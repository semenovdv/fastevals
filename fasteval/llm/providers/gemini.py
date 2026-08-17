import asyncio
import contextvars
import json
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from asgiref.sync import async_to_sync
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError, ServerError
from langfuse.decorators import langfuse_context, observe  # noqa: F401 (langfuse_context needed for test mocking)
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from config import GEMINI_FALLBACK_MODELS, GEMINI_THINKING_BUDGET, OPENAI_API_KEY
from fasteval.llm import LLMServiceError
from fasteval.llm.providers.base_processor import BaseLLMProcessor, GenerationResult
from fasteval.llm.providers.utils import before_sleep_log_with_langfuse


@dataclass(frozen=True)
class GeminiGenerationRequest:
    """Immutable container for all generation parameters."""

    model: str
    system_prompt: str = "You are a helpful assistant."
    user_prompt: str = "Hello!"
    temperature: float = 1.0
    seed: Optional[int] = None
    max_output_tokens: Optional[int] = None
    thinking_budget: Optional[int] = None
    media_resolution: Optional[str] = None
    file_paths: Optional[tuple[Union[str, Path], ...]] = None
    image_bytes_list: Optional[tuple[bytes, ...]] = None
    history: Optional[tuple[types.Content, ...]] = None
    response_schema: Optional[Dict[str, Any]] = None
    response_mime_type: Optional[str] = None
    uploaded_files: Optional[tuple] = None  # tuple[UploadedGeminiFile, ...]


@dataclass
class GeminiGenerationResult:
    """Container for generation result with usage metadata."""

    content: Union[str, Dict[str, Any]]
    raw_response: Any = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    thoughts_tokens: int = 0
    finish_reason: Optional[str] = None
    thoughts: Optional[str] = None
    effective_model: Optional[str] = None
    effective_provider: Optional[str] = None
    requested_model: Optional[str] = None
    attempt_number: int = 1
    is_fallback: bool = False
    fallback_reason: Optional[str] = None
    is_followup_request: bool = False

    @classmethod
    def from_gemini_response(
        cls,
        response: Any,
        response_schema: Optional[Dict[str, Any]] = None,
        response_mime_type: Optional[str] = None,
    ) -> "GeminiGenerationResult":
        """Factory method to extract all metadata from raw Gemini response.

        Args:
            response: Raw response from Gemini API.
            response_schema: Optional JSON schema (if set, parses response as JSON).
            response_mime_type: Optional MIME type (if "application/json", parses as JSON).

        Returns:
            GeminiGenerationResult with all metadata extracted.
        """
        # Parse content based on response type
        if response_schema or response_mime_type == "application/json":
            parsed_content: Union[str, Dict[str, Any]] = json.loads(response.text)
        else:
            parsed_content = response.text

        # Extract usage metadata
        input_tokens = 0
        output_tokens = 0
        cached_tokens = 0
        thoughts_tokens = 0

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            cached_tokens = getattr(response.usage_metadata, "cached_content_token_count", 0) or 0
            thoughts_tokens = getattr(response.usage_metadata, "thoughts_token_count", 0) or 0
            # Subtract cached tokens from input tokens
            input_tokens = max(0, input_tokens - cached_tokens)

        # Extract finish reason (FinishReason is an enum, get its name)
        finish_reason = None
        if response.candidates and response.candidates[0].finish_reason:
            fr = response.candidates[0].finish_reason
            # Handle both enum (has .name) and string representations
            finish_reason = fr.name if hasattr(fr, "name") else str(fr)

        # Extract thoughts from response parts
        thoughts_list: List[str] = []
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if getattr(part, "thought", False) and part.text:
                    thoughts_list.append(part.text)

        thoughts = "\n\n---\n\n".join(thoughts_list) if thoughts_list else None

        return cls(
            content=parsed_content,
            raw_response=response,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cached_tokens=cached_tokens,
            thoughts_tokens=thoughts_tokens,
            finish_reason=finish_reason,
            thoughts=thoughts,
        )

    @staticmethod
    def _extract_cached_tokens_from_openai(usage: Any) -> int:
        """Extract cached tokens from OpenAI usage details."""
        if not usage:
            return 0
        if hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
            return getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            return getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        return 0

    @staticmethod
    def _map_openai_status_to_finish_reason(response: Any) -> Optional[str]:
        """Map OpenAI response status to finish reason string."""
        status = getattr(response, "status", None)
        if status == "completed":
            return "STOP"
        if status == "incomplete":
            incomplete_details = getattr(response, "incomplete_details", None)
            incomplete_reason = getattr(incomplete_details, "reason", None) if incomplete_details else None
            return f"INCOMPLETE:{incomplete_reason}" if incomplete_reason else "INCOMPLETE"
        return status.upper() if status else None

    @staticmethod
    def _extract_thoughts_from_openai(response: Any) -> Optional[str]:
        """Extract reasoning/thoughts from OpenAI response output items."""
        if not hasattr(response, "output") or not response.output:
            return None
        thoughts_list: List[str] = []
        for item in response.output:
            if getattr(item, "type", None) == "reasoning" and hasattr(item, "summary"):
                for part in item.summary:
                    if hasattr(part, "text") and part.text:
                        thoughts_list.append(part.text)
        return "\n\n---\n\n".join(thoughts_list) if thoughts_list else None

    @classmethod
    def from_openai_response(
        cls,
        response: Any,
        response_schema: Optional[Dict[str, Any]] = None,
    ) -> "GeminiGenerationResult":
        """Factory method to extract metadata from OpenAI Responses API response.

        Args:
            response: Raw response from OpenAI Responses API.
            response_schema: Optional JSON schema (if set, parses response as JSON).

        Returns:
            GeminiGenerationResult with all metadata extracted from OpenAI response.
        """
        # Parse content based on response type
        output_text = getattr(response, "output_text", "") or ""
        parsed_content: Union[str, Dict[str, Any]] = (
            json.loads(output_text) if response_schema and output_text else output_text
        )

        # Extract usage metadata
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        # Extract reasoning tokens from output_tokens_details
        reasoning_tokens = 0
        if usage and hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
            reasoning_tokens = getattr(usage.output_tokens_details, "reasoning_tokens", 0) or 0

        # Completion tokens = output - reasoning
        completion_tokens = max(0, (output_tokens or 0) - (reasoning_tokens or 0))

        # Extract cached tokens and compute non-cached input
        cached_tokens = cls._extract_cached_tokens_from_openai(usage)
        non_cached_input = max(0, (input_tokens or 0) - cached_tokens)

        return cls(
            content=parsed_content,
            raw_response=response,
            input_tokens=non_cached_input,
            output_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            thoughts_tokens=reasoning_tokens,
            finish_reason=cls._map_openai_status_to_finish_reason(response),
            thoughts=cls._extract_thoughts_from_openai(response),
        )


# Define which errors should trigger a retry
def is_retryable(e: Exception) -> bool:
    """
    Retry on Server Errors (5xx) or Too Many Requests (429).
    Stop immediately on Client Errors (400, 401, 403, 404).
    """
    if isinstance(e, ServerError):
        return True  # 5xx errors
    if isinstance(e, APIError) and e.code == 429:
        return True  # Rate limit
    return False  # Don't retry bad requests (400) or auth (401)


class RecitationError(LLMServiceError):
    """Raised when Gemini returns a RECITATION finish reason."""

    pass


class GeminiProcessor(BaseLLMProcessor):
    """Gemini LLM processor with OpenAI fallback support.

    Extends BaseLLMProcessor to leverage shared Langfuse logging and utilities.
    """

    provider_name: str = "Gemini"
    _recitation_fallback_source_model: str = "gemini-3-flash-preview"
    _recitation_fallback_model: str = "gemini-2.5-pro"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.llm_client = self._set_up_llm()
        self._openai_fallback_processor = None  # Lazy-initialized for fallback

    def _set_up_llm(self):
        # Timeout is in milliseconds - 600,000ms (10 min) to handle peak usage when model reasoning takes longer
        http_options = types.HttpOptions(timeout=600_000)
        client = genai.Client(api_key=self.api_key, http_options=http_options)
        return client

    def _get_openai_fallback_processor(self):
        """Lazily initialize OpenAI processor for fallback."""
        if self._openai_fallback_processor is None:
            if not OPENAI_API_KEY:
                raise LLMServiceError("OpenAI API key not configured for fallback")
            from fasteval.llm.providers.openai_provider import OpenaiProcessor

            self._openai_fallback_processor = OpenaiProcessor(api_key=OPENAI_API_KEY)
        return self._openai_fallback_processor

    def _get_fallback_model(self, gemini_model: str) -> str:
        """Get the OpenAI fallback model for a given Gemini model."""
        fallback_model = GEMINI_FALLBACK_MODELS.get(gemini_model)
        if not fallback_model:
            # Use first available fallback as default
            fallback_model = next(iter(GEMINI_FALLBACK_MODELS.values()), None)
            if not fallback_model:
                raise LLMServiceError(f"No fallback model configured for {gemini_model}")
            logger.warning(f"No specific fallback for {gemini_model}, using default: {fallback_model}")
        return fallback_model

    @staticmethod
    def _build_attempt_metadata(
        request: GeminiGenerationRequest,
        *,
        attempt_model: str,
        attempt_provider: str,
        attempt_number: int,
        fallback_reason: Optional[str] = None,
    ) -> Dict[str, Any]:
        is_followup_request = attempt_number > 1
        is_fallback_attempt = is_followup_request or attempt_model != request.model or bool(fallback_reason)
        metadata: Dict[str, Any] = {
            "requested_model": request.model,
            "effective_model": attempt_model,
            "effective_provider": attempt_provider,
            "attempt_number": attempt_number,
            "is_fallback": is_fallback_attempt,
            "is_followup_request": is_followup_request,
            "request_type": "followup_request" if is_followup_request else "initial_request",
        }
        if fallback_reason:
            metadata["fallback_reason"] = fallback_reason
        if attempt_model != request.model:
            metadata["fallback_from_model"] = request.model
        return metadata

    @staticmethod
    def _build_fallback_status_message(
        *,
        requested_model: str,
        effective_provider: str,
        effective_model: str,
        fallback_reason: Optional[str],
    ) -> str:
        reason = fallback_reason or "fallback"
        return (
            f"Handled {reason} with a new request. "
            f"Requested model: {requested_model}. "
            f"Effective model: {effective_provider} {effective_model}."
        )

    def _mark_fallback_on_current_observation(
        self,
        *,
        request: GeminiGenerationRequest,
        next_provider: str,
        next_model: str,
        fallback_reason: str,
        attempt_number: int,
    ) -> None:
        langfuse_context.update_current_observation(
            level="WARNING",
            status_message=(
                f"Starting follow-up request after {fallback_reason}. "
                f"Switching from {request.model} to {next_provider} {next_model}."
            ),
            metadata=self._build_attempt_metadata(
                request,
                attempt_model=next_model,
                attempt_provider=next_provider,
                attempt_number=attempt_number,
                fallback_reason=fallback_reason,
            ),
        )

    @staticmethod
    def _is_recitation_reason(reason: Any) -> bool:
        """Check if a finish/block reason indicates RECITATION."""
        if not reason:
            return False
        try:
            reason_str = reason.name if hasattr(reason, "name") and isinstance(reason.name, str) else str(reason)
            return "RECITATION" in reason_str.upper()
        except (TypeError, AttributeError):
            return False

    def _check_recitation_finish_reason(self, response: Any) -> bool:
        """Check if response has RECITATION finish reason."""
        if not response or not response.candidates:
            return False
        return self._is_recitation_reason(response.candidates[0].finish_reason)

    def _check_recitation_block_reason(self, response: Any) -> bool:
        """Check if response was blocked due to RECITATION (empty candidates case).

        When Gemini blocks a request before generation, candidates is empty
        and the reason lives in prompt_feedback.block_reason.
        """
        if not response:
            return False
        prompt_feedback = getattr(response, "prompt_feedback", None)
        if not prompt_feedback:
            return False
        return self._is_recitation_reason(getattr(prompt_feedback, "block_reason", None))

    def _convert_history_for_openai(self, history: Optional[List[types.Content]]) -> Optional[List[Dict[str, Any]]]:
        """Convert Gemini history format to OpenAI format.

        Gemini uses types.Content with role and parts, OpenAI uses dicts with role and content.
        Handles both text and inline image data (from types.Part.from_bytes()).
        """
        if not history:
            return None

        import base64

        openai_history: List[Dict[str, Any]] = []
        for turn in history:
            role = turn.role
            # OpenAI uses "assistant" instead of "model"
            if role == "model":
                role = "assistant"

            # Extract text and image parts
            content_parts: List[Dict[str, Any]] = []
            for part in turn.parts:
                # Handle text parts
                if hasattr(part, "text") and part.text:
                    content_parts.append({"type": "text", "text": part.text})
                # Handle inline image data (from types.Part.from_bytes())
                elif hasattr(part, "inline_data") and part.inline_data:
                    inline_data = part.inline_data
                    if hasattr(inline_data, "data") and hasattr(inline_data, "mime_type"):
                        b64_data = base64.standard_b64encode(inline_data.data).decode("utf-8")
                        content_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:{inline_data.mime_type};base64,{b64_data}"},
                            }
                        )

            if content_parts:
                # If only text parts, simplify to string content for compatibility
                if len(content_parts) == 1 and content_parts[0]["type"] == "text":
                    openai_history.append({"role": role, "content": content_parts[0]["text"]})
                else:
                    openai_history.append({"role": role, "content": content_parts})

        return openai_history if openai_history else None

    def _map_thinking_budget_for_openai(self, thinking_budget: Optional[int]) -> Optional[str]:
        """Map Gemini thinking budget (int 0-24576) to OpenAI reasoning effort (str).

        Mapping:
        - 0 or None: None (no reasoning)
        - 1-1023: 'low'
        - 1024-8191: 'medium'
        - 8192+: 'high'
        """
        if thinking_budget is None or thinking_budget == 0:
            return None
        if thinking_budget < 1024:
            return "low"
        if thinking_budget < 8192:
            return "medium"
        return "high"

    # _detect_image_mime is inherited from BaseLLMProcessor
    # Uses "application/octet-stream" as default fallback

    def _prepare_contents_with_images(
        self,
        user_prompt: str,
        history: Optional[List[types.Content]],
        file_paths: Optional[List[Union[str, Path]]],
        image_bytes_list: Optional[List[bytes]] = None,
        uploaded_files: Optional[List[Any]] = None,
    ) -> List[types.Content]:
        """Prepare contents including inline image bytes.

        Args:
            user_prompt: The text prompt to send.
            history: Optional conversation history.
            file_paths: List of local files to attach (will be uploaded).
            image_bytes_list: List of image bytes (PNG/JPEG) to attach inline.
            uploaded_files: Pre-uploaded UploadedGeminiFile refs — skips upload.

        Note:
            media_resolution is set at GenerateContentConfig level via
            _build_generation_config, not per-Part.
        """
        contents: List[types.Content] = []

        if history:
            for turn in history:
                if not isinstance(turn, types.Content):
                    raise TypeError("history must be a list of google.genai.types.Content")
                contents.append(turn)

        user_parts: List[Any] = [types.Part.from_text(text=user_prompt)]

        if uploaded_files:
            for uf in uploaded_files:
                logger.info(f"[Gemini] Using pre-uploaded file: {uf.name} ({uf.mime_type})")
                user_parts.append(types.Part.from_uri(file_uri=uf.uri, mime_type=uf.mime_type))
        elif file_paths:
            user_parts.extend(self._upload_files(file_paths))

        # Add inline image bytes (more efficient for small images, no upload needed)
        # Note: media_resolution is set at GenerateContentConfig level, not per-Part
        if image_bytes_list:
            self._save_image_bytes_list(
                image_bytes_list,
                provider=self.provider_name.lower(),
                context="inline",
            )
            total_bytes = 0
            for idx, img_bytes in enumerate(image_bytes_list):
                img_size_kb = len(img_bytes) / 1024
                total_bytes += len(img_bytes)
                mime = self._detect_image_mime(img_bytes)
                logger.debug(
                    f"[Gemini] Adding image {idx + 1}/{len(image_bytes_list)}: " f"{img_size_kb:.1f}KB, mime={mime}"
                )
                # Validate image bytes are not empty or corrupted
                if len(img_bytes) < 100:
                    logger.warning(f"[Gemini] Image {idx + 1} is suspiciously small ({len(img_bytes)} bytes)")
                user_parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))

            total_mb = total_bytes / (1024 * 1024)
            logger.info(f"[Gemini] Prepared {len(image_bytes_list)} images, total size: {total_mb:.2f}MB")

        contents.append(types.Content(role="user", parts=user_parts))
        return contents

    def _build_generation_config(
        self,
        model: str,
        temperature: float,
        seed: Optional[int],
        system_prompt: str,
        response_schema: Optional[Dict[str, Any]],
        response_mime_type: Optional[str],
        thinking_budget: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        media_resolution: Optional[str] = None,
    ) -> types.GenerateContentConfig:
        generation_config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt,
        )

        # Only set seed if explicitly provided (not all models support it)
        if seed is not None:
            generation_config.seed = seed

        if max_output_tokens is not None:
            generation_config.max_output_tokens = max_output_tokens

        # Set media resolution for image/video processing quality
        # Valid values: "low", "medium", "high" -> maps to MediaResolution enum
        if media_resolution:
            resolution_map = {
                "low": types.MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                "high": types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            }
            if media_resolution in resolution_map:
                generation_config.media_resolution = resolution_map[media_resolution]
            else:
                logger.warning(f"Unknown media_resolution '{media_resolution}', ignoring. Valid: low, medium, high")

        self._apply_thinking_budget(generation_config, thinking_budget, model)

        if response_schema:
            generation_config.response_mime_type = "application/json"
            generation_config.response_schema = response_schema
        elif response_mime_type:
            generation_config.response_mime_type = response_mime_type

        return generation_config

    def _prepare_contents(
        self,
        user_prompt: str,
        history: Optional[List[types.Content]],
        file_paths: Optional[List[Union[str, Path]]],
        uploaded_files: Optional[List[Any]] = None,
    ) -> List[types.Content]:
        """Prepare contents without inline images. Wrapper for backwards compatibility."""
        return self._prepare_contents_with_images(
            user_prompt,
            history,
            file_paths,
            image_bytes_list=None,
            uploaded_files=uploaded_files,
        )

    def _upload_files(
        self,
        file_paths: List[Union[str, Path]],
        poll_interval: float = 2.0,
        max_wait_time: float = 300.0,
    ) -> List[types.Part]:
        """Upload files to the Gemini Files API and wait for processing to complete.

        Args:
            file_paths: List of file paths to upload.
            poll_interval: Seconds to wait between status checks (default: 2s).
            max_wait_time: Maximum seconds to wait for processing (default: 300s / 5min).

        Returns:
            List of Part objects ready to be used in API requests.

        Raises:
            FileNotFoundError: If a file doesn't exist.
            LLMServiceError: If file processing fails or times out.
        """
        parts: List[types.Part] = []
        for fp in file_paths:
            p = Path(fp)
            if not p.exists() or not p.is_file():
                raise FileNotFoundError(f"File not found: {p}")

            self._save_pdf_file(p, provider=self.provider_name.lower(), context="file")

            file_size_mb = p.stat().st_size / (1024 * 1024)
            logger.info(f"[Gemini Files API] Uploading file: {p.name} ({file_size_mb:.2f}MB)...")

            try:
                # 1. Upload the file
                uploaded = self.llm_client.files.upload(file=str(p))
                logger.info(f"[Gemini Files API] Upload started: {uploaded.name}")

                # 2. Wait for processing to complete
                start_time = time.time()
                while getattr(uploaded.state, "name", str(uploaded.state)) == "PROCESSING":
                    elapsed = time.time() - start_time
                    if elapsed > max_wait_time:
                        raise LLMServiceError(f"File processing timed out after {max_wait_time}s for {p.name}")
                    time.sleep(poll_interval)
                    # Refresh file status
                    uploaded = self.llm_client.files.get(name=uploaded.name)

                # 3. Check for errors after processing
                state_name = getattr(uploaded.state, "name", str(uploaded.state))
                if state_name != "ACTIVE":
                    raise LLMServiceError(f"File processing failed for {p.name} with state: {state_name}")

                # 4. Use the mime_type determined by Google
                mime = uploaded.mime_type or mimetypes.guess_type(p.name)[0] or "application/octet-stream"

                logger.info(f"[Gemini Files API] Ready: {p.name} -> uri={uploaded.uri}, mime={mime}")

                parts.append(types.Part.from_uri(file_uri=uploaded.uri, mime_type=mime))
            except LLMServiceError:
                raise
            except Exception as e:
                logger.opt(exception=e).debug(f"[Gemini Files API] Error processing {p.name}: {type(e).__name__}: {e}")
                raise LLMServiceError(f"File upload failed for {p.name}: {e}") from e

        return parts

    @staticmethod
    def _apply_thinking_budget(
        config: types.GenerateContentConfig,
        budget: Optional[int],
        model: str,
    ) -> None:
        # Use configured default if no budget specified
        effective_budget = budget if budget is not None else GEMINI_THINKING_BUDGET

        if effective_budget is None:
            return

        if not isinstance(effective_budget, int) or not (0 <= effective_budget <= 24576):
            raise ValueError("thinking_budget must be an int between 0 and 24576.")

        config.thinking_config = types.ThinkingConfig(thinking_budget=effective_budget, include_thoughts=True)

    # @observe(as_type="generation", name="Gemini._call_llm_async")
    @retry(
        wait=wait_exponential(multiplier=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(is_retryable),
        reraise=True,
        before_sleep=before_sleep_log_with_langfuse,
    )
    async def _call_llm_async(
        self,
        model: str,
        contents: List[types.Content],
        config: types.GenerateContentConfig,
    ) -> Any:
        """Call the Gemini API asynchronously.

        Retryable exceptions (ServerError, 429) propagate for @retry to handle.
        Non-retryable exceptions propagate as-is.
        """

        return await self.llm_client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )

    @staticmethod
    def _format_api_exception(exc: Exception) -> str:
        """Format an API exception for logs (handles empty httpx error messages)."""
        exc_str = str(exc).strip()
        if exc_str:
            return f"{type(exc).__name__}: {exc_str}"

        for attr in ("request", "response"):
            val = getattr(exc, attr, None)
            if val is not None and str(val).strip():
                return f"{type(exc).__name__}: {val}"

        cause = exc.__cause__ or exc.__context__
        if cause is not None:
            cause_str = str(cause).strip()
            if cause_str:
                return f"{type(exc).__name__} (caused by {type(cause).__name__}: {cause_str})"

        return type(exc).__name__

    @staticmethod
    def _raise_for_api_exception(exc: Exception, label: str) -> None:
        """Re-raise an API exception as RecitationError or LLMServiceError."""
        detail = GeminiProcessor._format_api_exception(exc)
        # Recoverable upstream failures are logged as warnings; terminal failures
        # are logged once at error level by the analysis pipeline.
        logger.opt(exception=exc).debug(f"[Gemini API] {label}: {detail}")
        if "recitation" in str(exc).lower():
            raise RecitationError(f"{label} with recitation: {exc}") from exc
        raise LLMServiceError(f"API request failed: {exc}") from exc

    def _validate_response(self, response: Any) -> None:
        """Validate a Gemini response, raising RecitationError or LLMServiceError as needed."""
        if response is None or not hasattr(response, "candidates"):
            raise LLMServiceError("Gemini returned a malformed or None response (no candidates attribute).")

        if not response.candidates:
            if self._check_recitation_block_reason(response):
                block_reason = getattr(getattr(response, "prompt_feedback", None), "block_reason", "RECITATION")
                block_reason_str = block_reason.name if hasattr(block_reason, "name") else str(block_reason)
                logger.warning(
                    f"RecitationError: Request blocked with reason '{block_reason_str}'. "
                    "Content may be too similar to training data."
                )
                raise RecitationError(
                    f"Request blocked with reason '{block_reason_str}' (RECITATION). "
                    "Content may be too similar to training data."
                )
            logger.warning(
                "LLMServiceError: Gemini returned empty response with no candidates. "
                "This may be due to safety filters, rate limiting, or content policy violations."
            )
            raise LLMServiceError(
                "Gemini returned empty response with no candidates. "
                "This may be due to safety filters, rate limiting, or content policy violations."
            )

        candidates = response.candidates
        try:
            first_candidate = candidates[0]
        except (TypeError, IndexError) as exc:
            raise LLMServiceError("Gemini response candidates is not a valid indexable sequence.") from exc

        finish_reason = getattr(first_candidate, "finish_reason", None)
        finish_reason_str = finish_reason.name if hasattr(finish_reason, "name") else str(finish_reason)

        if self._check_recitation_finish_reason(response):
            logger.warning(
                "RecitationError: Generation finished with reason 'RECITATION'. "
                "Content may be too similar to training data."
            )
            raise RecitationError(
                f"Generation finished with reason '{finish_reason_str}' (RECITATION). "
                "Content may be too similar to training data."
            )

        if finish_reason_str != "STOP":
            logger.warning(f"Generation finished with reason '{finish_reason_str}' (expected 'STOP')")

    async def _execute_gemini_call_async(
        self,
        model: str,
        contents: List[types.Content],
        generation_config: types.GenerateContentConfig,
        response_schema: Optional[Dict[str, Any]] = None,
        response_mime_type: Optional[str] = None,
    ) -> GeminiGenerationResult:
        """Execute a single async Gemini API call and process the response.

        Returns:
            GeminiGenerationResult with parsed content and all metadata.
        """
        try:
            response = await self._call_llm_async(model, contents, generation_config)
        except ClientError as exc:
            self._raise_for_api_exception(exc, "Client error")
        except (ServerError, APIError) as exc:
            self._raise_for_api_exception(exc, "Retries exhausted")
        except Exception as exc:
            self._raise_for_api_exception(exc, "Unexpected error")

        self._validate_response(response)

        try:
            return GeminiGenerationResult.from_gemini_response(
                response=response,
                response_schema=response_schema,
                response_mime_type=response_mime_type,
            )
        except (ValueError, TypeError) as exc:
            if "recitation" in str(exc).lower():
                raise RecitationError(f"Response parsing failed due to recitation: {exc}") from exc
            raise LLMServiceError(f"Failed to parse Gemini response: {exc}") from exc

    async def _fallback_to_openai_async(
        self,
        gemini_model: str,
        original_error: Exception,
        request: GeminiGenerationRequest,
        *,
        attempt_number: int = 2,
        fallback_reason: Optional[str] = None,
    ) -> GeminiGenerationResult:
        """Fall back to OpenAI asynchronously when Gemini fails.

        Args:
            gemini_model: The original Gemini model that failed.
            original_error: The exception that caused the fallback.
            request: The original generation request.

        Returns:
            GeminiGenerationResult wrapping the OpenAI response with full metadata.
        """
        fallback_model = self._get_fallback_model(gemini_model)
        logger.debug(
            f"Gemini failed with {type(original_error).__name__}, falling back to OpenAI model: {fallback_model}"
        )

        openai_processor = self._get_openai_fallback_processor()

        # Convert history from tuple to list for OpenAI
        history_list = list(request.history) if request.history else None
        file_paths_list = list(request.file_paths) if request.file_paths else None
        # Derive file paths from uploaded refs when no explicit paths are available
        if not file_paths_list and request.uploaded_files:
            file_paths_list = [uf.source_path for uf in request.uploaded_files]
        image_bytes_list = list(request.image_bytes_list) if request.image_bytes_list else None

        _, raw_response = await openai_processor.send_message_async(
            model=fallback_model,
            temperature=request.temperature,
            system_prompt=request.system_prompt,
            user_prompt=request.user_prompt,
            file_paths=file_paths_list,
            image_bytes_list=image_bytes_list,
            response_schema=request.response_schema,
            max_output_tokens=request.max_output_tokens,
            history=self._convert_history_for_openai(history_list),
            thinking_budget=self._map_thinking_budget_for_openai(request.thinking_budget),
            return_raw_response=True,
        )

        # Use factory method to extract all metadata from OpenAI response
        result = GeminiGenerationResult.from_openai_response(
            response=raw_response,
            response_schema=request.response_schema,
        )
        result.effective_model = fallback_model
        result.effective_provider = "OpenAI"
        result.requested_model = request.model
        result.attempt_number = attempt_number
        result.is_fallback = True
        result.fallback_reason = fallback_reason or type(original_error).__name__
        result.is_followup_request = attempt_number > 1
        return result

    @observe(as_type="generation", name="Gemini.generate_content_attempt")
    async def _execute_generation_attempt_async(
        self,
        *,
        request: GeminiGenerationRequest,
        attempt_model: str,
        contents: List[types.Content],
        generation_config: types.GenerateContentConfig,
        attempt_number: int,
        fallback_reason: Optional[str] = None,
    ) -> GeminiGenerationResult:
        """Execute a single Gemini API request as its own traced attempt."""
        attempt_metadata = self._build_attempt_metadata(
            request,
            attempt_model=attempt_model,
            attempt_provider=self.provider_name,
            attempt_number=attempt_number,
            fallback_reason=fallback_reason,
        )
        try:
            result = await self._execute_gemini_call_async(
                model=attempt_model,
                contents=contents,
                generation_config=generation_config,
                response_schema=request.response_schema,
                response_mime_type=request.response_mime_type,
            )
        except Exception as exc:
            self._log_error_gemini(
                request=request,
                error=exc,
                result=None,
                model_override=attempt_model,
                extra_metadata=attempt_metadata,
            )
            raise

        result.effective_model = attempt_model
        result.effective_provider = self.provider_name
        result.requested_model = request.model
        result.attempt_number = attempt_number
        result.is_fallback = attempt_metadata["is_fallback"]
        result.fallback_reason = fallback_reason
        result.is_followup_request = attempt_metadata["is_followup_request"]

        status_message = None
        if result.is_followup_request:
            status_message = self._build_fallback_status_message(
                requested_model=request.model,
                effective_provider=self.provider_name,
                effective_model=attempt_model,
                fallback_reason=fallback_reason,
            )

        self._log_usage_gemini(
            request,
            result,
            model_override=attempt_model,
            extra_metadata=attempt_metadata,
            status_message=status_message,
        )
        return result

    async def _send_message_async_impl(
        self,
        request: GeminiGenerationRequest,
    ) -> GeminiGenerationResult:
        """Internal implementation: Generate a response asynchronously with fallback to OpenAI.

        This is the core implementation method. Use send_message_async() for the public API.

        Args:
            request: GeminiGenerationRequest containing all generation parameters.

        Returns:
            GeminiGenerationResult containing the response content and metadata.
        """
        # Prepare contents once (file uploads happen here)
        loop = asyncio.get_running_loop()
        # Copy context to propagate Langfuse trace context into the executor thread
        ctx = contextvars.copy_context()

        # Convert tuples to lists for internal methods
        history_list = list(request.history) if request.history else None
        file_paths_list = list(request.file_paths) if request.file_paths else None
        image_bytes_list = list(request.image_bytes_list) if request.image_bytes_list else None
        uploaded_files_list = list(request.uploaded_files) if request.uploaded_files else None

        def _prepare():
            return self._prepare_contents_with_images(
                request.user_prompt,
                history_list,
                file_paths_list,
                image_bytes_list,
                uploaded_files=uploaded_files_list,
            )

        contents = await loop.run_in_executor(None, lambda: ctx.run(_prepare))

        generation_config = self._build_generation_config(
            model=request.model,
            temperature=request.temperature,
            seed=request.seed,
            system_prompt=request.system_prompt,
            response_schema=request.response_schema,
            response_mime_type=request.response_mime_type,
            thinking_budget=request.thinking_budget,
            max_output_tokens=request.max_output_tokens,
            media_resolution=request.media_resolution,
        )

        try:
            return await self._execute_generation_attempt_async(
                request=request,
                attempt_model=request.model,
                contents=contents,
                generation_config=generation_config,
                attempt_number=1,
            )
        except RecitationError as exc:
            if request.model == self._recitation_fallback_source_model:
                logger.warning(
                    f"RecitationError on {request.model}; retrying with {self._recitation_fallback_model} before OpenAI fallback."
                )
                self._mark_fallback_on_current_observation(
                    request=request,
                    next_provider=self.provider_name,
                    next_model=self._recitation_fallback_model,
                    fallback_reason="RECITATION",
                    attempt_number=2,
                )
                recitation_config = self._build_generation_config(
                    model=self._recitation_fallback_model,
                    temperature=request.temperature,
                    seed=request.seed,
                    system_prompt=request.system_prompt,
                    response_schema=request.response_schema,
                    response_mime_type=request.response_mime_type,
                    thinking_budget=request.thinking_budget,
                    max_output_tokens=request.max_output_tokens,
                    media_resolution=request.media_resolution,
                )
                try:
                    return await self._execute_generation_attempt_async(
                        request=request,
                        attempt_model=self._recitation_fallback_model,
                        contents=contents,
                        generation_config=recitation_config,
                        attempt_number=2,
                        fallback_reason="RECITATION",
                    )
                except Exception as fallback_exc:
                    fallback_model = self._get_fallback_model(request.model)
                    self._mark_fallback_on_current_observation(
                        request=request,
                        next_provider="OpenAI",
                        next_model=fallback_model,
                        fallback_reason="RECITATION",
                        attempt_number=3,
                    )
                    return await self._fallback_to_openai_async(
                        gemini_model=request.model,
                        original_error=fallback_exc,
                        request=request,
                        attempt_number=3,
                        fallback_reason="RECITATION",
                    )
            # For other models, fall back directly to OpenAI
            fallback_model = self._get_fallback_model(request.model)
            self._mark_fallback_on_current_observation(
                request=request,
                next_provider="OpenAI",
                next_model=fallback_model,
                fallback_reason="RECITATION",
                attempt_number=2,
            )
            return await self._fallback_to_openai_async(
                gemini_model=request.model,
                original_error=exc,
                request=request,
                attempt_number=2,
                fallback_reason="RECITATION",
            )
        except Exception as exc:
            # Non-RECITATION error: log and fall back immediately
            fallback_model = self._get_fallback_model(request.model)
            self._mark_fallback_on_current_observation(
                request=request,
                next_provider="OpenAI",
                next_model=fallback_model,
                fallback_reason=type(exc).__name__,
                attempt_number=2,
            )
            return await self._fallback_to_openai_async(
                gemini_model=request.model,
                original_error=exc,
                request=request,
                attempt_number=2,
                fallback_reason=type(exc).__name__,
            )

    def _send_message_sync_impl(
        self,
        request: GeminiGenerationRequest,
    ) -> GeminiGenerationResult:
        """Internal sync implementation: Generate a response with fallback to OpenAI.

        This is the core sync implementation method. Use send_message() for the public API.

        Args:
            request: GeminiGenerationRequest containing all generation parameters.

        Returns:
            GeminiGenerationResult containing the response content and metadata.
        """
        return async_to_sync(self._send_message_async_impl)(request=request)

    @observe(as_type="generation", name="Gemini.send_message_async")
    async def send_message_async(
        self,
        model: str,
        temperature: Optional[float] = 1.0,
        seed: Optional[int] = 12345,
        system_prompt: str = "You are a helpful assistant.",
        user_prompt: str = "Hello!",
        file_paths: Optional[List[Union[str, Path]]] = None,
        image_bytes_list: Optional[List[bytes]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        history: Optional[List[Any]] = None,
        thinking_budget: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        media_resolution: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Generate a response asynchronously using Gemini API.

        This is the public API matching the LLMProcessor protocol. Internally uses
        GeminiGenerationRequest for type safety and GeminiGenerationResult for metadata.

        Args:
            model: The Gemini model to use (required).
            temperature: Temperature for generation. Default: 1.0.
            seed: Random seed for reproducibility. Default: 12345.
            system_prompt: Instructions for the model.
            user_prompt: The prompt to send to the model.
            file_paths: List of local files to include (uploaded via Files API).
            image_bytes_list: List of image bytes (PNG/JPEG) to attach inline.
            response_schema: Optional JSON schema to structure the response.
            history: Optional conversation history as List[google.genai.types.Content].
            thinking_budget: Optional thinking budget for Gemini 2.5 models (0-24576).
            max_output_tokens: Maximum number of tokens to generate.
            media_resolution: Resolution for images/media ("low", "medium", "high").
            **kwargs: Additional provider-specific arguments (e.g., response_mime_type).

        Returns:
            The model's response as a string or JSON object (dict).
        """
        # Extract provider-specific kwargs
        response_mime_type = kwargs.pop("response_mime_type", None)
        uploaded_files_raw = kwargs.pop("uploaded_files", None)

        # Build immutable request (convert lists to tuples)
        request = GeminiGenerationRequest(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature if temperature is not None else 1.0,
            seed=seed,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
            media_resolution=media_resolution,
            file_paths=tuple(file_paths) if file_paths else None,
            image_bytes_list=tuple(image_bytes_list) if image_bytes_list else None,
            history=tuple(history) if history else None,
            response_schema=response_schema,
            response_mime_type=response_mime_type,
            uploaded_files=tuple(uploaded_files_raw) if uploaded_files_raw else None,
        )

        result = await self._send_message_async_impl(request)

        # Log usage to Langfuse (now inside @observe context)
        self._log_usage_gemini(request, result)

        return result.content

    @observe(as_type="generation", name="Gemini.generate_content")
    def send_message(
        self,
        model: str,
        temperature: Optional[float] = 1.0,
        seed: Optional[int] = 12345,
        system_prompt: str = "You are a helpful assistant.",
        user_prompt: str = "Hello!",
        file_paths: Optional[List[Union[str, Path]]] = None,
        image_bytes_list: Optional[List[bytes]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        history: Optional[List[Any]] = None,
        thinking_budget: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        media_resolution: Optional[str] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Generate a response synchronously using Gemini API.

        This is the public API matching the LLMProcessor protocol. Internally uses
        GeminiGenerationRequest for type safety and GeminiGenerationResult for metadata.

        Args:
            model: The Gemini model to use (required).
            temperature: Temperature for generation. Default: 1.0.
            seed: Random seed for reproducibility. Default: 12345.
            system_prompt: Instructions for the model.
            user_prompt: The prompt to send to the model.
            file_paths: List of local files to include (uploaded via Files API).
            image_bytes_list: List of image bytes (PNG/JPEG) to attach inline.
            response_schema: Optional JSON schema to structure the response.
            history: Optional conversation history as List[google.genai.types.Content].
            thinking_budget: Optional thinking budget for Gemini 2.5 models (0-24576).
            max_output_tokens: Maximum number of tokens to generate.
            media_resolution: Resolution for images/media ("low", "medium", "high").
            **kwargs: Additional provider-specific arguments (e.g., response_mime_type).

        Returns:
            The model's response as a string or JSON object (dict).
        """
        # Extract provider-specific kwargs
        response_mime_type = kwargs.pop("response_mime_type", None)
        uploaded_files_raw = kwargs.pop("uploaded_files", None)

        # Build immutable request (convert lists to tuples)
        request = GeminiGenerationRequest(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature if temperature is not None else 1.0,
            seed=seed,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
            media_resolution=media_resolution,
            file_paths=tuple(file_paths) if file_paths else None,
            image_bytes_list=tuple(image_bytes_list) if image_bytes_list else None,
            history=tuple(history) if history else None,
            response_schema=response_schema,
            response_mime_type=response_mime_type,
            uploaded_files=tuple(uploaded_files_raw) if uploaded_files_raw else None,
        )

        result = self._send_message_sync_impl(request)

        # Log usage to Langfuse (now inside @observe context)
        self._log_usage_gemini(request, result)

        return result.content

    def _log_usage_gemini(
        self,
        request: GeminiGenerationRequest,
        result: GeminiGenerationResult,
        *,
        model_override: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        status_message: Optional[str] = None,
    ) -> None:
        """Log usage metrics to Langfuse using the base class method.

        Args:
            request: The generation request parameters.
            result: The generation result with metadata.
        """
        # Build input dict
        history_length = len(request.history) if request.history else None
        input_dict: Dict[str, Any] = {"system": request.system_prompt}
        if history_length is not None:
            input_dict["history_turns"] = history_length
        input_dict["user"] = request.user_prompt

        # Convert to GenerationResult for base class
        gen_result = GenerationResult(
            content=result.content,
            raw_response=result.raw_response,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            cached_tokens=result.cached_tokens,
            reasoning_tokens=result.thoughts_tokens,
            finish_reason=result.finish_reason,
            thoughts=result.thoughts,
            provider=self.provider_name,
        )

        metadata: Dict[str, Any] = dict(extra_metadata or {})
        if result.requested_model:
            metadata.setdefault("requested_model", result.requested_model)
        if result.effective_model:
            metadata.setdefault("effective_model", result.effective_model)
        if result.effective_provider:
            metadata.setdefault("effective_provider", result.effective_provider)
        metadata.setdefault("attempt_number", result.attempt_number)
        metadata.setdefault("is_fallback", result.is_fallback)
        metadata.setdefault("is_followup_request", result.is_followup_request)
        if result.fallback_reason:
            metadata.setdefault("fallback_reason", result.fallback_reason)

        final_status_message = status_message
        if not final_status_message and result.is_fallback and result.effective_model and result.effective_provider:
            final_status_message = self._build_fallback_status_message(
                requested_model=result.requested_model or request.model,
                effective_provider=result.effective_provider,
                effective_model=result.effective_model,
                fallback_reason=result.fallback_reason,
            )

        # Use base class logging
        BaseLLMProcessor._log_usage(
            self,
            model=model_override or request.model,
            input_dict=input_dict,
            output=result.content,
            result=gen_result,
            temperature=request.temperature,
            seed=request.seed,
            response_schema=request.response_schema,
            extra_metadata=metadata,
            status_message=final_status_message,
        )

    def _log_error_gemini(
        self,
        request: GeminiGenerationRequest,
        error: Exception,
        result: Optional[GeminiGenerationResult] = None,
        *,
        model_override: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log observation when an error occurs using the base class method.

        Args:
            request: The generation request parameters.
            error: The exception that occurred.
            result: Optional partial result if error occurred after LLM call.
        """
        # Build input dict
        history_length = len(request.history) if request.history else None
        input_dict: Dict[str, Any] = {"system": request.system_prompt}
        if history_length is not None:
            input_dict["history_turns"] = history_length
        input_dict["user"] = request.user_prompt

        # Convert to GenerationResult if we have one
        gen_result = None
        if result is not None:
            gen_result = GenerationResult(
                content=result.content,
                raw_response=result.raw_response,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cached_tokens=result.cached_tokens,
                reasoning_tokens=result.thoughts_tokens,
                finish_reason=result.finish_reason,
                thoughts=result.thoughts,
                provider=self.provider_name,
            )

        # Use base class logging
        BaseLLMProcessor._log_error(
            self,
            model=model_override or request.model,
            input_dict=input_dict,
            error=error,
            result=gen_result,
            temperature=request.temperature,
            seed=request.seed,
            response_schema=request.response_schema,
            extra_metadata=extra_metadata,
        )
