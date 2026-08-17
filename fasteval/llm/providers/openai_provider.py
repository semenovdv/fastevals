import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai
from langfuse.decorators import langfuse_context, observe  # noqa: F401 (langfuse_context needed for test mocking)
from loguru import logger
from openai import AsyncOpenAI
from openai.lib._parsing._responses import type_to_text_format_param
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

import config
from fasteval.llm import LLMServiceError
from fasteval.llm.providers.base_processor import BaseLLMProcessor, GenerationResult
from fasteval.llm.providers.utils import before_sleep_log_with_langfuse


def is_retryable_openai(e: Exception) -> bool:
    """
    Retry on rate limits (429) and server errors (5xx).
    Stop immediately on client errors (400, 401, 403, 404).
    """
    if isinstance(e, openai.RateLimitError):
        return True
    if isinstance(e, openai.APIStatusError) and e.status_code >= 500:
        return True
    return False


@dataclass(frozen=True)
class OpenAIGenerationRequest:
    """Immutable container for all generation parameters."""

    model: str
    system_prompt: str = "You are a helpful assistant."
    user_prompt: str = "Hello!"
    temperature: Optional[float] = 1.0
    seed: Optional[int] = None
    max_output_tokens: Optional[int] = None
    thinking_budget: Optional[str] = None
    file_paths: Optional[tuple[Union[str, Path], ...]] = None
    image_bytes_list: Optional[tuple[bytes, ...]] = None
    image_detail: Optional[str] = None
    history: Optional[tuple[Dict[str, Any], ...]] = None
    response_schema: Optional[Union[type, Dict[str, Any]]] = None
    previous_response_id: Optional[str] = None


@dataclass
class OpenAIGenerationResult:
    """Container for generation result with usage metadata."""

    content: Union[str, Dict[str, Any]]
    raw_response: Any = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: Optional[str] = None
    thoughts: Optional[str] = None
    response_id: Optional[str] = None

    @staticmethod
    def _extract_cached_tokens(usage: Any) -> int:
        """Extract cached tokens from OpenAI usage details."""
        if not usage:
            return 0
        if hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
            return getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
            return getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        return 0

    @staticmethod
    def _map_status_to_finish_reason(response: Any) -> Optional[str]:
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
    def _extract_thoughts(response: Any) -> Optional[str]:
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
        response_schema: Optional[Union[type, Dict[str, Any]]] = None,
    ) -> "OpenAIGenerationResult":
        """Factory method to extract all metadata from OpenAI Responses API response.

        Args:
            response: Raw response from OpenAI Responses API.
            response_schema: Optional JSON schema or Pydantic model (if set, parses response as JSON).

        Returns:
            OpenAIGenerationResult with all metadata extracted.
        """
        # Parse content based on response type
        output_text = getattr(response, "output_text", "") or ""

        if response_schema and output_text:
            try:
                parsed_content: Union[str, Dict[str, Any]] = json.loads(output_text)
                # Validate if Pydantic model
                if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                    validated = response_schema.model_validate(parsed_content)
                    parsed_content = validated.model_dump()
            except json.JSONDecodeError as exc:
                raise LLMServiceError(f"Failed to parse JSON response: {exc}")
            except Exception as exc:
                raise LLMServiceError(f"Schema validation failed: {exc}")
        else:
            parsed_content = output_text

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
        cached_tokens = cls._extract_cached_tokens(usage)
        non_cached_input = max(0, (input_tokens or 0) - cached_tokens)

        return cls(
            content=parsed_content,
            raw_response=response,
            input_tokens=non_cached_input,
            output_tokens=completion_tokens,
            cached_tokens=cached_tokens,
            reasoning_tokens=reasoning_tokens,
            finish_reason=cls._map_status_to_finish_reason(response),
            thoughts=cls._extract_thoughts(response),
            response_id=getattr(response, "id", None),
        )


def _add_additional_properties_false(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively add 'additionalProperties': false to all object types in a JSON schema.

    OpenAI's strict mode requires additionalProperties to be explicitly set to false
    for all object types in the schema.

    Args:
        schema: A JSON schema dictionary.

    Returns:
        A new schema with additionalProperties: false added to all objects.
    """
    if not isinstance(schema, dict):
        return schema

    result = schema.copy()

    # If this is an object type, add additionalProperties: false
    if result.get("type") == "object":
        result["additionalProperties"] = False

        # Recursively process properties
        if "properties" in result:
            result["properties"] = {
                key: _add_additional_properties_false(value) for key, value in result["properties"].items()
            }

    # Handle nested schemas in 'items' (for arrays)
    if "items" in result:
        result["items"] = _add_additional_properties_false(result["items"])

    # Handle 'allOf', 'anyOf', 'oneOf'
    for key in ("allOf", "anyOf", "oneOf"):
        if key in result:
            result[key] = [_add_additional_properties_false(item) for item in result[key]]

    # Handle '$defs' / 'definitions'
    for key in ("$defs", "definitions"):
        if key in result:
            result[key] = {
                def_key: _add_additional_properties_false(def_value) for def_key, def_value in result[key].items()
            }

    return result


class OpenaiProcessor(BaseLLMProcessor):
    """OpenAI LLM processor using the Responses API.

    Extends BaseLLMProcessor to leverage shared Langfuse logging and utilities.
    """

    provider_name: str = "OpenAI"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.client = self._set_up_llm()
        self._async_client: Optional[AsyncOpenAI] = None

    def _set_up_llm(self) -> openai.OpenAI:
        return openai.OpenAI(api_key=self.api_key)

    def _get_async_llm(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(api_key=self.api_key)
        return self._async_client

    def _convert_schema_to_params(
        self, response_schema: Union[type[BaseModel], Dict[str, Any]]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Converts a Pydantic model or JSON schema dict to API and metadata formats.

        Args:
            response_schema: Either a Pydantic BaseModel class or a raw JSON schema dictionary.

        Returns:
            Tuple of (text_param_for_api, json_schema_for_metadata)
        """
        # If it's a Pydantic model, use OpenAI's private converter for 1:1 compatibility
        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            response_format_param = type_to_text_format_param(response_schema)
            text_param = {"format": response_format_param}
            metadata_schema = response_schema.model_json_schema()
            return text_param, metadata_schema

        # Otherwise, assume it's a raw JSON schema that needs wrapping
        if isinstance(response_schema, dict):
            # Ensure additionalProperties: false is set for all objects (required by OpenAI strict mode)
            strict_schema = _add_additional_properties_false(response_schema)
            text_param = {
                "format": {
                    "type": "json_schema",
                    "name": "ResponseSchema",
                    "schema": strict_schema,
                    "strict": True,
                }
            }
            return text_param, response_schema

        raise TypeError(f"Unsupported type for response_schema: {type(response_schema)}")

    def list_available_models(self):
        logger.info("Available models for text generation:")
        models = self.client.models.list()
        for m in models.data:
            logger.info(f"- {m.id}")
        return models

    async def _send_message_async_impl(
        self,
        request: OpenAIGenerationRequest,
    ) -> OpenAIGenerationResult:
        """Internal implementation: Generate a response asynchronously.

        This is the core async implementation method. Use send_message_async() for the public API.

        Args:
            request: OpenAIGenerationRequest containing all generation parameters.

        Returns:
            OpenAIGenerationResult containing the response content and metadata.
        """
        # Build request parameters
        request_params = self._build_request_params(
            model=request.model,
            temperature=request.temperature,
            system_prompt=request.system_prompt,
            thinking_budget=request.thinking_budget,
            seed=request.seed,
            max_output_tokens=request.max_output_tokens,
            previous_response_id=request.previous_response_id,
        )

        # Handle response schema
        if request.response_schema:
            text_param, _ = self._convert_schema_to_params(request.response_schema)
            request_params["text"] = text_param

        # Convert tuples to lists for internal methods
        history_list = list(request.history) if request.history else None
        file_paths_list = list(request.file_paths) if request.file_paths else None
        image_bytes_list = list(request.image_bytes_list) if request.image_bytes_list else None

        # Run content preparation in thread pool to avoid blocking (file uploads use time.sleep)
        contents = await asyncio.to_thread(
            self._prepare_contents,
            request.user_prompt,
            history_list,
            file_paths_list,
            request.image_detail,
            image_bytes_list,
        )
        request_params["input"] = contents

        try:
            response = await self._call_llm_async(request_params)
        except Exception as exc:
            self._log_error_openai(request=request, error=exc, result=None)
            raise

        # Check for incomplete status and log warning
        status = getattr(response, "status", None)
        incomplete_details = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(incomplete_details, "reason", None) if incomplete_details else None

        if status == "incomplete":
            warning_message = "Generation finished with status 'incomplete' (expected 'completed')"
            if incomplete_reason:
                warning_message += f" - reason: {incomplete_reason}"
            logger.warning(warning_message)

        # Parse response and build result
        try:
            result = OpenAIGenerationResult.from_openai_response(response, request.response_schema)
        except Exception as exc:
            self._log_error_openai(request=request, error=exc, result=None)
            raise

        return result

    def _send_message_sync_impl(
        self,
        request: OpenAIGenerationRequest,
    ) -> OpenAIGenerationResult:
        """Internal sync implementation: Generate a response.

        This is the core sync implementation method. Use send_message() for the public API.

        Args:
            request: OpenAIGenerationRequest containing all generation parameters.

        Returns:
            OpenAIGenerationResult containing the response content and metadata.
        """
        # Build request parameters
        request_params = self._build_request_params(
            model=request.model,
            temperature=request.temperature,
            system_prompt=request.system_prompt,
            thinking_budget=request.thinking_budget,
            seed=request.seed,
            max_output_tokens=request.max_output_tokens,
            previous_response_id=request.previous_response_id,
        )

        # Handle response schema
        if request.response_schema:
            text_param, _ = self._convert_schema_to_params(request.response_schema)
            request_params["text"] = text_param

        # Convert tuples to lists for internal methods
        history_list = list(request.history) if request.history else None
        file_paths_list = list(request.file_paths) if request.file_paths else None
        image_bytes_list = list(request.image_bytes_list) if request.image_bytes_list else None

        # Prepare contents
        contents = self._prepare_contents(
            request.user_prompt,
            history_list,
            file_paths_list,
            request.image_detail,
            image_bytes_list,
        )
        request_params["input"] = contents

        try:
            response = self._call_llm(request_params)
        except Exception as exc:
            self._log_error_openai(request=request, error=exc, result=None)
            raise

        # Check for incomplete status and log warning
        status = getattr(response, "status", None)
        incomplete_details = getattr(response, "incomplete_details", None)
        incomplete_reason = getattr(incomplete_details, "reason", None) if incomplete_details else None

        if status == "incomplete":
            warning_message = "Generation finished with status 'incomplete' (expected 'completed')"
            if incomplete_reason:
                warning_message += f" - reason: {incomplete_reason}"
            logger.warning(warning_message)

        # Parse response and build result
        try:
            result = OpenAIGenerationResult.from_openai_response(response, request.response_schema)
        except Exception as exc:
            self._log_error_openai(request=request, error=exc, result=None)
            raise

        return result

    @observe(as_type="generation", name="OpenAI.responses.create")
    def send_message(
        self,
        model: str,
        temperature: Optional[float] = 1.0,
        seed: Optional[int] = 12345,
        system_prompt: str = "You are a helpful assistant.",
        user_prompt: str = "Hello!",
        history: Optional[List[Dict[str, str]]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        image_bytes_list: Optional[List[bytes]] = None,
        response_schema: Optional[Union[type[BaseModel], Dict[str, Any]]] = None,
        thinking_budget: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        previous_response_id: Optional[str] = None,
        return_response_id: bool = False,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], tuple]:
        """Generate a response using the OpenAI Responses API.

        This is the public API matching the LLMProcessor protocol. Internally uses
        OpenAIGenerationRequest for type safety and OpenAIGenerationResult for metadata.

        Args:
            model: Model ID to use (required).
            temperature: Temperature for generation. Higher values make output more random.
            seed: Not used by the Responses API, kept for signature compatibility.
            system_prompt: Instructions for the model (maps to 'instructions').
            user_prompt: The text prompt to send.
            history: Optional conversation history. List of dicts with "role" and "content" keys.
            file_paths: list[Path | str] | None
                One or more local files to attach to the prompt (PDF or images: PNG, JPG, JPEG, WEBP, GIF).
            image_bytes_list: List of image bytes (PNG/JPEG) to attach inline.
            response_schema: Optional structured output schema. Can be either:
                - A Pydantic BaseModel class (recommended for type safety)
                - A JSON schema dictionary (for custom schemas)
                When provided, the response will be a dict conforming to the schema.
            thinking_budget: Optional reasoning effort ('low', 'medium', 'high'). Only for 'o' models.
            max_output_tokens: Maximum number of tokens to generate in the response.
            previous_response_id: (OpenAI Responses API) Continue a multi-turn chain by
                referencing the prior response's ID. When provided, OpenAI reuses prior
                response state (incl. reasoning items & tool context) for this turn.
            return_response_id: If True, returns a tuple of (response, response_id).
            **kwargs: Additional provider-specific arguments.
                image_detail: Detail level for image processing ('low', 'high', 'auto', 'low+high').
                    Applied to all images in file_paths. Default: auto (OpenAI decides).
                    'low+high' attaches each image twice - once with low detail, once with high.

        Returns:
            Either a string response or a dict if response_schema is specified.
            If return_response_id is True, returns a tuple of (response, response_id).
        """
        # Extract provider-specific kwargs
        image_detail = kwargs.pop("image_detail", None)

        # Build immutable request (convert lists to tuples)
        request = OpenAIGenerationRequest(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            seed=seed,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
            file_paths=tuple(file_paths) if file_paths else None,
            image_bytes_list=tuple(image_bytes_list) if image_bytes_list else None,
            image_detail=image_detail,
            history=tuple(history) if history else None,
            response_schema=response_schema,
            previous_response_id=previous_response_id,
        )

        result = self._send_message_sync_impl(request)

        # Log usage to Langfuse (now inside @observe context)
        self._log_usage_openai(request, result)

        if kwargs.get("return_result"):
            return result
        if return_response_id:
            return result.content, result.response_id
        return result.content

    @observe(as_type="generation", name="OpenAI.responses.create_async")
    async def send_message_async(
        self,
        model: str,
        temperature: Optional[float] = 1.0,
        seed: Optional[int] = 12345,
        system_prompt: str = "You are a helpful assistant.",
        user_prompt: str = "Hello!",
        history: Optional[List[Dict[str, str]]] = None,
        file_paths: Optional[List[Union[str, Path]]] = None,
        image_bytes_list: Optional[List[bytes]] = None,
        response_schema: Optional[Union[type[BaseModel], Dict[str, Any]]] = None,
        thinking_budget: Optional[str] = None,
        max_output_tokens: Optional[int] = None,
        previous_response_id: Optional[str] = None,
        return_raw_response: bool = False,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any], tuple]:
        """Generate a response asynchronously using OpenAI Responses API.

        This is the public API matching the LLMProcessor protocol. Internally uses
        OpenAIGenerationRequest for type safety and OpenAIGenerationResult for metadata.

        Args:
            model: Model ID to use (required).
            temperature: Temperature for generation. Higher values make output more random.
            seed: Not used by the Responses API, kept for signature compatibility.
            system_prompt: Instructions for the model (maps to 'instructions').
            user_prompt: The text prompt to send.
            history: Optional conversation history. List of dicts with "role" and "content" keys.
            file_paths: list[Path | str] | None
                One or more local files to attach to the prompt (PDF or images: PNG, JPG, JPEG, WEBP, GIF).
            image_bytes_list: List of image bytes (PNG/JPEG) to attach inline.
            response_schema: Optional structured output schema. Can be either:
                - A Pydantic BaseModel class (recommended for type safety)
                - A JSON schema dictionary (for custom schemas)
                When provided, the response will be a dict conforming to the schema.
            thinking_budget: Optional reasoning effort ('low', 'medium', 'high'). Only for 'o' models.
            max_output_tokens: Maximum number of tokens to generate in the response.
            previous_response_id: (OpenAI Responses API) Continue a multi-turn chain by
                referencing the prior response's ID. When provided, OpenAI reuses prior
                response state (incl. reasoning items & tool context) for this turn.
            return_raw_response: If True, returns tuple of (parsed_response, raw_response).
            **kwargs: Additional provider-specific arguments.
                image_detail: Detail level for image processing ('low', 'high', 'auto', 'low+high').
                    Applied to all images in file_paths. Default: auto (OpenAI decides).
                    'low+high' attaches each image twice - once with low detail, once with high.

        Returns:
            Either a string response or a dict if response_schema is specified.
            If return_raw_response is True, returns a tuple of (response, raw_response).
        """
        # Extract provider-specific kwargs
        image_detail = kwargs.pop("image_detail", None)

        # Build immutable request (convert lists to tuples)
        request = OpenAIGenerationRequest(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            seed=seed,
            max_output_tokens=max_output_tokens,
            thinking_budget=thinking_budget,
            file_paths=tuple(file_paths) if file_paths else None,
            image_bytes_list=tuple(image_bytes_list) if image_bytes_list else None,
            image_detail=image_detail,
            history=tuple(history) if history else None,
            response_schema=response_schema,
            previous_response_id=previous_response_id,
        )

        result = await self._send_message_async_impl(request)

        # Log usage to Langfuse (now inside @observe context)
        self._log_usage_openai(request, result)

        if return_raw_response:
            return result.content, result.raw_response
        return result.content

    @retry(
        wait=wait_exponential(multiplier=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(is_retryable_openai),
        reraise=True,
        before_sleep=before_sleep_log_with_langfuse,
    )
    async def _call_api_async(self, request_params: Dict[str, Any], timeout: float) -> Any:
        """Make async API call with tenacity retry logic.

        Retryable exceptions (RateLimitError, 5xx) propagate for @retry to handle.
        Non-retryable exceptions propagate as-is.
        """
        client = self._get_async_llm()
        return await client.responses.create(**request_params, timeout=timeout)

    async def _call_llm_async(self, request_params: Dict[str, Any]) -> Any:
        """Async version of _call_llm with background polling support."""
        use_background = request_params.get("background", False)
        timeout = request_params.pop("timeout", None)
        if timeout is None:
            timeout = 3600

        try:
            response = await self._call_api_async(request_params, timeout)
        except (openai.RateLimitError, openai.APIStatusError) as exc:
            # Retries exhausted for retryable errors
            logger.opt(exception=True).error(f"[OpenAI API] Retries exhausted: {type(exc).__name__}: {exc}")
            raise LLMServiceError(f"API request failed after retries: {exc}") from exc
        except openai.APIError as exc:
            # Non-retryable API errors
            logger.opt(exception=True).error(f"[OpenAI API] API error: {type(exc).__name__}: {exc}")
            raise LLMServiceError(f"API request failed: {exc}") from exc
        except Exception as exc:
            # Unexpected errors
            logger.opt(exception=True).error(f"[OpenAI API] Unexpected error: {type(exc).__name__}: {exc}")
            raise LLMServiceError(f"API request failed: {exc}") from exc

        if use_background:
            response = await self._poll_background_response_async(response.id)

        return response

    async def _poll_background_response_async(self, response_id: str) -> Any:
        """Async version of _poll_background_response."""
        import asyncio

        poll_interval = 1.0
        queued_warning_threshold = 30.0
        start_time = time.time()
        warning_printed = False

        client = self._get_async_llm()
        result = await client.responses.retrieve(response_id)

        while result.status in {"queued", "in_progress"}:
            elapsed = time.time() - start_time

            if result.status == "queued" and elapsed > queued_warning_threshold and not warning_printed:
                logger.warning(
                    f"Request {response_id} has been queued for more than {queued_warning_threshold} seconds"
                )
                warning_printed = True

            await asyncio.sleep(poll_interval)
            result = await client.responses.retrieve(response_id)

        if result.status != "completed":
            error_msg = f"Background response finished with status: {result.status}"
            if hasattr(result, "error") and result.error:
                error_msg += f" - {result.error.message}"
            raise LLMServiceError(error_msg)

        return result

    def _upload_file(self, file_path: Union[str, Path]) -> tuple[str, str]:
        p = Path(file_path)
        ext = p.suffix.lower()

        if ext == ".pdf":
            purpose = "user_data"
            content_type = "input_file"
        elif ext in config.OPENAI_UPLOAD_EXTENSIONS - {".pdf"}:
            purpose = "vision"
            content_type = "input_image"
        else:
            raise ValueError(config.OPENAI_UNSUPPORTED_FILE_TYPE_MESSAGE)

        if not p.is_file():
            raise FileNotFoundError(p)

        self._save_pdf_file(p, provider=self.provider_name.lower(), context="file")

        for attempt in range(3):
            try:
                with open(p, "rb") as f:
                    uploaded_file = self.client.files.create(
                        file=f,
                        purpose=purpose,
                        timeout=240.0,
                        extra_body={"expires_after": {"anchor": "created_at", "seconds": 3600}},
                    )

                return uploaded_file.id, content_type
            except (openai.APITimeoutError, openai.APIConnectionError):
                if attempt == 2:
                    raise
                time.sleep(1)
            except Exception as e:
                logger.opt(exception=True).error(f"Error uploading file {p}: {e}")
                raise
        return None, None

    def _build_request_params(
        self,
        model: str,
        temperature: Optional[float],
        system_prompt: str,
        thinking_budget: Optional[str],
        seed: Optional[int] = None,
        max_output_tokens: Optional[int] = None,
        previous_response_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Builds the dictionary of request parameters for the API call."""
        params: Dict[str, Any] = {
            "model": model,
            "instructions": system_prompt,
        }

        if not (model.startswith("o") or "gpt-5" in model) and temperature is not None:
            params["temperature"] = temperature

        if max_output_tokens is not None:
            params["max_output_tokens"] = max_output_tokens

        if previous_response_id:
            params["previous_response_id"] = previous_response_id

        self._apply_thinking_budget(params, thinking_budget, model)

        use_background = model.startswith("o") or "gpt-5" in model
        if use_background:
            params["background"] = True
            params["store"] = True

        return params

    def _convert_history_turn(self, turn: Dict[str, str]) -> Dict[str, Any]:
        """Convert a single history turn to OpenAI Responses API format."""
        role = "assistant" if turn["role"] == "model" else turn["role"]
        turn_content = turn["content"]
        if role == "assistant":
            return {"role": role, "content": [{"type": "output_text", "text": turn_content}]}
        return {"role": role, "content": turn_content}

    def _add_file_to_content(
        self, user_content: List[Dict[str, Any]], fp: Union[str, Path], image_detail: Optional[str]
    ) -> None:
        """Upload file and add to user content list."""
        file_id, content_type = self._upload_file(fp)

        if content_type == "input_image" and image_detail == "low+high":
            user_content.append({"type": content_type, "file_id": file_id, "detail": "low"})
            user_content.append({"type": content_type, "file_id": file_id, "detail": "high"})
        else:
            item: Dict[str, Any] = {"type": content_type, "file_id": file_id}
            if content_type == "input_image" and image_detail:
                item["detail"] = image_detail
            user_content.append(item)

    def _add_image_bytes_to_content(
        self, user_content: List[Dict[str, Any]], img_bytes: bytes, image_detail: Optional[str]
    ) -> None:
        """Add base64-encoded image bytes to user content list."""
        import base64

        mime_type = self._detect_image_mime_openai(img_bytes)
        b64_data = base64.standard_b64encode(img_bytes).decode("utf-8")
        item: Dict[str, Any] = {
            "type": "input_image",
            "image_url": f"data:{mime_type};base64,{b64_data}",
        }
        if image_detail and image_detail != "low+high":
            item["detail"] = image_detail
        user_content.append(item)

    def _prepare_contents(
        self,
        user_prompt: str,
        history: Optional[List[Dict[str, str]]],
        file_paths: Optional[List[Union[str, Path]]],
        image_detail: Optional[str] = None,
        image_bytes_list: Optional[List[bytes]] = None,
    ) -> List[Dict[str, Any]]:
        """Prepares the 'input' content list, including history and files."""
        contents: List[Dict[str, Any]] = []

        if history:
            contents.extend(self._convert_history_turn(turn) for turn in history)

        user_content: List[Dict[str, Any]] = [{"type": "input_text", "text": user_prompt}]

        if file_paths:
            for fp in file_paths:
                self._add_file_to_content(user_content, fp, image_detail)

        if image_bytes_list:
            self._save_image_bytes_list(
                image_bytes_list,
                provider=self.provider_name.lower(),
                context="inline",
            )
            for img_bytes in image_bytes_list:
                self._add_image_bytes_to_content(user_content, img_bytes, image_detail)

        # If user content is only text (no images), use plain string for compatibility
        if len(user_content) == 1 and user_content[0].get("type") == "input_text":
            contents.append({"role": "user", "content": user_content[0]["text"]})
        else:
            contents.append({"role": "user", "content": user_content})
        return contents

    # _detect_image_mime uses inherited method from BaseLLMProcessor
    # Override fallback to "image/png" for OpenAI compatibility
    def _detect_image_mime_openai(self, img_bytes: bytes) -> str:
        """Detect image mime type with OpenAI-compatible fallback."""
        return self._detect_image_mime(img_bytes, fallback="image/png")

    @retry(
        wait=wait_exponential(multiplier=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception(is_retryable_openai),
        reraise=True,
        before_sleep=before_sleep_log_with_langfuse,
    )
    def _call_api(self, request_params: Dict[str, Any], timeout: float) -> Any:
        """Make sync API call with tenacity retry logic.

        Retryable exceptions (RateLimitError, 5xx) propagate for @retry to handle.
        Non-retryable exceptions propagate as-is.
        """
        return self.client.responses.create(**request_params, timeout=timeout)

    def _call_llm(self, request_params: Dict[str, Any]) -> Any:
        """Calls the OpenAI API with background polling if needed."""
        use_background = request_params.get("background", False)
        # Allow callers to override the SDK request timeout on a per-request basis.
        # OpenAI Python SDK expects timeout in seconds (float/int).
        timeout = request_params.pop("timeout", None)
        if timeout is None:
            timeout = 3600

        try:
            response = self._call_api(request_params, timeout)
        except (openai.RateLimitError, openai.APIStatusError) as exc:
            # Retries exhausted for retryable errors
            logger.opt(exception=True).error(f"[OpenAI API] Retries exhausted: {type(exc).__name__}: {exc}")
            raise LLMServiceError(f"API request failed after retries: {exc}") from exc
        except openai.APIError as exc:
            # Non-retryable API errors
            logger.opt(exception=True).error(f"[OpenAI API] API error: {type(exc).__name__}: {exc}")
            raise LLMServiceError(f"API request failed: {exc}") from exc
        except Exception as exc:
            # Unexpected errors
            logger.opt(exception=True).error(f"[OpenAI API] Unexpected error: {type(exc).__name__}: {exc}")
            raise LLMServiceError(f"API request failed: {exc}") from exc

        if use_background:
            response = self._poll_background_response(response.id)

        return response

    def _poll_background_response(self, response_id: str) -> Any:
        """Polls a background response until completion."""
        poll_interval = 1.0
        queued_warning_threshold = 30.0
        start_time = time.time()
        warning_printed = False

        result = self.client.responses.retrieve(response_id)

        while result.status in {"queued", "in_progress"}:
            elapsed = time.time() - start_time

            if result.status == "queued" and elapsed > queued_warning_threshold and not warning_printed:
                logger.warning(
                    f"Request {response_id} has been queued for more than {queued_warning_threshold} seconds"
                )
                warning_printed = True

            time.sleep(poll_interval)
            result = self.client.responses.retrieve(response_id)

        if result.status != "completed":
            error_msg = f"Background response finished with status: {result.status}"
            if hasattr(result, "error") and result.error:
                error_msg += f" - {result.error.message}"
            raise LLMServiceError(error_msg)

        return result

    def _log_usage_openai(
        self,
        request: OpenAIGenerationRequest,
        result: OpenAIGenerationResult,
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
            reasoning_tokens=result.reasoning_tokens,
            finish_reason=result.finish_reason,
            thoughts=result.thoughts,
            response_id=result.response_id,
            provider=self.provider_name,
        )

        # Extra metadata specific to OpenAI
        extra_metadata = {
            "response_id": result.response_id,
            "previous_response_id": request.previous_response_id,
        }

        # Use base class logging
        BaseLLMProcessor._log_usage(
            self,
            model=request.model,
            input_dict=input_dict,
            output=result.content,
            result=gen_result,
            temperature=request.temperature,
            seed=request.seed,
            response_schema=request.response_schema,
            extra_metadata=extra_metadata,
        )

    def _log_error_openai(
        self,
        request: OpenAIGenerationRequest,
        error: Exception,
        result: Optional[OpenAIGenerationResult] = None,
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
                reasoning_tokens=result.reasoning_tokens,
                finish_reason=result.finish_reason,
                thoughts=result.thoughts,
                response_id=result.response_id,
                provider=self.provider_name,
            )

        # Use base class logging
        BaseLLMProcessor._log_error(
            self,
            model=request.model,
            input_dict=input_dict,
            error=error,
            result=gen_result,
            temperature=request.temperature,
            seed=request.seed,
            response_schema=request.response_schema,
        )

    @staticmethod
    def _apply_thinking_budget(params: Dict[str, Any], budget: Optional[Any], model: str) -> None:
        # Handle None or falsy values (including 0 from Gemini-style calls) as "no thinking"
        if budget is None or budget == 0 or budget == "none":
            return
        # Validate string values
        if not isinstance(budget, str) or budget not in {"minimal", "low", "medium", "high"}:
            raise ValueError(
                "thinking_budget must be one of 'minimal', 'low', 'medium', 'high' (or 0/None to disable)."
            )
        if not (model.startswith("o") or "gpt-5" in model):
            logger.warning(
                f"Thinking budget not supported for model: {model}. It is only supported for reasoning models. This parameter will be ignored."
            )
            return
        params["reasoning"] = {"effort": budget, "summary": "detailed"}
