"""Base LLM processor with shared functionality.

This module provides a base class for LLM providers that implements common
functionality like Langfuse logging, error handling, and result dataclasses.
This eliminates significant code duplication across providers.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from langfuse.decorators import langfuse_context
from loguru import logger

from fasteval.llm.providers.utils import detect_image_mime

_SAVE_LLM_ARTIFACTS = os.getenv("LOCAL_DEV") == "1"


@dataclass
class GenerationResult:
    """Unified container for generation result with usage metadata.

    This dataclass is used by all LLM providers to return consistent
    result structures with token usage, thoughts, and other metadata.
    """

    content: Union[str, Dict[str, Any]] 
    raw_response: Any = None
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0  # thoughts_tokens for Gemini, reasoning_tokens for OpenAI
    finish_reason: Optional[str] = None
    thoughts: Optional[str] = None
    response_id: Optional[str] = None  # OpenAI-specific, but included for uniformity
    provider: str = ""  # Provider name for logging

    @property
    def thoughts_tokens(self) -> int:
        """Alias for reasoning_tokens (Gemini terminology)."""
        return self.reasoning_tokens

    def get_usage_dict(self) -> Dict[str, Any]:
        """Get standardized usage dictionary for Langfuse logging."""
        return BaseLLMProcessor.build_langfuse_usage_details(self)


@dataclass
class BaseGenerationRequest:
    """Base container for generation request parameters.

    Subclasses can add provider-specific fields while inheriting common ones.
    """

    model: str
    system_prompt: str = "You are a helpful assistant."
    user_prompt: str = "Hello!"
    temperature: Optional[float] = 1.0
    seed: Optional[int] = None
    max_output_tokens: Optional[int] = None
    response_schema: Optional[Any] = None
    # Provider-specific fields can be added in subclasses
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def get_input_dict(self, history_length: Optional[int] = None) -> Dict[str, Any]:
        """Get standardized input dictionary for Langfuse logging."""
        input_dict: Dict[str, Any] = {"system": self.system_prompt}
        if history_length is not None:
            input_dict["history_turns"] = history_length
        input_dict["user"] = self.user_prompt
        return input_dict


class BaseLLMProcessor(ABC):
    """Abstract base class for LLM processors.

    Provides common functionality for Langfuse logging, error handling,
    and utility methods that are shared across all providers.

    Subclasses must implement:
    - send_message() - Synchronous message sending
    - send_message_async() - Asynchronous message sending
    """

    # Provider name for logging - override in subclasses
    provider_name: str = "BaseLLM"

    @staticmethod
    def build_langfuse_usage_details(result: GenerationResult) -> Dict[str, int]:
        """Build usage keys that match the configured Langfuse pricing table."""
        usage_details = {
            "input": result.input_tokens,
            "output": result.output_tokens,
        }

        if result.cached_tokens > 0:
            usage_details["input_cached_tokens"] = result.cached_tokens

        if result.reasoning_tokens > 0:
            usage_details["output_reasoning"] = result.reasoning_tokens

        return usage_details

    @abstractmethod
    def send_message(
        self,
        model: str,
        temperature: Optional[float] = 1.0,
        seed: Optional[int] = 12345,
        system_prompt: str = "You are a helpful assistant.",
        user_prompt: str = "Hello!",
        file_paths: Optional[List[Union[str, Path]]] = None,
        response_schema: Optional[Dict[str, Any]] = None,
        history: Optional[List[Any]] = None,
        thinking_budget: Optional[Any] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Send a message to the LLM and get a response."""
        ...

    @abstractmethod
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
        thinking_budget: Optional[Any] = None,
        max_output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Union[str, Dict[str, Any]]:
        """Send a message to the LLM asynchronously."""
        ...

    def _detect_image_mime(self, img_bytes: bytes, fallback: str = "application/octet-stream") -> str:
        """Detect image mime type from magic bytes.

        Uses the shared utility function.

        Args:
            img_bytes: Raw image bytes to analyze.
            fallback: MIME type to return if format cannot be detected.

        Returns:
            MIME type string.
        """
        return detect_image_mime(img_bytes, fallback=fallback)

    @staticmethod
    def _mime_to_ext(mime_type: str) -> str:
        """Map MIME types to file extensions for saving images."""
        mime_map = {
            "image/png": "png",
            "image/jpeg": "jpg",
            "image/webp": "webp",
            "image/gif": "gif",
        }
        return mime_map.get(mime_type, "bin")

    @staticmethod
    @lru_cache(maxsize=1)
    def _find_project_root() -> Optional[Path]:
        """Find project root by locating pyproject.toml (cached)."""
        try:
            current = Path(__file__).resolve()
            for parent in [current] + list(current.parents):
                if (parent / "pyproject.toml").is_file():
                    return parent
            return None
        except Exception as exc:
            logger.opt(exception=True).warning(f"[LLM] Failed to locate project root: {exc}")
            return None

    @classmethod
    def _get_project_tmp_dir(cls) -> Optional[Path]:
        """Get (and create) the project-root tmp directory."""
        if not _SAVE_LLM_ARTIFACTS:
            return None
        project_root = cls._find_project_root()
        if project_root is None:
            return None
        try:
            tmp_dir = project_root / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            return tmp_dir
        except Exception as exc:
            logger.opt(exception=True).warning(f"[LLM] Failed to initialize tmp dir: {exc}")
            return None

    def _save_image_bytes_list(self, image_bytes_list: List[bytes], *, provider: str, context: str) -> List[Path]:
        """Persist inline image bytes to the project tmp directory.

        Skipped in production (LOCAL_DEV != 1) to avoid unnecessary disk I/O.
        """
        if not _SAVE_LLM_ARTIFACTS or not image_bytes_list:
            return []

        tmp_dir = self._get_project_tmp_dir()
        if tmp_dir is None:
            return []

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        saved_paths: List[Path] = []
        for idx, img_bytes in enumerate(image_bytes_list):
            if not img_bytes:
                logger.warning(f"[LLM] Skipping empty image bytes ({provider}, {context}, index={idx})")
                continue
            mime_type = self._detect_image_mime(img_bytes)
            ext = self._mime_to_ext(mime_type)
            filename = f"{timestamp}_{provider}_{context}_{idx + 1}_{uuid.uuid4().hex[:8]}.{ext}"
            file_path = tmp_dir / filename
            try:
                file_path.write_bytes(img_bytes)
                saved_paths.append(file_path)
            except Exception as exc:
                logger.opt(exception=True).warning(f"[LLM] Failed to save image to {file_path}: {exc}")

        if saved_paths:
            logger.info(f"[LLM] Saved {len(saved_paths)} images to {tmp_dir}")
        return saved_paths

    def _save_pdf_file(self, file_path: Union[str, Path], *, provider: str, context: str) -> Optional[Path]:
        """Persist a PDF file to the project tmp directory.

        Skipped in production (LOCAL_DEV != 1) to avoid unnecessary disk I/O.
        """
        if not _SAVE_LLM_ARTIFACTS:
            return None

        tmp_dir = self._get_project_tmp_dir()
        if tmp_dir is None:
            return None

        p = Path(file_path)
        if not p.exists() or not p.is_file():
            logger.warning(f"[LLM] PDF path does not exist: {p}")
            return None
        if p.suffix.lower() != ".pdf":
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{provider}_{context}_{uuid.uuid4().hex[:8]}.pdf"
        dest = tmp_dir / filename
        try:
            shutil.copy2(p, dest)
            logger.info(f"[LLM] Saved PDF to {dest}")
            return dest
        except Exception as exc:
            logger.opt(exception=True).warning(f"[LLM] Failed to save PDF {p} to {dest}: {exc}")
            return None

    @staticmethod
    def _serialize_response_schema(response_schema: Optional[Any]) -> Optional[Any]:
        """Convert a response schema into loggable metadata."""
        if response_schema is None:
            return None
        if hasattr(response_schema, "model_json_schema"):
            return response_schema.model_json_schema()
        if isinstance(response_schema, dict):
            return response_schema
        return None

    def _log_usage(
        self,
        *,
        model: str,
        input_dict: Dict[str, Any],
        output: Union[str, Dict[str, Any]],
        result: GenerationResult,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        response_schema: Optional[Any] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
        status_message: Optional[str] = None,
    ) -> None:
        """Log usage metrics to Langfuse.

        This method provides standardized logging across all providers.

        Args:
            model: The model used for generation.
            input_dict: Dictionary with system prompt, user prompt, and optional history.
            output: The output text or parsed JSON.
            result: GenerationResult with token counts and metadata.
            temperature: Temperature used for generation.
            seed: Seed used for generation.
            response_schema: Optional schema used for structured output.
            extra_metadata: Additional provider-specific metadata.
        """
        # Get output text
        output_text = output if isinstance(output, str) else json.dumps(output)

        usage_details = self.build_langfuse_usage_details(result)

        # Build metadata
        metadata: Dict[str, Any] = {
            "temperature": temperature,
            "seed": seed,
            "finish_reason": result.finish_reason,
            "provider": self.provider_name,
        }

        # Add thoughts if present
        if result.thoughts:
            metadata["thoughts"] = result.thoughts

        schema_metadata = self._serialize_response_schema(response_schema)
        if schema_metadata is not None:
            metadata["response_schema"] = schema_metadata

        # Add response_id if present (OpenAI-specific)
        if result.response_id:
            metadata["response_id"] = result.response_id

        # Merge extra metadata
        if extra_metadata:
            metadata.update(extra_metadata)

        observation_params: Dict[str, Any] = {
            "input": input_dict,
            "output": output_text,
            "model": model,
            "usage_details": usage_details,
            "metadata": metadata,
        }

        if status_message:
            observation_params["status_message"] = status_message

        # Add warning level if finish reason indicates incomplete or abnormal generation
        if result.finish_reason and result.finish_reason not in ("STOP", "stop"):
            observation_params["level"] = "WARNING"
            observation_params["status_message"] = (
                f"Generation finished with reason '{result.finish_reason}' (expected 'STOP')"
            )
            logger.warning(
                f"{self.provider_name} finish_reason warning: Generation finished with reason "
                f"'{result.finish_reason}' (expected 'STOP')"
            )

        langfuse_context.update_current_observation(**observation_params)

    def _log_error(
        self,
        *,
        model: str,
        input_dict: Dict[str, Any],
        error: Exception,
        result: Optional[GenerationResult] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        response_schema: Optional[Any] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log observation when an error occurs, so Langfuse shows proper input/output.

        Args:
            model: The model used for generation.
            input_dict: Dictionary with system prompt, user prompt, and optional history.
            error: The exception that occurred.
            result: Optional partial result if error occurred after LLM call.
            temperature: Temperature used for generation.
            seed: Seed used for generation.
            response_schema: Optional schema used for structured output.
            extra_metadata: Additional provider-specific metadata.
        """
        # Build metadata
        metadata: Dict[str, Any] = {
            "temperature": temperature,
            "seed": seed,
            "error_type": type(error).__name__,
            "provider": self.provider_name,
        }

        # Add response schema if present
        if response_schema is not None:
            if hasattr(response_schema, "model_json_schema"):
                metadata["response_schema"] = response_schema.model_json_schema()
            elif isinstance(response_schema, dict):
                metadata["response_schema"] = response_schema

        # Add finish reason if we have a result
        if result is not None:
            metadata["finish_reason"] = result.finish_reason

        # Merge extra metadata
        if extra_metadata:
            metadata.update(extra_metadata)

        observation_params: Dict[str, Any] = {
            "input": input_dict,
            "model": model,
            "level": "ERROR",
            "status_message": str(error),
            "metadata": metadata,
        }

        # If we have a result (error occurred after LLM call), extract what we can
        if result is not None:
            output_text = result.content if isinstance(result.content, str) else json.dumps(result.content)
            observation_params["output"] = output_text
            observation_params["usage_details"] = self.build_langfuse_usage_details(result)
            if result.thoughts:
                observation_params["metadata"]["thoughts"] = result.thoughts
        else:
            observation_params["output"] = f"[No response - {type(error).__name__}: {str(error)}]"

        langfuse_context.update_current_observation(**observation_params)

        logger.debug(f"{self.provider_name} error: {type(error).__name__}: {error}")
