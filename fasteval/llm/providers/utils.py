"""Shared utilities for LLM providers.

This module contains common functionality used across multiple LLM providers
to eliminate code duplication and ensure consistent behavior.
"""

from typing import Callable

from langfuse.decorators import langfuse_context
from loguru import logger
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)


def before_sleep_log_loguru(retry_state: RetryCallState) -> None:
    """Log retry attempts using loguru before sleeping.

    This is a tenacity callback function that logs information about
    retry attempts in a consistent format across all LLM providers.

    Args:
        retry_state: The current retry state from tenacity.
    """
    logger.info(
        "Retrying {func} in {wait:.2f}s, attempt {attempt} after {exc}",
        func=retry_state.fn.__name__ if retry_state.fn else "unknown",
        wait=retry_state.next_action.sleep if retry_state.next_action else 0,
        attempt=retry_state.attempt_number,
        exc=retry_state.outcome.exception() if retry_state.outcome else "unknown",
    )


def before_sleep_log_with_langfuse(retry_state: RetryCallState) -> None:
    """Log retry attempts to both loguru and Langfuse before sleeping.

    This is a tenacity callback function that logs retry information to:
    1. Loguru for application logs
    2. Langfuse for observability and tracing

    Args:
        retry_state: The current retry state from tenacity.
    """
    func_name = retry_state.fn.__name__ if retry_state.fn else "unknown"
    wait_time = retry_state.next_action.sleep if retry_state.next_action else 0
    attempt = retry_state.attempt_number
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    exc_str = str(exception) if exception else "unknown"
    exc_type = type(exception).__name__ if exception else "unknown"

    # Log to loguru
    logger.debug(
        "Retrying {func} in {wait:.2f}s, attempt {attempt} after {exc_type}: {exc}",
        func=func_name,
        wait=wait_time,
        attempt=attempt,
        exc_type=exc_type,
        exc=exc_str,
    )

    # Log to Langfuse as an event on the current observation
    try:
        langfuse_context.update_current_observation(
            metadata={
                f"retry_attempt_{attempt}": {
                    "function": func_name,
                    "wait_seconds": round(wait_time, 2),
                    "error_type": exc_type,
                    "error_message": exc_str[:500],  # Truncate long error messages
                }
            }
        )
    except Exception:
        # Don't fail the retry if Langfuse logging fails
        logger.debug("Failed to log retry to Langfuse")


def detect_image_mime(img_bytes: bytes, fallback: str = "application/octet-stream") -> str:
    """Detect image MIME type from magic bytes.

    Supports detection of PNG, JPEG, WebP, and GIF formats by examining
    the file signature (magic bytes) at the beginning of the data.

    Args:
        img_bytes: Raw image bytes to analyze.
        fallback: MIME type to return if format cannot be detected.
            Defaults to "application/octet-stream".

    Returns:
        Detected MIME type string, or fallback if unrecognized
        or if bytes are too short to identify.
    """
    if len(img_bytes) < 3:
        return fallback

    # PNG: 8-byte signature
    if len(img_bytes) >= 8 and img_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"

    # JPEG: 2-byte signature (SOI marker)
    if img_bytes[:2] == b"\xff\xd8":
        return "image/jpeg"

    # WebP: "RIFF" at offset 0, "WEBP" at offset 8 (requires 12 bytes)
    if len(img_bytes) >= 12 and img_bytes[:4] == b"RIFF" and img_bytes[8:12] == b"WEBP":
        return "image/webp"

    # GIF: 3-byte signature ("GIF")
    if img_bytes[:3] == b"GIF":
        return "image/gif"

    return fallback


def create_retry_decorator(
    is_retryable: Callable[[Exception], bool],
    max_attempts: int = 3,
    multiplier: float = 1,
    max_wait: int = 60,
) -> Callable:
    """Create a standardized retry decorator for LLM API calls.

    This factory function creates consistent retry decorators across all providers,
    reducing duplication while allowing provider-specific error checking.

    Args:
        is_retryable: Function that determines if an exception should trigger a retry.
        max_attempts: Maximum number of retry attempts. Defaults to 3.
        multiplier: Multiplier for exponential backoff. Defaults to 1.
        max_wait: Maximum wait time between retries in seconds. Defaults to 60.

    Returns:
        A tenacity retry decorator configured with the specified parameters.

    Example:
        @create_retry_decorator(is_retryable_openai)
        async def _call_api(self, params):
            ...
    """
    return retry(
        wait=wait_exponential(multiplier=multiplier, max=max_wait),
        stop=stop_after_attempt(max_attempts),
        retry=retry_if_exception(is_retryable),
        reraise=True,
        before_sleep=before_sleep_log_with_langfuse,
    )


# Standard retry decorator with default settings
# Can be used directly or as a template
DEFAULT_RETRY_DECORATOR = create_retry_decorator(
    is_retryable=lambda e: False,  # Override with provider-specific check
    max_attempts=3,
    multiplier=1,
    max_wait=60,
)


# Type aliases for common patterns
ImageBytes = bytes
MimeType = str
