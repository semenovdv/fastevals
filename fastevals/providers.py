"""Provider adapters: one documented contract over LiteLLM."""

import asyncio
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from .config import MAX_ATTACHMENT_BYTES, ModelSpec
from .exceptions import ConfigError, ProviderError
from .models import ModelResponse

__all__ = ["build_messages", "call_model", "parse_response", "scrub_secrets"]

# LiteLLM needs an explicit routing prefix for these providers.
PROVIDER_PREFIXES = {"openrouter": "openrouter/", "gemini": "gemini/"}

try:  # LiteLLM is optional; config parsing and report rendering work without it.
    from litellm import acompletion as _litellm_completion
except ImportError:  # pragma: no cover - exercised via import guard test
    _litellm_completion = None

RETRY_BASE_DELAY_S = 0.5


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)


def litellm_model_name(provider: str, model: str) -> str:
    prefix = PROVIDER_PREFIXES.get(provider.lower(), "")
    return model if not prefix or model.startswith(prefix) else prefix + model


def scrub_secrets(message: str) -> str:
    """Replace API key values found in ``message`` with ``***``."""
    for name, value in os.environ.items():
        if name.endswith("_API_KEY") and value and value in message:
            message = message.replace(value, "***")
    return message


def _base64_data(path: Path) -> str:
    payload = path.read_bytes()
    if len(payload) > MAX_ATTACHMENT_BYTES:
        raise ConfigError(f"Attachment exceeds {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MB limit: {path}")
    return base64.b64encode(payload).decode("ascii")


def _attachment_part(path: str) -> dict[str, Any]:
    """Encode one attachment for a multimodal message.

    Images become ``image_url`` parts, PDFs become OpenAI-style ``file``
    parts, and anything else is inlined as decoded UTF-8 text.
    """
    file_path = Path(path)
    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    if mime.startswith("image/"):
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{_base64_data(file_path)}"}}
    if mime == "application/pdf":
        return {
            "type": "file",
            "file": {"filename": file_path.name, "file_data": f"data:application/pdf;base64,{_base64_data(file_path)}"},
        }
    try:
        text = file_path.read_text()
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"Unsupported attachment type '{mime}' for {file_path} (images, PDFs and text files are supported)"
        ) from exc
    return {"type": "text", "text": f"--- Attached file: {file_path.name} ---\n{text}"}


def build_messages(prompt: str, file_paths: list[str] | None) -> list[dict[str, Any]]:
    content: str | list[dict[str, Any]] = prompt
    if file_paths:
        content = [{"type": "text", "text": prompt}]
        content.extend(_attachment_part(path) for path in file_paths)
    return [{"role": "user", "content": content}]


def parse_response(response: Any) -> ModelResponse:
    """Normalize a LiteLLM completion into a :class:`ModelResponse`."""
    choice = response.choices[0]
    usage = getattr(response, "usage", None)
    details = getattr(usage, "prompt_tokens_details", None) if usage else None
    cached = getattr(details, "cached_tokens", 0) or 0
    completion_details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning = getattr(completion_details, "reasoning_tokens", 0) or 0
    return ModelResponse(
        text=choice.message.content or "",
        input_tokens=max(0, (getattr(usage, "prompt_tokens", 0) or 0) - cached),
        output_tokens=max(0, (getattr(usage, "completion_tokens", 0) or 0) - reasoning),
        cached_tokens=cached,
        reasoning_tokens=reasoning,
        finish_reason=getattr(choice, "finish_reason", None),
        response_id=getattr(response, "id", None),
    )


def _build_request(
    spec: ModelSpec,
    prompt: str,
    file_path: str | None,
    image_path: str | None,
    response_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    api_key_env = spec.api_key_env or f"{spec.provider.upper()}_API_KEY"
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ProviderError(f"Missing API key environment variable: {api_key_env}")

    file_paths = [path for path in (file_path, image_path) if path]
    request: dict[str, Any] = {
        "model": litellm_model_name(spec.provider, spec.model),
        "messages": build_messages(prompt, file_paths or None),
        "api_key": api_key,
        "timeout": spec.timeout_s,
    }
    if spec.reasoning_effort and spec.reasoning_effort not in ("off", "none"):
        request["reasoning_effort"] = spec.reasoning_effort
    if response_schema:
        request["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": response_schema, "strict": True},
        }
    if spec.provider == "openrouter":
        request.setdefault("extra_headers", {"HTTP-Referer": "https://github.com/semenovdv/fastevals"})
    return request


async def call_model(
    spec: ModelSpec,
    prompt: str,
    file_path: str | None = None,
    image_path: str | None = None,
    response_schema: dict[str, Any] | None = None,
) -> ModelResponse:
    """Execute one model call through the provider adapter.

    Transient provider failures are retried with exponential backoff
    (``1 + max_retries`` attempts); the final exception propagates.
    """
    if _litellm_completion is None:
        raise ProviderError('LiteLLM is not installed. Install provider support: pip install "fastevals"')
    request = _build_request(spec, prompt, file_path, image_path, response_schema)
    attempts = 1 + max(0, spec.max_retries)
    for attempt in range(attempts):
        try:
            response = await _litellm_completion(**request)
            break
        except Exception:
            if attempt == attempts - 1:
                raise
            await _sleep(RETRY_BASE_DELAY_S * (2**attempt))
    return parse_response(response)
