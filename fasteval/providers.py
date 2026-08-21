"""Provider adapters: one documented contract, mock + LiteLLM implementations."""

import base64
import mimetypes
import os
from pathlib import Path
from typing import Any

from .config import ModelSpec
from .exceptions import ProviderError
from .models import ModelResponse
from .structured import mocked_answer

__all__ = ["call_model", "parse_response", "scrub_secrets"]

# LiteLLM needs an explicit routing prefix for these providers.
PROVIDER_PREFIXES = {"openrouter": "openrouter/", "gemini": "gemini/"}

try:  # LiteLLM is optional; mock runs and report work without it.
    from litellm import acompletion as _litellm_completion
except ImportError:  # pragma: no cover - exercised via import guard test
    _litellm_completion = None


def litellm_model_name(provider: str, model: str) -> str:
    prefix = PROVIDER_PREFIXES.get(provider.lower(), "")
    return model if not prefix or model.startswith(prefix) else prefix + model


def scrub_secrets(message: str) -> str:
    """Replace API key values found in ``message`` with ``***``."""
    for name, value in os.environ.items():
        if name.endswith("_API_KEY") and value and value in message:
            message = message.replace(value, "***")
    return message


def _image_part(path: str) -> dict[str, Any]:
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}


def build_messages(prompt: str, file_paths: list[str] | None) -> list[dict[str, Any]]:
    content: str | list[dict[str, Any]] = prompt
    if file_paths:
        content = [{"type": "text", "text": prompt}]
        content.extend(_image_part(path) for path in file_paths)
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


def _mock_response(spec: ModelSpec, prompt: str, response_schema: dict[str, Any] | None) -> ModelResponse:
    if response_schema:
        return ModelResponse(text=mocked_answer(response_schema))
    template = spec.response or "[{model}] {prompt}"
    return ModelResponse(text=template.format(model=spec.model, prompt=prompt))


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
        request.setdefault("extra_headers", {"HTTP-Referer": "https://github.com/semenovdv/fasteval"})
    return request


async def call_model(
    spec: ModelSpec,
    prompt: str,
    file_path: str | None = None,
    image_path: str | None = None,
    response_schema: dict[str, Any] | None = None,
) -> ModelResponse:
    """Execute one model call. Mock specs never touch the network."""
    if spec.is_mock:
        return _mock_response(spec, prompt, response_schema)
    if _litellm_completion is None:
        raise ProviderError('LiteLLM is not installed. Install provider support: pip install "fasteval[native]"')
    request = _build_request(spec, prompt, file_path, image_path, response_schema)
    response = await _litellm_completion(**request)
    return parse_response(response)
