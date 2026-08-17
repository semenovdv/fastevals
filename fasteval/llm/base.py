from dataclasses import dataclass, field
from typing import Any


class LLMServiceError(RuntimeError):
    """Base error for connector failures."""


@dataclass(frozen=True)
class LLMRequest:
    model: str
    prompt: str
    system_prompt: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    response_schema: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMResponse:
    text: str
    provider: str = ""
    model: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    cost_usd: float | None = None
    raw: Any = None


def complete(prompt: str, model: dict[str, Any], file_paths: list[str] | None = None) -> LLMResponse:
    """Call a configured model through the selected connector."""
    connector = model.get("connector", "native")
    if connector == "mock":
        template = model.get("response", "[{model}] {prompt}")
        return LLMResponse(text=template.format(model=model.get("model", model.get("name", "mock")), prompt=prompt), provider="mock", model=model.get("model", "mock"))
    if connector == "native":
        import os

        provider = model["provider"]
        api_key = os.environ.get(model.get("api_key_env", f"{provider.upper()}_API_KEY"))
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {model.get('api_key_env', f'{provider.upper()}_API_KEY')}")
        from .providers import get_provider

        processor = get_provider(provider, api_key=api_key)
        settings = model.get("settings", {})
        output = processor.send_message(model=model.get("model", model.get("name")), user_prompt=prompt, temperature=settings.get("temperature"), max_output_tokens=settings.get("max_tokens"), thinking_budget=model.get("reasoning_effort"), file_paths=file_paths, return_result=True)
        if hasattr(output, "content"):
            return LLMResponse(text=str(output.content), provider=provider, model=model.get("model", model.get("name", "")), input_tokens=output.input_tokens, output_tokens=output.output_tokens, reasoning_tokens=output.reasoning_tokens, cached_tokens=output.cached_tokens, finish_reason=output.finish_reason, response_id=output.response_id, raw=output.raw_response)
        return LLMResponse(text=output if isinstance(output, str) else str(output), provider=provider, model=model.get("model", model.get("name", "")))
    raise LLMServiceError(f"Unsupported connector: {connector}")
