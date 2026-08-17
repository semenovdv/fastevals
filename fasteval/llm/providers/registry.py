"""Lazy provider registry for the public LLM connector API."""

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Type


@dataclass(frozen=True)
class ProviderCapabilities:
    chat: bool = True
    async_chat: bool = True
    streaming: bool = False
    vision: bool = False
    tools: bool = False
    structured_output: bool = False
    reasoning: bool = False


_PROVIDERS: dict[str, tuple[str, str, ProviderCapabilities]] = {
    "openai": ("fasteval.llm.providers.openai_provider", "OpenaiProcessor", ProviderCapabilities(streaming=True, vision=True, tools=True, structured_output=True, reasoning=True)),
    "openrouter": ("fasteval.llm.providers.openrouter", "OpenRouterProcessor", ProviderCapabilities(streaming=True, vision=True, tools=True, structured_output=True, reasoning=True)),
    "gemini": ("fasteval.llm.providers.gemini", "GeminiProcessor", ProviderCapabilities(streaming=True, vision=True, tools=True, structured_output=True, reasoning=True)),
}


def list_providers() -> list[str]:
    return sorted(_PROVIDERS)


def get_provider(name: str, **kwargs: Any) -> Any:
    key = name.strip().lower()
    try:
        module_name, class_name, _ = _PROVIDERS[key]
    except KeyError as exc:
        available = ", ".join(list_providers())
        raise ValueError(f"Unknown provider: {name}. Available: {available}") from exc
    provider_class: Type[Any] = getattr(import_module(module_name), class_name)
    return provider_class(**kwargs)


def get_capabilities(name: str) -> ProviderCapabilities:
    try:
        return _PROVIDERS[name.strip().lower()][2]
    except KeyError as exc:
        raise ValueError(f"Unknown provider: {name}. Available: {', '.join(list_providers())}") from exc
