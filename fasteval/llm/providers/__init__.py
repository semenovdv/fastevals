"""
Provider namespace.

Important: keep this module lightweight. Some providers (e.g. Gemini) pull in heavy
deps and may fail in restricted environments if imported eagerly. We therefore
lazy-import providers on first attribute access.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "BaseLLMProcessor",
    "GenerationResult",
    "ProviderCapabilities",
    "get_capabilities",
    "get_provider",
    "list_providers",
]

from .registry import ProviderCapabilities, get_capabilities, get_provider, list_providers


def __getattr__(name: str) -> Any:  # pragma: no cover
    if name == "BaseLLMProcessor":
        from fasteval.llm.providers.base_processor import BaseLLMProcessor

        return BaseLLMProcessor
    if name == "GenerationResult":
        from fasteval.llm.providers.base_processor import GenerationResult

        return GenerationResult
    if name == "GeminiProcessor":
        from fasteval.llm.providers.gemini import GeminiProcessor

        return GeminiProcessor
    if name == "OpenaiProcessor":
        from fasteval.llm.providers.openai_provider import OpenaiProcessor

        return OpenaiProcessor
    if name == "OpenRouterProcessor":
        from fasteval.llm.providers.openrouter import OpenRouterProcessor

        return OpenRouterProcessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()) + __all__)
