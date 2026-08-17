"""Provider-neutral LLM connectors used by fasteval."""

from .base import LLMRequest, LLMResponse, LLMServiceError, complete

__all__ = ["LLMRequest", "LLMResponse", "LLMServiceError", "complete"]
