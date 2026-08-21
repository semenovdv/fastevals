"""Exception hierarchy for fastevals."""

__all__ = ["ConfigError", "FastEvalError", "ProviderError", "StructuredOutputError"]


class FastEvalError(Exception):
    """Base class for all fastevals errors."""


class ConfigError(FastEvalError):
    """Invalid configuration, registry, or provider selection."""


class ProviderError(FastEvalError):
    """A model call failed: missing credentials, network, or provider error."""


class StructuredOutputError(FastEvalError):
    """Model output did not satisfy the requested JSON Schema."""
