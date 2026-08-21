"""Fast, provider-agnostic LLM evaluation toolkit."""

from .config import ModelSpec, RunConfig, SUPPORTED_PROVIDERS
from .exceptions import ConfigError, FastEvalError, ProviderError, StructuredOutputError
from .models import ModelResponse, RunResult
from .registry import load_registry
from .report import save_report
from .runner import run

__version__ = "0.1.0"

__all__ = [
    "ConfigError",
    "FastEvalError",
    "ModelResponse",
    "ModelSpec",
    "ProviderError",
    "RunConfig",
    "RunResult",
    "SUPPORTED_PROVIDERS",
    "StructuredOutputError",
    "__version__",
    "load_registry",
    "run",
    "save_report",
]
