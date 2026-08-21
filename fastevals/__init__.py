"""Fast, provider-agnostic LLM evaluation toolkit."""

from importlib.metadata import PackageNotFoundError, version

from .config import SUPPORTED_PROVIDERS, ModelSpec, RunConfig
from .exceptions import ConfigError, FastEvalError, ProviderError, StructuredOutputError
from .models import ModelResponse, RunResult
from .registry import load_registry
from .report import save_report
from .runner import run_evals
from .tags import load_tags, remove_tag, resolve_tag, save_tag

try:  # Single source of truth is the installed package metadata.
    __version__ = version("fastevals")
except PackageNotFoundError:  # pragma: no cover - running from a bare source tree
    __version__ = "0.0.0.dev0"

__all__ = [
    "SUPPORTED_PROVIDERS",
    "ConfigError",
    "FastEvalError",
    "ModelResponse",
    "ModelSpec",
    "ProviderError",
    "RunConfig",
    "RunResult",
    "StructuredOutputError",
    "__version__",
    "load_registry",
    "load_tags",
    "remove_tag",
    "resolve_tag",
    "run_evals",
    "save_report",
    "save_tag",
]
