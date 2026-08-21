"""Typed configuration objects for runs and model registry entries."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .exceptions import ConfigError

__all__ = [
    "ALL_PROVIDERS",
    "DEFAULT_MAX_CONCURRENCY",
    "DEFAULT_TIMEOUT_S",
    "SUPPORTED_PROVIDERS",
    "ModelSpec",
    "RunConfig",
]

ALL_PROVIDERS = "all"
SUPPORTED_PROVIDERS = ("openai", "gemini", "openrouter", "mock")
DEFAULT_TIMEOUT_S = 120
DEFAULT_MAX_CONCURRENCY = 4
DEFAULT_OUT_DIR = "runs"
MAX_ATTACHMENT_BYTES = 20 * 1024 * 1024

_KNOWN_SPEC_KEYS = frozenset(
    {
        "id",
        "provider",
        "model",
        "api_key_env",
        "reasoning_effort",
        "reasoning_efforts",
        "reasoning_parameter",
        "input_cost_usd_per_mtok",
        "cached_input_cost_usd_per_mtok",
        "cached_cost_usd_per_mtok",
        "output_cost_usd_per_mtok",
        "reasoning_cost_usd_per_mtok",
        "timeout_s",
        "response",
    }
)


@dataclass(frozen=True)
class RunConfig:
    """Everything needed to execute one evaluation run."""

    prompt: str
    providers: frozenset[str] = frozenset({ALL_PROVIDERS})
    file: str | None = None
    image: str | None = None
    structured_output: dict[str, Any] | None = None
    dataset: str | None = None
    nruns: int = 1
    registry: str | None = None
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY
    out: str = DEFAULT_OUT_DIR

    def __post_init__(self) -> None:
        if not self.prompt.strip() and not self.dataset:
            raise ConfigError("Prompt must not be empty when no dataset is given")
        if self.max_concurrency < 1:
            raise ConfigError(f"max_concurrency must be >= 1, got {self.max_concurrency}")
        if self.nruns < 1:
            raise ConfigError(f"nruns must be >= 1, got {self.nruns}")
        for label, path in (("file", self.file), ("image", self.image), ("dataset", self.dataset)):
            if path and not Path(path).exists():
                raise ConfigError(f"{label} not found: {path}")
        unknown = self.requested_providers() - set(SUPPORTED_PROVIDERS) - {ALL_PROVIDERS}
        if unknown:
            supported = ", ".join((*SUPPORTED_PROVIDERS, ALL_PROVIDERS))
            raise ConfigError(f"Unknown provider(s): {', '.join(sorted(unknown))}. Supported: {supported}")
        if self.structured_output is not None:
            schema = self.structured_output
            if not isinstance(schema, dict) or "properties" not in schema:
                raise ConfigError("structured_output must be a JSON Schema object with 'properties'")

    def requested_providers(self) -> set[str]:
        return {provider.lower() for provider in self.providers}


@dataclass(frozen=True)
class ModelSpec:
    """One concrete provider/model/reasoning combination from the registry."""

    id: str
    provider: str
    model: str
    api_key_env: str | None = None
    reasoning_effort: str = "off"
    reasoning_parameter: str | None = None
    input_cost_usd_per_mtok: float | None = None
    cached_input_cost_usd_per_mtok: float | None = None
    cached_cost_usd_per_mtok: float | None = None
    output_cost_usd_per_mtok: float | None = None
    reasoning_cost_usd_per_mtok: float | None = None
    timeout_s: int = DEFAULT_TIMEOUT_S
    response: str | None = None

    @property
    def is_mock(self) -> bool:
        return self.provider == "mock"

    @classmethod
    def from_dict(cls, raw: dict[str, Any], spec_id: str) -> "ModelSpec":
        unknown = set(raw) - _KNOWN_SPEC_KEYS - {"type"}
        if unknown:
            raise ConfigError(f"Unknown key(s) in registry entry '{spec_id}': {', '.join(sorted(unknown))}")
        if not isinstance(raw.get("model"), str) or not raw["model"]:
            raise ConfigError(f"Registry entry '{spec_id}' is missing a valid 'model'")
        provider = raw.get("provider", spec_id.split(":", 1)[0])
        if not provider:
            raise ConfigError(f"Registry entry '{spec_id}' is missing a 'provider'")
        fields: dict[str, Any] = {"id": spec_id, "provider": str(provider).lower(), "model": raw["model"]}
        for key in (
            "api_key_env",
            "reasoning_effort",
            "reasoning_parameter",
            "input_cost_usd_per_mtok",
            "cached_input_cost_usd_per_mtok",
            "cached_cost_usd_per_mtok",
            "output_cost_usd_per_mtok",
            "reasoning_cost_usd_per_mtok",
            "response",
        ):
            if raw.get(key) is not None:
                fields[key] = raw[key]
        if raw.get("timeout_s") is not None:
            try:
                fields["timeout_s"] = max(1, int(raw["timeout_s"]))
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"Registry entry '{spec_id}' has invalid timeout_s: {raw['timeout_s']!r}") from exc
        return cls(**fields)
