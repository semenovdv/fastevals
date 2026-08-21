"""Model registry: TOML loading, validation, and provider selection."""

import tomllib
from pathlib import Path
from typing import Any

from .config import ALL_PROVIDERS, ModelSpec
from .exceptions import ConfigError

__all__ = ["default_registry_path", "load_registry", "select_specs"]


def default_registry_path() -> Path | None:
    """Prefer a registry in the working directory, fall back to the repo copy."""
    cwd_candidate = Path.cwd() / "config" / "models.toml"
    if cwd_candidate.exists():
        return cwd_candidate
    package_candidate = Path(__file__).resolve().parents[1] / "config" / "models.toml"
    return package_candidate if package_candidate.exists() else None


def load_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the flat TOML registry keyed by ``provider:model``."""
    path = Path(path)
    with path.open("rb") as config_file:
        data = tomllib.load(config_file)
    if not data:
        raise ConfigError(f"Model registry is empty: {path}")
    invalid = [key for key, value in data.items() if not isinstance(value, dict) or not value.get("model")]
    if invalid:
        raise ConfigError(f"Invalid model registry entries: {', '.join(invalid)}")
    return data


def _expand_efforts(raw: dict[str, Any]) -> list[dict[str, Any]]:
    raw_efforts = raw.get("reasoning_efforts")
    if isinstance(raw_efforts, str) and raw_efforts.strip():
        efforts = [item.strip() for item in raw_efforts.split("|") if item.strip()] or ["off"]
    else:
        efforts = [raw.get("reasoning_effort", "off")]
    return [{**raw, "reasoning_effort": effort} for effort in efforts]


def select_specs(entries: dict[str, dict[str, Any]], requested: set[str]) -> list[ModelSpec]:
    """Resolve requested providers to concrete, effort-expanded model specs.

    ``all`` selects every provider present in the registry; otherwise only
    entries matching the requested providers are kept.
    """
    if ALL_PROVIDERS in requested or not requested:
        rows = [dict(entry, id=model_id) for model_id, entry in entries.items()]
    else:
        rows = [
            dict(entry, id=model_id)
            for model_id, entry in entries.items()
            if str(entry.get("provider", "")).lower() in requested
        ]
    if not rows:
        raise ConfigError(
            f"No models found for provider(s): {', '.join(sorted(requested))}. "
            "Add entries to the model registry or pick another provider."
        )
    return [ModelSpec.from_dict(expanded, _spec_id(expanded)) for expanded in _expand_all(rows)]


def _spec_id(expanded: dict[str, Any]) -> str:
    return f"{expanded.get('provider', 'unknown')}:{expanded['model']}:{expanded.get('reasoning_effort', 'off')}"


def _expand_all(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        for effort_row in _expand_efforts(row):
            expanded.append({**effort_row})
    return expanded
