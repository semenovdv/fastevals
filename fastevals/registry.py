"""Model registry: TOML loading, validation, and provider selection."""

import tomllib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .config import ALL_PROVIDERS, ModelSpec
from .exceptions import ConfigError

__all__ = ["default_registry_path", "describe_registry", "load_registry", "parse_selectors", "select_specs"]


def default_registry_path() -> Path:
    """Resolve the registry: ``./config/models.toml`` wins, bundled data is the fallback.

    The bundled copy ships inside the wheel, so a plain ``pip install``
    works out of the box anywhere.
    """
    cwd_candidate = Path.cwd() / "config" / "models.toml"
    if cwd_candidate.exists():
        return cwd_candidate
    return Path(__file__).resolve().parent / "data" / "models.toml"


def load_registry(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the flat TOML registry keyed by ``provider:model``."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Model registry not found: {path}")
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


def parse_selectors(raw: Iterable[str]) -> list[tuple[str | None, str, set[str] | None]]:
    """Parse model selector tokens into ``(provider, model, efforts)`` triples.

    Grammar: ``provider/model[@effort[,effort...]]`` — the provider is
    **required**, because the same model string is often served by many
    providers and silently fanning out a paid run across all of them would
    be a footgun. ``model`` is the exact official model id (e.g.
    ``gpt-5.6-luna``); the verbatim registry entry id
    (``openai:gpt-5.6-luna``) is accepted as well, so ids from
    ``--list-models`` paste back unchanged. The effort list narrows rows
    after expansion. Selectors are joined with ``|`` at the call site.
    """
    parsed: list[tuple[str | None, str, set[str] | None]] = []
    for token in raw:
        token = token.strip()
        if not token:
            continue
        head, sep, efforts_raw = token.partition("@")
        if ":" in head:  # verbatim entry id: provider:model
            provider, _, model = head.partition(":")
        elif "/" in head:  # canonical form: provider/model
            provider, _, model = head.partition("/")
        else:  # bare name: provider omitted, resolved (and rejected) later
            provider, model = None, head
        model = model.strip().lower()
        if not model or (provider is not None and not provider.strip()):
            raise ConfigError(
                f"Invalid model selector '{token}': provider/model required "
                "(e.g. openai/gpt-5.6-luna@low) — bare names are ambiguous when "
                "several providers serve the same model"
            )
        provider = provider.strip().lower() if provider is not None else None
        efforts: set[str] | None = None
        if sep:
            efforts = {item.strip().lower() for item in efforts_raw.split(",") if item.strip()}
            if not efforts:
                raise ConfigError(f"Invalid model selector '{token}': effort list is empty")
        parsed.append((provider, model, efforts))
    return parsed


def select_specs(
    entries: dict[str, dict[str, Any]],
    requested: set[str],
    selectors: Iterable[str] | None = None,
) -> list[ModelSpec]:
    """Resolve requested providers and model selectors to concrete specs.

    ``all`` selects every provider present in the registry; otherwise only
    entries matching the requested providers are kept. When ``selectors``
    are given, they further narrow the rows by model-name/id substring and
    optional reasoning-effort filters.
    """
    if ALL_PROVIDERS in requested or not requested:
        rows = [dict(entry, id=model_id) for model_id, entry in entries.items()]
    else:
        rows = [
            dict(entry, id=model_id)
            for model_id, entry in entries.items()
            if str(entry.get("provider", "")).lower() in requested
        ]
    expanded = _expand_all(rows)
    selector_tokens = [token.strip() for token in selectors] if selectors else []
    selectors_parsed = parse_selectors(selector_tokens) if selector_tokens else []
    if selectors_parsed:
        for token, (provider, model, _efforts) in zip(selector_tokens, selectors_parsed, strict=True):
            if provider is not None:
                continue
            candidates = sorted(
                {
                    row["id"]
                    for row in expanded
                    if str(row.get("model", "")).lower() == model or row["id"].lower() == model
                }
            )
            if candidates:
                raise ConfigError(
                    f"Selector '{token}' omits the provider. Matching entries: "
                    f"{', '.join(candidates)}. Qualify as provider/model or paste an entry id."
                )
        narrowed = [
            row
            for row in expanded
            if any(
                (provider is None or provider == str(row.get("provider", "")).lower())
                and (model == str(row.get("model", "")).lower() or model == row["id"].lower())
                and (efforts is None or str(row.get("reasoning_effort", "off")).lower() in efforts)
                for provider, model, efforts in selectors_parsed
            )
        ]
        if not narrowed and rows:
            available = sorted({row["id"] for row in expanded})
            wanted = ", ".join(selector_tokens)
            raise ConfigError(f"No models match selector(s): {wanted}. Available: {', '.join(available)}")
        expanded = narrowed
    if not expanded:
        raise ConfigError(
            f"No models found for provider(s): {', '.join(sorted(requested))}. "
            "Add entries to the model registry or pick another provider."
        )
    return [ModelSpec.from_dict(row, _spec_id(row)) for row in expanded]


def describe_registry(entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Human/agent-readable summary of registry entries."""
    return [
        {
            "id": model_id,
            "provider": entry.get("provider"),
            "model": entry.get("model"),
            "reasoning_efforts": entry.get("reasoning_efforts", entry.get("reasoning_effort", "off")),
            "input_cost_usd_per_mtok": entry.get("input_cost_usd_per_mtok"),
            "cached_input_cost_usd_per_mtok": entry.get("cached_input_cost_usd_per_mtok"),
            "output_cost_usd_per_mtok": entry.get("output_cost_usd_per_mtok"),
        }
        for model_id, entry in entries.items()
    ]


def _spec_id(expanded: dict[str, Any]) -> str:
    return f"{expanded.get('provider', 'unknown')}:{expanded['model']}:{expanded.get('reasoning_effort', 'off')}"


def _expand_all(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        for effort_row in _expand_efforts(row):
            expanded.append({**effort_row})
    return expanded
