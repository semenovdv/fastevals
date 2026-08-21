"""Built-in tags: standard suites computed from whatever registry you have.

Unlike saved tags these are recipes, not stored data — they always reflect
the current registry, so they work on day zero and never go stale. Names are
reserved: user-defined tags cannot shadow them.
"""

from typing import Any

from .exceptions import ConfigError

__all__ = ["BUILTIN_TAG_NAMES", "builtin_description", "builtin_selectors"]

_EFFORT_SCALE = ("off", "none", "low", "medium", "high", "xhigh", "max")


def _effort_rank(effort: str) -> int:
    try:
        return _EFFORT_SCALE.index(effort.lower())
    except ValueError:
        return len(_EFFORT_SCALE)


def _entry_rows(entries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for _model_id, entry in entries.items():
        efforts_raw = entry.get("reasoning_efforts") or entry.get("reasoning_effort") or "off"
        efforts = [item.strip().lower() for item in str(efforts_raw).split("|") if item.strip()] or ["off"]
        rows.append(
            {
                "provider": str(entry.get("provider", "")).lower(),
                "model": str(entry.get("model", "")),
                "efforts": efforts,
                "input_cost": float(entry.get("input_cost_usd_per_mtok") or 0.0),
                "output_cost": float(entry.get("output_cost_usd_per_mtok") or 0.0),
            }
        )
    return rows


def _selector(row: dict[str, Any], effort: str | None = None) -> str:
    base = f"{row['provider']}/{row['model']}"
    return f"{base}@{effort}" if effort else base


def builtin_selectors(name: str, entries: dict[str, dict[str, Any]]) -> list[str] | None:
    """Expand a built-in tag name into selectors, or ``None`` if unknown.

    Raises :class:`ConfigError` for known built-ins that cannot apply (for
    example an empty registry).
    """
    if name.strip().lower() not in _DESCRIPTIONS:
        return None
    rows = _entry_rows(entries)
    if not rows:
        raise ConfigError(f"Built-in tag '{name}' needs a non-empty model registry")

    if name == "auto-fast":
        # One cheapest-thinking cell per model.
        return [_selector(row, min(row["efforts"], key=_effort_rank)) for row in rows]

    if name == "auto-deep":
        # One deepest-reasoning cell per model.
        return [_selector(row, max(row["efforts"], key=_effort_rank)) for row in rows]

    if name == "auto-cheap":
        row = min(rows, key=lambda r: (r["input_cost"] + r["output_cost"], r["model"]))
        return [_selector(row, min(row["efforts"], key=_effort_rank))]

    if name == "auto-flagship":
        row = max(rows, key=lambda r: (r["input_cost"] + r["output_cost"], r["model"]))
        return [_selector(row)]

    return None


_DESCRIPTIONS: dict[str, str] = {
    "auto-fast": "built-in: one fastest cell per registered model",
    "auto-deep": "built-in: one deepest-reasoning cell per registered model",
    "auto-cheap": "built-in: the cheapest model at its lightest effort",
    "auto-flagship": "built-in: the most expensive model across all efforts",
}


def builtin_description(name: str) -> str:
    return _DESCRIPTIONS.get(name.strip().lower(), "")


BUILTIN_TAG_NAMES: tuple[str, ...] = tuple(sorted(_DESCRIPTIONS))


def ensure_not_reserved(name: str) -> None:
    if name.strip().lower() in BUILTIN_TAG_NAMES:
        raise ConfigError(
            f"Tag name '{name}' is reserved for built-in suites ({', '.join(BUILTIN_TAG_NAMES)}); choose another name"
        )
