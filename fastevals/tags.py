"""Named model suites ("tags"): reusable presets of model selectors.

Tags live in a single TOML file (``~/.config/fastevals/tags.toml`` by
default, override with ``FASTEVAL_TAGS_FILE``) so a user and every agent on
the machine share the same presets. The file is written by hand-rolled
serialization — tags need strings and arrays only, and the core keeps its
one-runtime-dependency discipline.
"""

import json
import os
import re
import tomllib
from pathlib import Path
from typing import Any

from .builtin_tags import BUILTIN_TAG_NAMES, builtin_selectors, ensure_not_reserved
from .exceptions import ConfigError

__all__ = ["default_tags_path", "load_tags", "remove_tag", "save_tag"]

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def default_tags_path() -> Path:
    override = os.environ.get("FASTEVAL_TAGS_FILE")
    if override:
        return Path(override)
    return Path.home() / ".config" / "fastevals" / "tags.toml"


def _toml_string(value: str) -> str:
    # JSON strings are valid TOML basic strings for the characters we store.
    return json.dumps(value, ensure_ascii=False)


def _dump_tags(data: dict[str, dict[str, Any]]) -> str:
    lines: list[str] = []
    for name in sorted(data):
        entry = data[name]
        key = name if _BARE_KEY.match(name) else _toml_string(name)
        lines.append(f"[{key}]")
        description = entry.get("description")
        if description:
            lines.append(f"description = {_toml_string(str(description))}")
        models = ", ".join(_toml_string(model) for model in entry.get("models", []))
        lines.append(f"models = [{models}]")
        lines.append("")
    return "\n".join(lines)


def load_tags(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return all tags as ``{name: {"description": ..., "models": [...]}}``."""
    tags_path = Path(path) if path else default_tags_path()
    if not tags_path.exists():
        return {}
    try:
        data = tomllib.loads(tags_path.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Tags file is not valid TOML ({tags_path}): {exc}") from exc
    tags: dict[str, dict[str, Any]] = {}
    for name, entry in data.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("models"), list):
            raise ConfigError(f"Tag '{name}' in {tags_path} must be a table with a 'models' array")
        tags[name] = {
            "description": entry.get("description"),
            "models": [str(item) for item in entry["models"]],
        }
    return tags


def save_tag(
    name: str,
    models: list[str],
    description: str | None = None,
    path: str | Path | None = None,
) -> None:
    """Create or overwrite one tag, preserving unrelated entries."""
    ensure_not_reserved(name)
    if not name.strip():
        raise ConfigError("Tag name must not be empty")
    clean_models = [model.strip() for model in models if model.strip()]
    if not clean_models:
        raise ConfigError(f"Tag '{name}' needs at least one model selector")
    tags_path = Path(path) if path else default_tags_path()
    tags = load_tags(tags_path)
    tags[name.strip()] = {"description": description, "models": clean_models}
    tags_path.parent.mkdir(parents=True, exist_ok=True)
    tags_path.write_text(_dump_tags(tags))


def remove_tag(name: str, path: str | Path | None = None) -> bool:
    """Delete one tag; returns whether it existed."""
    tags_path = Path(path) if path else default_tags_path()
    tags = load_tags(tags_path)
    if name.strip() not in tags:
        return False
    del tags[name.strip()]
    tags_path.parent.mkdir(parents=True, exist_ok=True)
    tags_path.write_text(_dump_tags(tags))
    return True


def resolve_tag(
    name: str, path: str | Path | None = None, entries: dict[str, dict[str, Any]] | None = None
) -> list[str]:
    """Expand one tag into its selector list.

    Resolution order: user-saved tags first, then the built-in recipes
    (``auto-fast``, ``auto-deep``, ``auto-cheap``, ``auto-flagship``)
    computed against ``entries``. Fails listing everything available.
    """
    key = name.strip()
    tags = load_tags(path)
    if key in tags:
        selectors: list[str] = tags[key]["models"]
        return selectors
    if entries is None:
        from .registry import default_registry_path, load_registry

        entries = load_registry(default_registry_path())
    builtin = builtin_selectors(key, entries)
    if builtin is not None:
        return builtin
    user_available = ", ".join(sorted(tags)) or "none saved yet"
    raise ConfigError(
        f"Unknown tag '{key}'. Available — saved: {user_available}; built-in: {', '.join(BUILTIN_TAG_NAMES)}"
    )
