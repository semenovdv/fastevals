from pathlib import Path

import pytest

from fastevals.exceptions import ConfigError
from fastevals.registry import select_specs
from fastevals.tags import load_tags, remove_tag, resolve_tag, save_tag


def test_roundtrip_preserves_unrelated_tags(tmp_path: Path):
    save_tag("b-suite", ["openai/gpt-5.6-luna@low"], "second", path=tmp_path / "tags.toml")
    save_tag("a-suite", ["openai/gpt-5.6-terra"], "first", path=tmp_path / "tags.toml")
    tags = load_tags(tmp_path / "tags.toml")
    assert tags["a-suite"]["description"] == "first"
    assert tags["b-suite"]["models"] == ["openai/gpt-5.6-luna@low"]
    # overwrite one, other survives
    save_tag("a-suite", ["openrouter/meta-llama/llama-4"], path=tmp_path / "tags.toml")
    tags = load_tags(tmp_path / "tags.toml")
    assert tags["a-suite"]["models"] == ["openrouter/meta-llama/llama-4"]
    assert "b-suite" in tags


def test_empty_models_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="at least one model selector"):
        save_tag("empty", ["", "  "], path=tmp_path / "tags.toml")


def test_empty_name_rejected(tmp_path: Path):
    with pytest.raises(ConfigError, match="name must not be empty"):
        save_tag("  ", ["openai/gpt-5.6-luna"], path=tmp_path / "tags.toml")


def test_remove_tag_reports_existence(tmp_path: Path):
    path = tmp_path / "tags.toml"
    save_tag("gone", ["openai/gpt-5.6-luna"], path=path)
    assert remove_tag("gone", path=path) is True
    assert remove_tag("gone", path=path) is False


def test_resolve_unknown_tag_lists_available(tmp_path: Path):
    save_tag("known", ["openai/gpt-5.6-luna"], path=tmp_path / "tags.toml")
    with pytest.raises(ConfigError) as excinfo:
        resolve_tag("nope", path=tmp_path / "tags.toml")
    message = str(excinfo.value)
    assert "saved: known" in message
    assert "built-in: auto-cheap" in message


def test_broken_tags_file_reports_path(tmp_path: Path):
    bad = tmp_path / "tags.toml"
    bad.write_text("[not-a-table")
    with pytest.raises(ConfigError, match="not valid TOML"):
        load_tags(bad)


def test_saved_selectors_stay_resolvable(tmp_path: Path):
    """Tags store raw selectors; they must still resolve against a registry."""
    save_tag("suite", ["openai/gpt-5.6-luna@low", "openai/gpt-5.6-terra"], path=tmp_path / "t.toml")
    selectors = resolve_tag("suite", path=tmp_path / "t.toml")
    specs = select_specs(
        {
            "openai:gpt-5.6-luna": {"provider": "openai", "model": "gpt-5.6-luna", "reasoning_efforts": "none|low"},
            "openai:gpt-5.6-terra": {"provider": "openai", "model": "gpt-5.6-terra", "reasoning_efforts": "low"},
        },
        {"all"},
        selectors=selectors,
    )
    assert len(specs) == 2
