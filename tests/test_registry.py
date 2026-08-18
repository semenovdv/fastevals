from pathlib import Path

import pytest

from fasteval.runner import _expand_reasoning_efforts, _load_registry


def test_load_registry_reads_models_toml():
    registry_path = Path(__file__).resolve().parents[1] / "config" / "models.toml"
    registry = _load_registry(registry_path)
    assert "openai:gpt-5.6-luna" in registry
    assert registry["openai:gpt-5.6-luna"]["provider"] == "openai"


def test_load_registry_rejects_invalid_entries(tmp_path: Path):
    bad_registry = tmp_path / "models.toml"
    bad_registry.write_text('["missing-fields"]\nfoo = "bar"\n')
    with pytest.raises(ValueError, match="Invalid model registry entries"):
        _load_registry(bad_registry)


def test_expand_reasoning_efforts_splits_pipe_separated_values():
    models = [{"provider": "openai", "model": "gpt-test", "reasoning_efforts": "none|low|high"}]
    expanded = _expand_reasoning_efforts(models)
    assert [model["reasoning_effort"] for model in expanded] == ["none", "low", "high"]
