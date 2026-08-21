from pathlib import Path

import pytest

from fastevals.config import ModelSpec
from fastevals.exceptions import ConfigError
from fastevals.registry import _expand_efforts, default_registry_path, load_registry, select_specs


def test_load_registry_reads_models_toml():
    registry_path = Path(__file__).resolve().parents[1] / "config" / "models.toml"
    registry = load_registry(registry_path)
    assert "openai:gpt-5.6-luna" in registry
    assert registry["openai:gpt-5.6-luna"]["provider"] == "openai"


def test_load_registry_rejects_invalid_entries(tmp_path: Path):
    bad_registry = tmp_path / "models.toml"
    bad_registry.write_text('["missing-fields"]\nfoo = "bar"\n')
    with pytest.raises(ConfigError, match="Invalid model registry entries"):
        load_registry(bad_registry)


def test_load_registry_rejects_empty_file(tmp_path: Path):
    empty = tmp_path / "models.toml"
    empty.write_text("")
    with pytest.raises(ConfigError, match="registry is empty"):
        load_registry(empty)


def test_expand_efforts_splits_pipe_separated_values():
    expanded = _expand_efforts({"provider": "openai", "model": "gpt-test", "reasoning_efforts": "none|low|high"})
    assert [row["reasoning_effort"] for row in expanded] == ["none", "low", "high"]


def test_expand_efforts_defaults_to_single_off():
    expanded = _expand_efforts({"provider": "mock", "model": "demo"})
    assert len(expanded) == 1
    assert expanded[0]["reasoning_effort"] == "off"


def test_select_specs_builds_typed_models():
    entries = {
        "openai:gpt-test": {
            "provider": "openai",
            "model": "gpt-test",
            "api_key_env": "OPENAI_API_KEY",
            "reasoning_efforts": "off|low",
            "input_cost_usd_per_mtok": 1.0,
        }
    }
    specs = select_specs(entries, {"openai"})
    assert [spec.reasoning_effort for spec in specs] == ["off", "low"]
    assert all(spec.id.endswith(spec.reasoning_effort) for spec in specs)
    assert specs[0].input_cost_usd_per_mtok == 1.0


def test_select_specs_unknown_provider_raises():
    with pytest.raises(ConfigError, match="No models found"):
        select_specs({}, {"gemini"})


def test_select_spec_rejects_unknown_keys(tmp_path: Path):
    entries = {"openai:typo": {"provider": "openai", "model": "x", "reasoning_efftort": "low"}}
    with pytest.raises(ConfigError, match="Unknown key"):
        select_specs(entries, {"openai"})


def test_default_registry_path_prefers_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    package_fallback = default_registry_path()
    assert package_fallback is not None and package_fallback.exists()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.toml").write_text("")
    assert default_registry_path() == config_dir / "models.toml"


def test_model_spec_from_dict_coerces_and_validates():
    spec = ModelSpec.from_dict({"provider": "openai", "model": "gpt", "timeout_s": "30"}, "openai:gpt")
    assert spec.timeout_s == 30
    with pytest.raises(ConfigError, match="invalid timeout_s"):
        ModelSpec.from_dict({"provider": "openai", "model": "gpt", "timeout_s": "abc"}, "openai:gpt")
    with pytest.raises(ConfigError, match="missing a valid 'model'"):
        ModelSpec.from_dict({"provider": "openai"}, "openai:x")
