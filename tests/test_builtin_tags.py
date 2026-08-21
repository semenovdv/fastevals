import pytest

from fastevals import builtin_tags as bt
from fastevals.builtin_tags import BUILTIN_TAG_NAMES, builtin_selectors
from fastevals.config import RunConfig
from fastevals.exceptions import ConfigError
from fastevals.tags import resolve_tag, save_tag

REGISTRY = {
    "openai:gpt-5.6-luna": {
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "reasoning_efforts": "none|low|high",
        "input_cost_usd_per_mtok": 1.0,
        "output_cost_usd_per_mtok": 6.0,
    },
    "openai:gpt-5.6-terra": {
        "provider": "openai",
        "model": "gpt-5.6-terra",
        "reasoning_efforts": "off|low",
        "input_cost_usd_per_mtok": 2.0,
        "output_cost_usd_per_mtok": 15.0,
    },
    "openai:gpt-5.6-sol": {
        "provider": "openai",
        "model": "gpt-5.6-sol",
        "reasoning_efforts": "none|low|medium|high|xhigh",
        "input_cost_usd_per_mtok": 5.0,
        "output_cost_usd_per_mtok": 30.0,
    },
}


def test_builtin_names_are_stable_and_documented():
    assert BUILTIN_TAG_NAMES == ("auto-cheap", "auto-deep", "auto-fast", "auto-flagship")
    assert all(bt.builtin_description(name) for name in BUILTIN_TAG_NAMES)


def test_auto_fast_picks_lightest_effort_per_model():
    assert builtin_selectors("auto-fast", REGISTRY) == [
        "openai/gpt-5.6-luna@none",
        "openai/gpt-5.6-terra@off",
        "openai/gpt-5.6-sol@none",
    ]


def test_auto_deep_picks_deepest_effort_per_model():
    assert builtin_selectors("auto-deep", REGISTRY) == [
        "openai/gpt-5.6-luna@high",
        "openai/gpt-5.6-terra@low",
        "openai/gpt-5.6-sol@xhigh",
    ]


def test_auto_cheap_is_single_lightest_cell_of_cheapest_model():
    assert builtin_selectors("auto-cheap", REGISTRY) == ["openai/gpt-5.6-luna@none"]


def test_auto_flagship_is_priciest_model_all_efforts():
    assert builtin_selectors("auto-flagship", REGISTRY) == ["openai/gpt-5.6-sol"]


def test_unknown_name_returns_none_not_error():
    assert builtin_selectors("nope", REGISTRY) is None


def test_known_builtin_on_empty_registry_raises():
    with pytest.raises(ConfigError, match="non-empty model registry"):
        builtin_selectors("auto-fast", {})


def test_user_tag_shadows_nothing_builtin_is_fallback():
    selectors = resolve_tag("auto-fast", entries=REGISTRY)
    assert selectors == builtin_selectors("auto-fast", REGISTRY)


def test_save_tag_rejects_reserved_builtin_names(tmp_path):
    with pytest.raises(ConfigError, match="reserved"):
        save_tag("auto-fast", ["openai/gpt-5.6-luna"], path=tmp_path / "tags.toml")


def test_resolve_unknown_lists_saved_and_builtin(tmp_path):
    save_tag("mine", ["openai/gpt-5.6-luna"], path=tmp_path / "tags.toml")
    with pytest.raises(ConfigError) as excinfo:
        resolve_tag("nope", path=tmp_path / "tags.toml", entries=REGISTRY)
    message = str(excinfo.value)
    assert "saved: mine" in message
    assert "built-in: auto-cheap, auto-deep, auto-fast, auto-flagship" in message


def test_run_config_tag_and_models_mutually_exclusive():
    with pytest.raises(ConfigError, match="not both"):
        RunConfig(prompt="x", tag="auto-fast", models=frozenset({"openai/gpt-5.6-luna@low"}))


def test_run_config_accepts_tag_alone():
    config = RunConfig(prompt="x", tag="auto-fast")
    assert config.tag == "auto-fast"
