import pytest

from fastevals.config import ModelSpec
from fastevals.exceptions import ConfigError
from fastevals.registry import parse_selectors, select_specs

ENTRIES = {
    "openai:gpt-5.6-luna": {"provider": "openai", "model": "gpt-5.6-luna", "reasoning_efforts": "none|low|high"},
    "openai:gpt-5.6-terra": {"provider": "openai", "model": "gpt-5.6-terra", "reasoning_efforts": "none|low"},
    "openrouter:llama-4": {"provider": "openrouter", "model": "meta-llama/llama-4", "reasoning_efforts": "off"},
}


def ids(specs):
    return sorted(spec.id for spec in specs)


def test_no_selectors_returns_everything_for_providers():
    specs = select_specs(ENTRIES, {"openai"})
    assert len(specs) == 5  # luna 3 efforts + terra 2


def test_substring_selector_narrows_by_model_name():
    specs = select_specs(ENTRIES, {"all"}, selectors=["luna"])
    assert ids(specs) == [
        "openai:gpt-5.6-luna:high",
        "openai:gpt-5.6-luna:low",
        "openai:gpt-5.6-luna:none",
    ]


def test_effort_filter_after_expansion():
    specs = select_specs(ENTRIES, {"openai"}, selectors=["luna@none,high"])
    assert ids(specs) == ["openai:gpt-5.6-luna:high", "openai:gpt-5.6-luna:none"]


def test_multiple_tokens_combine():
    specs = select_specs(ENTRIES, {"all"}, selectors=["luna@low", "terra"])
    assert ids(specs) == [
        "openai:gpt-5.6-luna:low",
        "openai:gpt-5.6-terra:low",
        "openai:gpt-5.6-terra:none",
    ]
    # spec ids follow provider:model:effort
    assert all(spec.id.count(":") == 2 for spec in specs)


def test_selector_matches_full_id():
    specs = select_specs(ENTRIES, {"all"}, selectors=["llama"])
    assert ids(specs) == ["openrouter:meta-llama/llama-4:off"]


def test_case_insensitive():
    specs = select_specs(ENTRIES, {"openai"}, selectors=["LUNA@LOW"])
    assert ids(specs) == ["openai:gpt-5.6-luna:low"]


def test_no_match_lists_available_ids():
    with pytest.raises(ConfigError) as excinfo:
        select_specs(ENTRIES, {"openai"}, selectors=["claude"])
    assert "No models match selector(s): claude" in str(excinfo.value)
    assert "Available:" in str(excinfo.value)


def test_effort_filter_never_matches_reports_available():
    with pytest.raises(ConfigError, match="No models match"):
        select_specs(ENTRIES, {"openai"}, selectors=["terra@high"])


def test_empty_model_part_rejected():
    with pytest.raises(ConfigError, match="model part is empty"):
        parse_selectors(["@low"])


def test_efforts_use_comma_separator():
    specs = select_specs(ENTRIES, {"openai"}, selectors=["luna@none,high"])
    assert ids(specs) == ["openai:gpt-5.6-luna:high", "openai:gpt-5.6-luna:none"]


def test_empty_effort_list_rejected():
    with pytest.raises(ConfigError, match="effort list is empty"):
        parse_selectors(["luna@"])


def test_spec_still_validates_after_selection():
    entries = {"openai:x": {"provider": "openai", "model": "x", "max_retries": "many"}}
    with pytest.raises(ConfigError, match="invalid max_retries"):
        select_specs(entries, {"openai"}, selectors=["x"])


def test_from_dict_still_coerces_max_retries():
    spec = ModelSpec.from_dict({"provider": "openai", "model": "x", "max_retries": 3}, "openai:x")
    assert spec.max_retries == 3
