import pytest

from fastevals.config import ModelSpec
from fastevals.exceptions import ConfigError
from fastevals.registry import parse_selectors, select_specs

ENTRIES = {
    "openai:gpt-5.6-luna": {"provider": "openai", "model": "gpt-5.6-luna", "reasoning_efforts": "none|low|high"},
    "openai:gpt-5.6-terra": {"provider": "openai", "model": "gpt-5.6-terra", "reasoning_efforts": "none|low"},
    "openrouter:meta-llama/llama-4": {
        "provider": "openrouter",
        "model": "meta-llama/llama-4",
        "reasoning_efforts": "off",
    },
}


def ids(specs):
    return sorted(spec.id for spec in specs)


def test_no_selectors_returns_everything_for_providers():
    specs = select_specs(ENTRIES, {"openai"})
    assert len(specs) == 5  # luna 3 efforts + terra 2


def test_qualified_selector_selects_all_its_efforts():
    specs = select_specs(ENTRIES, {"all"}, selectors=["openai/gpt-5.6-luna"])
    assert ids(specs) == [
        "openai:gpt-5.6-luna:high",
        "openai:gpt-5.6-luna:low",
        "openai:gpt-5.6-luna:none",
    ]


def test_effort_filter_after_expansion():
    specs = select_specs(ENTRIES, {"all"}, selectors=["openai/gpt-5.6-luna@none,high"])
    assert ids(specs) == ["openai:gpt-5.6-luna:high", "openai:gpt-5.6-luna:none"]


def test_entry_id_pasted_from_list_models_is_accepted():
    specs = select_specs(ENTRIES, {"all"}, selectors=["openai:gpt-5.6-luna@low"])
    assert ids(specs) == ["openai:gpt-5.6-luna:low"]


def test_multiple_tokens_combine():
    specs = select_specs(ENTRIES, {"all"}, selectors=["openai/gpt-5.6-luna@low", "openai/gpt-5.6-terra"])
    assert ids(specs) == [
        "openai:gpt-5.6-luna:low",
        "openai:gpt-5.6-terra:low",
        "openai:gpt-5.6-terra:none",
    ]


def test_case_insensitive():
    specs = select_specs(ENTRIES, {"all"}, selectors=["OpenAI/GPT-5.6-LUNA@LOW"])
    assert ids(specs) == ["openai:gpt-5.6-luna:low"]


def test_bare_name_is_rejected_as_ambiguous():
    with pytest.raises(ConfigError, match="omits the provider"):
        select_specs(ENTRIES, {"all"}, selectors=["gpt-5.6-luna"])


def test_ambiguous_bare_name_lists_every_provider_serving_it():
    entries = dict(
        ENTRIES,
        **{"azure/gpt-5.6-luna": {"provider": "azure", "model": "gpt-5.6-luna", "reasoning_efforts": "low"}},
    )
    with pytest.raises(ConfigError) as excinfo:
        select_specs(entries, {"all"}, selectors=["gpt-5.6-luna@low"])
    message = str(excinfo.value)
    assert "openai:gpt-5.6-luna" in message
    assert "azure/gpt-5.6-luna" in message


def test_qualified_selector_disambiguates_same_model_across_providers():
    entries = dict(
        ENTRIES,
        **{"azure/gpt-5.6-luna": {"provider": "azure", "model": "gpt-5.6-luna", "reasoning_efforts": "low"}},
    )
    assert len(select_specs(entries, {"all"}, selectors=["openai/gpt-5.6-luna"])) == 3
    assert len(select_specs(entries, {"all"}, selectors=["azure/gpt-5.6-luna"])) == 1


def test_unknown_selector_lists_available_ids():
    with pytest.raises(ConfigError) as excinfo:
        select_specs(ENTRIES, {"openai"}, selectors=["openai/gpt-5.6-cyber@low"])
    assert "Available:" in str(excinfo.value)


def test_empty_model_part_rejected():
    with pytest.raises(ConfigError, match="provider/model required"):
        parse_selectors(["@low"])
    with pytest.raises(ConfigError, match="provider/model required"):
        parse_selectors(["/gpt-5.6-luna"])


def test_empty_effort_list_rejected():
    with pytest.raises(ConfigError, match="effort list is empty"):
        parse_selectors(["openai/gpt-5.6-luna@"])


def test_spec_still_validates_after_selection():
    entries = {"openai:x": {"provider": "openai", "model": "x", "max_retries": "many"}}
    with pytest.raises(ConfigError, match="invalid max_retries"):
        select_specs(entries, {"openai"}, selectors=["openai/x"])


def test_from_dict_still_coerces_max_retries():
    spec = ModelSpec.from_dict({"provider": "openai", "model": "x", "max_retries": 3}, "openai:x")
    assert spec.max_retries == 3
