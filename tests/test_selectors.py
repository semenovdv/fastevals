import pytest

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


def test_exact_model_name_selects_all_its_efforts():
    specs = select_specs(ENTRIES, {"all"}, selectors=["gpt-5.6-luna"])
    assert ids(specs) == [
        "openai:gpt-5.6-luna:high",
        "openai:gpt-5.6-luna:low",
        "openai:gpt-5.6-luna:none",
    ]


def test_effort_filter_after_expansion():
    specs = select_specs(ENTRIES, {"openai"}, selectors=["gpt-5.6-luna@none,high"])
    assert ids(specs) == ["openai:gpt-5.6-luna:high", "openai:gpt-5.6-luna:none"]


def test_provider_qualified_canonical_form():
    specs = select_specs(ENTRIES, {"all"}, selectors=["openai/gpt-5.6-luna@low"])
    assert ids(specs) == ["openai:gpt-5.6-luna:low"]


def test_provider_part_filters_across_providers():
    # bare model matches any provider; qualified one only its own
    entries = dict(
        ENTRIES,
        **{"azure/gpt-5.6-luna": {"provider": "azure", "model": "gpt-5.6-luna", "reasoning_efforts": "low"}},
    )
    assert len(select_specs(entries, {"all"}, selectors=["gpt-5.6-luna"])) == 4
    assert len(select_specs(entries, {"all"}, selectors=["openai/gpt-5.6-luna"])) == 3
    assert len(select_specs(entries, {"all"}, selectors=["azure/gpt-5.6-luna"])) == 1


def test_registry_entry_id_is_accepted_verbatim():
    """Ids printed by --list-models can be pasted back as selectors."""
    specs = select_specs(ENTRIES, {"all"}, selectors=["openai:gpt-5.6-luna@low"])
    assert ids(specs) == ["openai:gpt-5.6-luna:low"]


def test_multiple_tokens_combine():
    specs = select_specs(ENTRIES, {"all"}, selectors=["openai/gpt-5.6-luna@low", "gpt-5.6-terra"])
    assert ids(specs) == [
        "openai:gpt-5.6-luna:low",
        "openai:gpt-5.6-terra:low",
        "openai:gpt-5.6-terra:none",
    ]


def test_case_insensitive():
    specs = select_specs(ENTRIES, {"openai"}, selectors=["OpenAI/GPT-5.6-LUNA@LOW"])
    assert ids(specs) == ["openai:gpt-5.6-luna:low"]


def test_short_nickname_no_longer_matches():
    with pytest.raises(ConfigError, match="No models match"):
        select_specs(ENTRIES, {"openai"}, selectors=["luna"])


def test_wrong_effort_reports_available_ids():
    with pytest.raises(ConfigError) as excinfo:
        select_specs(ENTRIES, {"openai"}, selectors=["gpt-5.6-terra@max"])
    assert "Available:" in str(excinfo.value)


def test_empty_model_part_rejected():
    with pytest.raises(ConfigError, match="model part is empty"):
        parse_selectors(["@low"])


def test_empty_provider_part_rejected():
    with pytest.raises(ConfigError, match="provider and model are required"):
        parse_selectors(["/gpt-5.6-luna"])


def test_empty_effort_list_rejected():
    with pytest.raises(ConfigError, match="effort list is empty"):
        parse_selectors(["gpt-5.6-luna@"])
