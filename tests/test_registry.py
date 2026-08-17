from fasteval.llm.providers import get_capabilities, list_providers


def test_provider_registry_is_discoverable_without_provider_sdks():
    assert list_providers() == ["gemini", "openai", "openrouter"]
    assert get_capabilities("openai").structured_output is True
