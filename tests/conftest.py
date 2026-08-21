"""Shared fixtures: an offline LiteLLM replacement for deterministic tests."""

from types import SimpleNamespace

import pytest

from fastevals import providers


@pytest.fixture
def fake_llm(monkeypatch):
    """Install a deterministic fake for ``providers._litellm_completion``.

    Returns a factory: ``calls = fake_llm(text="...", ...)``. Every request
    is recorded into the returned list so tests can assert on what actually
    reached the provider boundary.
    """
    calls: list[dict] = []

    def install(
        text: str = "hello",
        prompt_tokens: int = 100,
        completion_tokens: int = 50,
        cached: int = 10,
        reasoning: int = 5,
        fail: Exception | None = None,
    ) -> list[dict]:
        async def _fake(**request):
            calls.append(request)
            if fail is not None:
                raise fail
            usage = SimpleNamespace(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
            )
            choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")
            return SimpleNamespace(choices=[choice], usage=usage, id="resp-test")

        monkeypatch.setattr(providers, "_litellm_completion", _fake)
        return calls

    return install


@pytest.fixture
def openai_registry(tmp_path):
    """A minimal registry pointing at OpenAI with pricing configured."""
    path = tmp_path / "models.toml"
    path.write_text(
        '["openai:gpt-test"]\n'
        'provider = "openai"\n'
        'model = "gpt-test"\n'
        'api_key_env = "OPENAI_API_KEY"\n'
        'reasoning_efforts = "off|low"\n'
        "input_cost_usd_per_mtok = 1.0\n"
        "cached_input_cost_usd_per_mtok = 0.1\n"
        "output_cost_usd_per_mtok = 2.0\n"
    )
    return path


@pytest.fixture
def api_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
