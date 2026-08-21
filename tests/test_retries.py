import pytest

from fastevals import providers
from fastevals.config import ModelSpec


def make_spec(max_retries: int = 2) -> ModelSpec:
    return ModelSpec(id="openai:gpt-test:off", provider="openai", model="gpt-test", max_retries=max_retries)


@pytest.fixture
def no_delay(monkeypatch):
    sleeps: list[float] = []

    async def instant(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(providers, "_sleep", instant)
    return sleeps


@pytest.mark.asyncio
async def test_transient_failure_is_retried_until_success(monkeypatch, no_delay):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    calls = {"n": 0}

    async def flaky(**request):
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("transient provider hiccup")
        return _fake_response()

    monkeypatch.setattr(providers, "_litellm_completion", flaky)
    response = await providers.call_model(make_spec(), "hi")
    assert response.text == "ok"
    assert calls["n"] == 3
    assert no_delay == [0.5, 1.0]  # exponential backoff between attempts


def _fake_response():
    from types import SimpleNamespace

    usage = SimpleNamespace(
        prompt_tokens=10,
        completion_tokens=5,
        prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
    )
    choice = SimpleNamespace(message=SimpleNamespace(content="ok"), finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=usage, id="resp-1")


@pytest.mark.asyncio
async def test_exhausted_retries_raise_last_error(monkeypatch, no_delay):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    calls = {"n": 0}

    async def always_down(**request):
        calls["n"] += 1
        raise RuntimeError("provider down")

    monkeypatch.setattr(providers, "_litellm_completion", always_down)
    with pytest.raises(RuntimeError, match="provider down"):
        await providers.call_model(make_spec(max_retries=2), "hi")
    assert calls["n"] == 3  # 1 attempt + 2 retries


@pytest.mark.asyncio
async def test_zero_retries_means_single_attempt(monkeypatch, no_delay):
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    async def boom(**request):
        raise RuntimeError("no luck")

    monkeypatch.setattr(providers, "_litellm_completion", boom)
    with pytest.raises(RuntimeError, match="no luck"):
        await providers.call_model(make_spec(max_retries=0), "hi")
    assert no_delay == []
