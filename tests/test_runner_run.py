import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from fasteval import providers
from fasteval.config import RunConfig
from fasteval.exceptions import ConfigError, FastEvalError
from fasteval.runner import run


def write_registry(tmp_path: Path, content: str) -> str:
    path = tmp_path / "models.toml"
    path.write_text(content)
    return str(path)


def fake_completion(text="hello", prompt_tokens=100, completion_tokens=50, cached=10, reasoning=5):
    async def _fake(**request):
        usage = SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=SimpleNamespace(cached_tokens=cached),
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning),
        )
        choice = SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage, id="resp-1")

    return _fake


MOCK_REGISTRY = """
["mock:alpha"]
provider = "mock"
model = "alpha"
reasoning_efforts = "off|low"
response = "alpha says {prompt}"
"""

OPENAI_REGISTRY = """
["openai:gpt-test"]
provider = "openai"
model = "gpt-test"
api_key_env = "OPENAI_API_KEY"
input_cost_usd_per_mtok = 1.0
cached_input_cost_usd_per_mtok = 0.1
output_cost_usd_per_mtok = 2.0
"""


@pytest.mark.asyncio
async def test_mock_run_expands_reasoning_efforts(tmp_path):
    registry = write_registry(tmp_path, MOCK_REGISTRY)
    config = RunConfig(prompt="hi", providers=frozenset({"mock"}), registry=registry)
    results = await run(config)
    assert [row.reasoning_effort for row in results] == ["off", "low"]
    assert all(row.ok for row in results)
    assert results[0].output == "alpha says hi"


@pytest.mark.asyncio
async def test_builtin_mock_used_when_registry_has_none(tmp_path):
    registry = write_registry(tmp_path, OPENAI_REGISTRY)
    results = await run(RunConfig(prompt="hi", providers=frozenset({"mock"}), registry=registry))
    assert len(results) == 1
    assert results[0].provider == "mock"
    assert results[0].model == "demo"


@pytest.mark.asyncio
async def test_all_excludes_mock_providers(tmp_path):
    registry = write_registry(tmp_path, MOCK_REGISTRY + OPENAI_REGISTRY)
    results = await run(RunConfig(prompt="hi", registry=registry, max_concurrency=2))
    assert {row.provider for row in results} == {"openai"}


@pytest.mark.asyncio
async def test_unknown_provider_raises(tmp_path):
    registry = write_registry(tmp_path, MOCK_REGISTRY)
    with pytest.raises(ConfigError, match="Unknown provider"):
        await run(RunConfig(prompt="hi", providers=frozenset({"nope"}), registry=registry))


@pytest.mark.asyncio
async def test_missing_registry_file_raises(tmp_path):
    with pytest.raises(FastEvalError, match="Model registry not found"):
        await run(RunConfig(prompt="hi", registry=str(tmp_path / "missing.toml")))


@pytest.mark.asyncio
async def test_no_models_for_provider_raises(tmp_path):
    registry = write_registry(tmp_path, OPENAI_REGISTRY)
    with pytest.raises(ConfigError, match="No models found"):
        await run(RunConfig(prompt="hi", providers=frozenset({"gemini"}), registry=registry))


@pytest.mark.asyncio
async def test_partial_failure_keeps_matrix(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    registry = write_registry(tmp_path, MOCK_REGISTRY + OPENAI_REGISTRY)
    results = await run(RunConfig(prompt="hi", providers=frozenset({"mock", "openai"}), registry=registry))
    by_provider = {row.provider: row for row in results}
    assert by_provider["mock"].ok
    assert "OPENAI_API_KEY" in by_provider["openai"].error
    assert by_provider["openai"].output is None


@pytest.mark.asyncio
async def test_structured_output_reaches_provider_request(tmp_path, monkeypatch):
    registry = write_registry(tmp_path, OPENAI_REGISTRY)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    captured = {}
    payload = json.dumps({"name": "Ada"})
    inner = fake_completion(text=payload)

    async def spy(**request):
        captured.update(request)
        return await inner(**request)

    monkeypatch.setattr(providers, "_litellm_completion", spy)
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    config = RunConfig(
        prompt="hi",
        providers=frozenset({"openai"}),
        registry=registry,
        structured_output=schema,
    )
    results = await run(config)

    assert captured["response_format"]["json_schema"]["schema"] == schema
    assert results[0].ok
    assert results[0].output == {"name": "Ada"}


@pytest.mark.asyncio
async def test_structured_output_validates_provider_response(tmp_path, monkeypatch):
    registry = write_registry(tmp_path, OPENAI_REGISTRY)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setattr(providers, "_litellm_completion", fake_completion(text="not json at all"))
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    config = RunConfig(prompt="hi", providers=frozenset({"openai"}), registry=registry, structured_output=schema)
    results = await run(config)
    assert "not valid JSON" in results[0].error


@pytest.mark.asyncio
async def test_mock_honors_structured_schema(tmp_path):
    registry = write_registry(tmp_path, MOCK_REGISTRY)
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "A name"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
        "additionalProperties": False,
    }
    config = RunConfig(prompt="hi", providers=frozenset({"mock"}), registry=registry, structured_output=schema)
    results = await run(config)
    assert results[0].output == {"name": "A name", "age": 1}


@pytest.mark.asyncio
async def test_ttft_is_none_and_costs_use_registry_rates(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setattr(
        providers,
        "_litellm_completion",
        fake_completion(prompt_tokens=1_000_000, completion_tokens=500_000, cached=100_000, reasoning=0),
    )
    registry = write_registry(tmp_path, OPENAI_REGISTRY)
    results = await run(RunConfig(prompt="hi", providers=frozenset({"openai"}), registry=registry))
    row = results[0]
    assert row.time_to_first_token_ms is None
    assert row.tokens_per_second is not None and row.tokens_per_second > 0
    assert row.input_tokens == 900_000
    assert row.cached_tokens == 100_000
    assert row.input_cost_usd == pytest.approx(0.9)
    assert row.cached_cost_usd == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_api_keys_never_leak_into_errors(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")
    registry = write_registry(tmp_path, OPENAI_REGISTRY)

    async def boom(**request):
        raise RuntimeError("request failed with key sk-super-secret")

    monkeypatch.setattr(providers, "_litellm_completion", boom)
    results = await run(RunConfig(prompt="hi", providers=frozenset({"openai"}), registry=registry))
    assert "sk-super-secret" not in results[0].error
    assert "***" in results[0].error


@pytest.mark.asyncio
async def test_missing_litellm_yields_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(providers, "_litellm_completion", None)
    registry = write_registry(tmp_path, OPENAI_REGISTRY)
    results = await run(RunConfig(prompt="hi", providers=frozenset({"openai"}), registry=registry))
    assert "fasteval[native]" in results[0].error


def test_run_config_rejects_blank_prompt():
    with pytest.raises(ConfigError, match="Prompt"):
        RunConfig(prompt="   ")
