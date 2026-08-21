import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

from fasteval import runner
from fasteval.runner import run


def write_registry(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "models.toml"
    path.write_text(content)
    return path


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
    results = await run({"prompt": "hi", "providers": {"mock"}, "registry": str(registry)})
    assert [row.reasoning_effort for row in results] == ["off", "low"]
    assert all(row.error == "" for row in results)
    assert results[0].output == "alpha says hi"


@pytest.mark.asyncio
async def test_builtin_mock_used_when_registry_has_none(tmp_path):
    registry = write_registry(tmp_path, OPENAI_REGISTRY)
    results = await run({"prompt": "hi", "providers": {"mock"}, "registry": str(registry)})
    assert len(results) == 1
    assert results[0].provider == "mock"
    assert results[0].model == "demo"


@pytest.mark.asyncio
async def test_all_excludes_mock_providers(tmp_path):
    registry = write_registry(tmp_path, MOCK_REGISTRY + OPENAI_REGISTRY)
    results = await run({"prompt": "hi", "providers": {"all"}, "registry": str(registry)})
    assert {row.provider for row in results} == {"openai"}


@pytest.mark.asyncio
async def test_unknown_provider_raises(tmp_path):
    registry = write_registry(tmp_path, MOCK_REGISTRY)
    with pytest.raises(ValueError, match="Unknown provider"):
        await run({"prompt": "hi", "providers": {"nope"}, "registry": str(registry)})


@pytest.mark.asyncio
async def test_missing_registry_file_raises(tmp_path):
    with pytest.raises(ValueError, match="Model registry not found"):
        await run({"prompt": "hi", "providers": {"mock"}, "registry": str(tmp_path / "missing.toml")})


@pytest.mark.asyncio
async def test_no_models_for_provider_raises(tmp_path):
    registry = write_registry(tmp_path, OPENAI_REGISTRY)
    with pytest.raises(ValueError, match="No models found"):
        await run({"prompt": "hi", "providers": {"gemini"}, "registry": str(registry)})


@pytest.mark.asyncio
async def test_partial_failure_keeps_matrix(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    registry = write_registry(tmp_path, MOCK_REGISTRY + OPENAI_REGISTRY)
    results = await run({"prompt": "hi", "providers": {"mock", "openai"}, "registry": str(registry)})
    by_provider = {row.provider: row for row in results}
    assert by_provider["mock"].error == ""
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

    monkeypatch.setattr(runner, "acompletion", spy)
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    results = await run({"prompt": "hi", "providers": {"openai"}, "registry": str(registry), "structured_output": schema})

    assert captured["response_format"]["json_schema"]["schema"] == schema
    assert results[0].error == ""
    assert results[0].output == {"name": "Ada"}


@pytest.mark.asyncio
async def test_structured_output_validates_provider_response(tmp_path, monkeypatch):
    registry = write_registry(tmp_path, OPENAI_REGISTRY)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setattr(runner, "acompletion", fake_completion(text="not json at all"))
    schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"], "additionalProperties": False}
    results = await run({"prompt": "hi", "providers": {"openai"}, "registry": str(registry), "structured_output": schema})
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
    results = await run({"prompt": "hi", "providers": {"mock"}, "registry": str(registry), "structured_output": schema})
    assert results[0].output == {"name": "A name", "age": 1}


@pytest.mark.asyncio
async def test_ttft_is_none_and_costs_use_registry_rates(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")
    monkeypatch.setattr(runner, "acompletion", fake_completion(prompt_tokens=1_000_000, completion_tokens=500_000, cached=100_000, reasoning=0))
    registry = write_registry(tmp_path, OPENAI_REGISTRY)
    results = await run({"prompt": "hi", "providers": {"openai"}, "registry": str(registry)})
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

    monkeypatch.setattr(runner, "acompletion", boom)
    results = await run({"prompt": "hi", "providers": {"openai"}, "registry": str(registry)})
    assert "sk-super-secret" not in results[0].error
    assert "***" in results[0].error


def test_default_registry_path_prefers_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert runner._default_registry_path() != Path("config") / "models.toml"
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "models.toml").write_text("")
    assert runner._default_registry_path() == config_dir / "models.toml"


def test_registry_toml_stays_loadable():
    registry_path = Path(__file__).resolve().parents[1] / "config" / "models.toml"
    data = tomllib.loads(registry_path.read_text())
    assert data, "committed registry must not be empty"
