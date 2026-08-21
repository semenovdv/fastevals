import json
from pathlib import Path

import pytest

from fastevals.config import RunConfig
from fastevals.exceptions import ConfigError, FastEvalError
from fastevals.runner import run


def write_registry(tmp_path: Path, content: str) -> str:
    path = tmp_path / "models.toml"
    path.write_text(content)
    return str(path)


OPENAI_REGISTRY = """
["openai:gpt-test"]
provider = "openai"
model = "gpt-test"
api_key_env = "OPENAI_API_KEY"
reasoning_efforts = "off|low"
input_cost_usd_per_mtok = 1.0
cached_input_cost_usd_per_mtok = 0.1
output_cost_usd_per_mtok = 2.0
"""


@pytest.mark.asyncio
async def test_run_expands_reasoning_efforts(tmp_path, fake_llm, api_key, openai_registry):
    calls = fake_llm(text="answer")
    results = await run(RunConfig(prompt="hi", providers=frozenset({"openai"}), registry=str(openai_registry)))
    assert [row.reasoning_effort for row in results] == ["off", "low"]
    assert all(row.ok for row in results)
    assert len(calls) == 2
    assert all(row.output == "answer" for row in results)


@pytest.mark.asyncio
async def test_unknown_provider_raises(tmp_path):
    registry = write_registry(tmp_path, OPENAI_REGISTRY)
    with pytest.raises(ConfigError, match="Unknown provider"):
        await run(RunConfig(prompt="hi", providers=frozenset({"nope"}), registry=registry))


@pytest.mark.asyncio
async def test_missing_registry_file_raises(tmp_path):
    with pytest.raises(FastEvalError, match="Model registry not found"):
        await run(RunConfig(prompt="hi", registry=str(tmp_path / "missing.toml")))


@pytest.mark.asyncio
async def test_no_models_for_provider_raises(openai_registry):
    with pytest.raises(ConfigError, match="No models found"):
        await run(RunConfig(prompt="hi", providers=frozenset({"gemini"}), registry=str(openai_registry)))


@pytest.mark.asyncio
async def test_partial_failure_keeps_matrix(tmp_path, monkeypatch, openai_registry):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Bypass the litellm install guard so the missing-credential check is
    # reached deterministically, regardless of installed extras.
    async def _stub(**request):
        raise AssertionError("should not be called without an API key")

    monkeypatch.setattr("fastevals.providers._litellm_completion", _stub)
    results = await run(RunConfig(prompt="hi", registry=str(openai_registry)))
    assert [row.ok for row in results] == [False, False]
    assert all("OPENAI_API_KEY" in row.error for row in results)
    assert all(row.output is None for row in results)


@pytest.mark.asyncio
async def test_structured_output_reaches_provider_request(tmp_path, fake_llm, api_key, openai_registry):
    calls = fake_llm(text=json.dumps({"name": "Ada"}))
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    config = RunConfig(
        prompt="hi",
        providers=frozenset({"openai"}),
        registry=str(openai_registry),
        structured_output=schema,
    )
    results = await run(config)

    assert calls[0]["response_format"]["json_schema"]["schema"] == schema
    assert results[0].ok
    assert results[0].output == {"name": "Ada"}


@pytest.mark.asyncio
async def test_structured_output_validates_provider_response(tmp_path, fake_llm, api_key, openai_registry):
    fake_llm(text="not json at all")
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    config = RunConfig(
        prompt="hi",
        providers=frozenset({"openai"}),
        registry=str(openai_registry),
        structured_output=schema,
    )
    results = await run(config)
    assert "not valid JSON" in results[0].error


@pytest.mark.asyncio
async def test_ttft_is_none_and_costs_use_registry_rates(tmp_path, fake_llm, api_key, openai_registry):
    fake_llm(prompt_tokens=1_000_000, completion_tokens=500_000, cached=100_000, reasoning=0)
    results = await run(RunConfig(prompt="hi", providers=frozenset({"openai"}), registry=str(openai_registry)))
    row = results[0]
    assert row.time_to_first_token_ms is None
    assert row.tokens_per_second is not None and row.tokens_per_second > 0
    assert row.input_tokens == 900_000
    assert row.cached_tokens == 100_000
    assert row.input_cost_usd == pytest.approx(0.9)
    assert row.cached_cost_usd == pytest.approx(0.01)


@pytest.mark.asyncio
async def test_api_keys_never_leak_into_errors(tmp_path, monkeypatch, openai_registry):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-super-secret")

    async def boom(**request):
        raise RuntimeError("request failed with key sk-super-secret")

    monkeypatch.setattr("fastevals.providers._litellm_completion", boom)
    results = await run(RunConfig(prompt="hi", providers=frozenset({"openai"}), registry=str(openai_registry)))
    assert "sk-super-secret" not in results[0].error
    assert "***" in results[0].error


@pytest.mark.asyncio
async def test_missing_litellm_yields_clear_error(monkeypatch, openai_registry):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("fastevals.providers._litellm_completion", None)
    results = await run(RunConfig(prompt="hi", providers=frozenset({"openai"}), registry=str(openai_registry)))
    assert "fastevals[native]" in results[0].error


def test_run_config_rejects_blank_prompt():
    with pytest.raises(ConfigError, match="Prompt"):
        RunConfig(prompt="   ")


def test_run_config_rejects_unsupported_provider():
    with pytest.raises(ConfigError, match="Unknown provider"):
        RunConfig(prompt="x", providers=frozenset({"mock"}))


@pytest.mark.asyncio
async def test_dataset_run_with_evaluator_and_nruns(tmp_path, fake_llm, api_key, openai_registry):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        "\n".join(
            [
                json.dumps({"id": "greet", "prompt": "say hi", "expected": "hello", "evaluator": "exact_match"}),
                json.dumps({"id": "miss", "prompt": "bye", "evaluator": "contains", "pattern": "zzz"}),
            ]
        )
    )
    fake_llm(text="hello")
    config = RunConfig(
        prompt="",
        providers=frozenset({"openai"}),
        registry=str(openai_registry),
        dataset=str(dataset),
        nruns=2,
    )
    results = await run(config)
    assert len(results) == 8  # 2 cases x 2 attempts x (off|low)
    greet = [row for row in results if row.case_id == "greet"]
    assert all(row.evaluation["passed"] is True for row in greet)
    assert sorted(row.attempt for row in greet) == [1, 1, 2, 2]
    miss = [row for row in results if row.case_id == "miss"]
    assert all(row.evaluation["passed"] is False for row in miss)


@pytest.mark.asyncio
async def test_dataset_with_unknown_evaluator_raises(tmp_path, openai_registry):
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(json.dumps({"prompt": "hi", "evaluator": "magic"}))
    with pytest.raises(ConfigError, match="unknown evaluator"):
        await run(
            RunConfig(
                prompt="",
                providers=frozenset({"openai"}),
                registry=str(openai_registry),
                dataset=str(dataset),
            )
        )
