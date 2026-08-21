from fasteval.runner import ModelResponse, RunResult, _compute_costs

import pytest


def test_costs_with_all_rates_configured():
    model = {
        "input_cost_usd_per_mtok": 1.0,
        "output_cost_usd_per_mtok": 6.0,
        "reasoning_cost_usd_per_mtok": 12.0,
        "cached_input_cost_usd_per_mtok": 0.1,
    }
    response = ModelResponse(text="x", input_tokens=1_000_000, output_tokens=500_000, reasoning_tokens=250_000, cached_tokens=100_000)
    costs = _compute_costs(model, response)
    assert costs["input"] == pytest.approx(1.0)
    assert costs["output"] == pytest.approx(3.0)
    assert costs["reasoning"] == pytest.approx(3.0)
    assert costs["cached"] == pytest.approx(0.01)


def test_reasoning_falls_back_to_output_rate():
    model = {"output_cost_usd_per_mtok": 6.0}
    response = ModelResponse(text="x", reasoning_tokens=1_000_000)
    assert _compute_costs(model, response)["reasoning"] == 6.0


def test_cached_accepts_legacy_alias_key():
    model = {"cached_cost_usd_per_mtok": 0.2}
    response = ModelResponse(text="x", cached_tokens=1_000_000)
    assert _compute_costs(model, response)["cached"] == 0.2


def test_zero_tokens_and_missing_rates_yield_none():
    response = ModelResponse(text="x")
    assert _compute_costs({}, response) == {"input": None, "output": None, "reasoning": None, "cached": None}


def test_total_cost_skips_unknown_components():
    row = RunResult(provider="p", model="m", reasoning_effort="off", output="x", input_cost_usd=None, output_cost_usd=0.5)
    assert row.total_cost_usd == pytest.approx(0.5)
    empty = RunResult(provider="p", model="m", reasoning_effort="off", output="x")
    assert empty.total_cost_usd is None
