import pytest

from fasteval.config import ModelSpec
from fasteval.models import ModelResponse
from fasteval.pricing import compute_costs


def make_spec(**rates) -> ModelSpec:
    return ModelSpec(id="openai:gpt-test:off", provider="openai", model="gpt-test", **rates)


def test_costs_with_all_rates_configured():
    spec = make_spec(
        input_cost_usd_per_mtok=1.0,
        output_cost_usd_per_mtok=6.0,
        reasoning_cost_usd_per_mtok=12.0,
        cached_input_cost_usd_per_mtok=0.1,
    )
    response = ModelResponse(
        text="x", input_tokens=1_000_000, output_tokens=500_000, reasoning_tokens=250_000, cached_tokens=100_000
    )
    costs = compute_costs(spec, response)
    assert costs.input == pytest.approx(1.0)
    assert costs.output == pytest.approx(3.0)
    assert costs.reasoning == pytest.approx(3.0)
    assert costs.cached == pytest.approx(0.01)
    assert costs.total == pytest.approx(7.01)


def test_reasoning_falls_back_to_output_rate():
    spec = make_spec(output_cost_usd_per_mtok=6.0)
    response = ModelResponse(text="x", reasoning_tokens=1_000_000)
    assert compute_costs(spec, response).reasoning == pytest.approx(6.0)


def test_cached_accepts_legacy_alias_key():
    spec = make_spec(cached_cost_usd_per_mtok=0.2)
    response = ModelResponse(text="x", cached_tokens=1_000_000)
    assert compute_costs(spec, response).cached == pytest.approx(0.2)


def test_zero_tokens_and_missing_rates_yield_none():
    response = ModelResponse(text="x")
    costs = compute_costs(make_spec(), response)
    assert (costs.input, costs.output, costs.reasoning, costs.cached) == (None, None, None, None)
    assert costs.total is None
