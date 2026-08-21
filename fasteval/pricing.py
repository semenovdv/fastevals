"""Cost calculation from token buckets and per-million USD rates."""

from dataclasses import dataclass

from .config import ModelSpec
from .models import ModelResponse

__all__ = ["Costs", "compute_costs"]


@dataclass(frozen=True)
class Costs:
    """Per-bucket USD costs. ``None`` means "not measurable".""" 

    input: float | None = None
    output: float | None = None
    reasoning: float | None = None
    cached: float | None = None

    @property
    def total(self) -> float | None:
        known = [cost for cost in (self.input, self.output, self.reasoning, self.cached) if cost is not None]
        return sum(known) if known else None


def _rate(spec: ModelSpec, *keys: str) -> float | None:
    for key in keys:
        value = getattr(spec, key)
        if value is not None:
            return float(value)
    return None


def _cost(tokens: int | None, rate: float | None) -> float | None:
    if not tokens or rate is None:
        return None
    return tokens / 1_000_000 * rate


def compute_costs(spec: ModelSpec, response: ModelResponse) -> Costs:
    """Price disjoint token buckets.

    Reasoning tokens fall back to the output rate when no dedicated rate is
    configured; cached tokens accept the ``cached_cost_usd_per_mtok`` legacy
    alias. Buckets with zero tokens or unknown rates cost ``None``.
    """
    return Costs(
        input=_cost(response.input_tokens, _rate(spec, "input_cost_usd_per_mtok")),
        output=_cost(response.output_tokens, _rate(spec, "output_cost_usd_per_mtok")),
        reasoning=_cost(response.reasoning_tokens, _rate(spec, "reasoning_cost_usd_per_mtok", "output_cost_usd_per_mtok")),
        cached=_cost(response.cached_tokens, _rate(spec, "cached_input_cost_usd_per_mtok", "cached_cost_usd_per_mtok")),
    )
