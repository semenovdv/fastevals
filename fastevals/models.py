"""Core result models shared across the package."""

from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = ["ModelResponse", "RunResult"]


@dataclass
class ModelResponse:
    """Normalized single-model response returned by provider adapters."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None
    finish_reason: str | None = None
    response_id: str | None = None
    time_to_first_token_ms: float | None = None


@dataclass
class RunResult:
    """Outcome of one cell in the evaluation matrix.

    Token buckets are disjoint: ``input_tokens`` excludes cached tokens and
    ``output_tokens`` excludes reasoning tokens, so each bucket is billed at
    most once.
    """

    provider: str
    model: str
    reasoning_effort: str

    output: Any

    case_id: str = "case-001"
    attempt: int = 1
    evaluation: dict[str, Any] | None = field(default=None)

    time_to_first_token_ms: float | None = None
    latency_ms: float | None = None

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None

    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    reasoning_cost_usd: float | None = None
    cached_cost_usd: float | None = None

    tokens_per_second: float | None = None

    error: str | None = None
    finish_reason: str | None = None
    response_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize including computed convenience fields."""
        data: dict[str, Any] = asdict(self)
        data["ok"] = self.ok
        data["total_cost_usd"] = self.total_cost_usd
        return data

    @property
    def ok(self) -> bool:
        return not self.error

    @property
    def total_cost_usd(self) -> float | None:
        costs = (
            self.input_cost_usd,
            self.output_cost_usd,
            self.reasoning_cost_usd,
            self.cached_cost_usd,
        )
        known_costs = [cost for cost in costs if cost is not None]
        return sum(known_costs) if known_costs else None
