"""Evaluation matrix orchestration."""

import asyncio
import time
from pathlib import Path
from typing import Any

from .config import RunConfig
from .dataset import Case, load_dataset
from .evaluators import EVALUATORS, evaluate_output
from .exceptions import ConfigError, FastEvalError
from .models import ModelResponse, RunResult
from .pricing import compute_costs
from .providers import call_model, scrub_secrets
from .registry import default_registry_path, load_registry, select_specs
from .structured import validated_instance

__all__ = ["ModelResponse", "RunResult", "run"]


def _resolve_cases(config: RunConfig) -> list[Case]:
    if not config.dataset:
        return [Case(id="case-001", prompt=config.prompt)]
    cases = load_dataset(config.dataset)
    for case in cases:
        if case.evaluator and case.evaluator not in EVALUATORS:
            raise ConfigError(
                f"Case '{case.id}' uses unknown evaluator '{case.evaluator}'. Supported: {', '.join(EVALUATORS)}"
            )
        if case.evaluator == "regex" and not case.pattern:
            raise ConfigError(f"Case '{case.id}' uses the regex evaluator but defines no pattern")
    return cases


async def run(config: RunConfig) -> list[RunResult]:
    """Execute the evaluation matrix: cases x attempts x models."""
    if config.registry and not Path(config.registry).exists():
        raise FastEvalError(f"Model registry not found: {config.registry}")
    registry_path = config.registry or default_registry_path()
    entries = load_registry(registry_path) if registry_path and Path(registry_path).exists() else {}
    specs = select_specs(entries, config.requested_providers())
    cases = _resolve_cases(config)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    async def call_one(spec: Any, prompt: str) -> tuple[Any, ModelResponse]:
        async with semaphore:
            response = await call_model(
                spec,
                prompt,
                file_path=config.file,
                image_path=config.image,
                response_schema=config.structured_output,
            )
        output: Any = response.text
        if config.structured_output:
            output = validated_instance(response.text, config.structured_output)
        return output, response

    async def run_one(spec: Any, case: Case, attempt: int) -> RunResult:
        started = time.perf_counter()
        try:
            output, response = await call_one(spec, case.prompt)
            latency_ms = (time.perf_counter() - started) * 1000
            costs = compute_costs(spec, response)
            seconds = latency_ms / 1000
            tokens_per_second = (response.output_tokens / seconds) if seconds > 0 and response.output_tokens else None
            return RunResult(
                provider=spec.provider,
                model=spec.model,
                reasoning_effort=spec.reasoning_effort,
                output=output,
                case_id=case.id,
                attempt=attempt,
                evaluation=evaluate_output(case, output),
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                reasoning_tokens=response.reasoning_tokens,
                cached_tokens=response.cached_tokens,
                input_cost_usd=costs.input,
                output_cost_usd=costs.output,
                reasoning_cost_usd=costs.reasoning,
                cached_cost_usd=costs.cached,
                finish_reason=response.finish_reason or "completed",
                response_id=response.response_id or "",
                error="",
                latency_ms=latency_ms,
                time_to_first_token_ms=None,
                tokens_per_second=tokens_per_second,
            )
        except Exception as exc:  # Keep the matrix report useful when one provider fails.
            return RunResult(
                provider=spec.provider,
                model=spec.model,
                reasoning_effort=spec.reasoning_effort,
                output=None,
                case_id=case.id,
                attempt=attempt,
                evaluation=evaluate_output(case, None),
                latency_ms=(time.perf_counter() - started) * 1000,
                error=scrub_secrets(f"{type(exc).__name__}: {exc}"),
            )

    jobs = [run_one(spec, case, attempt) for case in cases for attempt in range(1, config.nruns + 1) for spec in specs]
    return list(await asyncio.gather(*jobs))
