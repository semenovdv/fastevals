# Roadmap

fastevals follows a "small core, honest metrics" philosophy. Everything below
is direction, not promise — see [`CHANGELOG.md`](../CHANGELOG.md) for what
has already landed.

## Done in v0.1

- Provider-agnostic matrix runs over a TOML model registry (LiteLLM adapter,
  built-in mock provider).
- Structured output with compact schema syntax, strict JSON Schema requests,
  and local validation of every response.
- Disjoint token buckets with per-bucket pricing and secret-safe errors.
- Dataset evaluation (JSONL/CSV), deterministic evaluators, repeated runs.
- Standalone HTML dashboards with sorting, filtering and CSV/Markdown export.
- MCP server exposing `run_evaluation`, `list_models`, `get_run`.
- Quality gates: ruff, `mypy --strict`, branch coverage floor, matrix CI,
  wheel smoke test, pre-commit hooks.

## Next

Prioritized by impact: adoption first, then the features that turn fastevals
from a comparison tool into an engineering tool.

### P1 — Adoption

- **Publish to PyPI** via trusted publishing so `pip install fastevals` works.
- **Codecov** integration with a live coverage badge.
- **Demo GIF** in the README (mock provider, zero credits).
- **Docs site** on GitHub Pages (mkdocs-material) with an API reference and a
  rendered sample report.

### P2 — Engineering tool

- **Eval gates for CI** — store a baseline run, compare future runs against
  it (`fastevals check --baseline`), and fail a PR when quality regresses or
  cost grows beyond a threshold. The pytest moment for LLM evaluation.
- **LLM-as-judge evaluator** — opt-in rubric-based scoring for open-ended
  cases, clearly labeled as a subjective metric.
- **Request caching** so iterating on a dataset does not re-pay identical
  (prompt, model, params) calls.
- **Retries with exponential backoff** for transient provider failures.
- **Run budget** (`--max-cost`) as a stop-crane for expensive matrices.

### P3 — Metrics and scenarios

- ~~Streaming completions~~ — shipped: TTFT is measured on streamed
  completions with automatic non-streaming fallback.
- **Incremental throughput curves** per run (streaming groundwork done).
- **Local models** via ollama/vLLM for free, private experimentation.
- **System prompts and multi-turn cases** — real applications are dialogs.
- **Prompt templating** with dataset variables for few-shot cases.
- **Statistics across `--nruns`** — variance, confidence intervals,
  significance between models.
- **Run diffing** — compare two saved runs ("what did this prompt change
  cost me?").

### P4 — Ecosystem

- Plugin discovery via entry points for custom evaluators and providers.
- Run history in sqlite with quality-over-time charts.
- Dependabot and CodeQL in CI.
- Anthropic adapter; issue templates and GitHub Discussions.

## Non-goals

- Becoming a full observability platform — fastevals measures one matrix per run.
- Hosting or team features; the output must stay plain files you own.
- Hiding provider differences behind fake abstractions; honest per-provider
  behavior beats uniform illusions.
