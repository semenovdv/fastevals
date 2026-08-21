# Roadmap

fasteval follows a "small core, honest metrics" philosophy. Everything below
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

### Streaming metrics

Real time-to-first-token via streamed completions, plus incremental
throughput curves per run instead of a single end-to-end number.

### Smarter evaluation

- Per-case assertions composed from multiple evaluators with weights.
- Opt-in LLM-as-judge evaluator with a documented rubric prompt.
- Statistical stability scores across `--nruns` attempts (not just raw repeats).

### Registry ergonomics

- `fasteval models add` helper that validates pricing fields interactively.
- Optional live price sync for well-known providers, clearly labeled as
  external data.

### Reporting

- Diff view between two saved runs ("what did this prompt change cost me?").
- Optional self-contained report (inline Chart.js) for fully offline sharing.

## Non-goals

- Becoming a full observability platform — fasteval measures one matrix per run.
- Hosting or team features; the output must stay plain files you own.
- Hiding provider differences behind fake abstractions; honest per-provider
  behavior beats uniform illusions.
