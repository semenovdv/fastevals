# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- **Tags: named model suites as reusable presets.** `fastevals tag
  add/list/show/remove` stores validated selector suites in
  `~/.config/fastevals/tags.toml`, shared across terminals and MCP clients;
  `fastevals --tag NAME` (or the MCP `run_evaluation(tag=...)` argument)
  runs a suite without retyping models. New MCP tools `add_tag` and
  `list_tags` let agents define and discover suites themselves; invalid
  selectors are rejected at save time.
- Model selectors for cherry-picking the comparison matrix without editing a
  registry: `-m/--models` on the CLI, a `models` argument on the MCP
  `run_evaluation` tool, and `RunConfig(models=...)` in Python. Grammar:
  `provider/model[@effort,effort]` with the exact official model id —
  `openai/gpt-5.6-luna@high|openai/gpt-5.6-sol@low`. The provider is
  required: identical model strings are common across providers, so bare
  names are rejected with the list of matching entry ids instead of
  silently fanning out a paid run. Ids from `--list-models` paste back
  verbatim; unknown selectors fail listing available ids.

## [0.1.2] - 2026-08-21

### Fixed

- `fastevals --version` reported a stale hardcoded string (0.1.0) while the
  installed distribution was 0.1.1; the runtime version now comes from
  package metadata with a regression test.

## [0.1.1] - 2026-08-21

### Added

- `--version` and `--list-models` CLI flags; the latter prints the resolved
  registry as JSON (shared implementation with the MCP `list_models` tool).
- Automatic retries for transient provider failures with exponential
  backoff (`max_retries` per registry entry, default 2 — three attempts).

### Changed

- Dropped the non-standard `-pr` short flag; use `--providers`.

### Fixed

- A plain `pip install fastevals` was unusable outside a source checkout:
  the model registry now ships inside the wheel and is discovered as a
  fallback after `./config/models.toml`.

### Changed

- LiteLLM moved from the `native` extra into core dependencies; the extra is
  removed. `pip install fastevals` is now the only install step.

### Added

- Dataset evaluation: `--dataset` accepts JSONL/CSV case files with optional
  expected outputs, evaluator names and regex patterns.
- Deterministic evaluators: `exact_match`, `contains`, `json_valid`, `regex`,
  with per-row verdicts and per-model aggregate pass rates.
- `--nruns N` repeats every case for consistency checks; reports show attempts
  and aggregates.
- MCP server (`fastevals-mcp`) exposing `run_evaluation`, `list_models` and
  `get_run` tools over stdio.
- Smart attachments: images sent as vision parts, PDFs as file parts, text
  files inlined; 20 MB size limit validated up front.
- Dashboard: sortable/filterable comparison table, CSV/Markdown export,
  fastest/cheapest/top-throughput cards, per-model aggregate tables,
  fastevals version metadata.
- Quality gates: ruff + formatter, `mypy --strict`, branch coverage with an
  85% floor, GitHub Actions matrix CI (3.11–3.13), wheel smoke test,
  pre-commit hooks, Makefile.

### Changed

- Package now requires Python >= 3.11 (`tomllib` is standard library there).
- `jsonschema` is a core dependency so structured output always validates.
- LiteLLM became an optional extra (`.[native]`); mock-only usage no longer
  imports it.

### Fixed

- Cached and reasoning token costs were silently zero due to registry key
  mismatch; pricing now supports documented aliases and rate fallbacks.
- Time-to-first-token was reported as total latency without streaming; it is
  now reported honestly as unavailable.
- Registry path was resolved relative to the installed module; discovery now
  prefers the working directory and supports `--registry`.
- CLI provider choices disagreed with the registry; providers share one
  source of truth with fail-fast validation.
- Model responses were not validated against the structured-output schema;
  validation failures now surface as structured errors.
- API key values could leak into error messages; secrets are scrubbed.
