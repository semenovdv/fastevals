# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and the project adheres to [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- Dataset evaluation: `--dataset` accepts JSONL/CSV case files with optional
  expected outputs, evaluator names and regex patterns.
- Deterministic evaluators: `exact_match`, `contains`, `json_valid`, `regex`,
  with per-row verdicts and per-model aggregate pass rates.
- `--nruns N` repeats every case for consistency checks; reports show attempts
  and aggregates.
- MCP server (`fasteval-mcp`) exposing `run_evaluation`, `list_models` and
  `get_run` tools over stdio.
- Smart attachments: images sent as vision parts, PDFs as file parts, text
  files inlined; 20 MB size limit validated up front.
- Dashboard: sortable/filterable comparison table, CSV/Markdown export,
  fastest/cheapest/top-throughput cards, per-model aggregate tables,
  fasteval version metadata.
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
