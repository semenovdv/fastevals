# QA Checklist — full-repo verification by usage scenario

Repeat before every release. Statuses below reflect the audit of v0.1.3
(2026-08-21). ✅ passed · 🔧 bug found & fixed in this pass · 👁 manual/live.

## A. First run after `pip install` (new user)

| # | Check | Criterion | Status |
|---|---|---|---|
| A1 | `fastevals --version` / `--help` | version matches PyPI; all flags listed | ✅ |
| A2 | `fastevals --list-models` | bundled registry found in site-packages, ids shown | ✅ |
| A3 | Run without API key | `exit=1`, JSON error names `OPENAI_API_KEY`, no traceback | ✅ |
| A4 | Bare selector `luna@low` | rejected with matching entry ids listed | ✅ |
| A5 | Real run `--tag auto-cheap` 👁 | `ok:true`, artifacts written, cost < $0.001 | ✅ ($0.000032) |
| A6 | `report.html` opened offline 👁 | dashboard renders, no secrets inside | ✅ |

## B. Contributor workflow

| # | Check | Criterion | Status |
|---|---|---|---|
| B1 | Fresh clone → `make dev` → `make check` | ruff + mypy strict + tests ≥85% green | ✅ |
| B2 | Test suite fully offline | no network needed | ✅ |
| B3 | PR pipeline | CI matrix 3.11–3.13 gates merge | ✅ |
| B4 | Wheel build + smoke in clean venv | entry points work | ✅ |

## C. CLI power user: matrix / dataset / tags

| # | Check | Criterion | Status |
|---|---|---|---|
| C1 | `tag add` with unresolvable selector | `exit=1`, tag not saved | ✅ |
| C2 | `tag add/list/show` incl. built-ins | saved suite visible; cell count correct | ✅ |
| C3 | `--tag nightly --dataset cases.jsonl --nruns 2` 👁 | cells = cases×runs×models; per-row `evaluation.passed` correct | ✅ (live, 2/2 pass) |
| C4 | CSV dataset identical to JSONL | blank lines skipped in both | ✅ (regression test added) |
| C5 | `--tag` together with `--models` | rejected: "Use tag or models, not both" | ✅ |
| C6 | Unknown tag / selector / provider errors | each lists the available options | ✅ |
| C7 | `run.json` schema | `case_id, attempt, evaluation, total_cost_usd, ok` per row | ✅ |
| C8 | Partial provider failure | healthy cells finish; exit=1; both rows reported | ✅ |

## D. Agent via MCP

| # | Check | Criterion | Status |
|---|---|---|---|
| D1 | **Real** stdio handshake with spawned `fastevals-mcp` | initialize + list_tools succeed | ✅ (automated now) |
| D2 | Tool surface | ≥5 tools incl. add_tag/list_tags, all documented | ✅ |
| D3 | Agent cycle: add_tag → run_evaluation(tag) → get_run | structured JSON on success *and* failure, no server crashes | ✅ |
| D4 | Shared state across surfaces | MCP-created tag visible to CLI (same tags.toml) | ✅ |
| D5 | Built-in tags via MCP | resolve identically to CLI | ✅ |

## E. Python integrator

| # | Check | Criterion | Status |
|---|---|---|---|
| E1 | README python blocks execute verbatim | zero edits needed | ✅ (automated; fixed standalone `resolve_tag`) |
| E2 | Consumer project under `mypy --strict` | py.typed contract holds | ✅ (CI step added) |
| E3 | `__version__` == installed metadata | drift guard test | ✅ |
| E4 | `await run_evals()` inside a running loop | no loop conflicts | ✅ (test added) |

## F. Security & missing credentials

| # | Check | Criterion | Status |
|---|---|---|---|
| F1 | No secrets in git history (`git log -p \| grep sk-`) | clean | ✅ |
| F2 | 401 from provider with fake key | key value absent from error/artifacts | ✅ (live check) |
| F3 | `.env` precedence: env var > project .env > global store | confirmed | ✅ (tests added) |
| F4 | Registry `timeout_s` honored | slow provider fails fast | ✅ (unit tests) |
| F5 | Artifacts free of `API_KEY=` dumps | grep clean | ✅ |

## G. Package integrity

| # | Check | Criterion | Status |
|---|---|---|---|
| G1 | Wheel contents | data/models.toml + py.typed present; tests/.env excluded | ✅ (CI assertion added) |
| G2 | sdist installs and imports | clean-venv smoke | ✅ (CI step added) |
| G3 | `twine check` metadata | passes | ✅ (CI step added) |

## Bugs found & fixed during this audit

1. `fastevals --list-models --registry MISSING` crashed with a raw
   `FileNotFoundError` traceback → `load_registry` now raises
   `ConfigError`; every surface returns the structured JSON error.
2. MCP servers launched by Claude Desktop (cwd=/, minimal env) could not
   find any `.env`, so runs failed without keys even when the user had a
   project `.env`. Added a global agent store:
   `~/.config/fastevals/.env` is loaded after the project `.env`
   (environment variables still take priority).
3. `resolve_tag("auto-deep")` from Python raised "needs a non-empty model
   registry" instead of resolving against the default registry — built-ins
   now self-load it when `entries` is omitted.
