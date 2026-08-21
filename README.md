# fastevals

**Evaluation tooling your AI agents can drive.**

fastevals is a small, provider-agnostic evaluation runner for LLM
applications. Run one prompt — or a whole dataset — across a matrix of
models, reasoning efforts and providers, save every response, and get a
readable standalone HTML comparison report with cost, latency and token
metrics.

It ships as an **MCP server**, so Claude Desktop, Claude Code or any other
MCP client can run evaluations as a native tool: your agent decides *what*
to test, fastevals answers *which model does it best*.

[![CI](https://github.com/semenovdv/fastevals/actions/workflows/ci.yml/badge.svg)](https://github.com/semenovdv/fastevals/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![Coverage](https://img.shields.io/badge/coverage-89%25-brightgreen)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![mypy](https://img.shields.io/badge/mypy-strict-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<p align="center">
  <img src="docs/assets/report.png" alt="fastevals HTML report" width="820">
</p>

## Drive it from Claude (MCP)

Install the server extras and register the entry point with any MCP client:

```bash
python3 -m pip install 'fastevals[mcp,native]'
claude mcp add fastevals -- fastevals-mcp        # Claude Code
```

Claude Desktop (`claude_desktop_config.json`):

```json
{
  "mcpServers": { "fastevals": { "command": "fastevals-mcp" } }
}
```

Exposed tools:

| Tool | Purpose |
|---|---|
| `run_evaluation` | Run a prompt or dataset across providers; returns JSON summary + report paths |
| `list_models` | Registry inspector: models, reasoning efforts, pricing |
| `get_run` | Summarize a saved run: pass rate, errors, total cost |

Example agent prompts that now just work:

> Use fastevals to compare gpt-5.6-luna at reasoning low and high on "Summarize
> this contract in 5 bullets" — which one is cheaper per correct answer?

> List my registered models, then evaluate cases.jsonl on terra and report
> the pass rate per effort level.

Because the CLI is fully non-interactive and returns structured JSON, agents
can also drive evaluations through plain shell execution without MCP.

## Why fastevals

- **Structured output that verifies** — compact schema syntax compiles to JSON Schema, is sent to the provider, and every response is validated locally before it reaches `run.json`.
- **Honest metrics** — disjoint token buckets (input / output / reasoning / cached), per-bucket pricing from your registry, no fake TTFT without streaming.
- **Real evaluation loop** — JSONL/CSV datasets, deterministic evaluators (`exact_match`, `contains`, `json_valid`, `regex`), repeated runs for stability.
- **Boring engineering** — strict typing, ~90% branch coverage, ruff + mypy + coverage gates in CI, single-file reports with zero telemetry.

## Install

```bash
python3 -m pip install 'fastevals[native]'    # from PyPI once released
# or from source:
git clone https://github.com/semenovdv/fastevals && python3 -m pip install -e '.[native]'
```

## CLI quick start

```bash
export OPENAI_API_KEY=...                  # keys live in the environment only
fastevals --prompt "Explain evaluation in three bullets" \
          --providers openai --out runs
```

Every run writes a timestamped directory under `--out` containing
`run.json` (machine-readable) and `report.html` (a standalone dashboard you
can open or send to anyone). Exit codes: `0` when every model completed,
`1` otherwise — easy to script.

### Models and reasoning efforts

Entries in `config/models.toml` become cells in the matrix:

```toml
["openai:gpt-5.6-luna"]
provider = "openai"
model = "gpt-5.6-luna"
api_key_env = "OPENAI_API_KEY"
reasoning_efforts = "none|low"          # expands into two runs
input_cost_usd_per_mtok = 1.0           # USD per 1M tokens
output_cost_usd_per_mtok = 6.0
```

Providers are validated against the registry; unknown names fail fast with a
helpful message. API keys are read from environment variables only — never
from the registry, never logged, and scrubbed from error messages.

### Structured output

```bash
fastevals \
  --prompt "Extract all relevant invoice fields" \
  --structured-output 'invoice_number:str("Unique identifier"),total:float("Amount incl. tax"),line_items:str[]("Items"),notes:str?' \
  --providers openai --out runs/invoice
```

`?` marks optional fields, `[]` arrays, `"..."` descriptions passed to the
model (`str|int|float|bool` with aliases supported).

### Files and images

Images become vision parts, PDFs OpenAI-style file parts, text files inline:

```bash
fastevals --image screenshot.png --structured-output 'x:int,y:int,width:int,height:int' \
  --prompt "Bounding box of the main widget" --providers openai --out runs/image
```

### Datasets, evaluators, consistency

```jsonl
{"id": "capital-france", "prompt": "Capital of France? City name only.", "expected": "Paris", "evaluator": "exact_match"}
{"id": "json-output", "prompt": "Return {\"status\": \"ok\"} as JSON.", "evaluator": "json_valid"}
```

```bash
fastevals --dataset cases.jsonl --nruns 3 --providers openai --out runs/dataset
```

Reports aggregate pass rates, latency and cost per model across all attempts.

## The report

Each `report.html` is a self-contained dashboard (Chart.js from CDN, no
build step, no telemetry): summary cards with fastest / cheapest /
top-throughput runs, sortable and filterable comparison table with CSV and
Markdown export, latency / throughput / token / cost charts, detailed result
cards, per-model aggregates for datasets.

## Python API

```python
import asyncio
from fastevals import RunConfig, run, save_report

config = RunConfig(prompt="Summarize eval best practices", providers=frozenset({"openai"}))
results = asyncio.run(run(config))
save_report(config, results, "runs")
print(results[0].output, results[0].latency_ms, results[0].total_cost_usd)
```

## Architecture

```mermaid
flowchart LR
    CLI["cli.py"] --> RC["RunConfig"]
    RC --> Runner["runner.py"]
    DS["dataset.py"] --> Runner
    EV["evaluators.py"] --> Runner
    Runner --> Reg["registry.py"]
    Reg --> Specs["ModelSpec"]
    Runner --> Prov["providers.py<br/>LiteLLM adapter"]
    Prov --> ST["structured.py<br/>schema · validation"]
    Runner --> PR["pricing.py"]
    Runner --> Rep["report.py<br/>single-file HTML"]
    Rep --> Out["run.json + report.html"]

    MCP["mcp_server.py"] --> Runner
```

Adding a provider means implementing the single `call_model` contract in
`providers.py`; adding a model means adding five lines to the TOML registry.
No other layers need to change.

## Development

```bash
make dev        # install with dev tooling
make check      # ruff + mypy --strict + tests with an 85% coverage floor
make format     # auto-fix style
```

The test suite is fully offline: provider calls are replaced by a recorded
stub at the LiteLLM boundary; live API calls never run in CI.

## Limitations (by design)

- No streaming yet — TTFT is reported as unavailable rather than faked; latency and throughput are end-to-end.
- One prompt template per case; no few-shot templating or conversation history.
- Evaluators are deterministic heuristics; LLM-as-judge scoring is not included.
- Pricing comes from your registry, not a live price feed — keep it current.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for where this is heading.

## License

MIT — see [LICENSE](LICENSE).
