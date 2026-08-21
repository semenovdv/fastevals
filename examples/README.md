# Examples

Runnable examples for the current CLI. All of them work with the mock
provider without any API key — replace `mock` with `openai` (after copying
`.env.example` to `.env`) to spend real credits.

| Example | Command | Shows |
|---|---|---|
| [01_simple_prompt](01_simple_prompt/01_simple_prompt.sh) | basic prompt, all reasoning efforts | matrix runs, JSON + HTML reports |
| [02_structured_output](02_structured_output.sh) | compact schema → JSON Schema | schema validation of responses |
| [03_dataset_evaluation](03_dataset_evaluation) | `cases.jsonl` + `--nruns` | datasets, evaluators, aggregates |
| [04_mcp_server](04_mcp_server.md) | `fasteval-mcp` | connecting fasteval to Claude |

## Zero-credit smoke run

```bash
fasteval --prompt "Hello from fasteval" --providers mock --out runs/demo
```

## Real runs

```bash
cp .env.example .env    # then fill in your key
fasteval --prompt "Explain evaluation in three bullets" --providers openai --out runs
```

The configured model is `gpt-5.6-luna`; its `none` and `low` reasoning runs
are expanded automatically from `config/models.toml`. Pricing in the registry
is USD per 1M tokens — keep it current for honest cost reports.
