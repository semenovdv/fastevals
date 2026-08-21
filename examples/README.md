# Examples

Runnable examples for the current CLI. All of them need a provider API key —
copy `.env.example` to `.env` and fill it in first.

| Example | Command | Shows |
|---|---|---|
| [01_simple_prompt](01_simple_prompt/01_simple_prompt.sh) | basic prompt, all reasoning efforts | matrix runs, JSON + HTML reports |
| [02_structured_output](02_structured_output.sh) | compact schema → JSON Schema | schema validation of responses |
| [03_dataset_evaluation](03_dataset_evaluation) | `cases.jsonl` + `--nruns` | datasets, evaluators, aggregates |
| [04_mcp_server](04_mcp_server.md) | `fastevals-mcp` | driving fasteval from Claude |

## Basic run

```bash
cp .env.example .env    # then fill in your key
fastevals --prompt "Explain evaluation in three bullets" --providers openai --out runs
```

The configured model is `gpt-5.6-luna`; its `none` and `low` reasoning runs
are expanded automatically from `config/models.toml`. Pricing in the registry
is USD per 1M tokens — keep it current for honest cost reports.
