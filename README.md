# fasteval

Run one prompt across a configurable matrix of models and settings, save every response, and get a readable HTML comparison report.

## Quick start

```bash
python3 -m pip install -e '.[native]'
fasteval --prompt "Explain evaluation in three bullets" --providers openai --out runs
```

## Agent-friendly usage

The CLI is non-interactive and returns structured JSON. Its complete interface is:

```bash
fasteval --prompt "" --providers openai --out "runs"
```

The model matrix is configured in `config/models.toml`. Exit code `0` means every model completed; exit code `1` means at least one model failed.

Useful discovery commands:

```bash
fasteval --help
```

The command writes every run into its own timestamped directory under `runs/`, containing `run.json` and a standalone `report.html`.

## Structured output

Use `--structured-output` when the model must return a predictable JSON object. The compact schema is a comma-separated list of `field:type` definitions:

```bash
fasteval \
  --prompt "Extract the invoice data" \
  --structured-output "invoice_number:str,total:float,currency:str" \
  --providers openai \
  --providers openai
```

### Supported types

| Compact type | JSON Schema type |
|---|---|
| `str`, `string` | `string` |
| `int`, `integer` | `integer` |
| `float`, `number` | `number` |
| `bool`, `boolean` | `boolean` |
| `object` | `object` |
| `any` | any JSON value |

Example:

```text
name:str,age:int,score:float,active:bool,metadata:object,raw:any
```

### Optional fields

Add `?` after the type. Optional fields are omitted from JSON Schema `required`:

```text
invoice_number:str,currency:str?,notes:str?
```

Required fields:

```json
["invoice_number"]
```

### Arrays

Add `[]` after the type:

```text
tags:str[],page_numbers:int[],confidence_scores:float[],flags:bool[]
```

This creates schemas such as:

```json
{
  "type": "array",
  "items": {"type": "string"}
}
```

### Descriptions

Put a quoted description in parentheses after the type:

```text
invoice_number:str("Unique invoice identifier"),total:float("Amount including tax"),currency:str("ISO currency code")
```

Descriptions are copied to the JSON Schema `description` property and help models understand the expected output.

### Commas and quotes in descriptions

Descriptions may contain commas:

```text
total:float("Total amount, including taxes and discounts"),notes:str("Short explanation, if available")
```

Use either single or double quotes:

```text
title:str('Document title'),summary:str("One-sentence summary")
```

### Complete example

```bash
fasteval \
  --file invoice.pdf \
  --prompt "Extract all relevant invoice fields" \
  --structured-output 'invoice_number:str("Unique invoice identifier"),vendor:str("Seller name"),total:float("Total amount, including taxes"),currency:str("ISO currency code"),line_items:str[]("Purchased item descriptions"),notes:str?' \
  --providers "openai|gemini" \
  --out runs/invoice
```

The equivalent JSON Schema is:

```json
{
  "type": "object",
  "properties": {
    "invoice_number": {"type": "string", "description": "Unique invoice identifier"},
    "vendor": {"type": "string", "description": "Seller name"},
    "total": {"type": "number", "description": "Total amount, including taxes"},
    "currency": {"type": "string", "description": "ISO currency code"},
    "line_items": {"type": "array", "items": {"type": "string"}, "description": "Purchased item descriptions"},
    "notes": {"type": "string"}
  },
  "required": ["invoice_number", "vendor", "total", "currency", "line_items"],
  "additionalProperties": false
}
```

## Model registry

Permanent model settings live in `config/models.toml`. Each TOML section is one concrete provider/model/reasoning combination:

```toml
["openai:gpt-5.6-sol:low"]
provider = "openai"
model = "gpt-5.6-sol"
api_key_env = "OPENAI_API_KEY"
reasoning_effort = "low"
```

Quotes around the section name are required by TOML because it contains colons. TOML keeps the registry readable and supports comments without requiring an extra dependency. The CLI selects providers and a model strategy:

```bash
fasteval --prompt "Extract the key risks" --providers "openai|gemini" --out runs
```

Each entry describes one model. `reasoning_efforts` uses pipe-separated values and the runner expands them into separate evaluation runs. Pricing fields are USD per 1M tokens: `input_cost_usd_per_mtok`, `cached_input_cost_usd_per_mtok`, `output_cost_usd_per_mtok`, and `cached_write_cost_usd_per_mtok`. API keys are read from `api_key_env` environment variables and never belong in the repository.

Create local credentials from the template:

```bash
cp .env.example .env
```

Load the variables in your shell before running `fasteval`. The real `.env` file must remain local.

## LLM providers

Model calls go through [LiteLLM](https://github.com/BerriAI/litellm). Configure models in `config/models.toml`; the runner resolves API keys from environment variables listed in each entry.

Install provider dependencies when running real models:

```bash
python3 -m pip install -e '.[native]'
```

Without them, mock runs and CLI/report development continue to work; a real provider run returns a structured error for the affected model.

## Development

```bash
python3 -m pip install -e '.[dev]'
python3 -m pytest
```

MIT License.
