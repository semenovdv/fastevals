# Examples

These examples use the current `fasteval` CLI and the model registry in
`config/models.toml`.

Before running them, activate the project environment and configure a real
API key:

```bash
source .venv/bin/activate
cp .env.example .env
```

## Basic comparison

Runs the prompt for every configured reasoning effort of the selected provider
and writes JSON and HTML reports to `runs/basic/`:

```bash
fasteval \
  --prompt "Explain why evaluation matters for an LLM application in three bullet points." \
  --providers openai \
  --out runs/basic
```

## Structured output

Extracts a predictable object using the compact schema syntax:

```bash
fasteval \
  --prompt "Extract the company name and total amount from this invoice." \
  --file invoice.pdf \
  --structured-output 'company:str("Company name"),total:float("Invoice total")' \
  --providers openai \
  --out runs/invoice
```

## Image analysis

Pass an image with `--image` and describe the expected fields:

```bash
fasteval \
  --prompt "Return the bounding box of the main object." \
  --image screenshot.png \
  --structured-output 'x:int,y:int,width:int,height:int' \
  --providers openai \
  --out runs/image
```

The configured model is `gpt-5.6-luna`; its `none` and `low` reasoning runs
are expanded automatically from `config/models.toml`.
