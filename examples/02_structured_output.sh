#!/usr/bin/env bash
# Structured output: compact schema compiles to JSON Schema, every response
# is validated against it before it reaches run.json.
set -euo pipefail

fasteval \
  --prompt "Extract the company name and total amount from this invoice." \
  --structured-output 'company:str("Company name"),total:float("Invoice total, including taxes"),line_items:str[]("Purchased item descriptions"),notes:str?' \
  --providers "${FASTEVAL_PROVIDERS:-mock}" \
  --out runs/structured
