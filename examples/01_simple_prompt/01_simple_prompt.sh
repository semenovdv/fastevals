#!/usr/bin/env bash
# Basic comparison across reasoning efforts of the configured model.
# Writes JSON and HTML reports to runs/basic/.
set -euo pipefail

fasteval \
  --prompt "Explain why evaluation matters for an LLM application in three bullet points." \
  --providers openai \
  --out runs/basic
