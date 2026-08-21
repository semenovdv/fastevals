#!/usr/bin/env bash
# Dataset evaluation with automatic scoring, repeated for consistency.
set -euo pipefail

fasteval \
  --dataset "$(dirname "$0")/cases.jsonl" \
  --nruns 3 \
  --providers "${FASTEVAL_PROVIDERS:-openai}" \
  --out runs/dataset
