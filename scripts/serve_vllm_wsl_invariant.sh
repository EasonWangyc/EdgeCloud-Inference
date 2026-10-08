#!/usr/bin/env bash
# Experimental batch-invariant profile, verified separately from throughput mode.
set -euo pipefail

export VLLM_BATCH_INVARIANT=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_root/scripts/serve_vllm_wsl_batch.sh" 4 "$@"
