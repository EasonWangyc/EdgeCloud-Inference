#!/usr/bin/env bash
# Multi-sequence eager candidate for measured HTTP concurrency.
set -euo pipefail

sequence_limit="${1:-2}"
if [[ $# -gt 0 ]]; then
    shift
fi
case "$sequence_limit" in
    2|4) ;;
    *) echo "Usage: bash scripts/serve_vllm_wsl_batch.sh [2|4] [vLLM options]" >&2; exit 2 ;;
esac

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_root/scripts/serve_vllm_wsl.sh" \
    --max-num-seqs "$sequence_limit" "$@"
