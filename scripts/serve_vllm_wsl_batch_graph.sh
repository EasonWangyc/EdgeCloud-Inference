#!/usr/bin/env bash
# Experimental four-sequence decode Graph profile; retain eager prefill.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_root/scripts/serve_vllm_wsl_batch.sh" 4 \
    --no-enforce-eager \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4]}' \
    "$@"
