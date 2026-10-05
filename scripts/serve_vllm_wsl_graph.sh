#!/usr/bin/env bash
# Candidate profile: capture batch=1 decode only; retain eager prefill.
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$project_root/scripts/serve_vllm_wsl.sh" \
    --no-enforce-eager \
    --compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}' \
    "$@"
