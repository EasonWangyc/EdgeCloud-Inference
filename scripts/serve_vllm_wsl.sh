#!/usr/bin/env bash
# Single-image Qwen3-VL service for the RTX 4060 Laptop (8 GB).
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_NO_USAGE_STATS=1

# Optional user-space headers when python3.10-dev is unavailable system-wide.
headers_root="$project_root/artifacts/vllm-python-headers/usr/include"
if [[ -f "$headers_root/python3.10/Python.h" ]]; then
    export C_INCLUDE_PATH="$headers_root/python3.10:$headers_root${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
fi

exec .venv-vllm/bin/vllm serve \
    "$project_root/models/Qwen3-VL-2B-Instruct-89644892" \
    --served-model-name Qwen/Qwen3-VL-2B-Instruct \
    --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --generation-config vllm \
    --max-model-len 2048 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 1024 \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt '{"image":1,"video":0}' \
    --mm-processor-kwargs '{"min_pixels":200704,"max_pixels":200704}' \
    --enforce-eager \
    "$@"
