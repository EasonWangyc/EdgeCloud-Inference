#!/usr/bin/env bash
# Experimental four-sequence profile without cross-request prefix/MM reuse.
set -euo pipefail

profile="${1:-graph}"
if [[ $# -gt 0 ]]; then
    shift
fi
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
case "$profile" in
    eager) command=(bash "$project_root/scripts/serve_vllm_wsl_batch.sh" 4) ;;
    graph) command=(bash "$project_root/scripts/serve_vllm_wsl_batch_graph.sh") ;;
    *) echo "Usage: bash scripts/serve_vllm_wsl_no_reuse.sh [eager|graph] [vLLM options]" >&2; exit 2 ;;
esac

# Development endpoint is used only to record effective settings on localhost.
export VLLM_SERVER_DEV_MODE=1
exec "${command[@]}" --no-enable-prefix-caching --mm-processor-cache-gb 0 "$@"
