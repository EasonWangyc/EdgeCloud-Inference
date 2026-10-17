# 脚本地图

优先从 `parksight-analyze`、`parksight-study` 或 [操作入口](../docs/operations.md)开始；`scripts/` 存放训练、部署和诊断的显式命令。

| 任务 | 入口示例 | 说明 |
| --- | --- | --- |
| LoRA 与数据准备 | `data/generate_lora_dataset.py`、`finetune_qwen3_vl_lora.py`、`merge_qwen3_vl_lora.py` | 数据集准备、训练与合并 |
| 量化、导出、构建 | `quantize_qwen3_vl_int4_awq_domain.py`、`export_model.py`、`build_edgellm_vlm_engines.py` | 使用 `configs/flows/` 中对应配置 |
| vLLM 服务与性能 | `serve_vllm_wsl.sh`、`benchmark_vllm_http.py` | WSL 启动与 HTTP benchmark |
| Jetson 服务与性能 | `serve_edgellm.py`、`run_edgellm_concurrency_sweep.py` | 板端 HTTP 服务和并发研究 |
| 数据复核 | `data/` | 复核表、审核包和标注修订 |
| TensorRT 诊断 | `diagnostics/audit_edgellm_kv_cache.py`、`diagnostics/summarize_tensorrt_nsight_trace.py` | 候选 engine、KV cache 和 profiler 分析 |
| 实验准备 | `experiments/` | paged ONNX、融合、补丁及隔离构建候选 |

根目录保留常用 CLI 路径，专项脚本按 `data/`、`diagnostics/`、`experiments/` 分组。执行前先看脚本 `--help` 与相应 [部署地图](../docs/ops/README.md)；输出默认留在 `reports/` 或 `artifacts/`，不要直接上传模型权重和原始日志。
