# 配置地图

| 目录 | 用途 |
| --- | --- |
| `workloads/` | 冻结 prompt、输入尺寸、生成参数和 JSON schema；比较实验时保持不变 |
| `studies/server/`、`studies/jetson/`、`studies/wsl/` | 按运行环境划分的 runtime、模型身份、样本和报告配置 |
| `studies/archive/` | 已完成的 WSL vLLM 批处理与并发研究快照，仍可按原身份复现 |
| `flows/train/`、`flows/merge/`、`flows/quantize/` | 训练、合并与量化流程 |
| `flows/export/`、`flows/build/` | 模型导出与 engine 构建；先预检再执行 |
| `flows/archive/` | 历史 reduced-vocab 候选流程，保留原有解析路径和身份 |
| `training/` | LoRA 训练参数与数据路径 |
| `tensorrt/` | TensorRT 专项候选与对照配置 |
| `data/` | 数据集、校准样本选择等配置 |

默认单图 workload 为 [`workloads/parking_risk_v2_strict_json.json`](workloads/parking_risk_v2_strict_json.json)。运行入口和配置字段见 [操作入口](../docs/operations.md)。每个分组下的 `.example.json` 可复制为同目录 `*.local.json` 后填写本机路径；该副本由 Git 忽略。正式提交的配置保留可复现的相对路径与版本身份，服务密钥只从环境变量读取。
