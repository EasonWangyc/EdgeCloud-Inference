# 测试地图

`tests/` 随源码提交，用于保护模块接口、指标口径和部署脚本的无硬件行为；它不会进入从 `src/` 构建的运行包。`__pycache__/` 等生成文件由 Git 忽略。

| 目录 | 覆盖范围 |
| --- | --- |
| `app/`、`assessment/`、`casebook/` | 命令与配置、严格 JSON、样本与划分 |
| `inference/`、`studies/` | 运行时适配、路由、质量和性能指标 |
| `training/` | LoRA 数据、人工复核和校准样本 |
| `tensorrt/` | engine 构建、补丁、KV cache 和部署预检 |
| `benchmarks/` | vLLM/Edge-LLM benchmark、Nsight 和遥测汇总 |
| `fixtures/` | 小型、可提交的测试输入 |

在仓库根目录运行 `uv run python -m unittest discover -s tests`。新增测试放在对应领域目录，命名为 `test_*.py`；模型、CUDA 和 Jetson 实测另行记录在本地 `reports/`。
