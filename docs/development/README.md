# 开发地图

## 修改位置

| 任务 | 代码与配置 | 说明 |
| --- | --- | --- |
| 调整 JSON 契约或风险语义 | `src/parksight_vlm/assessment/` | 同步检查 [架构](../architecture.md)与 [评测口径](../evaluation.md) |
| 调整样本、标注或划分 | `src/parksight_vlm/casebook/`、`data/manifests/`、`data/annotations/` | 遵循 [数据约定](../data.md)的来源组规则 |
| 调整 Transformers、vLLM、Edge-LLM 或路由 | `src/parksight_vlm/inference/` | 保留 `InferenceRecord` 中的后端事实 |
| 调整质量与性能指标 | `src/parksight_vlm/studies/` | 保持 [指标定义](../evaluation.md)和失败记录一致 |
| 调整命令入口 | `src/parksight_vlm/app/`、`scripts/` | 先查 [脚本地图](../../scripts/README.md) |

配置放在 [配置地图](../../configs/README.md)列出的目录；冻结 workload 与实际运行的 study 分开维护。测试按模块放在 `tests/app/`、`tests/assessment/`、`tests/casebook/`、`tests/inference/`、`tests/studies/`；专项测试在 `tests/training/`、`tests/tensorrt/`、`tests/benchmarks/`。目录说明见 [测试地图](../../tests/README.md)。新增测试使用 `test_*.py`，优先覆盖接口行为及失败路径。

## 本地验证

Python 3.10+；四空格缩进、类型标注、`snake_case`、`pathlib.Path`。无硬件回归不加载模型权重：

```bash
uv run python -m unittest discover -s tests
git diff --check
```

模型加载、GPU benchmark 和 Jetson 验证按 [部署与运行地图](../ops/README.md)单独执行，结果写入本地 `reports/`。

## 哪些文件进入 Git

提交源码、测试、文档、SVG、可复现的配置、manifest 和人工标注。`data/raw/`、`data/processed/`、`models/`、`artifacts/`、`reports/`、虚拟环境、`docs/source-study/`、`.env*` 和 `configs/**/*.local.json` 是本地输入或产物，默认由 `.gitignore` 排除。`patches/` 是本地补丁缓存；历史上已跟踪的 `0012-configurable-timing-cache.patch` 是例外，其他补丁不会自动上传。不要把服务密钥写进 study JSON 或报告；使用环境变量。提交前检查 `git status --short` 和 `git diff --cached --stat`，仅暂存本次需要的文件。
