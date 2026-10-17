# 部署与运行地图

| 任务 | 入口 | 结果或注意事项 |
| --- | --- | --- |
| 本机 WSL 启动 vLLM | [WSL vLLM 部署](../vllm-wsl-deployment.md) | 使用独立 `.venv-vllm/`；记录服务配置与模型身份 |
| Jetson 构建并运行 Edge-LLM | [Edge-LLM 部署](../edgellm-deployment.md) | 检查板端依赖、engine、插件和 HTTP 服务 |
| 单图分析或运行 Study | [操作入口](../operations.md) | 选择对应 `configs/studies/`，结果写入 `reports/` |
| 比较性能或定位失败 | [运行时性能](../runtime-performance.md)、[评测口径](../evaluation.md) | 同机、同模型、同 workload 才做框架性能归因 |
| 查看 TensorRT 优化阶段 | [优化速览](../tensorrt-optimization-summary.md)、[完整实验](../tensorrt-optimization.md) | 区分默认配置、候选 engine 与低层 microbenchmark |
| 排查 Jetson P2/P3 | [动态批处理与 paged KV](../tensorrt-dynamic-batching-paged-kv.md)、[板端核验](../jetson-readiness.md) | 按实验状态和原始证据判定是否可部署 |

运行前核对设备、精度、模型 revision、workload、输入尺寸和报告路径。服务器结果用于正确性与云侧服务研究；板端性能只在 Jetson 同机比较。原始日志、engine 和模型权重保存在本地目录，不随 Git 上传。
