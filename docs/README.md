# 文档地图

从 [项目 README](../README.md) 了解功能、性能和快速入口。以下按任务定位现有文档；长篇实验记录保留原路径，避免证据链接失效。

## 开发与数据

- [开发地图](development/README.md)：模块、配置、脚本、无硬件测试和提交边界。
- [系统架构](architecture.md)：领域对象、模块接口、端云路由与运行时矩阵。
- [数据约定](data.md)：manifest、标注、来源组和划分。
- [评测口径](evaluation.md)：质量、性能、失败类型及可比性；指标定义以此为准。
- [数据集与 LoRA 方案](dataset-and-finetuning-roadmap.md)、[环视数据集调研](surround-view-datasets.md)、[PS20 pilot](ps20-pilot.md)：数据与效果研究。

## 部署与性能

- [部署与运行地图](ops/README.md)：按 WSL vLLM、Jetson 和 Study 任务选择操作文档。
- [运行时性能测量](runtime-performance.md)：指标口径和各次运行结果。
- [TensorRT 优化速览](tensorrt-optimization-summary.md)：阶段结论；[完整实验记录](tensorrt-optimization.md)保存细节。
- [动态批处理与 paged KV](tensorrt-dynamic-batching-paged-kv.md)：P2/P3 实施及验收边界。

## 状态与历史

- [当前实现状态](status.md)：持续更新的实现与验证记录。
- [Jetson 部署核验](jetson-readiness.md)：板端实验事实与证据索引。
- [WSL 迁移交接](project-handoff-wsl.md)、[个人学习记录](personal_record.md)：历史背景，具体操作以当前部署文档为准。

新增详细文档时，更新本页及所属的开发或运维地图；移动文档时同步修复所有相对链接。`reports/` 中的原始证据和 `docs/source-study/` 中的本地学习材料默认不随 Git 上传。
