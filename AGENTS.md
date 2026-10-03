# EdgeCloud-Inference 仓库指南

## 项目范围

本仓库实现低速泊车场景的单图风险理解、领域 LoRA 适配与 Jetson 部署评测。业务主链路为：

```text
ParkingCase -> RiskRuntime -> InferenceRecord -> StudyReport
```

目标模型为 `Qwen/Qwen3-VL-2B-Instruct`。Transformers 实验分为服务器正确性参考和
Jetson Transformers FP16 板端基线；TensorRT Edge-LLM 提供 Jetson 最终部署运行时。
服务器结果用于任务正确性、误差分析和训练，不作为板端性能基线。Jetson
Transformers FP16 与 Jetson TensorRT Edge-LLM 使用相同模型 revision、工作负载和
数据集进行同机比较。实验序列包括服务器正确性参考、Jetson Transformers FP16、
TensorRT Edge-LLM FP16、领域 LoRA、合并模型和 LLM backbone INT4，并保存每个阶段的
实际证据。

项目产出泊车安全提示。风险事件集合包括：

- `vru_near_maneuver_path`
- `vehicle_near_maneuver_path`
- `fixed_obstacle_near_path`
- `narrow_passage`
- `visibility_occlusion`
- `parking_space_conflict`

文档和项目描述围绕已实现功能、输入输出、技术路径、评测口径和实测证据展开，采用项目中心的客观表述。

## 领域对象与 Module

- `ParkingCase`：一张泊车场景图片、来源组、数据集划分和人工标注的组合。
- `ParkingAssessment`：风险等级、风险事件、证据和驾驶建议构成的严格 JSON 内容。
- `InferenceRecord`：一次运行时执行的输入标识、配置、`ParkingAssessment`、时延与失败事实。
- `StudyReport`：一个冻结工作负载上的质量、性能、环境与失败归因汇总。

核心 Module 及其 Interface：

| Module | Interface | 职责 |
| --- | --- | --- |
| `assessment` | `ParkingAssessment.from_mapping(payload)` | 解析、校验和序列化严格 JSON |
| `casebook` | `ParkingCaseCatalog.load()` / `ParkingCaseCatalog.validate()` | 管理泊车样本、标注、来源组和数据集划分 |
| `inference` | `RiskRuntime.analyze(case, workload) -> InferenceRecord` | 适配 Transformers 与 TensorRT Edge-LLM，保留运行时事实 |
| `studies` | `StudyRunner.run(casebook, runtime, study) -> StudyReport` | 完成质量评测、性能采集和报告生成 |

`app` 负责组合 Module 并提供命令入口。`scripts` 承担训练、合并、导出和构建等显式流程。

## 目录结构

```text
src/parksight_vlm/
  assessment/       # ParkingAssessment 与 JSON schema
  casebook/         # ParkingCase、标注、manifest 与划分校验
  inference/        # RiskRuntime Interface 和具体 Adapter
  studies/          # InferenceRecord、评测、benchmark 与环境快照
  app/              # 单图分析与实验运行入口
configs/
  workloads/        # 冻结 prompt、生成参数和 schema 版本
  studies/          # 基线、LoRA、FP16、INT4 实验配置
data/
  manifests/        # 可提交的样本元数据
  annotations/      # 可提交的人工标注
  raw/              # 本地图片输入
artifacts/          # adapter、导出模型和 engine
reports/            # StudyReport 与图表
scripts/            # train、merge、export、build 等显式入口
tests/              # 围绕 Module Interface 的无硬件测试
docs/               # 架构、数据卡、部署和报告说明
```

## 实施顺序

1. 实现 `ParkingAssessment`、`ParkingCase`、`ParkingCaseCatalog` 和冻结工作负载配置。
2. 在服务器运行 Transformers 正确性参考，生成 `InferenceRecord` 并完成质量研究。
3. 在 Jetson 运行 Transformers FP16 板端基线；无法加载或 OOM 也作为明确实验结果。
4. 建立 TensorRT Edge-LLM FP16 Adapter，完成 Jetson 同机运行时研究。
5. 基于冻结测试集的领域误差分析训练 LoRA，完成合并模型的同口径研究。
6. 生成 INT4 LLM backbone 版本，完成质量、时延、内存、功耗与温度对比。

每个阶段保存模型 revision、adapter revision、精度、输入尺寸、prompt、生成参数、数据集版本、执行命令、环境快照和结果状态。

## 数据与评测

- `source_group_id` 将同一视频、连续采集序列或不可分数据来源的样本归入同一数据集划分。
- 冻结测试集在 LoRA 调参、量化校准和阈值调整之前确定，并保留人工标注、数据版本和来源信息。
- 失败记录使用明确类别，包括 JSON 解析失败、模型拒答、超时、显存不足、算子支持状态和任务判断错误。
- `StudyReport` 记录 JSON 有效率、风险等级准确率、事件 micro-F1、不安全建议率、按类别错误、冷启动、预处理/视觉编码/prefill/decode/端到端时延、p50/p90/p99、首 token 延迟、tokens/s、峰值内存、功耗、温度和失败样例。
- 服务器 Transformers、Jetson Transformers FP16 和 Jetson TensorRT Edge-LLM
  分别使用独立 `study_id` 与报告；板端性能结论只比较两个 Jetson runtime。

## 编码、测试与操作

- 使用 Python 3.10+、四空格缩进、类型标注、`pathlib.Path`、dataclass 和 `snake_case`。
- 单元测试围绕 Module 的 Interface 编写，并可在无模型权重、CUDA、摄像头和 Jetson 的环境中运行。
- 模型与板端验证属于显式集成测试，运行记录包含 backend 状态和失败原因。
- 标准无硬件测试使用 `uv run python -m unittest discover -s tests`；修改后运行 `git diff --check`。
- 项目配置固定依赖和模型版本。ONNX Runtime 用于导出和数值一致性研究，TensorRT Edge-LLM 用于 Jetson 运行时研究。
- 系统依赖、远程 Jetson、大模型权重、Git 提交、推送和历史操作遵循明确授权流程。

## 命令执行协作方式

- 默认由用户亲自执行本机、服务器和 Jetson 终端命令。Codex 提供命令、执行目的、
  预期输出和成功/失败判断标准，等待用户贴回结果后再给下一步。
- Jetson 命令以 SSH 登录后的 Bash 形式给出，不把远端 Bash 命令包装成 Windows
  PowerShell 一键执行。
- 安装依赖、下载模型、模型转换、GPU 推理、benchmark、系统升级、重启、Git
  commit/push 等操作，不由 Codex 代执行，除非用户在当前请求中明确要求 Codex 执行。
- 较长流程按可验证的小步骤给命令；每一步先解释输出，避免一次给出无法定位失败位置
  的大段脚本。
- Codex 可以按请求编辑仓库文件；修改完成后给出由用户执行的测试和 Git 命令。
