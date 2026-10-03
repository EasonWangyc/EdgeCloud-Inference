# 系统架构

## 架构中心

系统围绕泊车样本、泊车风险评估、推理记录和研究报告组织。模型运行时、量化精度、adapter 和板端环境通过推理记录与研究报告进入同一条数据流。

```text
ParkingCaseCatalog -> ParkingCase -> RiskRuntime -> InferenceRecord -> StudyRunner -> StudyReport
```

## 深 Module

| Module | Interface | 输入 | 输出 | 内部职责 |
| --- | --- | --- | --- | --- |
| `assessment` | `ParkingAssessment.from_mapping(payload)` / `audit_assessment_semantics(assessment)` | JSON mapping 或已解析 assessment | `ParkingAssessment` / `SemanticAudit` | JSON 契约校验，以及风险等级、事件和驾驶建议的可解释语义审计 |
| `casebook` | `ParkingCaseCatalog.load()` / `ParkingCaseCatalog.validate()` | manifest、标注、工作负载选择 | `ParkingCase` 集合 | 来源组约束、数据集划分、冻结测试集与图片引用解析 |
| `inference` | `RiskRuntime.analyze(case, workload)` | `ParkingCase`、冻结工作负载 | `InferenceRecord` | 输入准备、Adapter 调用、输出解析、阶段计时与运行失败记录 |
| `studies` | `StudyRunner.run(casebook, runtime, study)` | casebook、runtime、研究配置 | `StudyReport` | 推理记录归档、任务质量、性能分位数、环境快照与失败归因 |

每个 Module 通过一个小 Interface 向调用方提供深度。`inference` 内部包含 `TransformersRuntime`、`EdgeLlmRuntime` 与 `VllmRuntime` 三个 Adapter。`studies` 通过 `RiskRuntime` Interface 运行不同后端的任务与性能研究。

## 对象关系

```text
ParkingCase
  - case_id
  - image_ref
  - source_group_id
  - split
  - reference_assessment

ParkingAssessment
  - schema_version
  - risk_level
  - events
  - evidence
  - driver_advice

InferenceRecord
  - case_id
  - runtime_identity
  - workload_identity
  - assessment | failure
  - stage_timings
  - resource_snapshot

StudyReport
  - study_identity
  - environment_snapshot
  - quality_metrics
  - performance_metrics
  - failure_summary
```

`ParkingAssessment` 作为人工标注和模型解析结果的共享结构。`InferenceRecord` 保存单次执行事实。`StudyReport` 聚合多个推理记录，保留实验配置和环境快照。

## 目录职责

```text
src/parksight_vlm/
  assessment/
    model.py          # ParkingAssessment、枚举和校验错误
  casebook/
    model.py          # ParkingCase、来源组和数据集划分
    catalog.py        # ParkingCaseCatalog Interface 与 manifest 读取
  inference/
    runtime.py        # RiskRuntime Interface、InferenceRecord 与失败类型
    transformers.py   # TransformersRuntime Adapter
    edge_llm.py       # EdgeLlmRuntime Adapter
    vllm.py           # vLLM OpenAI-compatible HTTP Adapter
  studies/
    model.py          # Study 与 StudyReport
    runner.py         # StudyRunner
    quality.py        # micro/macro 质量指标与语义一致性汇总
    performance.py    # 时延、资源和环境汇总
  app/
    analyze_image.py  # 单图分析命令
    run_study.py      # 批量研究命令
  flows.py            # 外部训练、合并、导出和构建的计划与执行证据
configs/
  workloads/parking_risk_v1.json
  studies/transformers_base.json          # 服务器 Transformers 正确性参考
  studies/edgellm_fp16.json               # Jetson TensorRT Edge-LLM FP16
  studies/jetson_transformers_fp16.json   # Jetson Transformers FP16 基线
  flows/*.example.json
data/
  manifests/parking_risk_v1.jsonl
  annotations/parking_risk_v1.jsonl
  raw/
artifacts/
reports/
scripts/
  train_lora.py
  merge_lora.py
  export_model.py
  build_engine.py
tests/
  assessment/
  casebook/
  inference/
  studies/
```

## 固定工作负载

`workload` 固定任务语义、prompt、生成参数、输入尺寸、schema version 和泊车风险事件集合。`study` 记录 runtime、backend revision、模型 revision、adapter revision、精度、样本选择、运行次数和功耗模式。

泊车风险事件集合：

- `vru_near_maneuver_path`
- `vehicle_near_maneuver_path`
- `fixed_obstacle_near_path`
- `narrow_passage`
- `visibility_occlusion`
- `parking_space_conflict`

建议 `driver_advice` 使用：`maintain_observation`、`slow_down`、`yield`、`prepare_to_stop`、`change_maneuver_when_safe`。

`workload` 保持任务可比性；`study` 表达一次实验的运行条件。该拆分使 FP16、LoRA 和 INT4 结果在同一工作负载下可直接比较。

## Runtime 实验矩阵

| Study | 环境 | Runtime | 作用 | 性能比较 |
| --- | --- | --- | --- | --- |
| `transformers_base` | GPU 服务器 | Transformers | 正确性参考、完整质量研究和误差分析 | 不与 Jetson 时延直接比较 |
| `server_vllm_http` | GPU 服务器 | vLLM HTTP | 云侧服务后端研究 | 记录云端请求时延；不与 Jetson 时延直接比较 |
| `edge_vllm_router` | Jetson + 可访问的 GPU 服务 | Edge-LLM + vLLM HTTP | 按授权、健康状态和策略分发，并记录失败回退 | 报告端到端路径与回退；不将异构路径时延解释为框架 A/B |
| `jetson_transformers_fp16` | Jetson | Transformers FP16 | 板端框架基线；保留成功、OOM 或依赖失败事实 | 与 Jetson Edge-LLM 同机比较 |
| `edgellm_fp16` | Jetson | TensorRT Edge-LLM FP16 | 最终部署 FP16 基线 | 与 Jetson Transformers FP16 同机比较 |

跨后端质量对照必须冻结相同模型 revision、workload、输入尺寸、prompt、生成参数和测试集。
服务器 Transformers 与 vLLM 的结果用于服务器侧模型任务和服务后端研究；Jetson
Transformers 与 Jetson Edge-LLM 的同机结果用于回答部署优化带来的板端时延、内存、功耗
和温度变化。不同设备间的时延不用于归因推理框架性能差异。

`EdgeCloudRouterRuntime` 组合现有两个 HTTP Adapter。每次执行先获取
`RoutingSignalsProvider` 快照：`cloud_allowed` 明确约束图片能否上传，HTTP 健康探测更新
端/云服务可达状态，`RoutingPolicy` 选择 primary、按端侧初判对指定高风险场景升级云复核，
并限制哪些失败可以回退。生成的
`InferenceRecord.routing_decision` 保存策略、信号、尝试路径及失败类别，阶段时延包含
路由开销。当前默认 provider 不采集温度、显存或链路质量遥测；缺失值不会被伪造成测量，
需由设备/网络 provider 接入后才可用于资源门槛判断。

实现和示例配置见 `src/parksight_vlm/inference/routing.py` 与
`configs/studies/jetson_edge_vllm_router_ps20_pilot.example.json`。

## 测试 seam

1. `ParkingAssessment.from_mapping`：结构化内容的解析与校验。
2. `ParkingCaseCatalog.validate`：来源组、样本标识与数据集划分的校验。
3. `RiskRuntime.analyze`：Adapter 对泊车样本和工作负载的实际执行。
4. `StudyRunner.run`：推理记录到研究报告的聚合。

四个 seam 均已有无硬件测试。真实 Transformers、TensorRT Edge-LLM 和 Jetson
资源采集属于显式集成测试，只有实际执行记录可以进入研究证据。
