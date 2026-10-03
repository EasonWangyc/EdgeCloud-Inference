# 评测口径

## 冻结单位

一次可比较研究由以下身份共同确定：

- workload SHA-256；
- 数据集 manifest、annotation 与 split；
- backend 名称及 revision；
- model id 及不可变 revision；
- LoRA adapter revision；
- 精度、power mode 与 repetitions。

任一项变化都应产生新的 `study_id` 和报告，不覆盖既有证据。

## 实验角色与可比性

| 实验 | 主要结论 | 不用于 |
| --- | --- | --- |
| 服务器 Transformers | 模型、prompt、严格 JSON 和任务质量的正确性参考 | Jetson 部署性能结论 |
| 服务器 vLLM | 云侧服务后端的模型质量、请求时延和失败研究 | Jetson 部署性能结论；纯模型服务端 TTFT（除非服务端显式上报） |
| Jetson Transformers FP16 | 原生框架在目标板上的可运行性、OOM、质量和性能基线 | 服务器训练吞吐比较 |
| Jetson TensorRT Edge-LLM FP16/INT4 | 最终部署质量与板端性能 | 替代服务器误差分析 |

质量一致性可以在不同 runtime 间比较，但必须固定模型 revision、workload 和数据集。
时延、内存、功耗和温度的加速结论只比较 Jetson Transformers 与 Jetson TensorRT
Edge-LLM。服务器 GPU 的性能数值单独保存，不进入 Jetson 加速比。

Jetson Transformers FP16 如果无法加载、发生 OOM 或依赖不兼容，仍需生成失败记录；
该结果构成板端原生框架基线的一部分，不得省略后直接宣称 TensorRT 加速。

## 质量指标

- **JSON 有效率**：严格解析为 `ParkingAssessment` 的记录数 / 总记录数。
- **风险等级准确率**：预测 `risk_level` 与人工标注一致的记录数 / 总记录数；解析失败按错误计。
- **事件 micro precision/recall/F1**：在六类风险事件上累计 TP、FP、FN 后计算。
- **事件 macro-F1**：分别计算六类事件的 F1 后取算术平均；每类的 support、TP、FP、FN、
  precision、recall 和 F1 同时写入 `event_metrics`。它用于暴露稀有事件或单一类别的
  退化，不能替代整体事件 micro-F1。
- **不安全建议率**：人工标注为 high risk，或包含行人/车辆接近行驶路径事件时，预测必须至少包含
  `yield` 或 `prepare_to_stop`；解析失败也计为不安全。
- **按事件错误**：每个事件分别统计 false positive 与 false negative。
- **语义一致性率**：在 JSON 有效的预测中，没有触发跨字段语义告警的记录数 / JSON 有效记录数。
  当前告警包括低风险配合 `prepare_to_stop`、高风险或近路径事件缺少 `yield` /
  `prepare_to_stop`，以及非低风险没有事件。该指标是诊断信息，不会自动修正输出，
  也不替代风险准确率或事件 micro-F1。
- **语义告警计数**：`semantic_issue_counts` 按稳定告警名称统计上述冲突，便于定位
  `risk_level`、`events` 与 `driver_advice` 的跨字段偏差。

当前安全策略是 `parking_risk_v1` 的保守规则。修改规则需要提升 schema/workload 版本，
不得用新规则重算并覆盖旧报告。

## 性能指标

- 质量指标只对成功解析的记录计算；性能指标对已返回 `raw_output` 的后端完成记录计算，
  因为 JSON 解析失败不应抹掉已经发生的推理耗时。未返回模型输出的记录不进入性能分位数。
- 每个实际采集到的阶段分别计算 p50、p90、p99。
- 首个成功执行的端到端时延记为 cold start。
- `tokens_per_second` 只在 runtime 同时报告 output tokens 与 decode latency 时计算，表示
  decode-only tokens/s；`aggregate_output_tokens_per_end_to_end_second` 使用全部后端完成记录
 计算，表示端到端链路的聚合输出速率，不能替代 decode-only tokens/s。
- 峰值内存和峰值温度取最大值，平均功耗对已采集记录取算术平均。
- 未被 runtime 实际测量的指标保持 `null`，不使用估算值补齐。
- Transformers profile 可记录 `vision_encode_ms`、`prefill_ms`、`decode_ms` 和
  `time_to_first_token_ms`；Edge-LLM HTTP adapter 默认请求流式响应，并从首个非空
  `delta.content` 到达时记录客户端观测 TTFT，同时记录请求构造时间和
  `http_round_trip_ms`。vLLM HTTP Adapter 使用 OpenAI 图像 data URI、可选 Bearer API key
  环境变量和 JSON mode；JSON mode 只约束 JSON 语法，最终仍由 `ParkingAssessment` 做 schema
  校验。vLLM 侧若未提供服务端阶段时延，TTFT 表示客户端发出请求到收到首个非空内容的时延，
  包含网络和服务排队，不代表纯模型 TTFT。将 `runtime.options.stream_responses` 设为 `false` 可保留非流式
  兼容路径；非流式响应只有在服务端显式返回时才记录 TTFT，不能从完整 HTTP RTT 推算。
  vLLM 云端与 Jetson Edge-LLM 之间的时延包含不同硬件和网络条件，不作为纯框架性能对照；
  如需隔离服务框架差异，应在同一服务器、同一模型与冻结 workload 上比较 Transformers 与 vLLM。

## 失败类别

稳定类别包括输入错误、依赖不可用、JSON 解析失败、模型拒答、超时、内存不足、
不支持算子和其他 runtime 错误。每个 `InferenceRecord` 恰好包含 assessment 或
failure 之一，不会用默认业务输出掩盖失败。
