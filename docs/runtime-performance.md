# 运行时性能测量

当前阶段固定模型 revision、输入图片、workload、精度与输出预算，只研究 vLLM
和 TensorRT Edge-LLM 的运行时行为。模型正确性结果仍保存，但不在本阶段调参。

## 客户端指标

| 报告字段 | 口径 |
| --- | --- |
| `time_to_first_token_ms` | 从 HTTP 请求开始，到首个非空 SSE 文本片段；客户端可见 TTFT |
| `client_total_ttft_ms` | 在上述时间上加客户端请求构造时间，包含图片缩放、编码和 JSON 序列化 |
| `client_tpot_ms` | （最后文本片段时间 − 首片段时间）÷（服务端 completion_tokens − 1）；客户端平均 TPOT 估计 |
| `stream_timings.content_arrival_ms` | 每个非空 SSE 文本事件相对 HTTP 请求开始的到达时间 |
| `stream_chunk_interval_ms` | 全部相邻文本事件到达间隔的 p50/p90/p99；不称为逐 token ITL |
| `end_to_end_ms` | 单次 Runtime analyze 的全部耗时，包含预处理、HTTP、生成和结果校验 |
| `completed_requests_per_second` | 完整返回模型输出的请求数 ÷ 正式执行墙钟时间 |
| `wall_output_tokens_per_second` | 完整返回的输出 token 总数 ÷ 正式执行墙钟时间 |

SSE 片段可能包含多个 token，也可能因解码缓冲而延迟，因此片段数量不是输出 token
数量。TPOT 使用 usage 中的 token 数；缺少 usage、只有一个 token 或只有一个文本
事件时保持 null。usage 可能包含不可见结束 token，客户端 TPOT 是估计，不能替代
服务端算子时间。角色、空文本、usage 和 `[DONE]` 事件不计入文本事件。

逐样本保存服务端 `prompt_tokens` 和 `completion_tokens`。`token_counts` 同时保存
usage 覆盖条数；不能把缺失 usage 的总数解释为真实零 token。墙钟窗口包含失败请求
和客户端执行开销，排除预热及后续环境采集、报告落盘。它适用于将来的并发汇总。
旧字段 `aggregate_output_tokens_per_end_to_end_second` 使用逐请求耗时之和，保留兼容，
不能当作并发负载下的服务吞吐。`tokens_per_second` 没有服务端 decode 计时时保持 null。

## vLLM 服务端指标

在本机服务已启动、没有其他请求流量时运行：

```bash
.venv-vllm/bin/python scripts/benchmark_vllm_http.py \
  --config configs/studies/wsl_vllm_workload_resize_ps20.json \
  --output reports/vllm-wsl/新的运行名.json \
  --gpu-telemetry \
  --metrics-url http://127.0.0.1:8000/metrics
```

benchmark 独立完成预热，在正式运行前后保存 `.metrics.before.prom` 和
`.metrics.after.prom`。`.server_metrics.json` 对累积计数和直方图作窗口差分，
提供服务端 TTFT、ITL、逐请求 TPOT、排队、prefill、decode 和 E2E 的观察数量及
均值，并保留全部桶差分。缺失指标保持 null；计数回退将窗口标记为无效。
这里没有生成精确服务端 p50/p90/p99，直方图只能支持桶分辨率下的估计。

服务端统计是全局窗口，要求服务没有其他请求、窗口内不重启，并检查成功请求计数
与正式实验是否一致。服务端窗口吞吐包含 HTTP 之外的报告及采集开销，与客户端
执行窗口分别报告。均值不是单张图片的精确阶段记录；不回填逐请求 `decode_ms`。
GPU 遥测仍为设备级每秒采样，不能捕获全部瞬时峰值。

## 并发测试含义

并发数表示客户端同时未完成的请求数量。并发 1 逐个提交，并发 2 同时维护最多
两个请求，并发 4 同时维护最多四个请求；一个请求结束后再补下一个。
这与“将同一图片重复执行四次”不同，也不等于启动四个模型进程。

并发实验需要独立 HTTP 连接，不能让多个线程共用一个有状态的 backend。
客户端并发数和服务端 `max-num-seqs` 分别记录：当前服务为 `max-num-seqs=1`，
更高客户端并发主要测试排队；批处理研究需另建服务配置，核对 8 GB 显存余量。
比较 requests/s、tokens/s、TTFT 与 E2E 的尾延迟、失败率，不能只看总吞吐。
benchmark CLI 使用 `--concurrency 1`、`2` 或 `4` 选择客户端并发数。
每个 worker 独立构造 runtime，并复用自己的连接；执行结束后关闭连接。
预热始终以并发 1 执行，排除在正式计时之外。报告按提交顺序保存记录，
正式执行窗口包含线程调度、连接建立和全部请求结束的时间。
`first_request_ms` 是报告中第一条提交记录的耗时，不是服务冷启动。

```bash
.venv-vllm/bin/python scripts/benchmark_vllm_http.py \
  --config configs/studies/wsl_vllm_eager_seq1_concurrency2_ps20_repeat3.json \
  --concurrency 2 --gpu-telemetry \
  --metrics-url http://127.0.0.1:8000/metrics

.venv-vllm/bin/python scripts/benchmark_vllm_http.py \
  --config configs/studies/wsl_vllm_eager_seq1_concurrency4_ps20_repeat3.json \
  --concurrency 4 --gpu-telemetry \
  --metrics-url http://127.0.0.1:8000/metrics
```

三档配置均为 PS20、每张重复三次、相同任务提交顺序，共 60 次正式请求。
命令按档位顺序运行，避免不同测试的流量混入服务端统计窗口。
已有证据时使用 `--output` 指定新文件名，默认拒绝覆盖。

## 2026-10-05 本机验收

RTX 4060 Laptop、vLLM 0.11.2、BF16、eager、`workload_resize`、单序列，
20 次预热后测量 PS20 的 20 次请求，全部完成。

| 指标 | 本轮结果 |
| --- | --- |
| 客户端 HTTP TTFT p50 / p90 / p99 | 33.04 / 35.52 / 35.65 ms |
| 包含请求构造的 TTFT p50 / p90 / p99 | 90.81 / 106.07 / 107.04 ms |
| 客户端 TPOT 估计 p50 / p90 / p99 | 18.26 / 18.88 / 19.38 ms |
| SSE 文本片段间隔 p50 / p90 / p99 | 18.21 / 20.42 / 24.49 ms，1619 个间隔 |
| 客户端 E2E p50 / p90 / p99 | 1.456 / 2.221 / 2.412 s |
| 客户端正式执行窗口 / 请求吞吐 / 输出吞吐 | 32.229 s / 0.621 requests/s / 51.476 tokens/s |
| 输入 / 输出 token 总数 | 11400 / 1659，全部 20 条均有 usage |
| 服务端 TTFT / ITL 均值 | 22.79 / 18.52 ms |
| 服务端逐请求 TPOT 均值 | 18.50 ms |
| 服务端排队 / prefill / decode 均值 | 0.059 / 19.44 / 1517.97 ms |

客户端 TPOT 分布按请求统计，服务端 ITL 均值按 token 间隔统计，两者加权口径不同。
服务端窗口差分的完成数为 20、输出 token 数为 1659，与客户端相符。
原始证据：本地 `reports/vllm-wsl/metrics_eager_ps20.json`、`.warmup.json`、
`.execution.json`、`.server_metrics.json`、`.metrics.before.prom`、
`.metrics.after.prom`、`.gpu.csv`，以及 `metrics_eager_server.log`。
这些文件被 Git 忽略，分享结果时需一并保留。

## 运行条件

预热、prefix cache、视觉缓存、实际输入/输出长度、功耗与温度都会影响测量。
当前 PS20 测量是在缓存开启的预热条件下完成，输出长度由冻结生成预算内的实际
生成结果决定；它不是等长输出或冷缓存吞吐实验。
vLLM 的 4060 结果与 Jetson 结果分别保存，跨设备性能差异不能直接归因于框架。

## 2026-10-05 并发 1/2/4 实测

使用同一 vLLM 服务进程、eager、BF16、`max-num-seqs=1`、`workload_resize`。
每档独立串行预热 20 次，再执行 PS20 每张三次，共 60 次正式请求。
三档均 60/60 成功，输入 token 总数均为 34200，输出 token 总数均为 4977。
逐样本原始输出和 assessment 一致；服务端窗口完成数、输出 token 数均与客户端匹配。

| 客户端并发 | 输出吞吐（tokens/s） | 请求吞吐（requests/s） | HTTP TTFT p50 / p90 / p99（秒） | E2E p50 / p90 / p99（秒） |
| --- | --- | --- | --- | --- |
| 1 | 36.212 | 0.437 | 0.044 / 0.050 / 0.060 | 2.053 / 3.311 / 4.106 |
| 2 | 21.011 | 0.253 | 2.751 / 5.822 / 14.333 | 5.893 / 12.900 / 25.197 |
| 4 | 22.389 | 0.270 | 9.709 / 15.155 / 21.210 | 13.217 / 19.510 / 26.573 |

服务端平均排队时间分别为 0.073 ms、3.735 s、10.637 s。
日志显示并发 2 时 1 个运行、1 个等待，并发 4 时 1 个运行、3 个等待；
由客户端开始时间和耗时重建的最大 Runtime 重叠数分别为 1/2/4。
该重建假定墙钟没有跳变，不等同于 GPU 同时执行序列数。

本轮未观察到吞吐收益，排队与尾延迟明显增长。当前配置限制单活跃序列，
不能据此判定 vLLM 多序列批处理的收益。生成速度也有波动：并发 1/2 的
设备平均功耗采样分别约 56.54 / 43.04 W，服务端 ITL 均值约 26.39 / 47.53 ms。
未控制笔记本功耗、温度和其他负载，不能把全部吞吐下降归因于客户端并发。
设备显存采样峰值分别为 6924 / 7875 / 7859 MiB；临时服务及其 engine
进程退出后，设备仍报告 1214 MiB 占用。设备遥测包含其他消费者，不能全部归因于
vLLM，也不能据此得出单个请求的显存峰值。
后续批处理研究需单独使用更高 `max-num-seqs`，并重新建立匹配基线。

本地证据保存于 `reports/vllm-wsl/`：

- `concurrency1_seq1_ps20.json`、`concurrency2_seq1_ps20.json`、`concurrency4_seq1_ps20.json`。
- 每档对应的 `.warmup.json`、`.execution.json`、`.gpu.csv`、
  `.metrics.before.prom`、`.metrics.after.prom`、`.server_metrics.json`。
- `concurrency_seq1_comparison.json`、`concurrency_seq1_comparison.csv`。
- `concurrency_server_profile.json`、`concurrency_server.log`。

汇总核对客户端源码、运行时选项、workload、manifest 和 annotation 的身份哈希；
原始报告保持独立，不覆盖前轮证据。该目录被 Git 忽略，迁移时需另行保留。
