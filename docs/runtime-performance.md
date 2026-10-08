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
| `attempted_sample_count` / `failed_sample_count` | 全部 analyze 尝试数 / 未得到有效 assessment 的记录数，包含输入错误、超时和 JSON 错误 |
| `attempted_requests_per_second` | 全部尝试数 ÷ 正式执行墙钟时间 |
| `successful_requests_per_second` | 有效 assessment 数 ÷ 正式执行墙钟时间；不表示业务判断正确率 |
| `all_request_stage_latency_ms` | 全部记录已有阶段测量的 p50/p90/p99，包含超时等失败的实测耗时 |
| `failed_request_stage_latency_ms` | 失败记录已有阶段测量的 p50/p90/p99，包含 JSON 校验失败 |
| `decode_tokens_per_second` | 有明确 decode token 数及对应 `decode_ms` 的记录：计数总和 ÷ 阶段时间总和 |

SSE 片段可能包含多个 token，也可能因解码缓冲而延迟，因此片段数量不是输出 token
数量。TPOT 使用 usage 中的 token 数；缺少 usage、只有一个 token 或只有一个文本
事件时保持 null。usage 可能包含不可见结束 token，客户端 TPOT 是估计，不能替代
服务端算子时间。角色、空文本、usage 和 `[DONE]` 事件不计入文本事件。

逐样本保存服务端 `prompt_tokens` 和 `completion_tokens`。`token_counts` 同时保存
usage 覆盖条数；不能把缺失 usage 的总数解释为真实零 token。墙钟窗口包含失败请求
和客户端执行开销，排除预热及后续环境采集、报告落盘。它适用于将来的并发汇总。
旧字段 `aggregate_output_tokens_per_end_to_end_second` 使用逐请求耗时之和，保留兼容，
不能当作并发负载下的服务吞吐。历史 `tokens_per_second` 使用全部输出 token
数归一化 `decode_ms`；它不保证剔除了 prefill 产出的首 token，不能当作精确
decode token 吞吐。缺少阶段计时仍保持 null；新研究使用明确计数的
`decode_tokens_per_second`，并保留历史字段。

`InferenceRecord.decode_tokens` 保存明确属于 decode 阶段的 token 数。
Transformers 插桩只在单序列语言模型 forward 次数与完整输出计数一致时
填写：第一次 forward 为 prefill，后续 forward 为 decode。单 token 输出
可记录 decode 数为 0，但没有 decode 时间时速率仍为 null。未插桩、模块
未找到、批量输出或调用次数不一致时保留未知。HTTP 可读取服务端明确提供的
`usage.decode_tokens` 扩展，不从 `completion_tokens` 自动推定阶段计数。
`token_counts.decode_usage_record_count` 与 `decode_rate_record_count` 分别记录
计数覆盖条数、同时具备计数和时间的条数；部分覆盖速率只代表这些样本。

2026-10-08 RTX 4060 Laptop 单图 Transformers BF16 诊断得到 115 个输出
token、115 次语言模型 forward，decode 计数为 114，阶段总时间为 4221.35 ms，
对应 27.006 decode tokens/s。这是带 CUDA 同步 hook 的计数验证，不是未插桩
性能基线或 Jetson 性能结论。第一次 `device_map` 加载因缺少 `accelerate`
失败；第二次在 CPU 加载后显式移到 GPU，保持现有 vLLM 环境依赖锁定。
原始输出、逐次 forward 时间、代码哈希和两次日志保存在
`reports/vllm-wsl/decode-count-20261008/`，失败日志同样保留。

`stage_latency_ms` 保持完整返回模型输出的口径，包括 JSON 校验失败但有
`raw_output` 的记录；传输失败通常只有端到端耗时，另见全请求和失败请求
分位数。每个阶段的 `count` 是实际测量条数，缺失阶段不填 0。两个总体
允许重叠：JSON 错误既是完整后端执行，也是失败 assessment。
资源峰值和请求平均功耗汇总保留所有记录中的已知测量，包括失败；缺失
资源不补估。请求平均功耗的算术平均不等于设备连续采样的时间平均功耗。

2026-10-08 离线聚合验证使用此前保存的 `timeout.json`、
`service_unavailable.json` 和 `ps20_pilot.json`，分别生成新口径汇总，
没有合并为同一 study 或重新运行模型。结果与源文件 SHA 保存在
`reports/vllm-wsl/failure-metrics-20261008/recomputed_metrics.json`，
原始报告不回写。未记录的墙钟窗口、input usage 和 stream arrival 保持缺失。

## TensorRT 低层 benchmark 指标

Edge-LLM HTTP Adapter 默认 `image_preprocessing="source"`，保留已有实验的原图路径。
新板端配置 `configs/studies/jetson_edgellm_fp16_workload_resize_ps20_pilot.json`
显式使用 `workload_resize`：客户端按冻结尺寸生成 RGB bicubic PNG，直到 HTTP
响应结束才删除临时目录。图片解码、缩放、编码和临时文件准备纳入
`preprocess_ms`、客户端总 TTFT 及调用方端到端计时；HTTP TTFT 和服务端阶段时间
保持原来的边界。请求失败、PNG 写入失败也清理文件并丢弃持久连接。

路径协议要求客户端与 server 能访问同一文件系统及临时目录；这份配置用于
Jetson 同机运行。SSH 端口转发不提供共享文件系统，不能直接把 WSL 的临时路径
传给 Jetson。2026-10-08 离线核验 20 张 PS20 图片，实际 Pillow 生成的 PNG
与 vLLM `workload_resize` 上传图片逐像素一致，原图不变且临时目录全部清理。
原始结果见 `reports/jetson-readiness-20261008/image_preprocessing_verification.json`；
HTTP 调用被 mock，没有执行板端推理，也未验证 TensorRT 内部 patch/tensor 布局。

`scripts/run_edgellm_benchmark.py` 将服务端明确返回的 decode token 数写入
JSONL 的可选 `decode_tokens`。汇总只配对同一条 completed 记录中的 token 数
和耗时；一条记录只有 token、另一条只有时间时，不计算两者之间的速率。
总体与每个 repetition 都保存参与速率计算的 `*_sample_count`。

低层 JSONL 的历史 `decode_tokens_per_second` 仍表示完整 output token 数除以
decode 时间；新增 `measured_decode_tokens_per_second` 使用明确的 decode 计数。
这与 StudyReport 的严格 `decode_tokens_per_second` 口径对应。未提供计数的旧
JSONL 和直接 C++ benchmark 不自动推定 `N-1`，新字段保持 null。部分字段缺失时，
速率仅覆盖有完整配对的请求，不能解释为整个 workload 的吞吐。

2026-10-08 板端 server 源码检查发现流式异常会返回
`finish_reason="error"` 后继续发送 `[DONE]`。HTTP Adapter 将这种响应记为
运行时失败，即使已经收到符合 schema 的文本也不计为成功。当前 SSE 路径未返回
usage 或阶段计时；非流式 `prompt_tokens=0` 是 server 占位值，不能当作真实输入
token 数。补齐服务端测量之前，仅已有客户端 TTFT、文本片段到达和端到端时延
具备可采集路径，阶段速率不补估。

### Edge-LLM 可选流式 usage

新增 server 启动选项 `--stream-usage` 在进程内包装已核验的
`experimental.server.api_server._generate_stream_sse`。每个请求独立汇总
`StreamDelta.token_ids`，在 `[DONE]` 前发送空 choices 的 usage 事件；它包括
未显示为文本的特殊 token，不是字符数或 SSE 事件数。启动时显式开启后，该进程的
流式响应均可附带 usage，不依赖请求的 `stream_options`。默认关闭，现有板端源码
和生成参数不改写；已经包含 usage 的响应原样保留。

缺少有效 token IDs、没有观察到任何 delta 或 error 结束时，不补估 token 数。
扩展不返回占位 prompt 计数，也不从 completion 数推定 decode 计数。初始化时
检查 generator 签名，runtime metadata 记录开关、包装代码 SHA-256 和计数边界。
普通与工具 SSE 分支共用的 `generate_stream` 调用均可经过包装；当前已验证普通
单图 SSE 的离线集成和板端 native GPU 诊断，尚未验收 HTTP 传输或工具调用。

2026-10-08 使用采集的 Jetson SSE 函数、模拟 native delta 和真实客户端解析器，
验证两个文本事件可包含四个 token IDs，得到 completion_tokens=4，input/decode
计数和 decode 时间保持 null。证据见
`reports/jetson-readiness-20261008/stream_usage_verification.json` 和源码快照；
该离线结果不作为 GPU 吞吐证据。

### Edge-LLM native Timer 桥接准备

已核验的 C++ Timer 记录 CUDA event 时间，现有 Python binding 尚未暴露其数据。
`scripts/prepare_edgellm_timing_binding.py` 生成绑定源码身份的补丁，准备导出每个
原始 stage ID 的 `total_gpu_time_ms` 和 `run_count`。这些值是累计值，不是逐请求
报告；必须在 worker 全部停止执行时读取，且开启实际 profiling 才可能有数据。
在独占、串行验收中核对快照、stage 名称及包围范围后，才可映射为报告阶段字段。

2026-10-08 已对采集的 Jetson CRLF 源码验证补丁可应用并保持其他内容与行尾；
随后已完成独立 binding 编译和新进程导入；默认 profiling 为 false，初始快照为空。
初次导入未执行模型；后续单图 CUDA event 验收结果如下。SSE 阶段回传尚未接入，
现有研究报告中的阶段指标仍为 null。
补丁、准备记录和验证结果保存在
`reports/jetson-readiness-20261008/native_stage_timing_binding.patch` 及相关 JSON。

板端单图诊断入口为 `scripts/run_edgellm_native_probe.py`，默认不执行模型；
显式执行时保存候选绑定、engine、workload 和输入身份，以及初始化失败事实。
其流式 TTFT 不包含 HTTP 传输，不能填写 HTTP TTFT 字段。FP16 本轮在视觉 engine
加载阶段 OOM，没有生成结果；不补填 token 数或阶段时间。已有 INT4 引擎可完成
初始化；upstream metrics 的继承成员绑定错误已在独立候选绑定中修复。
各次失败与成功初始化的边界见 [Jetson 就绪记录](jetson-readiness.md)。

2026-10-09 已有 INT4 产物完成一次带 profiling 的 native 单图诊断：SSE usage 与
native generated 计数均为 95，computed/reused prompt 为 570/0；vision 和
prefill 各 1 次，分别 203.98/425.85 ms，`llm_generation` 94 次共 12001.78 ms。
已核验的 vanilla decoder 每次 decode step 包含一个该 stage；prefill 产生首个
token，因此本次 94 次 decode 与后续 token 数吻合，对应 7.832 decode tokens/s。
这条诊断不推广到 batch、speculative decoding、并发请求或其他 engine。
首文本延迟 659.54 ms、流总耗时 12.670 秒仅为进程内 SSE 路径，不含 HTTP。
严格 JSON 有效不代表风险判断正确；本阶段保留输出，不调整 prompt 或模型。
完整记录见 `reports/jetson-readiness-20261008/native_int4_diagnostic.json`。

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

## 2026-10-07 关闭额外 GPU 程序后复测

用户确认上一轮测试期间另有 GPU 程序运行，本轮在该程序关闭后重新执行。
复用同一组配置、模型、输入和提交顺序；eager、BF16、`max-num-seqs=1`。
每档预热 20 次、正式执行 60 次，三档共 180 次全部成功。每档输入 / 输出
token 总数均为 34200 / 4977，服务端完成数与 token 数均匹配客户端。
客户端源码、运行时选项、workload 和数据元信息哈希与旧轮一致，原始输出也一致。

| 客户端并发 | 旧轮 / 本轮吞吐（tokens/s） | 本轮 HTTP TTFT p50 / p90 / p99（秒） | 本轮 E2E p50 / p90 / p99（秒） |
| --- | --- | --- | --- |
| 1 | 36.212 / 45.757 | 0.035 / 0.041 / 0.047 | 1.758 / 2.138 / 3.256 |
| 2 | 21.011 / 42.081 | 1.734 / 2.665 / 2.985 | 3.587 / 5.180 / 6.054 |
| 4 | 22.389 / 42.165 | 5.333 / 7.472 / 8.678 | 7.219 / 9.903 / 11.202 |

本轮服务端平均排队时间分别为 0.061 ms、1.836 s、5.615 s；
客户端最大 Runtime 重叠数分别为 1/2/4。并发 2/4 吞吐较旧轮分别约为
2.00 / 1.88 倍，旧轮的受干扰结果不能作为框架固有性能结论。
本轮并发 2/4 吞吐接近，但较单请求基线没有提高；单活跃序列限制下仍有明显排队。
后续应提高服务端序列上限，另行测试实际多序列批处理。

启动前设备显存占用 930 MiB、GPU 利用率 16%、温度 58 °C；CUDA 计算进程
列表为空，不能证明所有 Windows 图形消费者均已隔离。设备显存采样峰值三档均为
7612 MiB，温度峰值分别为 87 / 75 / 75 °C，平均功耗约 60.11 / 46.14 / 45.60 W。
笔记本功耗与温度仍未固定，不能把新旧差异全部量化归因于单一程序。
新旧日志的 KV cache 容量均为 20880 tokens。

新证据独立保存在本地 `reports/vllm-wsl/retry-20261007/`：三档完整报告及各自
预热、执行身份、GPU 遥测和 `/metrics` sidecar；`concurrency_seq1_comparison.json`
与 `.csv` 保存本轮汇总，`previous_vs_retry.json` 保存新旧对照，
`concurrency_server_profile.json` 和 `server.log` 保存环境与启动记录。
旧证据保持原样；此目录被 Git 忽略，迁移或分享时需单独保留。

## 2026-10-08 服务端多序列批处理

建立当日单序列并发 2/4 匹配基线，再提高 `max-num-seqs` 到 2 和 4。
使用相同模型 revision、BF16、eager、2048 上下文、1024 调度 token 预算和
85% 显存预算；客户端图片、prompt、生成参数、提交顺序保持一致。
每档先串行预热 20 次，再执行 60 次正式请求。四档均 60/60 成功。
日志确认候选实际出现 2/4 个运行序列，KV cache 容量均为 20880 tokens。

| 服务端序列上限 / 客户端并发 | 输出吞吐（tokens/s） | 请求吞吐（requests/s） | HTTP TTFT p50 / p90 / p99（秒） | E2E p90（秒） |
| --- | --- | --- | --- | --- |
| 1 / 2 | 42.381 | 0.511 | 1.815 / 2.525 / 3.258 | 5.175 |
| 2 / 2 | 68.595 | 0.839 | 0.056 / 0.067 / 0.465 | 3.086 |
| 1 / 4 | 39.356 | 0.474 | 5.716 / 8.189 / 9.202 | 10.994 |
| 4 / 4 | 127.216 | 1.556 | 0.067 / 0.085 / 0.107 | 3.305 |

双序列吞吐为匹配基线的 1.62 倍，四序列为 3.23 倍。四序列独立完成
100 次并发复验：100/100 成功、134.605 tokens/s、1.661 requests/s，
HTTP TTFT p50 / p90 / p99 为 63.75 / 73.52 / 88.73 ms，
E2E p50 / p90 / p99 为 2.227 / 3.008 / 3.378 秒。
没有 OOM、超时或 JSON 校验失败。此复验仍只代表本机冻结 pilot 工作负载。

批处理输出不是逐字等价：双序列有 10/60 条 assessment 变化，涉及证据文案
10 条、驾驶建议 4 条，风险等级与事件集合一致；四序列有 12/60 条变化，
涉及证据文案 11 条、驾驶建议 6 条、事件集合 2 条，风险等级一致。
单序列输出 token 总数为 4977，双/四序列分别为 4903 / 4906；吞吐使用实际
usage 计数，同时报告 requests/s。没有为了等长输出修改冻结生成参数。
本阶段不调模型或标注；全部差异保留为运行时切换的实际证据，不将候选称为
输出完全等价的替代方案。

## 多序列候选的使用

四序列 eager 是本机吞吐候选，双序列适合先验证两个同时请求的负载。
默认单序列脚本保持现有行为。分别启动：

```bash
bash scripts/serve_vllm_wsl_batch.sh 2
# 或在前一服务退出后启动四序列
bash scripts/serve_vllm_wsl_batch.sh 4
```

另一终端执行对应负载，`seq2/concurrency2` 或 `seq4/concurrency4` 配对：

```bash
.venv-vllm/bin/python scripts/benchmark_vllm_http.py \
  --config configs/studies/wsl_vllm_batch_seq4_concurrency4_ps20_repeat3.json \
  --concurrency 4 --gpu-telemetry \
  --metrics-url http://127.0.0.1:8000/metrics
```

已有证据时必须通过 `--output` 指定新文件。100 次复验配置为
`configs/studies/wsl_vllm_batch_seq4_concurrency4_ps20_soak100.json`。
当前 decode Graph 脚本只捕获 batch=1，尚不能当作多序列 Graph 优化证据；
输出重复性和多序列 Graph 是后续独立研究项。

本轮启动前 GPU 利用率 0%、显存占用 735 MiB、温度 57 °C；四档设备显存
采样峰值均为 7654 MiB，功耗和温度未固定。测量包含热 prefix/视觉缓存和
重复请求，不作为冷缓存或其他硬件的通用加速比。
原始证据位于本地 `reports/vllm-wsl/batch-20261008/`：`seq1_c2.json`、
`seq1_c4.json`、`seq2_c2.json`、`seq4_c4.json`、`seq4_c4_soak100.json` 及
各自全部 sidecar；`batch_comparison.json`、`.csv`、
`seq2_output_differences.json`、`seq4_output_differences.json`、
`server_profiles.json` 和三份服务端日志。该目录被 Git 忽略，需单独保存。

## 批次不变模式验证

`scripts/serve_vllm_wsl_invariant.sh` 是独立实验入口：设置
`VLLM_BATCH_INVARIANT=1`、`VLLM_ATTENTION_BACKEND=FLASH_ATTN`，沿用四序列
eager、固定版本和冻结工作负载。默认启动入口不启用此模式。
[vLLM 官方说明](https://docs.vllm.ai/en/latest/features/batch_invariance/)
将该功能标为实验性；本仓库以固定 0.11.2 在本机的实际结果判断适用范围。

```bash
bash scripts/serve_vllm_wsl_invariant.sh
# 另一终端；分别用 concurrency1/2/4 配置及 --concurrency 1/2/4
.venv-vllm/bin/python scripts/benchmark_vllm_http.py \
  --config configs/studies/wsl_vllm_invariant_seq4_concurrency4_ps20_repeat3.json \
  --concurrency 4 --gpu-telemetry \
  --metrics-url http://127.0.0.1:8000/metrics
```

同一服务分别执行并发 1/2/4，每档串行预热 20 次、正式请求 60 次；比较
逐条 raw output、assessment 各字段和同一样本三次重复输出。原始报告与
服务端日志保存在 `reports/vllm-wsl/invariant-20261008/`，汇总脚本为
该目录下的 `summarize_invariance.py`。不修改 prompt、生成参数或样本标注。

2026-10-08 实测三档均 60/60 成功；服务端请求/token 计数与客户端一致，
抢占次数为零。所有 TTFT 均为 HTTP 首个非空文本片段到达时间。

| 客户端并发 | 输出吞吐（tokens/s） | 请求吞吐（requests/s） | TTFT p50（ms） | E2E p90（秒） | 相对本模式串行的输出变化 / 事件变化 |
| --- | --- | --- | --- | --- | --- |
| 1 | 27.313 | 0.339 | 48.21 | 3.679 | 0 / 0 |
| 2 | 40.407 | 0.501 | 94.06 | 5.223 | 15/60 / 3/60 |
| 4 | 76.272 | 0.938 | 96.50 | 5.102 | 15/60 / 5/60 |

串行中每张图片三次输出完全一致，双/四并发各有 5/20 张图片在重复运行时
出现不同 raw output。风险等级保持一致；证据和驾驶建议存在变化。
相对此前普通四序列批处理，模式切换本身有 20/60 条输出变化，包括 6 条
事件集合变化。完整字段差异与原文保存在 `invariance_comparison.json`。
显存采样峰值为 7697 MiB，温度峰值 76 °C，无 OOM 或超时。

**结论：固定 vLLM 0.11.2、当前 Qwen3-VL 和 RTX 4060 Laptop 的实测未达到
跨批次输出不变。此模式保持实验状态，不作为默认或确定性配置。** 普通
四序列候选此前为 127.216 tokens/s，本模式为 76.272 tokens/s；功耗与
热状态未受控、输出长度也不同，因此这些数值只是分别实测的结果。

## 2026-10-08 四序列 decode CUDA Graph

使用同一冻结配置完成 eager → Graph → eager 顺序复验，三档分别串行预热
20 次、四并发正式请求 60 次。服务重启后重新预热。模型 revision、BF16、
图片、prompt、生成参数、客户端源码哈希、工作负载与数据集哈希均一致。
仅调整 Graph 配置，服务端 `max-num-seqs=4` 保持一致。

`scripts/serve_vllm_wsl_batch_graph.sh` 设置 `compilation_config.mode=0`，
`cudagraph_mode=FULL_DECODE_ONLY`、捕获尺寸 `[1,2,4]`，prefill 保持 eager。
日志确认三个 decode Graph 捕获完成，新增约 0.06 GiB；KV cache 仍为
20880 tokens。出现 cascade attention 时服务可回退 eager，不据此声称
所有 GPU 操作均使用 Graph。

| 配置 | 输出吞吐（tokens/s） | 请求吞吐（requests/s） | HTTP TTFT p50（ms） | E2E p90（秒） | 设备显存采样峰值（MiB） |
| --- | --- | --- | --- | --- | --- |
| eager 前测 | 131.002 | 1.606 | 61.39 | 3.150 | 7706 |
| decode Graph | 178.105 | 2.203 | 57.82 | 2.245 | 7764 |
| eager 复验 | 103.699 | 1.278 | 69.30 | 4.611 | 7735 |

三档均 60/60 成功、零抢占，服务端请求和 output token 增量与客户端一致。
Graph 吞吐分别为两个 eager 实测值的 1.36 / 1.72 倍。两个 eager 结果本身
存在波动，功耗与温度没有固定，且输出 token 总数不同；不作为通用加速比。
Graph 与 eager 前测有 8/60 条 assessment 变化，涉及证据 7 条、建议 2 条、
事件集合 1 条，风险等级一致。两次 eager 有 7/60 条文案变化、事件集合一致。
因此 Graph 保持性能候选，未作为输出等价替代方案或默认启动配置。

启动候选并在另一终端测量：

```bash
bash scripts/serve_vllm_wsl_batch_graph.sh
# 另一终端；已有报告时用 --output 指定新文件
.venv-vllm/bin/python scripts/benchmark_vllm_http.py \
  --config configs/studies/wsl_vllm_batch_graph_graph_ps20_repeat3.json \
  --concurrency 4 --gpu-telemetry \
  --metrics-url http://127.0.0.1:8000/metrics
```

原始证据保存在 `reports/vllm-wsl/batch-graph-20261008/`：三档报告及全部
sidecar、独立服务日志、`server_profiles.json`、`summarize_graph.py` 和
`graph_comparison.json`（包含完整输出差异）。此目录被 Git 忽略，需单独保存。

同配置重启后完成独立 100 次四并发复验，使用
`configs/studies/wsl_vllm_batch_graph_soak100.json`：100/100 成功、
189.144 tokens/s、2.317 requests/s，HTTP TTFT p50 / p90 / p99 为
53.77 / 68.47 / 109.05 ms，E2E p50 / p90 / p99 为
1.578 / 2.215 / 2.362 秒。零抢占、无 OOM 或超时，服务端计数与客户端一致。
设备显存采样峰值 7807 MiB、温度峰值 75 °C；仍为热缓存、重复 PS20 请求，
不代表长时间稳定性或冷缓存负载。报告 `graph_soak100.json` 及 sidecar、
启动日志和脚本哈希与上述 A/B/A 证据一同保存。服务退出后 GPU 利用率 0%，
设备显存 848 MiB；该数值含系统及其他图形组件，不能当作模型显存。

## 缓存口径与实际服务配置

之前的重复 PS20 测量先预热同一组图片，代表热 prefix/多模态缓存负载。
独立输入的延迟与吞吐应使用独立口径，不把预热重复请求当作新图片性能。
`scripts/serve_vllm_wsl_no_reuse.sh` 同时设置
`--no-enable-prefix-caching --mm-processor-cache-gb 0`，固定四序列，支持
`eager` 和 `graph`。固定 vLLM 0.11.2 的 `v1/engine/processor.py` 在两项
缓存都关闭时使用请求独立的多模态标识，避免跨请求复用视觉特征。
这不等于模型冷启动：模型、CUDA kernel 和连接可预热，图片和 prompt 仍冻结。
默认启动配置继续启用缓存。

该实验入口在 localhost 启用 `VLLM_SERVER_DEV_MODE=1`，以读取
`/server_info?config_format=json`。benchmark 新增可选 `--server-info-url`：
保存模型 revision、精度、缓存、调度、并行与 Graph 配置的前后快照，
不保存 HF token 等未选择字段；同时记录 benchmark 脚本哈希。
快照不一致时 execution 标为失败、命令返回 2。已有 sidecar 时拒绝覆盖。
未提供此参数或字段缺失时，不推定服务端采用默认值。

```bash
bash scripts/serve_vllm_wsl_no_reuse.sh graph
# 另一终端；eager 前/后测使用对应 eager 配置与启动参数
.venv-vllm/bin/python scripts/benchmark_vllm_http.py \
  --config configs/studies/wsl_vllm_no_reuse_graph_ps20_repeat3.json \
  --concurrency 4 --gpu-telemetry \
  --metrics-url http://127.0.0.1:8000/metrics \
  --server-info-url 'http://127.0.0.1:8000/server_info?config_format=json'
```

实际缓存关闭状态通过前后 `server_config` 快照与服务端计数核对。仅关闭
prefix cache 或只清空一次缓存不足以说明重复请求没有跨请求复用。

2026-10-08 同日无跨请求复用 A/B/A：三档均串行预热 20 次、四并发正式
请求 60 次，全部成功；实际配置前后相同，prefix/MM cache 查询和命中
增量均为 0，零抢占，服务端请求/token 计数与客户端一致。

| 配置 | 输出吞吐（tokens/s） | 请求吞吐（requests/s） | HTTP TTFT p50（ms） | E2E p90（秒） |
| --- | --- | --- | --- | --- |
| eager 前测 | 133.140 | 1.656 | 172.57 | 2.930 |
| decode Graph | 150.128 | 1.853 | 165.95 | 2.705 |
| eager 复验 | 81.709 | 1.010 | 191.74 | 5.632 |

Graph 与 eager 前测有 17/60 条 assessment 变化，事件集合变化 5 条，
风险等级一致；两次 eager 同样有 17/60 条变化，事件集合变化 5 条。
Graph 与 eager 复验有 12/60 条变化，其中事件集合 2 条。三档设备显存
采样峰值分别为 7891 / 7861 / 7754 MiB，平均功耗为
48.72 / 53.30 / 46.66 W。eager 前后明显波动，无法将差异全部归因于
Graph；本轮不改默认配置，也不作为跨批次确定性证据。缓存关闭后的
TTFT 与之前热缓存结果分别报告，不把两轮测量视为受控缓存消融实验。

原始证据位于 `reports/vllm-wsl/no-reuse-20261008/`：三档报告、全部
sidecar、配置前后快照、日志、`server_profiles.json`、
`summarize_no_reuse.py`、`no_reuse_comparison.json` 和 `.csv`。
每次正式测量的 benchmark 脚本哈希已保存；报告目录被 Git 忽略。

本轮测量完成后，benchmark 增加有效性门槛：显式采集服务端 metrics 时，
计数器重置、请求数或 token 数与客户端不匹配会使 execution 标为失败并
返回 2，即使推理本身成功。后续 GPU CSV 同时采集 SM/显存时钟与功耗、
温度及降频原因；字段已在本机驱动核对。旧 CSV 不补填新字段，采样仍为
设备级每秒一次；未受控功耗、系统负载和热状态不能由这些字段自动消除。
