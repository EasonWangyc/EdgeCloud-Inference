# TensorRT Edge-LLM 优化实验

快速查看近期已核验结果： [TensorRT 优化阶段速览](tensorrt-optimization-summary.md)。本文保留
完整实验矩阵、命令、候选分支和证据边界，作为详细实验记录。

本页定义 Jetson Orin Nano 上 TensorRT Edge-LLM 的可复现实验入口。实验固定TensorRT Edge-LLM v0.9.1、Qwen3-VL-2B、15W、batch 1、`i768/k1024` 和现有INT4 AWQ LLM engine；量化、LoRA、标注和 workload 不在本轮中作为变量。

## 1. 实验矩阵

矩阵配置位于[`configs/tensorrt/jetson_orin_nano_int4_v1.json`](../configs/tensorrt/jetson_orin_nano_int4_v1.json)。

`builder_variants()` 生成第一阶段的 level `0/1/2/3` 与固定 workspace `1024 MiB`组合。后续阶段按单变量顺序选择最佳候选：workspace、CUDA Graph、profile，再单独测试 FP16 weight-streaming budget。

不要把不同模型 revision、visual engine、功耗模式或输入上限的结果合并成一个 A/B。每个候选 engine 使用新的目录、StudyReport 和 provenance 文件，level 0 正式 engine不被覆盖。

截至 2026-09-06，INT4 W4A16 `CTA_N=256` 候选已在 Jetson Orin Nano 上完成独立
plugin 编译，并通过候选 plugin 注入后的 engine 加载、CUDA Graph capture 和一次真实
多模态请求 smoke；这不是性能结论。候选仍需与正式 level0 plugin 在相同 20 样本、3 次
重复下比较 TTFT、prefill、decode、E2E、峰值内存和严格 JSON，再决定是否进入后续矩阵。

## 2. Engine provenance

构建完成后，在 Jetson 上准备一个 metadata JSON，至少包含：

```json
{
  "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
  "edge_llm_revision": "7f061f21f0a581ba234a1e233c9315b89d8e47d6",
  "tensorrt_version": "10.3",
  "cuda_version": "12.6",
  "gpu_name": "Jetson Orin Nano",
  "power_mode": "15W_MODE_0",
  "precision": "int4_awq",
  "max_batch_size": 1,
  "max_input_len": 768,
  "max_kv_cache_capacity": 1024,
  "builder_optimization_level": 1,
  "workspace_limit_mib": 1024,
  "cuda_graph": "enabled",
  "weight_streaming": "disabled",
  "weight_streaming_budget_bytes": null
}
```

然后记录 engine 文件大小和 SHA-256：

```bash
PYTHONPATH=src python3 scripts/record_engine_provenance.py \
  --engine artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024/llm/llm.engine \
  --metadata-json reports/tensorrt/opt1.metadata.json \
  --output reports/tensorrt/opt1.engine.provenance.json
```

构建命令可额外传入 `--build-report reports/tensorrt/opt1.build.json`。该报告按
LLM/visual 组件分别记录开始时间、耗时、返回码、实际 Edge-LLM revision、builder
环境变量、命令和最终 engine 的 size/SHA-256；构建耗时只用于评估 builder/timing-cache
成本，不能与推理 latency 或 tokens/s 混合。

TensorRT 的 `ITimingCache` 可复用已经测过的 layer tactic timing，降低重复构建的搜索
时间。它不是推理期 cache，也不改变已序列化 engine 的运行时 latency；因此 cache 命中
前后的构建耗时必须单独记录，不能写成推理加速。仓库的
`0012-configurable-timing-cache.patch` 使用 `EDGELLM_TIMING_CACHE_PATH` 加载并在构建
完成后回写 cache，默认以 `setTimingCache(..., false)` 做严格设备校验，不接受跨 GPU
cache 的静默复用。构建 wrapper 还要求提供与 cache 配套的 sidecar，例如
`configs/tensorrt/timing_cache_binding_jetson_orin_nano_llm_level2.json`：

```json
{
  "schema_version": "parksight_tensorrt_timing_cache_binding_v1",
  "device": {
    "gpu_name": "Jetson Orin Nano",
    "cuda_version": "12.6",
    "tensorrt_version": "10.3.0"
  },
  "builder_config": {
    "component": "llm",
    "max_batch_size": 1,
    "max_input_len": 768,
    "max_kv_cache_capacity": 1024,
    "workspace_limit_mib": 1024,
    "builder_optimization_level": 2
  }
}
```

sidecar 的设备字段和每个 BuilderConfig 字段会在 `llm_build` 启动前校验；因此 cache
首次生成、尚不存在时仍可使用，但不能省略绑定或复用不匹配的 level/workspace/profile。
构建报告会保存 sidecar 的大小、SHA-256 和绑定内容。示例：

```bash
git -C /home/ubuntu/TensorRT-Edge-LLM apply \
  /home/ubuntu/JetsonVLM/patches/tensorrt-edge-llm/0012-configurable-timing-cache.patch
PYTHONPATH=src python3 scripts/build_edgellm_vlm_engines.py \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --expected-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
  --onnx-root artifacts/onnx/qwen3_vl_2b_int4_awq_n128 \
  --engine-root artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024_opt2_profile \
  --component llm --max-batch-size 1 --max-input-len 768 \
  --max-kv-cache-capacity 1024 --workspace-limit-mib 1024 \
  --builder-optimization-level 2 --profiling-verbosity detailed \
  --timing-cache reports/tensorrt/jetson_orin_nano_trt103.cache \
  --timing-cache-binding reports/tensorrt/jetson_orin_nano_trt103.binding.json \
  --build-report reports/tensorrt/opt2.build.json
```

只有在同一 Jetson GPU、CUDA/TensorRT 版本和匹配的 BuilderConfig 下复用 cache；每个
candidate 仍需保留独立 engine、provenance、build report 和 inference benchmark。

构建脚本会先清除父 shell 中残留的 workspace、builder level、profiling 和 weight-streaming
变量，再按本次命令行重新设置，避免环境变量污染单变量 A/B。weight streaming 也只有在
环境值明确为 `1` 时才打开。

构建后可在 benchmark 前验证 provenance：

```bash
PYTHONPATH=src python3 scripts/record_engine_provenance.py \
  --engine artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024_vocab65536/llm/llm.engine \
  --metadata-json reports/tensorrt/opt1.engine.metadata.json \
  --reduced-vocab-dir artifacts/reduced_vocab/qwen3_vl_v1_65536_character_coverage \
  --output reports/tensorrt/opt1_vocab65536.engine.provenance.json

PYTHONPATH=src python3 scripts/validate_tensorrt_provenance.py \
  --provenance reports/tensorrt/opt1.engine.provenance.json \
  --tuning-config configs/tensorrt/jetson_orin_nano_int4_v1.json \
  --reduced-vocab-dir artifacts/reduced_vocab/qwen3_vl_v1_65536 \
  --output reports/tensorrt/opt1.provenance-validation.json
```

该检查要求 engine SHA-256 合法，并验证模型、Edge-LLM revision、平台、precision、batch
和固定 input/KV profile；KV 缩小候选应使用对应 profile 的独立配置，不直接套用默认
`i768/k1024` fixed 字段。

对于 visual engine，可直接以构建报告作为期望配置来源，不套用 LLM 的
`TensorRTTuningConfig`：

```bash
PYTHONPATH=src python3 scripts/validate_tensorrt_provenance.py \
  --provenance reports/tensorrt/visual_opt2.engine.provenance.json \
  --build-report reports/tensorrt/visual_opt2.build.json \
  --engine artifacts/engines/qwen3_vl_2b_fp16_visual_opt2/visual/visual.engine \
  --output reports/tensorrt/visual_opt2.provenance-validation.json
```

该模式会同时检查 build report 为 `succeeded`、报告中的 `component`/image profile/
builder level/workspace 与 provenance 完全一致，并重新计算目标 visual engine 的
SHA-256 和文件大小。报告输出 SHA 不匹配时，验证结果为失败，不能进入 benchmark
比较；这能避免把错误的 visual engine 或残留目录误标为 level2/3 候选。

## 3. 低层阶段 benchmark

Edge-LLM 低层 runtime/launcher 应输出 UTF-8 JSONL，每行一个样本：

```json
{
  "sample_id": "ps20-indoor-001",
  "repetition": 1,
  "status": "completed",
  "output_tokens": 78,
  "timings_ms": {
    "prefill_ms": 480.0,
    "time_to_first_token_ms": 710.0,
    "decode_ms": 2200.0,
    "end_to_end_ms": 2750.0
  }
}
```

如果 HTTP server 在响应中返回 `timings_ms` 或 `performance` 对象，ParkSight
HTTP adapter 会把其中的 `prefill_ms`、`time_to_first_token_ms`、`decode_ms`、
`server_e2e_ms` 等真实服务端字段写入 `InferenceRecord.backend_end_to_end_ms`。普通 OpenAI-compatible
响应没有这些字段时，报告会保留 `null`，不会用 HTTP RTT 代替 decode。

Edge-LLM HTTP adapter 默认发送 `stream=true` 并解析 OpenAI-compatible SSE。TTFT 定义为
客户端发起请求到首个非空 `delta.content` 到达的时间，记录在
`stage_timings.time_to_first_token_ms`；它是可观测的流式 TTFT，不等同于单独的
`prefill_ms`。如果服务端或代理不支持流式响应，可在 runtime options 中设置
`"stream_responses": false`，此时只有响应自身返回 TTFT 字段时才填充该指标。

失败样本使用 `"status": "failed"`，`output_tokens` 可以为 `null`，但仍保留
`sample_id` 和 `repetition`。`ttft_ms`、`e2e_ms` 兼容别名会被标准化为
`time_to_first_token_ms`、`end_to_end_ms`。

汇总命令：

```bash
PYTHONPATH=src python3 scripts/benchmark_edgellm.py \
  --input-jsonl reports/tensorrt/opt1.low-level.jsonl \
  --warmup-input-jsonl reports/tensorrt/opt1.warmup.jsonl \
  --metadata-json reports/tensorrt/opt1.metadata.json \
  --tegrastats reports/tensorrt/opt1.tegrastats.log \
  --output reports/tensorrt/opt1.benchmark.json
```

如果需要直接从冻结测试 manifest 产生 JSONL，可在服务已启动的 Jetson 上执行。
当 `--warmup` 大于 0 时必须保存 `--warmup-output-jsonl`，否则无法形成独立的
cold-start 证据：

```bash
PYTHONPATH=src python3 scripts/run_edgellm_benchmark.py \
  --manifest data/manifests/ps20_pilot_v1.jsonl \
  --workload configs/workloads/parking_risk_v1.json \
  --data-root data \
  --limit 20 \
  --repetitions 3 \
  --warmup 1 \
  --warmup-output-jsonl reports/tensorrt/opt1.warmup.jsonl \
  --output-jsonl reports/tensorrt/opt1.low-level.jsonl
```

验证 Edge-LLM 动态 batch 时，将并发度作为单独变量，并为每个工作线程创建独立的
HTTP backend；脚本仍按 repetition 和 manifest 顺序写出 JSONL，避免完成顺序影响统计：

```bash
PYTHONPATH=src python3 scripts/run_edgellm_benchmark.py \
  --manifest data/manifests/ps20_pilot_v1.jsonl \
  --workload configs/workloads/parking_risk_v1.json \
  --data-root data \
  --limit 20 \
  --repetitions 3 \
  --warmup 1 \
  --warmup-output-jsonl reports/tensorrt/opt1.c8.warmup.jsonl \
  --concurrency 8 \
  --engine-max-batch-size 4 \
  --run-metadata-json reports/tensorrt/opt1.c8.run-metadata.json \
  --output-jsonl reports/tensorrt/opt1.c8.low-level.jsonl
```

以 `--concurrency 1` 和 `--concurrency 8` 对同一 engine 做 A/B，才可判断服务端
是否通过动态 batch 提升吞吐。并发结果中的单请求 TTFT/E2E 与整体 wall-clock 吞吐
应分开汇总；不能用并发请求的 HTTP RTT 代替 decode latency。`--run-metadata-json`
记录并发度、stream 和连接复用开关，汇总时可作为 `--metadata-json` 输入；同时写入
steady-state wall-clock、完成请求数、总输出 token 数和
`aggregate_output_tokens_per_second`。该运行级吞吐才用于 dynamic batch 的系统吞吐比较。

执行脚本把 warm-up 单独输出；汇总器只有在传入 `--warmup-input-jsonl` 时才填充
`cold_start_ms`，其值来自 warm-up JSONL 的首个 completed sample；未传入时保留
`null`，避免把稳态请求冒充冷启动。这里的 cold-start 指服务 ready 后的首个请求，
不包含进程启动和 engine load。失败样本保留分类。每条记录同时保留客户端 `end_to_end_ms`、
`http_round_trip_ms`、流式 TTFT 以及服务端返回的 prefill/decode 字段；没有服务端
阶段字段时保留 `null`，不从 HTTP RTT 推导 decode latency。
若 warm-up 失败，执行脚本会停止 steady-state benchmark，但会先写出 warm-up 失败行，
保留初始化失败证据。

只有存在 `decode_ms` 时才计算 decode-only tokens/s；只有存在完整的
`end_to_end_ms` 时才计算端到端 tokens/s。HTTP client 的 round-trip 不应填入
`decode_ms`。

benchmark 汇总还可直接传入 `--provenance-json`；engine 的 SHA-256、文件大小和构建配置会
保存在 `metadata.engine_provenance`，并在 `evidence_sources.provenance_json` 留下来源路径。
这使同一套指标可以绑定到确切的 level、workspace、profile 和 engine 文件。

若是同一 engine 的 runtime 开关 A/B，可再传入 `--runtime-metadata-json`。对比工具会在
`runtime` 字段单独列出 engine 是否相同及具体变化的开关，不把有意的 host-side 变量变化
混入 fixed metadata 的匹配结论。

服务启动时还可传入 `--runtime-metadata-output`，记录 LLM/visual engine 的 SHA-256、路径、
CUDA Graph、pinned context、binding cache、profile switch 和 fallback scan 开关。该文件是
runtime A/B 的身份快照，不替代 benchmark 指标或 Engine Inspector 证据。

两份 benchmark 汇总可进一步计算每个阶段的 p50/p90/p99 改善率：

```bash
PYTHONPATH=src python3 scripts/compare_tensorrt_benchmarks.py \
  --baseline reports/tensorrt/level1.benchmark.json \
  --candidate reports/tensorrt/level2.benchmark.json \
  --output reports/tensorrt/level1_vs_level2.benchmark-comparison.json
```

输出会分别列出 prefill、TTFT、decode、E2E、HTTP RTT 和运行级 aggregate token/s 的 speedup/improvement；
并发度、stream 和连接复用会在独立的 `run` 字段标为 matched/changed，不会伪装成 engine
fixed A/B；metadata
字段不足时标记为 `unverified`，字段冲突时标记为 `mismatch`。汇总中的
`execution.repetitions` 和对比结果中的 `execution.repetitions` 还会保留每次重复的
样本数及阶段分位数；`execution.repetition_gate` 会按默认 3 次、p50 至少 10%、p90
至少 5% 的门槛逐次给出 `pass/fail/unverified`。缺少逐次统计时不会自动放行。

## 4. Attention、kernel 与 CUDA Graph 证据

构建用于分析的 engine 时，可在固定的 builder/profile/workspace 条件下应用
`patches/tensorrt-edge-llm/0010-configurable-profiling-verbosity.patch`，并传入
`--profiling-verbosity detailed`。该选项只改变 engine inspector 可见的层级信息，
不应与正式性能 engine 混用。之后在 Jetson 上使用 TensorRT Engine Inspector 或
Edge-LLM 的 `--dumpProfile`，将 layer、tactic、workspace 和 CUDA event 数据保存到
独立的报告目录。

若已构建 profiling engine，可直接导出 inspector JSON：

```bash
PYTHONPATH=src python3 scripts/inspect_tensorrt_engine.py \
  --engine artifacts/engines/<candidate>/llm/llm.engine \
  --plugin-path /home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
  --output reports/tensorrt/<candidate>.engine-inspector.json
```

两个候选的 inspector JSON 可进一步做层级对齐：

```bash
PYTHONPATH=src python3 scripts/compare_tensorrt_engine_inspectors.py \
  --left reports/tensorrt/level1.engine-inspector.json \
  --right reports/tensorrt/level2.engine-inspector.json \
  --output reports/tensorrt/level1_vs_level2.engine-inspector-comparison.json
```

对比结果中的 `layers.diff_summary` 汇总匹配 layer 的类型、tactic、workspace 和 timing
字段变化，`layers.operator_count_delta` 给出右侧 engine 相对左侧 engine 的算子数量差异。
这些字段用于快速筛选 attention、GEMM、Softmax 或 fusion/split 的 profiling 重点；它们仍然
只是 serialized Inspector 证据，不能单独证明某个算子造成了端到端收益。

比较结果优先按唯一 layer name 对齐，名称缺失或重复时才按 index 兜底；因此 TensorRT
发生 operator fusion/split 时会显式显示新增和移除的 layer。结果同时保留可见的
layer/tactic/workspace/timing 差异；若 profiling verbosity 或执行 profile 不足，对应字段
会标为 `missing`，不能据此声称两个 engine 使用了相同或不同 tactic，或直接把某个 layer
认定为运行时瓶颈。

runtime 日志可用以下命令汇总：

```bash
PYTHONPATH=src python3 scripts/summarize_edgellm_runtime_log.py \
  --log reports/jetson-tensorrt-stage1/opt0_vlm32_run.log \
  --log reports/jetson-tensorrt-stage1/opt1_vlm32_run.log \
  --output reports/jetson-tensorrt-stage1/fmha_graph_comparison.json
```

该报告区分 FMHA cubin 的加载尝试、被 CUDA driver 拒绝的候选、未被拒绝的候选、
aux/worker stream 和 Graph capture，并保存 `numKVHeads`、`headDim`、`kvCacheDtype`、
`usePagedKVCache` 和 `specDecodeType` 等 attention/KV 路由事实；“未被拒绝”仍不等于 TensorRT 在实测 layer 上
选择了该 tactic，最终选择必须以 Engine Inspector 或 Nsight Systems/Compute 为准。
当前 level 0/1 的日志显示 engine 配置均为 `headDim=128`、`numKVHeads=8`，两者
FMHA 候选集合一致，decoder CUDA Graph 均捕获成功。针对长序列 head_dim=128，已建立
独立的 `configs/tensorrt/jetson_orin_nano_int4_fmha_runtime_v1.json`：control 和
candidate 共享 0043 patch，只有 `EDGELLM_FMHA_FORCE_GRANULAR_TILING` 不同；随后再结合
level `2/3` 的 layer/tactic profile 判断收益来源，而不是重复排查视觉 encoder 或 Graph 开关。

0043 应在固定 checkout 上先完成 patch-chain 校验并重新构建 plugin；服务启动时再用矩阵
选择 control 或 candidate。两次启动必须复用同一 engine 和 plugin，只改变 variant：

```bash
FMHA_MATRIX=configs/tensorrt/jetson_orin_nano_int4_fmha_runtime_v1.json
COMMON="--edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --engine-root artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024 \
  --plugin-path /home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
  --runtime-tuning-config ${FMHA_MATRIX}"

# control
.venv-jetson/bin/python scripts/serve_edgellm.py ${COMMON} \
  --runtime-variant control_head128_default_tiling

# candidate
.venv-jetson/bin/python scripts/serve_edgellm.py ${COMMON} \
  --runtime-variant candidate_head128_force_granular_tiling
```

上述命令中的 `${COMMON}` 仅用于示意；正式采集时应分别保存 runtime metadata，并在干净
shell 中启动，避免旧的 `EDGELLM_FMHA_FORCE_GRANULAR_TILING` 环境变量污染 control。

固定 v0.9.1 的 XQA 源码进一步确认：runtime key 由 Q/KV dtype、`head_dim`、Q/KV
head ratio、sliding window 和 `tokensPerPage` 组成；同一 key 的候选按 kernel variant
priority 选择。当前 Qwen3-VL-2B 的实际配置是 `SM87 + FP16 Q/KV + head_dim=128 +
16/8 GQA ratio=2 + contiguous KV + spec_decode=false`，源码路径应为普通 XQA decode
attention。源码同时包含 paged-KV 和 FP8-KV 分支，但当前日志的
`usePagedKVCache=false`，且 Orin Nano `SM87` 不满足 v0.9.1 XQA FP8-KV 的 `SM89+`
条件；这些能力不能写成当前 engine 已启用。

为把“源码推断”落实为“实际函数选择”，可在固定 v0.9.1 checkout 应用
`patches/tensorrt-edge-llm/0028-xqa-selection-diagnostic.patch`，仅在诊断运行时设置
`EDGELLM_LOG_XQA_SELECTION=1`。补丁只在每个 XQA kernel list 首次选择时记录 function、
variant、M tile、SM、spec-decode 和 paged-KV，不改变选择逻辑；默认关闭。该日志仍需
与 Nsight Systems/Compute 的 kernel 时间线结合，不能单独证明 attention 是端到端瓶颈。

板端诊断时，在已应用补丁并重新构建 runtime 的 Edge-LLM 进程前设置该变量，然后运行
一条固定 decode 请求；例如：

```bash
export EDGELLM_LOG_XQA_SELECTION=1
.venv-jetson/bin/python scripts/serve_edgellm.py \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --engine-root artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024 \
  --plugin-path /home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
  --host 127.0.0.1 --port 8000 2>&1 | tee reports/tensorrt/xqa-selection.log
```

随后用低层 benchmark 发起至少一条 decode 请求，并检查
`rg "XQA selected function" reports/tensorrt/xqa-selection.log`。没有该行时，只能记录
“未采集到实际 XQA function”，不能从 FMHA cubin 加载日志反推 decode attention kernel。
服务启动时生成的 runtime metadata 会同时记录 XQA 诊断、greedy block-size 和 direct
embedding、greedy reduction implementation 以及 INT4 GEMM stages 相关环境变量，需将该 metadata
与 benchmark JSON 一起归档。
为避免手工设置变量造成 provenance 漏记，服务入口也提供对应的 CLI 参数，例如
`--log-xqa-selection`、`--greedy-argmax-block-size 1024`、`--greedy-argmax-impl warp`、
`--direct-device-token-embedding`、`--int4-gemm-stages 4` 和
`--int4-gemv-block-size 128`；省略参数表示保持当前环境，
正式 control 应在干净环境启动。对于候选服务，还应重复传入 `--runtime-patch` 绑定实际
应用的补丁文件，例如 `--runtime-patch patches/tensorrt-edge-llm/0024-greedy-argmax-fast-path.patch`；
metadata 会保存补丁的路径、大小和 SHA-256。

`0049-int4-gemv-block-size-candidate-v2.patch` 需要在 `0038-int4-gemv-nperblock4.patch`
之后应用。默认 GEMV 仍使用 `NPerBlock=2`、`256` threads/block；设置
`--int4-gemv-block-size 128` 或 `512` 只切换 `NPerBlock=2` 的 decode GEMV，设置 `256`
会清除该变量。建议先做 `128/256/512` 单变量 A/B，并在 Nsight/ptxas 中同时记录
register 数、occupancy、launch failure 和 GEMV kernel 时间；不能只根据 E2E 变化判断
block size 是否有效。

重新编译包含 `0038`、`0044` 的 plugin 后，control 和 candidate 只改变 block-size
参数，示例启动方式如下：

```bash
# control: 256 threads/block
.venv-jetson/bin/python scripts/serve_edgellm.py \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --engine-root artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024 \
  --plugin-path /home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
  --int4-gemv-block-size 256 \
  --runtime-patch patches/tensorrt-edge-llm/0038-int4-gemv-nperblock4.patch \
  --runtime-patch patches/tensorrt-edge-llm/0049-int4-gemv-block-size-candidate-v2.patch \
  --runtime-metadata-output reports/tensorrt/gemv_block256.runtime.json

# candidate: 128 or 512 threads/block; engine/workload must remain unchanged
.venv-jetson/bin/python scripts/serve_edgellm.py \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --engine-root artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024 \
  --plugin-path /home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
  --int4-gemv-block-size 128 \
  --runtime-patch patches/tensorrt-edge-llm/0038-int4-gemv-nperblock4.patch \
  --runtime-patch patches/tensorrt-edge-llm/0049-int4-gemv-block-size-candidate-v2.patch \
  --runtime-metadata-output reports/tensorrt/gemv_block128.runtime.json
```

`--int4-gemv-n-per-block 4` 应作为另一组独立 A/B，不要与 block-size 变化合并成一个
结论；否则无法判断收益来自 output-channel 分块还是 threads/block。

在把候选服务启动到 Jetson 前，可用 `scripts/check_tensorrt_patch_chain.py` 对固定
Edge-LLM checkout 做链式校验。工具在临时 detached worktree 中逐个执行
`git apply --check` 和 apply，源 checkout 本身不会被修改；输出还会记录 revision、补丁顺序
和每个补丁的 SHA-256：

```bash
PYTHONPATH=src python3 scripts/check_tensorrt_patch_chain.py \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --expected-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
  --patch patches/tensorrt-edge-llm/0024-greedy-argmax-fast-path.patch \
  --patch patches/tensorrt-edge-llm/0027-fused-greedy-reduced-vocab-map.patch \
  --patch patches/tensorrt-edge-llm/0029-greedy-argmax-block-size.patch \
  --output reports/tensorrt/greedy_reduced_vocab_patch_chain.json
```

`--runtime-patch` 仍需在实际服务启动时重复传入同一组补丁；链式校验报告只证明补丁能按序
应用，不证明编译后的 plugin 已加载或 kernel 已被运行时选择。

上述矩阵已固化到
`configs/tensorrt/jetson_orin_nano_int4_gemv_runtime_v1.json`：
`control_n2_block256`、`block128_n2`、`block512_n2` 和 `n4_block256`。
服务入口支持直接指定 `--runtime-tuning-config` 与 `--runtime-variant`，会把变体中的
runtime/kernel 参数注入进程，并在 metadata 中记录配置路径和 variant id；若同时显式传入
冲突的 CLI 值会直接报错。每个变体仍需将对应 `--runtime-patch` 链、runtime metadata
和 benchmark summary 一起归档。GEMV 矩阵已经在每个 variant 中声明
`runtime_patches`，因此服务入口会自动注入同一 patch 顺序；若命令行显式传入 patch，必须与
配置完全一致。增加 `--verify-runtime-patch-chain` 后，服务会在加载 engine 前对该顺序做
临时 worktree 校验，并把成功结果写入 runtime metadata：

```bash
.venv-jetson/bin/python scripts/serve_edgellm.py \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --engine-root artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024 \
  --runtime-tuning-config configs/tensorrt/jetson_orin_nano_int4_gemv_runtime_v1.json \
  --runtime-variant block128_n2 \
  --verify-runtime-patch-chain \
  --runtime-metadata-output reports/tensorrt/gemv_block128.runtime.json
```

建议顺序为
先完成 `control_n2_block256` 与 `block128_n2/block512_n2` 的单变量复验，再单独测试
`n4_block256`，并对每个候选至少保留 kernel 时间、TTFT、decode、E2E、严格 JSON、
峰值内存与 soak 结果。

在已有 greedy top-1 candidate 上，还可以单独测试归约 block size。默认控制值为 `256`；
设置 `EDGELLM_GREEDY_ARGMAX_BLOCK_SIZE=1024` 会选择同一语义的 `1024`-thread single-block
kernel。该开关只用于板端 A/B，不能与 reduced-vocab、动态 batch 或其他 runtime patch
同时首次验证；先固定 engine、请求和服务参数，比较采样 kernel、TTFT、decode 和 E2E。

在 block-size A/B 之后，可以用 `--greedy-argmax-impl warp`（或
`EDGELLM_GREEDY_ARGMAX_IMPL=warp`）测试 `0035-greedy-warp-reduction.patch`。它用 warp
shuffle 加共享内存汇总替代 CUB block reduction；`cub` 或未设置时仍是控制路径。该候选
只覆盖 greedy top-1，不影响 top-k/top-p、logprobs 和动态 batch；必须与
`--runtime-patch .../0035-greedy-warp-reduction.patch` 一起记录，并以逐 token 对齐、
sampling kernel 时间、TTFT/decode/E2E 和 soak 结果决定是否保留。

基于 Nsight Systems，当前 INT4 W4A16 T2 kernel
`gemm_w4a16_T2<(64,128,64,64,32,64,4,128)>` 约占 GPU kernel 时间的 `46.4%`
（约 `262.39 ms`），因此对 shared-memory pipeline 的 `STAGES=2/3/4` 做过板端单变量
复验。`STAGES=2` 比 `STAGES=4` 的 TTFT/E2E p50 慢约 `5.38%/2.14%`；`STAGES=3`
虽有较低的观测 E2E，但 20 条诊断请求严格 JSON 为 `0/20` 且输出长度不稳定，不能视为
有效加速。当前正式插件继续使用 `STAGES=4`，完整 provenance 和结果见
`reports/jetson-tensorrt-revalidation/int4_opt1_w4a16_stages_sweep.json`。

需要注意，TensorRT Edge-LLM v0.9.1 的高层 Python server 默认会在加载 runtime
后捕获 decoding CUDA Graph，原版 server 没有可直接传给 HTTP 启动命令的关闭开关。
仓库提供 `patches/tensorrt-edge-llm/0011-configurable-cuda-graph.patch`，将关闭选项
限定为 `EDGELLM_DISABLE_CUDA_GRAPH=1`，并由 `scripts/serve_edgellm.py --cuda-graph`
统一设置。只有应用该 patch 后，`enabled/disabled` 才构成有效的 CUDA Graph A/B；
未应用 patch 时，命令行的 `disabled` 只能算未验证，不能把启动成功当作 Graph 已关闭。

板端应用 patch 并启动关闭 Graph 的候选服务：

```bash
git -C /home/ubuntu/TensorRT-Edge-LLM apply \
  /home/ubuntu/JetsonVLM/patches/tensorrt-edge-llm/0011-configurable-cuda-graph.patch
.venv-jetson/bin/python scripts/serve_edgellm.py \
  --engine-root artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024_opt1 \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --cuda-graph disabled
```

关闭 Graph 的日志必须出现 `CUDA graph capture disabled by
EDGELLM_DISABLE_CUDA_GRAPH=1.`，且不能把 `Successfully captured decoding CUDA graphs`
作为该次运行的证据。启用组显式使用 `--cuda-graph enabled`，两组应分别保存 runtime
日志并通过本节汇总脚本比较。汇总结果中的 `replay_observed` 只有在日志出现明确的
CUDA Graph replay/launch 标记时才为 true；当前历史日志尚未观测到该标记，因此不能
仅凭 capture 成功宣称 decode 已稳定复用 Graph。Graph 是否真正改善 kernel launch
仍需结合 Nsight Systems 的 GPU/CPU 时间线判断。

当前日志会记录 TensorRT optimization profile 的实际 transition；汇总器输出
`profile_switch_count`、`profile_transitions`、同 profile transition 计数，以及带时间戳日志中
相邻切换消息的间隔分布。该间隔是定位线索，不等于
`setOptimizationProfileAsync` API 的纯耗时，因为两条日志之间还可能包含执行、同步和其他工作。
应用 0013 后，汇总器还会解析 `profile_switch_api_calls` 和
`profile_switch_api_call_summary`，只统计日志明确标记的 `setOptimizationProfileAsync`
Host API 调用耗时。
现有 level1 日志主要是 `1→0→1` 的必要 prefill/decode 交替，没有发现连续同 profile 的冗余切换；
下一步应以 Nsight Systems 或 Edge-LLM 源码中的前后打点确认 profile API 的真实代价，再评估
“每个 optimization profile 固定 execution context”的独立 A/B。

若需要把主机侧 API 调用时间从消息间隔中分离出来，可在 Edge-LLM 源码依次应用
`patches/tensorrt-edge-llm/0013-profile-switch-timing.patch` 和
`patches/tensorrt-edge-llm/0014-pinned-profile-contexts.patch`，重新编译后设置
`EDGELLM_PROFILE_SWITCH_TIMING=1`（或启动服务时传入
`--profile-switch-timing`）。日志中的 `profile switch call ... elapsed_ms` 只表示
`setOptimizationProfileAsync` 的主机调用耗时；在 pinned 模式下只会记录每个 profile 的首次初始化调用。
它是异步接口，不代表 GPU 已完成切换，仍需
用 Nsight Systems 或显式同步实验判断端到端影响。

固定 profile context 候选可通过 `EDGELLM_PIN_OPTIMIZATION_PROFILES=1` 或
`scripts/serve_edgellm.py --pin-optimization-profiles` 启用。应用补丁时使用
`git apply --recount --unidiff-zero`，因为该实验补丁刻意采用零上下文以兼容固定的 v0.9.1
源码；它为每个
optimization profile 懒创建独立的 `IExecutionContext`，首次使用 profile 时只初始化一次，
之后由当前 profile 的 context 复用 CUDA Graph。TensorRT 官方接口允许非并发 execution contexts
共享一块 enqueue memory，但并发复用会产生未定义行为，因此该实验要求调用方串行执行，不能
直接用于并发 batch server（参见 [IExecutionContext API](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/c-api/classnvinfer1_1_1_i_execution_context.html)）。

在固定 context 之外，还可依次应用 `patches/tensorrt-edge-llm/0015-cache-binding-state.patch`，
并通过 `EDGELLM_CACHE_BINDING_STATE=1` 或
`scripts/serve_edgellm.py --cache-binding-state` 开启 host-side binding cache。原始
`EngineExecutor::execute()` 每一步会重新读取所有 I/O tensor 的 address/shape 两遍，分别计算
CUDA Graph key 和 snapshot；该补丁在 `prepare()` 完成绑定后用一次遍历同时生成这两个结果，
使 decode 的 `execute()` 不再重复读取 binding state，并减少每步一次 host-side TensorRT API 遍历。
它依赖调用方只通过 `prepare()` 改变 binding，且不改变 kernel、数值
或 graph 本身；默认关闭，必须与关闭开关的同 engine、同 workload A/B，并同时记录 host E2E、
TTFT、decode、CPU 时间线和内存。该补丁应在 0013、0014 之后应用；当前只完成源码级 patch
sequence 校验，未将潜在收益写入正式 benchmark。

该 host-side 候选的板端 A/B 应保持同一 engine、同一 workload 和同一服务进程配置：

```bash
# control: 0013/0014/0015 均不启用
.venv-jetson/bin/python scripts/serve_edgellm.py \
  --engine-root artifacts/engines/<candidate> \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM

# candidate: 同时开启 profile pinning 与单次 binding-state introspection
.venv-jetson/bin/python scripts/serve_edgellm.py \
  --engine-root artifacts/engines/<candidate> \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --pin-optimization-profiles \
  --cache-binding-state
```

若只想测量 `setOptimizationProfileAsync` 的主机调用时间，单独启动一次诊断服务并传入
`--profile-switch-timing`；该日志开关不要混入正式延迟 A/B。候选 metadata 应额外记录
`pin_optimization_profiles`、`cache_binding_state` 和 `profile_switch_timing`，便于解释
CPU 时间线和结果差异。

`0016-registry-name-index.patch` 是独立的 host-side 微优化：`TensorRegistry::contains()`
原先在线性扫描展开后的 registry；补丁在 registry materialize 时建立
`std::unordered_set<std::string_view>` 名称索引，使 `EngineExecutor::prepare()` 的未注册
I/O 判断从线性查找变为常数期望复杂度。索引中的 `string_view` 只引用 registry 自身拥有的
展开名称，并在 registry 变脏时同步重建；该补丁不改变 TensorRT binding、shape、kernel 或
数值结果。它尚未在 Jetson 编译和测量，必须与未应用补丁的同 engine、同 workload 做 CPU
launch、TTFT、decode 和 E2E A/B，不能仅凭源码复杂度写入性能结论。

在已应用 `0014-pinned-profile-contexts.patch` 和 `0016-registry-name-index.patch` 后，
可选应用 `0017-cache-registered-bindings.patch`，并设置
`EDGELLM_CACHE_REGISTERED_BINDINGS=1` 或启动服务时传入
`--cache-registered-bindings`。它按 `IExecutionContext` 和 optimization profile 分层缓存
registry-managed tensor 的 device address 和 resolved shape；地址或 shape 未变化时跳过
对应的 `setTensorAddress()` / `setInputShape()`。profile-pinned context 具有独立缓存；
共享 context 在 profile 切换时清空缓存，以避免复用可能被 TensorRT 失效的动态 shape/binding
状态；LoRA 权重地址变化会自动重新绑定。该开关默认关闭，必须与 control 做同 engine、同 workload A/B，并结合
`prepare()` CPU trace 和正确性检查；当前没有板端性能证据。

在 `0017` 之后还可选应用 `0018-skip-redundant-profile-switch.patch`，并设置
`EDGELLM_SKIP_REDUNDANT_PROFILE_SWITCH=1` 或启动服务时传入
`--skip-redundant-profile-switch`。它在首次使用或 profile 发生变化时仍调用
`setOptimizationProfileAsync()`，同一 profile 连续准备时跳过重复调用，只针对
host-side API 调度，不改变 TensorRT engine、kernel、CUDA Graph 或数值结果。该开关默认
关闭，必须与未开启补丁的同 engine、同 workload A/B，并记录 profile-switch 调用次数、
CPU 时间线、TTFT、decode、E2E 和内存；当前仅完成源码级 sequence 校验，尚无板端性能证据。

在 `0018` 之后还可选应用 `0020-skip-redundant-registry-scan.patch`，并设置
`EDGELLM_SKIP_FALLBACK_BINDING_SCAN=1` 或启动服务时传入
`--skip-fallback-binding-scan`。构造阶段先检查 engine 的全部 I/O 是否已经由
`TensorRegistry` 覆盖；只有全覆盖时才在 `bindAll()` 后跳过 fallback scan，否则保留原路径并
输出 warning。该开关默认关闭，只减少 host-side engine I/O introspection，不改变 binding、
kernel、CUDA Graph 或数值；必须与 control 做同 engine、同 workload A/B，并记录
`prepare()` CPU 时间、TTFT、decode、E2E、内存和正确性。

本开关已在 Jetson 上完成一次严格匹配 A/B。由于正式源码 checkout 存在既有未提交改动，
板端 candidate 使用固定 v0.9.1 revision 的隔离 worktree 和
`0023-safe-skip-fallback-binding-scan.patch` 编译；control 与 candidate 复用同一
level1 INT4 engine、plugin、CUDA Graph 和服务参数，仅改变该环境变量。前 20 个冻结 test
样本各运行 `3×20`，两边均 `60/60` 完成且每次输出均为 32 chunks：

| 指标 | control | candidate | candidate 改善 |
|---|---:|---:|---:|
| TTFT mean | 690.54 ms | 578.66 ms | 16.20% |
| TTFT p50 | 690.53 ms | 577.75 ms | 16.33% |
| TTFT p90 | 693.05 ms | 581.34 ms | 16.12% |
| E2E mean | 1566.38 ms | 1441.12 ms | 8.00% |
| E2E p50 | 1566.57 ms | 1440.54 ms | 8.04% |
| E2E p90 | 1569.83 ms | 1445.50 ms | 7.92% |

完整输出 smoke 的 control/candidate 严格 JSON 均为 `9/20`，失败样本一致，未观察到该
host-side 变更引入的质量回归；但严格 JSON 门槛本身尚未通过，且本轮未重新采集候选的
峰值内存和 100 次 soak。因此该结果只证明当前口径下的 host-side 性能收益，candidate
仍不晋级正式默认。原始逐请求数据、quality smoke 和 provenance 见
`reports/jetson-tensorrt-revalidation/int4_opt1_host_fallback_scan_sweep.json` 及其同目录
的 `int4_opt1_hostscan_*.json` 文件。

HTTP adapter 还提供独立的 `reuse_http_connection` 候选（默认关闭），通过复用同一条
HTTP/1.1 keep-alive 连接减少逐请求 TCP 建连。它只影响服务链路的客户端 E2E/TTFT，
不改变 TensorRT engine、CUDA Graph、kernel、prefill 或 decode；因此不能把该开关带来的
E2E 改善写成 TensorRT 加速。若启用，应与关闭开关的同一 engine、同一 server、同一
workload 做 A/B，并单独记录 HTTP RTT、TTFT 与服务端阶段时间；正式板端 TensorRT
结论仍以 `prefill_ms`、`decode_ms` 和 engine-level evidence 为准。
低层 benchmark 可通过 `scripts/run_edgellm_benchmark.py --reuse-http-connection` 开启，
应用配置则在 `tensorrt_edge_llm_http` 的 `runtime.options` 中设置
`"reuse_http_connection": true`。

仓库已提供两个独立的 profiling 构建流程：
`configs/flows/build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt2_profile.json` 和
`configs/flows/build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt3_profile.json`。它们固定
`i768/k1024`、workspace `1024 MiB`，仅将 builder level 分别设为 2/3，并打开
`profiling-verbosity=detailed`；构建报告单独保存，不覆盖正式 engine。

workspace 第二阶段也已提供独立流程：
`configs/flows/build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt1_workspace512.json` 和
`configs/flows/build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt1_workspace1536.json`。两者固定
level1、`i768/k1024`，仅改变 workspace 上限；视觉 engine 继续复用已核验版本。

host-side runtime 开关使用独立矩阵
`configs/tensorrt/jetson_orin_nano_int4_runtime_v1.json`。其中 `control`、单变量候选和
`pinned_plus_binding_caches`、`all_host_fast_path` 分开列出；其中 `all_host_fast_path`
同时打开 profile pinning、binding-state cache、registered-binding cache、重复 profile
switch 消除和 fallback scan 消除，只能作为组合候选，不能替代单变量归因。profile switch
timing 只作为诊断开关，不混入性能候选。每个变体应使用同一 engine、同一 workload、同一
power mode，并将服务生成的
`runtime-metadata-output` 与 benchmark 结果一并保存。

当 visual engine 通过 symlink 复用（当前 INT4 LLM 部署即采用该布局）时，runtime
provenance 同时记录 `requested_path`、`resolved_path` 和文件 SHA-256。报告应以 SHA-256
判断是否为同一 engine，以 `requested_path` 说明部署语义，避免把共享的 FP16 visual
engine 误记为独立的 INT4 visual engine。

## 5. KV cache profile contract

profile 缩小必须先通过 prompt contract 门禁，再在板端构建和测量。当前 v1 workload
的实测输入为 735 token；对 `maxGenerateLength=32`，`i768/k768` 需要的容量为
`735+32=767`，有 1 token headroom，可以作为下一轮 KV cache 候选。`i512/k768`
会在输入长度检查阶段失败，不能拿它作为性能结果。

```bash
PYTHONPATH=src python3 scripts/check_tensorrt_profile_contract.py \
  --prompt-contract reports/prompt-contract/qwen3_vl_v1_i768_budget_20260830.json \
  --max-input-len 768 \
  --max-kv-cache-capacity 768 \
  --max-generate-length 32 \
  --max-batch-size 1 \
  --num-kv-heads 8 \
  --head-dim 128 \
  --element-size-bytes 2 \
  --output reports/tensorrt/i768_k768_contract.json
```

传入 KV 维度后，contract 报告还会给出连续 FP16 K/V buffer 的理论字节数。对于当前
`batch=1、numKVHeads=8、headDim=128`，`k1024` 约为 4 MiB，`k768` 约为 3 MiB；该数字
不含 allocator 对齐和其他 runtime buffer，不能替代 tegrastats 的实测峰值内存。

该检查只证明 profile 覆盖输入和输出预算，不证明 engine 能够构建或推理更快；
通过后仍需完成 engine load、decode profile、E2E、内存和质量门禁。

通过后，使用独立 engine 目录构建 KV 缩小候选；保持 builder level=1、workspace=1024 MiB，
避免把 profile 变化和 builder 变化混在一起：

```bash
PYTHONPATH=src python3 scripts/build_edgellm_vlm_engines.py \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --expected-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
  --onnx-root artifacts/onnx/qwen3_vl_2b_int4_awq_n128 \
  --engine-root artifacts/engines/qwen3_vl_2b_int4_awq_i768_k768_opt1 \
  --component llm \
  --max-batch-size 1 \
  --max-input-len 768 \
  --max-kv-cache-capacity 768 \
  --workspace-limit-mib 1024 \
  --builder-optimization-level 1
```

## 6. 候选晋级门禁

每个 level 先运行 3 次、每次 20 个冻结测试样本，再执行 100 个样本 soak test。StudyReport
必须使用 `repetitions=3`，候选门禁命令为：

```bash
PYTHONPATH=src python3 scripts/evaluate_tensorrt_candidate.py \
  --baseline reports/tensorrt/level0.study.json \
  --candidate reports/tensorrt/level1.study.json \
  --soak reports/tensorrt/level1.soak.json \
  --candidate-runtime-log-summary reports/tensorrt/level1.runtime-summary.json \
  --baseline-jetson-summary reports/tensorrt/level0.runtime-summary.jetson.json \
  --candidate-jetson-summary reports/tensorrt/level1.runtime-summary.jetson.json \
  --tuning-config configs/tensorrt/jetson_orin_nano_int4_v1.json \
  --output reports/tensorrt/level1.gate.json
```

命令只有在后端完成率、严格 JSON、风险准确率、事件 micro-F1、p50/p90、内存和 soak
均满足门禁，且若提供 runtime summary 则 decoder CUDA Graph capture 与 replay 均确认时返回 0；
候选与基线提供的 runtime identity（backend、revision、model、precision 等）不一致时
同样返回 2 并保留失败原因。gate 通过后仍需人工确认，再将
候选配置切换为正式默认。

如果同时传入 `--benchmark-comparison`，候选 evaluator 还要求低层 benchmark 的
`execution.repetition_gate.eligible=true`；这会把逐次 p50/p90 稳定性纳入最终晋级，而不只
依赖 StudyReport 的总体分位数。

对于 FMHA head-dim=128 的 tiled candidate，还必须额外提供候选 runtime metadata 和同次
运行日志摘要，并显式打开证据门禁：

```bash
PYTHONPATH=src python3 scripts/evaluate_tensorrt_candidate.py \
  --baseline reports/tensorrt/level0.study.json \
  --candidate reports/tensorrt/fmha_candidate.study.json \
  --soak reports/tensorrt/fmha_candidate.soak.json \
  --candidate-runtime-metadata reports/tensorrt/fmha_candidate.runtime.json \
  --candidate-runtime-log-summary reports/tensorrt/fmha_candidate.runtime-summary.json \
  --require-fmha-tiled-active \
  --output reports/tensorrt/fmha_candidate.gate.json
```

该选项要求 metadata 中的 `fmha_force_granular_tiling` 为启用状态，且日志出现
`forced tiled head_dim=128 candidate` marker；配置被写入但实际分支未命中时，候选不会被
标记为有效。该门禁只用于 0043 FMHA candidate，不改变正式 level0/level1 晋级口径。

StudyReport 如果没有保存 RAM、功耗或温度，可以成对传入由同次 tegrastats 生成的
`parksight_jetson_runtime_summary_v1` 文件。两份资源摘要必须分别匹配 baseline/candidate
的 workload、runtime、功耗 identity，并且 `runtime_execution.record_count` 必须等于对应
StudyReport 的 records 数量；否则 evaluator 直接拒绝，不能用较短或不同实验的遥测替代。

若要同时确认 provenance 没有指向错误的 engine 文件，可额外传入 `--engine`；验证器会重新
计算文件 SHA-256 和大小，并将结果写入 `engine_file`。仅有 provenance 中的 64 位字符串
格式正确，并不等于当前磁盘文件就是被记录的 engine。

## 7. 证据解释边界

- builder optimization level 改变 tactic 搜索投入，必须将构建时间与推理时间分开记录。
- timing cache 只复用 builder 的 tactic timing；它不能替代 engine 的 latency benchmark，
  且 cache 必须绑定目标 GPU、CUDA/TensorRT 版本和 BuilderConfig。
- workspace、profile 和 KV capacity 改变 engine 的形状/内存假设，不能与 level A/B 混为一个变量。
- CUDA Graph 的“捕获成功”必须由 runtime 日志或 Nsight Systems 验证。
- FP16 weight streaming 的驻留预算是内存—速度曲线实验；现有 INT4 engine 未启用该功能，
  不应传入 weight-streaming runtime 参数。

## 8. 当前 engine 的技术能力边界

下表只描述当前项目产物和日志已经能证明的状态，不把 Edge-LLM 源码中存在的插件或
TensorRT API 自动等同为本次 Qwen3-VL engine 已使用的能力。

| 技术 | 当前证据 | 当前结论 | 下一步证据 |
|---|---|---|---|
| FMHA / fused attention | level0/1 日志有 FMHA cubin 加载尝试，engine 配置为 `headDim=128`、`numKVHeads=8` | attention 融合路径存在；具体 layer tactic 未确认 | Engine Inspector、Nsight Systems/Compute |
| GQA | Qwen3-VL 配置为 16 个 Q heads、8 个 KV heads | 是模型结构，不是本轮新增优化开关 | 对比 attention layer 的 tactic 和 kernel 时间 |
| CUDA Graph | 同一 level1 engine、同一 workload 的开关 A/B 各完成 60/60；开启组 E2E p50 `1441.25 ms`，关闭组 `1510.45 ms`，慢约 `4.80%`；关闭组使用隔离 worktree + `EDGELLM_DISABLE_CUDA_GRAPH=1` | 在该板端和请求口径下应保持开启；收益体现在 decode/E2E，TTFT 基本不变。仍不把它归因到某个具体 kernel，也不声称每步 graph 内容恒定 | runtime replay marker、Nsight Systems 连续 decode trace；摘要见 `reports/jetson-tensorrt-revalidation/int4_opt1_cuda_graph_sweep.json` |
| TensorRT timing cache | 已提供 `ITimingCache` 加载/回写补丁和 build report digest | 仅用于重复构建时复用 tactic timing，不是推理优化证据 | cache 命中前后构建耗时、匹配环境与 engine A/B |
| profile 切换 | opt0/opt1 均记录 43 次 `0↔1` 切换、同 profile 为 0；相邻切换消息间隔 p50 分别为 751/654 ms，p90 为 3992/893 ms | 间隔不是 `setOptimizationProfileAsync` 纯耗时；已提供 0013 打点和 0014 固定 context 候选，仍需板端 A/B 验证 | `reports/jetson-tensorrt-stage1/opt0_vlm32_runtime-summary.json`、`opt1_vlm32_runtime-summary.json`，以及 Nsight CPU/GPU 时间线 |
| host binding state cache | `0015-cache-binding-state.patch` 在 `prepare()` 后一次遍历生成 graph key 与 snapshot；默认关闭 | 只减少 graph lookup 前的 host-side address/shape introspection，不改变 kernel 或数值；尚未板端验证 | 关闭/开启 `EDGELLM_CACHE_BINDING_STATE` 的同 engine A/B，结合 CPU launch trace |
| registry name index | `0016` 为展开后的 binding name 建立 host-side unordered index | 只减少 `TensorRegistry::contains()` 的线性查找开销，不改变 binding、shape 或 kernel；尚未板端验证 | 同 engine A/B，结合 `prepare()` CPU trace |
| registered binding cache | `0017` 按 execution context 和 optimization profile 分层缓存已注册 tensor 的 address/shape；pinned context 保留 profile-local 状态，共享 context 在 profile 切换时清空缓存 | 只减少 host-side binding setter 调用；默认关闭，尚未板端验证 | 同 engine A/B、正确性和 `prepare()` CPU trace |
| same-profile switch skip | `0018` 跳过同 profile 的重复 `setOptimizationProfileAsync()` | 只减少 host-side profile-switch API 调用；默认关闭，尚未板端验证 | 同 engine A/B、profile-switch 调用计数与 CPU trace |
| INT4 GEMM pipeline stages | `0032-configurable-int4-gemm-stages.patch` 保留默认 `4` stages，并通过 `EDGELLM_INT4_GEMM_STAGES=2/3/4` 选择编译期实例；已有 `0021/0022` 板端复验 | `STAGES=2` 相对 `4` 的 TTFT/E2E p50 慢约 `5.38%/2.14%`；`STAGES=3` 严格 JSON `0/20`，当前保留 `4` | 若重新验证，必须使用固定 engine/workload 输出 kernel 聚合时间、数值、显存和 soak |
| redundant registry scan skip | `0023` 在确认全量 registry 覆盖后跳过 fallback I/O scan；同 engine `3×20` A/B 两边均 `60/60` 完成 | candidate TTFT p50 `577.75 ms` vs control `690.53 ms`，E2E p50 `1440.54 ms` vs `1566.57 ms`；严格 JSON 两边均 `9/20`，未晋级默认 | 补采峰值内存、100 次 soak，并通过严格 JSON/质量门禁后再评估 |
| vocabulary reduction | [v0.9.1 官方能力](https://nvidia.github.io/TensorRT-Edge-LLM/0.9.1/user_guide/features/reduce-vocab.html)支持 task-specific output vocabulary reduction；当前 engine 仍是完整 `151936` vocab，Nsight 已定位最终 `lm_head` 为高价值热点 | 这是当前最值得优先验证的 GPU-side 候选：只缩小输出 logits/lm_head，不改变 transformer hidden states；但会改变可生成 token 集合，不能默认视为无损。历史输出-only 的 `16384` 候选在独立 holdout 仅覆盖 `47.31%`，已否决 | `65536/32768` plus-history map 均在冻结 `ps20_pilot_v1` 覆盖 `632/632` reference token、`20/20` 样本无缺失；先用 `65536` 做独立构建，检查 EOS、严格 JSON、风险质量、lm_head tactic、decode、显存和 soak |
| dynamic batching | 正式 level0/level1 engine 的运行配置为 `maxBatch=1`；已有 `maxBatchSize=4` 的独立 opt1 flow，并新增 `maxBatchSize=4 + maxKVPoolPages=64` paged-KV flow；`0026` 负责 batch-eviction token compaction 候选 | 不能用并发 HTTP 请求直接推断现有 engine 获得 dynamic batch；需分别构建 contiguous/paged batch4 engine，再比较 `concurrency=1/4` 的运行级 wall-clock token/s；paged 主要观察容量复用和并发吞吐，不预设单请求延迟收益 | batch4 engine Inspector/runtime identity、`usePagedKVCache`/page capacity、单请求数值对齐、并发吞吐、TTFT/E2E、显存、功耗、温度和 100 次 soak |
| reduced-vocab greedy map fusion | `0027` 将 greedy top-1 的 reduced→full ID 查表放入 `greedyTop1Kernel`；仅影响 reduced-vocab + greedy 分支 | 预期减少每个 decode round 的一次小型 map kernel 和 ID 全局读写；当前只有源码级可应用性证据，未计入收益 | reduced-vocab engine 的 kernel/Nsight trace、full-token 数值对齐、TTFT/decode/E2E、严格 JSON 和 soak |
| INT4 AWQ lm_head | 现有量化 provenance 明确 `lm_head_precision=fp16`；新增候选 flow 显式传 `--lm-head-quantization int4_awq`，[官方 v0.9.1 量化文档](https://nvidia.github.io/TensorRT-Edge-LLM/0.9.1/user_guide/features/quantization.html)将 `int4_awq` 列为 LM head 支持方法 | 这是针对最终输出 GEMM 的独立权重压缩候选；可能降低权重搬运和 GEMM 成本，也可能影响 logits/JSON 质量；当前无板端结果 | 独立 checkpoint/ONNX/engine，校准与反量化误差、Inspector tactic、Nsight kernel、TTFT/decode/E2E、严格 JSON、风险质量、显存和 soak |
| reduced-vocab + INT4 lm_head | 新增 `32768/65536` 两组 `export` 与 `build_*_opt1` flow，同时绑定 reduced-vocab coverage 和 INT4 calibration provenance | 这是两个高风险变量的组合候选，不能把收益归因给单一变量；exporter/build 兼容性、token 映射、INT4 误差和输出质量均需实测 | 先确认 export/build report 成功，再与正式 level1 严格 A/B；记录 lm_head tactic/kernel、严格 JSON、风险准确率、事件 micro-F1、TTFT、decode、E2E、峰值内存和 100 次 soak |

`lm_head` 候选的 quantize/export/build flow 还必须携带量化阶段的
`calibration_provenance.json`。build wrapper 要求显式传入
`--expected-lm-head-precision int4_awq`，并在调用 `llm_build` 前校验量化状态、固定
Edge-LLM revision、校准 workload、样本数和 `lm_head_precision`，同时将 provenance 的
大小与 SHA-256 写入 build report。这个门禁只证明候选 checkpoint 的声明和构建配置一致，
不替代量化误差、严格 JSON、质量和板端性能验证。若将 reduced-vocab 与 packed INT4
`lm_head` 组合，build gate 还要求 `reduced_vocab_size % 128 == 0`，以满足 group size
128 的权重打包约束；当前 `32768/65536` 候选均满足该结构条件。

量化脚本在实际校准前还会检查固定版本的 `quantize_and_export` 函数是否接受
`lm_head_quantization` 参数；如果版本不支持，会在加载大模型和执行校准前失败，避免把
参数兼容性错误误判为量化或 TensorRT 性能问题。
| TensorRT operator fusion | 当前只有 runtime/engine profile 入口，尚无已归档 layer diff | 不能仅凭 engine load 声称融合收益 | detailed Inspector JSON |
| greedy top-1 sampling kernel | 当前 vanilla greedy `selectAllTopK(topK=1)` 使用 8-block `topKStage1` 加第二阶段 reduction，并写临时 logits；`0024-greedy-argmax-fast-path.patch` 提供单 block CUB argmax 候选，仅在不需要 top-K values 时启用 | 只改变采样 kernel 的实现路径，不改变 top-1 输出语义；尚无板端性能证据，不能提前计入 decode 加速 | Jetson 编译、采样输出逐 token 对齐、Nsight kernel/launch trace、decode A/B |
| greedy argmax block size | `0029-greedy-argmax-block-size.patch` 在 `0024/0027` 基础上保留 `256` 控制路径，增加可选 `1024`-thread single-block reduction；由 `EDGELLM_GREEDY_ARGMAX_BLOCK_SIZE=1024` 选择 | 仅改变 greedy argmax 的线程并行度；源码级候选，尚无 Orin Nano 性能证据 | 固定 engine/workload 的 `256/1024` A/B、逐 token 对齐、采样 kernel 时间、TTFT/decode/E2E 和 soak |
| greedy warp reduction | `0035-greedy-warp-reduction.patch` 在 CUB 控制路径之外增加 warp-shuffle + shared-memory argmax 候选；由 `EDGELLM_GREEDY_ARGMAX_IMPL=warp` 选择，默认关闭 | 只改变 greedy top-1 reduction 实现；尚无 Jetson 编译和性能证据，不能提前计入 decode 加速 | 固定 block size 的 `cub/warp` A/B、逐 token 对齐、sampling kernel 时间、TTFT/decode/E2E 和 soak |
| device-side token feedback | `0025-device-token-feedback.patch` 仅在 vanilla `activeBatchSize==1` 且非调试路径将上一轮 `sampling.indices` 直接 D2D copy 到下一轮 `idsInput`；其他 batch size 和调试路径保留 host 反馈 | 只减少 batch=1 每轮 host token loop 和 H2D token copy；仍需 token D2H 与同步来完成 EOS/stop-word 和请求状态更新，尚无板端证据 | Jetson 编译、逐 token 对齐、D2D/H2D copy trace、decode A/B |
| direct device-token embedding | `0030-direct-device-token-embedding.patch` 在 vanilla、非 debug、无 Gemma4 PLE 路径直接使用 GPU `sampling.indices` 做 embedding lookup，显式设置 `EDGELLM_DIRECT_DEVICE_TOKEN_EMBED=1` 才启用 | 预计减少每轮一个极小 D2D token copy；不改变模型计算和采样语义，但只有源码级证据 | 固定 engine 的开关 A/B、dynamic batch eviction 对齐、D2D/H2D trace、TTFT/decode/E2E 和 soak |
| direct device-token reshape refinement | `0031-skip-unused-device-token-reshape.patch` 在 `0030` direct 路径跳过未使用的 `idsInput.reshape`，传统和 fallback 路径保留 reshape | 只减少 host-side shape metadata 操作；默认关闭、无板端证据 | 与 `0030` 组合进行固定 engine A/B，并核对 dynamic batch、debug/Gemma4 fallback |
| dynamic-batch device token feedback | `0026-device-token-feedback-dynamic-batch.patch` 在 `onBatchEvict` 中用已有 `compactTensorBatch` 压缩 `sampling.indices`，然后将 `0025` 的 D2D 路径扩展到非调试 vanilla batch | 解决 batch eviction 后 sampling slot 顺序变化这一安全前提；仍需动态 batch 数值对齐和吞吐 A/B，尚无板端证据 | Jetson 编译、batch eviction 对齐、D2D/H2D trace、并发吞吐 A/B |
| empty-batch eviction compaction/sync skip | `0033-skip-empty-batch-compaction.patch` + `0034-skip-empty-batch-eviction-sync.patch` 在 `newActiveBatch==0` 时跳过无 survivor 可搬运的 KV/Mamba compaction、mapping H2D 和尾部同步，仍执行 active-batch 状态更新和 strategy eviction callback | 主要减少请求结束时的无效 cache kernel/API 操作；依赖固定 v0.9.1 各 decoder 在全量结束时不消费 GPU mapping，默认关闭且尚无板端证据 | batch=1/全量结束和 dynamic-batch 两种路径的 CUDA API/Nsight、结果对齐、TTFT/E2E 和 soak |
| XQA decode attention selection | v0.9.1 源码按 dtype、head_dim、GQA ratio、sliding window、`tokensPerPage` 选择 XQA；当前配置为 `SM87/FP16/head_dim=128/ratio=2/contiguous/spec_decode=false`；`0028` 可记录首次实际选中的 function/variant | 当前只能确认普通 XQA 路由条件，尚无日志证明具体 function，也没有 shape-specific kernel 收益证据；不把 paged KV、FP8 KV 或 spec decode 写成当前实现 | 板端启用诊断日志、Nsight Systems/Compute kernel 名称与时间、固定 shape 的 kernel A/B |
| XQA list lookup cache | `0036-xqa-thread-local-list-cache.patch` 为每个 host thread 缓存固定 `(dtype, KV dtype, SM, spec-decode, paged-KV)` 对应的 immutable XQA list，避免普通 decode 每步重复获取全局 loader mutex；未改变 kernel selection 或 launch 参数 | 已在 Jetson 隔离 worktree 编译并加载正式 engine；与 `0037` 组合的 3 请求 screen 中没有可见 E2E p50 收益，且未采集阶段级和 telemetry 证据，不能归因于 attention kernel 加速 | 若继续研究，拆分 `0036` 单独 A/B，并补充 CPU/API trace、XQA kernel 时间、TTFT/decode/E2E 和 soak |
| XQA selection cache | `0037-xqa-selection-cache.patch` 在 `0036` 基础上缓存按 runtime key 选出的 `XQAKernelFuncInfo`，避免每个 decode step 重复 `unordered_map.find` 与 `std::min_element`；key 同时绑定 SM、spec-decode 和 paged-KV | 与 `0036` 组合的 Jetson 3 请求 screen 输出与 control 一致，E2E p50 为 `8089.00 ms` vs `8085.39 ms`，候选变化 `+0.04%`；只减少 host-side selection 开销，未达到继续扩展测试的收益信号 | 只有出现可复现正向信号才进入 3x20；需同时补充 CPU/API trace、XQA kernel 时间、TTFT/decode/E2E 和 soak |
| FMHA selection cache | `0040-fmha-selection-cache.patch` 缓存每个 host thread 的 FMHA immutable list 和按完整 `FMHAKernelHashKey` 选出的 function，避免重复 loader mutex、map lookup 和 function copy | 只改变 context prefill 的 host-side FMHA dispatch；不改变 FMHA cubin、grid、mask、Q/K/V 地址或数值；尚无 Jetson 性能证据 | 同 engine/workload 的 control/candidate A/B、FMHA CPU/API trace、FMHA kernel 时间、TTFT/prefill/E2E 和 soak |
| INT4 GEMV N-per-block | `0038-int4-gemv-nperblock4.patch` 为 `M=1` 且 `N % 16 == 0` 的 INT4 GEMV 增加 `NPerBlock=4` 候选；默认 `NPerBlock=2` 保持不变 | 通过每个 block 计算 16 个输出通道，将 output-channel block 数减半；只改变 INT4 GEMV 的线程块覆盖，不改变 scale、权重布局或结果语义；尚无 Jetson 性能证据 | 固定 INT4 engine 和 greedy workload 的 `2/4` A/B、逐 token 对齐、GEMV kernel 时间、TTFT/decode/E2E、显存和 soak |
| INT4 GEMV block size | `0049-int4-gemv-block-size-candidate-v2.patch` 在保留默认 `256` threads/block 的前提下，为 `NPerBlock=2` 增加显式 `128/512` threads/block 候选，由 `EDGELLM_INT4_GEMV_BLOCK_SIZE` 选择 | 只改变 decode GEMV 的线程块并行度；可能改善内存延迟隐藏，也可能因寄存器压力和 occupancy 变差而变慢；板端 matched decode A/B 未见 material gain | 固定 engine/workload 的 `128/256/512` A/B、`ptxas` 寄存器与 occupancy、逐 token 对齐、GEMV kernel 时间、TTFT/decode/E2E、显存和 soak |
| INT4 GEMM CTA-N | `0039-int4-gemm-cta-n256.patch` 将 `kGemmCtaN` 从 `128` 改为 `256`，编译出 `gemm_w4a16_T2<...,256,...>`；这是独立 build-time candidate，不改变正式源码默认值 | 固定 Qwen3-VL-2B 文本侧 INT4 输出维度 `2048/1024/6144` 均 256 对齐；共享内存估算约 66 KB，仍低于 Orin 99 KB 上限；可能因线程数、寄存器和 occupancy 变化而变慢，尚无板端证据 | 独立 plugin build 的 `CTA_N=128/256` A/B、W4A16 kernel 时间、逐层 N shape 数值对齐、TTFT/decode/E2E、显存和 soak |
| INT4 GEMM CTA-N=512 | `0045-int4-gemm-cta-n512.patch` 将 `kGemmCtaN` 从 `128` 改为 `512`，但固定 `CTA_M=64/CTA_K=64/STAGES=4` 的 shared memory 为 `102400` bytes，超过源码 `99*1024` 的 compile-time 上限 | 该 patch 是负向资源校验样例，不能作为独立 build candidate；512 tile 若要继续研究，必须重新设计 tile/stages，并将其作为新的 compound candidate | 先通过资源/编译 gate，再做实际 W4A16 kernel launch、ptxas 寄存器/shared memory/occupancy、逐层数值对齐、TTFT/decode/E2E、显存和 soak |
| INT4 GEMM CTA-M | `0041-int4-gemm-cta-m128.patch` 将 prefill W4A16 GEMM 的 `CTA_M` 从 `64` 改为 `128`，保持 `CTA_N=128`；这是独立 build-time candidate | 对 M 方向 block 做更大 tile，理论上减少 prefill block 数；`CTA_M=128, CTA_N=128, STAGES=4` 按源码公式为 `82944` bytes，低于 Orin 99 KiB，但寄存器/occupancy 可能抵消收益；尚无板端证据 | 独立 plugin build 的 `CTA_M=64/128` A/B、prefill W4A16 kernel 时间、逐层数值对齐、TTFT/E2E、显存和 soak |
| INT4 GEMM CTA-K | `0042-int4-gemm-cta-k128.patch` 将 `CTA_K` 从 `64` 改为 `128`，保持 `CTA_M=64/CTA_N=128`；独立 plugin 已构建并能加载正式 engine | 候选符号和 shared-memory 静态约束均通过，但同一单样本输出重复括号/空格并打满 64 token，严格 JSON=false；属于功能 gate 失败，不能使用其 latency 结果 | 保留失败日志、候选 plugin SHA 和正式 plugin 对照；若继续研究，先修复 CTA_K 的权重/共享内存布局数值问题，再重新做逐 token 对齐与性能 A/B |
| FMHA head_dim=128 tiled | `0043-fmha-head128-tiled-candidate.patch` 仅在 `SM87 + head_dim=128 + S>64` 且显式设置环境变量时，将现有 non-tiled FMHA 选择切换为已编译 tiled cubin；矩阵见 `configs/tensorrt/jetson_orin_nano_int4_fmha_runtime_v1.json` | 直接针对当前约 70.7 ms 的 FMHA 类别做 kernel 选择 A/B；tiled cubin shared memory 为 81,920 bytes，可能减少长序列 tile 量化损失，也可能因 occupancy 变慢；默认路径和其他 head/SM 不变 | 同一 engine/workload 的 control/candidate A/B、FMHA kernel 名称与时间、TTFT/prefill/E2E、显存和 soak |
| FMHA head_dim=64 tiled | `0047-fmha-head64-tiled-candidate.patch` 仅在 `SM87 + head_dim=64 + S>64` 且显式设置 `EDGELLM_FMHA_FORCE_GRANULAR_TILING_D64=1` 时，强制走 granular/tiled FMHA；默认启发式和其他 head/SM 不变 | 候选 plugin 已在隔离 worktree 编译，服务日志确认实际进入 forced tiled 分支；但 `S=1444` 功能 smoke 抛出 `There must be one kernel to implement the MHA` 并导致服务 abort，功能 gate 失败，未采集有效 latency | 保留 plugin SHA、运行时错误和正式 engine 未修改证据；若继续研究，先为该 shape 找到可用的 D64 tiled kernel 或恢复 shape 选择条件，再做数值对齐、TTFT/prefill/E2E、显存、功耗、温度和 soak |
| 视觉 engine builder level | `build_qwen3_vl_2b_fp16_visual_engine.json` 为 level1；opt2/opt3 只改变 visual builder level，LLM engine、ONNX、workspace 和 image-token profile 不变 | level2/3 已在 Jetson 构建，3 请求 E2E p50 比 level1 低约 `12.65%/12.75%`，但生成长度从 `59` 变为 `51` tokens 且完整输出改变，功能/数值对齐失败；不能将该信号称为有效 builder 加速 | 先定位 visual engine level2/3 的数值差异和停止条件变化，再以同一输出/质量口径做严格 A/B；补充 Engine Inspector、FMHA kernel 时间、TTFT/E2E、峰值内存和 soak |
| paged KV cache | 默认 level0/level1 仍使用 contiguous KV；独立 P3 candidate 已在固定 v0.9.1 checkout 接入 `maxKVPoolPages=32`、`kv_page_table`、paged pool、prefill gather、KV 写入和 decode XQA `pageList/tokensPerPage` | paged engine 已 build；batch=1/4 prefill/decode exit 0 且 CUDA Graph capture 成功。短 workload 下 batch=4 aggregate decode 为 `90.7376 tok/s`，dense control 为 `92.2316 tok/s`，暂未证明单请求速度收益 | 保留独立 paged engine/provenance；后续完成 dense-vs-paged 数值 parity、page allocate/reclaim 计数、Engine Inspector/Nsight route、HTTP scheduler 和 100 次 soak 后再进入正式性能矩阵；参考 [v0.9.1 release discussion](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions/138) 与 [builder](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/examples/llm/llm_build.cpp) |
| speculative decoding / MTP / EAGLE | Edge-LLM 源码存在 `spec_decode_type`、draft/base engine 和 acceptance 统计接口；当前 health 结果为 `speculative_decoding=false` | 当前运行未启用；源码接口不等于当前 engine 使用 | 独立 draft model/strategy、base+draft engine 及 acceptance trace |
| FP8 KV cache | 导出侧支持 `--kv_cache_quantization fp8`，runtime config 使用 `kv_cache_dtype=fp8`，attention plugin 还要求 `enable_fp8_kv_cache` 与 `[q,k,v]` `qkv_scales`；当前 XQA 源码对 FP8 KV 有 `smVersion>=89` 检查 | 当前 INT4 AWQ provenance 为 KV cache quantization `none`；Jetson Orin Nano 为 SM87，因此当前 XQA 路径不满足该源码条件 | 若目标分支提供其他 FP8-KV 路径，仍需在目标板独立验证质量、显存、decode 和插件日志；不能把导出参数等同于 engine 已启用 |
| FP8 / NVFP4 weight | 当前为 INT4 AWQ；TensorRT 10.3 兼容补丁禁用 FP4 plugin format | 本板端路径未使用 FP8/NVFP4 weight | 更换满足版本与硬件条件的独立 engine |
| KV cache quantization | AWQ provenance 明确 `KV cache quant = none` | KV cache 仍为 runtime 默认精度 | 仅在 runtime 明确支持后单独做质量/容量 A/B |
| weight streaming | 当前 INT4 engine 构建和运行均关闭 | 不应将其解释为 INT4 的加速来源 | FP16 独立 engine 的驻留预算曲线 |

### 8.1 当前 XQA host-cache 候选的板端复验

`0036` 和 `0037` 只改 host 侧 XQA list/function selection，不需要重新构建 engine；复验时
应保持同一 `llm.engine`、同一 profile、同一 prompt contract 和同一冻结样本。control 不
应用补丁，candidate 按顺序应用 `0036`、`0037`，并显式记录两个 patch 的 SHA-256。

板端最低证据应同时包含：

1. 20 个冻结样本的低层 benchmark：TTFT、prefill、decode-only、E2E、output tokens 和
   p50/p90/p99；
2. backend completion、strict JSON、风险准确率和事件 micro-F1；
3. `tegrastats` 的峰值 RAM/swap、GPU 利用率、功耗和温度；
4. Nsight Systems CPU/API trace，确认 `getXQAKernels`/selection lookup 的调用变化，及
   GPU XQA/FMHA 时间没有因 host cache 引入异常；
5. 三次重复后再进行 100 次连续 soak。

只有在输出逐 token 对齐、质量门禁、内存门禁和重复性门禁均通过后，才报告该候选的
TTFT/decode/E2E 变化。若 Nsight 只显示 GPU kernel 时间不变，这是预期的：该候选的
收益应出现在 host dispatch 开销，而不是 XQA kernel 本身。

### 8.2 INT4 GEMM CTA-N=256 的 shape 安全性

`0039-int4-gemm-cta-n256.patch` 只改变 W4A16 GEMM 的编译期 `CTA_N`，但 plugin
dispatcher 会用 `N % kernel::kGemmCtaN` 判断 GEMM/GEMV 路径，因此必须先审计固定模型的
所有 INT4 linear 输出维度。当前 Qwen3-VL-2B 文本 backbone 的相关输出维度为：

| 线性层类别 | 输出维度 N | `N % 256` | 结论 |
| --- | ---: | ---: | --- |
| Q/K/O projection | 2048 | 0 | 可走 CTA-N=256 GEMM |
| K/V projection（GQA） | 1024 | 0 | 可走 CTA-N=256 GEMM |
| MLP gate/up | 6144 | 0 | 可走 CTA-N=256 GEMM |
| MLP down | 2048 | 0 | 可走 CTA-N=256 GEMM |

`151936` 是完整词表大小，但当前正式 INT4 provenance 的 `lm_head` 为 FP16，不属于
该 W4A16 plugin 的 shape 审计范围。若更换模型、打开 GDN/Mamba 分支或量化 `lm_head`，
必须重新收集 Engine Inspector 的 plugin 输入 shape；非对齐 N 应保留 GEMV/fallback，不能
仅凭“编译成功”判定候选安全。该候选目前仍只具备源码和 shape 审计证据，未有板端性能
或数值对齐结果。

W4A16 CTA 候选在提交到板端 build 前，可先运行静态资源审计：

```bash
PYTHONPATH=src python3 scripts/audit_int4_gemm_candidates.py \
  --output reports/tensorrt/int4_gemm_candidate_audit.json
```

该审计复现固定 v0.9.1 的 `kSmemByteSize` 公式，并同时检查输出维度对齐、warp 数和
threads/block；`build_queue_eligible=false` 的组合不得进入编译队列。它不替代真实编译、
数值对齐、Nsight 或 soak test。

### 8.3 INT4 GEMM CTA-M=128 的 prefill 候选

`0041-int4-gemm-cta-m128.patch` 与 `0039` 独立：它保持 `CTA_N=128`，只把 prefill
W4A16 GEMM 的 M 方向 tile 从 `64` 扩展到 `128`。在 `CTA_K=64`、`STAGES=4` 下，
`CTA_M=128/CTA_N=128` 的动态 shared memory 按固定 v0.9.1 源码公式为 `82944` bytes（约
81 KiB，包含末尾的 `CTA_N` scale buffer），低于
Orin Nano 的 99 KiB 上限；它预计减少 M 方向 block 数，但会增加单 block warp 数，必须
通过 Nsight 检查寄存器、occupancy 和实际 W4A16 kernel 时间。该候选只适合 prefill/GEMM
路径，不能用 decode GEMV 的指标替代。

### 8.4 INT4 GEMM CTA-K=128 的 pipeline 候选

`0042-int4-gemm-cta-k128.patch` 保持 `CTA_M=64/CTA_N=128`，只把 K 方向 tile 从
`64` 扩展到 `128`。固定 `STAGES=4` 时，动态 shared memory 估算为
`(64*128 + 128*128/4 + 128) * 4 * sizeof(half) = 99,328` bytes（约 97 KiB），
仍低于源码的 `99*1024` static assertion，但距离上限很近。它可能减少 K 方向迭代、
同步和 pipeline 管理开销，也可能因 shared memory/寄存器压力变慢；必须把构建成功、
kernel launch 成功和实际 kernel 时间分别记录，不能只看 engine 生成成功。

`0041`、`0042`、`0039` 均按单变量独立构建；只有单变量 A/B 通过数值、质量、内存和
重复性门禁后，才考虑组合实验。

### 8.5 INT4 GEMM CTA-N=512 资源拒绝

`0045-int4-gemm-cta-n512.patch` 将 `kGemmCtaN` 从 `128` 提升到 `512`，针对固定
Qwen3-VL-2B INT4 线性层的 `N=1024/2048/6144` 形状做 build-time A/B。虽然这些形状
满足 `N % 512 == 0`，但当前 kernel 的 `CTA_M=64/CTA_K=64/STAGES=4` 资源计算为
`(64*64 + 512*64/4 + 512) * 4 * sizeof(half) = 102400` bytes，超过 `99*1024`
的静态断言，因此 0045 不能进入编译队列。若要继续研究 512 tile，必须重新设计
`CTA_K` 或 pipeline stages，并作为新的 compound candidate 进行资源、数值、kernel、
TTFT、decode、E2E、显存和 soak 验证。

### 8.6 W4A16 candidate 的隔离构建

`0039`、`0041`、`0042` 都是 plugin 的编译期变量，不能只替换运行时环境变量。建议为每个
候选从固定 v0.9.1 revision 创建独立 worktree，单独应用一个 patch、单独编译 plugin，避免
正式 plugin 和候选 plugin 互相覆盖：

下面的示例假定候选 worktree 已先应用正式部署所需的 JetPack/TensorRT 兼容补丁和已验证的
runtime 基础补丁；这些基础补丁必须与正式 level1 checkout 保持同一顺序，不能只应用 CTA
候选 patch 后直接把结果与正式 engine 比较。

```bash
EDGE_BASE=/home/ubuntu/TensorRT-Edge-LLM
EDGE_REV=7f061f21f0a581ba234a1e233c9315b89d8e47d6
CANDIDATE=cta_n256
CANDIDATE_EDGE=/home/ubuntu/TensorRT-Edge-LLM-${CANDIDATE}

git -C "${EDGE_BASE}" worktree add --detach "${CANDIDATE_EDGE}" "${EDGE_REV}"
git -C "${CANDIDATE_EDGE}" apply --check --recount --unidiff-zero \
  /home/ubuntu/JetsonVLM/patches/tensorrt-edge-llm/0039-int4-gemm-cta-n256.patch
git -C "${CANDIDATE_EDGE}" apply --recount --unidiff-zero \
  /home/ubuntu/JetsonVLM/patches/tensorrt-edge-llm/0039-int4-gemm-cta-n256.patch

cd "${CANDIDATE_EDGE}"
mkdir -p build
PYBIND11_DIR=$(/home/ubuntu/JetsonVLM/.venv-jetson/bin/python \
  -m pybind11 --cmakedir)
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=12.6 \
  -DENABLE_CUTE_DSL=ALL \
  -DBUILD_PYTHON_BINDINGS=ON \
  -Dpybind11_DIR="${PYBIND11_DIR}"
cmake --build build --parallel 2
```

`CTA-M=128` 和 `CTA-K=128` 只需将 `CANDIDATE`、patch 文件和输出目录替换为对应候选；
三组必须串行构建。随后使用该 worktree 的 `build/examples/llm/llm_build`、
`build/libNvInfer_edgellm_plugin.so` 构建唯一 engine 目录，并在 runtime metadata 中记录
worktree revision、patch SHA、plugin SHA 和 engine SHA。`0045` 不应执行上述构建流程，资源
审计已将其判定为 compile-time rejection。

### 8.7 视觉 encoder builder level A/B

当前正式视觉 engine 使用 `builder_optimization_level=1`。由于 level0/level1 的 LLM
对比复用了同一个视觉 engine，level1 trace 中约 `57.4 ms` 的
`fmha_v2_flash_attention...S_q_k_v_64...sm87` 仍属于视觉侧路径，不能用 LLM engine
的 level1 结果代表视觉 builder 已经优化。新增的 visual opt2/opt3 flow 只改变视觉
engine 的 builder level：

```bash
PYTHONPATH=src python3 scripts/build_edgellm_vlm_engines.py \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --expected-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
  --onnx-root artifacts/onnx/qwen3_vl_2b_fp16 \
  --engine-root artifacts/engines/qwen3_vl_2b_fp16_visual_opt2 \
  --component visual \
  --workspace-limit-mib 1024 \
  --builder-optimization-level 2 \
  --min-image-tokens 8 \
  --max-image-tokens 2048 \
  --max-image-tokens-per-image 2048 \
  --build-report reports/flows/build_qwen3_vl_2b_fp16_visual_opt2.build.json
```

level3 只需将输出目录改为 `qwen3_vl_2b_fp16_visual_opt3`、build report 改为
`build_qwen3_vl_2b_fp16_visual_opt3.build.json`，并将 optimization level 改为 `3`。
构建完成后，服务端仍使用正式的 INT4 LLM engine，只替换 `--visual-engine-root`，以
隔离视觉 builder 变量。至少应比较 visual engine SHA、build time、Engine Inspector
层/tactic、D64 FMHA kernel 名称和总时延；随后再做相同 20 样本的 TTFT/prefill/E2E、
峰值内存、质量和 100 次 soak。低层 comparator 同时输出 `vision_encode_ms` 与
`model_generate_ms`；若 engine 生成成功但 FMHA kernel/timing 和视觉阶段时延均未变化，应记录
为“builder level 未改变该热点”，不能把 level2/3 自动写成加速结论。

如果复用 TensorRT timing cache，visual level2/3 分别使用
`configs/tensorrt/timing_cache_binding_jetson_orin_nano_visual_level2.json` 和
`timing_cache_binding_jetson_orin_nano_visual_level3.json`；不要使用 LLM sidecar。sidecar
中的 `component=visual`、默认 visual builder profile 和 level 必须与本次命令一致，且 cache
命中只用于比较构建耗时，不能作为推理加速证据。

比较两份视觉 benchmark 时，额外传入 `--changed-engine-component visual`。比较器会要求
两边 runtime metadata 同时包含 LLM/visual engine SHA-256，且只允许 visual SHA 变化、
LLM SHA 和 runtime options 保持不变；否则结果标记为不可比。

构建完成后可直接用 build report 生成视觉 engine provenance，避免再次手工录入 profile：

```bash
PYTHONPATH=src python3 scripts/record_engine_provenance.py \
  --engine artifacts/engines/qwen3_vl_2b_fp16_visual_opt2/visual/visual.engine \
  --build-report reports/flows/build_qwen3_vl_2b_fp16_visual_opt2.build.json \
  --output reports/tensorrt/visual_opt2.engine.provenance.json
```

该入口只接受 `status=succeeded` 且输出路径、文件大小、SHA-256 均与实际 engine 匹配的
build report；因此构建失败或 provenance 指向错误 engine 时会在 benchmark 前直接失败。

### 8.8 词表裁剪候选的离线门禁

词表裁剪会改变 logits 的列数和采样 token 集合，因此先对 Edge-LLM 生成的
`vocab_map.safetensors` 做离线校验，再进行 engine 构建。校验器不依赖 PyTorch，检查
`I32` 一维 map、原始/裁剪后词表大小、严格递增顺序、token id 范围和必需 token（至少
包含 EOS 以及当前 JSON 输出链路需要的特殊 token）：

项目提供一个面向泊车标注 JSONL 的候选 map 生成器。它只读取指定字段，不读取冻结测试
集；默认从 `assessment` 的字符串字段统计 token，自动加入 tokenizer 的 EOS/BOS/PAD，
并以频次排序后用低 id 补齐到目标大小：

```bash
PYTHONPATH=src python3 scripts/build_reduced_vocab_map.py \
  --tokenizer /path/to/Qwen3-VL-2B-Instruct \
  --input-jsonl data/annotations/ps80_reviewed_v1.jsonl \
  --field assessment \
  --target-size 65536 \
  --output-dir artifacts/reduced_vocab/qwen3_vl_v1_65536 \
  --local-files-only
```

当前仓库的 `65536` 独立 flow 使用更严格的联合语料候选：将校准集的完整
`text`（包含冻结工作负载和校准输出）与人工确认标注的 `assessment` 一起统计，输出到
`artifacts/reduced_vocab/qwen3_vl_v1_65536_character_coverage`，并额外保留 tokenizer
单 token 中文/JSON 字符覆盖候选。该目录中的
`selection_report.json` 也必须保留，用于审计输入来源和 tokenizer SHA-256：

```bash
PYTHONPATH=src python3 scripts/build_reduced_vocab_map.py \
  --tokenizer models/Qwen3-VL-2B-Instruct-89644892 \
  --input-jsonl data/processed/calibration/ps16_int4_calibration_v1.jsonl \
  --input-jsonl data/annotations/ps80_human_confirmed_v1.jsonl \
  --field text --field assessment \
  --target-size 65536 \
  --original-vocab-size 151936 \
  --output-dir artifacts/reduced_vocab/qwen3_vl_v1_65536_character_coverage \
  --include-decoded-character-coverage \
  --local-files-only
```

若已有同口径的历史 StudyReport，可在上面的生成命令中追加一个或多个
`--study-report reports/...json`，将成功请求的 `raw_output` 也纳入频次选择；生成器会把
报告路径和 SHA-256 写入 `selection_report.json`。当前对应的增强候选目录为
`qwen3_vl_v1_32768_character_coverage_plus_history` 和
`qwen3_vl_v1_65536_character_coverage_plus_history`，并配有独立的 export/build flow。
它们只用于候选实验，不能使用包含最终 holdout 的报告来替代独立质量评测。

如果目标机已安装 Edge-LLM 的官方参考命令，也可以先用 `input_aware` 方法生成候选，
再使用本项目校验器检查结果；官方脚本的参考数据集不是泊车数据，不能跳过任务语料的
覆盖检查：

```bash
tensorrt-edgellm-reduce-vocab \
  --model_dir /path/to/Qwen3-VL-2B-Instruct \
  --output_dir artifacts/reduced_vocab/qwen3_vl_v1_65536_official \
  --reduced_vocab_size 65536 \
  --method input_aware \
  --max_samples 50000
```

完成 map 校验后，仓库提供了固定 `65536` 词表的独立 export/build flow；它们从现有
`qwen3_vl_2b_int4_awq_n128` checkpoint 开始，保持 `i768/k1024`、workspace `1024 MiB`
和 builder level `1`，不会覆盖正式 engine：

```bash
PYTHONPATH=src python3 scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_int4_awq_reduced_vocab65536.json

PYTHONPATH=src python3 scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_int4_awq_reduced_vocab65536_llm_i768_k1024_opt1.json
```

以上命令默认只打印 readiness；确认 required inputs 和输出目录均正确后，再分别追加
`--execute`。正式 export 命令必须包含 `--reduced-vocab-dir`，而 engine build wrapper 会
在调用 `llm_build` 前要求 source 目录包含 `selection_report.json`，并核对其 map SHA-256
与 token count；同时核对 source map、ONNX export map、metadata 和
`config.json.reduced_vocab_size`。因此不能用一个脱离选择语料审计的孤立 map 替代构建输入。
构建 flow 还显式传入 `--reduced-vocab-required-token-id 151645`，将 Qwen3-VL 的 EOS
纳入构建前门禁；如果更换模型或 tokenizer，必须重新解析并更新该 token id。
构建完成后应使用带 `--reduced-vocab-dir` 的 provenance 命令记录 map、metadata 和
`selection_report.json` 的大小与 SHA-256，再进行完整的严格 JSON、质量、decode、E2E、
显存和 soak A/B。后续 provenance 校验会同时检查三份文件，避免 map 来源在运行阶段丢失。

对于纳入历史成功输出的 `..._history` 候选，历史输出只用于 map 选择；对应 build flow
额外绑定 `reduced_vocab_*_ps20_character_coverage_plus_history.json`，要求冻结
`ps20_pilot_v1` holdout 的 token coverage 和 sample coverage 均为 `1.0`。这样可以在
启动 `llm_build` 前阻止“只覆盖历史输出、但无法表达独立样本”的候选。

当前候选链路分为“导出环境”和“Jetson engine 构建环境”两步：`export_model.py` 在已安装
固定 Edge-LLM Python exporter、量化 checkpoint 和 tokenizer 的导出环境执行；完成后将
`artifacts/onnx/...` 整目录及 flow record 复制到 Jetson，再在 Jetson 执行
`build_engine.py`。两组 flow 都应先去掉命令末尾的 `--execute` 做 readiness 检查，确认
路径和版本无误后再执行：

```bash
PYTHONPATH=src python3 scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_int4_awq_reduced_vocab65536_history.json \
  --execute
PYTHONPATH=src python3 scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_int4_awq_reduced_vocab65536_history_llm_i768_k1024_opt1.json \
  --execute
```

完成 `65536` 后，将两个配置文件名中的 `65536` 替换为 `32768` 进行第二组实验；两组
均使用独立 engine 目录，不覆盖 level0 或 level1。导出环境和 Jetson 的 Edge-LLM
revision 必须一致，不能用其他版本生成的 ONNX 直接构建。

该启发式 map 的 `selection_report.json` 会记录观测 token 数、补齐 token 数、输入文件和
SHA-256。它用于生成可构建的实验候选，不代表覆盖了模型所有可能的输出；生产评测必须
使用未参与 map 选择的冻结样本，并与完整词表 engine 做严格 JSON、风险质量和生成结果对齐。

除参考文本外，还可以把历史 StudyReport 中成功请求的 `raw_output` 纳入覆盖检查。该
检查只验证这些历史输出能否由候选 map 表达，不把失败请求计为“已覆盖”，也不替代新
engine 的严格 JSON 和质量评测：

```bash
PYTHONPATH=src python3 scripts/evaluate_reduced_vocab_coverage.py \
  --tokenizer models/Qwen3-VL-2B-Instruct-89644892 \
  --map artifacts/reduced_vocab/qwen3_vl_v1_65536_character_coverage/vocab_map.safetensors \
  --metadata artifacts/reduced_vocab/qwen3_vl_v1_65536_character_coverage/reduced_vocab.json \
  --original-vocab-size 151936 \
  --required-token-id 151645 \
  --study-report reports/jetson-metrics-20260830/int4_opt0_report.json \
  --study-report reports/jetson-metrics-20260830/int4_opt1_report.json \
  --output reports/tensorrt/reduced_vocab_65536_historical_output_coverage.json \
  --local-files-only
```

报告会单独记录 `study_reports`、有效 `raw_output` 样本数和缺失 token；只有参考文本、
历史输出和冻结 holdout 均通过后，才进入板端 reduced-vocab engine A/B。

```bash
PYTHONPATH=src python3 scripts/validate_reduced_vocab.py \
  --map artifacts/reduced_vocab/qwen3_vl_v1/vocab_map.safetensors \
  --metadata artifacts/reduced_vocab/qwen3_vl_v1/reduced_vocab.json \
  --original-vocab-size 151936 \
  --required-token-id 151645 \
  --output reports/tensorrt/reduced_vocab_validation.json
```

返回码 `0` 表示 map 通过，`2` 表示发现门禁问题。该工具只验证映射文件的结构和覆盖
约束，不证明裁剪后的模型质量、严格 JSON、`lm_head` tactic 或 decode 性能；这些仍需在
独立的 `65536`/`32768` engine 上完成板端 A/B、质量评测和 soak test。

如果候选还会量化 packed INT4 `lm_head`，在同一命令中追加
`--packed-int4-lm-head`；校验器会提前检查 group size 128 对齐。

在 engine 构建前，还应使用独立的 holdout coverage evaluator 检查参考输出 token 是否均
在 map 中。它不读取或修改 map 选择语料；当前启用联合 CJK/JSON character 与历史成功
输出 coverage 的 `65536` map 在冻结 `ps20_pilot_v1` 上达到 `632/632` token 覆盖、`20/20`
样本无缺失，已通过 reference-token coverage 门禁：

```bash
PYTHONPATH=src python3 scripts/evaluate_reduced_vocab_coverage.py \
  --tokenizer models/Qwen3-VL-2B-Instruct-89644892 \
  --map artifacts/reduced_vocab/qwen3_vl_v1_65536_character_coverage_plus_history/vocab_map.safetensors \
  --metadata artifacts/reduced_vocab/qwen3_vl_v1_65536_character_coverage_plus_history/reduced_vocab.json \
  --original-vocab-size 151936 \
  --input-jsonl data/annotations/ps20_pilot_v1.jsonl \
  --field assessment \
  --required-token-id 151645 \
  --output reports/tensorrt/reduced_vocab_65536_ps20_character_coverage_plus_history.json \
  --local-files-only
```

返回非零表示至少一个参考样本含缺失 token；这不是模型质量失败，但应在构建和性能 A/B
之前处理或显式记录风险。当前通过 coverage 只证明参考 token 可表达，不证明严格 JSON、
风险准确率、事件 micro-F1 或推理性能。

在对应 coverage report 通过后，`65536` reduced vocabulary 与 packed INT4 `lm_head` 的组合候选
使用独立 flow 执行，避免覆盖正式 engine；`32768` 只需将 flow 文件名中的词表规模替换为
`32768`：

```bash
PYTHONPATH=src python3 scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_int4_awq_lm_head_reduced_vocab65536_history.json
PYTHONPATH=src python3 scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_int4_awq_lm_head_reduced_vocab65536_history_llm_i768_k1024_opt1.json
```

该 flow 同时要求 `calibration_provenance.json`、reduced-vocab 的
`selection_report.json` 和上述 coverage report；如果 exporter 不接受 reduced-vocab 与
packed INT4 `lm_head` 的组合，应保留失败日志并停止在候选阶段，不将其当作模型或性能结论。

`32768` map 也使用同一联合 coverage 策略；其冻结 `ps20_pilot_v1` 检查同样为
`632/632` token、`20/20` 样本无缺失：

```bash
PYTHONPATH=src python3 scripts/evaluate_reduced_vocab_coverage.py \
  --tokenizer models/Qwen3-VL-2B-Instruct-89644892 \
  --map artifacts/reduced_vocab/qwen3_vl_v1_32768_character_coverage_plus_history/vocab_map.safetensors \
  --metadata artifacts/reduced_vocab/qwen3_vl_v1_32768_character_coverage_plus_history/reduced_vocab.json \
  --original-vocab-size 151936 \
  --input-jsonl data/annotations/ps20_pilot_v1.jsonl \
  --field assessment \
  --required-token-id 151645 \
  --output reports/tensorrt/reduced_vocab_32768_ps20_character_coverage_plus_history.json \
  --local-files-only
```

返回非零表示至少一个参考样本含缺失 token；这不是模型质量失败，但应在构建和性能 A/B
之前处理或显式记录风险。两个 map 仍需分别完成 ONNX/engine 构建、严格 JSON、质量、
decode/E2E、内存和 soak A/B。
仓库提供 `scripts/summarize_tensorrt_nsight_trace.py` 将 Nsight Systems 导出的
`cuda_gpu_trace.csv` 按 W4A16、FMHA/XQA、sampling、embedding 和 memcpy 分类，保存每类的
count、total/p50/p90/max 时延以及 top kernels。该摘要只做 GPU trace 事件聚合，不替代
匹配 workload 的 TTFT/decode/E2E，也不能单独证明算子融合或 layer 对应关系：

```bash
PYTHONPATH=src python3 scripts/summarize_tensorrt_nsight_trace.py \
  --trace reports/jetson-tensorrt-revalidation/int4_opt1_nsight_stats_cuda_gpu_trace.csv \
  --label int4_opt1 \
  --output reports/jetson-tensorrt-revalidation/int4_opt1_nsight_kernel_summary.json
```

两份摘要可用 `compare_tensorrt_nsight_summaries.py` 生成候选相对基线的绝对差值和百分比
差值。差值定义为 `candidate - baseline`，时延为负表示候选更快；该报告适合和同一批次
的 TTFT/decode/E2E 报告并列阅读：

```bash
PYTHONPATH=src python3 scripts/compare_tensorrt_nsight_summaries.py \
  --baseline reports/jetson-tensorrt-revalidation/int4_opt0_nsight_kernel_summary.json \
  --candidate reports/jetson-tensorrt-revalidation/int4_opt1_nsight_kernel_summary.json \
  --output reports/jetson-tensorrt-revalidation/int4_opt0_vs_opt1_nsight_comparison.json
```

该工具仅比较已导出的 GPU trace 摘要；它不会把 kernel 差值自动解释为算子融合、KV cache
优化或端到端加速，也不会替代质量、内存、功耗和 soak 门禁。

### 当前视觉 builder level 诊断边界

在隔离 worktree 中为 visual builder level2 设置 `EDGELLM_PROFILING_VERBOSITY=detailed` 后，
TensorRT Engine Inspector 可以直接看到 `fusion` 层、`kgen` 层、FP16 GEMM tactic、
multi-stream 的 signal/wait 节点以及 `ViTAttentionPlugin` PluginV3 边界。这些是“引擎中
实际选择了什么”的证据：TensorRT 对多个线性层/elementwise 节点做了图级融合或 codegen，
视觉 attention 仍由 Edge-LLM plugin 承担。Inspector 不会自动证明端到端收益，且 plugin
本身没有暴露 tactic metadata。

同一图片、prompt 和 INT4 LLM 的 level1/level2 Nsight screen 显示，D64 FMHA 总时延约
`169.01/171.10 ms`，W4A16 总时延约 `789.97/789.31 ms`；level2 的 E2E 降低与输出从
`59` tokens 变为 `51` tokens 同时发生，因此该信号不能写成 visual builder 或 attention
kernel 加速。原始 Inspector 日志和诊断 engine provenance 分别见
`reports/jetson-tensorrt-revalidation/visual_detailed_opt2_trtexec.log` 与
`reports/jetson-tensorrt-revalidation/visual_detailed_profiling_evidence.json`。

针对 Nsight 中的 INT4 decode 热点，已在同一隔离 plugin 中筛选 `N_PER_BLOCK=4`。该路径
只对 `M=1` 且输出维度满足无尾块索引的 shape 开启；编译产物中可见
`gemv_kernel<4,1,256,128>`，说明候选确实进入了 device code。板端 3 次 smoke 的 HTTP
状态均为 200，但每次都输出 256 tokens 的重复/乱码，严格 JSON 为 0/3，因此不测 latency
也不晋级。该结果说明 decode kernel 的 block 数减少不能脱离数值/协议门禁单独评价；失败
日志和 plugin provenance 见 `reports/jetson-tensorrt-revalidation/int4_gemv_nperblock4_candidate_failure.json`。

随后对同一 GEMV kernel 做 block-size screen：保留 `N_PER_BLOCK=2`，只通过
`EDGELLM_INT4_GEMV_BLOCK_SIZE=128` 选择 `gemv_kernel<2,1,128,128>`，默认 256-thread
路径保持不变。3 次 screen 均得到相同 59-token 输出，E2E p50 为 `8044.16 ms`；与已有
level1 control screen 的 `8085.77 ms` 相比约 `-0.52%`，但 control 没有同轮重跑，且尚未
采集 TTFT/decode、内存、功耗和严格质量指标，所以暂不称为加速。完整结果见
`reports/jetson-tensorrt-revalidation/int4_gemv_block128_screen.json`。

### 视觉 profile 收紧与内存边界

对当前实际视觉长度约 `1444` 的 workload，独立构建了 `max_image_tokens=1536` 的 level1
visual engine。runtime 日志显示 shared execution context memory 从 `423,624,704` bytes
降到 `317,718,528` bytes，约减少 `25%`，因此该 profile 能在当前板端连续内存压力下成功
加载；但输出从 level1 的 `59` tokens/`obstacle` 变为 `51` tokens/`low_speed`。这说明
profile 上限不仅影响内存，也可能改变 TensorRT engine 的 shape/tactic 数值路径；在严格
输出对齐前，不能将其作为性能候选晋级。证据见
`reports/jetson-tensorrt-revalidation/visual_max1536_profile_screen.json`。

继续把上限放宽到 `1792` 后，activation memory 为 `370,671,616` bytes，约比正式 level1
低 `12.5%`，但输出仍固定为 `low_speed/51 tokens`，没有恢复 level1 的 `obstacle/59 tokens`。
因此当前问题不是单纯的 1536 边界；profile 上限变化会影响 engine 的 shape/tactic 数值
路径，必须先通过功能/质量对齐再讨论性能收益。结果见
`reports/jetson-tensorrt-revalidation/visual_max1792_profile_screen.json`。

### INT4 GEMV block size 的隔离 decode A/B

视觉 runner 会同时引入视觉编码、HTTP 和输出长度因素，因此进一步使用 `llm_bench` 对正式
`i768/k1024` INT4 LLM engine 做 decode-only matched A/B。候选通过
`EDGELLM_INT4_GEMV_BLOCK_SIZE=128` 选择 `gemv_kernel<2,1,128,128>`，control 使用正式
256-thread 路径；batch=1、warm-up=10、50 次 layer profile 在 `pastKVLen=256/768/1024`
上的候选/ control 总时延分别为：

| pastKVLen | candidate 128 threads | control 256 threads | delta |
| ---: | ---: | ---: | ---: |
| 256 | 128.4026 ms | 128.6130 ms | -0.16% |
| 768 | 129.2875 ms | 129.5464 ms | -0.20% |
| 1024 | 129.7568 ms | 129.9656 ms | -0.16% |

在 `pastKVLen=768` 下使用 CUDA Graph 进行 100 次 decode，128-thread 为 `126.4849 ms`，
256-thread 为 `126.6739 ms`，差异仅 `-0.15%`；同一 patch 的 512-thread 分支为
`132.1313 ms`，比 control 慢 `4.31%`。因此 128-thread 结果属于测量噪声范围，未达到
项目的重复性能晋级门槛，正式 256-thread 路径继续保留。layer profile 将主导的
`node_linear` 归入 `kgen_other`，不能将该行直接解释为 GEMV kernel 时间；完整报告、CSV
和原始日志见 `reports/jetson-tensorrt-revalidation/int4_gemv_block_size_llm_bench.json`。

### INT4 LLM builder level2/level3 与 LM-head 定位

在固定 INT4 ONNX、`maxInputLen=768`、`maxKVCacheCapacity=1024`、workspace `1024 MiB`
下，level2 与 level3 的 LLM engine 构建均在 tactic 选择阶段失败：TensorRT 尝试申请约
`622,329,856` bytes 时返回 `CUDA error 2`，最终没有找到
`ForeignNode[n1...node__to_copy_226]` 的实现。该结果只说明当前板端构建资源不足，不能
把它解释为 level2/3 的推理性能结论；正式 level1 engine 未覆盖。

使用隔离 worktree 的 patched builder 生成 level1 detailed engine 后，Inspector 给出实际
LM-head 边界：`node_linear [profile 1]_myl1687_3` 是
`[1,1,2048] x [1,2048,151936] -> [1,1,151936]` 的 FP16 `gemm`，其 tactic 为
`sm80_xmma_gemm_f16f16_f16f16_f16_nn_n_tilesize128x128x32_stage4_warpsize2x2x1_tensor16x8x16`。
因此当前 decode 主热点不是前一轮尝试的 INT4 GEMV，而是大词表 LM-head GEMM；下一轮应
围绕 LM-head tactic/shape、输出投影专用 kernel 和 CPU launch/synchronization 做严格 A/B。
完整 build failure 与 Inspector 证据见
`reports/jetson-tensorrt-revalidation/int4_llm_builder_level2_level3_screen.json`，详细
layer metadata 见 `reports/jetson-tensorrt-revalidation/int4_llm_detailed_candidate_1412_layerinfo.json`。

### INT4 level0/level1 decode repeatability

修正 engine 目录为末级 `/llm` 并显式设置正式 Edge-LLM plugin 后，使用同一 `llm_bench`
命令完成 level0/level1 各 3 次重复：batch=1、`pastKVLen=768`、warm-up=10、20 次
decode、OSL=1、CUDA Graph、seed=0。level0 三次为 `126.6943/126.6258/126.7423 ms`，
均值 `126.6875 ms`；level1 三次为 `27.4520/27.3945/27.3921 ms`，均值 `27.4129 ms`。
因此低层 decode 的 level1/level0 平均加速为 `4.6215x`，平均延迟下降 `78.3618%`。

该结果验证了 level0 到 level1 的低层收益稳定性，但不改变 level1 的候选身份：完整
VLM 的 TTFT/E2E、严格 JSON、风险准确率、事件 micro-F1、峰值内存和 100 次 soak 仍需
单独满足晋级门槛。六个进程均在 benchmark 成功后出现 TensorRT
`IRuntime::~IRuntime` 生命周期 API-usage warning，已作为运行时工程问题记录，不能
忽略为普通 stderr。逐次日志、engine SHA 和环境 provenance 见
`reports/jetson-tensorrt-revalidation/int4_level0_level1_decode20_repeats_20260909.json`。

### llm_bench runtime 生命周期修复候选

只读源码检查确认，`llm_bench` 的 `loadStandaloneEngine()` 在局部作用域创建
`IRuntime`，却把反序列化得到的 `ICudaEngine` 返回给调用方；函数返回时 runtime 先于
engine 析构，触发 TensorRT 的 `mEngineCounter.use_count() == 1` API-usage warning。该
engine 只用于 metadata 提取，不是主推理 executor，但 warning 会污染所有低层 benchmark
日志并应在晋级前清除。

`patches/tensorrt-edge-llm/0050-llm-bench-keep-standalone-runtime-alive.patch` 将 runtime
所有权提升到 `llm_bench` main，并保持 engine/runtime 的正确销毁顺序。该 patch 尚未复制
到板端或重新编译，正式 checkout 和当前性能结论均不受影响。

### INT4 level0/level1 low-level prefill repeatability

固定 `inputLen=768`、`reuseKVLen=0`、batch=1、warm-up=10、20 次重复，对正式 level0
和候选 level1 各完成 3 次 prefill。level0 三次为 `526.9898/526.6765/526.9469 ms`，
均值 `526.8711 ms`；level1 三次为 `426.8245/426.8257/426.5893 ms`，均值
`426.7465 ms`，低层 prefill 平均加速 `1.2346x`，延迟下降约 `19.04%`。

这是 engine-level 的 prefill 证据，不等同于客户端 TTFT，也不替代完整 VLM 的 TTFT/E2E、
质量、内存和 soak 晋级门禁。逐次日志和 provenance 见
`reports/jetson-tensorrt-revalidation/int4_level0_level1_prefill768_decode20_20260909.json`。

### level1 low-level soak

对 level1 `i768/k1024` engine 在 `pastKVLen=768`、batch=1、warm-up=10、CUDA Graph 条件下
连续执行 1000 次 decode。进程退出码为 0，Graph capture 成功，无 runtime failure、崩溃或
超时；平均 decode E2E `27.3821 ms`，报告标准差 `0.0197 ms`，E2E 吞吐 `36.5202 tok/s`。
当前 `llm_bench` CSV 只保存聚合值，因此没有伪造逐 iteration 的 p50/p90/p99。由于仍有
standalone metadata engine 的 runtime 生命周期 warning，该结果是低层长稳证据，不是最终
生产晋级证明。完整状态见
`reports/jetson-tensorrt-revalidation/int4_level1_decode1000_soak_20260909.json`。

### k768/k1024 layer profile boundary

在 `pastKVLen=767`、batch=1、warm-up=10、20 次非 Graph decode 条件下，k768/k1024 可见
非零 layer timing 总和为 `30.0687/30.1159 ms`，最终 `node_linear` LM-head GEMM 为
`10.2523/10.2497 ms`。两者没有实质差异；当前 profiler 也没有将 attention/XQA plugin
执行暴露为可独立归因的时间行，因此不把该结果解释为 attention kernel 加速。完整 profile
证据见 `reports/jetson-tensorrt-revalidation/int4_level1_kv_capacity_profile20_20260909.json`。

### 候选 plugin 的非 CUDA Graph 调度 A/B

为验证 XQA kernel list/function selection cache 在 CPU launch 路径上的效果，固定同一个
level1 `i768/k1024` engine、`pastKVLen=768`、batch=1、warm-up=10、20 次 decode、OSL=1、
seed=0，对正式 plugin 和隔离候选 plugin 各进行了 3 次 `--noCudaGraph` 测量。候选 plugin
包含已有的 XQA cache 及相关候选源码改动，因此这是 plugin-stack A/B，不是单一 patch 的
因果归因。

候选结果为 `29.5547/29.5476/29.5509 ms`，均值 `29.5511 ms`；control 为
`30.0580/29.5455/29.5604 ms`，均值 `29.7213 ms`。虽然均值下降 `0.5728%`，但中位数
只下降 `0.0321%`，control 的首轮存在 `30.0580 ms` 波动，不能视为稳定收益。候选 plugin
不晋级，正式 plugin 和默认 engine 均未改变。完整 CSV、日志和 provenance 见
`reports/jetson-tensorrt-revalidation/int4_candidate_plugin_nograph_ab_20260909.json`。

### Nsight Systems 对 decode 热点的重新定位

在正式 level1 engine、`pastKVLen=768`、batch=1、warm-up=3、5 次非 Graph decode 上采集
Nsight Systems trace。GPU kernel 汇总显示：

| kernel | trace GPU time share | total time | 作用 |
| --- | ---: | ---: | --- |
| `gemv_kernel<2,1,256,128>` | 52.3% | 141.149952 ms | INT4 decode GEMV |
| `trt_ampere_h16816gemm_128x64_ldg8_tn_v1` | 34.2% | 92.243072 ms | level1 FP16 lm-head GEMM |
| `kernel_mha` | 6.9% | 18.640224 ms | decode attention |

因此当前 batch=1 decode 的首要 GPU 优化对象是 INT4 GEMV，其次是已经由 level1 tactic
改善过的 FP16 lm-head；attention 在该 trace 中不是第一热点。`CTA_N/CTA_K` 只会影响
`M>1` 的 W4A16 GEMM 路径，应放到 prefill/批处理实验中评估。Nsight 的 CUDA API 汇总含
初始化和 warm-up，只作为调度线索，不替代 decode E2E 计时。完整 trace、stats 和口径见
`reports/jetson-tensorrt-revalidation/int4_level1_formal_nograph_nsys_20260909.json`。

### INT4 GEMV block-size 192/320 screen

在隔离 plugin 中增加 `BlockSize=192/320` 的编译期实例，保留 `NPerBlock=2` 和原有量化
索引。固定 level1 `i768/k1024` engine、`pastKVLen=768`、batch=1、warm-up=10、20 次
decode，各候选在 Graph 和非 Graph 路径完成 3 次重复：

| candidate | non-Graph mean | delta vs control | Graph mean | delta vs control |
| --- | ---: | ---: | ---: | ---: |
| 192 threads | 30.1138 ms | +1.32% | 27.9317 ms | +1.89% |
| 320 threads | 31.8464 ms | +7.15% | 29.6242 ms | +8.07% |

control 为正式 256-thread 路径。两个候选都拒绝，默认 256 不变；因此当前应停止无目的的
block-size 扫描，转向能减少反量化/访存或 kernel launch 数的融合方案。该 screen 只覆盖
低层 decode，不替代严格 JSON、领域质量、内存和 soak 门禁。完整证据见
`reports/jetson-tensorrt-revalidation/int4_gemv_block192_320_ab_20260909.json`。

### INT4 GEMV read-only cache annotations

Nsight 定位的 `gemv_kernel<2,1,256,128>` 保持原有线程配置，新增候选 patch 仅将权重、scale
和 activation 的只读加载标注为 `__ldg`。候选与 control 使用同一隔离 candidate stack，
并通过反转/应用单个 patch 保持因果边界；两者 batch=1 资源均为 40 registers、0 local
memory。

固定 level1 `i768/k1024` engine、`pastKVLen=768`、batch=1、warm-up=10、20 次 decode、
OSL=1、seed=0，结果如下：

| path | control mean | candidate mean | latency delta |
| --- | ---: | ---: | ---: |
| non-Graph | 29.565533 ms | 29.515100 ms | -0.170582% |
| CUDA Graph | 27.325133 ms | 27.335100 ms | +0.036474% |

非 Graph 的微小下降没有在 Graph 路径复现，不能作为稳定收益；patch 0052 拒绝，正式 plugin
和默认 engine 保持不变。该实验只覆盖低层 decode，不包含严格 JSON、领域质量、完整 VLM
TTFT、内存或长时 soak。下一步优先评估能减少反量化与中间访存的融合 GEMV，以及结合
`lm_head` shape/tactic 的专用优化。完整 provenance、重复 CSV、原始日志和 build log 见
`reports/jetson-tensorrt-revalidation/int4_gemv_readonly_cache_ab_20260909.json`。

### INT4 W4A16 GEMM CTA_N=256 prefill A/B

此前 `Nsight` 只把 `CTA_N/CTA_K` 归入 `M>1` 的 W4A16 GEMM 路径，因此单独在 prefill
口径验证 `CTA_N=256`。候选仅修改编译期 `CTA_N`，`CTA_M/CTA_K`、stages、权重/scale 布局、
反量化算术、engine 和 workload 均保持不变；control 与 candidate 在同一隔离 stack 中
通过应用/反转 patch 物化。

固定 level1 `i768/k1024` engine、`inputLen=768`、`reuseKVLen=0`、batch=1、warm-up=10、
20 次 prefill，三次重复结果为：

| path | repeats | mean | median | E2E tokens/s |
| --- | --- | ---: | ---: | ---: |
| CTA_N=128 control | 426.7046/426.8751/426.7125 ms | 426.764067 ms | 426.7125 ms | 1799.589 |
| CTA_N=256 candidate | 500.3264/500.2616/500.3736 ms | 500.320533 ms | 500.3264 ms | 1535.016 |

CTA_N=256 比 control 慢 `17.235862%`，因此拒绝该候选；这也说明更大的 N tile 在当前
Orin Nano 与 Qwen3-VL-2B prefill shape 上没有带来收益，不能仅凭 shared-memory 可行就
推断性能提升。该实验只覆盖低层 prefill，不包含完整 VLM TTFT、严格 JSON、领域质量、
内存和 soak；完整 provenance、CSV、原始日志和 build log 见
`reports/jetson-tensorrt-revalidation/int4_gemm_cta_n256_prefill_ab_20260909.json`。

### INT4 W4A16 GEMM CTA_M=128 prefill A/B

在 `CTA_N=256` 候选被拒绝后，继续隔离验证 prefill 的 M 方向 tile。候选仅将
`CTA_M=64` 改为 `128`，固定 `CTA_N=128`、`CTA_K=64`、stages、权重/scale 布局、反量化
算术、engine 和 workload；control 与 candidate 在同一隔离 stack 中通过应用/反转 patch
物化。

固定 level1 `i768/k1024` engine、`inputLen=768`、`reuseKVLen=0`、batch=1、warm-up=10、
20 次 prefill，三次重复结果为：

| path | repeats | mean | median | E2E tokens/s |
| --- | --- | ---: | ---: | ---: |
| CTA_M=64 control | 426.7801/426.7598/426.6508 ms | 426.730233 ms | 426.7598 ms | 1799.732 |
| CTA_M=128 candidate | 497.8697/497.9312/497.8521 ms | 497.884333 ms | 497.8697 ms | 1542.527 |

CTA_M=128 比 control 慢 `16.674258%`，因此拒绝该候选；当前 shape 下更大的 M tile 没有
带来收益。该实验只覆盖低层 prefill，不包含完整 VLM TTFT、严格 JSON、领域质量、内存和
soak；完整 provenance、CSV、原始日志和 build log 见
`reports/jetson-tensorrt-revalidation/int4_gemm_cta_m128_prefill_ab_20260909.json`。

### INT4 GEMV `__restrict__` pointer A/B

在 GEMV block-size 与只读加载标注均未形成稳定收益后，验证一个不改变数据布局和数学语义的
编译器候选：为 `gemv_kernel<2,1,256,128>` 的 inputs、weight、scales、outputs 增加
`__restrict__`。control 与 candidate 使用同一隔离 stack，仅应用/反转该 patch。

固定 level1 `i768/k1024` engine、`pastKVLen=768`、batch=1、warm-up=10、20 次 decode、
OSL=1、seed=0，结果如下：

| path | control mean | candidate mean | latency delta |
| --- | ---: | ---: | ---: |
| non-Graph | 29.526367 ms | 29.503667 ms | -0.076880% |
| CUDA Graph | 27.323733 ms | 27.477100 ms | +0.561295% |

non-Graph 的下降低于测量与晋级门槛，且没有在 CUDA Graph 路径复现；patch 0053 拒绝，默认
256-thread GEMV 保持不变。该实验只覆盖低层 decode，不包含完整 VLM TTFT、严格 JSON、领域
质量、内存或 soak。下一步应停止编译器提示级微调，转向减少反量化/中间访存的融合 GEMV，
或围绕大词表 `lm_head` GEMM 做专用 tactic/算子归因。完整证据见
`reports/jetson-tensorrt-revalidation/int4_gemv_restrict_pointers_ab_20260909.json`。
