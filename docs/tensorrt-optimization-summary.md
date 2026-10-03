# TensorRT Edge-LLM 优化阶段速览

本文只记录近期 TensorRT Edge-LLM 性能优化的固定条件、已核验结果和候选结论。完整命令、
实验过程、provenance 字段和原始证据索引见 [完整实验记录](tensorrt-optimization.md)。

## 当前状态

当前正式参考为 TensorRT Edge-LLM INT4 level 0 engine；level 1 为已完成性能复验的候选。
本轮优化变量集中在 builder tactic、CUDA Graph、INT4 GEMV/GEMM tile、`lm_head` GEMM 和
低层 benchmark，不把量化、LoRA、数据标注或 workload 作为变量。

## P0-P3 执行顺序与验收边界

后续工作按板端可验证的性能链路排序，配置快照见
[`jetson_orin_nano_priority_v1.json`](../configs/tensorrt/jetson_orin_nano_priority_v1.json)。
投机解码本轮明确不纳入范围。

| 优先级 | 方向 | 当前实现 | 板端验收边界 |
|---|---|---|---|
| P0 | C++ direct runtime benchmark | [`run_edgellm_direct_benchmark.py`](../scripts/run_edgellm_direct_benchmark.py) 已实现命令、CSV、日志和 JSONL 归档；已在板端复验 | 同一 engine/workload 下分别采集 prefill/decode；不把 HTTP RTT 或 prefill 伪装成 TTFT |
| P1 | `lm_head` / INT4 GEMV/GEMM | 已完成 Engine Inspector、Nsight 热点复核；候选 A/B 仍保留为隔离实验 | 候选必须完成 3 次低层 A/B、完整 VLM 和 soak；当前热点仍优先是大词表 FP16 `lm_head` GEMM 和 INT4 GEMV |
| P2 | dynamic batching / continuous batching | 已修复 active-batch eviction、vanilla sampling indices 和空 batch compact；新增单 worker bounded native microbatch scheduler，统一流式/非流式 route；batch=1/4 低层 smoke 均 exit code 0 | c=2、c=4 各 60/60 成功，c=4 100 请求 soak 100/100 成功；aggregate output tok/s 分别为 `27.5115/41.0209/40.2069`；已补 telemetry，但因缺少同口径 baseline 暂不默认化 |
| P3 | paged KV | 已完成最小 builder/plugin/runtime backport：`maxKVPoolPages`、page pool、`kv_page_table`、prefill gather、decode XQA page-list 和 KV 写入接线 | 独立 paged engine 已 build；batch=1/4 prefill/decode 均 exit 0 且 CUDA Graph capture 成功；数值 parity、HTTP paged path、Nsight route 和 100 次 soak 仍未完成，不替换默认 engine |

P0 runner 的输出 JSONL 可以直接交给 `scripts/benchmark_edgellm.py` 汇总；它同时写出
`.metadata.json`，保存每次命令、CSV 路径、完整日志路径、退出码和 warning。`llm_bench`
当前 CSV 是每个进程的聚合结果，因此仍不生成不存在的逐 iteration p50/p90/p99。
P2 sweep 只负责复用现有 HTTP benchmark，不把独立连接的吞吐自动解释为 continuous batching。

## 本轮 P0-P3 执行结果

| 优先级 | 已执行内容 | 实测结果 | 当前结论 |
|---|---|---|---|
| P0 | level0/level1，各自 prefill/decode、3 次重复，`i768/k1024`、batch=1、CUDA Graph | level1 相对 level0：prefill `526.8768 -> 426.7482 ms`，降低 `19.0042%`；decode `126.7803 -> 27.4670 ms/token`，加速 `4.6157x` | direct C++ runtime benchmark 已完成；结果已归档 |
| P1 | level1 non-Graph Nsight + Engine Inspector | `gemv_kernel` 占 kernel time `52.12%`，FP16 `lm_head` GEMM 占 `34.43%`，`kernel_mha` 占 `6.88%`；level1 fresh Inspector 未输出可解析 tactic 字段 | 热点已复核；没有新的稳定 GEMV/GEMM 候选晋级 |
| P2 | 修复 max batch=4 engine 的 active-batch eviction，并增加 HTTP native microbatch scheduler；重编译 runtime/plugin 后复验 batch=1/4 与并发 | 低层 batch=1 `29.4693 ms`；batch=4 `43.7650 ms`，按 4 条序列折算 aggregate 约 `91.4 tok/s`；HTTP c=2 `60/60`、c=4 `60/60`、soak `100/100`，无 HTTP500/slot/graph race | 请求级 dynamic batching 已完成功能复验；`27.5115/41.0209/40.2069 tok/s` 是 HTTP aggregate 指标，不能替代低层 decode latency；已采集资源事实，但缺同口径 baseline，暂不默认化 |
| P3 | 修复 ONNX dense-cache shape、AttentionPlugin binding、KV 写入维度和 page-table/runtime 接线；构建独立 `i768/k1024/level1/maxBatch4/maxKVPoolPages32` engine | 新 engine 做 3×20：paged batch=1 decode 均值 `857.7339 ms`；batch=4 decode 均值 `1342.0580 ms`，按 4 条序列折算 aggregate `92.4076 tok/s`；prefill batch=1/4 均值 `428.3120/1655.6338 ms`；batch=1/4 均 exit 0、Graph capture 成功；C++ batch=4 输出 smoke 4/4 成功；新 engine telemetry RAM 峰值 `5197 MB`、SWAP `1569 MB`、GR3D `99%`、`tj` `55.718 C`、VDD_IN `11588 mW` | paged 低层可执行性和实际 token 输出已核验；dense control 当前为不同 iteration 数的回归数据，不能宣称微小延迟差异为加速；等待 parity、HTTP paged scheduler、Nsight 和 soak 门禁 |

P2 的 batch=4 engine 构建峰值为 GPU allocator `1617 MiB`、CPU allocator `5496 MiB`，engine
SHA-256 为 `2502e35bf979dba95a2158a90b79d09551ea2a3638d423ef4207fdc5b154833a`。低层
aggregate throughput 是一个 batch 内 4 条序列的总输出 token 除以 batch E2E 时间；HTTP
aggregate throughput 则是多个请求的总输出 token 除以 steady-state wall clock，两者都不等同
于单请求 decode latency。P1/P2 原始证据位于 `reports/tensorrt-priority/p1/`
和 `reports/tensorrt-priority/p2/`。一次伴随 `tegrastats` 的 c=4 运行记录 RAM
`5636--5656/7619 MB`、SWAP `1555/3810 MB`、GR3D 峰值 `99%`、温度峰值 `58.187 C`、
VDD_IN 峰值 `13680 mW`；因没有同口径 baseline telemetry，只作为资源事实归档。

板端执行入口示例（路径按板端实际安装位置替换）：

```bash
# llm_bench 需要显式加载 Edge-LLM plugin
export EDGELLM_PLUGIN_PATH=/home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so

PYTHONPATH=src python3 scripts/run_edgellm_direct_benchmark.py \
  --llm-bench /path/to/llm_bench \
  --engine-dir /home/ubuntu/JetsonVLM/artifacts/engines/<engine>/llm \
  --mode decode \
  --output-jsonl reports/direct/decode.jsonl \
  --repetitions 3 --iterations 20 --warmup 10 \
  --past-kv-len 768 --osl 1

PYTHONPATH=src python3 scripts/run_edgellm_concurrency_sweep.py \
  --manifest data/manifests/<manifest>.jsonl \
  --workload configs/workloads/parking_risk_v2_strict_json.json \
  --data-root data/raw \
  --output-dir reports/concurrency_sweep \
  --engine-max-batch-size 4 \
  --concurrency 1 --concurrency 2 --concurrency 4
```

如果板端 `llm_bench` 使用的输出目录参数不是 `--outputDir`，P0 命令通过
`--output-dir-flag <actual-flag>` 覆盖；若二进制只能从日志报告 CSV 路径，则传空值关闭
该参数，并让 runner 使用日志中的 `E2E timing CSV saved to:` 路径。

## 固定实验身份

| 项目 | 固定值 |
|---|---|
| 设备 | Jetson Orin Nano Super，SM87，15W_MODE_0 |
| 软件 | JetPack R36.5，CUDA 12.6.68，TensorRT 10.3.0.30 |
| Edge-LLM | v0.9.1，commit `7f061f21f0a581ba234a1e233c9315b89d8e47d6` |
| 模型 | Qwen3-VL-2B-Instruct，revision `89644892e4d85e24eaac8bacfd4f463576704203` |
| 精度 | INT4 AWQ LLM backbone，FP16 `lm_head` |
| profile | batch=1，`i768/k1024`，KV capacity=1024 |
| 构建 | workspace=1024 MiB，CUDA Graph enabled，weight streaming disabled |
| 插件 SHA-256 | `9437996d36b659092e7d4da244b43da6feb7c69e79a5654a8df53e9f5c2ac7eb` |
| level 0 engine SHA-256 | `33466f3f1149801bf496737fbe1e69634b8018ece9a0c5bccd8ff83af64afab4` |
| level 1 engine SHA-256 | `207e109fcca28ac29ae0348d8dd517e61a6291704a8fea08c53644574f28d345` |

## 最终已核验性能

### 基线与完整请求

| 阶段 | Runtime / 配置 | Prefill | TTFT | Decode / generation | E2E p50 / p90 / p99 |
|---|---|---:|---:|---:|---:|
| 基线 | Jetson Transformers FP16 | 486.32 ms | 716.69 ms | 7,897.63 ms/request；9.54 tok/s | 9.569 / 14.439 / 28.155 s |
| 阶段 1 | TensorRT Edge-LLM INT4，level 0 | 531.61 ms | 872.66 ms<sup>1</sup> | 124.02 ms/token；8.06 tok/s | 10.523 / 11.387 / 12.723 s |
| 阶段 2 | TensorRT Edge-LLM INT4，level 1 | 430.97 ms | 702.50 ms<sup>1</sup> | 27.25 ms/token；36.70 tok/s | 2.826 / 2.953 / 3.315 s |

基线为 `1×20` 请求，Edge-LLM 两档为 `3×20` 请求。不同 runtime 的 decode profile 不是严格 A/B；TTFT 是请求发送到 SSE 首个非空 `delta.content` 的客户端时间，包含视觉编码、调度、首 token decode、网络和流式输出，因此不能用 prefill 代替。E2E 包含预处理、视觉编码、prefill、decode、HTTP 和序列化。

### Edge-LLM 低层严格 A/B

低层 workload 固定 `inputLen=768`、`pastKVLen=768`、OSL=1、warm-up=10、每次 20 次、seed=0，并对 level 0/1 各做 3 次重复。

| 阶段 | 构建变量 | Prefill（20 次均值） | Decode Graph（20 次均值） | Decode 吞吐 | 相对 level 0 |
|---|---|---:|---:|---:|---:|
| level 0 | `builderOptimizationLevel=0` | 526.8768 ms | 126.7803 ms/token | 7.8877 tok/s | — |
| level 1 | `builderOptimizationLevel=1` | 426.7482 ms | 27.4670 ms/token | 36.4073 tok/s | prefill `-19.0042%`；decode `4.6157×` |
| level 1 soak | level 1，CUDA Graph，1000 steps | — | 27.3821 ms/token | 36.5202 tok/s | 0 次 runtime failure |

level 1 相对 level 0 的 decode latency 降低 `78.3618%`。1000-step soak 进程退出码为 0、
Graph capture 成功；同时保留独立 metadata loader 的 TensorRT runtime destructor warning。

## Nsight Systems 与 Engine Inspector

### 主要 tactic 差异

Engine Inspector 显示 level 0/1 的层数和 INT4 W4A16 主体保持一致，主要结构性差异集中在最终 `lm_head` GEMM。该矩阵为：

```text
[1, 1, 2048] × [1, 2048, 151936] -> [1, 1, 151936]
```

| 热点 | level 0 | level 1 | 变化 |
|---|---:|---:|---:|
| 最终 FP16 `lm_head` GEMM（单次 trace） | 111.4517 ms | 10.9825 ms | `-90.14%` |
| INT4 W4A16 主体（196 instances） | 262.3125 ms | 262.3898 ms | `+0.03%` |
| FMHA head-dim 64（24 instances） | 62.0573 ms | 57.4192 ms | `-7.48%` |

level 1 non-Graph decode trace 的 GPU kernel 时间分布为：

| Kernel | 总时间 | Trace share |
|---|---:|---:|
| `gemv_kernel<2,1,256,128>` | 141.1500 ms | 52.3% |
| `trt_ampere_h16816gemm_128x64_ldg8_tn_v1`（FP16 `lm_head`） | 92.2431 ms | 34.2% |
| `kernel_mha` | 18.6402 ms | 6.9% |

当前瓶颈优先级是大词表 `lm_head` tactic、INT4 decode GEMV 以及 launch/Graph 行为，不能将 level 1 的收益笼统描述为“所有算子都融合或加速”。Nsight 的 CUDA API 总时间包含初始化和 warm-up，不能直接当作单步 decode 时间。

本轮 fresh level1 non-Graph trace（5 次、warm-up=3）记录到 kernel 总时间 `268.640160 ms`：
`gemv_kernel<2,1,256,128>` 为 `140.025056 ms`（1568 次），`trt_ampere_h16816gemm_128x64_ldg8_tn_v1`
为 `92.478944 ms`（8 次），`kernel_mha` 为 `18.469248 ms`（224 次）。该 trace 用于热点定位，
不与历史 level0 单次 trace 拼成新的严格 A/B。

## 算子与矩阵候选结论

| 候选 | 测试口径 | 结果 | 结论 |
|---|---|---|---|
| builder level `0 -> 1` | 整体 engine tactic | decode `4.6215×`，prefill `-19.039%` | 已核验的主要收益阶段 |
| INT4 GEMV block/tile `128/192/320/512` | batch=1 decode | `128` 约 `-0.15%`；`192/320/512` 变慢约 `1.32%/7.15%/4.31%` | 未形成稳定收益 |
| `CTA_N=256` | INT4 W4A16 prefill GEMM | `+17.24%` | 否决 |
| `CTA_M=128` | INT4 W4A16 prefill GEMM | `+16.67%` | 否决 |
| `__ldg` read-only cache hint | INT4 GEMV decode | no-Graph `-0.17%`，Graph `+0.04%` | 否决 |
| `__restrict__` pointer hint | INT4 GEMV decode | no-Graph `-0.08%`，Graph `+0.56%` | 否决 |
| `NPerBlock=4` + CUDA stream 修复 | INT4 GEMV decode | 仅完成候选 Graph 局部运行 | control/A-B 和完整 VLM 未完成，不纳入结果 |

上述候选都保持 engine、权重布局、scale 布局和 workload 不变，并在隔离 plugin stack 上构建；正式 plugin 和正式 engine 没有被这些候选覆盖。

## 当前技术边界

- 当前路径使用 Edge-LLM 已有的 INT4 W4A16、attention/FMHA plugin、TensorRT tactic 选择和 CUDA Graph。
- 当前项目增加的是构建配置、provenance、benchmark、Nsight 汇总和隔离候选 patch；没有将自研
  CUDA kernel 或新 TensorRT plugin 宣称为正式生产路径。
- 投机采样、NVFP4、FP8 和 FP16 weight streaming 尚未形成本轮可比较的性能结果；paged KV 已形成
  独立低层候选，但尚未通过 parity、HTTP、Nsight 和 soak 门禁，因此不进入默认性能结论。
- builder level `2/3` 在板端构建阶段因约 622 MB CUDA allocation 失败，尚未生成可比较 engine；
  不对其推理性能作推断。

## 证据入口

- [阶段 1 报告](../reports/jetson-tensorrt-stage1/phase1_summary.md)
- [level 0/1 prefill 复验](../reports/jetson-tensorrt-revalidation/int4_level0_level1_prefill768_decode20_20260909.json)
- [level 0/1 decode 重复性](../reports/jetson-tensorrt-revalidation/int4_level0_level1_decode20_repeats_20260909.json)
- [level 1 1000-step soak](../reports/jetson-tensorrt-revalidation/int4_level1_decode1000_soak_20260909.json)
- [Nsight 对比摘要](../reports/jetson-tensorrt-revalidation/int4_opt0_vs_opt1_nsight_summary.json)
- [level 1 热点摘要](../reports/jetson-tensorrt-revalidation/int4_level1_formal_nograph_nsys_20260909.json)
- [GEMV/GEMM 候选 A/B 目录](../reports/jetson-tensorrt-revalidation)
- [dynamic batching 与 paged KV 实施记录](tensorrt-dynamic-batching-paged-kv.md)
- [paged KV engine provenance 与原始 CSV/telemetry](../reports/tensorrt-priority/p3/paged-kv-20260922)
