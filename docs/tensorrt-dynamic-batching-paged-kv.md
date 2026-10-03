# TensorRT Edge-LLM 动态 Batch 与 Paged KV 实施记录

本文记录固定 TensorRT Edge-LLM v0.9.1 checkout 上 P2/P3 的实现边界、修复内容和板端验收结果。
正式 level 0 engine 不被覆盖；paged KV 使用独立 engine、独立 provenance 和独立原始证据。

## P2：dynamic batching / continuous batching

### 上次 batch 构建失败的实际边界

`maxBatchSize=4` 的 LLM engine 已经能够构建，失败发生在请求级运行时，而不是 TensorRT
builder。旧 runtime 在以下路径存在状态不一致：

- decode token 反馈仍通过 Host 临时 buffer 重新拼接，batch active 状态变化后容易与 runtime
  sampling indices 不一致；
- batch eviction 后没有压缩 vanilla decoder 的 device-side sampling indices；
- 最后一个请求退出时，仍对 `newActiveBatch=0` 执行 batch mapping 上传、KV/cache compact
  和 stream synchronize，触发 `slots missing` 或 `oldActiveBatch` 断言；
- speculative decoder 的 draft cache 也对空 batch 执行 compact。

因此，之前的低层 batch=4 成功只能说明静态 batched engine 可运行，不能说明 HTTP 请求级
continuous batching 已经可用。

### 已实施的 runtime 修复

修复已应用到板端 `/home/ubuntu/TensorRT-Edge-LLM` 并完成 `make -C build -j2`：

1. vanilla decoder 在普通 batch=1、非 debug 路径直接使用 device-side sampling indices
   做 D2D token feedback；batch>1 或 debugger 路径保留 host fallback。
2. `VanillaDecoder::onBatchEvict` 使用 `compactTensorBatch` 压缩 sampling indices，并校验
   old active batch 的 shape。
3. `LLMInferenceRuntime::performBatchEvict` 在 `newActiveBatch=0` 时跳过 mapping 上传、
   base cache compact 和不必要的同步。
4. DFlash/Eagle/MTP decoder 在空 batch 时跳过 draft cache compact。

本地保留的可复现补丁和 CRLF-safe 应用脚本：

- [`dynamic-batch-runtime-valid.patch`](../reports/tensorrt-priority/p2/dynamic-batch-runtime-valid.patch)
- [`apply_dynamic_batch_patch.py`](../reports/tensorrt-priority/p2/apply_dynamic_batch_patch.py)

### 已取回的板端低层复验

固定 batch4 engine、`i768/k1024`、`pastKVLen=768`、OSL=1、warm-up=5、iterations=10，
使用重编译后的 plugin/runtime：

| batch | E2E | 单序列吞吐字段 | 进程结果 |
|---:|---:|---:|---|
| 1 | `29.4693 ms` | `33.9337 tok/s` | exit code 0 |
| 4 | `43.7650 ms` | `22.8493 tok/s` | exit code 0，CUDA Graph capture 成功 |

`llm_bench` CSV 中的 `throughput_tps` 是单序列字段；batch=4 的 aggregate throughput 应按
`4 / 43.7650 ms` 计算，约为 `91.4 tok/s`。它仍是一个静态 batch 的低层指标，不等同于
HTTP continuous batching 的请求级吞吐。

### HTTP native microbatch scheduler 修复

仅修复 C++ eviction 仍不足以支持 HTTP continuous batching：旧 Python server 为每个请求
启动一个 `handle_request` 线程，多个 VLM 请求会并发使用同一个 visual execution context。
本轮在固定 v0.9.1 checkout 增加了 bounded microbatch scheduler：

- 用单个 native worker 串行进入共享 runtime/visual context；
- 在 `2 ms` admission window 内合并兼容请求，最多形成 `maxBatch=4` 的一次
  `LLMGenerationRequest`；
- 保留每个请求独立的 `StreamChannel`，让 C++ runtime 按 slot 完成和 eviction；
- 非流式 route 也经过同一 dispatch，v0.9.1 pybind 缺失的可选字段采用兼容读取，
  `finish_reasons` 只读则用 Python response view 拆分。

板端服务启动参数为 `--http-max-batch-size 4`，scheduler patch 由
[`apply_http_batch_scheduler.py`](../reports/tensorrt-priority/p2/apply_http_batch_scheduler.py)
及后续 v0.9.1 pybind compatibility helpers 回放。该 patch 不改变 engine plan，
只改变 HTTP admission 和 runtime 调用边界。

### HTTP 复验结果

固定 `ps20_pilot_v1.jsonl`、20 个样本、warm-up=1、每次 3 次重复、同一 maxBatch=4
INT4 engine、复用 HTTP connection：

| concurrency | 请求数 | 完成率 | aggregate output tok/s | 结果 |
|---:|---:|---:|---:|---|
| 2 | 60 | 60/60 | 27.5115 | 通过 |
| 4 | 60 | 60/60 | 41.0209 | 通过 |
| 4 soak | 100 | 100/100 | 40.2069 | 通过 |

三组运行均未出现 `HTTP 500`、`slots missing`、`already loaded binary graph` 或
`native microbatch failed`。原始证据目录为
[`dynamic-http-scheduler-c2-r3`](../reports/tensorrt-priority/p2/dynamic-http-scheduler-c2-r3)、
[`dynamic-http-scheduler-c4-r3`](../reports/tensorrt-priority/p2/dynamic-http-scheduler-c4-r3)
和 [`dynamic-http-scheduler-c4-soak100`](../reports/tensorrt-priority/p2/dynamic-http-scheduler-c4-soak100)。
这里的 aggregate throughput 是请求级输出 token 除以 steady-state wall clock，不能与
低层单序列 `throughput_tps` 混用；请求级 TTFT/E2E 仍需从 JSONL 分布单独分析。

另做了一次 c=4、20 样本、1 次重复的 `tegrastats` 伴随采样：steady-state aggregate
output throughput `40.9566 tok/s`；59 个采样点中 RAM 为 `5636--5656/7619 MB`，SWAP
为 `1555/3810 MB`，GR3D 利用率最高 `99%`，`tj` 温度最高 `58.187 C`，`VDD_IN` 最高
`13680 mW`。这组 telemetry 已归档在
[`dynamic-http-scheduler-c4-telemetry`](../reports/tensorrt-priority/p2/dynamic-http-scheduler-c4-telemetry)，
目前没有同口径 baseline telemetry，因此只作为资源事实，不作为“内存无恶化”的 A/B 结论。

### P2 完成条件

- concurrency=1/2/4 在相同 20 个冻结样本和相同 engine 下完成 3 次重复；当前已完成
  c=2、c=4，c=1 仍作为串行 control 保留；
- 无 `slots missing`、`oldActiveBatch`、CUDA Graph invalidation 和超时；
- 记录请求完成率、请求级 TTFT/E2E、wall-clock aggregate output tok/s、显存/RAM/swap；
- 再做 100 次连续 soak，确认中途 admission、eviction、reclaim 和最后一个请求退出均稳定。

## P3：paged KV cache

### 当前 v0.9.1 的真实状态

固定 checkout 中已经存在 paged FMHA/XQA kernel 和 `pageList/tokensPerPage` ABI 符号；本轮以
最小 backport 方式补齐了 Qwen3-VL v0.9.1 所需的 builder、plugin 和 runtime page-table
接线。当前状态如下：

| 层次 | 当前状态 | 必须补齐的内容 |
|---|---|---|
| Builder/config | 已支持 `--maxKVPoolPages`，并将 `max_kv_pool_pages=32` 写入 paged engine config | 后续补齐 config 中显式的 layout/page-size 字段，并加入启动时一致性校验 |
| ONNX/TensorRT Plugin | 已增加 `kv_page_table` binding、paged pool shape `[2,pages,128,Hkv,D]`、prefill gather 和 decode XQA page-list 路由 | 继续补充 Engine Inspector/Nsight 的 binding 与 kernel 参数证据 |
| Runtime | 已连接 page pool、page table、KV 写入和 batch=1/4 的低层执行；dense engine 回归通过 | HTTP paged scheduler、跨 page 增长/释放日志和长 soak 仍待补齐 |
| 运行验证 | paged engine 已成功 build；batch=1/4 prefill/decode 均 exit code 0，CUDA Graph capture 成功 | dense-vs-paged 数值 parity、HTTP paged path、100 次 soak 尚未完成 |

因此，单独把 plugin 中的 `usePagedKVCache` 改成 `true` 不能修复问题；本轮失败复验正是由
ONNX cache 仍是 dense shape、plugin binding 数量不一致，以及 KV 写入 kernel 把 paged pool
误当成 `[batch,2,Hkv,capacity,D]` 引起的。

### v0.10.x 对照与固定版本边界

对照 NVIDIA 上游 v0.10.0/v0.10.1 源码后确认，完整 paged-KV 路径不是一个孤立开关：它
同时引入 `maxKVPoolPages` builder 参数、`kv_page_table` binding、paged pool shape 校验、
`KVPageTable` device upload/compact、runtime page allocator，以及 attention plugin 的 prefill
`runPaged` 和 decode XQA `pageList/tokensPerPage` 路由。该版本还伴随 attention/context-kernel
和 runtime 配置的大范围重构。

因此本轮没有把 v0.10.x 文件直接复制到固定 v0.9.1 checkout，也没有把版本升级当成
“修复 paged KV”。当前可验证结论是：v0.9.1 的 paged builder/plugin/runtime 低层路径已经
闭环到可执行 engine，但尚未达到正式性能候选的全部质量门禁；后续仍需补齐数值 parity、
HTTP paged scheduler、Nsight route 证据和长 soak。每层继续保留 contiguous engine 作为
control。

### 实施计划

#### P3.1：配置与 builder

本轮实际使用以下 engine provenance 字段：

```json
{
  "kv_cache_layout": "paged",
  "tokens_per_page": 128,
  "max_kv_pool_pages": 32,
  "max_kv_cache_capacity": 1024,
  "max_batch_size": 4
}
```

Jetson Orin Nano SM87 的当前源码证据表明，paged FMHA 的候选 `tokensPerPage` 需要优先从
`128` 开始验证；`k1024` 对应单序列最多 8 个逻辑 page。batch=4 的最小物理 pool 为
`4 * ceil(1024 / 128) = 32` pages，本轮先用该最小值验证 binding、page table 和 batch=4
执行；后续并发/碎片实验再增加 pool 余量。

Builder 必须把这些值真正序列化进 engine config，并在启动时拒绝：

- builder CLI 声明 paged KV，但最终 engine config 仍是 contiguous；
- page pool 小于 `maxBatchSize * ceil(maxKVCacheCapacity/tokensPerPage)`；
- `tokensPerPage` 不满足当前 FMHA/XQA kernel 约束；
- 旧版 `llm_build` 仍不识别 paged 参数。

#### P3.2：AttentionPlugin binding 与 kernel 路由（已实施）

Paged engine 的 binding contract 需要明确区分：

- 物理 KV page pool；
- `[active_batch, max_pages_per_sequence]` 的 int32 page table；
- sequence lengths；
- 现有 Q/K/V、RoPE 和 batch mapping。

prefill 走 paged FMHA 的 `runPaged`，decode 走 XQA 的 `KVCache.pageList` 和
`tokensPerPage`。两条路径都必须在 plugin profile 中声明正确的动态 shape；只验证 engine
能 deserialize 不足以证明 page path 已被执行。

#### P3.3：Runtime allocator 与 dynamic batch 连接（低层已实施）

每个请求持有逻辑 page table，page pool 维护 free list。请求 admission 按 prompt 长度分配
页面，decode 跨 page 边界时增量分配；请求结束或 eviction 时归还页面。batch compact 只
压缩 page table 和 active metadata，不搬运整块 KV payload。该设计与 P2 的 active-batch
mapping 修复必须共用 eviction 顺序，避免 page table、sampling indices 和 cache owner
出现不同步。

#### P3.4：验证门禁

Paged candidate 只有同时满足以下条件才进入性能矩阵：

1. `config.json` 声明 `kv_cache_layout=paged`、page size 和 pool capacity；
2. runtime 启动日志出现 page pool 初始化，运行中有 allocate/reclaim 计数；
3. Engine Inspector/Plugin profile 显示 page pool/page table bindings；
4. contiguous KV 与 paged KV 在固定短 prompt、长 prompt、batch=1/4 下数值一致；
5. Nsight 能看到 paged FMHA/XQA 路由，而不是仍执行 contiguous fallback；
6. 完成同口径 latency、throughput、显存、100 次 soak 和动态并发测试。

当前已满足第 1 项的独立 provenance、paged engine build 和第 3 项的 page-table binding
contract 的源码/shape 证据；第 2、4、5、6 项尚未全部满足，因此 paged engine 仍保持
候选状态，不替换 level 0/1 默认 engine。

### P3.5：2026-09-23 板端复验结果

环境固定为 Jetson Orin Nano Super、15W、SM87、TensorRT 10.3.0、CUDA 12.6.68、Edge-LLM
v0.9.1 commit `7f061f2`、INT4 AWQ、`i768/k1024`、builder level 1、workspace 1024 MiB。
完整身份见 [`engine-provenance.json`](../reports/tensorrt-priority/p3/paged-kv-20260922/engine-provenance.json)。

| engine / batch | prefill E2E | decode E2E（OSL=32） | decode 单序列 tok/s | batch aggregate tok/s | 进程 / Graph |
|---|---:|---:|---:|---:|---|
| paged / 1 | 428.3120 ms | 857.7339 ms | 36.1428 | 36.1428 | exit 0 / captured |
| paged / 4 | 1655.6338 ms | 1342.0580 ms | 23.1019 | 92.4076 | exit 0 / captured |
| dense control / 1 | 未采集 | 872.1420 ms | 35.5447 | 35.5447 | exit 0 / captured |
| dense control / 4 | 未采集 | 1344.4401 ms | 23.0579 | 92.2316 | exit 0 / captured |

这里的 batch aggregate 是 `batch * effective_decode_steps / E2E`；`llm_bench` 的
`throughput_tps` 本身是单序列字段，OSL=32 实际执行 31 个 decode steps。paged 正式结果
为 3 次重复、每次 20 iterations；dense control 的当前回归 CSV 为 3 iterations，二者
尚未构成同重复数的正式性能 gate，因此不能把微小差异解释为 paged 加速。paged 待验证
收益仍是物理 page 复用、碎片控制和并发容量。原始 CSV、日志和 telemetry 归档在
[`paged-kv-20260923-formal`](../reports/tensorrt-priority/p3/paged-kv-20260923-formal)，
早期 builder smoke 证据在 [`paged-kv-20260922`](../reports/tensorrt-priority/p3/paged-kv-20260922)。

本轮 telemetry（新 paged engine，batch=4 decode）为 71 个采样点：RAM used 峰值 5197 MB、SWAP
1569 MB、GR3D 峰值 99%、`tj` 峰值 55.718 C、`VDD_IN` 峰值 11588 mW。它是资源事实，
尚未与相同采样窗口的 dense control 组成正式内存/功耗 A/B。

另外使用 Edge-LLM 自带 `ragged_batch.json` 做了 C++ batch=4 输出 smoke：4 个请求全部
完成，输出分别为 `The capital of France is Paris`、`Indeed, artificial intelligence has undergone`、
`The three primary colors are red` 和 `A **Transformer neural network**`；该结果证明
paged engine 可以实际生成 token，但不是 dense-vs-paged 的数值 parity。输出证据见
[`numerical_smoke_output.json`](../reports/tensorrt-priority/p3/paged-kv-20260923-formal/numerical_smoke_output.json)。

## 与当前实验矩阵的关系

- P2 runtime 修复属于同一 maxBatch=4 engine 的运行时 A/B，不覆盖正式 level0/level1。
- P3 paged KV 必须新建独立 engine 目录和 provenance，不覆盖 contiguous KV engine。
- speculative decoding、LoRA、FP8/NVFP4 和质量指标不进入本阶段变量。
- 所有 latency 结论仍以固定 workload 的低层 benchmark 为准，HTTP RTT 只作为链路指标。
