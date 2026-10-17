# TensorRT Edge-LLM 动态 Batch 与 Paged KV 实施记录

本文记录固定 TensorRT Edge-LLM v0.9.1 checkout 上 P2/P3 的实现边界、修复内容和板端验收结果。
正式 level 0 engine 不被覆盖；paged KV 使用独立 engine、独立 provenance 和独立原始证据。
构建入口在启动 builder 前拒绝复用非空的目标组件目录，并独占创建 build report；失败后若目录中留有产物，下一轮需使用新的候选输出路径，避免混入旧 engine 文件。
batch=4 的普通和 detailed 候选还固定了预检时的 LLM builder 与 plugin 二进制哈希；任一二进制变化均需重新预检并更新候选配置，不能沿用旧指纹启动构建。

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

### P2 完成条件与结果

已在同一服务进程、同一 batch=4 engine 上完成 PS20 c=1→4→1，每段 3×20
请求、warm-up 1 次。三段均 60/60 完成，逐图输出哈希与 token 数一致；墙钟吞吐为
`27.259→40.169→27.261 output tok/s`，c=4 比两侧 c=1 均值高 47.36%。此前
c=4 的 100 请求 soak 也达到 100/100；未见 `slots missing`、CUDA Graph 错误或
HTTP 500。整机 telemetry 与原始归档见
[优化阶段速览](tensorrt-optimization-summary.md#p2-同机并发对照2026-10-13)。
该结论限于冻结 PS20；默认部署前仍需覆盖不同输入长度及图像 token 数。

## P3：paged KV cache

### 当前 v0.9.1 的真实状态

固定 checkout 中已经存在 paged FMHA/XQA kernel 和 `pageList/tokensPerPage` ABI 符号；本轮以
最小 backport 方式补齐了 Qwen3-VL v0.9.1 所需的 builder、plugin 和 runtime page-table
接线。当前状态如下：

| 层次 | 当前状态 | 必须补齐的内容 |
|---|---|---|
| Builder/config | 已支持 `--maxKVPoolPages`，现有 paged engine config 声明 `kv_cache_layout=paged`、`tokens_per_page=128` 和 `max_kv_pool_pages=32` | 新构建需核对序列化后的实际字段，并在运行时继续核验 |
| ONNX/TensorRT Plugin | 已增加 `kv_page_table` binding、paged pool shape `[2,pages,128,Hkv,D]`、prefill gather 和 decode XQA page-list 路由；Nsight 已区分 dense/paged decode kernel 指纹 | 继续补充 page-list 实参及 prefill gather 的直接运行证据 |
| Runtime | 已连接 page pool、page table、KV 写入和 batch=1/4 低层执行；eviction KV compact 按 engine 布局选择步长，dense 回归通过 | 页分配/释放计数仍需补齐 |
| 运行验证 | paged engine 已成功 build；batch=1/4 prefill/decode 均 exit code 0；最终分流版 HTTP c=1/4/1 各 3×20 完成，180/180 严格 JSON 有效且逐图原文一致；c=4 100 请求 soak 完成 | 长输入 dense/paged 严格数值一致性仍未完成 |

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
闭环到可执行 engine，但尚未达到正式性能候选的全部质量门禁；冻结 PS20 的 HTTP 并发
原文一致性和 100 请求 soak 已通过；Nsight 已区分 decode XQA 的 dense/paged kernel 指纹，
后续仍需补齐长输入数值 parity 与 page-list 实参证据。
每层继续保留 contiguous engine 作为 control。

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

构建前须先运行 `configs/flows/export/prepare_qwen3_vl_2b_int4_awq_n128_pagedkv32.json`
（batch=1 则用 `pagedkv16` 配置），以独立目录保存 patched ONNX、外部权重和
`paged_onnx_provenance.json`。对应 build flow 只接受这个 staged ONNX，并核对
batch、页数、完整文件清单及哈希、28 个 AttentionPlugin 的 paged binding；阶段目录
中的额外文件或符号链接会被拒绝。准备和构建预检
均需要可导入 `onnx` 的 Python 环境；新生成的 stage 还会逐节点核对第 8 个 page-table 输入、
INT32 表形状、batch 维容量和每层 KV pool 输入/输出形状，并把校验计数写入 provenance。
预检还要求 past/present cache 的层标识唯一且逐层配对，dtype 与完整形状一致；
准备阶段和构建预检均核对 AttentionPlugin 数量与 `llm/config.json` 的
`num_hidden_layers`，避免不完整图和 provenance 相互吻合却被放行。
构建脚本的 `--check-only` 现可只读加载 ONNX 元数据，复核已有 staged ONNX 的相同
结构契约，不加载外部 INT4 权重或重新复制文件。build flow 已将该校验脚本列为
必需输入。Jetson 上已有的 batch=4 stage 尚需用加入逐层 KV 配对和模型层数检查的当前版本重跑
`--check-only`；完整 engine 重建和 tactic 对照仍待执行。CLI 传入
`--max-kv-pool-pages` 时强制启用
`--verify-paged-kv-source`。源码审计检查固定 checkout 的
`KVPageTable::setRows/compactRows/upload`，只证明源码路径存在；结果保存于本地
`reports/jetson-readiness-20261012/p3_paged_source_audit_20261016.json`。Jetson
checkout 有未提交的 C++ 变更，单独记录 Git HEAD 不足以标识构建源码；审计同时记录
`cpp/` 与 `examples/llm/` 的源码内容指纹，engine 构建报告还记录 builder 和 plugin
二进制哈希。

2026-10-16 在 Jetson 对 batch=4 staged ONNX 执行当时版本构建脚本的 `--check-only`：源码审计、
28 个 paged AttentionPlugin binding、pool=32、文件哈希及 builder/plugin 二进制身份均
通过，返回 `ready_for_build`，且未创建 engine 目录。结果保存在本地
`reports/jetson-readiness-20261012/p3_paged_build_preflight_20261016.json`。
该结果早于逐层 KV 配对检查，只证明当时的构建输入就绪；engine 序列化、tactic 和
推理性能仍需独立验证。

同步当前仓库脚本与 flow 配置到 Jetson 后，在 SSH 登录后的 Bash 中重跑只读构建预检：

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python - <<'PY'
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys

flow = json.loads(Path(
    "configs/flows/build/build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt1_dynamic_batch4_paged_kv.json"
).read_text(encoding="utf-8"))
if flow["command"][:2] != ["python3", "scripts/build_edgellm_vlm_engines.py"]:
    raise RuntimeError("unexpected build command in paged-KV flow")
command = [sys.executable, *flow["command"][1:], "--check-only"]
result = subprocess.run(command, capture_output=True, text=True)
if result.returncode:
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    raise SystemExit(result.returncode)
report = json.loads(result.stdout)
if (report.get("status") != "ready_for_build"
        or report["paged_onnx_stage"]["attention_plugin_count"] != 28):
    raise RuntimeError("paged-KV build preflight did not validate all 28 layers")
output = Path("reports/jetson-readiness-20261012") / (
    f"p3_paged_build_preflight_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
)
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("x", encoding="utf-8") as stream:
    stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(f"{report['status']}: {output}")
PY
```

成功时打印 `ready_for_build` 与新报告路径，报告包含 `attention_plugin_count=28`；
预检不会新建 engine 文件或 build report。层数、哈希或二进制身份不匹配时应保留报错
并停止构建。此命令不执行
`scripts/build_engine.py --execute`，也不替代后续 engine 加载与请求级验收。
构建脚本现会在 LLM builder 返回后核对生成的 `llm/config.json` 中布局、page size、
batch、输入长度、KV 容量和 pool 页数；不匹配时将构建报告标为失败。该规则已用
现有 Jetson paged engine 的实际 config 格式做只读验证。
详细 profiling 的隔离配置为
`configs/flows/build/build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt1_dynamic_batch4_paged_kv_detailed.json`；
Jetson 只读预检确认当前 builder 提供原生 `--profilingDetailed`。普通与详细构建分别
使用独立 engine 目录，并固定同一源码指纹；详细 profiling 可能改变 tactic，不能把
两份 engine 的性能差异直接归因于 KV 布局。
同一 Jetson checkout 上还对源码指纹锁定做了正负例预检：实际指纹返回
`ready_for_build`，故意替换的指纹被拒绝；两次运行后目标 engine 目录均不存在。
本地证据为 `reports/jetson-readiness-20261012/p3_paged_pinned_preflight_20261016.json`
及同目录的 `p3_paged_wrong_fingerprint_stderr_20261016.txt`。

Builder 必须把这些值真正序列化进 engine config，并在启动时拒绝：

- builder CLI 声明 paged KV，但最终 engine config 仍是 contiguous；
- page pool 小于 `maxBatchSize * ceil(maxKVCacheCapacity/tokensPerPage)`；
- `tokensPerPage` 不满足当前 FMHA/XQA kernel 约束；
- 旧版 `llm_build` 仍不识别 paged 参数。

服务启动时可先用 `scripts/serve_edgellm.py --require-paged-kv --check-only`
核验候选 LLM engine，再用相同 `--require-paged-kv` 参数正式启动。该门禁在导入
GPU runtime 前读取 `llm/config.json`，要求 paged 布局、128-token 页及覆盖
`max_batch_size × ceil(max_kv_cache_capacity / 128)` 的物理 page pool；
使用 `--runtime-metadata-output` 时，启动元数据记录配置 SHA-256。此静态检查不替代
运行中 page-list 与分配/归还证据。

#### P3.2：AttentionPlugin binding 与 kernel 路由（已实施）

Paged engine 的 binding contract 需要明确区分：

- 物理 KV page pool；
- `[active_batch, max_pages_per_sequence]` 的 int32 page table；
- sequence lengths；
- 现有 Q/K/V、RoPE 和 batch mapping。

prefill 走 paged FMHA 的 `runPaged`，decode 走 XQA 的 `KVCache.pageList` 和
`tokensPerPage`。两条路径都必须在 plugin profile 中声明正确的动态 shape；只验证 engine
能 deserialize 不足以证明 page path 已被执行。

#### P3.3：Runtime page table 与 dynamic batch

当前低层实现创建固定容量 page pool，以 identity 映射初始化 page table，并支持
`setRows`、`compactRows` 和 GPU upload；HTTP eviction 已按 dense/paged 布局分流。
动态 free list、请求 admission 分配、跨页增量分配及归还计数尚无验收证据，
后续须与 P2 的 active-batch eviction 顺序一起验证。

#### P3.4：验证门禁

Paged candidate 只有同时满足以下条件才进入性能矩阵：

1. `config.json` 声明 `kv_cache_layout=paged`、page size 和 pool capacity；
2. runtime 启动日志出现 page pool 初始化，运行中有 allocate/reclaim 计数；
3. Engine Inspector/Plugin profile 显示 page pool/page table bindings；
4. contiguous KV 与 paged KV 在固定短 prompt、长 prompt、batch=1/4 下数值一致；
5. Nsight 能看到 paged FMHA/XQA 路由，而不是仍执行 contiguous fallback；
6. 完成同口径 latency、throughput、显存、100 次 soak 和动态并发测试。

当前已有独立 provenance、paged engine config 布局字段和 page-table binding 的源码/shape 证据；
Nsight 支持 decode 专用路由，HTTP 并发及 100 请求 soak 已完成。运行时的
页分配/归还计数、page-list 实参与长输入数值一致性仍缺证据，因此 paged engine 保持
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

### P3.6：2026-10-13 数值与 HTTP 并发复验

同一批 dense/paged engine、tokenizer 和生成参数下，SDK 短 ragged batch=4 的
4/4 输出在 dense→paged→dense 三段完全一致。长度为 8/232/517/682 token 的
长 ragged batch=4 全部运行成功，但 paged 的 2/4 原文不同；单请求重试也复现
其中差异。top-5 logprobs 显示首个分歧位置的候选 token 接近，例如短请求
`street`/`paved` 的 paged logprob 只差 0.0156；这支持数值近似导致选择变化的
解释，但不能证明全部 logits 一致。原始归档分别为
`reports/jetson-readiness-20261012/p3_dense_paged_ragged_aba_artifacts.tgz`
（SHA-256 `28ba8c1b6a9d677cb8860e03232d56e5ad74e90f9f8b6c29a8dc3dfbf9a17288`）、
`p3_dense_paged_long_aba_artifacts.tgz`
（`3844719ab2b6d572c889e3362191ec0ff981da57290afe6b1c7d764577dd5097`）和
`p3_logprobs_probe_artifacts.tgz`
（`7a9871a3d67f785eb244df8c2bb284f214d49915fa5ad83bd64e76a2c6de13a0`）。

HTTP paged 单服务 PS20 c=1→4→1 各 20/20 完成，吞吐为
`27.295→39.845→27.356 output tok/s`，但 c=4 有 5/20 原文哈希偏离 c=1。
独立的原文诊断 c=1、c=4 各 20/20，其中 c=4 有 3 条输出不是有效 JSON，
即“请求完成”不能作为质量通过。源码检查发现 KV pool 实际布局为
`[K/V,batch,token,head,dim]`，eviction compact kernel 却按
`[batch,K/V,head,token,dim]` 计算地址；这会在请求提前结束、存活请求迁移时
复制错误的 KV。最初全局改用 NHD 步长虽消除了 paged 的 JSON 损坏，却令 dense
c=4 有 8/20 条严格 JSON 无效，因此没有保留该全局改动。最终使用
[`fix_edgellm_kv_compaction.py`](../scripts/experiments/fix_edgellm_kv_compaction.py) 按 engine
配置的 `numPages` 分流：dense 保留原拷贝路径，仅 paged 使用 NHD 步长。
补丁可从原源码逐字节回放，原源码及构建日志保存在板端独立实验目录。

修复后同一 paged engine 再做单服务 c=1→4→1，各 20/20 完成，吞吐为
`27.328→40.167→27.366 output tok/s`；从保留的原文回放，三段共 60/60
通过 `ParkingAssessment` 严格校验，两侧 c=1 为 20/20 原文一致，c=4 与
c=1 为 19/20 一致。唯一差异是 `ps20-indoor-001` 的证据描述措辞变化
（94→95 tokens），仍是有效评估。修复前、后的原文诊断归档分别为
`p3_paged_http_raw_aba_partial_artifacts.tgz`
（SHA-256 `4f82ab96818eae661764dcfafc90de9881ccb3d477691f3ebebc476c5cca2daf`）和
`p3_paged_http_compaction_fix_aba_artifacts.tgz`
（`0b305d03c964b65a8eb04abbfbadd1d694da3c23c881a652bb6271a351493e91`）。
最终分流版本的 dense engine 在同一服务 c=1→4→1 各 20/20 完成，60/60 严格
JSON 有效，逐图原文哈希三段 20/20 一致；吞吐为
`27.149→40.851→27.218 output tok/s`。原始归档
`p2_dense_http_layout_select_aba_artifacts.tgz` 的 SHA-256 为
`4f5f839aea3c2a720b76765029b10d5285915c132b5a16f2e1a10b90cad33830`；
严格对照报告 `p2_dense_http_layout_select_aba_comparison.json` 的 SHA-256 为
`7a6cde3a94912acc307e27c582d840e6cd7483063c470405ff7d4ac92849fd31`。
在该初次修复轮次，paged 的严格原文 parity 尚未通过；后续使用最终分流版继续复验。

最终分流版 paged engine 又在**同一服务进程**完成 PS20 c=1→4→1，每段 3×20
请求、warm-up 1 次。三段均 60/60 完成，`assessment_valid` 均为 60/60，
按 `(sample_id, repetition)` 比较的原文哈希及 token 数 60/60 完全一致。
吞吐为 `27.349→40.200→27.348 output tok/s`，c=4 相对两侧 c=1 均值
为 1.470 倍，两侧基线漂移 -0.005%。原始归档
`p3_paged_http_layout_select_long_aba_artifacts.tgz` SHA-256 为
`a89b9ae497998496d678ad9d20fddee8f906a7d8a0e026c47844bfeb8d543302`；
严格对照报告 `p3_paged_http_layout_select_long_aba_comparison.json` SHA-256 为
`46cf216f0a32353f7b470ab745359a01861a927f28bd8c8d8735dc3d79cc038e`。
该结果闭环了冻结 PS20 的 HTTP 并发格式和原文一致性；此前单轮 1/20 的
措辞差异没有在后续 180 请求中复现。

2026-10-15 使用同一 paged engine、PS20 工作负载与 15W 模式，另完成 c=4、
20 图 × 5 次的 100 请求 soak：100/100 完成，100/100 通过严格
`ParkingAssessment` JSON 校验，7840 个输出 token / 195.248 秒，墙钟吞吐
`40.154 output tok/s`。E2E p50/p90/p99 为 `7.808/7.969/8.386 s`；
各重复的 p90 为 `8.016/7.974/7.968/7.960/7.959 s`。20/20 图在五次重复中
原文哈希与 token 数一致，并与上一轮 paged c=4 A-B-A 归档的输出一致；服务日志
未检出 HTTP 500、CUDA Graph、slot 或 microbatch 错误。整机 `tegrastats`
422 点覆盖服务启动与关闭，RAM 峰值 `5660/7619 MB`、swap 峰值 `1604 MB`、
温度峰值 `56.593 °C`、输入功耗峰值 `13.764 W`；这些不是进程显存或单请求功耗。
原始归档为本地忽略目录中的
`reports/jetson-readiness-20261012/p3_paged_http_layout_select_soak100_artifacts.tgz`
（SHA-256 `afaba814ddf05717c4568e06c15f35e861f5f5ef088b61777cd19e56353ae920`），
汇总为同目录的 `p3_paged_http_layout_select_soak100_summary.json`。
同一 517-token 长输入的 Jetson Nsight Systems 2024.5 采集使用上述两份 engine、同一
二进制/插件和 15W 模式；dense/paged 各自成功导出 GPU trace。两轮均有 1848 次
`kernel_mha` decode XQA 事件，但所有 dense 事件为 `152 Reg/Trd`，所有 paged 事件为
`202 Reg/Trd`；grid/block 相同。固定源码在 `mEnablePagedKV` 时传入 page table 和
`tokensPerPage=128`，XQA runner 据此选 paged kernel。Nsight 的通用名称不会显示
`xqa` 或 `paged`，不同寄存器指纹结合源码和 engine 配置支持 paged decode 专用路由；
trace 不直接读取 page-list 实参。两份 engine 的 FP16 GEMM kernel 名称也不同，
因此长输入 2/4 原文差异不能直接归因于 KV 布局，带 profiler 的 GPU 时长也不能用于
宣称 paged 加速。原始归档为本地忽略目录中的
`reports/jetson-readiness-20261012/p3_paged_nsight_route_artifacts.tgz`
（SHA-256 `b675085bc2991b19bf742970c8c079c9dd104deb84fe20f0e4594a18b359e3fe`），
同目录有 dense/paged 的 v5 kernel 分类摘要。

为检查 page 边界，将同一合成场景文本截成格式化 prompt 长度
`126–129`、`254–257`、`510–513` token，使用相同 engine、插件、15W 模式做
三组 batch=4 的 dense→paged→dense。两 token 运行中，12/12 条 dense 返回组的原文
及 top-10 logprobs 完全一致，paged 与 dense 也均选出相同的两个 token；但两组
top-10 分数并不相同，且首两个 token 的最小 top-1 优势约为 2，不能据此证明
长 decode 的数值一致性。延长到 24 token 后，dense 返回组仍 12/12 完全一致；
paged 与 dense 的原文为 9/12 一致，差异发生在 511、512、513 token 输入的
第 7、18、18 步。三处首分歧均为近似并列的候选，dense top-2 差值
`0.09375/0.09375/0.03125`，paged 为 `0/0.03125/0.09375`。输入文本随长度改变，
两份 engine 的 FP16 GEMM tactic 也不同，因此不能把这三处差异单独归因于跨页
读取。原始证据为本地忽略目录中的 `p3_page_boundary_artifacts.tgz` 和
`p3_page_boundary_decode24_artifacts.tgz`，同目录保存逐输入分析 JSON。

长输入 dense/paged 严格数值一致性与 page-list 实参核验仍是部署门禁；
本轮不推断 paged KV 加速，也不替换默认 engine。

## 与当前实验矩阵的关系

- P2 runtime 修复属于同一 maxBatch=4 engine 的运行时 A/B，不覆盖正式 level0/level1。
- P3 paged KV 必须新建独立 engine 目录和 provenance，不覆盖 contiguous KV engine。
- speculative decoding、LoRA、FP8/NVFP4 和质量指标不进入本阶段变量。
- 所有 latency 结论仍以固定 workload 的低层 benchmark 为准，HTTP RTT 只作为链路指标。
