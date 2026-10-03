# 当前实现状态

## 2026-09-22 TensorRT P2 native microbatch revalidation and P3 paged-KV source boundary

- 固定 Edge-LLM v0.9.1 checkout 的 max batch=4 engine 已确认可构建；上次失败属于请求级
  active-batch/eviction 状态管理，不是 builder 不能生成 batch=4 engine。
- 板端已应用并编译 runtime 修复：vanilla decoder 普通 batch=1 decode 使用 device-side
  sampling indices，batch eviction 压缩 sampling indices，空 batch 跳过 base/draft cache
  compact 和不必要的同步。重编译后低层 batch=1 为 `29.4693 ms`，batch=4 为 `43.7650 ms`，
  两个进程均 exit code 0，batch=4 CUDA Graph capture 成功。
- 旧 Python HTTP server 的并发调用会共享 VLM visual execution context；已在固定 checkout 的
  server 层增加单 worker、2 ms admission window、最大 native batch=4 的 microbatch scheduler，
  并让流式/非流式 route 共用 dispatch。v0.9.1 pybind 的可选字段和只读 finish reason 也已
  做兼容处理。
- 板端 c=2、c=4 各完成 3 次重复：均为 `60/60`，aggregate output tok/s 分别为 `27.5115`
  和 `41.0209`；c=4 100 请求 soak 为 `100/100`、`40.2069 tok/s`。错误扫描未发现 HTTP500、
  `slots missing`、`already loaded binary graph`、native microbatch failure；仍待补充显存/RAM/
  swap telemetry，故不直接替换默认 runtime。
- 新增 `docs/tensorrt-dynamic-batching-paged-kv.md`，将 P3 拆为 builder/config、AttentionPlugin
  page pool/page table binding、runtime page allocator 和验证门禁。当前 checkout 虽有 paged
  FMHA/XQA kernel 符号，但仍缺少完整 builder/plugin/runtime 接线；对照 v0.10.x 后确认该路径
  伴随 page table、pool allocator 和 attention/runtime 大范围接口变化，固定 v0.9.1 暂不做
  直接跨版本复制；审计 gate 已加强为同时要求 plugin paged binding 与 runtime page allocator，
  不能仅凭 ABI 符号构建 paged engine。

## 2026-09-06 TensorRT INT4 GEMM CTA-N=256 Jetson compile and smoke validation

- 在 Jetson Orin Nano、固定 Edge-LLM v0.9.1 revision `7f061f21f0a581ba234a1e233c9315b89d8e47d6`
  上，基于独立临时 worktree 应用现有兼容性补丁和 `0039-int4-gemm-cta-n256.patch`，
  成功编译 `NvInfer_edgellm_plugin`；候选 plugin target SHA-256 为
  `c7739d3be7c647c013cd576e5918ef8e907a4b6c96598d3297c88edd68f10e85`。
- nvlink 符号确认编译产物包含 `gemm_w4a16_T2<64,256,64,...>`，说明 `CTA_N=256`
  不是只写入 metadata，而是进入了 plugin 二进制。正式 plugin、level0/level1 engine
  和默认配置均未覆盖。
- 通过 `EDGELLM_PLUGIN_PATH` 注入候选 plugin 后，复用现有 INT4 `i768/k1024` engine
  启动服务，`/health` 返回 200；对冻结样本 `001.jpg` 的一次真实 multimodal 请求返回
  HTTP 200 和 JSON completion。日志同时确认 CUDA decoding graph capture 成功。
- 以上只证明候选 plugin 的加载、engine 反序列化和请求链路可用；尚未形成候选与正式
  level0 plugin 的严格 3×20 性能 A/B，因此不能宣称 CTA-N=256 已降低 TTFT、decode 或
  E2E，也不能替换默认 engine。

## 2026-09-06 TensorRT INT4 GEMM CTA-N=256 strict A/B result

- 在同一 Jetson Orin Nano、同一 `i768/k1024` INT4 engine、同一服务进程参数和同一
  `ps20_pilot_v1` 20 样本×3 repetition 下，正式 plugin 与 CTA-N=256 candidate 均为
  `60/60` completed、`0` failed；候选没有 OOM、HTTP error 或 CUDA Graph capture failure。
- 正式 plugin SHA-256 为
  `9437996d36b659092e7d4da244b43da6feb7c69e79a5654a8df53e9f5c2ac7eb`，候选 plugin
  SHA-256 为 `c7739d3be7c647c013cd576e5918ef8e907a4b6c96598d3297c88edd68f10e85`。
- 流式客户端 TTFT p50：正式 `787.93 ms`，候选 `861.93 ms`，候选慢约 `9.39%`；
  完整 E2E p50：正式 `10529.88 ms`，候选 `10603.54 ms`，候选慢约 `0.70%`。
  候选在 TTFT p90/p99 也分别为 `865.52/867.73 ms`，高于正式的
  `791.05/793.22 ms`。
- 该服务未在 SSE 中返回 usage、服务端 prefill 或 decode 字段，因此本轮只将 TTFT、
  E2E、完成率和失败率作为有效 A/B 指标；不从 HTTP RTT 推导 decode tokens/s。
- 结论：CTA-N=256 在当前多模态 workload 上否决，不进入默认 plugin 或后续组合；
  原始证据保存在 `reports/jetson-tensorrt-revalidation/cta_n256_*`，正式 level0/level1
  engine 未修改。

## 2026-09-06 TensorRT INT4 GEMM CTA-M=128 screening result

- CTA-M=128 plugin 在同一 Jetson Orin Nano、同一 engine 和同一 v0.9.1 compatibility
  base 上完成编译；plugin target SHA-256 为
  `0e4de226e2528aa0d9f09267c87385f54e7a556c5a8b2daa18028a870ea875e8`。
- 5 个冻结测试样本的筛选请求均完成，`0/5` failed，CUDA Graph capture 和服务启动均
  成功。与同序正式控制样本逐样本对齐后，候选 TTFT 增加约 `67.2–71.3 ms`，E2E
  增加约 `51.5–71.9 ms`；候选筛选 p50 为 `860.14 ms` TTFT、`10711.65 ms` E2E。
- 该方向在首轮筛选即表现为稳定回归，不进入完整 3×20；原始证据保存在
  `reports/jetson-tensorrt-revalidation/cta_m128_screen_*`。正式 level0/level1 和
  默认 plugin 未修改。

## 2026-09-06 TensorRT patch-chain reproducibility checker

- 新增 `scripts/check_tensorrt_patch_chain.py`，要求固定 Edge-LLM revision，并在临时
  detached worktree 中按给定顺序执行 `git apply --check` 与 apply；正式 checkout 不会被修改。
- 输出报告记录补丁顺序、文件名和 SHA-256，适用于 greedy/reduced-vocab、XQA/FMHA、INT4
  GEMV/GEMM 等需要多补丁叠加的候选。该工具只验证源码级可应用性，不代替 plugin 编译、
  runtime 加载、kernel 选择或板端性能结果。

## 2026-09-06 TensorRT Edge-LLM v0.9.1 paged-KV capability boundary

- NVIDIA v0.9.1 release notes 明确列出 paged KV-cache prefill 和 paged XQA decode；因此
  “v0.9.1 完全不支持 paged KV”是不准确的。能力是否可用于本项目，仍需以固定
  `7f061f21f0a581ba234a1e233c9315b89d8e47d6` checkout 的 builder、runtime 和 engine
  metadata 为准。
- 当前 Qwen3-VL INT4 engine 的 runtime summary 为 `usePagedKVCache=false`，所以只能确认
  本次 baseline 使用 contiguous 路径；不能从源码 ABI、release note 或 kernel 名称推断
  当前请求已经走 paged layout。
- 对固定 v0.9.1 checkout 的源码快照复核显示，`examples/llm/llm_build.cpp` 只暴露
  `maxKVCacheCapacity`，没有 `maxKVPoolPages`；`cpp/plugins/attentionPlugin/attentionPlugin.cpp`
  在两个 attention 构造路径都将 `usePagedKVCache` 设为 `false`。因此该 revision 的 paged
  KV 是源码/ABI 能力残留，但没有接入当前 builder-to-plugin 运行链路。
- 新增 `scripts/audit_edgellm_kv_cache.py`，在 Jetson checkout 上扫描 builder pool option、
  paged runtime/pool/XQA symbols，并检查 attention plugin 是否将
  `usePagedKVCache` 硬编码为 `false`；报告可输出
  `paged_kv_path_exposed_requires_runtime_validation`、`paged_kv_not_wired_in_checkout`、
  `paged_kv_source_evidence_incomplete` 或 `paged_kv_not_detected_in_checkout`。报告只做
  源码边界审计，不替代运行时日志、kernel 参数和 Nsight 证据。
- 官方入口：[v0.9.1 release discussion](https://github.com/NVIDIA/TensorRT-Edge-LLM/discussions/138)、
  [llm_build.cpp](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/main/examples/llm/llm_build.cpp)。
  固定 v0.9.1 若审计发现 builder 没有 pool 参数且 attention plugin 硬编码关闭，paged flow
  应作为兼容性探针失败并停止，不能继续构建或宣称可用；只有审计结论为完整路径暴露后，
  才测试 page pool 容量、地址表生命周期、请求释放复用和动态 batch 吞吐；不与 builder
  level、workspace 或 CUDA Graph A/B 混合。

## 2026-09-06 TensorRT paged-KV candidate build entry

- `build_qwen3_vl_2b_int4_awq_llm_i768_k1024_opt1_paged_kv.json` 保留为独立 LLM 兼容性探针，
  保持 Qwen3-VL-2B、INT4 AWQ、`i768/k1024`、batch=1、workspace `1024 MiB` 和 builder
  level1，只额外尝试传入 `--maxKVPoolPages 16`；该值按 64-token page 对齐 1024 capacity。
  固定 v0.9.1 若审计未发现完整 build/runtime 入口，该 flow 会在构建前失败，不产生 paged
  engine，也不改变正式 level0/level1。
- `build_edgellm_vlm_engines.py` 现在可选透传 `--max-kv-pool-pages`，默认省略时完全保持
  原有 contiguous/default 构建行为；paged candidate 不覆盖正式 level0/level1 engine。
- 两个 paged flow 同时启用 `--verify-paged-kv-source`：构建前检查固定 revision 的 builder
  pool option，成功后把完整源码审计报告写入 build report；缺少入口时在构建前失败。
- benchmark compare 将 `max_kv_pool_pages` 作为可选固定 identity 字段：旧 baseline 两侧
  均缺失时保持兼容，但 baseline/candidate 一侧出现 `16` 时会标记为 mismatch，避免 paged
  与 contiguous 结果被错误合并。
- `run_edgellm_benchmark.py` 新增 `--engine-max-kv-pool-pages`，在未直接传入 build
  provenance 时也能把 page pool 配置写入运行 metadata；该字段只描述 engine 身份，不
  会改变请求并发或推理行为。
- 构建成功后必须先核对 `usePagedKVCache`、page 数/容量和 XQA 参数，再做同一 20 样本的
  TTFT、decode、E2E、峰值内存和严格 JSON A/B；若 builder 不接受该参数，记录为版本/checkout
  兼容性失败，不将其解释为推理性能结论。
- candidate evaluator 新增 `--require-paged-kv-active`；paged-KV 晋级评测必须提供 runtime
  summary 且明确报告 `attention.use_paged_kv_cache=true` 与正数
  `attention.max_kv_pool_pages`，否则即使性能提升也不会放行。
- 另新增 `maxBatchSize=4、maxKVPoolPages=64` 的动态 batch paged-KV flow；它面向并发吞吐
  和物理 page 复用，不与 batch=1 延迟 candidate 混为同一实验。64 pages 按 4 个请求、每个
  1024-token capacity 和 64-token page 的估算设置，最终以构建日志暴露的 page size/capacity
  校验。

## 2026-09-06 TensorRT reduced-vocab + INT4 lm_head compound candidate

- 新增 `32768/65536` 两组 `export`/`build` flow，将已有 packed INT4 AWQ `lm_head` 候选与
  plus-history reduced vocabulary 组合，保持 batch=1、`i768/k1024`、workspace `1024 MiB` 和
  builder level1；`65536` 作为覆盖更保守的首选候选，`32768` 作为更激进的对照。
- 该组合复用 `calibration_provenance.json`、`selection_report.json` 和
  `reduced_vocab_65536_ps20_character_coverage_plus_history.json` 三类离线证据；后者在冻结
  `ps20_pilot_v1` 上覆盖 `632/632` reference token、`20/20` 样本无缺失。
- 这是候选 flow，不代表 Edge-LLM exporter 已支持该组合，也没有板端性能或质量结论。只有
  export/build 成功后，才继续做严格 JSON、风险准确率、事件 micro-F1、lm_head kernel、TTFT、
  decode、E2E、峰值内存和 soak test；正式 level0/level1 engine 不变。

## 2026-09-06 TensorRT INT4 GEMM CTA-N=512 static rejection

- 复核 `patches/tensorrt-edge-llm/0045-int4-gemm-cta-n512.patch` 后发现，固定 v0.9.1
  W4A16 GEMM 在 `CTA_M=64/CTA_K=64/STAGES=4` 下的 shared memory 计算为 `102400` bytes，
  超过源码 `static_assert` 的 `99*1024=101376` bytes 上限。
- 固定 Qwen3-VL-2B INT4 线性层的已审计输出维度 `1024/2048/6144` 均 512 对齐，
  但这不能抵消编译期资源约束。0045 现在只保留为负向资源校验样例，不进入 build queue；
  若要研究 512 tile，必须重新设计 tile/stages 并作为新的 compound candidate 单独验证。
- 补丁 SHA-256：`3d442c309060b231ffa699fdcea33926b73f68a7a66c75cd8968684b4577fe2e`。
- 新增 `scripts/audit_int4_gemm_candidates.py`，在板端 build 前复现固定 v0.9.1 的
  `kSmemByteSize` 公式、warp/thread 上限和 N 对齐检查；CTA-N=512 被自动标记为不可进入
  build queue，CTA-N=256、CTA-M=128、CTA-K=128 仍需真实编译和板端 A/B。

## 2026-09-06 TensorRT INT4 GEMV runtime sweep matrix

- 新增 `configs/tensorrt/jetson_orin_nano_int4_gemv_runtime_v1.json`，固定
  Qwen3-VL-2B、INT4 AWQ、`i768/k1024`、batch=1、CUDA Graph 和 15W，定义
  `NPerBlock=2/4` 以及 `128/256/512 threads/block` 的可审计变体。
- 变体只记录 runtime/kernel 选择；实际生效仍须由 `serve_edgellm.py` 的 CLI 参数、
  运行时 metadata、plugin patch chain 和板端日志共同证明。该矩阵没有新增性能结论。
- `serve_edgellm.py` 现在支持 `--runtime-tuning-config` + `--runtime-variant`，会注入
  矩阵参数并把配置路径/variant id 写入 metadata；显式 CLI 冲突会直接失败，减少手工
  启动参数与实验记录不一致的风险。
- GEMV 矩阵的每个 variant 现在额外声明 `runtime_patches`；服务入口会自动注入并记录补丁
  provenance，若命令行显式传入的 patch 顺序不同则直接失败。该自动注入不执行补丁，仍需
  先通过 patch-chain 校验并在对应 checkout 中完成编译。
- `serve_edgellm.py` 新增可选 `--verify-runtime-patch-chain`，启动前执行同一临时 worktree
  校验，并将成功报告写入 runtime metadata；默认关闭，不改变既有服务行为。

## 2026-09-06 TensorRT INT4 GEMV block-size candidate

- 新增 `patches/tensorrt-edge-llm/0049-int4-gemv-block-size-candidate-v2.patch`，保留默认
  `256` threads/block，并为 `NPerBlock=2` 的 decode GEMV 增加显式 `128/512` threads/block
  路径，由 `EDGELLM_INT4_GEMV_BLOCK_SIZE` 选择；同时修正 `0038` 候选 NPerBlock=4
  launch 对传入 CUDA stream 的遗漏。
- 候选只改变 GEMV 的 block 并行度；原有 `NPerBlock=4`、`M=1` 路径和默认 `256` 路径保留。
  需要在 Orin 上结合寄存器数、occupancy、GEMV kernel 时间和逐 token 对齐验证，当前没有
  板端性能证据，不能提前宣称 decode 加速。
- 补丁 SHA-256：`fc9c4db72e578b3cca2a6f455b4df4db5916aee52a2c441c510e4a801d661ef8`。

## 2026-09-06 TensorRT visual-profile A/B guard

- `scripts/compare_tensorrt_benchmarks.py` 现在在存在视觉 benchmark 元数据时额外校验
  `min_image_tokens`、`max_image_tokens` 和 `max_image_tokens_per_image`；旧的 LLM-only
  summary 仍兼容，但一侧缺失视觉 profile 或两侧 profile 不一致时不会判定为可比。
- 该校验用于隔离 visual builder level 2/3 与现有 visual level1 的实验变量，不代表任何
  性能收益；仍需结合 visual engine SHA、Engine Inspector、Nsight 和阶段级 benchmark。
- 低层 benchmark comparison 现在同时比较 `vision_encode_ms` 和 `model_generate_ms`，使
  visual builder A/B 可以单独观察视觉编码阶段，而不把变化全部归入 E2E 或文本 prefill。
- `record_engine_provenance.py` 新增 `--build-report`，会校验成功构建、输出路径、文件大小
  和 SHA-256，并自动继承 visual component、builder level、workspace 与 image-token profile。
- 新增 visual level2/3 专用 timing-cache binding sidecar，固定 `component=visual` 和
  visual builder 的默认 profile；timing cache 仍只作为构建时间变量，不计入推理收益。

## 2026-09-06 TensorRT FMHA head_dim=128 tiled candidate

- 新增 `patches/tensorrt-edge-llm/0043-fmha-head128-tiled-candidate.patch`，针对当前
  `SM87 + FP16 + head_dim=128` 长序列路径，显式切换到已编译的 tiled FMHA cubin。
- 默认启发式、其他 SM/head_dim 和短序列均不变；`serve_edgellm.py --fmha-force-granular-tiling`
  会设置环境变量并写入 runtime metadata，便于核对候选是否真正生效。
- 补丁 SHA-256 为 `4f7cfb1603895a9410b06808389f26cee241874e24989050c6f489fcb2ec7f38`。
- 当前 level1 trace 的 FMHA 类别约 `70.7 ms`，但该候选尚未在 Jetson 编译或验证，不能提前
  宣称 attention kernel 或 TTFT 加速。

## 2026-09-06 TensorRT INT4 GEMM CTA-K=128 candidate

- 新增 `patches/tensorrt-edge-llm/0042-int4-gemm-cta-k128.patch`，保持 `CTA_M=64`、
  `CTA_N=128`，将 W4A16 GEMM 的 K tile 从 `64` 改为 `128`。
- 固定 `STAGES=4` 时动态 shared memory 为 `99,328` bytes，低于源码 `99*1024` 的限制，
  但接近上限；该候选与 `0039/0041` 独立构建，不改变正式 plugin 或 engine。
- 尚未在 Jetson 编译，未取得 launch、W4A16 kernel、prefill/TTFT/E2E、显存或数值对齐证据。

## 2026-09-06 TensorRT INT4 GEMM CTA-M=128 candidate

- 新增 `patches/tensorrt-edge-llm/0041-int4-gemm-cta-m128.patch`，在保持 `CTA_N=128`
  的前提下，将 prefill W4A16 GEMM 的 `CTA_M` 从 `64` 改为 `128`。
- `CTA_M=128/CTA_N=128/STAGES=4` 按固定 v0.9.1 源码公式（含 `CTA_N` scale buffer）的动态
  shared memory 为 `82944` bytes，低于 Orin Nano 的 99 KiB 限制；该候选与 `0039` 分开
  构建和测量，当前未改变正式 plugin 或 engine。
- 尚未在 Jetson 编译，未取得 W4A16 kernel、prefill/TTFT/E2E、显存或数值对齐证据。

## 2026-09-06 TensorRT CTA-N=256 shape audit

- 固定 Qwen3-VL-2B 文本 backbone 的 INT4 linear 输出维度为 `2048/1024/6144`，均可被
  `256` 整除；视觉 encoder 不经过该 W4A16 plugin，正式 `lm_head` 仍为 FP16。
- 已将 `0039` 的 dispatcher 注释改为引用统一的 `kernel::kGemmCtaN`，并增加无硬件 shape
  contract 测试；候选仍需板端编译、逐 token 对齐和性能 A/B，尚未改变默认 engine。

## 2026-09-06 TensorRT FMHA selection cache candidate

- 新增 `patches/tensorrt-edge-llm/0040-fmha-selection-cache.patch`，在 FMHA context
  prefill 路径缓存 immutable kernel list 和完整 `FMHAKernelHashKey` 对应的 function，
  避免重复 loader mutex、unordered-map lookup 和 function copy。
- 该补丁不改变 FMHA cubin、mask、launch grid、Q/K/V 地址或数值语义；仅在显式应用补丁
  后生效，尚无 Jetson 编译、CPU/API trace 或端到端收益证据。

## 2026-09-06 TensorRT INT4 GEMM CTA-N=256 candidate

- 新增 `patches/tensorrt-edge-llm/0039-int4-gemm-cta-n256.patch`，将固定 v0.9.1
  W4A16 GEMM 的 `kGemmCtaN` 从 `128` 改为 `256`，目标是减少 N 方向 output tile 数。
- 该候选是独立 plugin build-time 变量；当前正式 plugin 和 level0/level1 engine 不变。
  `CTA_N=256` 的估算 shared memory 约 66 KB，仍在 Orin Nano SM87 的 99 KB 上限内，
  但是否优于 128 必须由 kernel 时间和端到端 A/B 决定。
- 已完成 Jetson plugin 编译、单请求 smoke 和严格 3×20 A/B；候选在 TTFT 与 E2E 均未
  达到收益门槛，已按负向结果退出后续 build queue。不能把 262 ms 的 W4A16 trace 热点
  归因到该候选，也不将该候选写入默认配置。

## 2026-09-06 TensorRT INT4 GEMV N-per-block candidate

- 新增 `patches/tensorrt-edge-llm/0038-int4-gemv-nperblock4.patch`，针对 `M=1` 且
  `N % 16 == 0` 的 INT4 GEMV 编译 `NPerBlock=4` 路径；默认 `NPerBlock=2` 不变。
- 候选每个 block 覆盖 16 个输出通道，理论上将 output-channel block 数减半；只改变
  GEMV 的输出通道分块，不改变 INT4 权重布局、group scale 或计算结果语义。
- `serve_edgellm.py --int4-gemv-n-per-block 4` 会显式设置环境变量并写入 runtime metadata；
  该候选尚未在 Jetson 编译或测得 GEMV/TTFT/decode/E2E 收益，不能提前计入性能结论。

## 2026-09-06 TensorRT KV-cache 路线边界复核

- 固定 Edge-LLM v0.9.1 的 XQA ABI 与 release note 均显示 paged KV 是版本能力；但当前
  Qwen3-VL INT4 engine 的 runtime summary 为 `usePagedKVCache=false`，当前性能结论仍只
  属于 contiguous-KV baseline。
- 因此 paged KV 不能仅凭 ABI 字段或“服务成功启动”判定生效；必须同时核对 builder
  配置、engine/runtime metadata、实际 XQA 参数以及 Nsight kernel trace。
- 后续如进入连续请求/动态 batch 阶段，paged KV 应作为独立 runtime/cache A/B：先由
  `audit_edgellm_kv_cache.py` 确认固定 checkout 的入口，再完成 page pool、page table、
  请求释放/复用和地址校验，最后测并发吞吐与容量；不与 builder level、workspace 或
  CUDA Graph A/B 混合。

## 2026-09-06 TensorRT XQA selection cache candidate

- 新增 `patches/tensorrt-edge-llm/0037-xqa-selection-cache.patch`，在 `0036` 的
  thread-local XQA list cache 之上，缓存按 runtime key 选出的 `XQAKernelFuncInfo`，避免
  普通 decode 每步重复执行 XQA function map lookup 和 variant ranking。
- selection key 同时包含 runtime key、SM、spec-decode 和 paged-KV 状态；因此不会在不同
  kernel 路由之间复用函数。补丁不改变 kernel、launch grid、KV 地址或数值语义，默认不启用。
- `0037` 已对固定 v0.9.1 源码归档通过 `git apply --check`；尚未在 Jetson 编译、采集
  CPU/API trace 或取得 TTFT/decode/E2E 证据，不能提前宣称 attention 加速。

## 2026-09-06 TensorRT greedy warp-reduction candidate

- 新增 `patches/tensorrt-edge-llm/0035-greedy-warp-reduction.patch`，在现有 greedy top-1
  CUB reduction 控制路径之外，增加 warp-shuffle 加共享内存汇总的候选；它同时兼容
  `0027` 的 reduced-vocab mapping 和 `0029` 的 256/1024 block-size 选择。
- 通过 `EDGELLM_GREEDY_ARGMAX_IMPL=warp` 或 `serve_edgellm.py --greedy-argmax-impl warp`
  显式启用；未设置或使用 `cub` 时仍走 CUB 控制路径，故不会改变正式 engine 的默认行为。
  runtime metadata 会记录该环境变量，候选启动时还应传入 `--runtime-patch` 记录补丁 SHA-256。
- 已通过固定 Edge-LLM v0.9.1 源码的 `0024 -> 0027 -> 0029 -> 0035` 补丁链校验；尚无
  Jetson 编译、逐 token 对齐或性能证据，不能提前计入 decode 加速。

## 2026-09-06 TensorRT XQA host lookup cache candidate

- 审计 v0.9.1 `dispatchXQAKernel` 后确认，普通 decode 每一步都会通过全局 loader 查询相同
  的 XQA kernel list；loader 使用 mutex，随后还会按 runtime key 查找候选并选择 variant。
- 新增 `patches/tensorrt-edge-llm/0036-xqa-thread-local-list-cache.patch`，为每个 host
  thread 缓存固定 `(dtype, KV dtype, SM, spec-decode, paged-KV)` 对应的 immutable list，
  仅在 key 变化时回到原 loader。该补丁不改变 kernel、launch grid、KV 地址或数值语义。
- 已通过固定 Edge-LLM v0.9.1 源码的完整 `0024 -> ... -> 0035 -> 0036` 补丁链校验；尚无
  Jetson 编译、CPU/API trace 或端到端性能证据，不能宣称 attention 加速。

## 2026-09-06 TensorRT INT4 W4A16 pipeline-stage reproducibility

- Nsight Systems GPU trace 将 `gemm_w4a16_T2` 识别为当前 INT4 单请求聚合热点：level0/level1
  均约 `262 ms`，因此新增 `patches/tensorrt-edge-llm/0032-configurable-int4-gemm-stages.patch`。
- 补丁将现有 `STAGES=4` 提取为模板 launch helper，保留 `4` 为默认控制，并通过
  `EDGELLM_INT4_GEMM_STAGES=2` 或 `3` 选择较浅 pipeline；每个实例独立设置动态 shared
  memory，避免在 CUDA Graph capture 期间调用 `cudaFuncSetAttribute`。
- 该补丁只改变 W4A16 GEMM 的编译期 pipeline 深度，不改变权重、量化 scale、tile 或输出
  语义；此前 `0021/0022` 已完成同一 level1 engine 的板端 `STAGES=2/3/4` 复验：
  `STAGES=2` 的 TTFT/E2E p50 相对 `4` 慢约 `5.38%/2.14%`，`STAGES=3` 严格 JSON 为
  `0/20`，因此正式插件继续使用 `STAGES=4`。
- `0032` 将该实验改为单一可复现的 runtime 环境开关，`serve_edgellm.py` provenance 已
  记录该变量；它用于后续重现和 Nsight kernel 摘要，不构成新的加速结论。

## 2026-09-06 TensorRT Nsight kernel aggregation

- 新增 `scripts/summarize_tensorrt_nsight_trace.py`，将 Nsight Systems 的
  `cuda_gpu_trace.csv` 按 W4A16、FMHA/XQA、sampling、embedding、memcpy 和其他 kernel
  分类，输出每类的调用次数、总时延、分位数和 top-kernel 列表，并保存输入 trace SHA-256。
- 该摘要明确排除 memcpy 后再计算 `kernel_duration_ms`，用于后续 `STAGES=2/3/4` A/B 的
  kernel 归因；它不把 kernel 名称映射为模型 layer，也不替代 TTFT/decode/E2E 或数值门禁。
- 已对现有 level0/level1 trace 生成
  `reports/jetson-tensorrt-revalidation/int4_opt0_nsight_kernel_summary.json` 和
  `int4_opt1_nsight_kernel_summary.json`：总 kernel 时间由 `684.795 ms` 降至
  `565.230 ms`（约 `-17.46%`），但 W4A16 由 `262.312 ms` 变为 `262.390 ms`
  （约 `+0.03%`），FMHA 由 `75.264 ms` 降至 `70.680 ms`。这支持“level1 主要改变
  TensorRT tactic/其他生成路径”的解释，不支持把收益归因给 W4A16 kernel。
- 新增 `scripts/compare_tensorrt_nsight_summaries.py`，并生成
  `reports/jetson-tensorrt-revalidation/int4_opt0_vs_opt1_nsight_comparison.json`；该报告
  统一记录每个 trace/category/top-kernel 的 `candidate - baseline` 差值，后续候选可沿用
  同一格式进行单变量归因。当前差异报告仍只代表 GPU trace，不替代 TTFT、decode、E2E、
  质量、内存、功耗和 soak 门禁。

## 2026-09-06 TensorRT XQA attention selection diagnostic

- 已审计固定 Edge-LLM v0.9.1 的 XQA 选择逻辑：runtime key 包含 Q/KV dtype、head
  dimension、Q/KV head ratio、sliding window 和 `tokensPerPage`；同一 key 按 kernel
  variant priority 选择候选，当前 `SM87 + FP16 KV + head_dim=128 + ratio=2 + contiguous
  KV + spec_decode=false` 对应普通 XQA 路径。
- 新增 `patches/tensorrt-edge-llm/0028-xqa-selection-diagnostic.patch`。设置
  `EDGELLM_LOG_XQA_SELECTION=1` 时，每个 XQA kernel list 只记录第一次实际选择的
  function、head dimension、M tile、variant、SM、spec-decode 和 paged-KV 状态；默认关闭，
  不改变 kernel 选择和运行路径。
- 补丁已对固定 Edge-LLM v0.9.1 源码归档通过 `git apply --check`，当前仅有源码级和
  patch-chain 证据，尚未在 Jetson 编译、采集实际 XQA function 或据此宣称 attention
  加速。板端日志确认后，再决定是否增加 shape-specific kernel 或调整 XQA 配置。

## 2026-09-06 TensorRT greedy argmax block-size candidate

- 新增 `patches/tensorrt-edge-llm/0029-greedy-argmax-block-size.patch`，在已有 `0024`
  /`0027` greedy top-1 kernel 上保留 `256` 线程控制路径，并增加 `1024` 线程单 block
  归约候选；通过 `EDGELLM_GREEDY_ARGMAX_BLOCK_SIZE=1024` 选择候选，其他值和未设置时
  使用 `256`。
- 该候选针对完整 Qwen 词表单 batch 行的归约并行度，未改变 logits、reduced-vocab
  映射或 top-k/top-p 语义。已通过固定 v0.9.1 源码的补丁链校验，尚无 Jetson 编译、数值
  对齐或性能证据，不能提前计入 decode 加速。

## 2026-09-06 TensorRT direct device-token embedding candidate

- 新增 `patches/tensorrt-edge-llm/0030-direct-device-token-embedding.patch`：在 vanilla
  decode、无 layer debugger、无 Gemma4 PLE 且显式设置
  `EDGELLM_DIRECT_DEVICE_TOKEN_EMBED=1` 时，直接把已在 GPU 上的 `sampling.indices` 作为
  embedding lookup 输入，跳过 `sampling.indices -> idsInput` 的 D2D copy；其他路径保持原实现。
- 该候选依赖 `0026` 对 dynamic batch 的 sampling index 压缩保证，并增加 batch/shape 校验。
  已通过固定 v0.9.1 源码的补丁链校验，尚无 Jetson 编译、逐 token 对齐或性能证据。

## 2026-09-06 TensorRT candidate environment provenance

- `serve_edgellm.py` 的 runtime metadata 现在记录 `EDGELLM_LOG_XQA_SELECTION`、
  `EDGELLM_GREEDY_ARGMAX_BLOCK_SIZE`、`EDGELLM_GREEDY_ARGMAX_IMPL`、
  `EDGELLM_DIRECT_DEVICE_TOKEN_EMBED` 和
  `EDGELLM_INT4_GEMM_STAGES`，并新增对应 CLI 参数，使 XQA 诊断、greedy block-size、
  direct embedding 和 INT4 stages A/B 的启动状态可追溯。省略候选参数时保持原有环境，
  不改变正式 engine 的默认行为。

## 2026-09-06 TensorRT empty-batch eviction compaction candidate

- 审计 v0.9.1 `performBatchEvict` 后发现 `newActiveBatch==0` 时仍会调用 base
  `HybridCacheManager::compactBatch`，从而为没有 survivor 的请求结束路径发起每层 KV/Mamba
  compaction kernel，随后再次同步；该路径不需要搬运任何 batch 数据。
- 新增 `patches/tensorrt-edge-llm/0033-skip-empty-batch-compaction.patch`：仅在存在 survivor
  时执行 cache compaction，仍保留 `setActiveBatchSize(0)`、strategy eviction callback 和
  CPU context 收尾。补丁已对固定 v0.9.1 源码归档通过 `git apply --check`，SHA-256 为
  `039c2773490b0c23bb3f24084e5471c277d88073df4c4a2287a1a588b2da1c6a`。
- 这是 decode 请求结束路径的源码级候选，尚无 Jetson 编译、Nsight、TTFT/E2E 或动态 batch
  数值证据，默认不启用，也不计入正式性能结论。

## 2026-09-06 TensorRT empty-batch eviction synchronization candidate

- 在 `0033` 基础上继续审计固定 v0.9.1 的 vanilla、DFlash、Eagle、MTP 和 Gemma4 MTP
  eviction callback：当 `newActiveBatch==0` 时均不需要读取 GPU batch mapping；base/draft
  compaction 也已由 `0033` 跳过。
- 新增 `patches/tensorrt-edge-llm/0034-skip-empty-batch-eviction-sync.patch`，在全量结束
  路径跳过 mapping 的 H2D 上传和 compaction 后的尾部 stream synchronization；仍保留
  strategy callback、active-batch 状态和 CPU context 收尾。该补丁依赖 `0033`，完整链式
  `git apply --check` 已通过，SHA-256 为
  `9692e49d9c68accde98dfe5d56bb3ca0959f744cc7718449fb84015681d6a339`。
- 这是 decode 请求结束路径的源码级候选，尚无 Jetson 编译、Nsight、TTFT/E2E 或动态 batch
  数值证据，默认不启用，也不计入正式性能结论。
- 新增的 CLI 配置和可逆性测试纳入本地回归，当前无硬件测试为 `201/201` 通过；这只证明
  参数配置与 provenance 记录逻辑，不替代 Jetson 编译和候选性能证据。
- `--runtime-patch` 现在可重复传入并把实际补丁文件的路径、大小和 SHA-256 写入 runtime
  metadata；候选环境变量因此能与源码实现绑定，避免未应用补丁时误判开关已生效。

## 2026-09-06 TensorRT direct device-token embedding host-path refinement

- 新增 `patches/tensorrt-edge-llm/0031-skip-unused-device-token-reshape.patch`：在
  `0030` direct-device 路径下不再对未使用的 `idsInput` 做 reshape；传统、debug 和 D2D
  fallback 路径仍执行原有 reshape。
- 该改动只减少一个 host-side shape metadata 操作，默认关闭并依赖 `0030`；已通过源码
  补丁链校验，尚无板端性能证据。

## 2026-09-06 TensorRT greedy top-1 sampling kernel candidate

- 固定 v0.9.1 的 vanilla greedy decode 当前通过 `selectAllTopK(topK=1)` 走
  `topKStage1`（8 个 block）和 `topKStage2` 两阶段路径，并写入临时 logits/top-k
  workspace；泊车 workload 只需要每个 batch row 的 top-1 index。
- 新增 `patches/tensorrt-edge-llm/0024-greedy-argmax-fast-path.patch`：在
  `topK=1 && !topKValues.has_value()` 时使用单个 block 的 CUB argmax，复用原有
  `TopK_2` reduction 语义，避免临时 logits 写入和第二次 kernel launch。带 top-K value
  的 logprobs 路径保持原实现，不改变该接口语义。
- 补丁已对固定 Edge-LLM v0.9.1 源码归档通过 `git apply --check`；SHA-256 为
  `915b272a9efdf0e0e93f978304ef2c90e2c398fb66511887f1754d8d1fb0f5d3`。该结果只是
  源码级可应用性证据，尚未在 Jetson 上编译、用 Nsight 验证或计入正式 latency/quality 结论。

## 2026-09-06 TensorRT device-side token feedback candidate

- 新增 `patches/tensorrt-edge-llm/0025-device-token-feedback.patch`：vanilla decode
  在 `activeBatchSize==1` 且非 `layerDebugger` 路径时，直接将上一轮 GPU sampling index
  以 D2D copy 写入下一轮 `idsInput`，再执行 embedding lookup；其他 batch size、调试和
  teacher-forcing 路径保留 host token 构造和 H2D copy，避免 batch eviction 后复用未压缩的
  sampling buffer。
- 该候选减少每个 decode round 的 host token loop 和 H2D token copy，但仍保留 token
  D2H 与 stream synchronization，因为运行时仍需在 host 侧判断 EOS/stop words 并维护
  请求状态。
- 补丁已对固定 Edge-LLM v0.9.1 源码归档通过 `git apply --check`；SHA-256 为
  `1c3371fce51a10983288f963683c8da08d2c63552cc6825836e1f1bda462cd3c`。尚未在 Jetson
  编译或取得 decode、TTFT、严格 JSON 和 Nsight 证据，不计入正式性能结论。
- 本地无硬件回归测试为 `189/189` 通过，`git diff --check` 通过；该结果不替代板端
  编译和运行时验证。

## 2026-09-06 TensorRT dynamic-batch token feedback candidate

- 新增 `patches/tensorrt-edge-llm/0026-device-token-feedback-dynamic-batch.patch`，调用
  v0.9.1 已有的 `kernel::compactTensorBatch` 在 VanillaDecoder 的 `onBatchEvict` 中压缩
  `sampling.indices`，并在 `0025` 基础上允许非调试 vanilla dynamic batch 走 D2D token
  feedback。
- `newActiveBatch==0` 时不执行压缩；调试/teacher-forcing 仍保留 host 路径。该候选依赖
  `0024 -> 0025 -> 0026` 顺序，已对保存的 v0.9.1 源码归档完成链式 `git apply --check`，
  SHA-256 为 `83dd3c885285d3dc636072d9cb2f4cac75bbe1bd469de4fe6fa6dbcb41f607fc`。
- v0.9.1 的 `buildBatchMapping` 按原 batch 顺序生成连续的新 index，且通用 compaction
  kernel 支持 source/destination 原地使用；因此该候选的 in-place `sampling.indices` 压缩与
  runtime 的 slot mapping 语义一致，仍需运行时数值测试确认。
- v0.9.1 的 `cpp/CMakeLists.txt` 通过 `file(GLOB_RECURSE KERNELS_CU_SRCS
  "kernels/*.cu")` 和 `RUNTIME_CPP_SRCS` 纳入该 kernel 与 decoder；不需要额外修改
  CMake target，重新构建 `_edgellm_runtime` 即可进入编译验证。
- 该结果只证明源码补丁链和 batch-index compaction 逻辑可应用，尚未在 Jetson 编译、做
  dynamic-batch 数值对齐或取得吞吐/延迟/Nsight 证据，不能提前计入正式性能结论。
- 现有正式 level0/level1 engine 的 `maxBatch=1`，不能用并发 HTTP 请求推断 dynamic
  batching。新增 `configs/tensorrt/jetson_orin_nano_int4_dynamic_batch4_v1.json` 和
  对应 build flow，将 `maxBatchSize=4` 作为独立 engine 变量；只有该 engine 构建并在
  `concurrency=1/4` 下完成数值、吞吐、内存和 soak A/B 后，才可评价该候选。
- 另新增 `65536 reduced-vocab + maxBatch4 + level1` compound build flow，用于验证
  输出 `lm_head` 缩小后能否在统一内存设备上容纳更高 batch；该 flow 不替代单变量
  reduced-vocab 或 dynamic-batch 实验，结果必须单独标记为 compound candidate。

## 2026-09-06 TensorRT fused reduced-vocabulary greedy candidate

- 新增 `patches/tensorrt-edge-llm/0027-fused-greedy-reduced-vocab-map.patch`：仅在
  greedy top-1 路径把 reduced-vocab ID 到 full-vocab ID 的查表融合进 argmax kernel；
  top-k/top-p 路径仍使用原有独立映射，保持采样语义边界。
- 补丁 SHA-256 为
  `c0d0d1da7911bbf9b805aa3028f864191f2bf747e10470b2b66578a5de5efaa8`，已与
  `0024 -> 0025 -> 0026` 在固定 v0.9.1 源码归档上完成链式 `git apply --check`。
  该结果尚未包含 Jetson 编译、数值对齐或 Nsight 性能证据。

## 2026-09-06 TensorRT fallback-scan patch integrity

- 修正 `patches/tensorrt-edge-llm/0023-safe-skip-fallback-binding-scan.patch` 的 unified
  diff hunk 计数，补齐启用 binding-state cache 时的 `refreshPreparedBindingState()`，避免
  跳过 fallback scan 后复用过期的 host snapshot。
- 该补丁已对固定 Edge-LLM v0.9.1 源码归档通过 `git apply --check`；SHA-256 为
  `0f57032eeac166b93cb65b5dcbed6fd3e212cb114d5e2128520f43cfec40b5e2`，并新增 hunk
  完整性单元测试。该结果只证明补丁可应用，不替代 Jetson 编译、性能和正确性 A/B。

## 2026-09-06 TensorRT reduced-vocabulary 16384 候选否决

- 使用 level0/level1 历史成功输出生成了独立 `16384` map；历史输出 coverage 为
  `3079/3079`、`40/40` 样本通过，但独立 `ps20_pilot_v1` 仅覆盖 `299/632` token，
  token coverage 为 `47.31%`，且 `20/20` 样本均存在缺失 token。
- 该结果说明仅用历史生成结果选择更小词表会产生明显分布过拟合；`16384` 候选不进入
  export/build，当前保留 `32768/65536` history map 作为后续板端候选，并继续要求
  独立 holdout coverage、严格 JSON、任务质量和性能门禁。

## 2026-09-06 TensorRT reduced-vocabulary historical-output coverage

- `scripts/build_reduced_vocab_map.py` 和
  `scripts/evaluate_reduced_vocab_coverage.py` 现在支持读取 StudyReport 中成功请求的
  `raw_output`；失败记录不会被当作已覆盖，并单独保存报告路径和样本来源。
- 对现有 level0/level1 各 20 条历史输出进行检查，原 `65536` map 的覆盖率为
  `3076/3079=99.90%`，有 3 个样本缺少 token `45879/63365`，因此没有直接进入构建。
- 将历史输出纳入频次选择后生成独立增强候选：`65536` map 的 SHA-256 为
  `b38da7cf86c079303982d22bc59d12bdcc6dddd660f5d51a3c832a413cdf05e0`，`32768` map 的
  SHA-256 为 `d536672ab04536933f6a5d190242bac55e329b332a5e3853b0528fbfd5b23657`；两者
  对 40 条历史成功输出均达到 `3079/3079` token 覆盖、无缺失。该结果仍只是构建前
  可表达性证据，尚未构建 TensorRT engine 或证明质量/性能收益。
- 新增 `...reduced_vocab32768_history...` 和 `...reduced_vocab65536_history...` 的独立
  export/build flow，分别绑定增强 map、冻结 `ps20` holdout coverage report 和
  `i768/k1024`、level1、workspace `1024 MiB`；历史输出只参与 map 选择，不作为独立
  构建门禁。它们只提供后续板端实验入口，不改变正式 flow。
- 本地无硬件测试当前为 `189/189` 通过，`git diff --check` 通过。

## 2026-09-06 TensorRT packed-INT4 vocabulary alignment gate

- `scripts/validate_reduced_vocab.py` 新增 `--packed-int4-lm-head`，可在离线 map 校验阶段
  直接检查 reduced vocabulary 是否满足 packed INT4 `lm_head` 的 group size `128` 对齐。
- engine build wrapper 会在声明 `expected_lm_head_precision=int4_awq` 时复用该约束；
  非对齐 map 会在 `llm_build` 启动前失败，`32768/65536` 候选均通过结构对齐检查。

## 2026-09-06 TensorRT attention/KV runtime evidence

- `summarize_edgellm_runtime_log.py` 现在将 `numKVHeads`、`headDim`、`kvCacheDtype`、
  `usePagedKVCache` 和 `specDecodeType` 单独写入 `attention` 摘要，并在双日志比较中
  检查两侧路由事实是否一致。
- 该摘要只证明 runtime/configuration 路径；它不把存在 paged-XQA kernel、FMHA cubin
  加载或 speculative 配置误判为实际执行，具体 kernel/tactic 仍需 Inspector 和 Nsight。
- 对固定 v0.9.1 的当前 engine 只确认到 runtime summary 的 `usePagedKVCache=false`；release
  note 与源码能力不能替代实际 route 证据。因此 paged KV 是后续需要单独构建、加载和
  trace 验证的 runtime/cache A/B，不把它写成现有 engine 已启用或完全不支持。

## 2026-09-06 TensorRT reduced-vocabulary 独立构建入口

- 新增 `65536` 和 `32768` 两组独立 export/build flow，将现有 INT4 AWQ checkpoint
  与经过离线校验的联合语料 vocab map 连接为候选链路；两组均固定 level1、
  `i768/k1024` 和 workspace `1024 MiB`。
- export flow 显式传入 `--reduced-vocab-dir`，并将 `llm/vocab_map.safetensors` 与
  `llm/reduced_vocab.json` 列为构建输入/输出；build flow 保持 level1、workspace
  `1024 MiB`、`i768/k1024`，并显式校验 EOS token `151645`，不覆盖正式
  level0/level1 engine。
- `build_edgellm_vlm_engines.py --reduced-vocab-dir` 现在会同时核对 source map、ONNX
  export map 的 SHA-256，以及 ONNX `config.json.reduced_vocab_size` 与 metadata 的一致性；
  同时要求 source 目录中的 `selection_report.json` 与 map SHA-256、token count 一致；任一
  不一致都会在 TensorRT build 启动前失败。
- 当前字符覆盖候选的联合语料仍覆盖 `458` 个观测 token，但额外保留 tokenizer 中
  单 token 中文/JSON 字符 token；它们已通过冻结 holdout 的独立 coverage evaluator，
  共保留 `29,412` 个字符覆盖 token；仍不等于模型质量通过，engine 构建前必须保留
  coverage 报告。`65536`/`32768` map SHA-256 分别为
  `beeeebb1b071fce03a9fca0fc1e3f61202652edd895790b1b9d76d848fc9fca3` /
  `599da52117369d6544fa0655bdb41ab691e5deca636f251942a9a51932f041582`。
- 两组 reduced-vocab build flow 现在还要求独立 holdout coverage report；report 必须为
  `valid=true`、token/sample coverage 均为 `1.0`，且 map SHA-256 与 source map 一致，
  否则不会启动 `llm_build`。这只是构建前 token 可表达性门禁，不等价于模型质量通过。
- 对固定 v0.9.1 的 `reduce-vocab` 实现进一步核对：transformer 和输入 embedding table
  保持原始 `151936` 词表，只有导出 logits/`lm_head` 和 runtime sampling vocabulary
  使用 reduced size；因此当前 map 不会改变 prompt token 的 embedding lookup，但仍会限制
  可生成 token 集合，必须继续通过严格 JSON 和质量 A/B。

## 2026-09-06 TensorRT reduced-vocabulary holdout coverage

- 新增 `scripts/evaluate_reduced_vocab_coverage.py`，在 map 构建完成后独立检查冻结
  holdout 的参考输出 token 是否均可被 map 表达；该检查不修改 map，也不参与候选选择。
- 联合字符覆盖与历史成功输出策略的 `65536` 和 `32768` map 在冻结
  `ps20_pilot_v1` 的 20 个样本上均覆盖 `632/632` 个 token、`20/20` 个样本无缺失；
  对应报告为 `reports/tensorrt/reduced_vocab_65536_ps20_character_coverage_plus_history.json`
  和 `reports/tensorrt/reduced_vocab_32768_ps20_character_coverage_plus_history.json`。
- 该结果只表示 reference-token coverage，不等价于模型严格 JSON、风险准确率或事件 F1；
  仍需独立 engine A/B 和完整质量门禁。
- 该结果只表示参考 token 覆盖，不等价于模型严格 JSON、风险准确率或事件 F1；应优先
  增加不与 holdout 重叠的开发/校准输出语料，再重新生成 map 并独立验证。

## 2026-09-06 TensorRT reduced-vocabulary 候选准备

- 在本机固定 Qwen3-VL-2B revision tokenizer 上生成了 `65536` 和 `32768` 两份
  `vocab_map.safetensors`，metadata 的原始词表均固定为 `151936`，并通过 I32、排序、
  范围、EOS/PAD 覆盖校验。
- 选择报告显示，当前 `ps80_reviewed_v1` 标注语料只覆盖 `194` 个 token；对 65536 map
  的观测覆盖比例为 `0.296%`，其余主要由低 id filler 补齐。两份 map 因此仅是生成器
  smoke artifact，不进入 engine 构建或性能结论。
- 新增 `scripts/build_reduced_vocab_map.py`、`scripts/validate_reduced_vocab.py`，并将
  reduced-vocab map/metadata/选择语料文件的 SHA-256 纳入 engine provenance 与校验流程。下一步应在
  Edge-LLM v0.9.1 工具环境中使用更大、更贴近输出分布的语料生成 map，再进行独立 engine
  A/B；不能把结构 `valid` 当作质量通过。

## 2026-09-06 TensorRT LM-head INT4 候选入口

- 现有 `int4_awq` 量化产物的 `lm_head` 明确保持 FP16；Engine Inspector 中最终
  `[1,1,2048] x [1,2048,151936]` GEMM 是 level0/level1 之间发生 tactic 变化的关键输出层。
- 新增 `--lm-head-quantization int4_awq` 可选参数及独立 flow
  `configs/flows/quantize_qwen3_vl_2b_int4_awq_lm_head_candidate.json`。正式量化命令默认
  不传该参数，因此不改变已有 checkpoint 或 engine。
- 该候选需要单独完成 quantize -> export -> build -> Engine Inspector/Nsight -> 3×20
  质量与性能 -> 100 次 soak；不能与 reduced-vocab 同时首测。官方 v0.9.1 文档将
  `int4_awq` 列为 LM head 支持方法，但当前项目尚未产生该候选的板端 engine 证据。
- 候选 export/build flow 现在显式携带 `calibration_provenance.json` 和
  `--expected-lm-head-precision int4_awq`；构建 wrapper 会在调用 `llm_build` 前校验
  quantization 状态、Edge-LLM revision、校准 workload、样本数和 lm_head 精度，并将
  provenance 的 SHA-256 写入 build report。该门禁不替代量化误差和板端质量/性能验证。
- 量化入口在校准前新增 `quantize_and_export` signature preflight；固定 Edge-LLM 版本若不
  接受 `lm_head_quantization`，会提前失败，不进入大模型加载和校准阶段。

## 2026-09-06 TensorRT reduced-vocabulary provenance 闭环

- engine provenance 现在同时记录并校验 `selection_report.json`，并验证其中记录的 map
  SHA-256、token count 与实际 map/metadata 一致；缺失选择报告的 reduced-vocab 目录会在
  record 阶段拒绝，避免后续 benchmark 只剩一个无法追溯来源的二进制 map。

## 2026-09-06 TensorRT host fast-path 实验矩阵与 provenance 修正

- 在 runtime tuning 配置中新增 `all_host_fast_path` 组合候选，明确组合启用 profile
  pinning、binding-state cache、registered-binding cache、重复 profile switch 消除和
  fallback scan 消除；该候选只用于后续受控 A/B，不改变正式 level0/level1 engine。
- `scripts/serve_edgellm.py` 的 runtime provenance 现在同时记录 engine 的
  `requested_path`、实际解析后的 `resolved_path` 和 SHA-256。这样 INT4 LLM 复用 FP16
  visual symlink 时，部署语义与实际被加载文件均可审计，避免按目录名误判精度。
- 本地无硬件回归测试为 `174/174` 通过，`git diff --check` 通过；板端组合候选尚未
  产生性能结论，仍需在同一 engine、workload、15W 和独立内存采集下复验。

## 2026-09-06 TensorRT reduced-vocabulary report self-consistency

- 当前 `32768` 和 `65536` 字符覆盖 map 的实际 SHA-256 分别为
  `599da52117369d6544fa0655bdb41ab691e5deca636f251942a9a51932f041582` 和
  `beeeebb1b071fce03a9fca0fc1e3f61202652edd895790b1b9d76d848fc9fca3`，与对应
  coverage report 中的 map identity 一致。
- 两份 report 均通过构建前校验，`token_coverage_fraction=1.0`；该检查只证明报告与
  当前 map 自洽，不替代 Jetson engine 构建、严格 JSON、任务质量和性能门禁。

## 2026-09-06 TensorRT candidate gate configuration fix

- `scripts/evaluate_tensorrt_candidate.py` 现在实际读取并应用 tuning config 中的
  `require_backend_completion`、`require_json_validity` 和
  `require_quality_non_regression`，同时在 comparison 中保留 completion、JSON 和质量
  差异，避免配置声明与实际晋级逻辑不一致。
- 当前正式配置三项仍均为 `true`；该修正没有降低 level0/level1 或生产候选的门禁，只使
  诊断性实验能够明确区分“记录指标”和“阻断晋级”的策略。
- 本地无硬件测试为 `174/174` 通过，`git diff --check` 通过。
- candidate gate 的 comparison 现在同时输出 prefill、TTFT、decode 和 E2E 的 p50/p90/p99、
  speedup 与 improvement；只有 E2E p50/p90 继续作为当前正式晋级门槛，阶段数据用于定位
  attention、kernel、launch 或链路变化。
- soak gate 现在要求至少 `100` 条记录、无 failure，并核对 workload、runtime 和功耗 identity
  与 candidate 一致；不再接受只有空 `failure_summary` 而缺少来源绑定的 soak 文件。
- gate 新增可选的成对 `parksight_jetson_runtime_summary_v1` 资源证据输入；它必须与对应
  StudyReport 的 identity 和 record_count 一致，才可为缺失的 RAM、功耗和温度指标提供
  明确来源。资源摘要不会覆盖 StudyReport 中已有值，也不会接受单边或样本数不匹配的摘要。
- 使用现有 level0/level1 StudyReport、level1 `soak100` 和正式 tuning config 做端到端重算后，
  identity、重复次数、严格 JSON、质量和 soak 均通过；gate 唯一剩余阻断原因为两份历史
  StudyReport 的 `peak_memory_mb` 均未保存。该结果记录于
  `reports/jetson-tensorrt-revalidation/int4_opt1_gate_recheck.json`，未用其他来源补填内存。
- 新增 `scripts/compare_tensorrt_study_reports.py`，直接比较两份 StudyReport 的 workload/runtime
  identity、prefill/TTFT/decode/E2E、tokens/s、RAM、功耗、温度和质量指标；缺失字段标为
  `unverified`，且明确声明不能仅凭报告归因到具体算子或 kernel。
- 已用真实 level0/level1 StudyReport 生成
  `reports/jetson-tensorrt-revalidation/int4_opt0_vs_opt1_study_comparison.json`；identity
  可比，E2E p50/p90 分别为 `10522.99/2825.56 ms` 和 `11386.74/2952.99 ms`，对应
  `3.72x/3.86x` speedup。历史 StudyReport 未保存阶段和资源字段，相关项保持 `unverified`。
- 已核对并接受同一 `ps20_pilot_v1` 的两份真实 Jetson 资源摘要：level0/level1 的峰值
  RAM 为 `5153/5110 MB`，swap 均为 `1035 MB`，平均 GPU 利用率为 `98.45%/95.61%`，
  峰值温度为 `61.16/59.78 °C`，平均输入功耗为 `10.05/12.74 W`。这些值已绑定
  StudyReport identity 和 20 条记录，可用于资源回归比较；不替代后续 3×20 正式复验与
  100 条 soak 门禁。
- 本地无硬件测试已更新为 `174/174` 通过。

## 2026-09-06 TensorRT host-side fallback binding scan A/B

- 在同一 level1 INT4 AWQ engine、`i768/k1024`、batch=1、workspace `1024 MiB`、CUDA
  Graph 开启和 15W 模式下，使用固定 `ps20_pilot_v1` 前 20 个 test 样本，完成 control
  与 candidate 各 `3×20` 请求，另各执行 1 次 warm-up。两组均为 `60/60` 完成，输出
  chunk 均为 32，engine SHA-256 均为
  `207e109fcca28ac29ae0348d8dd517e61a6291704a8fea08c53644574f28d345`。
- candidate 仅在构造阶段确认全部 65 个 engine I/O 已由 `TensorRegistry` 覆盖后，跳过
  `prepare()` 的冗余 fallback binding scan；control 保留原扫描路径。该变量由
  `0023-safe-skip-fallback-binding-scan.patch` 和
  `EDGELLM_SKIP_FALLBACK_BINDING_SCAN=1` 实现，不改变 engine、binding、kernel、Graph
  或数值计算路径。
- control/candidate 的 TTFT p50 为 `690.53/577.75 ms`，candidate 降低约 `16.33%`；
  E2E p50 为 `1566.57/1440.54 ms`，candidate 降低约 `8.04%`。p90/p99 也保持相近
  的改善，说明该 host-side 路径在当前服务口径下存在可测收益。完整逐请求原始数据和
  provenance 见 `reports/jetson-tensorrt-revalidation/int4_opt1_host_fallback_scan_sweep.json`。
- 完整输出质量 smoke 使用同一 20 个样本和 `max_tokens=256`：control/candidate 均为
  `20/20` 请求完成、严格 JSON `9/20`，且失败样本 ID 完全一致。因此本轮观察到的是
  性能收益而非质量改善，也没有观察到新增回归；但由于严格 JSON 门槛未通过、未采集该
  candidate 的峰值内存且未完成 100 次 soak，candidate 暂不替换正式默认 runtime。

## 2026-09-06 TensorRT Engine Inspector 板端复核

- 通过 SSH 在 Jetson Orin Nano、TensorRT `10.3.0` 上直接反序列化 level0/level1
  INT4 engine；两者均为 `2144 layers / 65 I/O tensors`。
- `device_memory_size_v2` 从 level0 的 `113,995,264 B` 降至 level1 的
  `39,323,136 B`，减少约 `65.50%`；这证明序列化 TensorRT execution-context
  memory metadata 有变化，但不能单独证明 latency、kernel tactic 或 Graph replay 改善。
- 两份 engine 的 Inspector 均未暴露 tactic、workspace、per-layer timing 字段；已补充
  更高 profiling verbosity 的 engine Inspector 和 Nsight Systems 单请求 trace。
- 原始 runtime 日志已汇总为
  `reports/jetson-tensorrt-revalidation/int4_opt0_vs_opt1_runtime-summary_v2.json`；
  level0/level1 都有 decoder Graph capture，但没有显式 replay marker。
- 同口径 detailed Inspector 对比显示只有 5 个 layer metadata 发生变化，其中两个
  `node_linear` 是 `[1,1,2048] x [1,2048,151936]` 的最终 lm_head GEMM；level1
  选择了不同的 Ampere GEMM tactic。该结果给出了 kernel 优化候选，但没有 per-layer
  执行时间，不能单独证明它解释了全部 decode 加速。摘要见
  `reports/jetson-tensorrt-revalidation/int4_opt0_vs_opt1_detailed_tactic-summary.json`。
- 在同一张冻结图片和同一请求参数下，各 engine 采集 1 次 Nsight Systems trace：两边均有
  58 次 `cudaGraphLaunch`；level 1 的 `cudaStreamSynchronize` 累计时间为
  `2,072.59 ms`，level 0 为 `7,923.40 ms`。最终 lm_head GEMM 从 `111.45 ms` 降至
  `10.98 ms`，而 INT4 W4A16 GEMM 基本不变（`262.31 ms` vs `262.39 ms`）。该证据
  支持“level 1 重新选择了关键 GEMM tactic”的判断，但单次 trace 仍不是完整 per-node
  归因或正式吞吐统计。结构化摘要见
  `reports/jetson-tensorrt-revalidation/int4_opt0_vs_opt1_nsight_summary.json`。
- 在相同 `i768/k1024/batch1/workspace1024/15W` 口径下尝试 level2，构建进程被系统
  `Killed`，未生成可用 engine；失败证据见
  `reports/jetson-tensorrt-revalidation/int4_opt2_build_failure.json`，不将其解释为
  推理阶段 OOM 或性能结论。
- 完成 level1 workspace 单变量复测：固定 `i768/k1024`、batch=1、CUDA Graph、INT4
  AWQ，在同一 Jetson 上测试 `512/1024/1536 MiB`。三档均为 60/60 成功；TTFT p50
  分别为 `580.10/579.93/578.76 ms`，E2E p50 分别为
  `1443.03/1441.25/1447.38 ms`。相对 1024 MiB，512 MiB 的 E2E p50 约慢 `0.12%`，
  1536 MiB 约慢 `0.43%`，没有稳定收益，因此暂不调整 workspace 默认值。完整 engine
  provenance 和统计见
  `reports/jetson-tensorrt-revalidation/int4_opt1_workspace_sweep.json`。
- 完成 level1 profile `i768/k768` 复测；prompt contract 显示 735 输入 token 加 32
  输出需要 767 KV 容量，768 仅剩 1 token headroom。与固定 workspace=1024 的
  `i768/k1024` 相比，60 次请求均成功，TTFT p50 为 `580.78 ms` vs `579.93 ms`，
  E2E p50 为 `1442.67 ms` vs `1441.25 ms`，单请求延迟没有收益。该候选的峰值 RAM、
  swap 和并发容量尚未采集，暂不晋级；完整记录见
  `reports/jetson-tensorrt-revalidation/int4_opt1_profile_sweep.json`。
- 完成同一 level1 engine 的 CUDA Graph 开关 A/B。Graph 开启和关闭各完成 60/60；
  TTFT p50 为 `579.93 ms` vs `578.49 ms`，基本不变，但 E2E p50 为
  `1441.25 ms` vs `1510.45 ms`，关闭后慢约 `4.80%`，p90/p99 也慢约 `4.81%/4.84%`。
  结果支持保留 Graph 开启，并将主要收益归因方向放在 decode 侧的 launch/replay 开销；
  不将其解释为某个具体 kernel 的收益。关闭组使用隔离 clean worktree 和
  `EDGELLM_DISABLE_CUDA_GRAPH=1`，完整 provenance 见
  `reports/jetson-tensorrt-revalidation/int4_opt1_cuda_graph_sweep.json`。
  关闭组原始启动日志为
  `reports/jetson-tensorrt-revalidation/int4_opt1_cuda_graph_disabled_startup.log`。
- 针对 Nsight 中占主要 GPU kernel 时间的 INT4 W4A16 T2，完成 `STAGES=2/3/4` 单变量
  复验。`STAGES=2` 在 60/60 成功下，TTFT/E2E p50 为 `611.12/1472.15 ms`，相对
  `STAGES=4` 的 `579.93/1441.25 ms` 分别慢约 `5.38%/2.14%`。`STAGES=3` 的 60/60
  HTTP 请求不能作为性能结果：平均输出仅约 `30.1` chunks，随后 20 条诊断请求严格
  JSON 为 `0/20`，并出现 schema-invalid、Markdown/YAML-like、截断和非 schema 文本。
  因此当前保留正式 `STAGES=4`，不将 STAGES=3 的较低 E2E 数字解释为加速。补丁、临时
  plugin SHA 和完整统计见
  `reports/jetson-tensorrt-revalidation/int4_opt1_w4a16_stages_sweep.json`；正式源码与
  plugin 已恢复到测试前 SHA。

## 2026-09-05 TensorRT benchmark 与候选分析链路

- 新增 `scripts/run_edgellm_benchmark.py`，可按冻结 manifest 固定顺序运行 warm-up、
  3×20 benchmark，并分别记录客户端 E2E、HTTP RTT、SSE TTFT、服务端 prefill/decode、
  output tokens 和失败类别；warm-up 单独输出，不进入正式统计。
- 该低层 runner 新增 `--concurrency` 和 `--run-metadata-json`：并发度大于 1 时每个
  工作线程使用独立 HTTP backend，并按 manifest/repetition 顺序写回结果，用于验证
  Edge-LLM dynamic batch 吞吐；并发结果仍需将单请求 TTFT/E2E 与整体 wall-clock 吞吐
  分开统计。
- benchmark summary/compare 现在会保留并比较运行级
  `aggregate_output_tokens_per_second`，并把并发度标为独立的 `run.changed` 变量，避免
  将并发请求下的请求级 tok/s 平均值误当作系统吞吐。
- compare 对 latency 使用 `baseline/candidate`，对 decode/E2E/aggregate token rate 使用
  `candidate/baseline`，并分别定义 improvement 方向；吞吐提升不再被报告为反向下降。
- `scripts/benchmark_edgellm.py` 现在支持挂接同次 `tegrastats`，汇总结果包含
  失败类别、GPU 利用率、RAM、swap、功耗和温度；传入 `--warmup-input-jsonl` 后才填充
  `cold_start_ms`，避免把 warm-up 后的稳态样本误标为冷启动。
- benchmark 汇总现在要求完整固定 metadata 才将 A/B 标记为 `matched`；字段缺失统一为
  `unverified`，防止不同 profile、功耗模式或 precision 的结果被误合并。
- 新增 `scripts/compare_tensorrt_engine_inspectors.py`，对齐算子类型、tactic、workspace
  与 device memory；现采用唯一 layer name 优先、index 兜底，并显式
  记录 fusion/split 导致的新增和移除 layer，并汇总类型、tactic、workspace、timing
  变化及算子数量 delta。字段不可见时标记为 `missing`，不将元数据差异直接解释为
  runtime 加速。
- 新增 level2/level3 `i768/k1024` profiling flow，构建报告显式记录 workspace、builder
  level、profiling verbosity 和 weight-streaming 配置。当前仍需在 Jetson 生成 engine、
  Inspector JSON 与 Nsight trace 后，才能继续归因 attention/kernel 瓶颈。
- 候选 evaluator 增加可选 runtime-log summary 输入；提供该证据时，decoder CUDA Graph
  capture 或 replay 未确认都会直接阻止候选晋级，避免仅凭 capture 或业务请求成功误判
  Graph 优化有效。
- 候选 evaluator 现在要求 baseline/candidate 的 runtime identity 均存在且一致，并将
  100-sample soak report 作为晋级必需证据；缺少 soak 或 baseline 重复次数不足时不会放行。
- 已新增 level1、`i768/k1024` 的 workspace `512 MiB` 与 `1536 MiB` 独立构建 flow，
  可在完成 level2/3 profiling 后继续按单变量顺序测试 workspace。
- 新增 `0012-configurable-timing-cache.patch` 和 `--timing-cache` 构建入口，按目标
  Jetson 的 TensorRT `ITimingCache` 复用 tactic timing，并在 build report 中记录 cache
  的路径、大小和 SHA-256；现在还要求 `--timing-cache-binding` sidecar 显式绑定
  GPU、CUDA/TensorRT 版本和 BuilderConfig，构建前会拒绝 level/workspace/profile 不匹配。
  该机制只优化重复构建时间，不作为推理 latency 提升证据。
- 新增 `0013-profile-switch-timing.patch` 与 `0014-pinned-profile-contexts.patch`：前者
  记录 `setOptimizationProfileAsync` 的主机调用时间，后者以
  `EDGELLM_PIN_OPTIMIZATION_PROFILES=1` 为开关，为每个 profile 懒创建独立 execution
  context。该候选已在 v0.9.1 源码副本完成补丁序列校验，但仍需板端编译、内存和延迟 A/B，
  不改变默认 level=0 或默认单 context 路径。
- 新增 `0015-cache-binding-state.patch` 与 `--cache-binding-state` 实验入口：在
  `prepare()` 完成所有 I/O 绑定后缓存 graph key/snapshot，减少 `execute()` 的重复 host-side
  binding introspection。该优化默认关闭，已通过 0013→0014→0015 源码级 patch sequence
  校验；尚未在 Jetson 编译或计入正式性能结论。
  - 新增 `0016-registry-name-index.patch`：为展开后的 TensorRegistry binding name 建立
    host-side unordered index，将 `contains()` 的线性查找降为常数期望复杂度；已在全新
    v0.9.1 源码副本完成 apply/check 校验，尚未在 Jetson 编译或计入正式性能结论。
  - 新增 `0017-cache-registered-bindings.patch` 与服务入口
    `--cache-registered-bindings`：按 execution context 缓存已注册 binding 的 address/shape，
    非 pinned 单 context 在 profile 变化时清空 cache，pinned context 保留各自 cache，
    仅在状态变化时调用 TensorRT setter；已完成
    0013→0017 源码级 apply/check 校验，尚未
    在 Jetson 编译、做正确性 A/B 或计入正式性能结论。
  - 新增 `0018-skip-redundant-profile-switch.patch` 与服务入口
    `--skip-redundant-profile-switch`：首次使用或 profile 变化时保留
    `setOptimizationProfileAsync()`，同 profile 重复准备时跳过调用；已完成
    0013→0018 源码级 apply/check 校验，尚未在 Jetson 编译或计入正式性能结论。
  - runtime 日志汇总器新增 `profile_switch_api_calls` 与
    `profile_switch_api_call_summary`，将 0013 明确记录的
    `setOptimizationProfileAsync` Host API 耗时与相邻日志间隔分离；现有历史日志未启用
    该打点，因此对应字段保持缺失，不能回填或估算。
  - 新增 `0020-skip-redundant-registry-scan.patch` 与服务入口
    `--skip-fallback-binding-scan`：构造阶段确认全部 engine I/O 均由 registry 覆盖后，
    可跳过 `prepare()` 的 fallback scan；未全覆盖时自动保留原路径。该候选的安全板端
    实现和 A/B 证据由 `0023-safe-skip-fallback-binding-scan.patch` 固化；源码级
    0013→0018→0020 序列仍保留用于历史复现。
- `validate_tensorrt_provenance.py` 新增可选 `--engine` 文件校验，会重新计算 engine
  SHA-256/大小并与 provenance 对照，防止仅凭配置字段误用其他 engine；未传入时保留原有
  配置级校验边界。
- `benchmark_edgellm.py` 新增 `--provenance-json`，将 engine SHA-256、大小和构建配置嵌入
  benchmark metadata，避免指标文件与实际 engine 身份脱钩。
- benchmark 新增 `--runtime-metadata-json`，compare 工具在独立 `runtime` 字段报告两次
  运行的 engine hash 与 runtime 开关变化，便于解释同 engine 的 host-side A/B。
- 低层 benchmark 汇总新增 `execution.repetitions`，compare 工具按 repetition 输出阶段
  分位数，并新增 `execution.repetition_gate` 按默认 3 次、p50 10%/p90 5% 门槛逐次判断；
  缺少逐次统计时标为 `unverified`。
- candidate evaluator 新增可选 `--benchmark-comparison`；传入后要求低层
  `repetition_gate.eligible=true`，将逐次延迟稳定性纳入候选晋级门禁。
- profile contract 新增可选 KV 维度参数，可估算连续 FP16 K/V cache 的理论字节数，当前
  Qwen GQA 形状下为 `k1024≈4 MiB`、`k768≈3 MiB`；仍需板端实测确认实际内存收益。
- `serve_edgellm.py` 新增 `--runtime-metadata-output`，在服务启动前记录两份 engine 的
  SHA-256、文件大小和本次 runtime 开关，便于 host-side A/B 与 benchmark 精确对齐。
- 新增独立 runtime 矩阵 `configs/tensorrt/jetson_orin_nano_int4_runtime_v1.json`，将
  control、单变量 host-side 候选和 pinned+binding cache 组合分开，避免将 runtime 开关
  与 builder/profile/KV cache 变化混为一个变量。

## 2026-09-05 TensorRT attention/kernel 证据链

- 基于 2026-09-04 的 level 0/1 VLM profile 日志新增
  `scripts/summarize_edgellm_runtime_log.py`，自动区分 FMHA cubin 加载尝试、driver
  拒绝项、未被拒绝项、engine 的 attention 配置、aux/worker stream 和 CUDA Graph capture。
- 对现有日志复核结果为：两级均为 `headDim=128`、`numKVHeads=8`，FMHA 加载候选集合一致，
  均有 2 个 `head_size=256 + custom_mask` 候选被 driver 拒绝，decoder CUDA Graph 均捕获成功。
  因此当前收益应优先归因于 builder tactic/engine 图差异，不能归因于视觉 encoder、FMHA
  候选集合变化或 Graph 开关变化；实际 tactic 仍需 Engine Inspector/Nsight 确认。
- 新增的 profile transition 解析显示，level0/level1 日志各有 43 次 profile 切换，
  `same_profile_transition_count=0`，实际序列为 prefill/decode 的必要交替；因此没有引入
  “跳过重复 profile 切换”的无收益 patch。
- 新增 `scripts/inspect_tensorrt_engine.py` 和
  `0010-configurable-profiling-verbosity.patch`，支持构建 profiling engine 后导出
  TensorRT Engine Inspector 的 layer/tactic/workspace 元数据。该 profiling engine 与正式
  性能 engine 分离，不改变当前默认 level=0。
- 新增 `scripts/check_tensorrt_profile_contract.py`，把 prompt token 预算与 KV capacity
  关系变成构建前门禁。现有 v1 workload 实测输入为 735 token；在 `maxGenerateLength=32`
  下，`i768/k768` 需要 767 capacity 并通过，`i512/k768` 因输入越界被拒绝。下一轮优先
  在板端复验 `i768/k768` 的内存与 decode 变化。

## 2026-08-31 TensorRT builder optimization level A/B

- 在同一 Jetson、15W 模式、同一 `ps20_pilot_v1` 20 条测试、同一 visual engine 和同一
  INT4 AWQ 权重下，仅将 LLM builder optimization level 从 `0` 改为 `1`，分别完成了
  20/20 后端推理和严格 JSON 100%。两次风险准确率均为 `35%`、事件 micro-F1 均为 `0`，
  质量结果一致。
- level=0 对照的 E2E p50/p90 为 `10526.63/11386.87 ms`，聚合输出速率为
  `7.32 token/s`；level=1 候选为 `2830.51/2946.80 ms` 和 `27.27 token/s`，p50 与
  聚合吞吐均约提升 `3.72x`。
- level=1 的 Jetson 系统观测为 RAM `5105.3/5110 MB`、swap `1035/1035 MB`、GPU
  利用率 `95.6%/98.5%`、输入功耗 `12.74/13.40 W`、温度 `56.67/59.78 °C`；没有看到
  由该 builder 选项引入的内存异常。该 engine 作为候选保留，尚未替换正式 flow 的默认
  level=0；需要重复运行和更长 workload 验证稳定性后再切换。
- 证据位于本地忽略目录 `reports/jetson-metrics-20260830/`，包括两个 StudyReport、
  两份 summary、tegrastats 和 level=1 builder 日志。候选 LLM engine 的 SHA-256 为
  `207e109fcca28ac29ae0348d8dd517e61a6291704a8fea08c53644574f28d345`。

## 2026-08-30 人工确认数据后的 v1-v5 INT4 对照

- 80 条人工确认数据已形成正式 provenance：48 条 LoRA train、16 条 validation、16 条
  独立 INT4 calibration；LoRA 与 calibration 无来源组交集，均使用
  `parking_risk_v2_strict_json@sha256:6ca953643f38a13b579a77090c77d3fca30d3ba9a1b181d88ae11692ea150fec`。
- 服务器 v3 calibration-aligned LoRA 训练和合并成功。16 条 validation 结果为严格 JSON
  `100%`、风险准确率 `56.25%`、事件 micro-F1 `0.4286`、不安全建议率 `0%`；与 v2
  服务器模型结果一致，说明本轮变化主要来自量化校准口径而不是训练文本。
- Jetson 同口径结果如下：v1 INT4 为 `100% / 62.50% / 0.3000 / 18.75%`（JSON / 风险
  准确率 / 事件 micro-F1 / 不安全建议率，p50 `7.84 s`）；v2 INT4 为
  `87.50% / 56.25% / 0.1429 / 18.75%`（p50 `7.32 s`）；v3 calibration-aligned INT4
  为 `100% / 68.75% / 0.1667 / 0%`（p50 `8.60 s`）。
- v3 的风险准确率和安全建议率改善伴随事件召回率降至 `10%`，逐样本检查还发现低风险
  场景普遍输出包含 `prepare_to_stop` 的完整建议集合，属于低风险/空事件偏置；因此
  v3 作为已验证实验候选保留，不替换当前 v1 事件识别参考版本。
- v3 的量化、导出、LLM/visual engine 构建、HTTP health check 和 16 条 study 均已完成，
  验证后服务已停止。完整 StudyReport 保存在本地忽略目录
  `reports/jetson_edgellm_int4_awq_ps64_reviewed_v3_calibration_aligned_validation_strict_json_i768_k1024.json`。
- v4 增加事件样本重采样（有效训练记录 80 条），但服务器 validation 的风险准确率由
  v3 的 `56.25%` 降至 `37.50%`，事件 micro-F1 由 `0.4286` 降至 `0.3529`，因此没有
  继续量化部署。训练入口新增的 `event_oversampling_factor` 默认值为 1，旧配置行为不变，
  并由无硬件测试覆盖。
- v5 保持 v1 合并模型不变，仅使用语义 v2 calibration 重新量化；Jetson 结果为严格 JSON
  `100%`、风险准确率 `62.50%`、事件 micro-F1 `0.2857`、不安全建议率 `18.75%`、p50
  `7.64 s`。相对 v1 的事件 micro-F1 `0.3000` 没有改善，说明单独替换 calibration
  不能解决板端事件漏检；v5 报告为
  `reports/jetson_edgellm_int4_awq_ps64_reviewed_v1_v2_calibration_validation_strict_json_i768_k1024.json`。
- `StudyReport` 新增 `event_macro_f1` 和 `event_metrics`，分别记录六类事件的平均 F1
  以及 support/TP/FP/FN/precision/recall/F1；既有 `event_micro_f1` 计算口径保持不变。
- 数据生成入口新增 `event_coverage` 审计。当前 64 条 LoRA 数据的训练覆盖为：
  `vehicle=12`、`narrow=11`、`visibility=3`、`vru=1`、`fixed=0`、`parking=0`；
  validation 含 1 条训练未见的 `fixed_obstacle_near_path`。该覆盖缺口是下一轮人工
  数据扩充的前置条件，不应通过继续调整量化参数规避。

## 2026-08-30 人工复核反馈：驾驶建议需要结合场景重新判断

- 复核样本 ps2-p2_img43_3396 时发现，Codex candidate 与模型均给出
  maintain_observation，但人工观察认为近场车辆/障碍可能已经影响当前机动路径，
  倾向于 prepare_to_stop。该样本尚未写入正式人工标注，当前结论仍由复核者最终确认。
- 该反馈说明人工复核不能只比较 candidate 与 model 是否一致；需要联合检查
  risk_level、events、evidence 和 driver_advice。如果确认存在近场路径冲突，
  应同步修正相互匹配的字段，并将状态设为 corrected、填写 review_note。
- 审核页面已为五种 driver_advice 增加中文语义说明和跨字段复核提示，重新生成的
  页面仍为 reports/label-review-20260830/ps80_candidate_review.html。该改动只改善
  复核可解释性，不自动改变 80 条候选标签或评测规则。

## 2026-08-30 候选 LoRA 80 条服务器开发集评测

- 在本地 RTX 4060 上复用候选 adapter、Qwen3-VL revision 和
  `parking_risk_v2_strict_json` workload，完成 `ps80_development_v1` 的 64 条 train 与
  16 条 validation 评测。两处分片均为严格 JSON `100%`，风险准确率分别为 `57.81%` 和
  `62.50%`，事件 micro-F1 均为 `0`，不安全建议率分别为 `32.81%` 和 `18.75%`；合计
  为 `80/80` 严格 JSON、风险准确率 `58.75%`、事件 micro-F1 `0`、不安全建议率
  `30.00%`。
- 该结果确认服务器训练后评测闭环已跑通，但候选 adapter 的风险事件识别仍不达标，且
  参考标签仍为 Codex 候选结果。正式模型仍需等待人工终审后的 `human_confirmed_v1`。
- 对应配置为
  `server_transformers_lora_ps64_codex_candidate_v1_ps80_train_strict_json.json` 和
  validation 版本；报告 SHA-256 分别为
  `d3bea513b671dfd5d84f034be1d5d1ec9b0f4bd259bcd7279b843cb067c853bf` 和
  `d1c81a98dfba0ed0f9b9ef3234988627aab6a8cd76ebc7052fc16c9b51afae87`。
- 新增 `scripts/apply_review_decisions.py`，支持按批次应用人工 `confirmed`/`corrected`
  决策，保留未复核记录，并在 `--require-complete` 下强制 80 条全部定稿；77 个无硬件
  测试通过。工具还支持生成全部 case 的待编辑决策模板；该模板不改变候选 annotation，
  也不绕过最终 provenance gate。
- `scripts/finetune_qwen3_vl_lora.py` 新增 `--validate-only`，可在不加载 CUDA 或模型前
  校验数据来源、schema、split、图片、workload 和模型路径。候选配置实测通过；当前
  `ps64_reviewed_v1` 正式配置因数据仍为 Codex 候选来源而在模型加载前拒绝。无硬件测试
  已累计 `77/77` 通过。
- 新增 `scripts/build_review_html.py`，将 80 条错误复核记录和本地图片打包为单文件离线
  复核页面；页面支持逐条填写 `confirmed`/`corrected` 决策，并下载可直接交给
  `apply_review_decisions.py` 的 JSONL。当前生成页面已嵌入 80 张图片，仍需人工实际
  复核后才能形成 `human_confirmed_v1`；页面另将未完成草稿保存到当前浏览器本地，不写入
  仓库；新增功能后无硬件测试累计 `78/78` 通过。
- 页面支持可选的 `--reference-annotations` 只读输入；当前实际使用
  `data/annotations/ps80_teacher_v1.jsonl`，80/80 case 对齐并显示 teacher/reference、
  Codex candidate 和模型输出三方上下文。teacher 仅用于复核上下文，不是人工金标。
- 正式 `ps64_reviewed_v1` 的后处理配置已补齐：合并、FP16 ONNX 导出、人工确认校准的
  INT4 AWQ、INT4 ONNX 导出、FP16/INT4 LLM engine 构建，以及对应 Jetson `ps20_pilot_v1`
  study。所有 flow 仍以 `human_confirmed_v1` 数据和 merged checkpoint 的实际存在为
  前置条件。
- README 已补充人工终审后的正式数据生成命令，固定输出
  `data/processed/calibration/ps16_human_confirmed_v1.jsonl`，与 INT4 flow 的输入一致；
  当前该文件尚不存在，仍需真实复核结果生成。

## 2026-08-30 严格 JSON workload A/B checkpoint

- 新增 `configs/workloads/parking_risk_v2_strict_json.json` 和对应的
  `jetson_edgellm_int4_awq_ps16_v1_ps20_pilot_strict_json` study。v2 保留
  `parking_risk_v1` schema、输入尺寸、生成参数和枚举集合，仅强化原始 JSON 输出边界，
  identity 为 `parking_risk_v2_strict_json@sha256:c4695a1bfa4d547f5ad90ec7697b82419dad12995829776e96c850707e15d1f4`。
- 在 Jetson 同一领域 INT4 engine、同一 `ps20_pilot_v1` 和相同运行时参数下完成 v1/v2
  完整 A/B。v1 为 `20/20` 后端完成、严格 JSON `20%`、风险准确率 `15%`、事件
  micro-F1 `0`、端到端 p50 `10618 ms`；v2 为 `20/20`、`95%`、`35%`、`0`、
  p50 `7428 ms`。v1 的 16 条失败均为 `json_parse_error`，v2 仅 1 条失败。
- v2 之前的单图 smoke 也返回 HTTP 200、严格 JSON 解析成功；完整 study 进一步证明格式
  修正不是单样本偶然现象，但风险事件识别质量仍未改善。
- 第一次 v2 请求曾因提示词达到 823 token、超过 `i768` engine 的 768 token 输入上限而
  返回 HTTP 500；压缩重复约束后重新验证通过。该事实纳入 workload 设计约束，后续完整
  study 需要同时报告输入 token 预算和格式有效率。
- prompt-contract 诊断已增加 token 计数和 `--max-input-tokens` 门禁；Jetson 实际 Qwen3-VL
  processor 测得固定 `001.jpg` 的 v1/v2 输入为 `735/700` tokens，均返回
  `within_budget=true`，并确认 `messages_equal=true`。
- 本次服务已停止；日志归档于
  `reports/jetson-int4-strict-json-smoke-20260830.log`，SHA-256 为
  `6850bee6589a026cd4e2a24cb6d7e8e5e090341bcf0ee94ac4619cfd577af5a1`。
- 完整 v1 StudyReport 归档于
  `reports/jetson_edgellm_int4_awq_ps16_v1_ps20_pilot_i768_k1024.json`，SHA-256 为
  `4e7a3faa97ddcf27a68b20ef9df54b544419793ede06e0b77d83b9317b682bd6`；完整 v2 StudyReport
  归档于 `reports/jetson_edgellm_int4_awq_ps16_v1_ps20_pilot_strict_json_i768_k1024.json`，
  SHA-256 为 `9f75374756305d820baf8efd635a5ef709dc867a453e2632f426c78d897c1cc0`。两次服务
  均已停止。
- 复用同一 engine 对 80 条 Codex 候选开发样本完成一次 train/validation 分片评测：
  `64+16` 条后端全部完成，严格 JSON `77/80=96.25%`，候选标签上的风险准确率
  `47/80=58.75%`，事件 micro-F1 `0`，不安全建议率 `23/80=28.75%`，共有 3 条
  `json_parse_error`。该结果只用于开发流程和错误分析，不是人工确认后的质量结论。
- 对应配置为
  `jetson_edgellm_int4_awq_ps16_ps80_codex_candidate_train_strict_json` 与 validation
  版本；报告 SHA-256 分别为
  `21a3ef7dd7f0a4d76f0846a03448cca6c296b2458c0a1ec065b70a5958d10671` 和
  `8edf7cd695cfffd919d521653bd7877421d470396705a12e089e76fb11ebb4dd`。评测服务已停止。

- 已生成 80 条候选开发评测的人工复核清单：覆盖 `80/80` 个 case，77 条 JSON 有效，47 条
  风险等级匹配，30 个 case 存在事件差异，33 个 case 为高优先级；事件差异全部是候选
  事件漏检。生成入口为 `scripts/build_candidate_error_review.py`，输出为
  `reports/label-review-20260830/ps80_candidate_error_review_v1.json`。
- 已在本地 RTX 4060 隔离训练环境完成候选 LoRA：3 epochs、48 个唯一训练样本、63 条
  过采样记录、48 个 optimizer steps，validation loss `0.72318`，峰值显存 `5.272 GiB`，
  训练耗时约 `146 s`。候选 adapter 在冻结 `ps20_pilot_v1` 上为风险准确率 `50%`、
  事件 micro-F1 `0.1818`；同环境 base 为 `35%`、`0.350`，事件识别反而下降。
- 候选 adapter 已成功 merge，merged checkpoint 与 adapter 的 20 条输出逐样本一致，
  对应配置为 `merge_qwen3_vl_2b_lora_ps64_codex_candidate_v1` 和
  `server_transformers_merged_lora_ps64_codex_candidate_v1_ps20_pilot`。该实验仍使用
  Codex 候选标签，不构成正式质量结论。

## 2026-08-29 现场复核 checkpoint

- 本地仓库已提交 Codex-assisted review workflow、训练 provenance 防误用和部署预检改动；
  当前 `main` 比 `origin/main` 超前 13 个本地提交，未执行 push。
- 无硬件测试为 `61/61` 通过；80 条 Codex 候选 package 可由
  `build_review_package.py` 重新生成，当前候选分布为 `low=47`、`medium=32`、
  `high=1`。
- 已使用真实候选 annotation 完成一次数据生成 dry-run/校验：LoRA 共 64 条（48 train、
  16 validation），INT4 calibration 共 16 条；LoRA 与 calibration 来源组交集为 0，
  开发数据与冻结测试集来源组交集为 0。输出 summary 同时固定了 workload identity、
  manifest、teacher annotation、candidate annotation 和 calibration config 的 SHA-256。
- 定稿脚本现在会校验候选和人工两份严格 `ParkingAssessment`，强制
  `confirmed`/`corrected` 状态与内容一致，并输出候选相对人工结果的风险准确率、事件
  micro-F1 和字段修正统计。当前没有人工确认 package，因此不能把候选结果写成最终
  质量结论。
- 正式 LoRA 训练入口现在要求训练配置声明 `label_source`，并核对所有数据记录的来源
  一致性；候选 Codex 来源默认被拒绝。当前 `ps64_reviewed_v1` 配置因此会在启动早期
  停止，等待人工终审后的 `human_confirmed_v1` 数据。
- 为支持“先跑通 Codex 开发流程、后做人工复盘”，新增独立的
  `train_qwen3_vl_2b_lora_ps64_codex_candidate_v1.json`；它显式允许候选标签，但输出
  数据 `data/processed/lora/ps64_codex_candidate_v1.jsonl`、输出目录和 flow_id 均带
  `codex_candidate`，summary 的 `dataset_id` 为 `ps80_codex_candidate_v1`，与正式训练
  配置隔离。
- 新增 `inspect_review_package.py` 只读检查入口，可输出复核状态计数、待处理 case_id、
  候选风险/事件分布和 `ready_for_finalize`；使用 `--fail-on-incomplete` 可将未完成复核
  作为流程失败处理。
- Edge-LLM 服务入口新增 `--check-only` 静态预检；它已在 Jetson 现有 FP16、LoRA FP16、
  普通 INT4、领域 INT4 四种 engine 组合上完成检查，可区分文件/路径就绪与实际 GPU
  加载成功。
- 服务入口同时支持 `--llm-engine-root` 与 `--visual-engine-root` 分目录模式，覆盖
  LoRA/INT4 LLM engine 复用 FP16 visual engine 的实际部署结构。
- 已通过 SSH 只读连接 Jetson。板端工作树为旧提交 `f362a43` 且存在 33 项未提交/未跟踪
  改动，本轮未覆盖或清理。临时补充 venv 内 CUDA 库路径后，PyTorch `2.9.1`、CUDA
  `12.6` 和 Transformers `4.57.6` 可导入。
- 板端 HTTP 服务当前未运行。使用明确的 Edge-LLM 源码、pybind 和插件路径后，FP16 LLM
  engine、tokenizer 和 base context 可以加载，但图形桌面状态下 visual engine 申请约
  `811 MiB` 连续内存失败。该结果属于资源条件失败，不改变既有 headless 条件下的
  `20/20` FP16 成功证据。
- 2026-08-30 对领域 INT4 做了新的真实 smoke：去掉未启用的 weight-streaming 参数，
  显式加入 Jetson venv `site-packages` 后，LLM/visual engine、tokenizer、CUDA graph 和
  Uvicorn 均成功启动，`/health` 返回 HTTP 200；1 条单图请求也返回 HTTP 200 和 91 tokens，
  但严格 JSON 因 ```json 代码围栏解析失败。临时服务已停止，原始日志归档于
  `reports/jetson-int4-smoke-20260830.log`，SHA-256 为
  `9ee94b7b331368c7e7204288938eadd1bdd3f81e05e0c97209210bd7d77534b5`。

## 已实现并由无硬件测试覆盖

- `ParkingAssessment` 严格 JSON 解析、字段和枚举校验。
- `ParkingCaseCatalog` 的 manifest/annotation 一致性、来源组划分和图片引用校验。
- 冻结 workload 读取、字段约束和 SHA-256 身份。
- Transformers 与 TensorRT Edge-LLM HTTP 的 `RiskRuntime` Adapter seam。
- Transformers 可选 forward-hook profiling，可独立记录视觉编码、prefill、decode 和
  首个 logits 时刻；Edge-LLM HTTP adapter 记录请求构造和 HTTP 往返时间；
  插桩 study 与未插桩性能基线使用不同 `study_id`。
- 成功、JSON 失败、超时、输入缺失等 `InferenceRecord` 事实记录。
- 质量指标、阶段时延分位数、资源汇总和 `StudyReport`。
- 从 `StudyReport` 与 `tegrastats` 原始日志派生后端完成率、端到端分位数和板端资源
  分位数的证据汇总入口。
- 单图分析、配置化研究和环境快照入口。
- LoRA 训练、合并、导出、engine 构建的可审计外部流程入口。

## 已形成的外部运行证据

- Jetson Orin Nano 的 L4T、CUDA、TensorRT、功耗模式、内存、swap 和磁盘已检查。
- `Qwen/Qwen3-VL-2B-Instruct` 已按不可变 commit
  `89644892e4d85e24eaac8bacfd4f463576704203` 缓存到 Jetson。
- 模型 snapshot 的文件数、总字节数、未完成文件和权重 SHA-256 已校验。
- 各实验与训练示例配置已统一固定到该模型 commit。
- Jetson Transformers FP16 已在冻结 `ps20_pilot_v1` 上完成 20/20 推理，严格 JSON
  有效率为 100%，失败汇总为空；未插桩端到端 p50/p90/p99 为
  `9.38/14.12/27.94 s`，进程 CUDA 峰值为 `4176.72 MB`。
- 独立插桩 study 同样 20/20 成功；阶段 p50 为预处理 `27.60 ms`、视觉编码
  `225.11 ms`、prefill `626.51 ms`、decode `17.04 s`。插桩强制 CUDA 同步，绝对
  时延不与未插桩基线混用；decode 占插桩生成 p50 的约 `88.5%`。
- x86 GPU 服务器已使用 TensorRT Edge-LLM `v0.9.1` 固定 commit 完成 LLM 与
  视觉编码器的 ONNX 导出；flow record 状态为 `succeeded`，全部声明输出存在。
- Jetson 已升级到 L4T R36.5.0 / JetPack 6.2.2，CUDA 12.6 与 TensorRT 10.3
  可用；ONNX 归档和内部 11 个文件均已重新校验。
- TensorRT Edge-LLM `v0.9.1` commit
  `7f061f21f0a581ba234a1e233c9315b89d8e47d6` 已在 Jetson 编译，插件、LLM/视觉
  builder、示例 runtime 与 Python binding 均存在，动态库依赖和 Python 导入通过。
- 视觉 FP16 engine 已在 Jetson 正式构建成功。`visual.engine` 为 786 MiB，SHA-256
  为 `3c6b4cce682e021b09c066d0e325335e31ef9edbf613c754be586035c26f5c2f`；flow
  record 状态为 `succeeded`，三个声明输出全部存在。
- 启用临时 8 GiB 磁盘 swap 后，LLM FP16 engine 已在 Jetson 构建成功。
  `llm.engine` 为 `3453798316` 字节，SHA-256 为
  `cbdf0300bf406dfbbcd06d47435c699c26403139d6bdd06b473ba00576583013`；flow
  record 状态为 `succeeded`，六个声明输出全部存在。
- 运行时补丁在创建 execution context 前调用 TensorRT
  `setWeightStreamingBudgetV2(0)`。LLM 与视觉 engine 已同时加载，HTTP `/health`
  返回 `healthy`，decode CUDA graph 捕获成功。
- 多模态 system content 已统一为 Qwen3-VL chat template 接受的 content 数组，并由
  固定 fixture 和 40 个无硬件测试覆盖。
- Edge-LLM C++ runtime 已应用“先分配 base/decoder context，再加载视觉 runner”的
  补丁；最终 FP16 LLM engine profile 为 input 768、KV 1024，单图连续 3/3 严格 JSON
  验收通过。
- 冻结 `ps20_pilot_v1` 的 20 个样本均完成后端推理；严格 JSON 有效率为 `20/20`，
  失败汇总为空。端到端 p50/p90/p99 分别为 `50.75/69.08/75.08 s`，聚合端到端
  输出速率为 `1.48 token/s`。
- 质量结果为风险等级准确率 `35%`、事件 micro-F1 `0.359`、不安全建议率 `0%`。
- 542 条 `tegrastats` 记录显示 GPU 利用率均值 `97.39%`、RAM 峰值 `7418 MB`、
  swap 峰值 `1904 MB`、板端输入功耗均值 `10.05 W`、GPU 峰值温度 `65.03°C`。
- 从 PS2.0 `training` 选取的 80 个独立来源组已生成联系表并完成 Codex 单轮视觉复核
  候选标注。相对 teacher 标签，33 条风险等级和 77 条事件集合被修正；当前拆分为
  48 条 LoRA train、16 条 validation 和 16 条独立 INT4 calibration，三者与 20 张
  pilot 冻结测试集均无来源组交集。该版本不是人工双人金标，仍需人工终审。
- RTX 4090 D 已完成 Qwen3-VL-2B LoRA：语言 attention 的 `q/k/v/o_proj` 共
  `6422528` 个可训练参数，占总参数 `0.301%`；1 epoch、16 个 optimizer step，
  验证损失 `0.0845`，峰值 CUDA 显存 `5.25 GiB`，训练用时 `21.54 s`。
- 同一服务器冻结测试集上，Base/LoRA 严格 JSON 均为 20/20，事件 micro-F1 从
  `0.3500` 提升至 `0.3889`，`vehicle_near_maneuver_path` 假阳性由 6 降为 0；风险
  准确率仍为 35%，`visibility_occlusion` 仍全部漏检。
- LoRA adapter 已合并为独立 checkpoint，合并模型复测质量指标与 adapter 一致；
  TensorRT Edge-LLM ONNX 导出 flow 状态为 `succeeded`。
- LoRA 合并模型的 LLM ONNX 归档与内部 7 个文件已在 Jetson 复算 SHA-256；新的
  FP16 weight-streaming LLM engine 构建 flow 状态为 `succeeded`，engine 为
  `3453786212` 字节，SHA-256 为
  `d38adc5d532615d7183a6b4aa8413020bd76a5991e49ecd51ca84d0442334224`。
- Jetson LoRA runtime 在 headless 模式和内存 compaction 后完成双 engine 加载、HTTP
  健康检查和同图连续 3/3 严格 JSON 验收；冻结 20 样本 20/20 完成，严格 JSON
  有效率为 100%，风险准确率为 35%，事件 micro-F1 从板端 Base FP16 的 `0.3590`
  提升至 `0.3889`。
- Jetson LoRA 端到端 p50/p90/p99 为 `30.60/33.08/40.90 s`，聚合输出速率为
  `1.44 token/s`；319 条遥测记录显示 RAM 峰值 `7351 MB`、swap 峰值 `580 MB`、
  GPU 利用率均值 `98.20%`、输入功耗均值 `10.10 W`、GPU 峰温 `61.94°C`。本轮输出
  token 总数比 Base 少 41.8%，因此延迟下降不能解释为 LoRA runtime 加速。
- Jetson INT4 AWQ 20 样本 20/20 完成且 JSON 有效率为 100%；相对 Edge-LLM FP16，
  平均延迟获得 `5.02x` 加速、输出速率获得 `4.95x` 提升、engine 减少 `60.5%`、
  RAM 峰值降低 `31.6%`。但事件 micro-F1 从 `0.359` 降为 0，明确记录为校准质量退化。
- 复核候选数据已完成三轮 LoRA 对照。最终 non-low x2、3 epoch adapter 在服务器冻结
  20 样本上严格 JSON 为 100%、风险准确率为 `50%`、事件 micro-F1 为 `0.182`；
  相比未平衡训练避免了全 low 塌缩，但事件 F1 低于旧弱监督 LoRA 的 `0.389`。
- 独立 ps16 领域文本已完成 LLM backbone W4A16 AWQ、ONNX 导出和 Jetson engine
  构建。新 engine SHA-256 为
  `589d8ba247a93cdf794c86697bb5a5d5fe3387fee812744c51d09806912b3026`，构建 flow
  状态为 `succeeded`。
- 新领域 INT4 在 Jetson 上 20/20 后端完成，但严格 JSON 仅 `4/20`，16 条
  `json_parse_error` 主要来自 Markdown JSON 围栏；风险准确率为 `15%`、事件
  micro-F1 为 0。端到端 p50 为 `10.68 s`、聚合输出速率为 `7.32 token/s`，性能与
  旧通用校准 INT4 相近，但质量更差，明确记录为负向实验。

上述状态证明固定版本的 Transformers FP16、Edge-LLM Base FP16、LoRA 合并模型和
INT4 均已完成相应的
`ONNX -> engine -> Edge-LLM HTTP -> ParkSight Adapter -> InferenceRecord/StudyReport`
板端实测。它不证明模型已经满足业务质量要求：当前主要误差已经从输出格式转为风险
等级和领域事件判断，仍需扩大独立人工标注数据并修正 LoRA 与量化校准偏置。
具体环境、命令和校验结果见 [`operations.md`](operations.md) 与 [`edgellm-deployment.md`](edgellm-deployment.md)。

## 尚未完成及证据边界

### 候选标注复核工具状态

80 条开发样本已经可以通过 `build_review_package.py` 固化为 Codex 候选 package，并在
人工复盘后通过 `finalize_review_package.py` 转换为标准 annotation JSONL。定稿过程现在
会校验候选和人工两份 `ParkingAssessment`、强制 `confirmed`/`corrected` 状态与内容一致，
并输出候选相对人工结果的风险准确率、事件 micro-F1 和字段修正统计。当前实际 package
仍是 `review_status=candidate`，因此尚未产生人工金标，也不能把候选结果写成最终质量结论。

- Transformers FP16 早期在旧环境/低连续内存状态下有明确 OOM 记录；当前
  R36.5.0、图形桌面和启用 swap 的环境中，固定模型完整映射到 `cuda:0` 并完成
  20 样本基线与 20 样本插桩 study。插桩期间系统 RAM 峰值 `7302/7619 MB`、swap
  峰值 `1174 MB`、最小 lfb 为 `1x2 MB`，说明当前配置可运行但内存余量很小。
- Base FP16/LoRA LLM engine 在图形桌面状态下均可能因 NvMap 无法分配约 811 MB
  视觉 engine 内存而 OOM；本轮 LoRA 实测通过临时切换 `multi-user.target`、释放显示栈
  NvMap 客户端并执行内存 compaction 完成。评测结束后已恢复图形桌面与系统服务。
- 已建立修正弱监督偏置的 80 条视觉复核候选标注并完成 LoRA 重训，但仍是 Codex 单轮
  复核，不是人工双人金标；新 LoRA 的事件 F1 低于旧实验，不能描述为整体质量提升。
- 已使用 16 条无泄漏泊车领域文本完成 INT4 重新量化和板端复测；格式有效率退化到
  20%，说明小规模领域校准没有通过质量验收。后续应先扩充并人工终审校准数据，而不是
  用宽松解析器掩盖 Markdown 围栏问题。
- 临时 `/home/ubuntu/parksight-build.swap` 已启用但未写入 `fstab`；`Device or resource
  busy` 表示重复执行 `swapon`，不是启用失败。

完整命令、结果和原始证据索引见 [`tensorrt-optimization.md`](tensorrt-optimization.md) 与本文件的当前状态记录。
## 2026-09-06 TensorRT visual builder level candidates

- 现有 level0/level1 LLM 对比复用同一个视觉 engine；但 level1 trace 中视觉侧
  `head_dim=64` FMHA 约 `57.4 ms`，是单类主要 GPU 开销。
- 新增 `build_qwen3_vl_2b_fp16_visual_engine_opt2.json` 与 `opt3.json`，固定
  Qwen3-VL-2B FP16 visual ONNX、workspace `1024 MiB`、image-token profile 和
  Edge-LLM revision，只改变 visual builder optimization level，并将输出放到独立
  engine 目录，避免覆盖正式 visual engine。
- 已在 Jetson 构建 level2/3：level2 engine generation `48.25 s`、构建期 CPU 峰值
  约 `4812 MiB`，engine SHA 为 `69083ca7...76985c6`；level3 generation `94.03 s`、
  构建期 CPU 峰值约 `4858 MiB`，engine SHA 为 `bae713b4...7f5cae`。两者均使用
  workspace `1024 MiB`、相同 image-token profile 和正式 plugin，输出目录独立。
- 后续统一 INT4 LLM、同一图片和 prompt 的 3 请求 screen 中，level2/3 E2E p50 约比
  level1 低 `12.65%/12.75%`，但输出由 `59` tokens 的 `obstacle` 变为 `51` tokens 的
  `low_speed`，完整内容和停止长度均不一致；该性能信号不能作为有效加速，候选在
  功能/数值对齐门禁拒绝，不提升默认。详细结果见
  `reports/jetson-tensorrt-revalidation/visual_builder_level2_level3_screen.json`。

## 2026-09-06 TensorRT FMHA tiled candidate screening

- 在固定 Edge-LLM `v0.9.1` revision、同一 INT4 AWQ `i768/k1024` engine、SM87
  Jetson Orin Nano 上，构建了隔离的 `0043-fmha-head128-tiled-candidate.patch`。
  候选仅在 `head_dim=128`、长序列且 `EDGELLM_FMHA_FORCE_GRANULAR_TILING=1` 时强制
  使用 tiled FMHA；插件 SHA-256 为
  `2835065c230433aff1bc01516b353500ecf2bc902054c8b6c280c18d3a7a025f`。
- 候选服务健康检查通过，decode CUDA Graph 捕获成功；请求日志确认实际出现
  `ContextFMHA: forced tiled head_dim=128 candidate`，因此不是仅靠编译产物或启动
  成功推断路径生效。
- 带 telemetry 的 5 样本筛选全部完成：TTFT p50 `783.07 ms`、E2E p50
  `10646.10 ms`、E2E mean `10721.29 ms`、cold start `12710.00 ms`。同序对比正式
  control 的前 5 个样本，候选 TTFT 均快 `5.81--8.99 ms`，E2E 均快 `3.00--23.95 ms`；
  该结果只作为筛选信号，尚不足以进入 3x20 严格 A/B 或提升默认配置。
- 筛选期间 RAM p50 `5047 MB`、swap `1230 MB`、GPU 利用率 p50 `99%`、输入功耗
  p50 `9.824 W`、GPU 温度峰值 `57.34 C`；未观察到请求失败或 OOM。原始 JSONL、摘要、
  telemetry 和 warm-up 证据位于 `reports/jetson-tensorrt-revalidation/fmha_tiled_screen_*`。

## 2026-09-09 TensorRT INT4 CTA_K=128 candidate build

- 在固定 Edge-LLM `v0.9.1` commit、JetPack R36.5/CUDA 12.6/TensorRT 10.3、SM87
  环境中，为 CTA_K 候选建立了独立 worktree。构建前排除了已确认损坏的临时 runtime
  patch，并复用已完成的隔离 kernel 构建缓存；正式 checkout、正式 plugin 和正式
  engine 均未修改。
- 最终 plugin SHA-256 为
  `af8074c10ea87c67fe6cdfe194dd75b68ff0c8d91c8750a7a025cf210259cc27`，`nm -C`
  确认包含 `gemm_w4a16_T2<64, 128, 128, 64, 32, 64, 4, 128>`；候选实际为
  `CTA_N=128、CTA_K=128`，shared-memory 静态审计值为 `99328/101376 bytes`。
- 候选服务启动和 CUDA Graph capture 已通过；恢复连接后用正式 engine 做了单样本功能
  smoke。候选输出重复括号/空格并打满 64 token 上限，严格 JSON=false；正式 plugin
  同条件输出正常 JSON 前缀。因此 CTA_K=128 未形成有效性能 A/B，已在功能 gate 拒绝，
  不提升为默认配置。

## 2026-09-09 TensorRT FMHA head_dim=64 candidate

- 基于 Nsight 中 `head_dim=64` FMHA 约 `57.4 ms` 的热点，新增隔离候选
  `0047-fmha-head64-tiled-candidate.patch`。候选只在 `SM87 + head_dim=64 + S>64` 且
  显式设置 `EDGELLM_FMHA_FORCE_GRANULAR_TILING_D64=1` 时强制 granular/tiled 路径，
  默认启发式不变。
- 候选在隔离 worktree 中编译成功，plugin SHA-256 为
  `22b944cb259ce396eff520bdafd477c26f7187367b5e89c2a0366254f687e59d`；正式 INT4
  engine 未修改。服务启动、engine 加载、视觉 runner 初始化和 CUDA Graph capture 均
  成功，日志也确认实际进入 `head_dim=64` forced tiled 分支。
- 但在同一 `ps20-indoor-001`、`max_new_tokens=64` 的功能 smoke 中，输入形状为
  `S=1444` 时 runtime 抛出 `There must be one kernel to implement the MHA`，HTTP
  连接随进程 abort 关闭。因此该候选在功能 gate 失败，未采集有效延迟 A/B，不提升为
  默认；完整证据见
  `reports/jetson-tensorrt-revalidation/fmha_d64_candidate_failure.json`。

## 2026-09-09 TensorRT INT4 CTA_K=128 functional rejection

- 使用别名 `jetson-orin-nano-super-tunnel` 恢复板端测试；正式 INT4 engine SHA 为
  `33466f3f...64afab4`，候选 plugin SHA 为 `311059ca...30b407`，符号确认包含
  `gemm_w4a16_T2<64,128,128,64,32,64,4,128>`。
- 候选服务能加载 engine、注册 plugin、完成视觉 runner 初始化和 CUDA Graph capture，
  但同一 `ps20-indoor-001`、`max_new_tokens=64` 单样本 smoke 输出重复括号/空格，
  `output_tokens=64` 打满上限且严格 JSON=false；正式 plugin 同条件输出正常 JSON 前缀。
- 因此 CTA_K=128 在功能 gate 失败，未形成有效延迟 A/B，也不提升为默认；完整证据见
  `reports/jetson-tensorrt-revalidation/cta_k128_candidate_failure.json`。

## 2026-09-09 TensorRT XQA host-side selection cache screen

- 在同一正式 INT4 `i768/k1024` engine 上，隔离编译并加载 `0036-xqa-thread-local-list-cache.patch`
  与 `0037-xqa-selection-cache.patch`。候选只缓存 XQA kernel list 和 runtime key 对应的
  function selection，不改变 engine、KV cache 布局、attention kernel、采样语义或 CUDA Graph
  配置；candidate plugin SHA-256 为
  `a781e66352ae06b4b5749cb8eafbb82074f256057b2624097e9145122e8e8eb0`。
- 候选和正式 plugin 均完成 engine 加载、视觉 runner 初始化、CUDA Graph capture，并在同一
  `ps20-indoor-001` 请求上返回相同内容、HTTP 200 和 59 completion tokens。使用
  `max_tokens=256` 的 3 次端到端 screen 中，E2E p50 为 candidate `8089.00 ms`、control
  `8085.39 ms`；没有可见收益。该 screen 只采集客户端 E2E，未采集 prefill/TTFT/decode
  阶段和板端 telemetry。
- 两侧该请求的业务结构均未通过严格 schema（短 screen 结果，不作为质量结论）；候选未发现
  回归，但收益不足以进入 3x20 严格 A/B 或默认晋级。完整数字、输出一致性和 provenance 见
  `reports/jetson-tensorrt-revalidation/xqa_selection_cache_screen.json`。

## 2026-09-09 TensorRT visual builder level2/3 screen

- 固定正式 INT4 LLM engine、FP16 visual ONNX、workspace `1024 MiB`、image-token profile、
  Jetson Orin Nano `SM87` 和 Edge-LLM `v0.9.1`，分别构建 visual builder level2/3；正式
  visual level1 engine 未覆盖。level2/3 engine SHA 分别为
  `69083ca7badc5f2e3d2147694913fc9c18af5bd620ad5151fd00fcddb76985c6` 和
  `bae713b48c2f5374994d1de574e243fd846858323056c7476c447873907f5cae`。
- 三组服务均能加载 engine、初始化 visual runner、完成 CUDA Graph capture 并返回 HTTP
  200。3 请求 E2E p50 为 level1 `8085.77 ms`、level2 `7062.58 ms`、level3
  `7054.15 ms`；但 level2/3 均只生成 `51` tokens，level1 生成 `59` tokens，且完整
  输出内容不同。该结果说明存在性能信号，但不满足 matched functional/quality gate，
  不能称为 builder level 加速。
- level2/3 均未进入 3x20、TTFT/prefill/decode、telemetry 或 soak；默认视觉 engine
  保持 level1。完整逐请求输出和构建 provenance 见
  `reports/jetson-tensorrt-revalidation/visual_builder_level2_level3_screen.json`。
- 为补足 tactic 证据，在隔离 worktree 使用 `EDGELLM_PROFILING_VERBOSITY=detailed` 构建
  visual level2 诊断 engine，并用带正式 Edge-LLM plugin 的 `trtexec --dumpLayerInfo` 读取
  Inspector；engine SHA 为 `e1ede1d90c0e4ff5c290292bd6fd048e433ef57e99443be066bf7a136c2e5de9`，
  原始日志见 `reports/jetson-tensorrt-revalidation/visual_detailed_opt2_trtexec.log`。
  Inspector 明确暴露了 TensorRT `fusion`/`kgen` 层及其 FP16 GEMM tactic，说明图级融合和
  codegen kernel 在视觉 engine 中实际存在；`ViTAttentionPlugin` 仍是 PluginV3 边界，
  且没有暴露 plugin tactic metadata。
- 同一批 level1/level2 Nsight trace 中，D64 FMHA 总 kernel 时间约为 `169.01/171.10 ms`，
  W4A16 总 kernel 时间约为 `789.97/789.31 ms`；没有证据表明 visual level2 改善了
  attention 或 INT4 主热点。level2 的 E2E 信号主要伴随少生成 8 tokens 以及较少的
  `cudaStreamSynchronize`/launch 事件，不能归因于视觉 kernel 加速。详细 Inspector 边界、
  SHA 和证据索引见 `reports/jetson-tensorrt-revalidation/visual_detailed_profiling_evidence.json`。

## 2026-09-09 TensorRT INT4 GEMV N_PER_BLOCK=4 rejection

- 针对单 token decode 的输出通道并行路径，新增隔离 `0038-int4-gemv-nperblock4.patch`，
  仅在 `M=1`、`N` 可被 16 整除且显式设置 `EDGELLM_INT4_GEMV_N_PER_BLOCK=4` 时调用
  `gemv_kernel<4,1,256,128>`；`nm -C` 确认该 kernel 符号进入候选 plugin，plugin SHA 为
  `9264afd03a5aa124e758db16514d4ce9fa297bec10b174cf1abbc16007f83df2`。
- 复用正式 INT4 `i768/k1024` LLM engine 和正式 visual engine 完成 3 次单样本 smoke；服务
  初始化和 decoder CUDA Graph capture 成功，但 3/3 请求均生成 `256` tokens 打满上限，
  输出为重复/乱码片段，严格 JSON 为 `0/3`。因此不进入 latency benchmark，正式 plugin、
  engine 和默认配置均未改变。完整失败证据见
  `reports/jetson-tensorrt-revalidation/int4_gemv_nperblock4_candidate_failure.json`。

## 2026-09-09 TensorRT INT4 GEMV block-size=128 screen

- 在保留 GEMV 默认 `N_PER_BLOCK=2` 的前提下，新增显式 `EDGELLM_INT4_GEMV_BLOCK_SIZE=128`
  候选，编译产物包含 `gemv_kernel<2,1,128,128>` 和 `gemv_kernel<2,1,512,128>`；本次只
  选择 128-thread 路径，候选 plugin SHA 为
  `6839c2a6c7716a1e24a05b363860945df6a5c23a3af501c0c320f3a9cb5f72aa`。
- 复用同一正式 INT4/visual engine 完成 3 次 screen，均 HTTP 200、59 completion tokens，
  内容逐次一致且与 formal level1 screen 相同；candidate E2E p50 为 `8044.16 ms`，历史
  control p50 为 `8085.77 ms`，初步差异约 `-0.52%`。由于 control 未在同一轮重跑、尚无
  TTFT/decode/telemetry 和完整严格质量门禁，该结果仅保留为候选信号，不进入默认 plugin。
- 隔离 worktree 的完整 build 过程还出现 `llm_inference` fatbin device-link 错误，但
  `NvInfer_edgellm_plugin.so` 已链接且可加载；后续正式 3×20 前需要先修复/隔离该构建
  并行链接边界；适配当前源码的可应用 patch 为 `0049-int4-gemv-block-size-candidate-v2.patch`。
  结果见 `reports/jetson-tensorrt-revalidation/int4_gemv_block128_screen.json`。

## 2026-09-09 TensorRT visual profile max_image_tokens=1536 rejection

- 当前实际视觉序列长度约为 `1444`，因此构建独立 level1 visual profile，将
  `max_image_tokens/max_image_tokens_per_image` 从 `2048` 收紧到 `1536`；engine SHA 为
  `97d9a5ed71c9d9868f0d273479363cd78341f54e95064147c63aaec19ae0e0dd`，大小为
  `823569284` bytes，未覆盖正式 visual engine。
- 该 profile 的 shared execution context memory 从正式 level1 的 `423624704` bytes 降到
  `317718528` bytes，约减少 `25%`，并在当前 swap/连续内存压力下成功加载和完成 decoder
  CUDA Graph capture。
- 但同一图片/prompt 的 3 次请求均生成 `51` tokens，内容由 level1 的 `obstacle` 变为
  `low_speed`；因此不能把较低的 E2E 写成 profile 加速，候选按功能对齐门禁拒绝。完整
  证据见 `reports/jetson-tensorrt-revalidation/visual_max1536_profile_screen.json`。

## 2026-09-09 TensorRT visual profile max_image_tokens=1792 rejection

- 继续进行 profile sweep，构建 `max_image_tokens/max_image_tokens_per_image=1792` 的独立
  level1 visual engine，activation memory 为 `370671616` bytes，相比正式 level1 的
  `423624704` bytes 减少约 `12.5%`，并成功完成 engine load 与 decoder CUDA Graph capture。
- 3 次同图片/prompt 请求均稳定生成 `51` tokens 的 `low_speed` 结果，仍与正式 level1 的
  `59` tokens/`obstacle` 不一致；较低 E2E 不能作为有效加速，候选拒绝。engine SHA、构建
  日志和完整输出见 `reports/jetson-tensorrt-revalidation/visual_max1792_profile_screen.json`。

## 2026-09-09 TensorRT INT4 GEMV block-size matched decode A/B

- 为隔离视觉 runner 的端到端噪声，使用正式 `i768/k1024` INT4 LLM engine 和 `llm_bench`
  完成 `pastKVLen=256/768/1024`、batch=1、warm-up=10、50 次 layer profile 的 matched A/B。
  候选仅通过 `EDGELLM_INT4_GEMV_BLOCK_SIZE=128` 选择 `gemv_kernel<2,1,128,128>`，control
  保持正式 256-thread 路径；三个 KV 形状候选总时延分别为 `128.4026/129.2875/129.7568 ms`，
  control 为 `128.6130/129.5464/129.9656 ms`，差异为 `-0.16%/-0.20%/-0.16%`。
- 为覆盖 CUDA Graph 路径，`pastKVLen=768` 再完成 100 次 decode：128-thread 候选为
  `126.4849 +/- 0.1609 ms`，256-thread control 为 `126.6739 +/- 0.1866 ms`，仅快
  `0.15%`；512-thread 分支为 `132.1313 +/- 0.1552 ms`，反而慢 `4.31%`。三组 Graph
  均 capture 成功，但 128-thread 未达到项目晋级门槛。
- 结论为 `rejected_no_material_gain`：正式 256-thread 路径保持不变，默认 plugin/engine
  未替换。layer profile 的主导 `node_linear` 被归入 `kgen_other`，不是直接 GEMV kernel
  计时；完整数值、CSV、日志和 provenance 见
  `reports/jetson-tensorrt-revalidation/int4_gemv_block_size_llm_bench.json`。

## 2026-09-09 TensorRT INT4 LLM builder level2/level3 build boundary

- 对固定 INT4 ONNX、`i768/k1024`、workspace `1024 MiB` 分别尝试 builder optimization
  level2 和 level3，均在 tactic 选择阶段因约 `622,329,856` bytes CUDA allocation 报
  `CUDA error 2`，随后出现 `Could not find any implementation for node ...`，没有生成
  可推理 engine。因此 level2/3 当前只有“构建资源失败”证据，不能推断其推理速度或质量。
- 使用隔离 worktree 的 patched builder 成功构建 level1 detailed 诊断 engine；Inspector
  明确显示最终 `node_linear` 为 `[1,1,2048] x [1,2048,151936]` 的 FP16 `gemm`，选择
  `sm80_xmma...128x128x32...stage4` Tensor Core tactic。该热点不是 INT4 GEMV，后续应
  优先围绕 LM-head GEMM tactic/shape 与 CPU launch 开销做实验。
- 正式 level1 engine、默认 plugin 和默认配置均未修改。build failure、诊断 engine
  provenance、Inspector JSON 和原始日志见
  `reports/jetson-tensorrt-revalidation/int4_llm_builder_level2_level3_screen.json`。

## 2026-09-09 TensorRT INT4 level0/level1 decode repeatability

- 修正 engine 目录和 plugin 环境后，使用同一 `llm_bench` 命令对正式 level0 与候选 level1
  各完成 3 次重复；每次 batch=1、`pastKVLen=768`、warm-up=10、20 次 decode、OSL=1、
  CUDA Graph、seed=0，六次运行均返回成功。
- level0 三次 E2E decode 为 `126.6943/126.6258/126.7423 ms`，均值 `126.6875 ms`；
  level1 为 `27.4520/27.3945/27.3921 ms`，均值 `27.4129 ms`。level1 相对 level0
  平均加速 `4.6215x`，平均延迟下降 `78.3618%`，复验范围内结果稳定。
- 该报告只覆盖低层 decode，不替代完整 VLM 的 TTFT、严格 JSON、风险准确率、事件
  micro-F1、峰值内存和 soak gate；level1 仍保持候选状态。每个进程退出前均有 TensorRT
  `IRuntime::~IRuntime` API-usage warning，已记录为需要单独修复/排查的 runtime 生命周期问题。
  详细 engine SHA、环境、逐次日志和计算口径见
  `reports/jetson-tensorrt-revalidation/int4_level0_level1_decode20_repeats_20260909.json`。

## 2026-09-09 TensorRT llm_bench runtime 生命周期诊断

- 只读检查 `llm_bench` 后确认 warning 来自 metadata 辅助路径：`loadStandaloneEngine()`
  在函数内部创建 `IRuntime`，返回 `ICudaEngine` 后立即析构 runtime，违反 TensorRT 要求的
  engine 先于 runtime 析构顺序。该对象不是主推理 executor，但会在每次 benchmark 结束时
  输出 `mEngineCounter.use_count() == 1` warning。
- 新增独立候选 `patches/tensorrt-edge-llm/0050-llm-bench-keep-standalone-runtime-alive.patch`，
  让 main 持有 standalone runtime，保证 engine 先析构。该 patch 仅在本地留存，未修改正式
  Edge-LLM checkout，尚未在板端重新编译验证；因此当前报告仍将 warning 标记为已知问题。

## 2026-09-09 TensorRT paged KV capability audit

- 固定 v0.9.1 checkout 的 `llm_build --help` 只暴露 `--maxKVCacheCapacity`，没有 paged pool、
  tokens-per-page、page table 或 paged-KV 模式选择参数；当前 Qwen3-VL engine provenance 的
  `usePagedKVCache` 为 `false`。
- 源码层面仍能看到 paged KV 的 kernel 类型、page-list 校验、paged XQA 选择以及 CuTe DSL
  paged FMHA（FP16/FP8-input）实现，但 attention plugin 两处将 `usePagedKVCache` 硬编码为
  `false`。因此“源码有 paged kernel”不等于当前 engine 已启用 paged KV。
- 该路线当前结论为 `not_wired_for_current_checkout`，未进行性能 A/B；只有 builder/runtime
  接口完整暴露后，才继续验证 page pool、page table、容量、数值一致性、动态 batch、decode
  和 soak。审计证据见
  `reports/jetson-tensorrt-revalidation/int4_paged_kv_capability_audit_20260909.json`。

## 2026-09-09 TensorRT 0023 fallback binding scan A/B 无效复验

- 曾用同一 level1 engine 和候选 plugin 做 control/`EDGELLM_SKIP_FALLBACK_BINDING_SCAN=1`
  各 3×20 decode；control 均值 `27.3826 ms`，skip 请求均值 `27.3815 ms`。
- 复核隔离 worktree 后发现其 `EngineExecutor` 源码没有 `0023` 的
  `mSkipFallbackBindingScan`/`mAllEngineIORegistered` 实现，环境变量未进入实际路径；因此
  这组结果标记为 `invalid_experiment_no_candidate_code`，既不能证明也不能否定 0023 的收益。
- 六份日志和无效原因保留在
  `reports/jetson-tensorrt-revalidation/int4_level1_skip_fallback_scan_invalid_20260909.json`；
  后续必须先构建真正包含 0023 的候选 benchmark/runtime，再复用同一 workload 重测。

## 2026-09-09 TensorRT level1 KV capacity screen

- 复用板端已有的 level1 `i768/k768` 与 `i768/k1024` engine，在共同可表示的
  `pastKVLen=512`、batch=1、warm-up=10、20 次 decode、CUDA Graph 条件下各完成 3 次重复，
  六次 Graph capture 均成功。
- k768 三次均值为 `26.9211 ms`，k1024 为 `26.9222 ms`，差异仅约 `0.004%`，没有低层
  decode material gain。KV profile 收紧可作为内存压力缓解手段，但不能据此声称延迟优化；
  k768 也不能覆盖原 `pastKVLen=768` workload 的额外 headroom。
- engine SHA、逐次数据和限制见
  `reports/jetson-tensorrt-revalidation/int4_level1_kv_capacity_decode20_20260909.json`。

## 2026-09-09 TensorRT level0/level1 low-level prefill repeatability

- 固定 `inputLen=768`、`reuseKVLen=0`、batch=1、warm-up=10、20 次重复，对正式 level0
  和候选 level1 各完成 3 次 prefill；六次运行成功。
- level0 三次为 `526.9898/526.6765/526.9469 ms`，均值 `526.8711 ms`；level1 为
  `426.8245/426.8257/426.5893 ms`，均值 `426.7465 ms`。低层 prefill 的 level1/level0
  平均加速 `1.2346x`，延迟下降约 `19.04%`。
- 该结果补充了 builder level 对 prefill 的隔离证据，但仍不替代完整 VLM TTFT/E2E、质量、
  内存与 soak 晋级门禁。逐次日志和 engine provenance 见
  `reports/jetson-tensorrt-revalidation/int4_level0_level1_prefill768_decode20_20260909.json`。

## 2026-09-09 TensorRT level1 KV capacity near-limit screen

- 在共同可表示的 `pastKVLen=767` 下，对已有 level1 `i768/k768` 与 `i768/k1024` engine
  各完成 3×20 decode，六次 CUDA Graph capture 均成功。
- k768 均值 `27.3488 ms`，k1024 均值 `27.3145 ms`；k768 反而慢约 `0.13%`，并出现一次
  `27.4442 ms` 波动。profile 收紧没有稳定 decode 收益，且 k768 无法为原 `pastKVLen=768`
  workload 提供额外 headroom。
- 详细数据与逐次日志见
  `reports/jetson-tensorrt-revalidation/int4_level1_kv_capacity_past767_decode20_20260909.json`。

## 2026-09-09 TensorRT level1 low-level 1000-step soak

- 对 level1 `i768/k1024` engine 在 `pastKVLen=768`、batch=1、warm-up=10、CUDA Graph 条件下
  连续执行 1000 次 decode，进程退出码为 0，Graph capture 成功，未发生 runtime failure、
  崩溃或超时。
- 平均 decode E2E 为 `27.3821 ms`，报告标准差 `0.0197 ms`，E2E 吞吐 `36.5202 tok/s`。
  `llm_bench` 当前 CSV 只保存聚合值，没有逐 iteration 样本，故不虚构 p50/p90/p99。
- soak 仍重复出现 standalone metadata engine 的 TensorRT runtime 生命周期 warning；它不影响
  本次进程退出码，但在修复 `0050` 并重新验证前，不能作为生产候选的最终稳定性证明。原始
  log、CSV 和完整状态见 `reports/jetson-tensorrt-revalidation/int4_level1_decode1000_soak_20260909.json`。

## 2026-09-09 TensorRT k768/k1024 layer profile boundary

- 对 k768/k1024 在 `pastKVLen=767`、batch=1、warm-up=10、20 次非 Graph decode 做逐层 profile。
  可见非零 layer timing 总和为 `30.0687/30.1159 ms`，最终 `node_linear` LM-head GEMM 为
  `10.2523/10.2497 ms`，均没有实质差异。
- profiler 没有将 attention/XQA plugin 执行暴露为可独立归因的时间行，因此不能从该 CSV
  宣称 attention kernel 加速；总 decode 结论仍以 CUDA Graph E2E A/B 为准。完整 CSV、日志和
  边界说明见 `reports/jetson-tensorrt-revalidation/int4_level1_kv_capacity_profile20_20260909.json`。

## 2026-09-09 TensorRT 候选 plugin 非 Graph 调度 A/B

- 使用同一个 level1 `i768/k1024` engine、`pastKVLen=768`、batch=1、warm-up=10、20 次
  decode、OSL=1、seed=0，对正式 plugin 和隔离候选 plugin 各完成 3 次非 CUDA Graph 重复。
  候选 plugin 包含 XQA list/function selection cache 等已有候选改动，属于 plugin-stack A/B，
  不是单个 patch 的归因实验。
- 候选三次为 `29.5547/29.5476/29.5509 ms`，均值 `29.5511 ms`；control 为
  `30.0580/29.5455/29.5604 ms`，均值 `29.7213 ms`。均值表面下降 `0.5728%`，但中位数
  仅下降 `0.0321%`，且 control 有一个 `30.0580 ms` 波动，未达到 material-gain 门槛。
- 该候选 plugin 不晋级，正式 plugin、engine 和默认配置均未改变。Graph 路径的前序 A/B
  也没有观察到稳定收益；后续优先处理真实 benchmark runtime 生命周期 warning，并继续围绕
  LM-head GEMM/tactic 和可归因的 attention/XQA kernel 做单变量实验。详细 provenance、CSV
  和限制见 `reports/jetson-tensorrt-revalidation/int4_candidate_plugin_nograph_ab_20260909.json`。

## 2026-09-09 TensorRT level1 Nsight decode kernel boundary

- 对正式 level1 engine 在 `pastKVLen=768`、batch=1、warm-up=3、5 次非 Graph decode 上采集
  Nsight Systems trace。GPU kernel 汇总中，`gemv_kernel<2,1,256,128>` 占 `52.3%`、FP16
  `lm_head` GEMM `trt_ampere_h16816gemm_128x64_ldg8_tn_v1` 占 `34.2%`，`kernel_mha`
  占 `6.9%`。
- 该证据把当前 decode 优先级从 XQA host selection cache 转移到 INT4 GEMV；`CTA_N/CTA_K`
  改动主要作用于 `M>1` 的 W4A16 GEMM，应在 prefill 口径测试，不能直接当作 batch=1 decode
  优化。CUDA API 汇总包含初始化和 warm-up，未被误算为单步 decode 时间。
- 原始 `.nsys-rep`、stats 和 benchmark CSV，以及 engine/plugin provenance 见
  `reports/jetson-tensorrt-revalidation/int4_level1_formal_nograph_nsys_20260909.json`。

## 2026-09-09 TensorRT INT4 GEMV 192/320-thread A/B

- 针对 Nsight 定位出的 `gemv_kernel<2,1,256,128>`，在隔离 plugin 中增加 192 和 320
  threads/block 候选，保持 `NPerBlock=2`、group size、权重/scale 布局、反量化算术和输出
  索引不变；两种候选的 batch=1 编译资源均为 40 registers、0 local memory。
- 固定 level1 engine、`pastKVLen=768`、batch=1、warm-up=10、20 次 decode，各完成 Graph
  和非 Graph 3 次重复。192-thread 的 Graph 均值为 `27.9317 ms`，相对 control
  `27.4129 ms` 慢 `1.8927%`；320-thread 为 `29.6242 ms`，慢 `8.0668%`。非 Graph 分别
  慢 `1.3206%` 和 `7.1501%`。
- 两个候选均拒绝，正式 256-thread 路径保留。该结果说明在当前 Orin Nano、Qwen3-VL-2B
  decode shape 下，单纯调整 GEMV block size 不能获得收益；后续应转向融合反量化/GEMV 或
  输出投影专用路径，并补齐严格输出和质量门禁。详细 CSV、日志、编译资源和 provenance
  见 `reports/jetson-tensorrt-revalidation/int4_gemv_block192_320_ab_20260909.json`。

## 2026-09-09 TensorRT INT4 GEMV read-only cache A/B

- 在同一隔离 candidate stack 中，仅对 INT4 batch=1 GEMV 的权重 `float4`、scale `half` 和
  activation `float4` 读取增加 `__ldg`，保持 kernel shape、launch geometry、反量化算术、
  索引、engine 和 workload 不变；control/candidate 的 batch=1 编译资源均为 40 registers、
  0 local memory。
- 固定 level1 `i768/k1024` engine、`pastKVLen=768`、batch=1、warm-up=10、20 次 decode，
  非 Graph 三次均值为 `29.565533/29.515100 ms`，candidate 下降 `0.170582%`；Graph 三次
  均值为 `27.325133/27.335100 ms`，candidate 反而上升 `0.036474%`，六次 Graph capture
  均成功。
- 该收益低于 material-gain 门槛，patch 0052 拒绝，正式 plugin、engine 和默认配置均未改变。
  该低层 screen 也未采集严格 JSON、领域质量或完整 VLM TTFT。完整 provenance、CSV、日志
  和构建日志见 `reports/jetson-tensorrt-revalidation/int4_gemv_readonly_cache_ab_20260909.json`。

## 2026-09-09 TensorRT INT4 GEMM CTA_N=256 prefill A/B

- 针对 `M>1` 的 W4A16 GEMM 路径，在同一隔离 candidate stack 中将 `CTA_N` 从 128 改为
  256，保持 `CTA_M/CTA_K`、pipeline stages、权重布局、反量化算术、engine 和 workload
  不变；control/candidate 均成功编译。
- 固定 level1 `i768/k1024` engine、`inputLen=768`、`reuseKVLen=0`、batch=1、warm-up=10、
  20 次 prefill，control 三次均值 `426.764067 ms`，candidate 三次均值 `500.320533 ms`，
  candidate 延迟上升 `17.235862%`、E2E tokens/s 下降 `14.699261%`。
- CTA_N=256 明确拒绝，正式 plugin、engine 和默认配置均未改变；该低层 prefill 结果不等同
  于完整 VLM TTFT。完整 provenance、CSV、日志和构建日志见
  `reports/jetson-tensorrt-revalidation/int4_gemm_cta_n256_prefill_ab_20260909.json`。

## 2026-09-09 TensorRT INT4 GEMM CTA_M=128 prefill A/B

- 针对 `M>1` 的 W4A16 GEMM 路径，在同一隔离 candidate stack 中将 `CTA_M` 从 64 改为
  128，保持 `CTA_N=128`、`CTA_K=64`、stages、权重布局、反量化算术、engine 和 workload
  不变；control/candidate 均成功编译。
- 固定 level1 `i768/k1024` engine、`inputLen=768`、`reuseKVLen=0`、batch=1、warm-up=10、
  20 次 prefill，control 三次均值 `426.730233 ms`，candidate 三次均值 `497.884333 ms`，
  candidate 延迟上升 `16.674258%`、E2E tokens/s 下降 `14.291291%`。
- CTA_M=128 明确拒绝，正式 plugin、engine 和默认配置均未改变；该低层 prefill 结果不等同
  于完整 VLM TTFT。完整 provenance、CSV、日志和构建日志见
  `reports/jetson-tensorrt-revalidation/int4_gemm_cta_m128_prefill_ab_20260909.json`。

## 2026-09-09 TensorRT INT4 GEMV __restrict__ pointer A/B

- 在现有 `gemv_kernel<2,1,256,128>` 上仅给 inputs、weight、scales、outputs 增加
  `__restrict__`，不改变 kernel shape、launch geometry、反量化算术、索引、engine 或
  workload；candidate/control 均成功编译。
- 固定 level1 `i768/k1024` engine、`pastKVLen=768`、batch=1、warm-up=10、20 次 decode，
  non-Graph 三次均值为 `29.526367/29.503667 ms`，candidate 下降 `0.076880%`；CUDA Graph
  三次均值为 `27.323733/27.477100 ms`，candidate 上升 `0.561295%`，所有 Graph capture
  均成功。
- 该编译器提示没有形成稳定收益，patch 0053 拒绝，正式 plugin、engine 和默认配置均未改变。
  完整 provenance、CSV、日志和构建日志见
  `reports/jetson-tensorrt-revalidation/int4_gemv_restrict_pointers_ab_20260909.json`。
