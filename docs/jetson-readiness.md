# Jetson 部署核验

## 2026-10-08 环境快照

通过 Windows OpenSSH 和 `C:/Users/22122/.ssh/config` 中的
`jetson-orin-nano-super-tunnel` 完成只读检查。项目目录为
`/home/ubuntu/JetsonVLM`，TensorRT Edge-LLM 目录为
`/home/ubuntu/TensorRT-Edge-LLM`。本轮未执行板端 GPU 推理、安装或功耗调整。

| 项目 | 实测值 |
| --- | --- |
| 系统 | aarch64，内核 `5.15.185-tegra`，L4T R36.5.0 |
| JetPack | `6.2.2+b24` |
| TensorRT 系统包 | `libnvinfer10 10.3.0.30-1+cuda12.5` |
| Python | 3.10.12 |
| 当前功耗模式 | `15W`，mode ID 0 |
| 板端项目 HEAD | `f362a4397a2bfd58b7367201c4f26d2b3fc3dd0e` |
| Edge-LLM HEAD | `7f061f21f0a581ba234a1e233c9315b89d8e47d6` |

Edge-LLM 工作树原始 diff 涉及 930 个文件；忽略行尾空白后仍有
36 个文件、2008 行增加和 216 行删除。另有未跟踪的 paged-KV 文件。
因此 HEAD 不能独立表示实际 runtime 源码身份；继续实验须保留现有改动并记录补丁、
插件和 binding 的文件哈希。

## 已有 FP16 产物

`artifacts/engines/qwen3_vl_2b_fp16/` 同时存在 LLM 和视觉 engine。
`build/examples/llm/llm_inference` 已存在。检查监听端口时没有发现 8000 服务，
不能据此排除其他服务方式。

LLM 配置记录 Edge-LLM 0.9.1、FP16 KV cache、batch 上限 1、输入上限
1024 和 KV capacity 2048；视觉配置记录每张图像最多 2048 image tokens。
这些字段不证明 engine 实际执行精度或与当前插件兼容。

板端两份 FP16 study 配置声明相同的模型 revision
`89644892e4d85e24eaac8bacfd4f463576704203`、15W 模式和 PS20 测试集。
工作负载内容与本地一致：448×448、最多 256 个新 token、`do_sample=false`。
随后 SHA-256 核验确认 PS20 全部 20 张图片、manifest、标注和两份 study 配置与
本地逐字节一致。固定 revision 缓存中的 `model.safetensors`、模型配置、视觉
预处理配置和 tokenizer 配置也一致；这仍不等同于完整 tokenizer 目录核验。
板端虚拟环境的包元数据为 Torch 2.9.1、Transformers 4.57.6、Accelerate 1.12.0。

当前 FP16 LLM engine 为 3453798316 字节，视觉 engine 为 824000540 字节。
两者与插件、binding、推理程序的实际哈希保存于 `file_identity.json`；权重与
环境元数据保存于 `model_environment.json` 和 `model_parity.json`。
文件哈希证明本次采集的文件身份，尚未证明与旧构建记录的绑定关系。

## 下一步验收

本地已新增 `jetson_edgellm_fp16_workload_resize_ps20_pilot.json`，显式固定客户端
448×448 RGB PNG 输入；旧配置仍传原图路径。实际 Pillow 离线核验的 20 张图片
与 vLLM 上传内容逐像素一致，结果保存于 `image_preprocessing_verification.json`。
Adapter 随后已在隔离板端目录完成真实 HTTP 单图验收；这份 FP16 study 配置尚未
运行完整 PS20，不能用本地图片核验或 INT4 单图替代 FP16 study。

C++ `qwenViTRunner.cpp` 从视觉 builder 配置读取 min/max image tokens，
从预处理配置读取 patch=16、merge=2；尺寸按 factor=32 对齐。
当前已核验的基础 FP16 视觉 engine 配置对应像素范围 8192～2097152，
448×448 已对齐且处于范围内，因此按该源码规则预期不会再次改变尺寸。
这不是现有二进制或视觉张量数值一致性的实测结论，其他 engine 需分别核验配置。

新增 `serve_edgellm.py --stream-usage` 可从 native delta token IDs 补齐 SSE
completion usage；安装只发生在服务进程内。当前签名及客户端解析的离线验证见
`stream_usage_source_snapshot.json` 和 `stream_usage_verification.json`。
新增入口随后已在板端隔离 HTTP 服务启用并完成单图验收；进一步开启串行阶段测量
后完成阶段时间与 prompt/decode 回传验收，普通 usage 模式仍不补估这些字段。

阶段接口核验确认：现有 binding 只导出 profiling 开关和累计 token/run metrics，
没有导出 `gTimer` 的 CUDA event 时间。新增
`scripts/experiments/prepare_edgellm_timing_binding.py` 可针对已核验源码生成增量补丁，增加
`get_stage_timing_snapshot()`，按原始 stage ID 返回累计 GPU 毫秒和测量次数。
生成前必须给出匹配的源码 SHA-256；工具不修改 checkout、不编译且不覆盖已有输出。

当前 Jetson 源码的 CRLF 补丁已在本地临时目录通过 `git apply --check`、实际应用
和结果源码哈希校验，记录为 `native_timing_patch_verification.json`。
随后已在 Jetson 独立目录完成 binding 单文件编译与链接，候选模块提供该接口；
默认部署使用的原 binding 未替换。
Timer 是全局可变状态，快照仅限全部 native worker 停止执行时读取；累计快照之差
也不能直接用于并发请求归因。验收先采用独占服务、单请求和单序列，再核对各 stage
实际边界，不能把一个总时间拆估为 vision/prefill/decode。

## 独立 binding 构建与导入验收

`scripts/experiments/build_edgellm_timing_binding.py` 默认生成只读构建计划，显式 `--execute`
才在新输出目录复制 binding 源码、校验和应用补丁、编译单个对象并链接模块。
复用已有静态库和 device-link 对象，不调用整个 CMake build；记录编译/链接参数与
输入哈希，输出路径已重定向，原源码及链接输入在结束时复核。目录存在时拒绝覆盖。

2026-10-08 板端产物位于
`/home/ubuntu/JetsonVLM/artifacts/runtime/native-timing-20261008/binding/`。
实际构建耗时 46.45 秒，候选 `.so` SHA-256 为
`c6eb4cabc0cc373e569617c3cc98f65e3828d4ef47343a36bc96b48f79e4c940`。
新 Python 进程导入了该路径，`get_stage_timing_snapshot()` 返回空字典，
`get_profiling_enabled()` 为 false。native PNG 加载验证为 448×448、3 通道、1 帧；
未创建 LLM runtime 或加载 engine，不能替代视觉张量和模型推理验收。

原 binding SHA-256 仍为
`68ef953cf30893524d3a5dc9a8cb37607aba58aa797dac005325a8c0a220faef`。
构建与导入证据见本地 `native_timing_build_result.json` 和板端 `build_record.json`、
`build.log`、`probe.json`。实际验收发现 upstream `_import_runtime()` 会按文件路径
重新加载模块；仅调整 `PYTHONPATH` 不足以选择候选绑定，提前导入还会触发重复类型
注册。`scripts/experiments/run_edgellm_native_probe.py` 在构造 runtime 时临时指定已验证模块，
结束后恢复 loader。该选择机制已复用至正式 HTTP 入口，并要求明确的候选目录和
SHA-256；默认仍使用 upstream loader。

## 2026-10-08/09 单图诊断

单图入口默认只检查输入，显式 `--execute` 才运行。它复制 engine 元数据、只对
engine/embedding 二进制建立符号链接，按冻结 workload 生成 448×448 RGB PNG，
调用实际 native SSE generator 与客户端解析器。读取全局 Timer 前确认新建 worker
已停止；这不是 HTTP 性能基线，也不写回既有 StudyReport。

FP16 `qwen3_vl_2b_fp16_i768_k1024` 在预算 0、batch 1、15W 模式下加载 LLM 后，
视觉 engine 反序列化申请 811032832 字节时出现 TensorRT OOM。没有生成 token，
不能报告 TTFT 或阶段吞吐。Python 顶层异常为 `No valid multimodal engine found`，
真实 OOM 归因来自 TensorRT 日志；证据见 `native_single_image_failures.json`。
此前两次诊断分别暴露隔离视图未遍历视觉目录符号链接、loader 重复导入问题；两者
均已修复，不归为 GPU OOM。失败后内存 available 为 5331 MiB，swap 已用 1563 MiB。

已有 INT4 `qwen3_vl_2b_int4_awq_i768_k1024` 不设置 weight streaming 预算，LLM、
视觉 engine 与 decode Graph 均初始化成功。随后指标读取出现绑定类型错误：
`get_total_runs()` 的继承成员指针要求未导出的 `BaseMetrics`。
新增补丁选项 `--fix-metrics-runs` 使用明确派生类型的 lambda 修复四类 metrics；
记录补丁身份并在新目录编译，不替换原绑定。该次诊断尚未执行生成，证据见
`native_int4_observation1.json`。

修复候选 binding 编译耗时 46.13 秒，SHA-256 为
`6dae87aed5c65afeb42a393e73f2346cb00f8af2c03f6c3e47dde36d5a2e79db`。
2026-10-09 使用该模块完成 INT4 单图诊断，严格 JSON 校验通过；native generated
计数与 SSE completion usage 均为 95，computed prompt 为 570、reused 为 0。
Timer 实测 vision 1 次 203.98 ms、prefill 1 次 425.85 ms、decode 94 次合计
12001.78 ms。首 token 由 prefill 产生，后续 94 次 vanilla decode 与计数吻合；
该单次诊断对应 7.832 decode tokens/s，不作为未插桩吞吐基线。

runtime 初始化 6.020 秒，客户端 PNG 准备 269.05 ms，进程内流式首文本延迟
659.54 ms、完整流耗时 12.670 秒；没有 HTTP 传输。初始化、生成到进程退出期间
500 ms tegrastats 采样共 48 条，设备 RAM/swap 峰值 5087/1580 MiB，GPU 温度
峰值 45.875°C，采样输入功耗均值 7.741 W；这些不是逐请求模型资源用量。
原 binding 哈希与 engine 元数据保持不变。完整原始流、输出、前后快照、二进制
身份、日志及 tegrastats 保存于 `native_int4_diagnostic.json`；板端运行目录为
`artifacts/runtime/native-timing-20261008/single-image-int4-20261009-retry2/`。

候选已验证 native 单图采集、HTTP 绑定选择及串行阶段回传；冻结 PS20 study
仍待完成。FP16 OOM 继续作为独立实验事实保留，不能用 INT4 成功替代它。

## 2026-10-09 真实 HTTP 单图

`serve_edgellm.py` 新增 `--native-binding-directory` 与
`--native-binding-sha256`，两项必须同时提供。启动与 `--check-only` 都验证目录
仅有一个候选 `.so` 且哈希匹配；静态检查不导入 CUDA。进程已加载其他绑定时拒绝
覆盖，构造完成恢复 upstream loader，metadata 保存实际候选及选择代码身份。

在新目录 `artifacts/runtime/http-metrics-20261009/` 复制包及 engine 元数据，
使用上述候选 binding 在 localhost 18080 启动实际服务。真实 HTTP SSE 返回
95 个 completion token，94 个非空文本事件，严格 JSON 有效。客户端按冻结
workload 生成临时 448×448 RGB PNG，同机路径协议已实际执行。

HTTP TTFT 为 671.79 ms，含客户端预处理的总 TTFT 为 922.53 ms；请求准备
250.74 ms、HTTP 往返 12.688 秒、完整客户端诊断 12.941 秒，客户端 TPOT 估计
126.44 ms。prompt/decode 计数及阶段时间仍为 null；不将 native 诊断的 94 次
decode 或 CUDA event 时间复制到该请求。单图不用于 p90/p99 或稳定性结论。

原始证据见本地 `reports/jetson-readiness-20261009/http_diagnostic.json`，包括
完整 SSE、metadata、输入/包身份、客户端源码、日志与 tegrastats。测试结束后
临时端口连接返回 111（无监听），原 binding 哈希保持不变。

## HTTP 阶段回传与准入验收

`--serial-stage-metrics` 默认关闭，要求显式候选绑定及 batch=1，并拒绝 draft 或
native batch scheduler。ASGI 锁覆盖推理响应与流式清理；runtime proxy 跟踪
`handle_request` 的执行区间与流式 generator 的准备/清理区间，snapshot 仅在这些
区间全部结束时读取。初始化不采集
profiling，服务开始前开启，入口结束时关闭。断流后 worker 未停则封闭后续推理，
不把仍在变动的 Timer 用于下一条记录。单请求窗口同时核对 native 与 SSE 计数。

2026-10-09 新隔离服务完成 HTTP 阶段单图验收，输入/输出/decode 为 570/95/94，
视觉编码 166.30 ms、prefill 426.48 ms、decode 12020.47 ms，HTTP TTFT 629.84 ms。
随后在另一新服务同时提交 PS20 indoor 001/114，提交时间相差 0.106 ms；114 先执行，
输出/decode 为 75/74，001 为 95/94。两个前后快照窗口首尾完全相接，均各含一次
prefill/generation；001 的 HTTP TTFT 为 10681.44 ms，包含前一请求排队时间。
这验证串行归因，不是原生并发吞吐或稳定性研究。两次服务均已关闭。

完整结果、SSE 快照、metadata、代码身份、日志和资源采样分别见
`reports/jetson-readiness-20261009/http_stage_diagnostic.json` 与
`http_stage_concurrency_diagnostic.json`；板端目录分别为
`artifacts/runtime/http-stage-metrics-20261009/` 与
`artifacts/runtime/http-stage-concurrency-20261009/`。下一步完成冻结 PS20 study，
另行验证未插桩基线，并定位 FP16 初始化内存峰值。

进一步增加 generator 准备/清理状态守护后，在新目录
`artifacts/runtime/http-stage-guard-20261009/` 重跑相同双请求验收；两个计数窗口
再次通过完整性、非交叠和首尾连续核验。最新代码身份与全部原始证据见
`reports/jetson-readiness-20261009/http_stage_guard_diagnostic.json`。取消 HTTP 响应
且 native worker 仍在执行的场景先由无硬件测试验证后续请求返回 503；
2026-10-11 的真实 GPU 断流结果见下节。

## 冻结 PS20 完整验收

2026-10-09 使用新 study 配置
`configs/studies/jetson/jetson_edgellm_int4_serial_stages_ps20_pilot.json`，标准 StudyRunner
完成 20/20 后端执行和严格 JSON 校验，失败汇总为空。三个阶段各 20 条，native
计数、SSE usage、报告字段和连续快照窗口逐条核对通过；输入/输出/decode 总数为
11400/1550/1530。HTTP TTFT p50 567.71 ms、E2E p50/p90/p99
10.489/11.343/12.769 秒，墙钟输出吞吐 7.335 tokens/s。完整口径和质量限制见
[性能记录](runtime-performance.md)。

首次尝试将 raw 图片符号链接到原 data 根目录，被 `ParkingCase.resolve_image`
拒绝，留下 20 条 input_error、零模型请求的报告。随后在新目录复制并核验 20 张
冻结 JPEG，保留源图和路径校验；修正后的 study 才执行真实推理。前后图片哈希与
原 engine 元数据均一致，临时服务已退出且 18080 无监听。

成功运行目录为 `artifacts/runtime/ps20-stage-study-20261009-retry1/`，报告位于
其 `package/reports/jetson_edgellm_int4_serial_stages_ps20_pilot.json`；逐请求原始
事件和 `measurement_validation.json` 在运行目录根部。本地完整证据为
`reports/jetson-readiness-20261009/ps20_stage_study.json`，SHA-256：
`3373649b91764b9c1542d02f1efca3e4c80cb23b7e27a976f43796aec52058e2`。
失败尝试单独保存为 `ps20_input_failure.json`。模型 revision 仍是配置声明，
engine 与历史导出记录的绑定证据尚缺；不能将本轮称为完整精度等价性验收。

FP16 内存排查已核验当前源码：TensorRT <10.7 的分支使用 mmap 整体 buffer
反序列化；已安装 10.3 头文件同时提供 `deserializeCudaEngine(IStreamReader&)`，
并注明 weight streaming 下可降低 host 内存使用。源码与头文件证据分别为
`ps20_progress_memory_source.json` 和 `tensorrt_legacy_reader_headers.json`。
该读取方式已完成隔离构建与单图验证，结果见下节。

## FP16 读取器实验

2026-10-09 使用 `prepare_edgellm_legacy_reader.py` 为已核验的 `trtUtils.cpp`
生成 SHA-256 绑定补丁。候选增加 `EDGELLM_LEGACY_STREAM_READER=1`，默认保留
原 mmap 分支；实验仅覆盖已安装的 TensorRT 10.3。读取器直接写入 TensorRT
提供的 host buffer，按 4 MiB 分段读取，对读过的文件页调用 advisory DONTNEED。
这不修改 engine 文件，也不清理全局系统缓存。

真实源码混用 LF/CRLF，补丁保留未修改文本和各段换行。链接命令对 Core 库
使用 whole-archive，因此 `build_edgellm_legacy_reader.py` 复制静态库并仅替换
副本中的 `trtUtils.cpp.o`，复用已验证的 timing binding 对象。构建记录核验
archive 成员顺序、替换对象内容、链接映射及原始输入哈希；原 checkout、库和
绑定未改动。脚本默认只生成计划，执行必须显式传入 `--execute`。

候选 binding SHA-256 为
`da86534af3e69d9cd83263af32be93899b02578565998e505d23e51bdea65694`。
板端目录为 `artifacts/runtime/legacy-reader-20261009-retry1/`。以相同 FP16
engine、冻结 indoor 001、15W 模式、weight streaming budget 0 做单图诊断；
LLM 和视觉加载日志均确认启用了候选读取器。视觉 engine 仍在申请
811032832 bytes 时 OOM，失败阶段为 `runtime_initialization`，零生成 token，
因此没有 TTFT、decode 或吞吐结果，不能宣称优化成功或精度等价。

日志记录 weight streaming scratch 1244660224 bytes、预分配 base execution
context 1347639296 bytes。250 ms 进程采样共 210 条，采样中最大 VmRSS
6500680 KiB、最大 VmHWM 6512536 KiB；这些不是逐请求 GPU 内存指标。
下一步核对 context、权重和视觉资源的同时驻留及分配顺序；不能据本轮推断
更换读取方式降低了多少峰值。本轮未开展配对的 mmap 内存采样比较。

完整构建证据为 `reports/jetson-readiness-20261009/legacy_reader_build_record.json`，
单图原始证据为 `legacy_reader_fp16_probe.json`，后者 SHA-256 为
`b7a7701857744c623299e18f919e86a949257d27d529414b8c5c90a1e491f874`。
诊断及资源采样进程均已退出；原 engine 元数据核验未变。以上文件位于被 Git
忽略的 reports 目录，需与源码提交分别归档。

## Context 分配顺序与预算研究

`prepare_edgellm_context_allocation.py` 为当前 `llmInferenceRuntime.cpp` 生成
源码哈希绑定补丁。`EDGELLM_DEFER_CONTEXT_ALLOCATION=1` 将共享 context 的分配
延后到可选视觉 runner 加载之后；未设置或设为 0 时保留原分配顺序。候选拒绝
draft 模型，未验证 speculative decoding。后置分配仍使用各 runner 需求的
最大值，不能缩小 buffer 来规避 OOM。

Core overlay 构建脚本可对已核验的候选 archive 再覆盖单个 runtime 对象，
参数 `--core-archive` 与 `--core-archive-sha256` 必须同时提供。原库、候选输入
和 checkout 均只读，输出必须位于新目录。本轮 binding SHA-256 为
`2859d58b4518b4b7d050c7eb901f50ddabb39eeca5b7790124f3c6345d66c7a7`。
两个单图实验使用相同 FP16 engine、workload、indoor 001 和 15W 模式：

| Context 分配 | Weight budget | 实测结果 |
| --- | --- | --- |
| 延后 | 0 | 视觉 runner 初始化成功；申请 1347639296 bytes 共享 context 时 CUDA OOM |
| 延后 | 640 MiB | scratch 未减少；视觉反序列化再次在 811032832 bytes 分配处 OOM |

两次均失败于 `runtime_initialization`，零生成 token，无 TTFT 和吞吐结果；
原 engine 元数据未变，诊断和采样进程已退出。分配顺序改变了失败位置，
尚未证明完整模型可驻留或生成，因此不启用候选作为默认。

已安装 10.3 头文件说明 scratch 可通过 `getDeviceMemorySizeV2()` 与
`setDeviceMemoryV2()` 提供；当前 executor 已使用这对接口。scratch 是全局
context 需求的一部分，不能重复相加；通用接口说明见 NVIDIA
[weight streaming 文档](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/weight-streaming.html)。

使用系统现有 TensorRT Python 10.3 独立查询同一 LLM engine，未创建用户执行
context 或运行模型。0/640/1280/1920/2560/3200 MiB 六个预算均核验
requested=actual，scratch 均为 1244660224 bytes，全局 context 均为
1347639296 bytes（约 1.26 GiB）。本版本两个 per-profile API 值为
102979072/78145536 bytes，未覆盖该 scratch，不能替代全局 buffer 需求。
预算 setter 本身可能分配 GPU 权重；查询不是完整模型 fit 或性能证据。

首轮查询尝试全量预算 3441150208 bytes 时 TensorRT 报 OOM，Python setter
却未抛异常，actual 仍是此前的 3355443200。首轮输出的 `queried` 不代表最后
预算生效。`inspect_edgellm_weight_streaming.py` 已增加 `budget_applied`、
requested/actual 核验和失败前的部分结果保存；存在拒绝时输出
`queried_with_rejected_budgets` 并返回 2。修正后的六预算查询全部通过。

完整构建、两个单图结果及预算查询分别保存在
`reports/jetson-readiness-20261009/context_build_record.json`、
`context_fp16_probe.json`、`context_fp16_budget640_probe.json` 和
`weight_budget_query_verified.json`；首轮原始拒绝证据单独保存在
`weight_budget_query_initial.json`。板端目录为
`artifacts/runtime/deferred-context-20261009/` 和
`artifacts/runtime/weight-budget-query-20261009-verified/`。
下一步核对既有 FP16 ONNX 与构建身份，研究非流式 FP16 候选；当前六个
流式预算没有提供减少 scratch 的证据，继续扩大预算不作为默认优化方向。

1. 补齐完整 tokenizer 目录、源码补丁与二进制构建身份，核对 TensorRT 内部视觉
   预处理尺寸和布局，并将新增 Adapter 放入独立板端 checkout 进行验收。
2. 核对构建日志、命令和源码补丁。旧 flow record 仅记录输出存在和成功状态，
   未绑定 engine SHA-256，不能证明当前文件就是当时产物。
3. 选定已有产物后，以新输出目录执行单图验收，再开展 Jetson Transformers FP16
   与 Edge-LLM 同机研究。TTFT、decode、端到端时延和失败记录采用
   [统一指标口径](runtime-performance.md)。

只读证据保存在本地 `reports/jetson-readiness-20261008/environment.json`
和 `fp16_snapshot.json`，后续文件身份与跨主机比较见 `file_identity.json`、
`parity_audit.json`、`model_environment.json` 和 `model_parity.json`。
`reports/` 被 Git 忽略，原始证据需单独归档。

## 真实断流与贪心参数复验（2026-10-11）

在隔离 INT4 服务收到首个可见 token 后关闭 TCP 连接；取消发生在请求开始后
664.38 ms。随后请求返回 503，等待 35 秒后再次请求仍为 503，明确要求重启
测量服务。这验证了该次 native worker 尚未停止时的持续拒绝行为，避免串行
阶段计数混入后续请求；不能推广为所有取消时机均已覆盖。取消请求不计为有效
推理记录。证据：`reports/jetson-readiness-20261011/http_cancel_evidence.json`，
SHA-256 `a04f37e2bfab829862106dd314b19cba00271f4d7d2d478e0e7247b5bfc7443f`。

原 HTTP 请求只指定 temperature=0，native 每个生成步骤都会将默认 top_k/top_p
归一化并输出告警。Adapter 与 native probe 现在显式发送 temperature=0、
top_k=1、top_p=1.0；do_sample=true 保留原采样行为。相同单图 A/B/A 三次请求
各输出 95 tokens，原始文本相同，阶段计数窗口连续；显式请求告警为零，隐式
请求各 95 条。此实验支持参数行为一致与日志消除，不构成稳定加速结论。
完整证据：`reports/jetson-readiness-20261011/http_greedy_evidence.json`，
SHA-256 `6aad60efc362bf7c90e577b84577771a2676b92bc9202b856f11528c00ad41af`。

## FP16 非流式构建的系统内存限制

2026-10-11 复用现有 FP16 ONNX，在新目录
`artifacts/runtime/fp16-nonstream-build-20261011/` 关闭 weight streaming，设置
batch=1、input=768、KV=1024、workspace=512 MiB、builder optimization level=0。
构建约 56 秒后 SIGKILL，未生成 engine，输入文件哈希未变。内核日志明确记录
PID 216353（llm_build）被 global OOM killer 杀死；进程采样最高 VmRSS
5812328 KiB、最低 MemAvailable 21180 KiB，SwapFree 降为零。
这是构建阶段的系统 OOM，不是候选 engine 的运行时适配或显存结论。

原始证据保存在 `reports/jetson-readiness-20261011/` 的
`fp16_nonstream_build_evidence.json`、`fp16_nonstream_kernel_audit.json` 与
`fp16_nonstream_failure_summary.json`。构建脚本分别记录进程信号、非零退出
和启动失败；仅凭 SIGKILL 不自动宣称 OOM，需另附内核证据。

已准备 mode=0600 的 8 GiB 临时构建 swap 文件，未写入 fstab；准备时未启用。
`sudo -n` 需要用户密码，因此启用步骤由用户在 Jetson Bash 执行：

```bash
sudo swapon -p 0 /home/ubuntu/JetsonVLM/artifacts/runtime/fp16-nonstream-build-20261011/swap/build.swap
cat /proc/swaps
```

启用后再核验状态并使用新目录构建；不能将交换空间增加等同于 CUDA 可用内存
增加。运行时评测需记录实际 swap 状态；构建完成后使用同一路径执行
`sudo swapoff`，确认不再活跃后再删除临时文件。准备证据为
`reports/jetson-readiness-20261011/build_swap_preparation.json`。

## 显式贪心参数的 PS20 验收（2026-10-12）

新配置 `configs/studies/jetson/jetson_edgellm_int4_explicit_greedy_ps20_pilot.json`
在隔离目录 `artifacts/runtime/ps20-explicit-greedy-20261012/` 完成 20/20 执行
与严格 JSON 校验，三个阶段各 20 条，失败为空。阶段窗口连续，native 计数、
SSE usage 和报告一致；输入/输出/decode 总数仍为 11400/1550/1530。全服务日志
的贪心归一化告警为零。逐样本与 2026-10-09 PS20 比较，20 条原始文本、
assessment、token 数和 workload identity 全部一致。

HTTP TTFT p50 567.40 ms，含预处理的客户端 TTFT p50 769.96 ms；E2E
p50/p90/p99 为 10.488/11.340/12.750 秒，墙钟输出吞吐 7.338 tokens/s，
严格 decode 7.825 tokens/s。跨天结果接近，不据此宣称性能提升。风险等级准确率
仍为 0.35、事件 micro-F1 为 0、语义一致率为 0.10；框架有效不代表业务效果合格。

本轮仍为串行、已加载服务、native profiling 与 SSE 持久化开启、无独立预热的
pilot；不是无插桩基线或 native 并行吞吐。前后只有六个 zram swap，临时磁盘
swap 未启用；测量服务已退出，端口 18080 无监听，图片与原 engine 元数据未变。
模型 revision 仍缺历史导出绑定。完整证据为
`reports/jetson-readiness-20261012/ps20_greedy_study.json`，SHA-256
`bdb3cdf0d8adbfa964c11fb2666c80ecc8f1957fdc3afc9fa7b396e879c0c866`；
同目录 `ps20_greedy_comparison.json` 保存逐样本比较和全部指标。

## 阶段 profiling 开启 / 关闭 / 开启复验

2026-10-12 在同一 Jetson 15W mode 0 依次完成三个独立服务的冻结 PS20
study，每档 20/20 后端执行与严格 JSON 校验，告警和失败均为零。三档 engine、
native binding、图片哈希与功耗模式一致；关闭档和开启复验各 20 条原始输出、
assessment、输出 token 数和 workload identity 均与开启前测一致。

| 配置 | HTTP TTFT p50（ms） | E2E p50（s） | 墙钟输出吞吐（tokens/s） |
| --- | --- | --- | --- |
| profiling 开启前测 | 567.40 | 10.488 | 7.3378 |
| profiling 关闭 | 567.23 | 10.488 | 7.3389 |
| profiling 开启复验 | 565.96 | 10.486 | 7.3361 |

关闭档使用 `configs/studies/jetson/jetson_edgellm_int4_profiling_off_ps20_pilot.json`。
隔离 launcher 显式关闭 profiling，逐请求检查实际状态与 timer 快照；20 条
均为关闭且计时器未增长。SSE 输出计数、事件持久化与状态审计仍存在，因此这是
关闭 CUDA 阶段计时的参考，不能称为完全无插桩基线。input/decode 计数和阶段
耗时保持未知；不能从 output tokens 自动推算并补填。

关闭档旧验收 scope 字符串误沿用 profiling-enabled 模板，原始文件保留。
派生摘要明确纠正该描述，实际状态由逐请求审计裁定；旧汇总的未知 token 总数
另依据原始记录纠正为 null。修正不改变时延、吞吐或原始 GPU 证据。

三档完整服务期设备 RAM 采样峰值为 5154/5106/5151 MB，zram 用量峰值
1638/1639/1640 MB，VDD_IN 均值为 9.534/9.510/9.555 W，GPU 温度峰值为
51.906/51.812/52.500°C。采样含初始化，均是设备级指标，不能当作单请求
GPU 内存或能耗；前后都没有启用磁盘 swap。性能差异很小，本轮不支持稳定加速
结论，也不把 48 MB 设备 RAM 差异归因于 profiling 本身。

开启复验记录新服务进程启动到 HTTP health 成功为 10077.95 ms；250 ms
健康检查轮询，排除包准备与 engine 哈希时间。该值受现有文件缓存影响，不是
系统冷启动或单独的 engine 加载时间；StudyReport 的 cold_start_ms 仍为空。

板端目录分别为 `artifacts/runtime/ps20-explicit-greedy-20261012/`、
`ps20-profiling-off-20261012/`、`ps20-profiling-on-recheck-20261012/`。
三档服务均已退出，18080 无监听，原图片与 engine 元数据未变。
本地 `reports/jetson-readiness-20261012/` 保存全部原始 SSE、日志、launcher、
代码哈希、报告和 `profiling_comparison.json`；新增原始文件 SHA-256：

- `ps20_profiling_off_study.json`：
  `cac74e1b06a7b7be148845da4243da3397a9a058600fdeaf276ff03e30f6cf49`。
- `ps20_profiling_on_recheck_study.json`：
  `88feb993fc19081f1af9be0d68346867d295bb9eefe2fc3ae1f0aff17ab42471`。

风险等级准确率仍为 0.35、事件 micro-F1 为 0；本轮只关闭框架测量差异的
验证缺口，业务效果、FP16 内存限制及历史导出 revision 绑定仍需继续处理。

## Decode CUDA Graph 捕获对照（2026-10-12）

当前 Python server 在加载 runtime 后默认调用 Graph 捕获，原源码没有
`EDGELLM_DISABLE_CUDA_GRAPH` 分支。`prepare_edgellm_cuda_graph.py` 以源码
SHA-256 绑定生成 Python 补丁：只有环境变量为 1 时跳过捕获，默认行为不变。
实验将 10 个 experimental Python 文件复制到独立目录，仅修改副本 engine.py，
并核验进程实际加载路径及哈希。native binding、Core、插件与 engine 均未重建；
原 checkout 的 Python 文件和 engine 元数据前后哈希未变。

固定原 engine.py SHA-256 为
`5a7d77fbc97e1a18c8dd44b7c922bfbf300e708e2cb957f0471ee50585d1fc43`，
候选为 `a1bcc2296e51ed8ccade20d16698ff66190c0833fc5c61ddebb34d04ab2e00a7`。
关闭与重开使用同一候选源码、同一 6dae timing binding；功耗模式仍为 15W。
关闭档 metadata 确认环境开关为 1，原生捕获日志为零；重开档环境开关移除，
捕获日志为一条。Python INFO 关闭提示被当前 logger 过滤，不能要求它已出现。
状态判断结合实际加载源码、运行环境和原生日志；捕获证据尚不是 replay 的
直接 CUDA trace。

| Graph 捕获 | HTTP TTFT p50（ms） | decode p50（s） | E2E p50（s） | 墙钟输出吞吐（tokens/s） |
| --- | --- | --- | --- | --- |
| 开启前测 | 565.96 | 9.712 | 10.486 | 7.3361 |
| 关闭 | 567.70 | 9.877 | 10.658 | 7.2096 |
| 开启复验 | 566.53 | 9.707 | 10.483 | 7.3383 |

三档各 20/20 执行与严格 JSON 校验通过，三阶段各 20 条且计数窗口连续。
关闭和重开各 20 条原始输出、assessment、input/output/decode 数均与前测一致，
总数为 11400/1550/1530。严格 decode 吞吐分别为 7.823/7.678/7.826 tokens/s。
本轮开启组墙钟吞吐比关闭组约高 1.8%，保留默认启用；20 张 pilot、无独立预热、
native profiling 与 SSE 持久化开启、未固定 GPU 时钟和温度，不推广为其他
工作负载的稳定加速幅度。质量指标仍为风险等级准确率 0.35、事件 micro-F1 0。

板端关闭与重开目录为 `artifacts/runtime/ps20-graph-disabled-20261012/` 和
`ps20-graph-enabled-recheck-20261012/`，两服务已退出且 18080 无监听。完整
本地证据在 `reports/jetson-readiness-20261012/`：

- `ps20_graph_disabled_study.json`：SHA-256
  `825a59de23d4ae9f0be975c779e6cd8019c023618f79669ad1b0770ae0595516`。
- `ps20_graph_enabled_recheck_study.json`：SHA-256
  `fa43c4f109c65a8c2f3ce00d6bf6a18957c9c0b9f2add3b159bb385747ed10db`。
- `graph_python_source_audit.json` 保存全部原始/副本 Python 内容与哈希，SHA-256
  `754194585e48ec4850717b1beab5c867a4419e738719a32dffb5b23c751fc4f3`。
- `graph_comparison.json` 保存逐样本比较、性能、启动时间及设备遥测。

下一步使用已安装的 Nsight Systems 2024.5.4.34 追踪实际 Graph replay、CUDA
同步与 decode kernel。尚未执行该 trace；不能仅凭总耗时指定具体 kernel 瓶颈。

## 单图 Nsight generation trace（2026-10-12）

在新目录 `artifacts/runtime/nsight-native-generation-20261012/` 使用已安装
Nsight Systems 2024.5.4.34，CUDA node trace、无 CPU sampling/context switch。
`run_edgellm_native_probe.py --cuda-profiler-range` 在生成前后显式调用
cudaProfilerStart/Stop，并同步 GPU 边界；初始化、Graph 捕获与客户端图片
resize 不在 capture range 内。默认不启用该选项。日志确认 capture range
开始和结束，采集与 CSV/SQLite 导出均退出 0。

固定 `ps20-indoor-001`，native 执行一次 prefill 和一次 generation，输入
570、输出 95、实际 decode stage 94 次；严格 JSON 有效，与此前同图 PS20
原始输出和输出计数一致。GPU event 共有 44160 条。SQLite 中 94 次成功的
cudaGraphLaunch 与 94 个 decode 步骤一致，通过 correlationId 关联到
42488 条 graph-node kernel 事件，每次 replay 452 条；这是实际 replay 的
直接证据，且 95 次主 GEMM 中 94 次属于这些 Graph launch。

| GPU kernel | 次数 | 总 GPU event 时长 | 平均时长 | kernel 总时长占比 |
| --- | --- | --- | --- | --- |
| sm80 FP16 GEMM，tile256x64x32、stage3、aligna2/alignc2 | 95 | 10502.18 ms | 110.55 ms | 83.55% |
| INT4 WoQ GEMV，NPerBlock=2、block=256 | 18424 | 1332.89 ms | 0.0723 ms | 10.60% |
| INT4 W4A16 GEMM T2 | 196 | 264.06 ms | 1.3473 ms | 2.10% |

全部 kernel event 时长之和为 12569.47 ms，transfer 为 1.69 ms；这不是
wall time，事件可能重叠。95 次 cudaStreamSynchronize 的 host 等待合计
12504.32 ms，与 GPU 执行重叠，不能再相加或作为独立瓶颈。首要目标是主
FP16 GEMM，当前证据不支持优先优化仅 3.76 ms 的 sampling kernel 总时长。

同机 engine inspector 成功读取 2144 个 layer 名，但原 engine 只保存
LAYER_NAMES_ONLY，没有完整 shape/tactic 或层到 kernel 的绑定。主 GEMM 的
grid 为 (1,2374,16)，与大输出投影的假设相符；尚不能将具体模型算子或 LM
head 身份写成已证明事实。本次 trace 的范围止于 kernel；后续 ONNX 图检查与
现有 level1 候选的完整工作负载复验见下节。项目 Python 环境仍无需 ONNX 依赖。

本轮是单图、有 native timer 和 Nsight 开销的诊断，native wall 约 15.805 s
包含 profiler 开始/停止开销，不更新常规吞吐基线。未改权重、输入、功耗模式
或原 engine 元数据；采集进程已退出，临时磁盘 swap 仍未启用。

完整本地证据位于 `reports/jetson-readiness-20261012/`：

- `nsight_native_evidence.json` 保存原始报告、SQLite、四份 CSV、native probe、
  执行命令与环境，SHA-256
  `966a996ff5899c737db7ad93bff27a8256f77724e074da792e6092054e10022e`。
- `nsight-native-artifacts/generation.nsys-rep`：SHA-256
  `590d10febd6ebf70971fffffb10de2856e53008f9abf7ad8674cf6801bb59ab0`。
- `nsight_native_kernel_summary_v2.json` 与 `nsight_native_analysis.json`
  保存分类、API/kernel 关联与验证边界；`analyze_native_nsight.py` 保留查询。
- `engine_inspector_evidence.json` 保存完整层名与 inspector 执行源码，SHA-256
  `2dac4d5905e1481faab873758ce928c70d7a442ce52797225b560d86eb7449c0`。

## Builder level1 候选全流程复验（2026-10-12）

在同一 Jetson、15W mode 0、冻结 PS20、448×448、显式贪心和 batch=1 下，
依次运行 level0 / level1 / level0 回测 / level1 复验，各自启动独立服务。
全部保持 Graph、native stage profiling 与 SSE 保存，无独立预热。

| 引擎与次序 | HTTP TTFT p50 | 端到端 p50 | 墙钟输出 tokens/s |
| --- | ---: | ---: | ---: |
| level0 前测 | 566.53 ms | 10.483 s | 7.3383 |
| level1 首测 | 465.65 ms | 2.784 s | 27.5918 |
| level0 回测 | 566.91 ms | 10.486 s | 7.3376 |
| level1 复验 | 468.49 ms | 2.787 s | 27.5744 |

level1 两次吞吐约为 level0 前测的 3.76 倍；四组各 20 条均完成且严格 JSON
有效，逐样本 raw output、assessment、workload identity 和三类 token 计数
完全一致。各组总输入/输出/decode 为 11400/1550/1530；阶段计数连续。
风险准确率仍为 0.35、事件 micro-F1 为 0，性能改善不代表领域效果改善。
全部服务已退出，端口关闭，原图片和元数据保持不变。

level0 LLM engine SHA-256 为
`33466f3f1149801bf496737fbe1e69634b8018ece9a0c5bccd8ff83af64afab4`，
现有 level1 候选为
`207e109fcca28ac29ae0348d8dd517e61a6291704a8fea08c53644574f28d345`。
两者配置均为 i768/k1024、全词表 151936、INT4 backbone；五份 LLM 元数据
及 embedding 内容相同，复用同一 FP16 visual engine 和当前 native binding。
候选通过独立目录组合与安全元数据副本加载，未改默认 engine。
历史 level1 构建记录缺少导出图/权重文件哈希，因而尚不能证明完整权重来源
等价；当前 pilot 输出一致也不能代替该来源证据或更大规模验收。

只读 ONNX 检查确认 `logits <- Cast <- node_linear(MatMul)`，匿名权重
`val_1246` 为 FLOAT16 `[2048,151936]`。图 protobuf SHA-256 为
`78c7badb4cfa1f81dd71a66742b811c89131e232c7af6522fd73956ea59e9603`。
检查仅解析图，不读外部权重；ONNX 1.19.0 安装于 `/tmp` 独立环境，未改项目
或 Jetson 依赖。历史 detailed engine 的输出投影 tactic 差异支持该优化方向，
但当前 kernel 到 layer 的直接绑定仍待 level1 trace 核验。

本地证据目录 `reports/jetson-readiness-20261012/`：

| 完整证据 | SHA-256 |
| --- | --- |
| `ps20_level1_study.json` | `1cf1456c527ffd6d76d6f5f2b8fdedde32ab9942bda6f23d8745bf3c5525aeca` |
| `ps20_level0_return_final_fetch.json` | `63aa70a2c0352a4ca846700e4cdbc643a65aac00f7cd736f90ab3621563a4ad4` |
| `ps20_level1_recheck_study.json` | `de571ae81f5b708afcd7162d9c698c4b829424ba165cab83211baeaa5f931410` |
| `onnx_output_projection.json` | `6ba6b65e2ce18ca0621977e365853a306235cfdae913fbdec4f57a24bd5592f3` |

`compare_builder_levels.py` 保留逐样本比较、engine 身份与退出检查，生成
`builder_level_comparison.json`。新增两个 study 配置区分候选与回测；engine
选择仍由服务启动参数确定，配置本身不切换 engine。

## Level1 Nsight 同图归因复验（2026-10-12）

独立目录 `artifacts/runtime/nsight-level1-generation-20261012/` 固定 level1
engine、原 visual engine、当前 plugin/native binding、冻结图片与贪心参数。
开始前核对各二进制哈希、15W mode 0 与无活动推理/构建进程；仅在生成范围
采集 CUDA node trace。采集和 CSV/SQLite 导出均退出 0，进程已退出。

同图输入 570、输出 95、decode stage 94；严格 JSON、原始输出和 assessment
与两组 PS20 报告一致。SQLite 中 94 次 Graph launch 全部成功，按进程命名空间
和 correlationId 关联到 42488 条非零 graphNodeId kernel 事件，每次 452 条。
原主 FP16 GEMM 在 level1 中换为 `trt_ampere_h16816gemm_128x64_ldg8_tn_v1`，
共 95 次，其中 94 次关联 Graph replay。

| Kernel | level0 总 event 时长 | level1 总 event 时长 | level1 平均 | level1 kernel 时长占比 |
| --- | ---: | ---: | ---: | ---: |
| 原主 FP16 GEMM / 新 Ampere half GEMM，95 次 | 10502.18 ms | 983.75 ms | 10.36 ms | 32.28% |
| INT4 WoQ GEMV，18424 次 | 1332.89 ms | 1334.64 ms | 0.0724 ms | 43.79% |
| INT4 W4A16 GEMM T2，196 次 | 264.06 ms | 264.07 ms | 1.3473 ms | 8.66% |

主 GEMM 平均从 110.55 ms 降至 10.36 ms，约 10.68 倍；INT4 GEMV 基本不变，
现在是最大的单一 kernel family。全部 kernel event 时长从 12569.47 ms 降至
3047.81 ms，这个比值不是端到端加速比。当前 native stream 约 6.108 s，
包含 profiler 开始/停止开销，不更新常规 PS20 基线。精确模型 layer 到该
kernel 的直接绑定及历史权重来源仍未证明，不能直接套用旧 detailed engine
中的 nn tactic 名称解释当前 tn kernel。

Nsight 汇总分类升级为 `parksight_cuda_kernel_names_v3`，补充当前实测 Ampere
half GEMM 和 fused FP16 GEMM 符号。两份原始 CSV 均用 v3 重算后比较；v2
归档保留。`analyze_edgellm_nsight_replay.py` 提供只读 SQLite 关联与 native
计数差值核验，不把同步 host 等待与 GPU 时长相加，不固定每次 replay 的
kernel 数量；缺失事件、失败 launch 或歧义 correlation 不通过核验。

本地完整证据 `reports/jetson-readiness-20261012/nsight_level1_evidence.json`
SHA-256 为 `0fadfa436a0ab7fd13d700fa59765645996ec42d55975e7a58f853f114eae613`，
其中原始 `generation.nsys-rep` 为
`549464dfd532cec1aaa73bc63fc3f93fb2c765c500638daea6e1cf8206357f79`。
解码文件位于 `nsight-level1-artifacts/`，每份按归档哈希校验；重算摘要和
关联结果为 `nsight_level1_kernel_summary_v3.json`、
`nsight_level0_level1_comparison_v3.json`、`nsight_level1_replay_analysis.json`。
下一步以当前 level1 为性能候选参考，评估 INT4 GEMV 访存/融合方向，并补齐
可复现的导出与构建来源，保持完整词表、冻结工作负载和输出验收。

## QKV 与 gate/up 融合候选（2026-10-12）

只读提取并核对现有导出模型的 196 份 pre-quant scale。28 层的 Q/K/V
逐字节相同，gate/up 也逐字节相同；这 140 份 scale 均为 FP16 全 1 向量。
其余 O/down scale 不属于本候选。原外部权重文件 1350303744 字节，SHA-256
`235606ea014b1377560e6cc8ea22792a7a65c0289f5edfdf9d546e4911720724`。

`prepare_edgellm_int4_fusion.py` 根据已核验的 plugin 源码布局生成独立图：
packed weight 按 N-major/interleave4 连接，group scale 按行连接，输出按
最后一维 Slice 恢复原名称。要求相同 hidden input、相同 scale 字节、group128
与 N/K 对齐；拒绝不完整分组、布局来源变化和已有输出目录。56 组融合使
INT4 plugin 196→112，图节点 1730→1702，依赖顺序和 graph outputs 保留。

`materialize_edgellm_int4_fusion.py` 校验原数据和候选图 SHA-256，以 1 MiB
以内的复制块写入临时目录，完成后发布新目录；原数据仅以只读访问及 symlink
复用。新增重排数据 484442112 字节，独立再次逐段比较全部 weight blocks 和
scale rows，均逐字节一致。candidate graph SHA-256 为
`8e65b9669790ce77b93ffeade998331d6e452f26c473f4951749475d75158c61`，
fusion data 为 `91165e177941a903b8f50b307b1d2f11e78f8a6960cac78f0597fe65dd807f6f`。
参数字节验证不能替代 GPU 执行验证。

初次构建前元数据检查发现 source config 与 engine config 的差别仅为新增
`builder_config`；embedding、tokenizer 和模板均与 control 一致。使用原导出
config 后，level1/workspace1024、i768/k1024、无 weight streaming 构建成功，
耗时 134.34 s。当时 wrapper 只传 `EDGELLM_PROFILING_VERBOSITY`，原 builder
不识别此环境键；后续 inspector 实测仍为 LAYER_NAMES_ONLY。输出 engine 1361856164 字节，SHA-256
`14f4cd6638c621caa2515daf13a36b1483747ce69068172dcef6d5c972d5aacb`。
源 config SHA-256 为 `4b56a8b3b8654cf199e52871ca68cf715bad0b5172bbe347331ca3f91830d179`；
构建后 engine config 与原 level1 完全一致。历史导出到 HF revision 的绑定仍未证明。

单图检查严格 JSON 有效，570/95/94 输入/输出/decode 与 control 相同，原始
输出和 assessment 相同。随后冻结 PS20 20/20 完成，所有输出、assessment、
workload identity 与三类 token 计数均匹配，质量指标不变。首测及原 level1
回测分别启动独立服务，保留相同 visual engine、native binding、15W mode 0、
Graph 与阶段计时；回测与此前首测跨运行时段，未固定 GPU 时钟。

| 运行 | 墙钟输出 tokens/s | 端到端 p50 | HTTP TTFT p50 | decode p50 |
| --- | ---: | ---: | ---: | ---: |
| 原 level1 既有复验 | 27.5744 | 2.787 s | 468.49 ms | 2.114 s |
| 完整融合首测 | 26.8691 | 2.854 s | 459.89 ms | 2.189 s |
| 原 level1 回测 | 27.5812 | 2.786 s | 465.91 ms | 2.113 s |

融合首测比回测吞吐低 2.58%，当前 engine **不晋级**。两个原 level1 结果
接近，不能把 plugin 数量减少当作全流程加速。全部服务退出、端口关闭，
原图片和元数据未改；冷启动及逐请求内存/功耗仍未由此证明。

同图 Nsight 95 次 forward、94 次成功 Graph replay，每次 kernel 从 452 降到
368，关联事件 42488→34592。INT4 GEMV 18424→10528 次，总 event 时长
1334.64→1273.03 ms，减少 4.62%；prefill INT4 GEMM 为 112 次、261.39 ms。
但原主 `trt_ampere_h16816gemm_128x64_ldg8_tn_v1` 被此构建的
`sm50_xmma_cublas_gemvx_f16f16_f32_f32_nn_n_int32_unit_n_launch_param64x2x1_strided_copyx_unit_stride_execute_kernel_trt`
替换，同为 95 次，平均 10.36→12.03 ms，总时长增加 158.79 ms，超过 GEMV
节省的 61.61 ms。完整 kernel event 时长 3047.81→3144.59 ms；这不是 wall time。
该次 engine 缺少 DETAILED inspector，故无法由它直接核对输出层 tactic。

更充分的 level2/workspace1024 搜索在 87.24 s 后退出 1：日志明确显示
622329856 字节分配的 CUDA error 2、tactic 被跳过，最终无可用实现；没有
生成 engine，也没有 SIGKILL。不能仅依据最终“无可用实现”判为算子不支持，
也不能将其写成内核 OOM killer 结果。后续单独降低 workspace 至 512 MiB，
保持相同融合图、权重、profile 和精度。workspace512 重试在 80.65 s 后
失败：真实 builder 返回 -9/SIGKILL，外层 Python 返回 1，未产生 engine。
本次未提前记录 builder 子进程 PID，不能自动将该信号确认为 OOM killer。
原始证据 `int4_fusion_build_level2_w512_evidence.json` 的 SHA-256 为
`88effe322cb0749e95985ec2d5d08b3d286dbd76d5cc5d2f12a43eafa099be7e`。

完整本地证据位于 `reports/jetson-readiness-20261012/`：

| 证据 | SHA-256 |
| --- | --- |
| `prequant_scale_evidence.json` | `4da659d714aabd20effe0b7b770453c07c98633188a28822f0c588527dddf9b5` |
| `int4_fusion_materialization.json` | `9d5ef14ca5622889103862a2f8eb287b496a6b804cba573f6d017a8a485892d1` |
| `int4_fusion_build_evidence.json` | `cc091ce88f6251e77047a80760addc0124c93f6840c215d6e0319faec4a880dd` |
| `ps20_int4_fusion_study.json` | `d41a94f9588e23be3d3490f13e946b55d886968758499989afed34779d41d95b` |
| `ps20_level1_fusion_return_study.json` | `42d7cb7a19bbd0b4213987a49dd5453c140f4884f9aad1ce115401075385919f` |
| `nsight_int4_fusion_evidence.json` | `ad807b1800f609637f056fd7609d7e3b8e5b44087a77f14fd1875cb1c3b23469` |

`int4_fusion_level1_return_comparison_v2.json` 保存逐样本比较；
`nsight_int4_fusion_replay.json` 保存 Graph 关联。新 GEMVX 用
`parksight_cuda_kernel_names_v4` 单列 `fp16_gemv`，两份原始 CSV 均重算 v4，
旧 v3 摘要保留。level2 失败原始报告与日志为
`int4_fusion_build_level2_evidence.json`。

首次成功的 fusion engine inspector 为 TensorRT 10.3.0、1640 layers、
LAYER_NAMES_ONLY，不能读取完整 tactic/shape。SDK 的 LLM builder 虽无
`EDGELLM_PROFILING_VERBOSITY` 环境键，但原生支持 `--profilingDetailed`；
以相同候选图和 level1/workspace1024 隔离重建成功，engine SHA-256 为
`3baab9239bc0a46327dc94271108e4b85f21de5234624e75e45722c14a4132f9`。
实际 inspector 返回 `ProfilingVerbosity.DETAILED`、1640 层；输出层
`node_linear` 的两个 profile 均为 `ampere_h16816gemm_128x64_ldg8_tn_v1`。
一张冻结图片的原生推理为有效 assessment；`native_stream_total_ms=3066.03`，
这个单样本结果不用于判定 20 样本吞吐提升。

使用独立编译的 timing-cache builder 重建相同 fusion 图后，实际生成
687700 字节 cache（SHA-256
`690659c06f3424fd256947116337d17aa0bf3d6cf14bc67ba9842669863639c6`）；
engine SHA-256 为
`48003ff1fa342ff07a5361c5a049dd578a76bfc45d01b3d3e4ab1d580873709a`，
inspector 同样为 `DETAILED`，输出层 tactic 同为 Ampere GEMM。
这一 engine 在冻结 PS20 上两次得到 28.122、28.178 output tokens/s；夹在中间
的旧 level1 对照为 27.558 output tokens/s，后一次对照提升 2.25%。
decode p50 为 2064.17/2059.90 ms，对照为 2110.59 ms；20/20 JSON 有效，
全部逐样本输出与旧 level1 一致。
`ps20_int4_fusion_detailed_study.json`、`ps20_fusion_aba_studies.json` 和
`int4_fusion_aba_control_to_repeat_comparison_v2.json` 保存完整事实。
此增益属于新 engine 的同机实测；timing cache 只参与构建，不是推理期加速机制。
新 engine 的详细 inspector 将 `node_linear` 两个 profile 均标为 Ampere GEMM
tactic；同一张冻结图的 Nsight 确认该 kernel 出现 95 次、总 GPU event 时长
981.45 ms，而未开启详细信息的融合 engine 使用 FP16 GEMVX 95 次、1142.54 ms。
详细候选与原 level1 的总 kernel event 时长为 2976.30 对 3047.81 ms；
INT4 类合计为 1531.64 对 1598.72 ms。94 次成功 Graph replay 均关联
368 个 kernel，原始输出与两份旧 trace 完全相同。
`nsight_int4_fusion_detailed_evidence.json`、`nsight_int4_fusion_detailed_replay.json`、
`nsight_int4_fusion_detailed_vs_level1_v4.json` 保存原始轨迹、关联与分类对比。
GPU event 时长不能直接当作端到端时延。

为隔离 builder 重建的影响，又用相同独立 builder SHA-256
`56172aa9ba0ef1c746b45ff404f9f7c1bbaec92ecb68f979377e663639b488d7`、
相同 level1/workspace1024、`--profilingDetailed` 和全新 timing cache 构建
**未融合**原图。对照 engine SHA-256
`7daedae0a0b8a24eb68690f40b947527523d1cc1f8e4633b8b6ab2c8ba18705a`，
inspector 为 `DETAILED`、2144 层，输出层同样选 Ampere GEMM。
冻结 PS20 对照吞吐为 27.627 output tokens/s、decode p50 为 2109.46 ms；
融合 engine 两次相对它的吞吐提升分别为 1.79% 和 1.99%，decode p50
分别降低 2.15% 和 2.35%，20/20 输出逐样本一致。
同图 Nsight 为 94 次 Graph replay，每次 kernel 从 452 降至 368；
INT4 类 GPU event 合计 1598.01→1531.64 ms，主 `lm_head` Ampere tactic
保持一致，整体 kernel event 3052.47→2976.30 ms。上述结果将性能差异
限定到融合图及由它触发的 tactic 选择，不能把全部收益简单解释为减少 launch。
证据见 `int4_unfused_detailed_build_evidence.json`、
`ps20_int4_unfused_detailed_study.json`、
`int4_fusion_vs_unfused_detailed_comparison_v2.json` 和
`nsight_int4_fusion_vs_unfused_detailed_v4.json`。

2026-10-13 再用上述两份实测 engine 做 direct C++ decode 长稳复验：
`i768/k1024`、batch=1、Graph、seed=0、warm-up=10，按未融合→融合→未融合
顺序各运行 1000 次。三段均正常退出，吞吐依次为
36.4928→37.3506→36.5156 tokens/s；融合相对两侧未融合均值提升
2.32%。三份原始 CSV/日志的 SHA-256 均与执行清单匹配，未见 CUDA 错误；
三段日志仍有已知 `IRuntime::~IRuntime` 析构警告。全程 tegrastats
202 个采样，整机 RAM 峰值 4440 MB、SWAP 峰值 1607 MB、平均瞬时
`VDD_IN` 12.335 W、结温峰值 56.687 °C。这是固定长度的纯 LLM decode，
可核对长时间执行稳定性和局部吞吐，不代替包含视觉与 HTTP 的 PS20 验收。
原始归档 `reports/jetson-readiness-20261012/int4_detailed_direct_soak_artifacts.tgz`
SHA-256 为 `e805648e7c079e8876d06b571ac4430f36c46042842d00ace9def95afc31606a`；
归档中的 `execution.json` 记录 engine、benchmark 二进制和 plugin 哈希。

对同一 fusion 图、同一 builder/config 再构建一次时，TensorRT 日志明确从
687700 字节 cache 加载 315 个 timing 条目；构建时间由首次创建的 113.17 秒
降至复用的 35.37 秒（约 3.20 倍）。复用后的 engine 可反序列化，仍为
`DETAILED`、1640 层，输出层 tactic 相同。完整构建报告和原始日志摘要见
`int4_fusion_cache_build_evidence.json`；复用 engine 的 SHA-256 为
`e32de6bd725bac00a2e63a4ecb2c8582d57b74fd838f89c31b4a5ebc90dbb753`。
`fusion_inspector_audit_evidence.json` 保存完整 inspector、执行代码、源码和
二进制环境键，SHA-256 为
`ed82a82ca4e0d2d3037a27c2f1e85b2746ad85325eee1d50a1dde5097a69ca18`。

构建脚本对 LLM 的 `detailed` 优先使用已核验的原生 CLI；其他精度级别
或 visual builder 需编译相应环境控制。timing cache 仍要求二进制中有
`EDGELLM_TIMING_CACHE_PATH`，并使用独立 builder 路径；键存在只是能力提示，
实际仍以 inspector 和生成的 cache 为准。报告未请求 profiling 时保留 null，
不再将未知默认值写成 none。
每个构建 step 保存真实子进程 PID，成功/非零退出/SIGKILL/启动失败保持独立
事实，`oom_confirmed` 仍需外部证据。原始旧报告保留，不补写未观测的 PID。
