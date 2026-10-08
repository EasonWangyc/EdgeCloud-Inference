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
`scripts/prepare_edgellm_timing_binding.py` 可针对已核验源码生成增量补丁，增加
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

`scripts/build_edgellm_timing_binding.py` 默认生成只读构建计划，显式 `--execute`
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
注册。`scripts/run_edgellm_native_probe.py` 在构造 runtime 时临时指定已验证模块，
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
且 native worker 仍在执行的场景由无硬件测试验证后续请求返回 503；本轮未执行
真实 GPU 断流测试，不能把正常完成请求的证据用于该分支。

## 冻结 PS20 完整验收

2026-10-09 使用新 study 配置
`configs/studies/jetson_edgellm_int4_serial_stages_ps20_pilot.json`，标准 StudyRunner
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
这是可验证的下一步优化方向，尚未修改、编译或证明能解决视觉加载 OOM。

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
