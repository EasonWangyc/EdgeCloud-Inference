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
这份新 Adapter 与配置尚未部署到 Jetson，不能用本地测试代替板端结果。

C++ `qwenViTRunner.cpp` 从视觉 builder 配置读取 min/max image tokens，
从预处理配置读取 patch=16、merge=2；尺寸按 factor=32 对齐。
当前已核验的基础 FP16 视觉 engine 配置对应像素范围 8192～2097152，
448×448 已对齐且处于范围内，因此按该源码规则预期不会再次改变尺寸。
这不是现有二进制或视觉张量数值一致性的实测结论，其他 engine 需分别核验配置。

新增 `serve_edgellm.py --stream-usage` 可从 native delta token IDs 补齐 SSE
completion usage；安装只发生在服务进程内。当前签名及客户端解析的离线验证见
`stream_usage_source_snapshot.json` 和 `stream_usage_verification.json`。
新增入口尚未在板端启用，阶段时间和 prompt/decode 计数仍待补齐。

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
结束后恢复 loader；该做法限于隔离诊断，尚未接入正式 HTTP 启动入口。

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

候选已验证 native 单图采集；正式 HTTP 的绑定选择、阶段回传和冻结 PS20 study
仍待接入验收。FP16 OOM 继续作为独立实验事实保留，不能用 INT4 成功替代它。

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
