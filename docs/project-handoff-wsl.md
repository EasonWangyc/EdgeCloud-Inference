# 项目进展与 WSL 迁移交接（2026-10-03）

## 项目定位与名称建议

**项目名称：EdgeCloud-Inference**。副标题为“场景驱动的端云协同视觉推理与评测”。*EdgeCloud* 对应 Jetson 端侧和 GPU 服务侧，*Inference* 对应统一推理接口、场景分发与结果评测。当前业务任务是低速泊车场景风险提示，输入为场景图片，输出为严格校验的风险等级、事件、证据和驾驶建议。

仓库名称确定为 `EdgeCloud-Inference`，WSL 目标目录为 `/home/easonwangyc/code/EdgeCloud-Inference`。Python 包和命令为 `parksight_vlm`、`parksight-analyze`、`parksight-study`，用于保持既有运行入口和实验记录兼容。

## 当前技术链路

```text
ParkingCase + 冻结 workload
    -> RiskRuntime（Transformers / Edge-LLM HTTP / vLLM HTTP / 路由组合）
    -> InferenceRecord（结构化结果、失败类别、时延、路由决策）
    -> StudyReport（质量、性能、环境、失败归因）
```

目标模型是 `Qwen/Qwen3-VL-2B-Instruct`，当前配置固定的模型 revision 为 `89644892e4d85e24eaac8bacfd4f463576704203`。风险事件共六类，schema 和 workload 在跨运行时评测中保持一致。服务器 Transformers 负责正确性参考和误差分析；板端性能结论来自 Jetson Transformers FP16 与 Jetson TensorRT Edge-LLM 的同机研究。vLLM 与端云路由是新增服务链路，不能直接继承已有板端性能结论。

## 已有进展与证据边界

| 方向 | 已实现或已记录 | 当前证据边界 |
| --- | --- | --- |
| 业务与评测 | `assessment`、`casebook`、`inference`、`studies` 模块；严格 JSON 校验、来源组和 split 校验、逐样本记录与研究报告 | 已有无硬件单元测试和历史研究报告；迁移后需在新环境重新运行 |
| 端侧推理 | Jetson Transformers FP16、TensorRT Edge-LLM FP16/INT4、LoRA 合并模型和 HTTP 接入的运行记录 | 已有 Jetson 实机证据；不同 engine、精度与采样口径需分别表述 |
| TensorRT 优化 | INT4 level 0/level 1 builder 对照；P2 请求级 microbatch；P3 paged-KV 独立 engine | level 1 是性能复验候选；P2/P3 均未完成默认运行时晋级门槛 |
| 云侧推理 | `VllmHttpBackend` 通过 OpenAI-compatible Chat Completions 发送图像 data URI 和冻结 prompt，解析流式结果并记录 usage/时延；服务地址、鉴权与模型名可配置 | 适配代码和无硬件用例已具备；未见真实 vLLM 服务的模型加载、图像请求、长时间运行或性能报告 |
| 端云分发 | `EdgeCloudRouterRuntime` 使用授权开关、HTTP 健康状态和场景升级条件选择 Edge-LLM/vLLM，记录尝试路径、回退原因和 `routing_decision` | 策略与失败分支已有无硬件测试；Jetson + laptop 两机闭环尚未实测 |
| 数据与 LoRA | PS20 pilot、PS80 人工确认版本、多个 LoRA 候选、合并/量化/板端研究流程和报告 | 现有 validation 很小且多轮参与调参；尚不能给出可靠泛化提升结论 |

板端已记录的一组完整 VLM/HTTP 性能：Jetson Transformers FP16 的 E2E p50/p90/p99 为 **9.569/14.439/28.155 s**（1×20 请求）；TensorRT Edge-LLM INT4 level 0 为 **10.523/11.387/12.723 s**，level 1 为 **2.826/2.953/3.315 s**（各 3×20 请求）。level 0/1 固定相同 INT4 AWQ LLM backbone、`i768/k1024`、workspace、KV capacity、CUDA Graph 和插件版本，仅改 builder optimization level，因此 level 0/1 是本组严格的 engine A/B；Transformers 与 Edge-LLM 的 decode profile 和采样次数不同，不能将各阶段数字笼统归因于推理框架。原始口径、artifact SHA 和资源指标见 [README](../README.md) 与 [TensorRT 优化摘要](tensorrt-optimization-summary.md)。

P2 已在板端完成 c=2/c=4 各 60/60 和 c=4 的 100/100 HTTP soak，并补充资源遥测；缺少同口径 baseline 遥测，因此尚未默认启用。P3 已完成独立 paged engine 的 batch=1/4 低层运行与输出 smoke；数值一致性、HTTP 路径、Nsight 归因和 soak 尚待完成。详情见 [P2/P3 实施记录](tensorrt-dynamic-batching-paged-kv.md)。

## 当前不足

1. **端云闭环尚无实测。** 配置中的 laptop 地址、vLLM 版本和服务模型名是模板值。需验证 WSL 内 GPU 可见、vLLM 加载指定模型 revision、图片请求符合协议、Jetson 可达 laptop 服务，以及断网/超时/服务退出时的回退。客户端填写的 `model_revision` 不会证明服务端实际加载了该 revision。
2. **路由信号尚不完整。** 当前默认 provider 刷新端、云 HTTP 健康状态；显存余量、温度、RTT 与丢包率由接口或配置提供，尚无实时采集。`/health` 仅证明服务可达，不能代替真实多模态推理。路由策略的延迟、质量、云使用率和失败回退需要同一工作负载的独立报告。
3. **数据来源和泛化证据不足。** `ps80_human_confirmed_v1` 为 train 64 / validation 16，当前每张图片各有一个 `source_group_id`，并不能证明视频连续帧和同场景图片已按真实来源归组。validation 没有 high、VRU 或车位冲突正例，固定障碍正例仅 1 个。PS20 只有 20 张，属于 pilot；PS80 validation 已参与多轮方案选择，不能充当独立测试集。
4. **微调效果不稳定。** 服务器 Transformers 的 PS80 validation 上，reviewed v2/v3 风险准确率均为 0.563、事件 micro-F1 均为 0.429；v4 分别下降到 0.375 和 0.353（n=16）。严格 JSON 有效率为 1.0 只表示输出格式可解析。现有小样本和类别不平衡不足以判断 LoRA 的稳定收益。完整误差与数据方案见 [数据集与 LoRA 优化方案](dataset-and-finetuning-roadmap.md)。
5. **本地证据尚未形成可直接重建的迁移包。** `.gitignore` 排除了 `data/raw/`、`data/processed/`、`artifacts/`、`reports/`、`models/`、`patches/`、`.venv/` 等目录；其中原图、训练/量化产物和原始报告在当前工作区确实存在。只克隆 Git 仓库会丢失这些材料。当前 `main` 的 HEAD 为 `374ab83`，工作区还有大量未提交的修改、新文件和删除记录，迁移时必须按实际工作区保存，不能只依据 HEAD 重建。

## WSL 迁移顺序与验收

以下是迁移计划，**本文件不表示已执行迁移或安装**。执行前先在 Windows/Jetson 留存目录清单、文件大小和关键模型/engine/报告 SHA-256，确认数据授权和目标磁盘空间。复制现有工作区时保留未提交文件、配置和版本化元数据；对被忽略的原图、处理后数据、adapter、模型、engine、patch 与原始报告单独清点和备份。不要把 Windows `.venv` 当成 WSL 环境复用，也不要把 Jetson 构建的 engine 当成 laptop GPU engine。

1. **代码与证据落盘。** 在 WSL 的 Linux 文件系统建立工作目录，恢复当前工作区和必要的本地数据/报告，核对文件数量、关键哈希、`git status` 和 HEAD。若选择 Git 传输，需另外转移未提交及忽略文件；当前已有删除记录应按原样保留，避免迁移工具把历史文件意外恢复。
2. **环境基线。** 在 WSL 记录发行版、Python、Windows/WSL GPU 驱动可见性、CUDA/PyTorch/vLLM 版本和 laptop 可用显存。vLLM 官方文档当前要求 Linux，并给出 WSL 作为 Windows 上的运行途径；微软文档说明 WSL 2 可使用 NVIDIA CUDA。具体版本组合与目标模型是否能在本机显存中稳定加载，仍需本机实测。[vLLM GPU 安装要求](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) · [Microsoft WSL CUDA 指南](https://learn.microsoft.com/en-us/windows/ai/directml/gpu-cuda-in-wsl)
3. **无硬件回归。** 在新建的 WSL Python 环境中安装项目依赖后，运行仓库无硬件单元测试、源码编译检查；记录通过数和失败原因。此步不需要模型权重、Jetson 或云服务。入口见 [操作文档](operations.md)。
4. **单机云侧验证。** 先启动仅 laptop/WSL 可访问的 vLLM 服务，确认实际加载的模型 revision、图像输入、严格 JSON、流式首 token、超时和显存占用；再用 `server_vllm_http_ps20_pilot.example.json` 的副本运行研究。模板占位符必须替换，原模板不直接作为实测配置。
5. **两机路由验证。** 在可信局域网/VPN 下让 Jetson 与 laptop 服务互通，先验证端侧结果、云侧结果，再验证 `cloud_allowed=false`、健康失败、请求超时、可回退故障和高风险/关键事件升级。逐样本保留路径、信号快照、失败类别和端到端时间；不要将端云异构时延当作框架 A/B。
6. **质量研究再启动。** 先按真实来源重做分组与近重复审计，补充稀有事件和高风险样本，冻结新的独立测试集；随后在开发集上做 base/LoRA 单变量对照，最后只对入选版本做一次冻结测试集评估。量化和跨后端质量结果分别记录。

迁移验收以“证据能复现”为准：关键数据和报告哈希一致；无硬件测试在 WSL 通过；真实 vLLM 多模态请求成功且服务端模型身份可核对；端云路径及失败回退有逐样本记录；新数据集的来源组、类别覆盖和独立测试边界明确。在上述实测完成前，展示材料应把 vLLM 和端云路由标为**已实现、待实机联调**。

## 主要入口

- [架构与路由语义](architecture.md)、[运行入口和配置说明](operations.md)
- [vLLM Adapter](../src/parksight_vlm/inference/vllm.py)、[端云路由实现](../src/parksight_vlm/inference/routing.py)
- [云侧 study 模板](../configs/studies/server_vllm_http_ps20_pilot.example.json)、[端云 study 模板](../configs/studies/jetson_edge_vllm_router_ps20_pilot.example.json)
- [TensorRT 性能与候选边界](tensorrt-optimization-summary.md)、[数据集与 LoRA 方案](dataset-and-finetuning-roadmap.md)
