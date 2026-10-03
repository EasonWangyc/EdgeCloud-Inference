# EdgeCloud-Inference

<p>
  <img src="https://img.shields.io/badge/Built%20with-Codex-412991" alt="Built with Codex">
<p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Qwen3--VL-2B--Instruct-6E56CF" alt="Qwen3-VL 2B Instruct">
  <img src="https://img.shields.io/badge/TensorRT--Edge--LLM-v0.9.1-76B900" alt="TensorRT Edge-LLM v0.9.1">
  <img src="https://img.shields.io/badge/JetPack-6.2.2-76B900" alt="JetPack 6.2.2">
</p>

面向低速泊车场景的视觉语言模型研究与部署项目。项目围绕
`Qwen/Qwen3-VL-2B-Instruct` 搭建风险理解、领域 LoRA 适配、INT4 AWQ 量化、
TensorRT Edge-LLM 部署和可审计评测链路。

## 技术栈与固定版本

| 组件 | 版本或固定身份 | 用途 |
|---|---|---|
| Python | `>=3.10` | 项目运行环境 |
| Qwen3-VL | `Qwen/Qwen3-VL-2B-Instruct`，revision `89644892e4d85e24eaac8bacfd4f463576704203` | 基础视觉语言模型 |
| Transformers | 服务器 `5.9.0`；Jetson `4.57.6` | 训练、质量参考和 Jetson FP16 基线 |
| vLLM | 云侧版本按部署记录 | 云端 GPU OpenAI-compatible 推理服务；当前接入 HTTP Adapter |
| PyTorch | 服务器 `2.8.0+cu128`；Jetson `2.9.1` | 训练与推理 |
| PEFT / Accelerate | `0.18.0` / `1.10.1` | LoRA 训练 |
| ModelOpt | `0.44.0` | INT4 AWQ 量化 |
| TensorRT Edge-LLM | `v0.9.1`，commit `7f061f21f0a581ba234a1e233c9315b89d8e47d6` | Jetson runtime、导出和 engine 构建 |
| JetPack / CUDA / TensorRT | `6.2.2` / `12.6` / `10.3` | Jetson Orin Nano 板端环境 |


默认评测 workload 为 `configs/workloads/parking_risk_v2_strict_json.json`，其
workload identity 为：

```text
parking_risk_v2_strict_json@sha256:6ca953643f38a13b579a77090c77d3fca30d3ba9a1b181d88ae11692ea150fec
```

该 workload 保留 `parking_risk_v1` schema，输入尺寸为 `448x448`，
`max_new_tokens=256`，`do_sample=false`，并要求模型直接输出原始 JSON。

## 功能与输出

输入为单张泊车场景图片，输出为一个严格校验的 `ParkingAssessment`：

```json
{
  "schema_version": "parking_risk_v1",
  "risk_level": "medium",
  "events": ["narrow_passage"],
  "evidence": ["车辆两侧空间较窄，可用横向通行区域有限。"],
  "driver_advice": ["slow_down", "maintain_observation"]
}
```

字段约束如下：

- `risk_level`：只能为 `low`、`medium` 或 `high`，表示场景整体风险。
- `events`：固定六类事件的多标签数组，可以为空。
- `evidence`：与图片可见线索对应的简短说明，不是思维链。
- `driver_advice`：枚举式安全提示，不是车辆控制指令。
- `prepare_to_stop`：表示高风险场景下随时准备停车，不表示“准备泊入车位”。

核心调用链为：

```text
ParkingCase -> RiskRuntime -> InferenceRecord -> StudyReport
```

支持服务器 Transformers 正确性参考与 vLLM HTTP 服务、Edge-LLM/vLLM HTTP 端云路由、
Jetson Transformers FP16、TensorRT Edge-LLM FP16、领域 LoRA、合并模型和 LLM backbone
INT4 AWQ 研究。服务器结果
用于质量和误差分析；板端性能只比较 Jetson 上的 runtime。

## 最终已核验结果

本节只汇总性能证据。测试板为 Jetson Orin Nano Super，SM87，15W，batch=1；JetPack
R36.5、CUDA `12.6.68`、TensorRT `10.3.0.30`、TensorRT Edge-LLM v0.9.1，模型和
workload identity 与上文固定。Transformers FP16 是跨 runtime 参考；Edge-LLM level 0
和 level 1 使用相同 INT4 AWQ LLM backbone、`i768/k1024` profile、workspace 1024 MiB、
KV capacity 1024、CUDA Graph 和插件版本，仅改变 builder optimization level。

本轮正式性能证据绑定以下 artifact provenance：level 0 engine SHA-256 为
`33466f3f1149801bf496737fbe1e69634b8018ece9a0c5bccd8ff83af64afab4`，level 1 candidate
engine SHA-256 为 `207e109fcca28ac29ae0348d8dd517e61a6291704a8fea08c53644574f28d345`，
Edge-LLM plugin SHA-256 为 `9437996d36b659092e7d4da244b43da6feb7c69e79a5654a8df53e9f5c2ac7eb`。

### 基线与完整请求性能

| 阶段 | Runtime / 配置 | Prefill | Prefill 吞吐 | TTFT | Decode latency | Decode-only 吞吐 | E2E p50 / p90 / p99 | E2E aggregate 吞吐 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 基线 | Jetson Transformers FP16 | 486.32 ms | 未采集 | 716.69 ms | 7,897.63 ms/request | 9.54 tok/s | 9.569 / 14.439 / 28.155 s | 7.35 tok/s |
| 优化阶段 1 | TensorRT Edge-LLM INT4，level 0 | 531.61 ms | 1,442.78 tok/s | 872.66 ms<sup>1</sup> | 124.02 ms/token | 8.06 tok/s | 10.523 / 11.387 / 12.723 s | 7.32 tok/s |
| 优化阶段 2 | TensorRT Edge-LLM INT4，level 1 | 430.97 ms | 1,779.70 tok/s | 702.50 ms<sup>1</sup> | 27.25 ms/token | 36.70 tok/s | 2.826 / 2.953 / 3.315 s | 27.27 tok/s |

上表是已完成的完整 VLM/HTTP 性能记录：基线为 `1×20` 请求，Edge-LLM 两档为
`3×20` 请求。不同 runtime 的 decode profile 不是严格 A/B；TTFT 是请求发送到 SSE
首个非空 `delta.content` 的客户端时间，包含视觉编码、调度、首 token decode、网络和
流式输出，不能用 prefill 代替。E2E 包含预处理、视觉编码、prefill、decode、HTTP 和
序列化。`Decode-only 吞吐` 来自模型生成阶段；`E2E aggregate 吞吐` 是完整请求的
总输出 token 数除以 E2E wall-clock，不能用前者替代后者。基线的 prefill 吞吐未在原始
Transformers profile 中采集，因此保留为“未采集”。

### 板端内存与资源占用

Jetson Orin Nano 使用统一内存架构，没有独立的离散显存。下表同时保留运行时 profile
的峰值内存和 `tegrastats` 的系统 RAM/SWAP 峰值；二者采样器、时间范围和统计对象不同，
不能当作同一份进程显存账本。level 0/1 的峰值统一内存来自真实 VLM profile 三次平均，
并且是严格可比的 builder level A/B。

| 阶段 | Runtime profile 峰值内存 | `tegrastats` RAM used 峰值 | `tegrastats` SWAP used 峰值 | 说明 |
|---|---:|---:|---:|---|
| 基线 | 4,168.8 MB | 6,231 MB | 1,059 MB | Transformers `peak_memory_mb`；未采集 native unified-memory 字段 |
| 优化阶段 1 / level 0 | 4,030.4 MB | 4,904 MB | 1,083 MB | Edge-LLM native profile，三次平均；INT4、`i768/k1024` |
| 优化阶段 2 / level 1 | 4,040.1 MB | 4,905 MB | 1,085 MB | 相对 level 0 峰值统一内存 `+9.66 MB`、`+0.24%` |

内存结论只在同一采样口径内比较：level 1 相对 level 0 没有明显内存恶化，且本轮
没有 OOM；不同 runtime 的 `Runtime profile 峰值内存` 不直接作为显存节省结论。

### Edge-LLM 低层严格 A/B

| 阶段 | 构建变量 | Prefill（20 次均值） | Decode Graph（20 次均值） | Decode 吞吐 | 相对 level 0 |
|---|---|---:|---:|---:|---:|
| level 0 | `builderOptimizationLevel=0` | 526.8768 ms | 126.7803 ms/token | 7.8877 tok/s | — |
| level 1 | `builderOptimizationLevel=1` | 426.7482 ms | 27.4670 ms/token | 36.4073 tok/s | prefill `-19.0042%`；decode `4.6157×` |
| level 1 soak | level 1，CUDA Graph，1000 steps | — | 27.3821 ms/token | 36.5202 tok/s | 运行完成，0 次 runtime failure |

level 0/1 的低层复验固定 `pastKVLen=768`、warm-up=10、每次 20 次、seed=0。level 1
相对 level 0 的 decode latency 降低 `78.3618%`；1000-step soak 的进程退出码为 0、
Graph capture 成功，仍记录到独立 metadata loader 的 TensorRT runtime destructor warning。
以上 low-level 结果用于确认 engine/tactic 差异的稳定性，不把 HTTP round-trip 当作 decode
latency，也不将 level 1 表述为已替换默认 engine。

原始报告位于本地忽略目录 `reports/jetson-tensorrt-stage1/` 与
`reports/jetson-tensorrt-revalidation/`，不随仓库版本化；仓库内的结论、命令和证据边界见
下方的 TensorRT 优化摘要与完整实验记录。

## TensorRT 优化文档

近期 TensorRT Edge-LLM 的 builder、CUDA Graph、Nsight、`lm_head`、INT4 GEMV/GEMM
和 profile 实验集中在 [TensorRT 优化阶段速览](docs/tensorrt-optimization-summary.md)。
本轮已按 P0 C++ direct runtime、P1 `lm_head`/INT4 GEMV-GEMM、P2 dynamic batching、
P3 paged KV 顺序执行：P0 已完成 direct benchmark 复验；P1 已完成 Nsight/Inspector 热点
复核但没有新的稳定 kernel 候选；P2 已修复 active-batch eviction、sampling indices
compact 和空 batch cache compact，新增单 worker bounded native microbatch scheduler，并在
板端重编译后完成 max batch=4 的 batch=1/4 低层 smoke；HTTP c=2、c=4 三次重复均为
60/60 成功，c=4 的 100 请求 soak 为 100/100 成功，aggregate output tok/s 为
`27.5115/41.0209/40.2069`，已补一轮 RAM/SWAP/GPU/功耗/温度 telemetry，但因缺少同口径
 baseline telemetry 暂不切换默认 runtime；P3 已在固定 v0.9.1 checkout 上完成最小
builder/plugin/runtime backport，独立 paged engine 已通过 batch=1/4 prefill/decode smoke 和
3×20 低层复验，显式记录 `maxKVPoolPages=32`、`tokens_per_page=128` 与 page table contract。
数值 parity、HTTP paged scheduler、Nsight route 和 100 次 soak 仍未完成，因此不替换默认
level0/level1 engine。对应的可执行入口和验收边界见
[`jetson_orin_nano_priority_v1.json`](configs/tensorrt/jetson_orin_nano_priority_v1.json)。
完整命令、候选 patch、provenance 和原始证据边界见 [完整实验记录](docs/tensorrt-optimization.md)，
P2/P3 的实现细节见 [dynamic batching 与 paged KV 实施记录](docs/tensorrt-dynamic-batching-paged-kv.md)。

## 项目结构

```text
src/parksight_vlm/
  assessment/       # ParkingAssessment、严格 JSON schema 和语义审计
  casebook/         # ParkingCase、annotation、manifest 和 split 校验
  inference/        # Transformers / TensorRT Edge-LLM adapters
  studies/          # InferenceRecord、指标、性能和 StudyReport
  app/              # 单图分析、Study 入口和 runtime factory
configs/
  workloads/        # 冻结 prompt、生成参数、输入尺寸和 schema
  studies/          # 质量/性能 Study 配置
  flows/             # 训练、合并、导出、量化和 engine 构建配置
data/
  manifests/        # 可提交的样本元数据
  annotations/      # 可提交的标注
  raw/               # 本地图片输入，默认不进入 Git
scripts/             # 训练、评测、导出、量化、构建和服务脚本
tests/               # 无硬件单元测试
docs/                # 架构、数据、评测、部署和过程记录
artifacts/           # adapter、ONNX、权重和 engine，本地生成
reports/             # StudyReport 和运行证据，本地生成
models/              # 本地模型权重，默认不进入 Git
```

工作区约定：`src/`、`scripts/`、`configs/`、`tests/` 和 `docs/` 是可审阅的项目实现、实验
入口和说明；`patches/` 是固定 Edge-LLM checkout 的本地候选 patch cache，当前不随仓库版本化，
只通过 patch SHA、源码 revision 和 provenance 记录实验身份。`models/`、`data/raw/`、
`data/processed/`、`artifacts/`、`reports/` 和两个 `.venv/` 目录是本地输入或生成物，不作为
源码层级互相引用。`reports/` 保留实验事实，`artifacts/` 保留模型、ONNX、engine 和传输包；
清理时先依据配置/报告引用关系，再删除临时包，不把可复现实验所需的证据误删为“中间产物”。

核心接口：

- `ParkingAssessment.from_mapping(payload)`：解析、校验和序列化严格 JSON。
- `ParkingCaseCatalog.load()` / `validate()`：加载样本并校验来源组、split 和图片引用。
- `RiskRuntime.analyze(case, workload)`：执行一次推理并保留成功或失败事实。
- `StudyRunner.run(casebook, runtime, study)`：聚合推理记录并生成 `StudyReport`。

## 使用方法

### 安装与无硬件测试

在已准备好项目依赖的环境中：

```powershell
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -e .
$env:PYTHONPATH = "src"
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

### 单图推理

Transformers 运行时示例：

```powershell
$env:PYTHONPATH = "src"
& ".\.venv\Scripts\python.exe" -m parksight_vlm.app.analyze_image `
  --image data\raw\example.jpg `
  --workload configs\workloads\parking_risk_v2_strict_json.json `
  --runtime transformers `
  --backend-revision transformers-5.9.0 `
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 `
  --precision bf16
```

使用已训练 LoRA 时增加 `--adapter-path <adapter-directory>`；使用 Jetson Edge-LLM
HTTP runtime 时将 `--runtime` 改为 `tensorrt_edge_llm_http`，并提供 `--edge-url`。
云侧 vLLM 使用 `--runtime vllm_http`、`--vllm-url` 和可选 `--api-key-env`；配置化
PS20 研究模板见 `configs/studies/server_vllm_http_ps20_pilot.example.json`。云端模型服务
必须加载与 study 中 `model_revision` 相同的不可变模型版本。vLLM HTTP 的 TTFT 是客户端
观测值，会包含网络与服务排队时间。

### 运行配置化 Study

```powershell
$env:PYTHONPATH = "src"
Copy-Item configs\studies\server_vllm_http_ps20_pilot.example.json `
  configs\studies\server_vllm_http_ps20_pilot.json
& ".\.venv\Scripts\python.exe" -m parksight_vlm.app.run_study `
  --config configs\studies\server_vllm_http_ps20_pilot.json
```

运行前需在复制出的配置中填入实际的 vLLM 版本、服务地址和部署配置。

Study 会将完整报告写入配置指定的 `reports/` 路径，包含 JSON、风险、事件、时延、
内存和失败归因。Jetson 研究使用 `configs/studies/` 中对应的 Edge-LLM 配置，并在
板端执行。

### 训练前门禁

正式 LoRA 训练前可先只做数据、来源、split、图片、workload 和模型路径校验：

```powershell
$env:PYTHONPATH = "src"
& ".\.venv-train\Scripts\python.exe" scripts\finetune_qwen3_vl_lora.py `
  --config configs\training\qwen3_vl_2b_lora_ps64_reviewed_v1.json `
  --validate-only
```

训练、合并、导出、量化和 engine 构建均通过 `scripts/` 与 `configs/flows/` 的显式
配置执行；完整操作顺序见 [`docs/operations.md`](docs/operations.md)，Jetson 部署细节
见 [`docs/edgellm-deployment.md`](docs/edgellm-deployment.md)。

## 文档入口

- [当前实现状态](docs/status.md)
- [系统架构](docs/architecture.md)
- [数据说明](docs/data.md)
- [评测口径](docs/evaluation.md)
- [操作入口](docs/operations.md)
- [数据集与 LoRA 优化方案](docs/dataset-and-finetuning-roadmap.md)
- [Edge-LLM 部署](docs/edgellm-deployment.md)
- [TensorRT 优化阶段速览](docs/tensorrt-optimization-summary.md)
- [TensorRT 优化完整实验记录](docs/tensorrt-optimization.md)
