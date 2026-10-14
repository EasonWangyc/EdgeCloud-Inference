# EdgeCloud-Inference

<p>
  <img src="https://img.shields.io/badge/Built%20with-Codex-412991" alt="Built with Codex">
</p>

<p>
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Qwen3--VL-2B--Instruct-6E56CF" alt="Qwen3-VL 2B Instruct">
</p>

<p>
  <img src="https://img.shields.io/badge/vLLM-0.11.2-30A2FF?logo=vllm&amp;logoColor=FDB515" alt="vLLM 0.11.2">
  <img src="https://img.shields.io/badge/TensorRT--Edge--LLM-v0.9.1-76B900?logo=nvidia&amp;logoColor=white" alt="NVIDIA TensorRT Edge-LLM v0.9.1">
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
| vLLM | WSL `0.11.2`；云侧版本按部署记录 | GPU OpenAI-compatible HTTP 服务、批处理与端云路由 |
| PyTorch | 服务器 `2.8.0+cu128`；Jetson `2.9.1` | 训练与推理 |
| PEFT / Accelerate | `0.18.0` / `1.10.1` | LoRA 训练 |
| ModelOpt | `0.44.0` | INT4 AWQ 量化 |
| TensorRT Edge-LLM | `v0.9.1`，commit `7f061f21f0a581ba234a1e233c9315b89d8e47d6` | Jetson runtime、导出和 engine 构建 |
| JetPack / CUDA / TensorRT | `6.2.2` / `12.6` / `10.3` | Jetson Orin Nano 板端环境 |


默认评测 workload 为 `configs/workloads/parking_risk_v2_strict_json.json`，
保留 `parking_risk_v1` schema，输入尺寸为 `448x448`，
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

![泊车风险推理与评测数据流](assets/inference-flow.svg)

支持服务器 Transformers 正确性参考与 vLLM HTTP 服务、Edge-LLM/vLLM HTTP 端云路由、
Jetson Transformers FP16、TensorRT Edge-LLM FP16、领域 LoRA、合并模型和 LLM backbone
INT4 AWQ 研究。服务器结果
用于质量和误差分析；板端性能只比较 Jetson 上的 runtime。

## 性能

下表汇总 `ps20_pilot_v1` 的单图完整请求结果，与上文默认的 v2 workload 分属不同研究口径。Jetson 使用 15W 模式；WSL 使用 RTX 4060 Laptop。设备、输入预处理和统计窗口不同，**不能用跨设备数字计算框架加速比**；同机的 TensorRT level 0/1 才是严格的 builder 变量对照。

| 设备 / 运行时 | 请求 | 客户端 TTFT | E2E p50 / p90 / p99 | 输出吞吐 |
| --- | ---: | ---: | ---: | ---: |
| Jetson Transformers FP16 基线 | 1×20 | 716.69 ms | 9.569 / 14.439 / 28.155 s | 7.35 tok/s |
| Jetson TensorRT Edge-LLM INT4，level 0 | 3×20 | 872.66 ms | 10.523 / 11.387 / 12.723 s | 7.32 tok/s |
| Jetson TensorRT Edge-LLM INT4，level 1 | 3×20 | 702.50 ms | 2.826 / 2.953 / 3.315 s | 27.27 tok/s |
| WSL vLLM BF16，eager，单序列 | 1×20 | 33.04 ms（p50） | 1.456 / 2.221 / 2.412 s | 51.476 tok/s |

Jetson 吞吐沿用报告的输出 token 总数 ÷ 各请求 E2E 耗时之和；vLLM 为正式执行墙钟窗口的输出吞吐。Jetson TTFT 沿用原报告汇总值，vLLM 为 HTTP 首个非空文本片段的 p50。两者的分母与统计量不同，表格用于定位各自实验，不构成跨设备排名。详细口径见 [运行时性能测量](docs/runtime-performance.md)。

### TensorRT Edge-LLM 优化

- **Builder level 0 → 1**：保持模型、INT4 权重、视觉 engine、profile 和 CUDA Graph 不变，仅调整 LLM builder optimization level。Jetson 完整请求 E2E p50 从 10.523 s 降至 2.826 s；低层 decode Graph 从 126.78 降至 27.47 ms/token（4.62×）。Engine Inspector 将主要变化定位在 `lm_head` GEMM tactic；level 1 仍是候选 engine。
- **P0/P1：测量与算子**：C++ direct benchmark 分离 prefill/decode；Nsight 与 Engine Inspector 定位 `lm_head` 和 INT4 GEMV 热点。INT4 Q/K/V、gate/up 融合候选在同构建条件下比未融合 engine 的 PS20 输出吞吐高 1.79%–1.99%，尚未替换默认配置。
- **P2：动态批处理**：修复 batch eviction、sampling indices 与空 batch cache compact，加入单 worker HTTP microbatch scheduler。同服务 c=1→4→1 各 60/60 请求完成，逐样本输出一致；吞吐为 27.259→40.169→27.261 tok/s，另有 c=4 的 100/100 请求稳定性复验。
- **P3：paged KV**：补齐 builder、plugin、page table 和 KV 写入，并按 dense/paged 布局分别处理 eviction。最终分流版 c=1→4→1 共 180/180 请求严格 JSON 有效、60/60 逐样本原文一致，吞吐为 27.349→40.200→27.348 tok/s。长输入 dense/paged 严格一致性、Nsight 路由与 100 请求 soak 尚未通过部署门禁，默认 engine 不变。

level 0/1 的同口径运行时 profile 峰值内存分别为 4,030.4 / 4,040.1 MB；整机 `tegrastats` RAM 与该字段不是同一口径。完整过程与证据见 [优化阶段速览](docs/tensorrt-optimization-summary.md)、[dynamic batching / paged KV 实施记录](docs/tensorrt-dynamic-batching-paged-kv.md)和[完整实验记录](docs/tensorrt-optimization.md)。

## 项目结构

![项目目录与模块结构](assets/project-structure.svg)

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

`inference/` 还包含 vLLM HTTP Adapter 与端云路由。`models/`、`data/raw/`、
`artifacts/`、`reports/` 等为本地输入或产物，不进入 Git；模块接口见
[系统架构](docs/architecture.md)。

## 使用方法

### 安装与无硬件测试

```powershell
py -3.12 -m venv .venv
& ".\.venv\Scripts\python.exe" -m pip install -e .
$env:PYTHONPATH = "src"
& ".\.venv\Scripts\python.exe" -m unittest discover -s tests -v
```

### 单图推理

Transformers 单图示例：

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

LoRA 增加 `--adapter-path`；Jetson HTTP 使用 `--runtime tensorrt_edge_llm_http`
与 `--edge-url`；vLLM HTTP 使用 `--runtime vllm_http` 与 `--vllm-url`。
WSL 服务准备见 [vLLM 部署](docs/vllm-wsl-deployment.md)。

### 运行配置化 Study

```powershell
$env:PYTHONPATH = "src"
Copy-Item configs\studies\server_vllm_http_ps20_pilot.example.json `
  configs\studies\server_vllm_http_ps20_pilot.json
& ".\.venv\Scripts\python.exe" -m parksight_vlm.app.run_study `
  --config configs\studies\server_vllm_http_ps20_pilot.json
```

先在副本中填写服务地址和部署身份；报告写入配置指定的 `reports/` 路径。
Jetson 研究使用 `configs/studies/` 中对应配置，在板端执行。

### 训练前门禁

正式训练前先校验数据、划分、工作负载与模型路径：

```powershell
$env:PYTHONPATH = "src"
& ".\.venv-train\Scripts\python.exe" scripts\finetune_qwen3_vl_lora.py `
  --config configs\training\qwen3_vl_2b_lora_ps64_reviewed_v1.json `
  --validate-only
```

训练、导出和构建入口见 [操作手册](docs/operations.md)；板端步骤见
[Edge-LLM 部署](docs/edgellm-deployment.md)。

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
