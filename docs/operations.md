# 操作入口

## 1. 无硬件验证

仓库的单元测试不加载模型权重，也不要求 CUDA、摄像头或 Jetson：

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
python3 -m compileall -q src tests scripts
git diff --check
```

安装为开发包后，也可以直接使用 `parksight-analyze` 与 `parksight-study` 两个命令。

## 2. 单图分析

当前模型配置固定到 Hugging Face 不可变 commit
`89644892e4d85e24eaac8bacfd4f463576704203`。单图执行示例：

```bash
PYTHONPATH=src python3 -m parksight_vlm.app.analyze_image \
  --image data/raw/example.jpg \
  --runtime transformers \
  --backend-revision REPLACE_WITH_TRANSFORMERS_VERSION \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --precision bf16 \
  --device-map auto \
  --dtype bfloat16 \
  --attn-implementation sdpa
```

命令输出完整 `InferenceRecord`。成功时包含严格校验后的 `assessment`，失败时包含
明确的失败类别和错误事实。单图分析不要求参考标注。`precision` 是实验身份，
`dtype` 是实际传给 Transformers 的加载精度，两者必须保持一致。

TensorRT Edge-LLM HTTP Adapter 使用实验性 OpenAI-compatible server：

```bash
PYTHONPATH=src python3 -m parksight_vlm.app.analyze_image \
  --image data/raw/example.jpg \
  --runtime tensorrt_edge_llm_http \
  --backend-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --adapter-revision edge-http-v2 \
  --precision fp16 \
  --edge-url http://127.0.0.1:8000
```

HTTP server 的启动方式、请求兼容性和目标模型支持情况必须以 Jetson 上实际安装的
TensorRT Edge-LLM revision 为准。

云侧 vLLM 使用独立 `vllm_http` Adapter，通过 OpenAI-compatible Chat Completions
API 发送图片 data URI 和冻结 workload prompt。PS20 研究配置模板为
`configs/studies/server_vllm_http_ps20_pilot.example.json`。复制为实际 study 配置后，
将 `backend_revision` 改为云服务安装的 vLLM 版本，将 `base_url`、`model_name` 和
`precision` 改为实际服务值，并确认服务加载了配置声明的模型 revision；客户端记录的
`model_revision` 不会自动验证远端权重。

WSL 本机部署、独立 Python 环境和 RTX 4060 Laptop 启动参数见
[`vLLM WSL 部署与验收`](vllm-wsl-deployment.md)。本机服务使用实际配置：

```bash
.venv-vllm/bin/python scripts/benchmark_vllm_http.py \
  --config configs/studies/wsl_vllm_workload_resize_ps20.json \
  --gpu-telemetry \
  --metrics-url http://127.0.0.1:8000/metrics
```

若服务要求鉴权，在 Bash 中设置 `VLLM_API_KEY`，并在 runtime options 中设置
`api_key_env`。密钥只从环境读取，不应写进 study JSON 或报告。
当前 HTTP TTFT 是客户端观察到的首个非空输出时延，包含网络与服务排队；
跨云端与 Jetson 的时延不用于框架性能归因。

## 端云协同研究

`configs/studies/jetson_edge_vllm_router_ps20_pilot.example.json` 提供一个 Edge-LLM
优先、vLLM 作为场景升级与故障备用的组合 Runtime。配置中的 laptop 地址、vLLM 版本和服务模型名
是模板值；运行前需改为实际值。端侧服务由 Jetson 本机 HTTP endpoint 提供，云侧服务
可以运行在同一局域网可访问的 laptop GPU 上。笔记本合盖休眠、网络切换或服务退出时，
健康探测会将云路径判为不可用。

vLLM 服务端需运行在其支持的 Linux GPU 环境；Windows 主机应通过 WSL2/Linux 部署，
而不是假设原生 Windows 环境受支持。Laptop 能否承载 Qwen3-VL-2B 还取决于具体 GPU、
可用显存、驱动/CUDA 组合和所选精度；模型权重、视觉编码器、KV cache 与服务开销都占用
显存，启动成功和长时间持续运行需分别验证。

模板默认 `cloud_allowed=false`。只有确认本次场景图片允许离开 Jetson 时，才将其设为
`true`；同时将 `network_available` 设为当前网络许可开关。路由器会请求 Edge 和 vLLM
服务的 `/health`，并记录策略、信号快照、主路径、回退原因、各路径执行结果和 `routing_ms`。
`/health` 只表示服务可达，不保证模型请求可成功；真实请求失败仍按配置的类别决定是否
回退。模型拒答和输入错误不在默认自动回退列表中，避免把业务拒答伪装成可恢复故障。
端侧输出命中策略配置的 high 风险等级或关键事件时，可升级到云端复核；升级遵循相同的
`cloud_allowed` 授权门控。若云端复核失败，保留已成功的端侧结果，并记录升级尝试和失败。

当前 health provider 实时刷新 Edge/vLLM 的 HTTP 可达状态；显存余量、温度、RTT 和丢包
率由 `RoutingSignalsProvider` 接口输入。若配置中未提供这些数值，对应资源/SLA 门槛不会
被应用。因此现阶段可以直接验证授权门控、健康路由和故障回退，资源感知路由需接入真实
设备与网络遥测后再启用并评估。场景升级条件来自端侧已校验的初判，不使用参考标注。

运行组合研究时，执行端必须能够访问配置中声明的服务地址。建议服务只绑定到可信局域网
或 VPN 接口，不直接暴露到公网；远程传输图像和 API key 的连接应使用 TLS。Router 入口
通过 `parksight-study --config` 执行，不作为单图 CLI 后端暴露。

## 两个推理框架的项目化学习

学习任务围绕同一 `ParkingCase`、冻结 workload 和 `InferenceRecord` 展开，先手写最小
请求构造与响应解析，再对照 `VllmHttpBackend`、`EdgeLlmHttpBackend` 和 `RiskRuntime`
补齐流式响应、超时、错误分类、计时和 schema 校验。之后沿两条实际链路分别追踪：vLLM
的 OpenAI-compatible 服务入口、调度/批处理、模型执行与响应；TensorRT Edge-LLM 的
HTTP 服务、engine 构建身份、视觉/LLM 执行与 KV cache。每个阶段保存可运行的小实现、
执行配置和观察到的事实；同一输入用于理解接口差异，不以不同硬件上的耗时差异推断框架
优劣。

板端启动服务时，建议显式传入 Edge-LLM checkout。入口会自动加入源码和 pybind 路径，
并从 checkout 的 `build/libNvInfer_edgellm_plugin.so` 发现插件；也可以用
`--plugin-path` 显式指定插件：

启动前可先做不加载 GPU runtime 的静态预检：

```bash
cd /home/ubuntu/JetsonVLM
export JETSON_PY_CUDA_LIB=$PWD/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib
export PYTHONPATH=$PWD/.venv-jetson/lib/python3.10/site-packages:/home/ubuntu/TensorRT-Edge-LLM:$PWD/src
export LD_LIBRARY_PATH=$JETSON_PY_CUDA_LIB:/home/ubuntu/TensorRT-Edge-LLM/build:$LD_LIBRARY_PATH

.venv-jetson/bin/python scripts/serve_edgellm.py \
  --check-only \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --engine-root artifacts/engines/qwen3_vl_2b_fp16_i768_k1024 \
  --plugin-path /home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
```

预检返回 `ready=true` 只证明文件和路径齐全，不代表统一内存、CUDA graph 或 HTTP
服务一定能启动；这些仍需实际板端加载和请求验证。

如果 engine 构建时未启用 weight streaming（现有 INT4 engine 属于此类），启动时不要
传 `--weight-streaming-budget-bytes`；该参数只适用于对应构建选项已启用的 engine。

LoRA 或 INT4 只有 LLM engine 时，可将其与 FP16 visual engine 分开指定；服务会把两个
目录传给同一个 Edge-LLM 实例：

```bash
.venv-jetson/bin/python scripts/serve_edgellm.py \
  --check-only \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --llm-engine-root artifacts/engines/qwen3_vl_2b_int4_awq_i768_k1024/llm \
  --visual-engine-root artifacts/engines/qwen3_vl_2b_fp16/visual \
  --plugin-path /home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
```

```bash
cd /home/ubuntu/JetsonVLM
export JETSON_PY_CUDA_LIB=$PWD/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib
export LD_LIBRARY_PATH=$JETSON_PY_CUDA_LIB:/home/ubuntu/TensorRT-Edge-LLM/build:$LD_LIBRARY_PATH

.venv-jetson/bin/python scripts/serve_edgellm.py \
  --edge-llm-root /home/ubuntu/TensorRT-Edge-LLM \
  --engine-root artifacts/engines/qwen3_vl_2b_fp16_i768_k1024 \
  --plugin-path /home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so \
  --weight-streaming-budget-bytes 0 \
  --host 127.0.0.1 \
  --port 8000
```

## 3. 冻结研究

1. 按 [`data.md`](data.md) 准备 manifest、人工标注和图片。
2. 确认 `configs/studies/*.json` 中的 `model_revision` 与实际缓存和待构建 engine
   使用同一个不可变 commit；升级模型时必须显式修改并重新验证。
3. 检查 backend、精度、power mode 和输出路径。
4. 为三类实验使用独立配置和报告：

| 配置 | 运行位置 | 状态 |
| --- | --- | --- |
| `configs/studies/transformers_base.json` | GPU 服务器 | 已提供，作为正确性参考 |
| `configs/studies/jetson_transformers_fp16.json` | Jetson | 已提供，作为板端 Transformers FP16 基线 |
| `configs/studies/edgellm_fp16.json` | Jetson | 已提供，作为 TensorRT Edge-LLM FP16 基线 |
| `configs/studies/jetson_edgellm_fp16_ps20_pilot.json` | Jetson | 与 PS2.0 Transformers FP16 基线同口径比较 |

服务器正确性参考：

```bash
PYTHONPATH=src python3 -m parksight_vlm.app.run_study \
  --config configs/studies/transformers_base.json
```

Jetson Transformers FP16：

```bash
PYTHONPATH=src python3 -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_transformers_fp16.json
```

Jetson TensorRT Edge-LLM FP16：

```bash
PYTHONPATH=src python3 -m parksight_vlm.app.run_study \
  --config configs/studies/edgellm_fp16.json
```

研究入口在运行前校验来源组划分和标注完整性。报告保存配置身份、环境快照、每条
`InferenceRecord`、质量指标、性能分位数和失败汇总。

通用 StudyReport 的 `first_request_ms` 表示本次研究第一个请求的端到端时延，
包括该请求失败的情况。`cold_start_ms` 没有独立的运行时初始化证据时保持 null；
不能将已预热 HTTP 服务的首个研究请求视为冷启动。历史报告不回写，新报告使用此口径。

HTTP Adapter 会在流式读取、响应解析或服务端错误后关闭持久连接，下一请求重新建连，
不隐式重发 POST。OpenAI-compatible SSE 必须以 `[DONE]` 结束；提前断流和协议错误
记录为运行时失败，明确的 `refusal` 或 `content_filter` 记录为模型拒答。

三类配置必须使用相同的 workload、模型 revision 和冻结测试集。服务器报告用于正确性
参考；Jetson Transformers 与 Jetson Edge-LLM 报告用于同机性能比较。

## 4. LoRA、合并、导出与 engine 构建

`configs/flows/*.example.json` 保留为新实验的待审核模板。Qwen3-VL Base FP16、LoRA、
合并、INT4 和 Jetson engine 构建均已提供不带 `.example` 的实际配置；其中基础 FP16
部署入口包括：

- `configs/flows/export_qwen3_vl_2b_fp16.json`
- `configs/flows/build_qwen3_vl_2b_fp16_engines.json`

完整服务器导出、传输、Jetson runtime 编译、engine 构建、server 和验收命令见
[`edgellm-deployment.md`](edgellm-deployment.md)。新增实验先复制模板为不带
`.example` 的实际配置，填入当前环境中已确认的命令、输入和输出，再做 dry-run：

```bash
PYTHONPATH=src python3 scripts/train_lora.py \
  --config configs/flows/train_lora.json
PYTHONPATH=src python3 scripts/merge_lora.py \
  --config configs/flows/merge_lora.json
PYTHONPATH=src python3 scripts/export_model.py \
  --config configs/flows/export_model.json
PYTHONPATH=src python3 scripts/build_engine.py \
  --config configs/flows/build_engine.json
```

默认行为只输出 readiness，不执行外部命令。确认以下条件后，才增加 `--execute`：

- 命令和版本已经人工复核；
- required inputs 全部存在；
- expected outputs 尚不存在；
- record 与 log 路径尚未被占用。

执行时命令参数直接传给子进程，不经过 shell。结果只有在进程返回 0 且全部预期输出
存在时才记为 `succeeded`；日志和结构化结果均进入 `reports/flows/`。

## 5. 证据边界

代码可用、命令可启动和模型完成部署是三种不同状态。报告中只有实际执行产生的记录
可以作为质量、时延、内存、功耗或温度证据。dry-run、engine 文件存在、README
描述以及服务器可连接均不能替代板端推理验证。
