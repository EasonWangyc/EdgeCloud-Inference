# TensorRT Edge-LLM FP16 部署流程

本文记录 `Qwen/Qwen3-VL-2B-Instruct` 的首个 TensorRT Edge-LLM板端实验流程。2026-08-04 已完成固定 revision 模型的服务器 ONNX 导出，并生成成功的 flow record；已完成 Jetson C++ runtime 编译和视觉 engine构建。

## 1. 固定实验身份

| 项目 | 固定值 |
| --- | --- |
| 模型 | `Qwen/Qwen3-VL-2B-Instruct` |
| 模型 revision | `89644892e4d85e24eaac8bacfd4f463576704203` |
| TensorRT Edge-LLM revision | `7f061f21f0a581ba234a1e233c9315b89d8e47d6` (`v0.9.1`) |
| 精度 | LLM FP16、视觉编码器 FP16 |
| Jetson 平台 | Orin Nano、JetPack 6.2.2、CUDA 12.6 |
| 工作负载 | `configs/workloads/parking_risk_v1.json` |
| 板端对照配置 | `configs/studies/jetson_transformers_fp16_ps20_pilot.json` |
| Edge-LLM 配置 | `configs/studies/jetson_edgellm_fp16_ps20_pilot.json` |

两个板端 Study 必须使用相同的模型 revision、工作负载、PS2.0 样本、功耗模式和生成上限。Edge-LLM 的 engine 只服务于上述 Edge-LLM revision、TensorRT 版本和构建参数；任一项变化都重新构建，不复用旧 engine。

## 2. 产物边界

```text
GPU 服务器
models/Qwen3-VL-2B-Instruct-89644892/
  -> artifacts/onnx/qwen3_vl_2b_fp16/{llm,visual}/

Jetson
artifacts/onnx/qwen3_vl_2b_fp16/{llm,visual}/
  -> artifacts/engines/qwen3_vl_2b_fp16/{llm,visual}/
  -> Edge-LLM HTTP server
  -> InferenceRecord / StudyReport
```

模型下载与 ONNX 导出在 x86 Linux NVIDIA GPU 服务器执行；C++ runtime 编译、TensorRT engine 构建和推理在 Jetson 执行。不要在 Jetson 上使用 server 的`--model` 参数，因为这会把模型解析和 ONNX 导出重新带回板端。

## 3. GPU 服务器：固定源码与导出环境

以下命令在 GPU 服务器 Bash 中逐条执行。先确认服务器 CUDA、GPU 和磁盘：

```bash
nvidia-smi
nvcc --version
python3 --version
df -h
```

成功标准：x86-64 Linux、NVIDIA GPU、CUDA 12.x 或 13.x、Python 3.10+，并为模型、ONNX 和临时文件保留足够空间。

固定 TensorRT Edge-LLM 源码：

```bash
cd /path/to/workspace
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout --detach 7f061f21f0a581ba234a1e233c9315b89d8e47d6
git submodule update --init --recursive
git rev-parse HEAD
```

最后一条命令必须输出
`7f061f21f0a581ba234a1e233c9315b89d8e47d6`。

在独立导出环境中安装该 checkout 的依赖：

```bash
python3 -m venv .venv-export
source .venv-export/bin/activate
python -m pip install --upgrade pip
python -m pip install .
tensorrt-edgellm-export --help
```

在服务器上的 ParkSight-VLM 仓库下载固定模型 revision。若已经存在模型目录，先用文件清单和 Hugging Face 缓存信息核对 revision，不覆盖未知来源目录。

```bash
cd /path/to/JetsonVLM
hf download Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --local-dir models/Qwen3-VL-2B-Instruct-89644892
```

先 dry-run，再显式导出：

```bash
PYTHONPATH=src python3 scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_fp16.json

PYTHONPATH=src python3 scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_fp16.json \
  --execute
```

成功标准：命令返回 0，并同时生成：

- `artifacts/onnx/qwen3_vl_2b_fp16/llm/model.onnx`
- `artifacts/onnx/qwen3_vl_2b_fp16/visual/model.onnx`
- 两个目录对应的 `config.json`，以及 LLM tokenizer/chat template、视觉预处理配置
- `reports/flows/export_qwen3_vl_2b_fp16.json`
- `reports/flows/export_qwen3_vl_2b_fp16.log`

## 4. 将 ONNX 传到 Jetson

将完整的 `artifacts/onnx/qwen3_vl_2b_fp16` 目录复制到 Jetson 的
`/home/ubuntu/JetsonVLM/artifacts/onnx/`。

传输前后分别执行：

```bash
find artifacts/onnx/qwen3_vl_2b_fp16 -type f -print0 \
  | sort -z \
  | xargs -0 sha256sum > qwen3_vl_2b_fp16.sha256

sha256sum -c qwen3_vl_2b_fp16.sha256
```

成功标准：Jetson 上全部文件校验为 `OK`。

## 5. Jetson：编译固定版本的 C++ runtime

以下命令在 SSH 登录 Jetson 后的 Bash 中执行：

```bash
cd /home/ubuntu
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout --detach 7f061f21f0a581ba234a1e233c9315b89d8e47d6
git submodule update --init --recursive
git rev-parse HEAD
```

为 HTTP server 安装最小 Python 依赖，不替换 ParkSight 已验证的 Jetson PyTorch：

```bash
/home/ubuntu/JetsonVLM/.venv-jetson/bin/python -m pip install \
  -r /home/ubuntu/TensorRT-Edge-LLM/requirements-server.txt
```

配置并以较低并行度编译，避免 8GB 板端因编译并发产生额外内存压力：

```bash
cd /home/ubuntu/TensorRT-Edge-LLM
mkdir -p build
cd build

PYBIND11_DIR=$(/home/ubuntu/JetsonVLM/.venv-jetson/bin/python \
  -m pybind11 --cmakedir)

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=12.6 \
  -DENABLE_CUTE_DSL=ALL \
  -DBUILD_PYTHON_BINDINGS=ON \
  -Dpybind11_DIR="$PYBIND11_DIR"

cmake --build . --parallel 2
```

成功标准：至少存在以下文件和程序：

```bash
cd /home/ubuntu/TensorRT-Edge-LLM
test -f build/libNvInfer_edgellm_plugin.so
test -n "$(find build/pybind -maxdepth 1 -name '*_edgellm_runtime*.so' -print -quit)"
test -x build/examples/llm/llm_build
test -x build/examples/llm/llm_inference
test -x build/examples/multimodal/visual_build
```

在 8 GB Jetson 上，固定 revision 的原始 runtime 会在加载视觉权重后才申请约 1.35 GB 共享 execution context，可能触发统一内存 OOM。应用仓库中已验证的分配顺序补丁并增量

编译 Python binding：

```bash
cd /home/ubuntu/TensorRT-Edge-LLM
git apply --check \
  /home/ubuntu/JetsonVLM/patches/tensorrt-edge-llm/preallocate-base-context-before-multimodal.patch
git apply \
  /home/ubuntu/JetsonVLM/patches/tensorrt-edge-llm/preallocate-base-context-before-multimodal.patch

/home/ubuntu/JetsonVLM/.venv-jetson/bin/cmake --build build \
  --target _edgellm_runtime -j2
```

补丁只改变共享 context 的分配顺序：先为 LLM base/decoder 分配，再加载可选视觉runner；没有改变 engine 数值计算。对已应用补丁的工作树不要重复执行 `git apply`。

## 6. Jetson：分别构建两个 engine

LLM 与视觉 engine 使用独立 flow，避免其中一个失败时覆盖另一个阶段的证据。先构建视觉 engine：

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_fp16_visual_engine.json

PYTHONPATH=src .venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_fp16_visual_engine.json \
  --execute
```

再检查并构建 LLM engine：

```bash
PYTHONPATH=src .venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_fp16_llm_engine_i768_k1024.json

PYTHONPATH=src .venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_fp16_llm_engine_i768_k1024.json \
  --execute

ln -s ../qwen3_vl_2b_fp16/visual \
  artifacts/engines/qwen3_vl_2b_fp16_i768_k1024/visual
```

两条入口都会再次核对 Edge-LLM commit，并拒绝覆盖已有 engine。成功标准：

- `artifacts/engines/qwen3_vl_2b_fp16_i768_k1024/llm/llm.engine`
- `artifacts/engines/qwen3_vl_2b_fp16/visual/visual.engine`
- LLM tokenizer/chat template/embedding 与 visual config/preprocessor sidecar
- 两个独立 flow record 的 `status` 均为 `succeeded`

2026-08-10 的实测中，视觉 flow 先生成 786 MiB `visual.engine`。启用已准备的临时8 GiB 磁盘 swap 后，独立 LLM flow 也执行成功并生成 3,453,798,316 字节的`llm.engine`：

```bash
sudo swapon --priority 1 /home/ubuntu/parksight-build.swap
swapon --show
free -h
```

该 swap 不写入 `/etc/fstab`。重复执行 `swapon` 返回 `Device or resource busy` 表示它已经启用。两个 engine 的 SHA-256、flow record 和日志均已保存；这仍是 FP16 结果，不能改称 INT4 成功。

## 7. Jetson：启动预构建 engine server

项目提供的入口只加载已有 engine，不隐式导出或重新构建：

```bash
cd /home/ubuntu/JetsonVLM
export EDGE_LLM_ROOT=/home/ubuntu/TensorRT-Edge-LLM
export PYTHONPATH=$EDGE_LLM_ROOT:$PWD/src
export EDGELLM_PLUGIN_PATH=$EDGE_LLM_ROOT/build/libNvInfer_edgellm_plugin.so
export JETSON_PY_CUDA_LIB=$PWD/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib
export PYTHONPATH=$JETSON_PY_CUDA_LIB:$EDGE_LLM_ROOT:$EDGE_LLM_ROOT/build/pybind:$PWD/src
export LD_LIBRARY_PATH=$JETSON_PY_CUDA_LIB:$EDGE_LLM_ROOT/build:$LD_LIBRARY_PATH

.venv-jetson/bin/python scripts/serve_edgellm.py \
  --engine-root artifacts/engines/qwen3_vl_2b_fp16_i768_k1024 \
  --weight-streaming-budget-bytes 0 \
  --host 127.0.0.1 \
  --port 8000
```

`--weight-streaming-budget-bytes 0` 依赖项目保存的 runtime 补丁，在创建 context 前调用 `setWeightStreamingBudgetV2(0)`。不设置预算时，TensorRT 会尝试让约 3.44 GiB streamable weights 全部驻留 GPU，并在本设备上 OOM。2026-08 的 0 预算实验完成过推理，但 2026-10-08 当前 runtime 的隔离复验在视觉 engine 加载时仍出现 OOM；0 预算不保证整个 VLM 能加载。保留两次实验身份，下一步定位初始化峰值，详情见 [板端就绪记录](jetson-readiness.md)。现有 INT4 engine 未按 weight streaming 构建，启动时不要传该参数。

新指标验收可在上述启动命令中增加 `--stream-usage`，并通过
`--runtime-metadata-output reports/runtime/新运行名.runtime.json` 保存包装代码身份。
该开关默认关闭，仅包装服务进程内的 SSE generator，不修改 Edge-LLM checkout。
完整计数边界见 [运行时指标](runtime-performance.md)。开启前先将新增入口及 Module
部署到独立板端 checkout，不覆盖已有未提交工作树。

另开一个 Jetson SSH 终端验证：

```bash
curl --fail http://127.0.0.1:8000/health
```

成功标准：HTTP 连接成功并返回健康状态。server 是实验性接口，因此必须保留固定Edge-LLM commit，不能把另一个版本的 server 与当前 engine 混用。

## 8. 单图与 PS2.0 Study

先选择 PS2.0 pilot 中的一张图片进行单图验收：

```bash
cd /home/ubuntu/JetsonVLM
PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.analyze_image \
  --image data/raw/ps2.0/pilot/indoor/001.jpg \
  --runtime tensorrt_edge_llm_http \
  --backend-revision 7f061f21f0a581ba234a1e233c9315b89d8e47d6 \
  --model-revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --adapter-revision edge-http-system-content-array-v1 \
  --precision fp16 \
  --edge-url http://127.0.0.1:8000
```

实际图片路径以 manifest 的 `image_ref` 为准。验收分为两层：HTTP 200、token 和`raw_output` 证明后端真实执行；只有存在通过严格 schema 的 `assessment` 才证明业务输出成功。2026-08-10 实测完成 HTTP 200 和 57 token，但字段类型错误，因此业务层如实记录为 `json_parse_error`。

单图成功后再运行完整 pilot：

```bash
PYTHONPATH=src .venv-jetson/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_edgellm_fp16_ps20_pilot_promptfix.json
```

最终报告为 `reports/jetson_edgellm_fp16_ps20_pilot_promptfix_i768_k1024.json`。只有该报告与`reports/jetson_transformers_fp16_ps20_pilot.json` 使用相同 workload identity、样本和功耗模式时，才计算加速比和质量差异。

2026-08-12 使用 prompt 修复、runtime 分配顺序补丁和 768/1024 engine 重新实测：20/20 样本完成后端执行并通过严格 JSON，失败汇总为空；风险等级准确率 35%，事件micro-F1 0.359，不安全建议率 0%。端到端 p50/p90/p99 为 50.75/69.08/75.08 秒，平均输入功耗 10.05 W，GPU 峰值温度 65.03°C，RAM/swap 峰值 7418/1904 MB。

可复现摘要命令：

```bash
PYTHONPATH=src python scripts/summarize_jetson_study.py \
  --study-report reports/jetson_edgellm_fp16_ps20_pilot_promptfix_i768_k1024.json \
  --tegrastats reports/runtime/tegrastats_promptfix_i768_k1024_20sample_20260812.log \
  --output reports/jetson_edgellm_fp16_ps20_pilot_promptfix_i768_k1024.runtime-summary.json
```
