# WSL vLLM 本机部署与验收

本流程使用 RTX 4060 Laptop 8 GB，在 WSL 的 Ubuntu 22.04 上运行
`Qwen/Qwen3-VL-2B-Instruct`，通过仓库的 `VllmHttpBackend` 执行单图风险理解。
本机服务验证通过后，再开展 Jetson 与 WSL 的两机联调。

## 独立环境

在仓库根目录执行。vLLM 使用 `.venv-vllm`，项目的无硬件开发环境使用 `.venv`。
移动仓库后，应重新执行 editable 安装，避免 `.pth` 仍指向原工作目录。
同时检查激活脚本里的 `VIRTUAL_ENV` 是否仍引用旧路径；直接运行
`.venv-vllm/bin/python` 不依赖激活脚本。

```bash
uv venv .venv-vllm --python /usr/bin/python3
uv pip install --python .venv-vllm/bin/python -r requirements/vllm-wsl.lock.txt
uv pip install --python .venv-vllm/bin/python --no-deps -e .
```

`requirements/vllm-wsl.txt` 保存直接依赖，`.lock.txt` 保存完整安装版本。
固定 vLLM 0.11.2、Transformers 4.57.1 与 PyTorch 2.9.0（CUDA 12.8 wheel）。
这是本项目的一组部署候选，不代表最新版本或性能最优版本。
环境与设备快照保存在本地 `reports/vllm-wsl/`。

Triton 首次运行需要 GCC 与当前 Python 的开发头文件。若缺少 `Python.h`，
可以安装 `python3.10-dev`；没有 sudo 时，本机使用以下用户目录方案：

```bash
mkdir -p artifacts/vllm-python-headers
cd artifacts/vllm-python-headers
apt-get download libpython3.10-dev
dpkg-deb --extract libpython3.10-dev_*.deb .
cd ../..
```

启动脚本会检测这里的头文件并设置 `C_INCLUDE_PATH`。下载完成后再解包。
本机包版本为 `3.10.12-1~22.04.18`，匹配系统 Python 3.10.12；在其他发行版或
Python 环境中应核对对应版本，不能直接复用这组头文件。

## 模型身份

复用 `models/Qwen3-VL-2B-Instruct-89644892/`，不重复下载权重。
模型 revision 为 `89644892e4d85e24eaac8bacfd4f463576704203`。
目录名和客户端的 `model_revision` 字段不能证明实际权重身份。
首次部署时将本地文件与该 revision 的 Hugging Face 元数据逐项核对：
权重使用 LFS SHA-256，普通配置文件使用 Git blob SHA-1，并保存本地 SHA-256。
验证结果保存在 `reports/vllm-wsl/model_identity.json`。

## 启动服务

```bash
bash scripts/serve_vllm_wsl.sh
```

脚本使用本地模型、BF16、2048 token 上下文、单请求、1024 token 调度预算、
85% GPU 显存预算和 eager 模式。只允许一张图片，不启用视频。
使用 `--generation-config vllm` 避免继承模型仓库的采样默认值；客户端按冻结
workload 发送 `temperature=0` 和 `max_tokens=256`。
服务端预处理像素预算为 200704（448 × 448）；处理器保持收到图片的比例并按
模型 patch 要求调整尺寸。历史原图模式不能表述为固定 448 × 448 输入。
启动参数、初始化时间、显存和请求时延分别记录。

服务默认绑定 `127.0.0.1:8000`，关闭 Hugging Face 联网加载与 vLLM 使用统计。
不要仅以 `/health` 成功作为多模态验收依据。

## 仓库链路验收

在另一终端运行：

```bash
.venv-vllm/bin/python scripts/benchmark_vllm_http.py \
  --config configs/studies/wsl_vllm_workload_resize_ps20.json \
  --gpu-telemetry \
  --metrics-url http://127.0.0.1:8000/metrics
```

报告写入 `reports/vllm-wsl/workload_resize_ps20.json`，保存逐样本原始输出、严格 schema
校验、失败分类、客户端 TTFT、输出 token 数和质量指标。
退出码 0 表示没有执行或输出校验失败；任务质量仍须读取质量指标。
退出码 2 表示报告中存在失败，应检查逐样本记录，不能直接视为服务不可用。
benchmark 先独立预热再测量，并保存 `.warmup.json`、`.execution.json` 和
`.gpu.csv`。执行证据包含客户端源码、配置、运行时选项与数据元信息哈希。
GPU 遥测仅适用于服务就在本机的情况。已有证据时拒绝覆盖；复验需使用
`--output reports/vllm-wsl/新的文件名.json`。
客户端 TPOT、流式片段间隔、墙钟吞吐及服务端 `/metrics` 的具体口径参见
[运行时性能测量](runtime-performance.md)。

## 输入预处理口径

`runtime.options.image_preprocessing` 支持两个模式：

- `source`：默认的历史兼容模式，上传原图，由服务器处理器调整尺寸。
- `workload_resize`：按 workload 转换 RGB、bicubic 缩放并上传无损 PNG，
  与 Transformers 共用客户端图像预处理函数。新研究应显式使用此模式。

本机 workload 为 448 × 448，20 张实际 PNG 上传均验证了尺寸、RGB 模式及
解码后像素与共享预处理结果一致。服务端 processor 仍负责 patch 和张量转换；
这不等于已证明跨框架视觉张量或 Jetson engine 输入完全一致。
两种模式应使用不同 study/report，不能混合比较历史质量和时延。

新口径在 eager 模式下完成 20 次独立预热与 20 次正式请求：20/20 成功，
JSON 有效率 100%，风险等级准确率 0.350，事件 micro-F1 0.350，语义一致率
0.800，E2E p50 / p90 / p99 为 1.482 / 2.250 / 2.506 秒。
仍有 narrow_passage 误报 12 次、visibility_occlusion 漏报 9 次，
输入契约修正不代表任务质量已达到可用标准。
证据为 `workload_resize_ps20.json` 及相应 sidecar，像素验证记录保存在
`workload_resize_image_identity.json`。

## CUDA Graph 候选

```bash
bash scripts/serve_vllm_wsl_graph.sh
```

该脚本只捕获 batch=1 的 decode，保留 eager prefill，不启用 Inductor。
同一 GPU 的 eager → Graph → eager 复验，每轮先预热 20 次，再测量 60 次。
以下三轮使用历史 `source` 输入口径，全部成功，逐样本 assessment 一致：

| 模式 | E2E p50 / p90 / p99（秒） | 设备显存采样峰值 |
| --- | --- | --- |
| eager 首轮 | 1.467 / 2.028 / 2.175 | 6942 MiB |
| decode Graph | 1.332 / 1.664 / 1.968 | 6992 MiB |
| eager 复验 | 1.599 / 2.107 / 2.463 | 6942 MiB |

Graph 的 p50 较两轮 eager 分别降低 9.2% 和 16.7%。笔记本功耗、温度未受控，
该结果是本机候选配置证据，不作为通用加速比或 Jetson 结论。
默认启动脚本继续采用 eager；Graph 仍需更长稳定性验证。
证据：本地 `graph_comparison.json`、`graph_repeat3.json`、
`eager_repeat3.json`、`eager_recheck.json` 及相应日志和 GPU 遥测。

## HTTP 与计时可靠性

SSE 验证 UTF-8 增量解码、终止帧、响应结构和 token 数；半截响应不能当作成功。
模型拒答、HTTP 协议错误与业务 JSON 校验失败分别记录。连接异常会关闭连接，
下一次请求重新连接，不自动重试已发出的 POST。
`timeout_seconds` 是连接/读等待超时，尚不是整个请求的绝对时间上限。
首请求时延保存在 `first_request_ms`，缺少独立初始化测量时
`cold_start_ms` 保持 null；旧报告的历史字段不回写。

PS20 是 pilot 数据集。本机时延包含 HTTP 与服务开销，未控制笔记本功耗状态，
不作为 Jetson 框架 A/B 或泛化能力结论。端云路由、网络中断和服务退出回退
需在后续两机研究中独立验收。

## 无硬件回归

```bash
uv run python -m unittest discover -s tests
git diff --check
```

## 2026-10-04 本机实测

WSL 2 / Ubuntu 22.04.5、Python 3.10.12、RTX 4060 Laptop 8188 MiB，
Windows 驱动 592.82。CUDA/BF16 矩阵运算通过，固定 revision 的 10 个本地
模型文件均与远端元数据匹配。服务加载 Qwen3VLForConditionalGeneration，
使用 FLASH_ATTN；首次启动缺少 Python.h，采用上述用户目录头文件方案后恢复。

| 项目 | 实测结果 |
| --- | --- |
| 本机服务与单图请求 | `/health`、`/v1/models`、图像 data URI、SSE 与严格 schema 校验通过 |
| PS20 请求 / JSON 有效率 | 20/20 成功，100% |
| 风险等级准确率 / 事件 micro-F1 | 0.350 / 0.316 |
| 不安全建议率 / 语义一致率 | 0.000 / 0.750（仓库指标口径） |
| E2E p50 / p90 / p99 | 1.428 / 1.869 / 2.329 s |
| 客户端 TTFT p50 | 99.6 ms |
| 连续请求复验 | 10/10 成功；仅为短期复验 |
| 1 秒采样的设备显存峰值 / 温度峰值 | 6942 MiB / 79 °C |
| 1 秒采样的设备功耗峰值 | 74.61 W |
| 真实客户端超时 | 1 ms 请求超时被记录为 `timeout` |
| 服务退出后的请求 | 连接拒绝被记录为 `runtime_error`，assessment 为 null |
| 无硬件回归 | 264 项通过；源码编译与 `git diff --check` 通过 |

PS20 之前已执行单图预热，报告的 `cold_start_ms` 是本次 study 首个请求，
不能作为服务冷启动时间。GPU 遥测是设备级的 1 秒采样结果，独立于逐请求
`ResourceSnapshot`；未获取的阶段计时、逐请求资源和纯 decode tokens/s 保持 null。

主要任务错误为 narrow_passage 误报 11 次、visibility_occlusion 漏报 9 次。
这些结果建立了 vLLM 本机链路的执行证据，后续仍需改善数据与任务质量。

原始证据位于本地 `reports/vllm-wsl/`：`model_identity.json`、
`execution_identity.json`、`environment.json`、`server.log`、`smoke.json`、
`ps20_pilot.json`、`study_gpu.csv`、`soak.json`、`timeout.json`、
`service_unavailable.json` 和 `summary.json`。
该目录被 Git 忽略；迁移或共享实测时应单独保留。
本轮验收结束后已关闭临时服务，显存回落至 44 MiB；需要使用时运行启动脚本。
