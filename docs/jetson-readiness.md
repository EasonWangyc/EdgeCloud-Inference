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
模型权重来源和实际图片身份仍需独立核验。

## 下一步验收

1. 核对图片、权重、engine、插件和 binding 的文件哈希及实际依赖版本。
2. 核对构建日志、命令和源码补丁。旧 flow record 仅记录输出存在和成功状态，
   未绑定 engine SHA-256，不能证明当前文件就是当时产物。
3. 选定已有产物后，以新输出目录执行单图验收，再开展 Jetson Transformers FP16
   与 Edge-LLM 同机研究。TTFT、decode、端到端时延和失败记录采用
   [统一指标口径](runtime-performance.md)。

只读证据保存在本地 `reports/jetson-readiness-20261008/environment.json`
和 `fp16_snapshot.json`。`reports/` 被 Git 忽略，原始证据需单独归档。
