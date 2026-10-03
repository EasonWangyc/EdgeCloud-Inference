# JetsonVLM项目学习记录

## 1. Jetson板端环境检查/依赖安装/版本更新

### 1.1 板端环境检查

查看开发套件版本：

```bash
cat /proc/device-tree/model
```

```text
NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super
```

查看 Jetson 系统版本：

```bash
cat /etc/nv_tegra_release
```

```text
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia
```

查看cuda版本和nvcc版本：

```bash
/usr/local/cuda/bin/nvcc --version
```

```text
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Wed_Aug_14_10:14:07_PDT_2024
Cuda compilation tools, release 12.6, V12.6.68
Build cuda_12.6.r12.6/compiler.34714021_0
```

查看JetPack版本和L4T版本：

```bash
apt-cache policy nvidia-jetpack nvidia-l4t-core
```

```text
nvidia-jetpack:
  已安装：6.2.1+b38
  候选： 6.2.1+b38
  版本列表：
 *** 6.2.1+b38 600
        600 https://repo.download.nvidia.com/jetson/common r36.4/main arm64 Packages
        100 /var/lib/dpkg/status
     6.2+b77 600
        600 https://repo.download.nvidia.com/jetson/common r36.4/main arm64 Packages
     6.1+b123 600
        600 https://repo.download.nvidia.com/jetson/common r36.4/main arm64 Packages
nvidia-l4t-core:
  已安装：36.4.7-20250918154033
  候选： 36.4.7-20250918154033
  版本列表：
 *** 36.4.7-20250918154033 600
        600 https://repo.download.nvidia.com/jetson/t234 r36.4/main arm64 Packages
        100 /var/lib/dpkg/status
     36.4.4-20250616085344 600
        600 https://repo.download.nvidia.com/jetson/t234 r36.4/main arm64 Packages
     36.4.3-20250107174145 600
        600 https://repo.download.nvidia.com/jetson/t234 r36.4/main arm64 Packages
     36.4.0-20240912212859 600
        600 https://repo.download.nvidia.com/jetson/t234 r36.4/main arm64 Packages
```

查看功率模式：

```bash
nvpmodel -q
```

```text
NV Power Mode: 15W
0
```

查看系统target：

```bash
systemctl get-default
```

```bash
graphical.target
```

查看硬件相关资源：

```bash
free -h
```

```text
               total        used        free      shared  buff/cache   available
Mem:           7.4Gi       1.5Gi       4.6Gi        45Mi       1.4Gi       5.7Gi
Swap:          3.7Gi          0B       3.7Gi
```

```bash
swapon --show --bytes
```

```text
NAME       TYPE           SIZE USED PRIO
/dev/zram0 partition 665829376    0    5
/dev/zram1 partition 665829376    0    5
/dev/zram2 partition 665829376    0    5
/dev/zram3 partition 665829376    0    5
/dev/zram4 partition 665829376    0    5
/dev/zram5 partition 665829376    0    5
```

**Swap（交换空间）** 是指系统将存储空间（如固态硬盘或内存压缩区）模拟当作**虚拟内存**使用的一种技术。当物理内存（RAM）快要用完时，系统会把不常用的数据暂时移到 Swap 里面，从而防止程序因为内存不足而崩溃。

```bash
df -h
```

```text
Filesystem       Size  Used Avail Use% Mounted on
/dev/nvme0n1p1   233G   41G  181G  19% /
tmpfs            3.8G  172K  3.8G   1% /dev/shm
tmpfs            1.5G   43M  1.5G   3% /run
tmpfs            5.0M  4.0K  5.0M   1% /run/lock
/dev/nvme0n1p10   63M  110K   63M   1% /boot/efi
tmpfs            762M  120K  762M   1% /run/user/1000
```

Jetson 板端专用Pytorch环境检查：

板端torch不能通过常规方式下载，需通过Jetson PyTorch wheel下载。

```bash
cd ~/JetsonVLM
.venv-jetson/bin/python -m pip list
```

```text
Package            Version
------------------ -------------
accelerate         1.12.0
certifi            2026.7.22
charset-normalizer 3.4.9
filelock           3.32.0
fsspec             2026.6.0
hf-xet             1.5.2
huggingface-hub    0.36.0
idna               3.18
Jinja2             3.1.6
MarkupSafe         3.0.3
mpmath             1.3.0
networkx           3.4.2
numpy              2.2.6
nvidia-cudss-cu12  0.4.0.2.post1
packaging          26.2
pillow             12.1.0
pip                26.1.2
psutil             7.2.2
PyYAML             6.0.3
regex              2026.7.19
requests           2.34.2
safetensors        0.8.0
setuptools         83.0.0
sympy              1.14.0
tensorrt           10.3.0
tensorrt_dispatch  10.3.0
tensorrt_lean      10.3.0
tokenizers         0.22.2
torch              2.9.1
torchvision        0.24.1
tqdm               4.70.0
transformers       4.57.6
typing_extensions  4.16.0
urllib3            2.7.0
wheel              0.47.0
```

其中，Transformer相关依赖通过板端UV下载。

### 1.2 更新JetPack和L4T

第一次运行Transformer时发现当前 torch 来自 PyTorch CUDA 12.6 通用`cp312-manylinux_2_28_aarch64` wheel。它能发现 Orin，但不包含 Orin 所需的 `sm_87`。目标 JetPack 6.2.2 对应 L4T R36.5，必须先切换软件源。

```bash
# 备份软件源
sudo cp -a /etc/apt/sources.list.d/nvidia-l4t-apt-source.list /etc/apt/sources.list.d/nvidia-l4t-apt-source.list.pre-r36.5
# 将36.4改为36.5
sudo sed -i 's/r36\.4/r36.5/g' /etc/apt/sources.list.d/nvidia-l4t-apt-source.list
# 确认结果，预期输出：common/t234/ffmpeg r36.5 main
cat /etc/apt/sources.list.d/nvidia-l4t-apt-source.list 
# 更新apt
sudo apt update
# 智能升级软件包
sudo apt dist-upgrade
```

## 2. 项目记录

### 2.1 评测链路搭建

评测链路总览：

```text
ParkingCase -> RiskRuntime -> InferenceRecord -> StudyReport
```

**模型输出、人工标注、评测标准使用同一套 JSON 契约（ParkingAssessment）**。该字段为模型输出 / 人工标注共用契约。

| 字段 | 类型 | 含义 | 校验规则 |
| --- | --- | --- | --- |
| `schema_version` | string | schema 版本标识 | 必须等于 `parking_risk_v1` |
| `risk_level` | string 枚举 | 总体风险等级 | 只能是 `low` / `medium` / `high` |
| `events` | string 数组 | 识别出的风险事件（可空） | 必须数组、元素在 6 个枚举内、不重复 |
| `evidence` | string 数组 | 图像证据（可见线索） | 必须非空数组、不重复 |
| `driver_advice` | string 数组 | 驾驶建议 | 必须非空数组、元素在 5 个枚举内、不重复 |

6 个风险事件枚举：`vru_near_maneuver_path`（行人接近路径）/ `vehicle_near_maneuver_path`（车辆接近路径）/`fixed_obstacle_near_path`（固定障碍物）/`narrow_passage`（狭窄通道）/ `visibility_occlusion`（可见性遮挡）/`parking_space_conflict`（车位冲突）。

5 个驾驶建议枚举：`maintain_observation`（保持观察）/ `slow_down`（减速）/ `yield`（让行）/`prepare_to_stop`（随时准备停车，注意，不是泊车）/`change_maneuver_when_safe`（安全时改变操作）。

#### 数据文件 JSON 格式

**manifest**（样本清单，`data/manifests/ps20_pilot_v1.jsonl`），每行一个样本：

```json
{"case_id":"ps20-indoor-001","image_ref":"raw/ps2.0/pilot/indoor/001.jpg","source_group_id":"ps2.0-testing-indoor","split":"test"}
```

| 字段 | 含义 |
| --- | --- |
| `case_id` | 样本唯一 ID |
| `image_ref` | 图片相对 `data/` 的路径 |
| `source_group_id` | 来源组（防泄漏划分） |
| `split` | 数据集划分：train / validation / test |

**annotation**（人工标注，`data/annotations/ps20_pilot_v1.jsonl`）：

```json
{"case_id":"ps20-indoor-001","assessment":{"schema_version":"parking_risk_v1","risk_level":"medium","events":["narrow_passage"],"evidence":["自车左右两侧均有近距离停放车辆，可见横向通行空间较窄。"],"driver_advice":["slow_down","maintain_observation"]}}
```

`RiskRuntime.analyze()` 执行一个样本后产出，关键约束：`assessment` 与 `failure` **恰好二选一**。

主要字段：`case_id` / `runtime_identity`（backend、model_id、model_revision、adapter_revision、precision）/
`workload_identity`（SHA-256）/ `assessment` | `failure` / `raw_output`（模型原始输出，留痕复盘）/
`stage_timings`（preprocess、model_generate、end_to_end 等，未测为 null）/ `resource_snapshot`（峰值显存）/ `output_tokens`。

失败分类（`_classify_failure`）：`json_parse_error` / `dependency_unavailable` / `out_of_memory` /
`timeout` / `model_refusal` / `unsupported_operator` / `runtime_error` / `input_error`。

`StudyRunner.run()` 聚合所有记录：

| 字段 | 含义 |
| --- | --- |
| `study_identity` | 实验完整身份（study_id + workload + runtime + split + 功耗模式） |
| `environment_snapshot` | 环境快照（L4T、Python、torch/transformers 版本） |
| `quality_metrics` | 质量指标（对照人工标注） |
| `performance_metrics` | 性能指标（**只统计成功记录**） |
| `failure_summary` | 失败类别计数 |
| `records` | 全部原始 InferenceRecord，一条不删 |

质量指标：`json_validity_rate`（JSON 有效率）/ `risk_level_accuracy`（风险等级准确率）/
`event_micro_f1`（事件 micro-F1，六类事件累计 TP/FP/FN）/ `unsafe_advice_rate`（不安全建议率）/
`event_errors`（逐事件 FP/FN）。

性能指标：`cold_start_ms`（仅在单独 warm-up 证据存在时记录服务 ready 后首个请求）/
`stage_latency_ms`（每阶段 p50/p90/p99）/
`tokens_per_second` / `peak_memory_mb` / `average_power_w` / `peak_temperature_c`。

#### 评测标准

本项目的评测标准如下：

| 概念             | 含义                     | 本项目的例子                                      |
| ---------------- | ------------------------ | ------------------------------------------------- |
| **TP（真阳性）** | 标注有事件，模型也预测有 | 标注有 `narrow_passage`，模型也标了 ✅             |
| **FP（假阳性）** | 标注没有，模型却预测有   | 标注没有，模型标了 `narrow_passage` ❌（过度报警） |
| **FN（假阴性）** | 标注有，模型漏掉了       | 标注有 `narrow_passage`，模型没标 ❌（漏报）       |

```
precision = TP / (TP + FP)   预测为有的事件里，有多少是真的
recall    = TP / (TP + FN)   真有的事件里，模型找回了多少
F1        = 2·P·R / (P+R)    两者的调和平均
```

最终使用`micro-F1`，即对所有类别的TP/FP/FN累加，最后计算得到。

#### 运行前测试

随后冷测试run1-run6，属于基本链路测试，其中遇到了不少问题，如格式不支持fp16、未下载Torchvision包、sm不匹配、JetPack版本错误等。

| run      | 环境 / 修改                        | 结果                                | 得到的结论                                   |
| -------- | ---------------------------------- | ----------------------------------- | -------------------------------------------- |
| 入口初测 | 原 CLI                             | `--dtype` 等参数无法识别            | 增加参数并传入 runtime options               |
| run1     | 原 Python 3.12 通用 torch          | NvMap error 12，约 15.25 s          | 此结果混有错误 torch wheel，不能判断模型容量 |
| run2     | 增加 CUDA arch 预检                | `dependency_unavailable`，约 7.09 s | 明确发现 torch 不含 `sm_87`                  |
| run3     | 新 `.venv-jetson`                  | 缺少 torchvision，12.26 s           | 安装 Jetson torchvision 0.24.1               |
| run4     | 依赖补齐                           | NvMap/NVML allocator error，12.15 s | 权重加载阶段连续大块分配失败                 |
| run5     | `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | `out_of_memory`，10.53 s            | 不是单纯 caching allocator 预留导致          |

升级到R36.5.0之后：

| run  | 结果                                             | 原因 / 修复                                                  |
| ---- | ------------------------------------------------ | ------------------------------------------------------------ |
| run6 | `runtime_error: string indices must be integers` | Qwen3-VL 多模态 system message 结构不符合 processor 期望     |
| run7 | 模型已生成 JSON，但严格解析失败                  | `events` 使用中文自由文本，`evidence`/`driver_advice` 还是字符串 |
| run8 | `failure=null`，严格 JSON 成功                   | 修复消息格式并加强 prompt 的英文 snake_case 枚举约束         |

> run6针对消息结构的关键修复：

```text
system.content = [{"type": "text", "text": ...}]
user.content = [
    {"type": "image", "image": image},
    {"type": "text", "text": ...},
]
```

几个关键设计：

1. 懒加载：`_ensure_loaded()`只在首次`generate`时才加载模型，CLI启动、配置校验、无硬件测试都不需要模型权重
2. `_require_cuda_architecture`进行CUDA架构预检

```python
required = f"sm_{major}{minor}"   # Orin Nano → sm_87
if required not in torch.cuda.get_arch_list():
    probe = torch.ones(1, device="cuda"); probe.add_(1)  # 真实 kernel 试跑
```

这里发现了通用aarch64 torch wheel只包含`sm_80/sm_90`，而实际需要`sm_87`。

3. 冻结workload：`max_new_tokens=256`、`do_sample=False`（贪心解码，可复现）、输入 448×448、prompt 严格约束输出 JSON 枚举等

### 2.2 板端基线FP16 VL Transformer推理实现

#### 数据集准备

面向泊车场景，主要调研了三个数据集：

| 数据集 | 描述 | 限制 |
| --- | --- | --- |
| WoodScape | 真实车载四向鱼眼，人、车、障碍、遮挡 | 不专注停车场，数据许可为 proprietary |
| Tongji PS2.0 | 四鱼眼拼接的 AVM 鸟瞰图、停车位 | 缺少通用障碍和风险标签 |
| nuScenes | 包含大量汽车在城市道路和停车场进行倒车、掉头、泊车时的同步后视图像 | 不专注停车场景 |

使用[Tongji PS2.0](https://cslinzhang.github.io/deepps/)数据集，并选取20pilot样本进行手动标注，结果见[对应jsonl文件](../data/annotations/ps20_pilot_v1.jsonl)。

#### HuggingFace模型下载

通过SHA下载固定commit版本的模型，保证服务器端、Jetson Transformer、Jetson TensorRT模型权重相同。

```bash
/home/ubuntu/project/llm-on-device/.venv/bin/hf download \
  Qwen/Qwen3-VL-2B-Instruct \
  --revision 89644892e4d85e24eaac8bacfd4f463576704203 \
  --cache-dir /home/ubuntu/.cache/huggingface/hub
```

结果如下：

- 模型：`Qwen/Qwen3-VL-2B-Instruct`
- 固定 commit：`89644892e4d85e24eaac8bacfd4f463576704203`
- snapshot：
  `/home/ubuntu/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203`
- 仓库缓存占用：约 4.0 GiB

##### Qwen3-VL-2B-Instruct模型结构

Qwen3-VL 是**混合架构 VLM**：一个视觉塔（ViT）+ 一个 LLM 骨干，中间通过视觉-文本对齐层融合。总参数 **21.34 亿**（项目实测 `2133954560`）。

```text
                     ┌─────────────────────────────────────┐
                     │          Qwen3-VL-2B 整体            │
                     │                                     │
 图片 ──► 视觉塔(ViT) ──► 视觉 token ──┐                    │
             24 层                     │  融合               │
             hidden 1024               ├─► LLM 骨干 (28 层) ─► 文本输出
             patch 16×16               │  hidden 2048        │
                                      └─► 文本 token ──┘     │
                     └─────────────────────────────────────┘
```

视觉塔（Vision Tower）—— `vision_config`

| 参数                       | 值        | 含义                                       |
| -------------------------- | --------- | ------------------------------------------ |
| `depth`                    | 24        | 24 层 Transformer                          |
| `hidden_size`              | 1024      | 每层隐藏维度                               |
| `num_heads`                | 16        | 注意力头数                                 |
| `patch_size`               | 16        | 图片切成 16×16 patch                       |
| `temporal_patch_size`      | 2         | 视频时间维采样（本项目只用单图）           |
| `spatial_merge_size`       | 2         | **视觉 token 合并**（下面重点讲）          |
| `out_hidden_size`          | 2048      | 输出维度，对齐到 LLM 的 2048               |
| `deepstack_visual_indexes` | [5,11,17] | 从第 5/11/17 层取**分层特征**（DeepStack） |

LLM 骨干

| 参数                      | 值     | 含义                                                         |
| ------------------------- | ------ | ------------------------------------------------------------ |
| `num_hidden_layers`       | 28     | **28 层** decoder（项目实测：LoRA 注入"28 层 self-attention"；32 个 AttentionPlugin 是含视觉等插件的总数） |
| `hidden_size`             | 2048   | 隐藏维度                                                     |
| `intermediate_size`       | 6144   | FFN 中间维度（3× hidden）                                    |
| `num_attention_heads`     | 16     | Q 头数                                                       |
| `num_key_value_heads`     | 8      | **KV 头数 = 8（GQA）**                                       |
| `head_dim`                | 128    | 每头维度（项目实测 FMHA 用 head_size=128 的关键！）          |
| `vocab_size`              | 151936 | 词表                                                         |
| `max_position_embeddings` | 262144 | 极长上下文支持                                               |
| `tie_word_embeddings`     | true   | **embedding 与 lm_head 共享权重**                            |
| `attention_bias`          | false  | 注意力无 bias                                                |

#### 冻结workload

配置实验身份的入口是[run_study.py](../src/parksight_vlm/app/run_study.py)，其中调用了`FrozenWorkload.load()`，在[workload.py](../src/parksight_vlm/workload.py)中冻结`prompt`、`448×448输入尺寸`、`max_new_tokens=256`、`do_sample=false`、`6 个风险事件`、`5 个驾驶建议`、`schema version`等。

#### 板端推理

`transformers.py`的`_ensure_loaded()` 加载对应模型和固定的commit：

```python
self._processor = AutoProcessor.from_pretrained(
    self._model_id,              # "Qwen/Qwen3-VL-2B-Instruct"
    revision=self._model_revision # 固定 commit，命中本地 snapshot
)
self._model = Qwen3VLForConditionalGeneration.from_pretrained(
    self._model_id,
    revision=self._model_revision,
    device_map="auto",           # 整模型映射到 cuda:0（实测 hf_device_map = {'': 0}）
    dtype="float16",             # FP16 精度
    attn_implementation="sdpa",  # SDPA 注意力省显存
)
```

配置`jetson_transformers_fp16_ps20_pilot.json`的`runtime`字段，固定模型信息：

```json
"runtime": {
  "backend": "transformers",
  "backend_revision": "transformers==4.57.6",
  "model_id": "Qwen/Qwen3-VL-2B-Instruct",
  "model_revision": "89644892e4d85e24eaac8bacfd4f463576704203",
  "adapter_revision": "none",
  "precision": "fp16",
  "options": {
    "device_map": "auto",
    "dtype": "float16",
    "attn_implementation": "sdpa"
  }
}
```

执行推理：

```bash
# 冻结20样本study
cd /home/ubuntu/JetsonVLM

LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
timeout 1800s \
.venv-jetson/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/jetson_transformers_fp16_ps20_pilot.json
```

运行结果结果如下：

| 指标 | 结果 |
| --- | ---: |
| 样本数 | 20 |
| JSON 有效率 | 1.000 |
| 风险等级准确率 | 0.350 |
| 事件 micro precision | 0.280 |
| 事件 micro recall | 0.4375 |
| 事件 micro F1 | 0.3415 |
| 不安全建议率 | 0.000 |
| 运行时失败 | 0 |

主要误差：20 张全部预测为 `low`；18 张预测 `narrow_passage`，7 张额外预测`vehicle_near_maneuver_path`，没有预测 `visibility_occlusion`。这表明 JSON 契约已经成立，但业务判断质量较低。

| 指标 | 结果 |
| --- | ---: |
| 冷启动端到端 | 31,081.31 ms |
| 预处理 p50 / p90 / p99 | 27.50 / 31.30 / 86.93 ms |
| 模型生成 p50 / p90 / p99 | 9,354.70 / 14,086.91 / 14,972.91 ms |
| 端到端 p50 / p90 / p99 | 9,384.95 / 14,117.36 / 27,939.48 ms |
| 进程峰值内存 | 4,176.72 MB |
| 系统 RAM 峰值 | 6,398 MB |
| swap 峰值 | 501 MB |
| GPU 利用率峰值 | 99% |
| GPU 温度峰值 | 61.781 °C |
| `VDD_IN` 瞬时峰值 | 12.655 W |
| `VDD_IN` 最终区间平均 | 10.115 W |

### 2.3 TensorRT-edge-llm部署

#### GPU服务器导出TensorRT Edge-LLM ONNX

选择[[AutoDL](https://www.autodl.com/market/list)]平台的单卡RTX 4090 D作为服务器端，具体环境如下：

| 项目                              | 实测值                                                    |
| --------------------------------- | --------------------------------------------------------- |
| 系统                              | Ubuntu 22.04.5 LTS，x86-64                                |
| GPU                               | NVIDIA GeForce RTX 4090 D，24 GiB                         |
| 驱动 / `nvidia-smi` CUDA          | 595.71.05 / 13.2                                          |
| 导出 Python                       | 3.12.3，独立 `.venv-export`                               |
| PyTorch                           | `2.12.0+cu126`                                            |
| TensorRT Edge-LLM                 | v0.9.1，commit `7f061f21f0a581ba234a1e233c9315b89d8e47d6` |
| Transformers / ONNX / ONNX Script | 5.9.0 / 1.19.0 / 0.7.0                                    |

```bash
# 固定源码和模型
cd /root/autodl-tmp
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout --detach 7f061f21f0a581ba234a1e233c9315b89d8e47d6 # 该revision是单个model.safetensors
git submodule update --init --recursive
git rev-parse HEAD
# dry-run与导出
export PYTHONPATH=/root/autodl-tmp/JetsonVLM/src
export PATH=/root/autodl-tmp/TensorRT-Edge-LLM/.venv-export/bin:$PATH

python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_fp16.json

python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_fp16.json \
  --execute
```

导出耗时约为3分钟，输出包含 LLM 和视觉编码器的 ONNX、external data、tokenizer/chat template、视觉preprocessor/config 等共 11 个文件。

```text
llm/
  model.onnx                    ← LLM 主干图结构（小，几 MB）
  model.onnx.data               ← LLM 权重 external data（大，~3.4 GB）★
  config.json                   ← LLM 配置
  embedding.safetensors         ← embedding 权重单独导出（~620 MB）
  tokenizer.json                ← 分词器
  tokenizer_config.json         ← 分词器配置
  processed_chat_template.json  ← 已处理的 chat 模板（engine 推理时用）
visual/
  model.onnx                    ← 视觉编码器图结构（小）
  model.onnx.data               ← 视觉编码器权重 external data（~786 MB）
  config.json                   ← 视觉配置
  preprocessor_config.json      ← 视觉预处理配置（归一化/resize 参数）
```

Qwen3-VL-2B-Instruct的总参数量为21.34亿，FP16导出大小约为4.27 GB。

#### Jetson编译TensorRT Edge-LLM

**engine 与 GPU 架构、TensorRT 版本、插件、构建参数强绑定**，因此编译构建需要在板端进行。

```bash
# Jetson板端， Edge-LLM v0.9.1
cd /home/ubuntu
git clone https://github.com/NVIDIA/TensorRT-Edge-LLM.git
cd TensorRT-Edge-LLM
git checkout --detach 7f061f21f0a581ba234a1e233c9315b89d8e47d6
git submodule update --init --recursive
git rev-parse HEAD   # 必须输出 7f061f21f0a581ba234a1e233c9315b89d8e47d6
```

仓库上游依赖NVTX、googletest、nlohmann/json三个子模块，从PC传输；CMake和Pybind11从`.venv-jeston`中安装。

通过CMake构建和编译：

```bash
cd /home/ubuntu/TensorRT-Edge-LLM
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRT_PACKAGE_DIR=/usr \
  -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64_linux_toolchain.cmake \
  -DEMBEDDED_TARGET=jetson-orin \
  -DCUDA_CTK_VERSION=12.6 \
  -DENABLE_CUTE_DSL=ALL \
  -DBUILD_PYTHON_BINDINGS=ON \
  -Dpybind11_DIR="$PYBIND11_DIR"

cmake --build . --parallel 2   # 低并行度！避免 8GB 板端编译 OOM
```

编译产物如下：

```text
build/libNvInfer_edgellm_plugin.so              ← TensorRT 插件（AttentionPlugin 等）
build/examples/llm/llm_build                    ← LLM engine 构建器
build/examples/llm/llm_inference                ← LLM 推理示例
build/examples/multimodal/visual_build          ← 视觉 engine 构建器
build/pybind/_edgellm_runtime.cpython-310-aarch64-linux-gnu.so  ← Python binding
```

根据板端实际环境，以补丁形式完成兼容性修复，如TensorRT 新旧版接口兼容问题、CuteDSL补丁。

##### FMHA cubin问题

***\*FMHA cubin\**** 是指针对 **FlashMulti-Head Attention（融合多头注意力机制）** 编译好的 **CUDA 二进制目标代码文件**（.cubin）。FlashAttention 的融合实现——把 attention 的 QK^T、softmax、PV 融合进一个 kernel，避免中间矩阵写回显存。**CUBIN**是CUDA 编译出的**二进制 kernel 文件**（类似 .exe 但给 GPU 执行）。Edge-LLM 针对每个 GPU 架构（这里是 sm_87）预先编译了多种 attention 配置的 CUBIN（比如不同 head_size、是否带 mask），运行时按模型配置挑一个加载。

遇到的问题是LLM ONNX 图优化到11AttentionPlugin时发现有2个创建失败，看 CUBIN 文件名（`*sm87.cubin.cpp` 是 C 数组包装）和元数据，发现被拒的两个都是：

```text
head_size = 256 + custom_mask
```

而Qwen3-VL-2B的实际attention配置是：

```text
head_size = 128
```

修复方法：跳过这两个CUBIN。

#### 启动TensortRT 推理服务

##### 运行时OOM问题修复

Jetson 是**统一内存架构**（UMA），CPU 和 GPU 共享同一块物理内存（8 GB），没有独立显存。

**weight streaming问题**：

TensorRT 的 **weight streaming** 机制：默认所有权重一次性常驻 GPU；开启后，权重可以**按需从内存流式加载**（用到哪层加载哪层），只保留一部分在 GPU 上。

探测实验发现：

```text
streamable weights              = 3441150208 bytes  ← 权重总量 3.44 GB
budget=0        → scratch = 1244660224 bytes（1.16 GiB 工作区）
budget=256MiB..3GiB → scratch 不降，总占用反而随 budget 涨
budget=全量(3.44GB) → scratch 归零，但常驻权重申请直接 OOM
```

因此只能选择budget=0，代价是权重流式读取让性能比 Transformers 还慢 4.45×。

**分配顺序问题**：

1.35 GB context + 0.79 GB visual 要同时存在。

假设先加载visual engine权重（0.79GB），再一次性申请LLM共享context，这时内存已碎，1.35GB连续内存申请失败

修正申请顺序：

```text
① 先查询 base executor / decoder strategy 的 context 大小
② 先申请并绑定 LLM 共享 GPU context (~1.35 GB)   ← 趁内存还完整时拿大块
③ 再加载 visual/audio/action runner (~0.79 GB)
④ 仅当 multimodal runner 需求确实更大时才重新分配
```

**engine profile问题**：

原始profile大小为`maxInputLen=1024`，`maxKVCacheCapacity=2048`，运行时KV Cache过大导致OOM，通过分析单次输入长度：

```text
Python processor: input_ids = 570 token（图片 196 visual tokens + 文本）
Edge-LLM C++ 实际展开 = 735 token（比 Python 多，模板展开不同）
```

将profile大小改为768/1024。

##### 正常启动服务

```bash
export PYTHONPATH=/home/ubuntu/TensorRT-Edge-LLM:/home/ubuntu/JetsonVLM/src
export BUILD_DIR=/home/ubuntu/TensorRT-Edge-LLM/build
export EDGELLM_PLUGIN_PATH=/home/ubuntu/TensorRT-Edge-LLM/build/libNvInfer_edgellm_plugin.so
export EDGELLM_WEIGHT_STREAMING_BUDGET_BYTES=0
export LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/home/ubuntu/TensorRT-Edge-LLM/build:/usr/local/cuda/targets/aarch64-linux/lib:/usr/lib/aarch64-linux-gnu

cd /home/ubuntu/JetsonVLM
.venv-jetson/bin/python scripts/serve_edgellm.py \
  --engine-root artifacts/engines/qwen3_vl_2b_fp16 \
  --weight-streaming-budget-bytes 0 \
  --host 127.0.0.1 \
  --port 8000
```

##### Transformers 与Edge-LLM同机比较

两份报告共同使用：

```text
model revision    89644892e4d85e24eaac8bacfd4f463576704203
workload identity parking_risk_v1@sha256:8350ace4574f8aa154319f7136ef831003d4dcc074ef20b74c1b419d69a2a493
dataset           ps20_pilot_v1 / test / 20 samples
power mode        15W_MODE_0
Jetson L4T        R36.5.0
precision         fp16
```

| 指标           | Transformers FP16 | Edge-LLM FP16 |
| -------------- | ----------------: | ------------: |
| 后端完成       |             20/20 |         20/20 |
| 严格 JSON 有效 |             20/20 |          0/20 |
| E2E p50        |           9.385 s |      41.758 s |
| E2E p90        |          14.117 s |      50.612 s |
| E2E p99        |          27.939 s |      55.419 s |
| RAM 峰值       |           6398 MB |       7414 MB |
| swap 峰值      |            501 MB |       2579 MB |
| 平均输入功耗   |          10.115 W |      10.110 W |

按当前数据计算，Edge-LLM 的 p50/p90/p99 分别约为 Transformers 的`4.45× / 3.59× / 1.98×`，即当前实现没有加速，反而更慢。主要背景是 LLM engine
必须使用 `weight-streaming-budget=0` 才能在 8 GB 设备上加载，权重流式访问和 1.24 GBscratch 带来明显代价。同时，两边模型权重相同，但 Edge-LLM 的 prompt/chattemplate、请求格式、解码行为或输出约束仍可能存在适配差异，导致严格JSON有效率极低。

### 2.4 量化

Jetson Orin + JetPack 6.2 支持 **FP16 / INT8 / INT4**，不支持 FP8/NVFP4。

#### AWQ 量化的具体方法

本轮量化使用 TensorRT Edge-LLM v0.9.1 的量化接口和 ModelOpt 0.44.0，配置为：

```json
{
  "quant_algo": "W4A16_AWQ",
  "group_size": 128,
  "zero_point": false,
  "pre_quant_scale": true,
  "quantized_scope": "LLM backbone",
  "visual_precision": "FP16",
  "lm_head_precision": "FP16",
  "kv_cache_quantization": null
}
```

量化边界不是整个 Qwen3-VL：

- 量化 LLM backbone 中 28 层 Transformer 的 attention 投影 `q_proj/k_proj/v_proj/o_proj`；
- 量化 FFN 的 `gate_proj/up_proj/down_proj`；
- visual encoder 保持 FP16；
- `lm_head` 保持 FP16；
- KV cache 不量化；
- `W4A16` 表示权重使用 INT4，激活仍使用 FP16；
- `group_size=128` 表示每 128 个权重共享量化 scale，`zero_point=false` 表示对称量化。

AWQ共分为两步：

```text
# 第一步：weight-only量化
原始 FP16 权重 W (shape [out, in])
  → 按 group_size=128 分组
  → 每组：scale = max(|W_group|) / 8（INT4 对称范围 [-8, 7]）
  → W_int4 = round(W / scale)    每个权重存 4 bit
  → 反量化：W_approx = W_int4 × scale（推理时）
# 第二步：activation-aware 的 scale 优化
用校准集跑一遍，统计每个通道的激活值分布（activation 重要性）
→ 找到"对输出影响大"的权重通道（激活值大的通道）
→ 对这些通道放大 scale（per-channel 保护），减少量化误差
→ 把放大吸收进相邻层（等效变换，数学上无损）
```

#### 校准集准备

**为什么需要校准集**：AWQ 的"activation-aware"——它要跑一遍数据统计**每个权重通道对输出的影响**，才能决定量化时保护哪些通道。没有校准集就没有感知，量化就退化成纯 round。

使用通用新闻文本 `cnn_dailymail` 前 128 条作为校准集。

#### 环境准备与服务器端量化

```text
# RTX 4090 D 环境确认
torch 2.12.0 (CUDA 13), torchvision 0.27.0, transformers 5.9.0
ModelOpt 0.44.0（NVIDIA 量化工具）, datasets 4.8.5, ONNX 1.19
```

```bash
python scripts/quantize_model.py \
  --config configs/flows/quantize_qwen3_vl_2b_int4_awq.json \
  --execute
```

实测结果如下：

```text
flow status         succeeded
flow elapsed        113.50 s
ModelOpt quantization 98.4 s
峰值显存            约 8.5 GiB
```

AWQ内部做了什么：

1. 对 LLM 骨干每个 Linear 权重（q/k/v/o + FFN 的 gate/up/down）按 **group_size=128** 分组
2. 每 128 个权重算一组 scale：`scale = max(|W_group|) / 8`（INT4 对称范围 [-8,7]）
3. **跑校准集**统计激活值分布 → 找出对输出影响大的通道 → **放大这些通道的 scale（per-channel 保护）**，并把放大等效吸收进相邻层（数学无损变换）
4. 权重存 INT4，scale/zero 元数据保存

#### ONNX导出与板端构建

```bash
python scripts/export_model.py \
  --config configs/flows/export_qwen3_vl_2b_int4_awq.json \
  --execute
```

耗时98.19s，导出：

```text
llm/model.onnx              4209515 字节  (~4 MB)      图结构
llm/model.onnx.data         1350303744 字节  (~1.26 GiB)  INT4 权重（external data）
llm/embedding.safetensors   622329944 字节  (~594 MB)   embedding 单独存
```

会同时导出FP16 visual engine，但与板端完全一致。将上述文件传输至板端后构建engine：

```bash
.venv-jetson/bin/python scripts/build_engine.py \
  --config configs/flows/build_qwen3_vl_2b_int4_awq_llm_engine_i768_k1024.json --execute
```

#### INT4量化模型板端推理

执行串行单图稳定性验证，然后冻结 20 样本 INT4 Study。与FP 16模型对比：

| 指标               | Edge-LLM FP16 | Edge-LLM INT4 AWQ |       变化 |
| ------------------ | ------------: | ----------------: | ---------: |
| engine 大小        |  3453786212 B |      1362769140 B |     -60.5% |
| mean 延迟          |   53638.78 ms |       10685.66 ms | 5.02x 加速 |
| p50 延迟           |   50753.15 ms |       10524.38 ms | 4.82x 加速 |
| p90 延迟           |   69082.40 ms |       11394.11 ms | 6.06x 加速 |
| p99 延迟           |   75081.88 ms |       12626.09 ms | 5.95x 加速 |
| aggregate tokens/s |        1.4803 |            7.3276 |      4.95x |
| RAM peak           |      7418 MiB |          5072 MiB |     -31.6% |
| swap peak          |      1904 MiB |           840 MiB |     -55.9% |
| GPU 温度 peak      |      65.03 °C |          60.75 °C |   -4.28 °C |
| JSON validity      |          1.00 |              1.00 |       持平 |
| risk accuracy      |          0.35 |              0.35 |       持平 |
| event micro-F1     |        0.3590 |            0.0000 |   明显退化 |
| unsafe advice rate |          0.00 |              0.00 |       持平 |

从结果上看，量化后的模型在延迟、吞吐、engine 大小和内存上获得明确收益。但 128 条通用新闻文本的校准集只覆盖语言统计，不覆盖泊车视觉指令分布；本实验的事件 F1 退化说明该校准方案只能作为部署与性能验证，不能作为最终质量版本。

### 2.5 LoRA

从2.3节的Transformer推理结果和Edge-LLM的同精度推理结果中可以看出，两者针对”泊车场景“的准确率均很低，即便是Transformer模型，其”风险等级准确率“也仅有0.35。因此，需要对模型进行训练与微调。

#### 数据准备

##### 隔绝训练/验证/测试集

从 PS2.0 `training` 9827 张按**文件名提取来源组**（`p2_img28_0408` → `p2_img28`，同一采集序列归一组），每组只选一张：

```text
train  64 张 / 64 来源组
val    16 张 / 16 来源组
test   20 张 / 独立 pilot 组
```

保证`train-val overlap=0`和`train-test overlap=0`。

##### 弱监督标签生成

用固定基础模型 + 固定 workload 对 80 张图生成弱监督 JSON，teacher模型本身具有偏置，因此本轮只能证明”链路可行“。

#### 训练

只微调语言模块的q/k/v/o，因为”风险判断“的语义推理发生在LLM的attention。

```text
# 训练脚本
precision: BF16（训练省显存）
gradient_checkpointing: 开启（省显存）
gradient_accumulation: 4
AdamW, lr=1e-4, warmup 10%, weight decay 0
epochs=1, optimizer steps=16（64 样本 / 4 累积 = 16 步）
loss 除以 accumulation → 等效 batch=4
```

#### 评测

#### 参数量计算

A/B的低维度均为16。

Qwen3-VL-2B 的 `hidden_size=2048`，但注意 **GQA**：`num_attention_heads=16`、`num_key_value_heads=8`、`head_dim=128`：

```text
q_proj: 2048 → 2048   (16 头)    参数 = 16×(2048+2048) = 65,536
k_proj: 2048 → 1024   (8 KV 头)  参数 = 16×(2048+1024) = 49,152
v_proj: 2048 → 1024   (8 KV 头)  参数 = 16×(2048+1024) = 49,152
o_proj: 2048 → 2048   (16 头)    参数 = 16×(2048+2048) = 65,536
────────────────────────────────────────────────────────────────
每层小计                                229,376
```

共28层decoder layer，总参数为：`229376 * 28 = 6,422,528`，约占总参数量的`6,422,528 / 2,133,954,560 = 0.3010%`。

### 2.6 人工标注数据集并重新LoRA、量化

#### 人工标注与数据拆分

前一阶段的 80 条开发样本先由 Codex 生成候选 assessment，再通过离线复核页面逐图人工确认或修正风险等级、风险事件、evidence 和 driver advice。最终 annotation 来源标记为`human_confirmed_v1`，不是候选的 `codex_visual_review_v1_single_pass`。

人工确认后的文件为：

```text
data/annotations/ps80_human_confirmed_v1.jsonl
```

80 条样本的风险等级分布为：`low=45`、`medium=32`、`high=3`。原始开发 manifest按来源组隔离为 64 条 train 和 16 条 validation；随后从原 train 来源组中另选 16 条作为独立 INT4 calibration。因此，实际监督训练数据和校准数据的关系为：

```text
80 条人工确认开发数据
├── 48 条 LoRA train
├── 16 条 LoRA validation
└── 16 条 INT4 calibration
```

在人工确认数据的 16 条 validation 上，随后对 LoRA 模型、workload 和 calibration做了组合对照：

| 版本 | LoRA / calibration 组合 | 严格 JSON | 风险准确率 | 事件 micro-F1 | 不安全建议率 | Jetson p50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 服务器 reviewed LoRA v2 | semantic v2 LoRA，未量化 | 100% | 56.25% | 0.4286 | 0% | 5.26 s |
| v1 INT4 | reviewed v1 LoRA + v1 calibration | 100% | 62.50% | 0.3000 | 18.75% | 7.84 s |
| v2 INT4 | semantic v2 LoRA + v1 calibration | 87.50% | 56.25% | 0.1429 | 18.75% | 7.32 s |
| v3 INT4 | semantic v2 LoRA + semantic v2 calibration | 100% | 68.75% | 0.1667 | 0% | 8.60 s |
| v5 INT4 | reviewed v1 LoRA + semantic v2 calibration | 100% | 62.50% | 0.2857 | 18.75% | 7.64 s |

严格 JSON 和风险等级在部分版本中改善，但事件 micro-F1 没有稳定提升；当前没有一个 INT4 版本同时在事件识别、格式、安全建议和时延上全面优于基线。

#### 扩大 validation 后的重训回归

为了检验 16 条 validation 是否导致指标波动，将人工确认 LoRA 数据重划分为 32 train和 32 validation，保留原 16 条 calibration 不变。新训练仍使用 3 epoch、non-low x2，在本机 RTX 4060 上完成：

```text
unique train: 32
effective train after oversampling: 43
optimizer steps: 33
validation loss: 0.817634
training time: about 164.65 s
peak CUDA memory: about 5.32 GiB
```

服务器 Transformers 在 32 条 validation 上的结果为：

| 样本数 | 严格 JSON | 风险准确率 | 事件 micro-F1 | 不安全建议率 | 端到端 p50 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 100% | 65.63% | 0.1905 | 18.75% | 4.41 s |

该版本输出明显收缩为 `low + no_event`，因此只作为 validation 扩展和工程回归证据，没有继续进入新的量化流程。

#### 阶段结论

本轮完成的事实链路为：

```text
80 条候选数据人工确认
  -> human_confirmed_v1 annotation
  -> 48/16 LoRA train/validation
  -> LoRA 微调与 merged 模型
  -> 16 条独立 calibration
  -> W4A16 AWQ 量化
  -> ONNX 导出
  -> Jetson engine 构建与 Edge-LLM study
```

人工标注解决了候选标签不能直接作为正式训练来源的问题；LoRA 训练恢复了部分风险等级和事件预测，但受样本规模与类别覆盖限制，质量仍不稳定。AWQ 量化完成了权重压缩和engine 构建，但当前实测只证明了格式、部署和部分资源指标，尚未证明相对 Jetson Transformers FP16 的综合性能优势。后续优化重点应转向 Edge-LLM 的 builder、kernel、weight streaming、KV cache、prefill/decode 和服务层开销，而不是继续把量化后的负向质量结果解释为模型能力提升。

### 2.7 TensorRT Edge-LLM 链路性能优化

考虑到数据集通用性和训练手段不足，下一步不准备做微调方面的优化，将目标放在系统链路的优化上。
