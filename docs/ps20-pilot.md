# PS2.0 AVM pilot 数据集

## 目的与状态

`ps20_pilot_v1` 从 Tongji PS2.0 的官方 testing 分类目录中分层选择 20 张
`600 x 600` AVM 鸟瞰图，用于尽快打通以下链路：

```text
AVM 图片 -> ParkingCaseCatalog -> Jetson Transformers FP16 -> InferenceRecord -> StudyReport
```

该数据集是第一轮工程 pilot，不是正式冻结测试集。人工标注只完成了单人第一轮目视
判断，质量指标对外使用前需要第二人复核并冻结 revision。

## 分层抽样

| PS2.0 场景 | 数量 |
| --- | ---: |
| indoor parking lot | 3 |
| outdoor normal daylight | 4 |
| outdoor rainy | 3 |
| outdoor shadow | 4 |
| outdoor slanted | 3 |
| outdoor street light | 3 |
| 合计 | 20 |

对应文件：

- manifest：`data/manifests/ps20_pilot_v1.jsonl`
- annotation：`data/annotations/ps20_pilot_v1.jsonl`
- 本地图片：`data/raw/ps2.0/pilot/`
- Jetson study：`configs/studies/jetson/jetson_transformers_fp16_ps20_pilot.json`

## 标注边界

PS2.0 只提供拼接后的 AVM 图片和停车位标注，不提供车辆当前挡位、转向、目标车位或
规划轨迹。因此本轮标注遵守以下保守规则：

- 只依据图中可见事实标注，不推断车辆意图。
- 没有规划轨迹时，不标注包含 `near_maneuver_path` 的事件。
- 没有目标车位时，不标注 `parking_space_conflict`。
- `narrow_passage` 只用于近距离停放车辆明显限制横向空间的图像。
- `visibility_occlusion` 只用于雨天模糊、强明暗差、低照度或噪声明显降低细节可见度的图像。
- 图中没有明确风险目标时使用 `risk_level=low`、空 `events` 和
  `maintain_observation`。

该边界意味着 pilot 不能覆盖行人横穿、动态车辆侵入、高风险刹停和轨迹相关判断。
后续需要从原始四路鱼眼或连续环视视频补充这些类别。

## 数据使用边界

原始 PS2.0 图片和 `.mat` 文件位于被 Git 忽略的 `data/raw/`，不提交和再分发。
仓库只保存 manifest、项目领域标注和本说明。正式发布衍生标注前仍需确认原始数据集
许可条款。

## 2026-08-02 Jetson Transformers FP16 pilot

### Smoke 与 prompt 冻结

第一次对 `ps20-indoor-001` 执行单图 smoke 时，模型生成了可解析的 JSON 对象，但将
`maintain_observation` 翻译为中文自然语言，严格契约因此记录
`json_parse_error`。随后在 workload 的 `user_prompt` 中明确要求英文
`snake_case` 枚举值必须逐字输出，禁止翻译、改写或扩写。修订后的 workload 身份为：

```text
parking_risk_v1@sha256:8350ace4574f8aa154319f7136ef831003d4dcc074ef20b74c1b419d69a2a493
```

同图重跑得到 `failure=null`，证明严格 JSON 生成和解析链路可以完成。该 prompt 在完整
pilot 运行前冻结，后续 runtime 对比必须使用相同 workload 身份。

### 执行命令

在 Jetson 的 Bash 中执行：

```bash
cd /home/ubuntu/JetsonVLM

LD_LIBRARY_PATH=/home/ubuntu/JetsonVLM/.venv-jetson/lib/python3.10/site-packages/nvidia/cu12/lib:/usr/local/cuda/lib64 \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
PYTHONPATH=src \
timeout 1800s \
.venv-jetson/bin/python -m parksight_vlm.app.run_study \
  --config configs/studies/jetson/jetson_transformers_fp16_ps20_pilot.json
```

外部同时以一秒间隔运行 `tegrastats`。完整运行包含模型冷启动和 20 次推理，监控日志
覆盖 226 秒。

### 质量结果

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

模型把 20 张图片全部预测为 `low`；18 张预测 `narrow_passage`，7 张额外预测
`vehicle_near_maneuver_path`，没有预测 `visibility_occlusion`。人工标注包含 7 个
`narrow_passage` 和 9 个 `visibility_occlusion`。因此当前低质量结果主要表现为风险等级
塌缩、窄通道过检、轨迹相关事件误报和可见性风险漏检，不能归因于 JSON 或运行时失败。

### 性能与资源结果

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
| GPU 温度峰值 | 61.781 C |
| `VDD_IN` 瞬时峰值 | 12.655 W |
| `VDD_IN` 最终区间平均 | 10.115 W |

`StudyReport` 当前还没有直接接入 `tegrastats`，所以报告内部的平均功耗和峰值温度仍为
`null`；表中的板级功耗、温度、系统 RAM 和 swap 来自同一次运行的外部监控日志。

本地原始证据保存在被 Git 忽略的以下路径：

- `reports/jetson_transformers_fp16_ps20_pilot.json`
- `reports/jetson_transformers_fp16_ps20_pilot_20260802_tegrastats.log`
- `reports/jetson_transformers_fp16_ps20_pilot_20260802_stdout.log`
- `reports/jetson_transformers_fp16_ps20_pilot_20260802_stderr.log`
