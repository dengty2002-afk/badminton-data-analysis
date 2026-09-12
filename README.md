# Badminton Data Analysis · ShuttleLab

面向羽毛球单打视频的本地分析项目：提取球员姿态与位置、羽毛球轨迹、击球候选、BST 球种预标注及结构化逐拍数据，并开展比赛级打法风格分析。

这是研究代码与方法的发布快照。自动预测不等同于人工真值；当前项目不提供经过验证的竞技能力评分或胜负因果结论。

## 项目结构

| 目录 | 用途 |
|---|---|
| `badminton_pipeline/` | 正式视频、轨迹、击球、球种及导出管线 |
| `configs/` | 配置和带 SHA256 的模型下载清单 |
| `scripts/` | Windows 安装、模型下载与运行验证 |
| `tests/` | Python 单元及集成契约测试 |
| `examples/` | 不需要比赛视频或 GPU 的合成示例 |
| `experiments/` | 独立 HitNet、SwingNet 对照实验及下游误差评估 |
| `research/` | 比赛级风格、配对比较和敏感性分析脚本 |
| `research_outputs/` | 精选研究协议、聚合结果及图表 |
| `vendor/` | 运行所需的最小第三方源码及原许可证 |
| `demo/` | 保留的可选本地界面源码，不是当前开发重点 |
| `docs/` | 数据边界、复现说明及历史文档 |

## 快速开始：不下载模型

当前完整管线支持 Windows、Python 3.12；批处理锁使用 Windows API，Linux 尚未适配。以下命令在项目根目录运行：

```powershell
py -3.12 -m venv .venv-model
.\.venv-model\Scripts\python.exe -m pip install -e .
.\.venv-model\Scripts\python.exe examples/evaluate_hits.py
.\.venv-model\Scripts\python.exe -m unittest discover -s tests -v
```

示例使用合成事件，演示一对一匹配的 Precision/Recall/F1；它不代表真实模型性能。缺少单独下载的权重时，权重完整性集成测试会显式跳过；不会把跳过当作 GPU 推理验证。

## Windows GPU 环境与真实视频

本地已使用 Python 3.12、PyTorch 2.13.0+cu130 与 RTX 5060 验证。其他显卡/系统需要调整 PyTorch、ONNX Runtime 和驱动组合。

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

默认安装 Python/GPU 依赖并按清单下载模型；可用 `-SkipModels` 跳过权重下载。界面依赖仅在指定 `-WithWeb` 时安装，不自动部署网站。

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.ingest.register_video --input D:\videos\match.mp4 --data-dir data
.\.venv-model\Scripts\python.exe -m badminton_pipeline.run_pipeline --help
```

真实视频处理需要自备视频、已接受的场地标定和下载后的权重。操作细节见 [管线指南](docs/archive/DATASET_BUILD_GUIDE.md)；历史指南中的机器目录应替换为自己的目录。安装后的完整模型检查可运行 `scripts/verify_portable.py --full`。

## 当前测得结果与限制

六段既有回归视频，174 条人工击球标注；严格匹配为 ±5 帧加逐标注不确定范围：

| 检测方案 | Precision | Recall | F1 |
|---|---:|---:|---:|
| 现有规则 | 62.32% | 74.14% | 67.72% |
| 官方权重 SwingNet 适配 | 63.86% | 74.14% | 68.62% |
| 固定参数融合 | 61.80% | 82.76% | 70.76% |

融合减少漏检但增加误检，尚未替换生产检测器。以上不是 BST 球种分类指标，也不是独立盲测结果。

- [SwingNet 实验](experiments/swingnet/REPORT.md)：官方 checkpoint 严格加载，恢复历史预处理；说明缺失源码适配与输入尺寸差异。
- [MonoTrack HitNet 实验](experiments/monotrack-hitnet/REPORT.md)：本地架构复现，不是官方预训练权重。
- [对 BST 和研究问题的影响](experiments/swingnet/downstream-impact-20260913/ASSESSMENT.md)：误差传播、序列完整性、半场差异及分析边界。
- [研究协议](research_outputs/shi_tactical_profile_v1/PLAYER_RADAR_RESEARCH_PROTOCOL_FINAL_v2.md)：24 场回顾性个案分析，原始算法标签未被当作无误差行为测量。

## 数据和复现

仓库不包含比赛视频、模型二进制、完整逐帧特征、私人输入路径、虚拟环境或缓存。公开的是源码、固定清单、方法、聚合结果和合成示例。完整实验需要恢复本地输入；不能仅凭聚合表重建全部实验。详见 [复现边界](docs/REPRODUCIBILITY.md)。

## 第三方与许可

参见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和 [LICENSE.md](LICENSE.md)。仓库混合使用不同来源的代码，公开可见不表示全部内容采用同一开源许可证。权重、上游依赖和原视频仍受其各自条款约束。
