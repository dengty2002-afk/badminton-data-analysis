# 石宇奇比赛自动标注管线

## 项目概述

端到端的职业羽毛球单打比赛视频自动标注管线。系统从比赛视频出发，自动完成场线标定、比分识别、回合切分、球员追踪、羽毛球轨迹追踪、击球事件检测、18类击球技术识别，以及人工审核与修正的闭环，最终输出可直接用于计算机视觉和比赛胜率建模的高质量标注数据集。

## 管线范围

- **比赛规模**: 20场石宇奇职业比赛（2场标定 + 8场训练 + 10场盲测）
- **标签体系**: ShuttleSet 官方18类击球技术（net shot, smash, lob, clear, drive, drop, push, rush 等变体）
- **执行方式**: 双人协作，Mac本地开发 + NVIDIA GPU远程训练
- **计划周期**: 基准14周，预计320-440人时

## 八阶段管线架构

| 阶段 | 名称 | 主要工作 | 输出 |
|------|------|----------|------|
| S1 | 视频提取 | 源视频校验、转码、哈希登记 | canonical.mp4, video.json |
| S2 | 比赛/回合切分 | 镜头切分、回放检测、比分OCR、发球状态机 | segments.parquet, rallies.parquet |
| S3 | 场地/球员 | 场地线标定、透视变换、球员检测、追踪、角色 | court/player observations |
| S4 | 羽毛球轨迹 | TrackNetV3轨迹追踪、轨迹插值与平滑 | shuttle trajectory |
| S5 | 击球事件 | 轨迹转折点检测、视频瞬态特征、球员临近约束 | hit candidates |
| S6 | 18类分类 | XGBoost基线 + TCN时序模型 + 辅助分类头 | 18类标签与置信度 |
| S7 | 自动审核 | 置信度阈值、规则约束、异常检测 | review tasks |
| S8 | 输出 | Schema校验、数据入库、哈希版本标签 | dataset_v0.x |

## 技术栈

- **语言/环境**: Python 3.11 + uv
- **视频处理**: FFmpeg + OpenCV
- **深度学习**: PyTorch (MPS/CUDA), Ultralytics YOLO, TrackNetV3
- **ML模型**: XGBoost, TCN (Temporal Convolutional Network)
- **OCR**: PaddleOCR
- **数据处理**: Parquet + DuckDB + SQLite
- **工作流**: Prefect
- **版本管理**: Git + DVC + MLflow
- **审核界面**: FastAPI + React/Vite + Canvas
- **硬件**: Apple M5 MacBook Air (16GB) + NVIDIA L4/A10 GPU (24GB)

## 关键指标

| 指标 | 目标值 |
|------|--------|
| 回合切分 F1 | ≥ 0.98 |
| 比分准确率 | ≥ 99% |
| 击球事件 P/R | ≥ 0.95 |
| 发球员准确率 | ≥ 0.98 |
| 9区位置准确率 | ≥ 0.90 |
| 18类 macro-F1 | ≥ 0.80 |
| 18类 Top-2 准确率 | ≥ 0.95 |
| 置信度校准 ECE | ≤ 0.05 |
| 人工审核效率 | ≤ 75分钟/场 |
| 自动审核率 | ≥ 65% |

## 数据目录结构

```
data/
├── raw/          # 原始比赛视频（仅授权播放源）
├── processed/    # 转码后的母版视频 (720p H.264)
├── fixtures/     # 5分钟黄金测试片段
└── external/     # ShuttleSet 等外部数据集
```

## 参考资料

- [ShuttleSet Paper (KDD 2023)](https://arxiv.org/abs/2306.04948)
- [CoachAI Projects - ShuttleSet](https://github.com/wywyWang/CoachAI-Projects)
- [TrackNetV3](https://github.com/qaz812345/TrackNetV3)

## 版本

当前版本: v0.1-foundation | 计划书版本: 2026年7月25日
