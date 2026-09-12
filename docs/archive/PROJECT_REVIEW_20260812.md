# ShuttleLab 项目回顾、拆解与后续路线图

更新日期：2026-08-12

## 1. 项目目标

从固定机位单打视频中，自动提取可复核的视觉与事件数据：场地边界、球员姿态、球员
场地位置、羽毛球轨迹、击球候选和回合活动区间，并导出逐帧运动学表和逐拍事件表。

当前产物应称为“视觉事件数据集（带 ShuttleSet 兼容占位字段）”，不是完整 ShuttleSet
真值。它保留了连续逐帧视觉信息，但击球类型、落点、比分和胜负原因尚未完成。

## 2. 当前系统拆解

```text
视频注册与校验
  -> 场地四角提议 / 人工确认 / Homography
  -> 人体检测 + COCO-17 姿态
  -> 羽毛球 top-k 检测 + 时序关联 + 轨迹质量审计
  -> upper/lower 球员身份 + 场地坐标 + 短时追踪
  -> 击球候选 + 击球者推断 + 比赛进行状态门控
  -> 单视频逐帧/逐拍导出
  -> 多视频批量校验与合并
```

### 代码模块

| 路径 | 职责 | 状态 |
|---|---|---|
| `badminton_pipeline/ingest` | 视频 SHA-256、元数据、注册目录 | 完成 |
| `badminton_pipeline/calibration` | 四角校验、Homography、Good-Badminton 自动提议 | 完成首版 |
| `badminton_pipeline/models` | YOLOX、RTMPose、YOLO-ball 推理适配 | 可运行 |
| `badminton_pipeline/tracking` | 球员/球轨迹、回合、运动状态门控 | 开发版已验证 |
| `badminton_pipeline/evaluation` | Gold 锁定、稳健击球匹配和评估 | 完成 |
| `badminton_pipeline/exports` | 单视频表、批量数据集、质量报告与 manifest | 完成 |
| `badminton_pipeline/visualization` | QA 叠加视频 | 完成 |
| `badminton_pipeline/stroke_classification` | BST 输入窗口适配 | 仅适配器；类型分类未完成 |
| `demo/` | 公开展示站和本地 `/gold` 击球标注工具 | Web v13；当前非开发主线 |

### 模型与依赖

| 模型 | 用途 | 大小 |
|---|---|---:|
| YOLOX nano ONNX | 人体检测 | 3.55 MiB |
| RTMPose ONNX | COCO-17 姿态 | 20.88 MiB |
| YOLO11s ball | 羽毛球检测 | 18.28 MiB |

模型合计约 42.8 MiB。当前完整 Python 环境约 3.67 GiB，主要来自 PyTorch/CUDA；它
可以在新电脑重建，不应复制或压缩分享。

## 3. 已经完成并真实验证的内容

- 9 段视频已建立独立视频记录与标定记录；原视频保持在私有原路径，没有复制进项目。
- 自动/半自动场地边界已接入 Good-Badminton OpenCV 检测器。真实固定机位样本中，
  自动四角相对既有人工四角平均约 8.6 像素；自动结果默认只生成 proposal。
- 首个完整视频形成 58 条人工 Gold；后续 holdout、validation 和 6 视频前瞻批次均按
  frame-blind 流程标注与锁定。
- 当前锁定前瞻 Gold 为 174 次真实击球；自动运动门控前后唯一一次评估为：
  Precision 66.37% -> 71.98%，Recall 86.21% -> 85.63%，F1 75.00% -> 78.22%。
- 6 视频批量数据集构建成功：6,308 帧、207 条机器击球候选，结构化产物约 4.58 MiB。
- Python 全量单元测试 55 项通过；Web 生产构建在上一阶段通过。

## 4. 当前数据契约

### 逐帧层 `frames.parquet`

每个处理帧一行，保存 COCO-17 骨架、关键点置信度、球员场地坐标、球 top-k 候选、
平滑/预测位置、创新残差、轨迹状态、模型版本和比赛活动状态。它是后续身体运动学、
步法、连续移动和球轨迹分析的基础。

### 逐拍层 `dataset_rows.parquet`

每个机器击球候选一行，保存候选帧、时间、upper/lower 击球者、双方位置、击球时球
像素位置、轨迹/手腕/姿态证据、综合置信度、比赛状态证据与版本信息。

它不是完整 ShuttleSet：`stroke_type`、`aroundhead`、`backhand`、击球/落点、比分、
真实 A/B 球员、胜负原因均为空或 unknown；候选本身还存在已知误报和漏报。

### 批量层

`badminton_pipeline.exports.batch_dataset` 校验输入 manifest、SHA-256、schema、行数和
复合唯一键，然后确定性合并 videos、calibrations、frames 与 dataset_rows 四张表。

## 5. 已知限制和风险

1. 当前逐拍行是机器候选，不是人工真值；已知 207 个候选中有 58 个未匹配 Gold，
   174 个 Gold 中有 25 个未被召回。
2. 姿态、球轨迹和球员场地位置目前只有覆盖率/连续性审计，没有大规模人工位置真值。
3. upper/lower 是场地侧身份，不能直接代替 ShuttleSet 的真实 Player A/B。
4. 单目飞行中的羽毛球不在地面平面，不能用 Homography 伪造完整三维位置。
5. 回合间隔规则和短回合首拍门控仍可能误切；被抑制候选保留在独立文件中以便恢复。
6. 当前前端工作树包含未提交的本地 Gold 页面和视频副本，打包时只取源码，不取视频。

## 6. 工作区体积

| 目录 | 当前大小 | 分享策略 |
|---|---:|---|
| `.venv-model` | 3,669 MiB | 排除；新电脑重建 |
| `demo/` | 668 MiB | 只带源码；排除 node_modules、缓存、构建和视频 |
| `data/` | 492 MiB | 只带最终4.58 MiB数据集、Gold、标定和批次协议 |
| `models/` | 42.8 MiB | 完整便携包保留 |
| `vendor/` | 30.4 MiB | 只带场地检测源码与许可证 |
| Python源码、配置和测试 | <1 MiB | 保留 |

## 7. 后续计划

### P0：可复现与可分享（本次完成）

- 固化项目总览、安装说明、模型校验、环境验证和便携 ZIP。
- 不在包中携带本机绝对视频、虚拟环境、Node依赖、缓存或大体积中间产物。

### P1：在当前限定范围内继续完善

- 把批量构建从“合并单视频结果”升级为带断点续跑的原视频批处理编排器。
- 用多个不同机位统计自动场地检测的成功率、角点误差和人工修正率。
- 增加数据集审计抽样：随机姿态帧、球轨迹高残差帧、身份切换帧和全部击球窗口。

### P2：是否转向严格 ShuttleSet，需要用户确认

若目标是战术逐拍分析，应新增严格 ShuttleSet 表，而不是覆盖逐帧运动学表：

1. 人工修正真实击球、回合和 ball_round。
2. 建立真实 A/B 球员身份与换边逻辑。
3. 标注/训练18类击球类型、反手和绕头。
4. 识别击球点、下一次接触或末拍落点及场地区域。
5. 对完整比赛补充比分、发球者、回合胜者和得失分原因。
6. 将逐拍表通过 `video_id + frame_num` 与现有逐帧运动学表连接。

在未选择 P2 前，项目仍保持“批量视觉数据集 + 自动/半自动场地边界”的精简范围。

## 8. 恢复工作的入口

- 新电脑安装与运行：`PORTABLE_README.md`
- 批量数据与场地标定：`DATASET_BUILD_GUIDE.md`
- 完整事实台账：`PROJECT_STATE.md`
- 决策记录：`DECISIONS.md`
- 逐阶段进度：`PIPELINE_PROGRESS.md`
- 模型环境：`MODEL_PIPELINE.md`
