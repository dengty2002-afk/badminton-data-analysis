# 羽毛球逐拍数据集半自动生产系统 PRD

> 项目代号：Badminton Silver/Gold Pipeline  
> 文档版本：v1.1  
> 日期：2026-08-06  
> 状态：准备开发  
> 目标平台：Windows 本地工作站，NVIDIA RTX 5060 8GB

## 1. 产品摘要

本项目用于把羽毛球单打视频半自动转换成类似 ShuttleSet 的逐拍结构化数据集。

系统首先利用 Good-Badminton、RTMPose 和羽毛球追踪模型，从完整比赛视频中提取球场、球员姿态、球员位置和羽毛球轨迹；随后检测可能发生击球的时间点，在标注界面中展示对应视频片段；标注员确认击球时间、击球者和技术动作，形成少量高质量 Gold 数据；最后使用 Gold 数据微调 BST-CG-AP 击球分类模型，对剩余击球批量预测，生成 Silver 数据，并导出 ShuttleSet-like 的逐拍表格。

核心原则：

- 原始视频始终是事实来源。
- Good-Badminton 输出的是逐帧视觉特征，不是逐拍战术真值。
- Gold 是人工确认的数据；Silver 是模型生成的数据，两者必须分开保存。
- 模型预测必须保留置信度、模型版本和处理记录。
- 第一版只支持固定机位单打，暂不追求双打和完全自动化。

## 2. 背景与问题

ShuttleSet 提供了击球时间、击球者、动作类型、双方位置、落点、比分和回合结果等逐拍信息，但创建这类数据需要大量人工逐帧观看和标注。

Good-Badminton 已经具备以下基础能力：

- 球场检测和透视映射；
- RTMPose、RTMO、YOLO Pose 人体姿态估计；
- 球员位置和移动轨迹；
- YOLO 羽毛球检测；
- 视频、JSONL 和位置可视化输出。

但 Good-Badminton 当前不能直接生成 ShuttleSet-like 数据，主要缺少：

- 稳定的击球时刻检测；
- 击球者识别；
- 技术动作分类；
- 完整骨架数据导出；
- 羽毛球落点识别；
- 人工审核与修正工具；
- Gold/Silver 数据质量控制；
- ShuttleSet-compatible 数据集导出。

## 3. 产品目标

### 3.1 核心目标

1. 批量处理固定机位羽毛球单打视频。
2. 为每一帧保存完整人体骨架、球员场地位置和羽毛球轨迹。
3. 自动提出高召回率的击球候选事件。
4. 提供高效率的逐拍人工审核和动作标注界面。
5. 使用 Gold 数据微调 BST-CG-AP 击球类型分类模型。
6. 对未标注击球批量推理并生成 Silver 数据。
7. 导出可用于击球预测、移动预测、动作识别和域适应研究的数据集。
8. 对每个标签保留来源、置信度、模型版本和人工修改历史。



### 3.2 非目标

第一版不包含：

- 双打四人持续身份追踪；
- 实时摄像头分析；
- 正式比赛级界内/界外鹰眼裁决；
- 全自动生成比分、得分方和失误原因；
- 3D 羽毛球轨迹重建；
- 手机端标注；
- 云端多租户产品化。

## 4. 关键术语

| 术语 | 定义 |
|---|---|
| Frame Data | 每一帧的球员骨架、球员位置、羽毛球位置等视觉检测数据 |
| Hit Event | 一次击球事件，至少包含击球时间和击球者 |
| Gold | 经过人工确认或复核的高质量逐拍标签 |
| Silver | 由模型自动生成、未经逐条人工确认的伪标签 |
| Stroke Clip | 以击球时刻为中心截取的短时间窗口，不一定永久保存为独立视频 |
| ShuttleSet-like | 字段和语义尽量对齐 ShuttleSet 的逐拍表格，但不宣称为官方 ShuttleSet |
| Domain | 视频来源域，例如职业转播、业余固定机位、不同场馆或不同帧率 |

## 5. 目标用户

### 5.1 研究人员

需要构建羽毛球击球预测、移动预测、动作识别或域适应数据集。

### 5.2 标注员或教练

需要在视频上下文中快速确认击球时间、击球者、技术动作、落点和回合结果。

### 5.3 算法开发者

需要重复运行视觉模型、训练 BST、比较实验版本并追溯数据来源。

## 6. 产品范围与端到端流程

```text
完整比赛视频
  ↓
视频登记与球场标定
  ↓
Good-Badminton 姿态/场地提取 + 羽毛球追踪
  ↓
逐帧数据 frames.parquet
  ↓
轨迹平滑、回合分割、击球候选检测
  ↓
机器候选 events_silver.parquet
  ↓
人工半自动标注
  ↓
Gold annotations.parquet
  ↓
微调 BST-CG-AP
  ↓
批量预测剩余击球
  ↓
Silver 逐拍记录
  ↓
质量过滤 + ShuttleSet-like 数据集导出
```

## 7. 功能需求

### FR-01 视频登记

系统必须支持：

- 导入 MP4、MOV、MKV 视频；
- 使用 ffprobe 读取 FPS、时长、分辨率、总帧数和音轨信息；
- 为每个视频生成唯一 `video_id`；
- 计算文件校验值，避免重复导入；
- 允许输入视频来源、比赛、球员、场馆、机位和 domain；
- 标记是否允许在研究数据中重新分发原视频；
- 生成适合浏览器播放的 CFR 代理视频，但保留原视频。

### FR-02 球场标定

系统必须：

- 调用 Good-Badminton 自动检测球场四角；
- 支持人工点击修正四角点；
- 保存 homography、球场方向、标定人和版本；
- 显示俯视球场预览；
- 将球员脚点映射到标准 6.1m × 13.4m 球场坐标；
- 对低质量标定阻止后续批量处理。

### FR-03 姿态提取

默认使用 Good-Badminton 的 RTMPose `balanced` 或 `performance` 模式。

必须导出：

- 每个检测人物的 bounding box；
- COCO 17 个二维关键点；
- 每个关键点的置信度；
- 上/下半场球员身份；
- 关键点是否缺失或插值；
- 姿态模型名称、权重版本和推理参数。

当前 Good-Badminton 仅在最终 JSONL 中保留中心和手腕，开发时必须扩展为完整骨架导出。

### FR-04 羽毛球轨迹提取

MVP 使用 Good-Badminton 的 YOLO 羽毛球检测器；同时保留接入 TrackNetV3 的接口。

必须导出：

- 每帧羽毛球图像坐标；
- 检测置信度；
- 可见、缺失或插值状态；
- 平滑前和轨迹修复后的坐标；
- 使用的模型和权重版本。

Pilot 阶段必须比较 YOLO 和 TrackNetV3 在目标视频上的召回率。若 YOLO 在击球前后连续漏检明显，则正式管线改用 TrackNetV3。

### FR-05 球员跟踪与身份

系统必须：

- 在单打中维护 `upper` 和 `lower` 两个稳定身份；
- 在换边时保留真实球员身份，不仅保存画面上下位置；
- 保存球员图像坐标和球场坐标；
- 对漏检进行短期插值；
- 标记身份不确定帧；
- 允许标注员修正击球者。

### FR-06 回合分割

系统必须：

- 检测比赛镜头和非比赛镜头；
- 提出回合开始和结束候选；
- 过滤回放、慢动作和场外镜头；
- 为所有逐帧数据写入 `rally_id`；
- 允许人工合并、拆分或删除回合；
- 不把模板匹配结果直接当作 Gold 回合边界。

### FR-07 击球候选检测

MVP 使用多信号规则融合，不要求一开始训练复杂检测模型。

候选分数至少融合：

- 羽毛球轨迹方向变化；
- 羽毛球速度变化；
- 羽毛球与左右手腕的距离；
- 球员肩、肘、腕运动幅度；
- 击球音频瞬态（可选）；
- 单打双方交替击球约束。

系统必须输出：

- `candidate_frame` 和 `candidate_time`；
- 推荐的片段起止帧；
- 预测击球者；
- 各信号分数；
- 综合置信度；
- 是否来自人工补录。

优先优化召回率：误报可以快速删除，漏报需要标注员主动发现。

### FR-08 半自动标注界面

界面必须同时展示：

- 原视频片段；
- 当前候选帧和逐帧前后移动；
- 可开关的人体骨架和羽毛球轨迹；
- 双方在俯视球场上的位置；
- 当前机器预测和置信度；
- 当前回合前后击球的上下文。

标注员必须可以：

- 接受候选；
- 删除误报；
- 前后调整击球帧；
- 补充漏掉的击球；
- 选择击球者；
- 选择动作类型；
- 标注正手/反手和是否头顶区；
- 点击修正击球点和落点；
- 标记不确定、遮挡或无法判断；
- 保存并自动进入下一条；
- 使用键盘快捷键完成主要操作。

标注界面应以事件表索引原视频，不要求预先永久切割所有小视频。

### FR-09 Gold 数据审核

Gold 标签状态包括：

```text
unreviewed → annotated → double_checked → adjudicated
```

系统必须：

- 保存标注员、复核员和时间；
- 保存机器原预测与人工修改后的值；
- 不允许覆盖历史版本；
- 支持 10%–20% 样本双人独立标注；
- 支持分歧裁决；
- 导出标注一致性报告。

### FR-10 BST 击球分类模型

主模型固定为 BST-CG-AP（BST-3）PyTorch 实现。

每个击球样本的输入：

```text
T 帧击球者 2D 关节及置信度
T 帧骨骼向量
T 帧羽毛球轨迹及置信度
T 帧击球者球场位置
T 帧对手球场位置
```

默认设置：

- `T = 30`，后续与 60/100 帧做消融实验；
- 使用 ShuttleSet 预训练权重；
- 使用 Gold 数据微调；
- 第一版使用 8–10 个粗粒度类别；
- 第二版映射到 ShuttleSet 18 类；
- 类别不均衡使用 class-weighted cross entropy；
- 训练、验证、测试按比赛或球员划分。

输出：

- 每个动作类别的概率；
- Top-1 和 Top-3 类别；
- 模型版本；
- 是否通过 Silver 置信度门槛。

### FR-11 Silver 批量生成

系统必须：

- 对所有未人工标注击球运行 BST；
- 生成动作类型、击球者和相关置信度；
- 低置信度事件进入人工复核队列；
- 高置信度事件可进入 Silver 数据集；
- Gold 记录永远优先于 Silver 预测；
- 模型升级后允许重新生成 Silver，而不改变 Gold；
- 支持按类别和 domain 采样，避免只保留容易样本。

### FR-12 ShuttleSet-like 导出

导出表至少包含：

```text
video_id
match_id
set_id
rally_id
stroke_index
hit_frame
hit_time
player
stroke_type
stroke_type_confidence
aroundhead
backhand
player_location_x
player_location_y
opponent_location_x
opponent_location_y
hit_x
hit_y
landing_x
landing_y
label_source
annotation_status
model_version
```

可选字段：

```text
score_A
score_B
getpoint_player
lose_reason
landing_height
landing_area
```

若某字段无法可靠生成，必须保存为空值或 `unknown`，不得伪造。

支持以下导出：

- Parquet：训练和批量分析的主格式；
- CSV：兼容 ShuttleSet 和常规统计工具；
- JSONL：调试和流式输出；
- 数据字典与类别映射表；
- 训练、验证、测试 split 文件。

### FR-13 实验与版本追踪

每次处理或训练必须记录：

- Git commit；
- 配置文件；
- 输入视频校验值；
- 模型权重校验值；
- Python、PyTorch 和 CUDA 版本；
- 随机种子；
- 训练数据版本；
- 指标和混淆矩阵；
- 输出数据集版本。

## 8. 动作类别设计

### 8.1 第一阶段粗粒度类别

建议先使用以下 9 类：

1. 短发球；
2. 长发球；
3. 高远球；
4. 吊球；
5. 杀球；
6. 挑球；
7. 平抽/推球；
8. 网前球；
9. 防守回球。

增加：

- `unknown`：画面不足或标注员无法判断；
- `not_a_hit`：机器误报。

### 8.2 第二阶段 ShuttleSet 18 类

在 Gold 数量和一致性足够后细分为：

- net shot；
- return net；
- smash；
- wrist smash；
- lob；
- defensive return lob；
- clear；
- drive；
- driven flight；
- back-court drive；
- drop；
- passive drop；
- push；
- rush；
- defensive return drive；
- cross-court net shot；
- short service；
- long service。

开始大规模标注前必须完成中文标注手册，给出每类定义、正例、反例和易混淆类别。

## 9. 数据设计

### 9.1 `videos.parquet`

| 字段 | 类型 | 说明 |
|---|---|---|
| video_id | string | 视频唯一标识 |
| path | string | 原视频路径 |
| proxy_path | string | 浏览器代理视频 |
| checksum | string | 文件校验值 |
| fps | float | 原视频 FPS |
| frame_count | int | 总帧数 |
| width/height | int | 分辨率 |
| duration_sec | float | 时长 |
| match_id | string | 比赛标识 |
| players | struct | 球员元数据 |
| domain | string | 视频域 |
| redistribution | string | 原视频再分发权限 |

### 9.2 `calibrations.parquet`

| 字段 | 类型 | 说明 |
|---|---|---|
| video_id | string | 视频标识 |
| corners | array | 四角图像坐标 |
| homography | array | 透视变换矩阵 |
| court_orientation | string | 上下半场方向 |
| quality_status | string | accepted/rejected |
| source | string | auto/manual |
| version | int | 标定版本 |

### 9.3 `frames.parquet`

按 `video_id` 分区，每个有效比赛帧一行。

核心字段：

```text
video_id, frame_idx, timestamp, rally_id, is_court_view
upper_bbox, upper_keypoints, upper_keypoint_scores
lower_bbox, lower_keypoints, lower_keypoint_scores
upper_court_position, lower_court_position
shuttle_raw, shuttle_smoothed, shuttle_confidence, shuttle_state
pose_model_version, shuttle_model_version
```

### 9.4 `events_silver.parquet`

```text
event_id, video_id, rally_id
candidate_frame, window_start_frame, window_end_frame
predicted_hitter, hitter_confidence
trajectory_score, hand_distance_score, pose_score, audio_score
event_confidence
predicted_stroke_type, stroke_probabilities
model_version, review_status
```

### 9.5 `annotations_gold.parquet`

```text
annotation_id, event_id
human_hit_frame, human_hitter, human_stroke_type
backhand, aroundhead
hit_position, landing_position
annotator_id, reviewer_id
status, confidence, notes
created_at, updated_at, revision
```

### 9.6 `dataset_rows.parquet`

最终逐拍训练数据。每一行必须包含 `label_source = gold|silver`，并可追溯到事件、视频、帧数据和模型运行。

## 10. 模型与算法方案

### 10.1 姿态模型

默认：Good-Badminton RTMPose balanced。若远端球员关键点质量不足，使用 performance 档位。

YOLO11n-Pose 作为速度基线，不作为默认高质量 Gold 特征源。

### 10.2 羽毛球模型

- Baseline：Good-Badminton YOLO shuttlecock detector；
- Candidate：TrackNetV3；
- 选择依据：在 Pilot Gold 上比较击球窗口内召回率、连续漏检长度和坐标误差。

### 10.3 击球事件检测

MVP：规则融合 + 时间非极大值抑制。

第二阶段可训练 1D TCN 或小型 Temporal Transformer：

```text
输入：连续帧的轨迹、手腕距离、关节速度和音频特征
输出：每帧是击球点的概率
```

### 10.4 击球类型分类

主模型：BST-CG-AP（BST-3）。

必须保留以下消融基线：

- 仅姿态；
- 姿态 + 羽毛球轨迹；
- 姿态 + 球员位置；
- 完整 BST 输入；
- ST-GCN 基线。

### 10.5 比分与结果

MVP 采用人工录入。OCR 和比分状态机作为后续独立模块，不能阻塞逐拍动作数据集开发。

## 11. 技术架构

### 11.1 本地技术栈

| 层 | 推荐技术 |
|---|---|
| 语言 | Python 3.11 |
| 深度学习 | PyTorch，支持 RTX 50 系列的 CUDA 构建 |
| 姿态 | Good-Badminton RTMPose/RTMO，rtmlib，ONNX Runtime GPU |
| 羽毛球 | Good-Badminton YOLO、TrackNetV3 |
| 视频 | FFmpeg、ffprobe、PyAV、OpenCV |
| 数值处理 | NumPy、SciPy |
| 数据 | PyArrow、Parquet、DuckDB |
| 任务状态 | SQLite |
| 后端 | FastAPI |
| 前端 | React + TypeScript + HTML5 Video/Canvas |
| 模型实验 | PyTorch + TensorBoard 或 MLflow |
| 测试 | pytest |

### 11.2 本机运行策略

本机 RTX 5060 8GB 足以：

- 运行姿态和羽毛球模型推理；
- 微调 BST；
- 从头训练 BST；
- 进行单机批量 Silver 推理。

建议：

- 每次只运行一个重型视觉 worker；
- BST 初始 batch size 16 或 32；
- 开启 mixed precision；
- 将视觉预处理和 BST 训练分阶段执行；
- 不采用 Good-Badminton README 中旧的 PyTorch 2.5.1 + CUDA 12.1 固定组合，先验证当前 PyTorch CUDA 构建对 RTX 5060 的支持。

### 11.3 目录建议

```text
badminton-dataset/
├── apps/
│   ├── api/
│   └── annotator-web/
├── badminton_pipeline/
│   ├── ingest/
│   ├── calibration/
│   ├── pose/
│   ├── shuttle/
│   ├── tracking/
│   ├── rallies/
│   ├── hit_detection/
│   ├── stroke_classification/
│   └── export/
├── configs/
├── data/
│   ├── videos/
│   ├── proxies/
│   ├── frames/
│   ├── events/
│   ├── annotations/
│   └── datasets/
├── models/
├── runs/
├── tests/
└── docs/
```

## 12. API 草案

```text
POST   /videos/import
GET    /videos
GET    /videos/{video_id}
POST   /videos/{video_id}/calibration
POST   /jobs/visual-inference
POST   /jobs/hit-detection
GET    /jobs/{job_id}
GET    /events/next
GET    /events/{event_id}
POST   /events/{event_id}/annotation
POST   /events/{event_id}/reject
POST   /events/{event_id}/split
POST   /events/manual
GET    /clips/{event_id}/frames/{frame_idx}
POST   /training/bst
POST   /inference/silver
POST   /datasets/export
GET    /metrics/annotation
GET    /metrics/model
```

## 13. 标注工作流

### 13.1 单条标注

1. 系统加载下一个候选事件；
2. 自动播放击球前后片段；
3. 标注员检查击球时间；
4. 接受、微调或删除候选；
5. 选择击球者和动作类别；
6. 必要时标记正反手、头顶区和落点；
7. 保存并跳至下一条。

### 13.2 快捷键建议

```text
Space：播放/暂停
← / →：前后 1 帧
Shift + ← / →：前后 5 帧
A / B：选择击球者
1–9：选择粗粒度动作
Enter：接受并保存
X：删除误报
U：不确定
N：下一条
```

### 13.3 主动学习队列

Gold-v1 之后，待标注事件优先级按以下组合排序：

- 模型不确定度高；
- 少数类别概率高；
- 新场馆或新 domain；
- 模型之间预测分歧；
- 轨迹缺失但事件可能真实；
- 随机样本，避免只看困难案例。

## 14. 评估指标与初始验收门槛

以下是 Pilot 后可调整的初始门槛。

### 14.1 视觉数据质量

| 指标 | 初始目标 |
|---|---:|
| 有效比赛帧球员检出率 | ≥ 95% |
| 关键点有效率 | ≥ 90% |
| 球员身份正确率 | ≥ 97% |
| 球场标定人工接受率 | ≥ 95% |
| 击球窗口羽毛球可见或可修复率 | ≥ 85% |

### 14.2 击球事件检测

以人工击球帧 ±5 帧为正确匹配窗口：

| 指标 | 初始目标 |
|---|---:|
| Recall | ≥ 90% |
| Precision | ≥ 70% |
| F1 | 报告，不作为第一阶段硬门槛 |
| 平均击球帧误差 | ≤ 3 帧 |

### 14.3 标注效率

| 指标 | 初始目标 |
|---|---:|
| 相比从零标注的时间降低 | ≥ 40% |
| 事件保存失败率 | < 0.1% |
| 100 条连续标注无界面阻塞 | 必须通过 |

### 14.4 动作分类

第一阶段 9 类：

| 指标 | 初始目标 |
|---|---:|
| Macro-F1 | ≥ 0.75 |
| Top-3 Accuracy | ≥ 0.90 |
| 每类 Recall | 必须报告 |
| 校准误差 | 必须报告 |

18 类不预设过高上线门槛，应以 Gold 规模、类别一致性和基线结果为依据。

### 14.5 Gold 一致性

| 指标 | 初始目标 |
|---|---:|
| 击球者一致率 | ≥ 98% |
| 击球时间 ±3 帧一致率 | ≥ 95% |
| 粗粒度动作 Cohen's κ | ≥ 0.80 |

## 15. 数据集划分与实验设计

严禁随机按击球行拆分相邻数据。

推荐：

- 训练集 70%；
- 验证集 15%；
- 测试集 15%；
- 按整场比赛划分；
- 域适应实验额外按场馆或视频来源划分；
- 同一场比赛不得跨越 train/val/test；
- 研究球员泛化时，同一球员不得同时出现在训练和测试。

核心实验：

1. Gold-only；
2. Gold + 全部 Silver；
3. Gold + 高置信度 Silver；
4. Gold + 类别均衡 Silver；
5. Teacher-Student 半监督；
6. 主动学习；
7. 跨场馆域适应；
8. 姿态、轨迹、位置输入消融。

## 16. 分阶段开发计划

### M0：环境和单视频验证（第 1 周）

- 创建 Python/CUDA 环境；
- 验证 RTX 5060 可用；
- 跑通 Good-Badminton；
- 跑通 BST 官方推理；
- 选择一个 3–5 分钟固定机位单打视频。

完成标准：所有模块可以独立执行并保存结果。

### M1：完整逐帧数据管线（第 2–3 周）

- 扩展 Good-Badminton 完整骨架导出；
- 增加模型置信度和版本；
- 写入 Parquet；
- 完成球场标定和球员身份；
- 比较 YOLO 与 TrackNetV3。

完成标准：一个完整视频可稳定生成 `frames.parquet`。

### M2：击球候选和标注界面 MVP（第 4–5 周）

- 实现规则融合击球检测；
- 生成事件表；
- 开发视频逐帧标注界面；
- 支持接受、调整、删除和补录；
- 完成快捷键。

完成标准：能够连续标注 100 次击球且结果可追溯。

### M3：Pilot Gold（第 6 周）

- 标注 200–500 次击球；
- 测量候选召回率和标注效率；
- 修订动作类别和标注手册；
- 确定正式羽毛球追踪模型。

Go/No-Go：若击球候选召回率低于 80%，暂停扩大标注，优先修复轨迹和事件检测。

### M4：Gold-v1 与 BST 微调（第 7–9 周）

- 标注 2,000–4,000 次击球；
- 双标 10%–20%；
- 微调 BST-CG-AP；
- 完成 9 类分类基线和输入消融；
- 输出混淆矩阵。

完成标准：在比赛隔离测试集上达到可解释的稳定基线。

### M5：Silver 批量生产（第 10–11 周）

- 批量推理未标注视频；
- 置信度过滤；
- 主动学习复核队列；
- 数据集版本化；
- 导出 ShuttleSet-like 表格。

完成标准：Silver 可重新生成，Gold 不被覆盖，所有记录可追溯。

### M6：研究实验与发布（第 12 周及以后）

- Gold-only 与半监督对比；
- 域适应实验；
- 统计标注时间节省；
- 完成数据卡、模型卡和论文实验记录；
- 根据视频授权决定是否发布原视频或仅发布衍生特征。

## 17. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|---|---|---|
| 羽毛球连续漏检 | 击球检测和落点失败 | 比较 TrackNetV3；保存插值状态；优先人工复核低质量回合 |
| 远端球员姿态不准 | 动作分类下降 | 使用 RTMPose performance；扩大 ROI；保存关键点置信度 |
| 18 类标签不一致 | Gold 噪声严重 | 先做 9 类；编写标注手册；双人标注与裁决 |
| 模型学习场馆特征 | 换场馆失效 | 使用骨架/轨迹模型；按场馆划分测试；做域适应 |
| 视频切镜和回放 | 回合边界错误 | 固定机位优先；比赛镜头分类；人工修正 |
| Silver 错误被当作真值 | 研究结论失真 | 强制 `label_source`；保留置信度；Gold 独立测试集 |
| RTX 5060 环境不兼容 | CUDA 无法使用 | 使用支持新架构的 PyTorch/CUDA；先做最小 GPU 验证 |
| 标注范围失控 | 项目无法按期完成 | MVP 只做单打、9 类、人工比分；18 类后置 |
| 视频版权限制 | 无法公开数据集 | 保存授权字段；必要时只发布表格、特征和代码 |

## 18. 隐私、版权与科研规范

- Good-Badminton、BST 和模型权重的许可证分别核验和保留归属；
- 代码开源许可证不自动授予比赛视频再分发权；
- 公开数据集前记录每个视频的来源和授权；
- 不将 Silver 描述为人工真值；
- 论文中明确报告自动标签比例、置信度过滤和人工复核比例；
- 数据删除必须能同步删除其衍生帧、事件和训练索引；
- 发布数据卡，说明采集场景、类别分布、已知偏差和限制。

## 19. MVP 完成定义

满足以下条件即认为 MVP 完成：

1. 至少一个固定机位单打视频可完整处理；
2. 导出完整 COCO 17 点骨架、球员场地坐标和羽毛球轨迹；
3. 自动生成击球候选并达到 Pilot 约定的召回率；
4. 标注员能在界面内修正并保存击球事件；
5. 至少完成 200–500 条 Pilot Gold；
6. BST 官方权重可以在本机推理；
7. BST 可以用自有 Gold 数据完成一次微调；
8. 未标注事件可以生成带置信度的 Silver 预测；
9. 可以导出一份逐拍 Parquet/CSV；
10. Gold、Silver、原始视觉检测和模型版本均可追溯。

## 20. 开发前必须确认的决策

以下决策应在大规模标注前冻结：

1. 第一版具体采用 8、9 还是 10 个粗粒度动作类别；
2. 视频是否全部为固定机位单打；
3. Gold-v1 的目标规模和标注人员；
4. 是否必须标注落点、比分和失分原因；
5. 目标是仅用于内部研究，还是最终公开数据集；
6. 测试集按比赛隔离还是进一步按球员隔离；
7. YOLO 和 TrackNetV3 的 Pilot 比较结果；
8. 动作标签与 ShuttleSet 18 类的最终映射规则。

## 21. 推荐的第一步

不要立即开发完整平台。先完成一个最小纵向切片：

1. 选择一段 3–5 分钟固定机位单打视频；
2. 跑出完整骨架、球员位置和羽毛球轨迹；
3. 人工标注 50 次击球时间；
4. 实现第一版击球候选检测；
5. 评估候选召回率；
6. 用一个极简页面完成 50 条动作标签；
7. 把这 50 条数据转换成 BST 输入格式并跑通训练。

只有这个闭环跑通后，再扩大到正式 Gold 和 Silver 批量处理。

## 22. 参考项目

- Good-Badminton：https://github.com/yo-WASSUP/Good-Badminton
- BST-CG-AP / BST-3：https://github.com/Va6lue/BST-Badminton-Stroke-type-Transformer
- ShuttleSet：https://github.com/wywyWang/CoachAI-Projects/tree/main/ShuttleSet
- TrackNetV3：https://github.com/qaz812345/TrackNetV3
- PyTorch 本地安装：https://pytorch.org/get-started/locally/
