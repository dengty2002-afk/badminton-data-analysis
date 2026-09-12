# 自动球种与落点：实施路线与真值标准

更新日期：2026-08-14

## 1. 目标和命名

目标输出是：

```text
视频 → 击球事件 → 18 类 ShuttleSet 球种 + ShuttleSet 语义落点
```

必须区分：

- `model_prediction / unreviewed`：模型自动预标注，不是真值；
- `human_semantic_gold / confirmed`：独立人工确认，可作为真值；
- 在固定测试集上达到约定指标后，可以把高置信自动结果称为“验收通过的自动标签”，但仍
  应保留预测来源和置信度，不能伪装成人工 Gold。

ShuttleSet 原始定义有 18 类球种。这里的落点不是简单的“球最后落地”：正常被回击的
一拍，目的地是对手下一次触球的位置；回合最后一拍，目的地才是羽毛球接触地面的位置。

## 2. 当前基础与缺口

已具备：击球候选、击球者、COCO-17 姿态、球员场地位置、羽毛球二维轨迹、场地
Homography、固定窗口 BST 输入和批量数据集导出。

当前缺口：

1. 现有 174 条 Gold 只有击球帧/击球者，没有球种和落点；
2. BST 目前只有输入适配器，没有接入经本项目域内验证的分类权重；
3. 现有球检测存在漏检，且单帧空中球不能直接通过球场 Homography 得到落点；
4. 没有球种宏 F1、落点米制误差和二者联合正确率的基线。

## 3. 已完成的第一阶段

- `evaluation/stroke_semantics.py`
  - 锁定 ShuttleSet 18 类词表；
  - 从现有 hits Gold 生成语义标注模板；
  - 校验球种、米制落点、目的地类型及帧号；
  - 评估球种 Accuracy/Macro-F1、落点 MAE/中位数/P90、0.5m/1m 命中率和联合指标。
- `stroke_classification/semantic_samples.py`
  - 用语义 Gold 而不是机器候选生成 BST 监督样本；
  - 输出姿态、骨架、球轨迹、双方位置、落点目标和缺失掩码。
- 数据集导出
  - 支持球种、落点、置信度、预测来源和人工确认状态；
  - 未提供预测时继续输出 `unknown/null`，不会臆造标签。

生成语义 Gold 模板：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.evaluation.stroke_semantics template `
  --hits-gold data\gold\vid_2e2258f261aa.hits_gold.csv `
  --output data\gold\semantic\vid_2e2258f261aa.semantic_gold.csv
```

语义 Gold 字段：

```text
video_id,rally_id,hit_frame,hitter,stroke_type,
landing_x,landing_y,landing_frame,landing_kind,
confidence,uncertainty_frames,notes
```

其中 `landing_kind` 只能为：

- `next_contact`：该拍被对手回击；
- `ground_contact`：该拍终结回合。

生成监督样本：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.stroke_classification.semantic_samples `
  --frames data\runs\<run>\frames.enriched.jsonl `
  --semantic-gold data\gold\semantic\<video_id>.semantic_gold.csv `
  --output data\training\semantic\<video_id>.npz
```

评估自动预测：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.evaluation.stroke_semantics evaluate `
  --predictions data\predictions\<video_id>.jsonl `
  --gold data\gold\semantic\<video_id>.semantic_gold.csv `
  --output-dir data\evaluation\semantic\<video_id>
```

## 4. 推荐模型路线

### 球种

以 BST 为起点，输入保持为击球帧前后窗口：击球者姿态/骨架、羽毛球轨迹、击球者和
对手场地位置。先验证公开 ShuttleSet 权重在本项目固定机位视频上的零样本迁移，再用
本地训练集微调。必须按视频/比赛划分 train/dev/test，不能随机拆同一视频的逐拍，否则
会造成数据泄漏。

### 落点

不采用“把空中球像素直接做 Homography”的方法。推荐两路融合：

1. **正常被回击的一拍**：下一击球事件的接球者位置 + 下一次触球附近球轨迹，预测该拍
   的 `next_contact` 目的地；
2. **回合终结的一拍**：检测最后一段下降轨迹/地面接触，用场地 Homography 投影；若球
   在落地前长期漏检，则输出低置信或 `null`，不强行生成坐标。

第一版可以做成“目的地区域分类 + 区域内坐标回归”的多任务模型，比直接全场坐标回归
更稳健。球种分类也可共享轨迹和位置编码，但建议先分别建立基线，避免故障归因不清。

## 5. 数据与验收建议

18 类长尾明显。若只标现有 174 拍，很多类别可能没有样本，无法证明准确。建议：

- Pilot：先标 200–300 拍，验证标注界面、词表歧义和落点定义；
- Train：至少 2,000–5,000 拍，并对稀有类定向补样；
- Test：至少 500 拍，按完整视频锁定，任何调参都不能查看测试标签；
- 对部分样本双人复标，报告球种一致率和落点标注差异。

建议首个验收门槛（后续可根据 pilot 难度调整）：

- 击球事件 Recall ≥ 95%，否则后续语义再准也会漏行；
- 球种 Macro-F1 ≥ 0.75，且必须报告每类 F1；
- 落点覆盖率 ≥ 90%，中位误差 ≤ 0.5m，P90 ≤ 1.2m；
- “球种正确且落点误差 ≤ 1m”联合正确率 ≥ 70%；
- 所有未达到置信阈值的结果保留为空或进入人工复核，不能硬填。

## 6. 后续开发顺序

1. 在 Gold Web 中增加“18 类快捷键/下拉 + 球场点击落点”，自动继承已有击球帧；
2. 完成 200–300 拍 Pilot 语义 Gold，并统计类别/落点分布和标注一致性；
3. 接入 BST 官方权重，建立不微调的域外基线；
4. 本地微调并锁定视频级 dev/test；
5. 开发 `next_contact` 落点基线，再开发终局 `ground_contact` 落点；
6. 将通过验收的预测器接入批量构建器；低置信样本进入人工复核队列。

在第 2 步完成前，继续堆复杂模型的收益无法客观判断，因此不建议跳过 Pilot Gold。
