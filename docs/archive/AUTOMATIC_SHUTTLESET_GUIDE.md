# 自动 ShuttleSet-like 数据集构建指南

## 一条主线

```text
视频注册 → 主视角源帧索引（不复制视频） → 场地四角/ Homography
→ 姿态、球轨迹、球员位置 → 击球与比赛进行状态门控
→ BST 球种预标注 → next-contact / terminal-track-end 目的地预标注
→ 单视频质量报告 → 批量合并 → Parquet + CSV + JSONL
```

批量运行示例：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.batch_builder `
  --batch-id broadcast-001 `
  --input .\uncensored_videos `
  --recursive `
  --device cuda `
  --main-view-template .\data\runs\main-view-8e0644\court-template-frame-4882.jpg
```

第一次运行会注册视频、生成主视角源帧清单和自动场地提议，并停在场地复核。确认场地后，
原样重跑即可从状态文件续跑。主视角步骤只保存源帧范围，不生成第二份 MP4。

## 输出含义

- `frames.parquet`：逐帧 COCO-17 姿态、球员米制位置、Top-k 球检测和连续轨迹。
- `dataset_rows.parquet/csv/jsonl`：逐拍事件；包括击球帧、击球方、球种、目的地和完整置信度/来源。
- `quality.json`：覆盖率和异常状态，不是准确率。
- `manifest.json`：输入、模型、SHA-256、行数和所有产物的可追溯清单。
- `batch_state.json`：失败续跑状态；单视频失败不会阻断其他视频。

审计球种时重点查看 `stroke_type_status`、`stroke_type_confidence`、
`stroke_type_input_quality`、`stroke_type_raw_class`、`stroke_type_top3_json`、
`semantic_label_source` 和 `semantic_review_status`。审计目的地时查看 `landing_kind`、
`landing_status`、`landing_proxy_kind`、`landing_confidence` 和 `landing_source`。

## 真值边界

当前产物是可复现的自动预标注，不是未经人工验证的“真值”。BST 上游模型输出 17 个基础
球种加 unknown，缺少 ShuttleSet 官方的 `driven flight`。正常拍目的地使用下次触球附近的
球投影和接球方位置融合；终局拍只在可靠球轨迹明确终止时输出最高 0.5 置信度的实验落点，
否则为空。任何 `automatic_prelabel`、`experimental_terminal_track_end` 或低置信度行都应进入
抽检，而不是直接当作 Gold。

现有人工 Gold 只标注了击球时间/击球方，不能评估球种和米制目的地。若要求“准确真值”，
下一项数据工作不是继续加规则，而是为约 200–500 拍补充球种与目的地人工标签，分开发集和
锁定测试集，再决定是否微调 BST 和训练终局落点模型。
