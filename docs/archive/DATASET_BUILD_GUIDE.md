# 批量数据集与场地标定操作指南

当前只维护两项生产能力：把多个单视频导出合并成 ShuttleSet-like 数据集，以及用
Good-Badminton 自动提议场地四角并由人确认或修正。两项功能均不下载新模型。

## 1. 一次批量任务：注册 → 场地 → 推理 → 校验 → 合并

把待处理视频放在一个目录中，然后运行：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.batch_builder `
  --batch-id august-fixed-camera `
  --input D:\videos\august `
  --device cuda
```

第一次运行会自动注册所有视频并生成场地提议，然后停在
`waiting_court_review`。这是有意设计的安全门：自动四角不能在未经确认时直接进入正式
数据集。查看 `data/court_proposals/*.court_preview.png`，逐个确认后执行汇总文件中给出的
`court_review_command`；自动结果错误时，应为对应命令补充
`--manual-corners "x1,y1;x2,y2;x3,y3;x4,y4"`。

确认完场地后，原样重跑同一条批量命令。构建器将自动跳过已完成的注册和场地阶段，
依次完成模型推理、单视频数据校验和最终合并。核心产物为：

- `data/batch_runs/<batch-id>/batch_state.json`：可续跑的完整状态机；
- `data/batch_runs/<batch-id>/batch_summary.json`：面向人的单视频状态和质量汇总；
- `data/batch_runs/<batch-id>/videos/<video_id>/`：单视频中间结果和数据集；
- `data/datasets/<batch-id>/`：最终合并数据集。

同一 `batch-id` 必须使用相同的视频列表和运行参数。某段视频失败时，其他视频继续；
再次运行同一命令只重试失败/未完成阶段。修改参数时应使用新的 `batch-id`，防止把不同
模型条件的数据混在同一批次。

`batch_summary.json` 会为每段视频给出：

- `completed`：完整且自动质量检查为 `ok`；
- `completed_with_warning`：产物完整，但覆盖率或候选数需要人工抽查；
- `waiting_court_review`：等待确认四角；
- `failed`：记录具体失败阶段与错误，可原命令续跑。

质量 `warning` 不等同于“数据错误”，也不是 Gold 准确率；它是球员/球轨迹覆盖率和空
候选等自动诊断。最终批次会保留这些逐视频质量信息。

构建器不复制原始视频。默认在批次生成文件超过 2 GiB 时写入容量警告，可用
`--workspace-warn-mb` 调整。逐帧 JSONL 是主要空间来源；不要把 `.venv-model`、原视频、
`data/batch_runs` 和最终 ZIP 重复复制到同一个分享目录。

常用选项：

```powershell
# 多个文件/目录可重复传入 --input；递归扫描子目录需显式开启
--input D:\videos\one.mp4 --input D:\videos\set2 --recursive

# 只做少量帧的工程冒烟测试，应使用独立 batch-id
--max-frames 300 --device cpu
```

## 2. 仅合并已有单视频导出（底层工具）

单视频管线应先分别产出 `dataset/manifest.json`。然后运行：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.exports.batch_dataset `
  --input-root data\runs\play-state-prospective-001 `
  --output-dir data\datasets\play-state-prospective-001
```

`--input-root` 可重复传入，也可以直接指向一个单视频 `dataset` 目录。构建器会：

- 校验所有输入的 manifest、SHA-256、schema、行数和唯一键；
- 拒绝重复 `video_id`、重复帧复合键、重复事件复合键或不一致的数据字典；
- 按视频和帧号确定性排序并合并四张 Parquet 表；
- 同时生成事件 CSV/JSONL、整批质量摘要及可追溯 manifest；
- 不复制原始 MP4、逐帧 JSONL 或模型权重。

当前真实批次已生成在 `data/datasets/play-state-prospective-001/`：6 段视频、6,308
帧、207 条事件，全部产物约 4.8 MB。

## 3. 自动/半自动场地边界

先只生成自动提议和预览，不写正式标定：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.calibration.auto_court `
  --input D:\path\to\video.mp4 `
  --video-id vid_xxxxxxxxxxxx `
  --frame-time-ms 10000 `
  --output-dir data\court_proposals
```

查看 `<video_id>.court_preview.png`，确认 1→4 依次为左上、右上、右下、左下，且
绿色四边贴合双打场地外边线。自动提议默认 `accepted=false`，不会污染正式数据。

确认无误后用同一命令加 `--accept`。这会在
`data/calibrations/<video_id>/vN.json` 新建一个版本，不覆盖旧版本：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.calibration.auto_court `
  --input D:\path\to\video.mp4 `
  --video-id vid_xxxxxxxxxxxx `
  --frame-time-ms 10000 `
  --accept
```

如果自动结果不准，传入人工修正后的四角并显式接受：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.calibration.auto_court `
  --input D:\path\to\video.mp4 `
  --video-id vid_xxxxxxxxxxxx `
  --manual-corners "430,289;850,290;1077,653;220,650" `
  --accept
```

自动检测复用 `vendor/Good-Badminton` 的 Apache-2.0 OpenCV 线结构匹配代码。失败时
应人工给四角，不应降低几何校验阈值强行接受。
