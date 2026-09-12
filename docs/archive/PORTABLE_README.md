# ShuttleLab 便携包说明

本包用于把项目交给另一台 Windows 电脑。它包含源码、配置、测试、三份模型权重、
Good-Badminton 场地检测源码、Web/Gold工具源码，以及小体积的最终结构化数据与评估
记录；不包含私有原视频、Python虚拟环境、`node_modules`、缓存、构建目录和大型运行
中间文件。

## 机器要求

- Windows 10/11 x64
- Python 3.12 x64
- 推荐 NVIDIA RTX GPU；当前已验证环境为 CUDA 模式
- Node.js 22.13 或更高版本（只在需要本地 Gold/Web 工具时使用）
- 安装后请预留约 5 GiB 磁盘；压缩包小，但 PyTorch/CUDA 环境约 3.7 GiB

建议解压到短的英文路径，例如 `D:\ShuttleLab`，可减少第三方工具在中文路径下的日志
编码问题。

## 1. 安装 Python 与可选 Web 环境

在项目根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
```

只安装本地视觉管线、不安装 Web：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -SkipWeb
```

脚本会创建 `.venv-model`。它不会修改系统 Python；如果模型权重未随包提供，会按照
`configs/models.good-badminton.json` 下载并校验 SHA-256。

## 2. 验证完整性

```powershell
.\.venv-model\Scripts\python.exe .\scripts\verify_portable.py --full
```

该命令校验核心文件、模型 SHA-256、最终批次数据 SHA-256、运行依赖，并执行全部测试。

## 3. 处理一段新视频

首先注册视频：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.ingest.register_video `
  --input D:\videos\match.mp4 `
  --data-dir data
```

记录输出的 `video_id`，生成场地提议：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.calibration.auto_court `
  --input D:\videos\match.mp4 `
  --video-id vid_xxxxxxxxxxxx `
  --frame-time-ms 10000
```

查看 `data/court_proposals/<video_id>.court_preview.png`。确认正确后重新执行并加
`--accept`；若不正确，使用 `--manual-corners "x1,y1;x2,y2;x3,y3;x4,y4" --accept`。

运行完整管线：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.run_pipeline `
  --input D:\videos\match.mp4 `
  --calibration data\calibrations\vid_xxxxxxxxxxxx\v1.json `
  --output-dir data\runs\my-run `
  --device cuda
```

## 4. 合并多视频数据集

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.exports.batch_dataset `
  --input-root data\runs\my-batch `
  --output-dir data\datasets\my-batch
```

详细说明见 `DATASET_BUILD_GUIDE.md`。

## 5. 启动本地 Gold 工具

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_gold_ui.ps1
```

然后打开 `http://127.0.0.1:3000/gold`。便携包不含原视频，需在页面或本地视频注册中
选择新电脑上的视频路径。

## 6. 数据含义

- `frames.parquet` 是逐帧视觉/运动学层。
- `dataset_rows.parquet` 是机器逐拍候选层，不是人工真值，也不是完整 ShuttleSet。
- 当前不提供18类击球类型、落点、比分或真实A/B球员身份。
- 最终批次数据用于展示结构与复核流程；来源视频因私有属性不在包中。

完整项目说明和后续计划见 `PROJECT_REVIEW_20260812.md`。
