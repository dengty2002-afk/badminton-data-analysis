# 模型管线（当前切片）

当前已接入 Good-Badminton v0.1.0 的三份官方权重：YOLOX 人体检测、RTMPose-S balanced 姿态和 YOLO11s 羽毛球检测。权重位于本地 `models/good-badminton/`，由 manifest 中的文件大小和 SHA-256 双重校验，不进入 Git。

统一输出为逐帧 JSONL。每帧包含视频与帧索引、时间戳、完整 COCO 17 点坐标与逐点置信度、缺失标记、人物框、固定机位单打的 `upper/lower` 场侧身份、羽毛球原始坐标/框/置信度/状态，以及模型 release 和权重校验值。旁路 metadata JSON 记录运行环境和耗时。

运行入口：

```powershell
python -m badminton_pipeline.models.infer_video `
  --input example.mp4 `
  --output data/runs/example/frames.jsonl `
  --device cuda `
  --max-frames 100
```

可以先加 `--skip-shuttle` 运行较小的 ONNX 姿态切片；此模式不要求 PyTorch/Ultralytics，羽毛球字段会明确写为 `state=not_run`，不会伪装成模型漏检。

CUDA 模式在创建 RTMPose 会话前加载并验证 PyTorch CUDA runtime；如果 ONNX Runtime 实际回退到 CPU，管线会直接报错，禁止把 CPU 结果错误记录为 CUDA。

已验证运行环境：PyTorch 2.13.0 + CUDA 13.0、ONNX Runtime GPU 1.28.0、Ultralytics 8.4.116，RTX 5060 的计算能力为 12.0，运行时包含 `sm_120`。案例视频已完成 3 帧联合烟雾测试：每帧均输出两个球员的完整 COCO 17 点，并输出 YOLO 羽毛球框、中心点和置信度。

## Homography、追踪与击球候选

`badminton_pipeline.tracking.postprocess` 读取原始模型 JSONL 与已接受的标定 JSON，完成：

- 保留多个人体候选，通过脚点投影到 6.1m × 13.4m 标准球场，过滤裁判和场外人员；
- 在远端/近端半场各选择一名球员并写入 `upper/lower`；
- 使用 α-β 时序滤波平滑羽毛球坐标，短时漏检写为 `predicted`，长时漏检保持 `missing`；
- 融合轨迹转向/变速、羽毛球到手腕距离和手腕运动幅度，输出高召回击球候选。

球员场地轨迹使用 EMA 平滑，短时姿态漏检最多延续 5 个源视频帧，并明确标记为 `measured`、`predicted` 或 `missing`。候选事件保存双方的 `player_locations`、羽毛球图像坐标和跟踪状态。由于飞行中的羽毛球不在球场平面上，事件中的 `shuttle_court_xy_m` 固定为 `null`，避免错误使用 Homography 伪造场地落点。

击球候选按时间排序后使用 6 秒无击球间隔提出回合边界，写入 `rally_id`、边界来源、与上一拍间隔和 `candidate` 状态。该结果仅是高召回候选；标注界面允许在任意事件前拆分回合或与上一回合合并，人工修改会继续追加到 D1 事件修订历史。

模型层始终保留稳定的 `upper/lower` 场侧身份，前端再按视频和生效帧把场侧解析为真实球员 A/B。人工点击“记录换边”会向 D1 追加一个场侧映射版本；新映射只重算尚未审核的机器候选，既有 Gold 与历史修订保持不变。

标注界面支持直接点击视频画面或俯视球场补充击球点与落点。视频点击通过最新已接受的 Homography 转为 6.1m × 13.4m 球场坐标；两类点位都会写入同一条 D1 事件修订，并随 CSV 与 Pilot 报告导出。无法可靠确定时保持为空，不生成伪造坐标。

Gold 审核在原有事件修订之上增加 `double_checked` 与 `adjudicated` 状态。每次独立复核或裁决都会记录复核人、基线修订、意见和时间；已双检记录只能继续裁决，已裁决记录不能回退。登录用户执行独立复核时，服务端会阻止标注者复核自己的记录。

一致性报告会将独立双检修订与其 `review_base_revision` 配对，计算击球者一致率、击球帧 ±3 帧一致率、动作一致率和粗粒度动作 Cohen's κ，并列出可追溯到具体版本的分歧项。样本不足时指标保持为空，不用零值制造虚假结论。

案例视频第 320–499 帧验证结果：180 帧约 8.1 秒完成 GPU 推理；平均每帧约 4 个人体候选，146 帧同时选中两名球员，已选球员脚点 97.55% 位于球场范围内，并产生 10 个待人工审核候选。输出文件合计约 1–2 MB。

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.tracking.postprocess `
  --frames data/runs/example/frames.rally-smoke.raw.jsonl `
  --calibration data/calibrations/vid_2e2258f261aa/v1.json `
  --output-frames data/runs/example/frames.rally-smoke.enriched.jsonl `
  --output-events data/runs/example/events.rally-smoke.jsonl `
  --fps 30
```

完整管线可以用一条命令运行：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.run_pipeline `
  --input example.mp4 `
  --calibration data/calibrations/vid_2e2258f261aa/v1.json `
  --output-dir data/runs/my-run `
  --device cuda
```

输出目录包含 `frames.raw.jsonl`、`frames.enriched.jsonl`、`events.jsonl` 和汇总运行元数据 `run.json`。运行前会核验标定状态与视频 ID，避免将其他视频的 Homography 误用到当前素材。

将候选事件同步为前端审核队列：

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.exports.demo_feed `
  --events data/runs/my-run/events.jsonl `
  --output demo/public/data/model-candidates.json
```

## BST-CG-AP 输入适配

`badminton_pipeline.stroke_classification.features` 将逐帧增强数据与事件表转换为固定 `T=30` 的压缩 NPZ。每个样本包含击球者 COCO 17 点与置信度、16 条骨骼向量、羽毛球平滑坐标与置信度、击球者/对手球场位置，以及每一种模态独立的缺失掩码。图像坐标按宽高归一化，球场坐标按 6.1m × 13.4m 归一化；补零值只有在对应 mask 为 `true` 时才允许进入模型，避免把漏检伪装成真实坐标。

```powershell
.\.venv-model\Scripts\python.exe -m badminton_pipeline.stroke_classification.features `
  --frames data/runs/example/frames.rally-smoke.enriched.jsonl `
  --events data/runs/example/events.rally-smoke.jsonl `
  --output data/runs/example/bst-input-smoke.npz `
  --window-size 30
```

也可以在完整管线命令末尾增加 `--export-bst-inputs`。案例切片生成 10 个样本，NPZ 约 63 KiB；窗口帧覆盖率 96%，击球者姿态关键点覆盖率约 84.25%，羽毛球有效轨迹覆盖率 96%。当前候选事件没有人工动作标签，因此标签明确保存为 `unknown`；必须接入 Gold 导出后才能用于监督训练。

当前本地模型虚拟环境约 3.58 GiB，工作区约 4.27 GiB，已经超过配置中的 2 GiB 提醒阈值。后续开发不应复制虚拟环境或重复下载权重；逐帧结果也应按运行批次清理或转移到外部存储。

复现安装顺序：

```powershell
python -m venv .venv-model
.\.venv-model\Scripts\python.exe -m pip install -r requirements-models-onnx.txt
.\.venv-model\Scripts\python.exe -m pip install --no-deps rtmlib==0.0.16
.\.venv-model\Scripts\python.exe -m pip install -r requirements-models-cuda.txt --index-url https://download.pytorch.org/whl/cu130
.\.venv-model\Scripts\python.exe -m pip install -r requirements-models-app.txt
.\.venv-model\Scripts\python.exe -m pip install --no-deps ultralytics==8.4.116
```
