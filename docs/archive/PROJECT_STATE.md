# ShuttleLab 项目状态

> 2026-08-16 06:54 运行检查点（`broadcast-uncensored-v1`）：11/24 场唯一原始
> 转播已完成并通过结构校验（均因语义尚非人工 Gold 标记为 warning），第 12 场
> `vid_5bd629a2bddb` 正在 CUDA 推理；其余 12 场排队。14 场待处理视频的场地外边界均已人工审计并保存 accepted
> Homography；总决赛和马来西亚公开赛因机位外观不同，分别改用赛事级主视角模板，
> 修复了全局巴黎模板造成的低覆盖或空片段。马来西亚高机位
> `vid_f289e47c5102` 经 14 个时点抽审确认全程为固定端线机位，100% 选择属实。
> 本轮主视角与标定修复后全量 82 项测试通过。批次生成物约 4.62 GiB，已超过
> 2 GiB 提醒阈值；原视频仍在原处引用、没有复制，D 盘剩余约 76.96 GiB。
> 第 11 场 `vid_aff76f296088` 产出 49,290 帧、1,229 个候选和 102 个候选回合；
> 球轨迹覆盖 91.2%、球种预标注覆盖 80.1%、目的地覆盖 90.9%，但远端球员跟踪
> 只有 75.4%，故列为“可复核但不优先”，不进入首批高可信 Gold 队列。

> 标注不等待整批：`/gold` 当前已有 9 段视频、337 个独立人工击球锚点，可立即并行
> 标球种和落地区域；本机 `http://127.0.0.1:3000/gold` 与同局域网
> `http://localhost:3000/gold` 已验证 HTTP 200。新转播视频按“单视频推理完成 →
> 候选接受/纠帧/删除/补漏 → 球种与落点”滚动进入标注，不把未复核候选写成 Gold。

> 2026-08-16 恢复检查点（`broadcast-uncensored-v1`）：用户已发出“继续”，正式批次
> 已按原命令和 CUDA 环境恢复。仍按 29 个清单文件、24 场唯一原始转播执行，5 个
> `_主视角.mp4` 派生重复项继续排除。恢复时 8/24 场已完成；当前从
> `vid_583aecb64490` 的严格 JSONL 前缀断点续跑，随后处理已接受场地外边界的
> `vid_0697c4032516`。人工标注采用滚动方式：不必等待全批推理，机器候选辅助审核只需
> 等对应单视频完成；完全盲标击球帧可立即开始。击球 Gold 与球种/落点 Gold 分层保存，
> 自动预标注不冒充真值。低质量视频在加入标注队列前必须先通过质量门槛或明确标记。

> 2026-08-14 16:59 暂停检查点（`broadcast-uncensored-v1`）：用户要求暂停，批次
> 进程已安全终止且文件锁可重新获取。输入清单共 29 个文件，其中 24 场唯一原始
> 转播需要处理，`main_view_outputs` 下 5 个旧 `_主视角.mp4` 已标记为
> `excluded_source_variant`，保留文件但不重复推理/合并。当前 8/24 场完成，累计
> 317,165 帧、10,422 条自动逐拍预标注；`vid_583aecb64490` 已完成 8,908/50,513
> 帧并可从 JSONL 前缀续跑；`vid_0697c4032516` 已有人工确认外边界 v1，状态表仍显示
> waiting review，恢复时会自动识别并进入推理。总决赛首段主视角扫描被安全中止，尚无
> manifest，恢复时重做。最终 24 场合并尚未生成。批次目录 3.91 GiB，D 盘剩余
> 77.53 GiB。恢复命令见本次任务记录；恢复前无需清理任何缓存。

> 本次新增可靠性改动：严格 JSONL 断点续写/末行截断与 source-frame seek；标定版本
> 变化时复用 raw 仅重算后处理；自动场地提议增加“疑似把单打内线当 6.1 m 双打外界”
> 对照评分；同场派生主视角文件自动去重。全量 80 项测试通过。巴黎首段使用 v3，丹麦
> 决赛与伍家朗视频使用 v2，均已按双打外边界重算完成。

> 2026-08-12 便携整理：完整回顾见 `PROJECT_REVIEW_20260812.md`，新电脑安装见
> `PORTABLE_README.md`。便携包只保留源码、模型、最终结构化数据和必要台账，排除
> 3.67 GiB Python环境、Node依赖/缓存、私有视频和481 MiB运行中间产物。

> 2026-08-12 增量：当前开发范围收敛为两项。批量 ShuttleSet-like 构建器已完成，
> 真实 6 视频批次合并为 6,308 帧和 207 条事件，总计约 4.8 MB；Good-Badminton
> 自动场地四角已接入为“提议→人工确认/修正→版本化 Homography”，自动结果默认不落
> 正式标定。操作与恢复命令见 `DATASET_BUILD_GUIDE.md`。其余 Web、动作分类、落点和
> Silver 扩展均不在当前范围。

最后更新：2026-08-10

这是项目恢复工作的唯一事实来源。每完成一个可验证阶段，应同步更新本文件、`DECISIONS.md` 和 `CHANGELOG.md`，不依赖聊天记录保存执行细节。

## 当前目标

先跑通轻量本地视觉数据管线：

```text
固定机位单打视频
  -> 视频登记与校验
  -> 四角标定 / Homography
  -> 球员检测与 COCO-17 姿态
  -> 羽毛球检测与轨迹平滑
  -> 球员场地坐标与短时追踪
  -> 击球候选与回合候选
  -> ShuttleSet-like Parquet / CSV / JSONL
```

本阶段不继续扩展 Web 界面、多人审核、Silver 批量生产或动作分类训练。

## 当前实现状态

| 模块 | 状态 | 产物或实现 |
|---|---|---|
| 视频登记 | 完成 | SHA-256、元数据、`data/videos/*.json` |
| 场地标定 | 完成（人工四角） | 版本化 JSON、Homography |
| 球员/姿态 | 完成 | YOLOX nano + RTMPose ONNX，COCO-17 |
| 羽毛球检测 | 完成 | YOLO11s ball，当前为 PyTorch/Ultralytics 运行时 |
| 球员位置 | 完成 | Homography 场地坐标、upper/lower、短时补全 |
| 球轨迹 | v0.2开发版 | top-k检测、时序关联、创新残差/不确定度审计、击球感知软门控、最多5帧速度外推；明确不做插值 |
| 击球候选 | v0.3已过holdout；v0.6前瞻通过 | v0.4扩展约±2帧手腕证据，v0.5接入鲁棒轨迹，v0.6在候选生成后应用可审计的自动回合运动门控 |
| 自动回合运动门控 | v0.1前瞻通过（可审计） | 以短窗口球速、轨迹连续性、测量可信度和交替击球序列建立回合活动区间；不做人工四类阶段分类，保留 provisional、kept、suppressed 三份事件 |
| 回合候选 | 可运行的规则基线 | 6 秒无击球间隔分割；本份 Gold 无跨回合边界，因此未调参 |
| 数据集导出 | 完成 v0.1 | `videos.parquet`、`calibrations.parquet`、`frames.parquet`、`dataset_rows.parquet/csv/jsonl`、数据字典、质量报告和 manifest |
| 本地 QA 视频 | 完成 | 640×360 全长叠加视频，含球场、骨架、场地坐标、球轨迹和68条 v0.4开发候选提示 |
| Gold 参考视频 | 完成 | 只显示原画面、帧号和时间，隐藏全部机器结果，避免人工真值被候选带偏 |
| 本地 Gold 标注界面 | 完成、已产出首份 Gold | `/gold`：播放/逐帧/慢放、U/L/K记录、新回合、撤销、本地自动保存和CSV导出 |
| 候选评估器 | 完成并运行 | Gold 单调一对一匹配；同时输出严格时序指标和含人工反应余量的稳健存在性指标，不自动平移 Gold |
| BST | 仅输入适配器 | 30 帧 NPZ；动作标签仍为 `unknown` |
| Web | 冻结于 v13 | 后续仅作为标定与人工质检控制台 |

## 已验证结果

- 单元测试：51 项通过（2026-08-10）；Web生产构建通过。
- 真实 GPU 数据集冒烟：案例视频第 380–439 帧，共 60 帧、3 条击球候选。
- 冒烟总产物约 0.60 MiB；`frames.parquet` 为 60 行 × 30 列，保留嵌套 COCO-17 骨架。
- 新运行统一使用规范 ID `vid_<sha256前12位>`；旧的无前缀 24 位 ID 仍兼容读取。
- 全长案例视频运行完成：1841 帧、初始规则 v0.2 产生 78 条候选、约 80 秒，推理约 24 FPS；产物位于 `data/runs/full-local-v0.1/`。
- 全长覆盖率：upper 追踪 93.70%、lower 追踪 100%、羽毛球轨迹 93.92%；upper/lower 场内坐标率分别为 96.61%/100%。这些是自动覆盖率，不代表相对人工真值的准确率。
- 全长运行产物约 17.74 MiB；当前 `data/` 合计约 20.61 MiB。
- 人工 Gold 共 58 条击球，帧号严格递增，upper/lower 各 29 条且交替，均声明 ±1 帧不确定度。
- 初始规则 v0.2 在主稳健窗口（基础 ±5 帧 + Gold ±1 帧 + 反应余量 6 帧）下：51/58 命中，Precision 65.38%、Recall 87.93%、F1 75.00%；中位有符号误差 -3 帧。
- Gold 校准后的规则 v0.3 产生 68 条候选；同一稳健窗口下：52/58 命中，Precision 76.47%、Recall 89.66%、F1 82.54%，误报 16、漏报 6；中位有符号误差 -3 帧。严格基础 ±5 帧加 Gold ±1 帧时 F1 为 63.49%。
- v0.3 结果属于单视频开发集调参成绩，不是独立泛化成绩；下一份视频必须作为不调参的 holdout 验证集。
- 第二段 holdout `vid_a598228fbc11` 已登记：1993帧、30 FPS、约66.43秒。抽查四个时间点确认与首段视频为相同固定机位，因此复制其已接受四角几何并保存为独立标定记录。
- holdout 已使用冻结的 `hit-rules-0.3.0` 完成机器运行和79行数据集导出，产物位于 `data/runs/holdout-a598-v0.3/`；没有生成机器提示 QA 视频。
- holdout Gold 已锁定：64条击球，upper/lower各32条，SHA-256 `82798b2e8780ce78e76143d67f15d11715466f7e75c894822ca761d7eef38a0a`。
- 冻结 v0.3 已完成唯一一次正式 holdout 评估。在主稳健窗口（基础±5 + Gold±1 + 反应余量6）下：59/64命中，Precision 74.68%、Recall 92.19%、F1 82.52%，中位有符号误差-6帧；与开发视频F1 82.54%基本一致。
- holdout 击球者 upper/lower 准确率为64.41%，明显低于开发视频84.62%；5个漏检全部为upper。该视频 upper 实测姿态率76.12%，lower为99.90%，远端姿态/球员关联是下一开发集需调查的问题。
- 严格主窗口（基础±5 + Gold±1，不含反应余量）F1为47.55%，说明候选存在性可以泛化，但精确击球帧定位仍受人工反应差异或候选峰值定义影响。
- v0.4开发版只在首段开发视频上验证：68个候选帧与v0.3完全相同，4条击球者归属发生变化；主稳健击球者准确率84.62%→86.54%，击球存在性F1保持82.54%，时序中位偏移保持-3帧。v0.4没有在已锁定holdout上重跑。
- v0.4新验证视频 `vid_576b0c1ab370` 已登记：1745帧、30 FPS、约58.17秒。四个抽查帧确认与前两段为同一固定机位，已保存独立标定记录。
- v0.4 validation Gold锁定为41条；原视频QA发现并由用户确认第1034帧击球侧误按，已从upper改为lower。锁定SHA-256为`b13f03d7a794596791e01861dccb9b1616d1e1bd15f345fc421f377a0de9aac3`。
- v0.4唯一一次正式验证：59条候选对41条Gold；主稳健窗口下37/41命中，Precision 62.71%、Recall 90.24%、F1 74.00%，击球者准确率83.78%，中位有符号误差+3帧；严格主窗口F1为56.00%。参数未修改。
- 22个误报中9个在第一条Gold之前、2个在最后一条Gold之后。只作诊断的Gold首尾oracle裁剪后F1为83.15%，说明主要新增问题是比赛进行状态门控；该oracle数字不是可部署指标，也不能替代正式74.00%。
- 4个漏检再次全部来自upper。v0.4击球者归属在独立验证中达到83.78%，接近开发集86.54%，邻域手腕证据得到初步支持；候选存在性规则仍需加入非比赛时段抑制。
- 比赛状态证据层 `play-state-evidence-0.1.0` 已接入后处理和 ShuttleSet-like 导出。开发集仍为68条候选，候选帧与击球者均未改变，自动过滤数为0；主稳健F1仍为82.54%、击球者准确率仍为86.54%。
- 开发集上，52条匹配候选的比赛状态分数中位数为0.946，16条误报为0.892；存在区分趋势但明显重叠，因此当前分数仅供审计，不设置生产过滤阈值。
- 轨迹鲁棒性诊断发现：开发视频相邻原始检测最大跳变约433像素，旧alpha-beta平滑后最大仍约266像素；大跳变附近同时存在真实击球和误报，说明不能用固定最大位移直接删除。改造前逐帧只保留最高置信度检测，且滤波器未对异常创新残差降权。
- `shuttle-track-0.2.0-dev` 已完成真实GPU全长开发运行：1841帧中153帧含多个球候选，时序关联20帧选择非最高置信度候选，45帧触发软抑制；未应用插值。新运行位于`data/runs/full-local-v0.2-track-dev/`。
- 相对v0.4轨迹基线，平滑轨迹单帧位移P95从79.89降至56.60像素、P99从157.13降至87.84像素、最大值从265.69降至173.26像素；真实击球附近的大变向通过手腕/姿态证据保留。
- 开发集击球评估：候选68→67，稳健Precision 76.47%→77.61%、Recall保持89.66%、F1 82.54%→83.20%；严格F1 63.49%→65.60%；误报16→15、漏报保持6、击球者准确率保持86.54%。该结果仍是开发集成绩，尚未晋升或在锁定集上重跑。
- `play-state-gate-0.1.0-dev` 与 `hit-rules-0.6.0-dev` 已接入完整管线。门控先生成 provisional 候选，再由候选序列支持、0.25–4秒的交替击球者链接及连续球飞行证据建立 active-play 区间；区间外候选写入 `events.suppressed.jsonl`，全部门控前候选写入 `events.provisional.jsonl`，不静默删除。
- 开发视频门控回放：67条 provisional 中保留66条、抑制1条准备期误报；主稳健Precision 77.61%→78.79%，Recall保持89.66%，F1 83.20%→83.87%，误报15→14，漏报保持6。
- 冻结参数后的回顾性验证：`vid_576b0c1ab370` 58条 provisional 中抑制3条，三者均为误报；主稳健Precision 62.07%→65.45%、Recall保持87.80%、F1 72.73%→75.00%。`vid_a598228fbc11` 76条 provisional 均位于推断 active-play 区间，门控未改变结果。两者用于门控的事后安全审计，不是新的前瞻盲测成绩。
- 原锁定 holdout 事件、validation 事件和 validation Gold 未被覆盖，SHA-256 仍分别为 `a4f518a8c8541e8a3a187d91cdc002bf73e2a612f188259b062e6907ee5ab6ae`、`b7aeeed4614ea5a21ec201afbd6b61afb6025e08b82dffa40e991e2b1ab68173`、`b13f03d7a794596791e01861dccb9b1616d1e1bd15f345fc421f377a0de9aac3`。
- 新前瞻批次`play-state-prospective-001`包含6段同机位回合视频，共6308帧、210.80秒；源视频合计约39.9 MiB，保留在原微信目录，仅登记路径和SHA-256，不复制进工作区。`data/videos.json`已从9份独立记录重建并纳入全部历史/新视频。
- 六段视频10%/50%/90%抽帧确认与案例视频为同一1280×720固定机位，分别保存独立的v1标定复用记录。冻结v0.6参数后全批GPU推理完成：226条provisional、207条kept、19条suppressed、7个active区间；总运行约280.57秒，产物约65.36 MiB。用户已知批次汇总数，但候选帧、逐视频数量、击球者、置信度和门控决定保持隐藏，因此本批次是frame-blind而非完全信息盲测。
- 本地`/gold`已升级为六视频批次盲标：开发服务按已登记`video_id`自动读取原始MP4并支持Range拖动，不上传/复制视频；若登记路径失效仍可用文件选择兜底。每段独立localStorage，只支持U/L/K击球Gold和整批导出；人工阶段分类、阶段快捷键与阶段CSV已删除。
- 六视频批次已完成并锁定174条人工击球标注；逐视频击球CSV格式、视频ID、帧范围、重复帧和SHA-256校验通过。唯一一次击球前瞻评估已完成：自动运动门控使稳健Precision 66.37%→71.98%、Recall 86.21%→85.63%、F1 75.00%→78.22%，抑制18个误报和1个首击真候选；本批不再调参或重跑。
- 全长 QA 视频 `qa-overlay.mp4` 已按 v0.4开发版重建，为1841帧、68条候选、约20.68 MiB；无机器提示的720p Gold参考视频为36.06 MiB。当前 `data/` 约99.16 MiB。

## 数据契约

- 图像坐标：像素，原点在左上。
- 场地坐标：米，标准单打/双打完整球场平面为 6.1m × 13.4m。
- `player` 在无人名映射时保存可靠的 `upper|lower` 场侧身份，不冒充 A/B 球员身份。
- 动作类型、击球点、落点等无法可靠生成的字段保存为 `unknown` 或 `null`。
- 飞行中的羽毛球不在场地平面上，不使用 Homography 伪造其场地坐标。
- 每次数据集导出生成 SHA-256 manifest，保证来源和产物可追溯。

## 当前模型与环境

- RTMPose ONNX：约 20.9 MiB。
- YOLO11s ball `.pt`：约 18.3 MiB。
- YOLOX ONNX：约 3.5 MiB。
- `.venv-model`：约 3.58 GiB，主要体积来自 PyTorch。
- 整个工作区：约4.79 GiB，已经超过2 GiB提醒阈值；当前`data/`约486.17 MiB。最大目录是`data/runs/main-view-8e0644/`约254.72 MiB；新六视频前瞻批次约65.36 MiB。
- 未经确认，不下载 500 MiB 以上模型或生成 500 MiB 以上单个产物。

## Web 状态

- 公开地址：https://shuttlelab-badminton-demo-0806.dengty2002.chatgpt.site/
- 已发布版本：v13。
- Demo Git 提交：`90189acdfaf3e3eae3155d460550012ce73d0079`。
- 当前定位：可选人工质检控制台；本地 Python 管线是数据生产主线。
- 当前访问策略：公开可写（用户已明确接受）；在重新进入人工标注阶段前评估数据污染和权限风险。
- `/gold` 是独立的本地盲标工具：不读取机器候选、不写 D1，只使用浏览器本地存储和下载导出；本轮未发布到公开站点。
- 本地标注页已由应用内浏览器验证可用，但当前桌面运行环境会在对话轮次结束时回收开发服务，因此 `localhost` 只适合短时测试，不能作为持续标注入口。可靠使用需要在用户明确批准后发布 `/gold`；该页面仍只保存到浏览器本地，不写 D1。

## 版本控制

- 工作区根目录当前不是 Git 仓库，本地 Python 管线的改动尚无提交历史保护。
- `demo/` 是独立 Git 仓库；当前含尚未提交的holdout标注页与视频副本改动。
- 下一次结构性开发前建议为根目录初始化独立 Git 仓库；执行前由用户确认。

## 下一步

1. 接受 v0.3 作为已通过一次holdout的“击球存在性候选”基线；不把它当成精确时序或可靠击球者识别模型。
2. v0.4邻域手腕证据在独立validation上获得83.78%击球者准确率，可保留；正式存在性F1为74.00%，下降主要来自第一条Gold前和最后一条Gold后的11个误报。
3. `shuttle-track-0.2.0-dev` 已完成开发集实现与验证；下一步在新的开发片段做轨迹困难区间抽样QA，再决定是否冻结参数并使用新的盲标视频验收，不在既有锁定集上重跑。
4. 六视频前瞻批次已完成：174条击球Gold与6份CSV已锁定，唯一一次纯击球评估已执行。四类阶段标注被取消；自动回合门控保留为precision过滤器，本批不再调参或重跑。
5. 用包含多个真实回合并正确按R分段的Gold视频评估6秒回合分割规则；当前三份Gold都只有一个回合，不能支持调整该规则。
6. 若基本视觉数据满足需求，评估将球模型转换为ONNX，以移除约2.8 GiB的PyTorch运行时。
7. 本地闭环稳定后，再精简Web并接入“导出人工修正 -> 合并最终数据集”。

## 常用命令

```powershell
# 单元测试
.\.venv-model\Scripts\python.exe -m unittest discover -s tests -v

# 完整本地管线（默认导出数据集）
.\.venv-model\Scripts\python.exe -m badminton_pipeline.run_pipeline `
  --input example.mp4 `
  --calibration data\calibrations\vid_2e2258f261aa\v1.json `
  --output-dir data\runs\my-run `
  --device cuda

# 对已有 JSONL 重新导出，不重复执行模型
.\.venv-model\Scripts\python.exe -m badminton_pipeline.exports.dataset `
  --frames data\runs\my-run\frames.enriched.jsonl `
  --events data\runs\my-run\events.jsonl `
  --calibration data\calibrations\vid_2e2258f261aa\v1.json `
  --video-record data\videos\vid_2e2258f261aa.json `
  --output-dir data\runs\my-run\dataset

# 生成轻量本地 QA 叠加视频
.\.venv-model\Scripts\python.exe -m badminton_pipeline.visualization.qa_overlay `
  --video example.mp4 `
  --frames data\runs\full-local-v0.1\frames.enriched.jsonl `
  --events data\runs\full-local-v0.1\events.jsonl `
  --calibration data\calibrations\vid_2e2258f261aa\v1.json `
  --output data\runs\full-local-v0.1\qa-overlay.mp4 `
  --scale 0.5

# 生成独立 Gold 参考视频（隐藏所有机器结果）
.\.venv-model\Scripts\python.exe -m badminton_pipeline.visualization.qa_overlay `
  --video example.mp4 `
  --frames data\runs\full-local-v0.1\frames.enriched.jsonl `
  --events data\runs\full-local-v0.1\events.jsonl `
  --calibration data\calibrations\vid_2e2258f261aa\v1.json `
  --output data\runs\full-local-v0.1\gold-reference.mp4 `
  --mode gold

# Gold 填写完成后评估候选
.\.venv-model\Scripts\python.exe -m badminton_pipeline.evaluation.hit_events `
  --candidates data\runs\full-local-v0.1\events.jsonl `
  --gold data\gold\vid_2e2258f261aa.hits_gold.csv `
  --output-dir data\runs\full-local-v0.1\evaluation-hit-rules-0.3.0 `
  --reaction-allowance-frames 6
```
