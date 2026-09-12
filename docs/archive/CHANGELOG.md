# Changelog

## 2026-08-14 — Stroke type and destination foundation

- Added the official 18-class ShuttleSet stroke vocabulary and semantic Gold CSV contract.
- Added supervised semantic BST sample export with destination targets and masks.
- Added type, metric destination and joint semantic evaluation reports.
- Extended dataset rows with destination frame/kind/confidence and semantic provenance.
- Kept unverified automatic values explicitly marked as predictions rather than human truth.

## 2026-08-14 — Resumable batch dataset builder v0.1

- Added one main batch command for register → court review → inference → per-video validation → merge.
- Added atomic per-stage state, failed-stage retry, configuration locking and per-video quality summaries.
- Kept court acceptance as an explicit human gate; automatic proposals never silently become formal calibration.
- Added generated-size reporting and a configurable 2 GiB warning threshold without copying source videos.
- Added batch orchestration tests; the full suite now covers 58 tests.

## Unreleased — local-visual-v0.1

### Added

- 便携项目发布流程：项目回顾、移机指南、Windows环境重建、模型下载/哈希校验、包内完整性测试与精简ZIP构建。
- 批量 ShuttleSet-like 构建器：校验输入 SHA-256/schema/行数/复合键，确定性合并多视频 Parquet、CSV、JSONL、质量摘要与 manifest。
- Good-Badminton 自动场地四角适配器：从视频抽帧生成可视 proposal，显式确认或人工修正后才新增版本化 Homography。
- 真实 6 视频批次 `data/datasets/play-state-prospective-001/`：6,308 帧、207 条事件、约 4.8 MB。
- ShuttleSet-like 本地数据导出器。
- `videos.parquet`、`calibrations.parquet`、`frames.parquet` 和 `dataset_rows.parquet/csv/jsonl`。
- 数据字典、行数/来源/文件大小/SHA-256 manifest。
- 球员追踪、姿态关键点、场内坐标、球轨迹与事件分布自动质量报告。
- 空事件数据集与真实嵌套 COCO-17 Parquet 的测试覆盖。
- `PROJECT_STATE.md` 与 `DECISIONS.md` 项目恢复台账。
- 本地 QA 叠加视频渲染器：球场四角、COCO-17、球员场地坐标、球轨迹和击球候选提示。
- 无机器提示的 Gold 参考视频模式，避免候选结果影响人工真值。
- 本地盲标界面 `/gold`：视频播放、逐帧、慢放、U/L/K击球快捷键、回合切换、撤销、删除、浏览器自动保存和Gold CSV导出。
- `hits_gold.csv` 真值模板、填写说明和严格输入校验。
- 击球候选一对一最优匹配评估器，支持人工不确定帧范围和多档容差。
- 人工反应延迟稳健评估：严格/稳健双轨指标、显式反应余量、有符号偏移、MAD、P90残差与歧义匹配计数；不自动平移Gold。
- `matched_events.csv`、`false_positives.csv`、`false_negatives.csv` 评估产物。
- 首份完整 Gold：58 条人工击球，upper/lower 各29条，逐条 ±1 帧不确定度。
- 第二段 holdout 视频登记、固定机位标定复用记录和冻结参数协议。
- `/gold` 已切换到 `vid_a598228fbc11`，使用独立帧范围、CSV文件名和浏览器本地存储。
- 第三段v0.4验证视频登记、固定机位标定复用记录、独立Gold模板和冻结参数协议。
- `/gold` 已切换到 `vid_576b0c1ab370`，候选数量与机器预测在Gold锁定前保持隐藏。
- 可审计的 `play-state-evidence-0.1.0`：逐候选输出局部球可见/追踪/运动、双方球员追踪和候选序列支持证据；当前仅打分，不过滤。
- ShuttleSet-like 事件行和质量报告中的比赛状态证据字段与分数分位数。
- 羽毛球检测top-k候选保存与向后兼容的单候选读取。
- `shuttle-track-0.2.0-dev`时序关联、创新残差、动态门宽、测量权重/状态、轨迹不确定度和击球证据审计字段。
- ShuttleSet-like帧表中的候选JSON、关联点、预测点、创新残差、软门控状态和质量分位数；明确记录未应用插值。
- `play-state-gate-0.1.0-dev`：使用候选序列支持、交替击球者和连续球飞行证据建立active-play区间。
- 门控审计三件套：`events.provisional.jsonl`、最终`events.jsonl`和`events.suppressed.jsonl`；逐帧导出比赛状态与区间ID。
- 视频总目录重建工具：从独立`data/videos/vid_*.json`校验并重建`data/videos.json`，拒绝重复ID或SHA-256。
- 六视频本地批次盲标：不复制/上传源MP4，支持视频切换、独立本地存储和击球Gold整批导出。

### Changed

- Gold协议精简为纯击球标注；回合活动由归一化球速、轨迹连续性和击球序列自动推断。
- 六视频锁定前瞻评估中，自动运动门控将稳健F1从75.00%提升至78.22%，以0.57个百分点召回损失换取5.61个百分点precision提升。

- 完整管线默认在运行目录的 `dataset/` 下生成最终数据产物。
- 新运行的视频 ID 统一为 `vid_<sha256前12位>`。
- 开发顺序调整为本地视觉闭环优先，Web v13 暂时冻结。
- 击球规则升级到 `hit-rules-0.3.0`：默认阈值 0.38，候选去重间隔从固定帧数改为按帧率计算的 0.4 秒。
- 全长案例数据集按 v0.3 重导出为 68 行事件，并重建 QA 叠加视频；保留 `events.hit-rules-0.2.0.jsonl` 作为基线。
- 新增 `hit-rules-0.4.0` 开发版：使用候选前后约2帧的最强手腕接近证据判断击球者，不移动事件帧；保存v0.3事件归档并重建开发数据集与QA视频。
- 后处理管线接入1秒比赛状态证据窗口和3秒候选序列窗口，明确保持 `play_state_filter_applied=false`。
- 击球候选开发版升级为`hit-rules-0.5.0-dev`：轨迹分数按追踪可靠性软加权，接触/击球者证据使用已接受的关联检测点，不把滤波滞后点冒充接触点。
- 击球候选开发版升级为`hit-rules-0.6.0-dev`：完整管线默认启用比赛进行状态门控；可用`--play-state-gate-mode off`无损关闭。

### Removed

- `/gold`中的`pre_serve|active_play|post_rally|waiting`四类人工阶段按钮、快捷键和阶段CSV导出。
- 人工阶段Gold逐帧评估器及其测试；项目不再把比赛阶段分类作为产品目标。

### Verified

- 50项Python单元测试与Web生产构建通过；六视频174条击球Gold已锁定SHA-256并完成唯一一次纯击球前瞻评估。

- 23 项 Python 单元测试通过。
- 案例视频 60 帧真实 CUDA 冒烟通过，生成 60 行帧数据和 3 行击球候选。
- 案例视频全长 1841 帧 CUDA 运行通过，约 80 秒并生成 78 条候选。
- 全长自动覆盖率：upper 球员 93.70%、lower 球员 100%、羽毛球轨迹 93.92%；全长运行产物 17.74 MiB。
- 27 项 Python 单元测试通过。
- 全长 QA 视频完成：1841 帧、640×360、78条候选提示、20.79 MiB；抽帧目视检查通过。
- 720p Gold参考视频完成：1841 帧、36.06 MiB、0条可见机器候选；抽帧目视检查通过。
- 本地 Gold 标注页构建通过，并确认 `/gold` 返回可用页面；未发布公开版本。
- 确认本地服务会随桌面对话运行会话结束而被回收；应用内浏览器已在活动服务期间完整验证 `/gold`，持续使用需改为稳定部署。
- 30 项 Python 单元测试通过。
- 初始 v0.2 在主稳健窗口下：51/58 命中，Precision 65.38%、Recall 87.93%、F1 75.00%。
- v0.3 在同一窗口下：52/58 命中，Precision 76.47%、Recall 89.66%、F1 82.54%；该结果明确标记为开发集成绩，等待 holdout 验证。
- 第二段 holdout 的1993帧冻结 v0.3 机器运行通过；数据质量状态为 `ok`，Gold 评估尚未执行。
- 第二段 holdout Gold 锁定为64条；唯一一次冻结评估在主稳健窗口下获得Precision 74.68%、Recall 92.19%、F1 82.52%，未在holdout上调参。
- holdout 击球者准确率为64.41%，5个漏检全部位于远端upper；该问题进入后续开发集调查，不修改已锁定结果。
- 31项 Python 单元测试通过。
- v0.4开发集验证：68个候选帧与v0.3完全一致，4条击球者归属改变，主稳健击球者准确率84.62%→86.54%；未重跑旧holdout。
- 第三段验证视频1745帧冻结v0.4机器运行通过，质量状态为`ok`；Gold评估尚未执行。
- 第三段validation Gold经原视频QA纠正一处击球侧后锁定为41条；唯一一次v0.4评估获得Precision 62.71%、Recall 90.24%、F1 74.00%、击球者准确率83.78%，未调参。
- 误差诊断显示22个误报中11个发生在第一条Gold之前或最后一条Gold之后；下一开发目标确定为比赛进行状态门控，oracle Gold裁剪不进入生产规则。
- 33项 Python 单元测试通过；开发集重导出仍为68条候选，候选帧、击球者与稳健F1均未改变，自动过滤数为0。
- 锁定holdout事件、validation事件和validation Gold的SHA-256保持不变，未用于比赛状态阈值调参。
- 38项 Python 单元测试通过；top-k归一化、连续性选择非首候选、大残差软抑制和手腕证据放宽门控均有测试覆盖。
- 120帧真实GPU冒烟与1841帧全长开发运行通过；全长153帧含多候选、20帧选择非首候选、45帧软抑制，`interpolation_applied=false`。
- 开发集稳健击球F1由82.54%升至83.20%，严格F1由63.49%升至65.60%，击球者准确率保持86.54%；版本尚未在新盲标视频验收。
- 47项Python单元测试通过；门控启用/关闭、孤立候选、同侧候选、交替击球链和ShuttleSet-like字段均有覆盖。
- 开发集门控67→66条，稳健F1 83.20%→83.87%，召回保持89.66%；旧validation回顾性审计58→55条，稳健F1 72.73%→75.00%，召回保持87.80%；旧holdout 76→76条。
- 原锁定holdout事件、validation事件和validation Gold的SHA-256保持不变；新增结果写入独立`*-play-gate-dev`目录。
- `play-state-prospective-001`六段共6308帧GPU推理通过：226条provisional、207条kept、19条suppressed，运行产物约65.36 MiB；批次汇总数已披露，但候选帧和逐视频预测对Gold标注端保持隐藏。
- 51项Python单元测试及Web生产构建通过。
