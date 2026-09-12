# SwingNet 击球检测隔离实验

该目录包含官方预训练羽毛球 SwingNet 的适配、规则融合、冻结选参和回归评估。
生产入口不依赖此目录。没有前端或数据标注界面改动。

结果见 [REPORT.md](REPORT.md)，预先约定见 [PROTOCOL.md](PROTOCOL.md)。
这套实验不训练 BST，也不衡量球种分类指标。

## 环境

在仓库根目录使用已有 `.venv-model` Python：PyTorch、NumPy、OpenCV。
本次实际环境和严格加载信息记录在 `results/model-load.json`。
推理支持 CUDA/CPU，当前预处理明确仅支持 1280×720 视频。

## 文件

- `swingnet.py`：历史运动图变换、网络、作者事件解码与独立融合。
- `run_experiment.py`：开发集 18 组配置、验证选择、冻结参数后回归评估。
- `infer.py`：不读取 Gold 的单视频推理入口；可选融合已有特征与规则事件。
- `fetch_assets.py`：固定来源获取脚本，HTTP Range 只提取约 63 MB checkpoint，不下载完整 14.5 GB ZIP。
- `upstream/`：固定版本的作者代码、网络及来源元信息。
- `test_contract.py`：权重/历史预处理/输入几何/解码/融合契约检查。
- `smoke_check.py`：开发视频前 64 帧的独立 CLI 端到端检查。
- `finish_report.py`：仅汇总固定结果，导出 BST 窗口；不选参、不重复神经网络推理。

## 复现评估

在 `<PROJECT_ROOT>` 运行，选择尚不存在的结果目录：

```powershell
.\.venv-model\Scripts\python.exe -B experiments/swingnet/run_experiment.py --output experiments/swingnet/results-repeat
```

实验复用 `run_experiment.specs()` 明确列出的已有注册视频、特征、基线事件和 Gold。
原素材路径必须仍可读取；不会自动替换成其他视频。
所有输出都限制在本实验目录中，并拒绝覆盖已有结果目录。

## 不使用 Gold 的单视频推理

```powershell
.\.venv-model\Scripts\python.exe -B experiments/swingnet/infer.py --video example.mp4 --video-id vid_2e2258f261aa --output experiments/swingnet/example-inference
```

可同时加 `--frames` 和 `--baseline`，分别指向已有完整视频的 `frames.enriched.jsonl` 和 `events.jsonl`，启用固定融合参数。
默认从 `results/frozen-selection.json` 读取配置；可用 `--selection` 指定另一次实验的冻结文件。
`selected-events.jsonl` 按验证门槛选择融合或旧基线；这只是实验输出，不会改写生产文件。

事件帧索引从 0 开始。BST 窗口的结束帧为 exclusive，起止已裁到视频范围内。
窗口清单尚未送入 BST；球种分类是否受益需要另行验证。

## 验证

```powershell
.\.venv-model\Scripts\python.exe -B -m unittest discover -s experiments/swingnet -p test_contract.py -v
```

首次 CLI 冒烟检查可以运行 `smoke_check.py`；它拒绝覆盖既有 `smoke-input.mp4` 和 `smoke-result/`。
已生成的 `smoke-result/summary.json` 记录帧数、权重/输入哈希及 `uses_gold: false`。

## 解释边界

六段回归素材之前被用来评估过其他方案，不是新的盲测。
官方权重严格加载不等于完整复现作者训练环境；作者公开代码缺失的部分及尺寸差异详见报告。
作者解码按每段概率分位数筛选，分数未校准，不能当成真实正确率。
未经更多独立视频验证，不应仅凭一次验证集门槛就自动替换生产检测器。
