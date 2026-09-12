from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "data" / "batch_runs" / "broadcast-uncensored-v1" / "videos"
MAPPING_PATH = ROOT / "research" / "identity_mapping.csv"
OUT = ROOT / "research_outputs" / "shi_tactical_profile_v1"
OUT.mkdir(parents=True, exist_ok=True)

COURT_WIDTH = 6.1
COURT_LENGTH = 13.4
NET_Y = COURT_LENGTH / 2

MACRO_ORDER = ["发球", "网前/前场", "后场进攻", "高远/挑球", "平抽/推挡", "吊球"]
DEPTH_ORDER = ["前场", "中场", "后场"]
WIDTH_ORDER = ["左侧", "中路", "右侧"]
PHASE_ORDER = ["开局阶段", "组织阶段", "延长阶段"]

RAW_TO_MACRO = {
    "short service": "发球",
    "long service": "发球",
    "net shot": "网前/前场",
    "return net": "网前/前场",
    "rush": "网前/前场",
    "cross-court net shot": "网前/前场",
    "smash": "后场进攻",
    "wrist smash": "后场进攻",
    "lob": "高远/挑球",
    "defensive return lob": "高远/挑球",
    "clear": "高远/挑球",
    "drive": "平抽/推挡",
    "defensive return drive": "平抽/推挡",
    "back-court drive": "平抽/推挡",
    "push": "平抽/推挡",
    "drop": "吊球",
    "passive drop": "吊球",
}


def add_chinese_font() -> None:
    candidates = [Path(r"C:\Windows\Fonts\msyh.ttc"), ROOT / "vendor" / "Good-Badminton" / "simhei.ttf"]
    for path in candidates:
        if path.exists():
            try:
                font_manager.fontManager.addfont(str(path))
                prop = font_manager.FontProperties(fname=str(path))
                plt.rcParams["font.family"] = prop.get_name()
                break
            except Exception:
                continue
    plt.rcParams["axes.unicode_minus"] = False


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: str | None) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def valid_court_xy(x: float | None, y: float | None) -> bool:
    return x is not None and y is not None and 0 <= x <= COURT_WIDTH and 0 <= y <= COURT_LENGTH


def depth_zone(y: float | None, side: str) -> str | None:
    if y is None or not 0 <= y <= COURT_LENGTH:
        return None
    depth_from_baseline = y / NET_Y if side == "upper" else (COURT_LENGTH - y) / NET_Y
    if not 0 <= depth_from_baseline <= 1:
        return None
    if depth_from_baseline >= 2 / 3:
        return "前场"
    if depth_from_baseline >= 1 / 3:
        return "中场"
    return "后场"


def width_zone(x: float | None) -> str | None:
    if x is None or not 0 <= x <= COURT_WIDTH:
        return None
    if x < COURT_WIDTH / 3:
        return "左侧"
    if x < COURT_WIDTH * 2 / 3:
        return "中路"
    return "右侧"


def zone9(depth: str | None, width: str | None) -> str | None:
    return f"{depth}-{width}" if depth and width else None


def phase(stroke_index: int | None) -> str | None:
    if stroke_index is None:
        return None
    if stroke_index <= 3:
        return "开局阶段"
    if stroke_index <= 6:
        return "组织阶段"
    return "延长阶段"


def load_mapping() -> dict[str, dict[str, str]]:
    rows = read_csv(MAPPING_PATH)
    return {row["video_id"]: row for row in rows}


def normalize_events(mapping: dict[str, dict[str, str]]) -> list[dict]:
    output: list[dict] = []
    for video_id, meta in mapping.items():
        path = BATCH / video_id / "dataset" / "dataset_rows.csv"
        if not path.exists():
            continue
        for raw in read_csv(path):
            if raw.get("player") != meta["shi_side"]:
                continue
            own_side = meta["shi_side"]
            opponent_side = "lower" if own_side == "upper" else "upper"
            own_x = as_float(raw.get("player_location_x"))
            own_y = as_float(raw.get("player_location_y"))
            opponent_x = as_float(raw.get("opponent_location_x"))
            opponent_y = as_float(raw.get("opponent_location_y"))
            landing_x = as_float(raw.get("landing_x"))
            landing_y = as_float(raw.get("landing_y"))
            stroke_type = (raw.get("stroke_type") or "unknown").strip().lower()
            macro = RAW_TO_MACRO.get(stroke_type)
            stroke_index = as_int(raw.get("stroke_index"))
            output.append(
                {
                    "video_id": video_id,
                    "opponent": meta["opponent"],
                    "competition": meta["competition"],
                    "shi_side": own_side,
                    "event_id": raw.get("event_id", ""),
                    "rally_id": raw.get("rally_id", ""),
                    "stroke_index": stroke_index,
                    "hit_frame": as_int(raw.get("hit_frame")),
                    "hit_time": as_float(raw.get("hit_time")),
                    "stroke_type": stroke_type,
                    "macro_type": macro or "未分类",
                    "own_x": own_x,
                    "own_y": own_y,
                    "opponent_x": opponent_x,
                    "opponent_y": opponent_y,
                    "landing_x": landing_x,
                    "landing_y": landing_y,
                    "own_depth": depth_zone(own_y, own_side),
                    "opponent_depth": depth_zone(opponent_y, opponent_side),
                    "own_width": width_zone(own_x),
                    "opponent_width": width_zone(opponent_x),
                    "own_zone9": zone9(depth_zone(own_y, own_side), width_zone(own_x)),
                    "opponent_zone9": zone9(depth_zone(opponent_y, opponent_side), width_zone(opponent_x)),
                    "landing_zone9": zone9(depth_zone(landing_y, opponent_side), width_zone(landing_x)),
                    "phase": phase(stroke_index),
                }
            )
    return output


def profile(rows: Iterable[dict], key: str, categories: list[str] | None = None, exclude_unclassified: bool = False) -> list[dict]:
    values = [row.get(key) for row in rows]
    values = [value for value in values if value]
    if exclude_unclassified:
        values = [value for value in values if value != "未分类" and value != "unknown"]
    counts = Counter(values)
    ordered = categories or sorted(counts)
    total = sum(counts.values())
    return [
        {"category": category, "count": counts.get(category, 0), "proportion": counts.get(category, 0) / total if total else 0}
        for category in ordered
    ]


def vector(rows: Iterable[dict], key: str, categories: list[str]) -> np.ndarray:
    values = [row.get(key) for row in rows]
    counts = np.array([sum(value == category for value in values) for category in categories], dtype=float)
    total = counts.sum()
    return counts / total if total else np.zeros(len(categories))


def entropy(probabilities: np.ndarray) -> float:
    p = probabilities[probabilities > 0]
    if len(p) <= 1:
        return 0.0
    raw = float(-(p * np.log(p)).sum())
    return raw / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() == 0 or q.sum() == 0:
        return float("nan")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float((a[mask] * np.log2(a[mask] / b[mask])).sum())

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def contingency(rows: list[dict], context_key: str, context_order: list[str], category_key: str = "macro_type") -> tuple[np.ndarray, float]:
    table = np.zeros((len(context_order), len(MACRO_ORDER)), dtype=float)
    context_index = {value: index for index, value in enumerate(context_order)}
    category_index = {value: index for index, value in enumerate(MACRO_ORDER)}
    for row in rows:
        context = row.get(context_key)
        category = row.get(category_key)
        if context not in context_index or category not in category_index:
            continue
        table[context_index[context], category_index[category]] += 1
    n = table.sum()
    if n == 0:
        return table, 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    mask = expected > 0
    chi2 = float(((table - expected) ** 2 / np.where(mask, expected, 1))[mask].sum())
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    v = math.sqrt((chi2 / n) / denominator) if denominator > 0 else 0.0
    return table, v


def write_context_profile(rows: list[dict], context_key: str, context_order: list[str], output_name: str) -> dict[str, float]:
    table, v = contingency(rows, context_key, context_order)
    result: list[dict] = []
    for i, context in enumerate(context_order):
        total = table[i].sum()
        for j, category in enumerate(MACRO_ORDER):
            result.append(
                {
                    "context": context,
                    "macro_type": category,
                    "count": int(table[i, j]),
                    "proportion": float(table[i, j] / total) if total else 0.0,
                }
            )
    write_csv(OUT / output_name, result)
    return {"cramers_v": v, "n": int(table.sum())}


def plot_bar(profile_rows: list[dict], title: str, output: Path, color: str = "#1f77b4") -> None:
    labels = [row["category"] for row in profile_rows]
    values = [row["proportion"] * 100 for row in profile_rows]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(labels, values, color=color)
    ax.set_ylabel("比例（%）")
    ax.set_title(title)
    ax.set_ylim(0, max(values + [1]) * 1.25)
    ax.tick_params(axis="x", rotation=25)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values + [1]) * 0.02, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def heatmap(rows: list[dict], key: str, title: str, output: Path) -> None:
    matrix = np.zeros((len(DEPTH_ORDER), len(WIDTH_ORDER)), dtype=float)
    for i, depth in enumerate(DEPTH_ORDER):
        for j, width in enumerate(WIDTH_ORDER):
            matrix[i, j] = sum(row.get(key) == f"{depth}-{width}" for row in rows)
    total = matrix.sum()
    display = matrix / total * 100 if total else matrix
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    image = ax.imshow(display, cmap="YlOrRd", vmin=0, vmax=max(1, float(display.max())))
    ax.set_xticks(range(len(WIDTH_ORDER)), WIDTH_ORDER)
    ax.set_yticks(range(len(DEPTH_ORDER)), DEPTH_ORDER)
    ax.set_xlabel("横向区域")
    ax.set_ylabel("纵深区域")
    ax.set_title(title)
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            ax.text(j, i, f"{display[i, j]:.1f}%\n(n={int(matrix[i, j])})", ha="center", va="center", color="black")
    fig.colorbar(image, ax=ax, label="比例（%）")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_context(rows: list[dict], context_key: str, context_order: list[str], title: str, output: Path) -> None:
    table, _ = contingency(rows, context_key, context_order)
    row_sums = table.sum(axis=1, keepdims=True)
    display = np.divide(table, row_sums, out=np.zeros_like(table), where=row_sums > 0) * 100
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    image = ax.imshow(display, cmap="Blues", vmin=0, vmax=max(1, float(display.max())))
    ax.set_xticks(range(len(MACRO_ORDER)), MACRO_ORDER, rotation=25, ha="right")
    ax.set_yticks(range(len(context_order)), context_order)
    ax.set_title(title)
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            ax.text(j, i, f"{display[i, j]:.1f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=ax, label="行内比例（%）")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_stability(match_stats: list[dict], output: Path) -> None:
    ordered = sorted(match_stats, key=lambda row: row["jsd_macro_to_overall"], reverse=True)
    labels = [row["video_id"][-6:] for row in ordered]
    jsd_values = [row["jsd_macro_to_overall"] for row in ordered]
    entropy_values = [row["entropy_macro"] for row in ordered]
    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(labels))
    ax1.bar(x, jsd_values, color="#d95f02", alpha=0.82, label="JSD（与总体画像差异）")
    ax1.set_ylabel("JSD")
    ax1.set_xticks(x, labels, rotation=65, ha="right")
    ax1.set_xlabel("比赛视频（video_id后6位）")
    ax2 = ax1.twinx()
    ax2.plot(x, entropy_values, color="#1b9e77", marker="o", linewidth=1.8, label="球种熵")
    ax2.set_ylabel("归一化球种熵")
    ax1.set_title("比赛间战术稳定性：画像差异与球种多样性")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_opponents(opponent_stats: list[dict], output: Path) -> None:
    stats = sorted(opponent_stats, key=lambda row: row["classified_strokes"], reverse=True)[:12]
    if not stats:
        return
    labels = [row["opponent"] for row in stats]
    fig, ax = plt.subplots(figsize=(11, 6))
    bottom = np.zeros(len(stats))
    colors = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
    for category, color in zip(MACRO_ORDER, colors):
        values = np.array([row.get(category, 0) * 100 for row in stats])
        ax.bar(labels, values, bottom=bottom, label=category, color=color)
        bottom += values
    ax.set_ylabel("已分类击球比例（%）")
    ax.set_title("不同对手下的石宇奇球种结构（按已分类击球）")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def make_transition(rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["video_id"], row["rally_id"])].append(row)
    counts = np.zeros((len(MACRO_ORDER), len(MACRO_ORDER)), dtype=int)
    index = {category: i for i, category in enumerate(MACRO_ORDER)}
    for rally_rows in grouped.values():
        rally_rows.sort(key=lambda row: row["stroke_index"] if row["stroke_index"] is not None else 10**9)
        previous = None
        for row in rally_rows:
            current = row["macro_type"]
            if current not in index:
                continue
            if previous in index:
                counts[index[previous], index[current]] += 1
            previous = current
    result: list[dict] = []
    for i, source in enumerate(MACRO_ORDER):
        total = counts[i].sum()
        for j, target in enumerate(MACRO_ORDER):
            result.append({"from_macro": source, "to_macro": target, "count": int(counts[i, j]), "row_proportion": float(counts[i, j] / total) if total else 0.0})
    return result


def build_report(summary: dict, top_raw: list[dict], top_macro: list[dict], top_opponent: list[dict]) -> str:
    overall_depth = summary["overall_depth"]
    phase_top = summary["phase_top"]
    lines = [
        "# 石宇奇男子单打技术—战术画像：第一版实证结果",
        "",
        "> 研究版本：v1.0；分析单位为石宇奇的一次逐拍击球事件。当前版本用于建立可复现的描述性基线，不解释得分效果或因果关系。",
        "",
        "## 1. 数据范围",
        "",
        f"- 纳入比赛：{summary['video_count']}场；对手：{summary['opponent_count']}名。",
        f"- 回合：{summary['rally_count']}个。",
        f"- 石宇奇逐拍事件：{summary['stroke_events']}次；其中已归入具体球种的事件：{summary['classified_strokes']}次；未分类事件：{summary['unclassified_strokes']}次。",
        f"- 有效击球位置：{summary['valid_own_position']}次；有效对手位置：{summary['valid_opponent_position']}次；有效预测落点：{summary['valid_landing']}次。",
        "- 上下场身份已按比赛画面完成第一版映射，映射表位于 `research/identity_mapping.csv`，后续可直接修改后重跑。",
        "",
        "## 2. 分析方法",
        "",
        "1. 将每场比赛的 upper/lower 转换为石宇奇/对手，并将纵深统一为击球者视角的前场、中场、后场。",
        "2. 将原始球种归并为发球、网前/前场、后场进攻、高远/挑球、平抽/推挡和吊球六类；unknown 不作为球种类别。",
        "3. 根据每个回合的 stroke_index 划分开局阶段（1—3拍）、组织阶段（4—6拍）和延长阶段（7拍及以后）。",
        "4. 使用球种比例、九宫格空间分布、条件比例、Cramér’s V、归一化球种熵和 Jensen–Shannon divergence 描述战术结构与稳定性。",
        "",
        "## 3. 第一版主要结果",
        "",
        "### 3.1 球种结构",
        "",
    ]
    for row in top_macro:
        lines.append(f"- {row['category']}：{row['proportion'] * 100:.1f}%（n={row['count']}）。")
    lines += ["", "原始球种中出现频率最高的类别："]
    for row in top_raw:
        lines.append(f"- {row['category']}：{row['proportion'] * 100:.1f}%（n={row['count']}）。")
    lines += ["", "### 3.2 击球区域与对手位置", ""]
    for row in overall_depth:
        lines.append(f"- 石宇奇击球位于{row['category']}：{row['proportion'] * 100:.1f}%（n={row['count']}）。")
    lines.append("")
    for row in summary["overall_opponent_depth"]:
        lines.append(f"- 对手位于{row['category']}：{row['proportion'] * 100:.1f}%（n={row['count']}）。")
    lines.append("")
    lines.append(
        f"- 球种与石宇奇击球纵深的 Cramér’s V={summary['context_association']['own_depth']['cramers_v']:.3f}；"
        f"与对手纵深的 Cramér’s V={summary['context_association']['opponent_depth']['cramers_v']:.3f}；"
        f"与回合阶段的 Cramér’s V={summary['context_association']['phase']['cramers_v']:.3f}。"
    )
    lines += ["", "### 3.3 回合阶段", ""]
    for row in phase_top:
        lines.append(f"- {row['category']}：{row['proportion'] * 100:.1f}%（n={row['count']}）。")
    lines += ["", "### 3.4 战术稳定性", ""]
    lines.append(f"- 总体六类球种归一化熵：{summary['overall_macro_entropy']:.3f}。")
    lines.append(f"- 比赛层面的 JSD（与总体球种画像的差异）中位数：{summary['median_match_jsd']:.3f}。")
    lines.append(f"- 比赛层面的球种熵中位数：{summary['median_match_entropy']:.3f}。")
    eligible_opponents = [row for row in top_opponent if row["classified_strokes"] >= 200]
    for row in sorted(eligible_opponents, key=lambda item: item["jsd_macro_to_overall"], reverse=True)[:3]:
        lines.append(f"- 对手层面画像差异较大的样本：{row['opponent']}（{row['matches']}场，n={row['classified_strokes']}，JSD={row['jsd_macro_to_overall']:.3f}）。")
    lines += ["", "## 4. 结果解释边界", "", "- 本版可以说明石宇奇在不同空间和回合情境下的击球分布与球种选择差异。", "- 本版不能说明某种球是否导致得分、哪种落点最有效，也不能作出胜负因果结论。", "- 落点变量是当前算法提供的下一次接触点/终端代理，应在论文中称为预测落点或落点代理。", "- 对手比较中，重复交手较少的对手应作为探索性结果，不宜过度泛化。", "", "## 5. 输出文件", "", "- `stroke_events.csv`：统一到石宇奇视角后的逐拍分析数据。", "- `overall_macro_type.csv`、`overall_stroke_type.csv`：总体球种结构。", "- `context_macro_by_own_depth.csv`、`context_macro_by_opponent_depth.csv`、`context_macro_by_phase.csv`：情境化球种分布。", "- `match_profiles.csv`、`opponent_profiles.csv`：比赛和对手层面的画像及稳定性指标。", "- `figure_*.png`：第一版图表。", "", "## 6. 参考思路", "", "- ShuttleSet：逐拍球种、击球位置、落点和双方位置的联合表示，以及场地网格化分析。https://arxiv.org/abs/2306.04948", "- Badminton Strike Positions：使用击球位置空间熵描述击球分布的集中程度。https://www.mdpi.com/1099-4300/23/7/799", "- Singles Badminton Technical-Tactical Instrument：以单次击球为单位组织情境化技术—战术变量。https://pmc.ncbi.nlm.nih.gov/articles/PMC7758221/", "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    add_chinese_font()
    mapping = load_mapping()
    rows = normalize_events(mapping)
    if not rows:
        raise SystemExit("No Shi Yuqi stroke events were found")
    rows.sort(key=lambda row: (row["video_id"], row["hit_frame"] if row["hit_frame"] is not None else 10**12))
    write_csv(OUT / "stroke_events.csv", rows)

    classified = [row for row in rows if row["macro_type"] != "未分类"]
    top_raw = sorted(profile(rows, "stroke_type", exclude_unclassified=True), key=lambda row: row["count"], reverse=True)
    top_macro = sorted(profile(classified, "macro_type", categories=MACRO_ORDER), key=lambda row: row["count"], reverse=True)
    write_csv(OUT / "overall_stroke_type.csv", profile(rows, "stroke_type", exclude_unclassified=True))
    write_csv(OUT / "overall_macro_type.csv", profile(classified, "macro_type", categories=MACRO_ORDER))
    write_csv(OUT / "overall_hit_depth.csv", profile([row for row in rows if row["own_depth"]], "own_depth", categories=DEPTH_ORDER))
    write_csv(OUT / "overall_opponent_depth.csv", profile([row for row in rows if row["opponent_depth"]], "opponent_depth", categories=DEPTH_ORDER))
    write_csv(OUT / "overall_landing_zone.csv", profile([row for row in rows if row["landing_zone9"]], "landing_zone9"))

    own_depth_assoc = write_context_profile(classified, "own_depth", DEPTH_ORDER, "context_macro_by_own_depth.csv")
    opponent_depth_assoc = write_context_profile(classified, "opponent_depth", DEPTH_ORDER, "context_macro_by_opponent_depth.csv")
    phase_assoc = write_context_profile(classified, "phase", PHASE_ORDER, "context_macro_by_phase.csv")
    write_csv(OUT / "context_association.csv", [
        {"context": "own_depth", **own_depth_assoc},
        {"context": "opponent_depth", **opponent_depth_assoc},
        {"context": "phase", **phase_assoc},
    ])

    match_groups: dict[str, list[dict]] = defaultdict(list)
    opponent_groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        match_groups[row["video_id"]].append(row)
        opponent_groups[row["opponent"]].append(row)
    overall_macro_vector = vector(classified, "macro_type", MACRO_ORDER)

    match_stats: list[dict] = []
    for video_id, group in sorted(match_groups.items()):
        group_classified = [row for row in group if row["macro_type"] != "未分类"]
        match_vector = vector(group_classified, "macro_type", MACRO_ORDER)
        record = {
            "video_id": video_id,
            "opponent": group[0]["opponent"],
            "competition": group[0]["competition"],
            "shi_side": group[0]["shi_side"],
            "stroke_events": len(group),
            "classified_strokes": len(group_classified),
            "unclassified_strokes": len(group) - len(group_classified),
            "valid_own_position": sum(row["own_depth"] is not None for row in group),
            "valid_opponent_position": sum(row["opponent_depth"] is not None for row in group),
            "valid_landing": sum(row["landing_zone9"] is not None for row in group),
            "entropy_macro": entropy(match_vector),
            "jsd_macro_to_overall": jsd(match_vector, overall_macro_vector),
        }
        record.update({category: float(match_vector[index]) for index, category in enumerate(MACRO_ORDER)})
        match_stats.append(record)
    write_csv(OUT / "match_profiles.csv", match_stats)

    opponent_stats: list[dict] = []
    for opponent, group in sorted(opponent_groups.items()):
        group_classified = [row for row in group if row["macro_type"] != "未分类"]
        group_vector = vector(group_classified, "macro_type", MACRO_ORDER)
        record = {
            "opponent": opponent,
            "matches": len({row["video_id"] for row in group}),
            "stroke_events": len(group),
            "classified_strokes": len(group_classified),
            "entropy_macro": entropy(group_vector),
            "jsd_macro_to_overall": jsd(group_vector, overall_macro_vector),
        }
        record.update({category: float(group_vector[index]) for index, category in enumerate(MACRO_ORDER)})
        opponent_stats.append(record)
    write_csv(OUT / "opponent_profiles.csv", opponent_stats)

    transition_rows = make_transition(rows)
    write_csv(OUT / "macro_transition_matrix.csv", transition_rows)

    plot_bar(top_macro, "石宇奇总体球种结构（已分类击球）", OUT / "figure_overall_macro_type.png", color="#2c7fb8")
    plot_bar(top_raw[:12], "石宇奇原始球种结构（已分类击球）", OUT / "figure_overall_raw_type.png", color="#41ab5d")
    heatmap([row for row in rows if row["own_zone9"]], "own_zone9", "石宇奇击球位置分布（九宫格）", OUT / "figure_hit_zone_heatmap.png")
    heatmap([row for row in rows if row["opponent_zone9"]], "opponent_zone9", "对手位置分布（九宫格）", OUT / "figure_opponent_zone_heatmap.png")
    heatmap([row for row in rows if row["landing_zone9"]], "landing_zone9", "石宇奇预测落点分布（九宫格）", OUT / "figure_landing_zone_heatmap.png")
    plot_context(classified, "own_depth", DEPTH_ORDER, "不同击球纵深下的球种结构", OUT / "figure_macro_by_own_depth.png")
    plot_context(classified, "opponent_depth", DEPTH_ORDER, "不同对手纵深下的球种结构", OUT / "figure_macro_by_opponent_depth.png")
    plot_context(classified, "phase", PHASE_ORDER, "不同回合阶段的球种结构", OUT / "figure_macro_by_phase.png")
    plot_stability(match_stats, OUT / "figure_match_stability.png")
    plot_opponents(opponent_stats, OUT / "figure_opponent_profiles.png")

    overall_depth = profile([row for row in rows if row["own_depth"]], "own_depth", categories=DEPTH_ORDER)
    overall_opponent_depth = profile([row for row in rows if row["opponent_depth"]], "opponent_depth", categories=DEPTH_ORDER)
    phase_profile = profile([row for row in rows if row["phase"]], "phase", categories=PHASE_ORDER)
    summary = {
        "version": "v1.0",
        "video_count": len(match_groups),
        "opponent_count": len(opponent_groups),
        "rally_count": len({(row["video_id"], row["rally_id"]) for row in rows if row["rally_id"]}),
        "stroke_events": len(rows),
        "classified_strokes": len(classified),
        "unclassified_strokes": len(rows) - len(classified),
        "valid_own_position": sum(row["own_depth"] is not None for row in rows),
        "valid_opponent_position": sum(row["opponent_depth"] is not None for row in rows),
        "valid_landing": sum(row["landing_zone9"] is not None for row in rows),
        "overall_macro_entropy": entropy(overall_macro_vector),
        "median_match_jsd": float(np.median([row["jsd_macro_to_overall"] for row in match_stats])),
        "median_match_entropy": float(np.median([row["entropy_macro"] for row in match_stats])),
        "overall_depth": overall_depth,
        "overall_opponent_depth": overall_opponent_depth,
        "phase_top": phase_profile,
        "context_association": {"own_depth": own_depth_assoc, "opponent_depth": opponent_depth_assoc, "phase": phase_assoc},
        "top_macro": top_macro,
        "top_raw": top_raw[:12],
        "files": sorted(path.name for path in OUT.iterdir() if path.is_file()),
    }
    (OUT / "analysis_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    report = build_report(summary, top_raw[:12], top_macro, opponent_stats)
    (OUT / "RESEARCH_REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ["video_count", "opponent_count", "stroke_events", "classified_strokes", "unclassified_strokes", "valid_own_position", "valid_opponent_position", "valid_landing", "overall_macro_entropy", "median_match_jsd", "median_match_entropy"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
