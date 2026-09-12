from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shi_tactical_profile_v1 as base


OUT = base.OUT
MACRO_ORDER = base.MACRO_ORDER


def cramers_v(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=float)
    n = table.sum()
    if n == 0:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    mask = expected > 0
    chi2 = float((((table - expected) ** 2) / np.where(mask, expected, 1))[mask].sum())
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt((chi2 / n) / denominator)) if denominator > 0 else 0.0


def rally_tables(rows: list[dict], context_key: str, context_order: list[str]) -> np.ndarray:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["macro_type"] not in MACRO_ORDER:
            continue
        grouped[(row["video_id"], row["rally_id"])].append(row)
    context_index = {value: index for index, value in enumerate(context_order)}
    macro_index = {value: index for index, value in enumerate(MACRO_ORDER)}
    tables: list[np.ndarray] = []
    for rally_rows in grouped.values():
        table = np.zeros((len(context_order), len(MACRO_ORDER)), dtype=np.int32)
        for row in rally_rows:
            context = row.get(context_key)
            macro = row.get("macro_type")
            if context in context_index and macro in macro_index:
                table[context_index[context], macro_index[macro]] += 1
        if table.sum() > 0:
            tables.append(table)
    if not tables:
        return np.zeros((0, len(context_order), len(MACRO_ORDER)), dtype=np.int32)
    return np.stack(tables)


def row_proportions(table: np.ndarray) -> np.ndarray:
    totals = table.sum(axis=1, keepdims=True)
    return np.divide(table, totals, out=np.zeros_like(table, dtype=float), where=totals > 0)


def bootstrap_context(
    rows: list[dict],
    context_key: str,
    context_order: list[str],
    reps: int = 1000,
    seed: int = 20260817,
) -> dict:
    tables = rally_tables(rows, context_key, context_order)
    if len(tables) == 0:
        raise ValueError(f"No classified rally tables for {context_key}")
    observed = tables.sum(axis=0)
    observed_probs = row_proportions(observed)
    rng = np.random.default_rng(seed)
    probability_draws = np.zeros((reps, len(context_order), len(MACRO_ORDER)), dtype=float)
    v_draws = np.zeros(reps, dtype=float)
    for index in range(reps):
        sample = rng.integers(0, len(tables), size=len(tables))
        table = tables[sample].sum(axis=0)
        probability_draws[index] = row_proportions(table)
        v_draws[index] = cramers_v(table)
    ci_rows: list[dict] = []
    for i, context in enumerate(context_order):
        for j, macro in enumerate(MACRO_ORDER):
            ci_rows.append(
                {
                    "context_variable": context_key,
                    "context": context,
                    "macro_type": macro,
                    "observed_proportion": float(observed_probs[i, j]),
                    "ci95_low": float(np.percentile(probability_draws[:, i, j], 2.5)),
                    "ci95_high": float(np.percentile(probability_draws[:, i, j], 97.5)),
                    "stroke_count": int(observed[i, j]),
                    "rally_count": int(len(tables)),
                }
            )
    association = {
        "context_variable": context_key,
        "observed_cramers_v": cramers_v(observed),
        "ci95_low": float(np.percentile(v_draws, 2.5)),
        "ci95_high": float(np.percentile(v_draws, 97.5)),
        "stroke_count": int(observed.sum()),
        "rally_count": int(len(tables)),
    }
    return {
        "context_key": context_key,
        "context_order": context_order,
        "observed": observed,
        "observed_probs": observed_probs,
        "probability_draws": probability_draws,
        "v_draws": v_draws,
        "ci_rows": ci_rows,
        "association": association,
    }


def build_contrasts(results: dict[str, dict], specs: dict[str, list[tuple[str, str]]]) -> list[dict]:
    output: list[dict] = []
    for context_key, pairs in specs.items():
        result = results[context_key]
        order = result["context_order"]
        index = {value: i for i, value in enumerate(order)}
        observed = result["observed_probs"]
        draws = result["probability_draws"]
        for first, second in pairs:
            if first not in index or second not in index:
                continue
            i, j = index[first], index[second]
            for macro_index, macro in enumerate(MACRO_ORDER):
                difference_draws = (draws[:, i, macro_index] - draws[:, j, macro_index]) * 100
                output.append(
                    {
                        "context_variable": context_key,
                        "reference_context": first,
                        "comparison_context": second,
                        "macro_type": macro,
                        "observed_difference_percentage_points": float((observed[i, macro_index] - observed[j, macro_index]) * 100),
                        "ci95_low_percentage_points": float(np.percentile(difference_draws, 2.5)),
                        "ci95_high_percentage_points": float(np.percentile(difference_draws, 97.5)),
                    }
                )
    return output


def add_font() -> None:
    base.add_chinese_font()


def plot_contrasts(contrast_rows: list[dict], output: Path) -> None:
    groups: list[tuple[str, str, str]] = []
    for row in contrast_rows:
        key = (row["context_variable"], row["reference_context"], row["comparison_context"])
        if key not in groups:
            groups.append(key)
    if not groups:
        return
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 3.2 * len(groups)), squeeze=False)
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
    for axis, group in zip(axes[:, 0], groups):
        context_key, first, second = group
        rows = [row for row in contrast_rows if (row["context_variable"], row["reference_context"], row["comparison_context"]) == group]
        x = np.arange(len(rows))
        values = np.array([row["observed_difference_percentage_points"] for row in rows])
        low = values - np.array([row["ci95_low_percentage_points"] for row in rows])
        high = np.array([row["ci95_high_percentage_points"] for row in rows]) - values
        axis.errorbar(x, values, yerr=[low, high], fmt="o", capsize=4, color="#333333")
        axis.axhline(0, color="#999999", linewidth=1)
        axis.set_xticks(x, [row["macro_type"] for row in rows], rotation=20)
        axis.set_ylabel("百分点差异")
        axis.set_title(f"{context_key}: {first} − {second}")
        for tick, color in zip(axis.get_xticklabels(), colors):
            tick.set_color(color)
    fig.suptitle("情境化球种比例差异（按回合聚类Bootstrap 95% CI）", y=1.01)
    fig.tight_layout()
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def paper_variables() -> list[dict]:
    return [
        {"variable": "stroke_type", "role": "主要因变量", "definition": "算法输出的原始击球类别", "unit": "一次击球"},
        {"variable": "macro_type", "role": "主要因变量", "definition": "原始球种归并为六类上位类别", "unit": "一次击球"},
        {"variable": "own_depth", "role": "情境变量", "definition": "石宇奇击球时位于前场、中场或后场", "unit": "一次击球"},
        {"variable": "opponent_depth", "role": "情境变量", "definition": "对手在石宇奇击球时位于前场、中场或后场", "unit": "一次击球"},
        {"variable": "phase", "role": "情境变量", "definition": "按回合内stroke_index划分开局、组织和延长阶段", "unit": "一次击球"},
        {"variable": "own_zone9", "role": "空间变量", "definition": "石宇奇击球位置的三纵深×三横向九宫格", "unit": "一次击球"},
        {"variable": "opponent_zone9", "role": "空间变量", "definition": "对手位置的三纵深×三横向九宫格", "unit": "一次击球"},
        {"variable": "landing_zone9", "role": "空间结果变量", "definition": "算法预测落点/下一次接触点的九宫格代理", "unit": "一次击球"},
        {"variable": "entropy_macro", "role": "稳定性指标", "definition": "六类球种分布的归一化Shannon熵", "unit": "一场比赛"},
        {"variable": "jsd_macro_to_overall", "role": "稳定性指标", "definition": "单场球种分布与总体画像的Jensen–Shannon divergence", "unit": "一场比赛"},
    ]


def build_v2_report(summary: dict, results: dict[str, dict], contrasts: list[dict]) -> str:
    association_lines = []
    for key, result in results.items():
        assoc = result["association"]
        association_lines.append(
            f"- {key}：Cramér’s V={assoc['observed_cramers_v']:.3f}，95% CI [{assoc['ci95_low']:.3f}, {assoc['ci95_high']:.3f}]，基于{assoc['rally_count']}个回合。"
        )
    key_contrasts = []
    for row in contrasts:
        if row["macro_type"] in {"发球", "后场进攻", "高远/挑球", "平抽/推挡"}:
            key_contrasts.append(
                f"- {row['context_variable']}：{row['reference_context']}−{row['comparison_context']} 的{row['macro_type']}差异为 {row['observed_difference_percentage_points']:.1f} 个百分点，95% CI [{row['ci95_low_percentage_points']:.1f}, {row['ci95_high_percentage_points']:.1f}]。"
            )
    lines = [
        "# 石宇奇男子单打技术—战术画像：第二版研究报告",
        "",
        "> 研究设计：单运动员、多比赛、逐拍事件的情境化技术—战术个案研究。第二版将描述性画像升级为按回合聚类的Bootstrap区间估计。",
        "",
        "## 一、当前内容能否支撑学术论文？",
        "",
        "### 可以支撑的论文类型",
        "",
        "当前数据和结果已经足以支撑一篇以石宇奇为对象的：",
        "",
        "- 男子单打技术—战术个案研究；",
        "- 基于自动逐拍数据的运动表现分析研究；",
        "- 情境化击球选择与战术稳定性研究；",
        "- 探索性或方法应用型体育数据分析论文。",
        "",
        "### 目前还不能支撑的论文结论",
        "",
        "目前不宜直接声称：",
        "",
        "- 某种球提高了石宇奇的得分概率；",
        "- 某个落点是最有效或最优落点；",
        "- 石宇奇的战术优于其他运动员；",
        "- 某种情境导致了胜负结果。",
        "",
        "原因是当前研究没有把得分、失分原因、回合胜负和对照运动员作为核心结果变量。因此，论文应明确定位为‘单运动员情境化技术—战术画像’，而不是‘击球效果因果研究’。",
        "",
        "## 二、第二版研究问题",
        "",
        "1. 石宇奇的总体球种、击球位置和落点结构是什么？",
        "2. 石宇奇的球种选择是否随自身击球纵深、对手纵深和回合阶段发生系统变化？",
        "3. 石宇奇是否具有稳定的核心球种结构，同时在不同比赛和对手之间进行局部调整？",
        "",
        "## 三、样本与变量",
        "",
        f"- {summary['video_count']}场比赛、{summary['opponent_count']}名对手、{summary['rally_count']}个回合。",
        f"- 石宇奇击球事件{summary['stroke_events']}次，其中已分类击球{summary['classified_strokes']}次。",
        "- 分析单位：击球事件；稳健性估计的聚类单位：回合。",
        "- 主要因变量：六类上位球种。",
        "- 主要情境变量：自身击球纵深、对手纵深、回合阶段。",
        "- 空间变量：击球位置、对手位置和预测落点的九宫格。",
        "",
        "完整变量表见 `paper_variables.csv`。",
        "",
        "## 四、第二版统计结果",
        "",
        "### 4.1 情境与球种结构的关联",
        "",
    ]
    lines.extend(association_lines)
    lines += ["", "### 4.2 具有解释价值的情境差异", ""]
    lines.extend(key_contrasts[:18])
    lines += [
        "",
        "这些差异是比例差异，不是得分效果差异。例如，后场进攻比例在后场击球情境中更高，只能说明球种选择结构不同，不能说明该选择一定更有效。",
        "",
        "## 五、论文完成度判断",
        "",
        "当前已经具备：研究对象、研究问题、数据来源、变量体系、空间划分、描述性结果、情境比较、稳定性指标和可复现脚本。",
        "",
        "若要形成可投稿初稿，还需要补齐：",
        "",
        "1. 将身份映射、数据筛选和变量归并写成正式方法章节；",
        "2. 将结果图表压缩为4—6幅论文主图，并统一图注和统计口径；",
        "3. 增加研究假设或明确的探索性研究声明；",
        "4. 完成讨论部分，将结果与ShuttleSet和既有羽毛球技术—战术研究对照；",
        "5. 由研究者最终核对身份映射、球种上位分类和关键图表。",
        "",
        "因此，当前版本可以视为‘论文实证部分已经成形’，但还不是完整投稿稿件。最适合的论文定位是单运动员、多比赛的探索性个案研究。",
        "",
        "## 六、参考项目与文献",
        "",
        "- ShuttleSet：逐拍球种、击球位置、落点和双方位置的联合表示。https://arxiv.org/abs/2306.04948",
        "- Badminton Strike Positions：使用空间熵描述击球分布。https://www.mdpi.com/1099-4300/23/7/799",
        "- Singles Badminton Technical-Tactical Instrument：以单次击球为单位组织技术—战术变量。https://pmc.ncbi.nlm.nih.gov/articles/PMC7758221/",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    add_font()
    mapping = base.load_mapping()
    rows = base.normalize_events(mapping)
    classified = [row for row in rows if row["macro_type"] != "未分类"]
    specs = {
        "own_depth": [("前场", "后场"), ("中场", "后场")],
        "opponent_depth": [("前场", "后场"), ("中场", "后场")],
        "phase": [("开局阶段", "延长阶段"), ("组织阶段", "延长阶段")],
    }
    context_orders = {
        "own_depth": base.DEPTH_ORDER,
        "opponent_depth": base.DEPTH_ORDER,
        "phase": base.PHASE_ORDER,
    }
    results: dict[str, dict] = {}
    for index, (key, order) in enumerate(context_orders.items()):
        results[key] = bootstrap_context(classified, key, order, reps=1000, seed=20260817 + index)

    all_ci_rows = [row for result in results.values() for row in result["ci_rows"]]
    base.write_csv(OUT / "context_bootstrap_ci.csv", all_ci_rows)
    base.write_csv(OUT / "context_association_bootstrap.csv", [result["association"] for result in results.values()])
    contrasts = build_contrasts(results, specs)
    base.write_csv(OUT / "context_contrasts.csv", contrasts)
    base.write_csv(OUT / "paper_variables.csv", paper_variables())
    plot_contrasts(contrasts, OUT / "figure_context_effects.png")

    summary = json.loads((OUT / "analysis_summary.json").read_text(encoding="utf-8"))
    summary["bootstrap"] = {
        key: result["association"] for key, result in results.items()
    }
    (OUT / "analysis_summary_v2.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "RESEARCH_REPORT_v2.md").write_text(build_v2_report(summary, results, contrasts), encoding="utf-8")
    print(json.dumps({key: result["association"] for key, result in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
