from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import shi_tactical_profile_v1 as base


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "data" / "batch_runs" / "broadcast-uncensored-v1" / "videos"
MAPPING_PATH = ROOT / "research" / "identity_mapping.csv"
OUT = ROOT / "research_outputs" / "shi_tactical_profile_v1"
OUT.mkdir(parents=True, exist_ok=True)

COURT_WIDTH = base.COURT_WIDTH
COURT_LENGTH = base.COURT_LENGTH

UNCLASSIFIED = "未分类"
LINE_ORDER = ["左偏", "中路", "右偏"]
ZONE9_ORDER = [f"{depth}-{width}" for depth in base.DEPTH_ORDER for width in base.WIDTH_ORDER]

SMASH_TYPES = {"smash", "wrist smash"}
DROP_TYPES = {"drop", "passive drop"}
ATTACK_MACROS = {"后场进攻"}
ACTIVE_MACROS = {"后场进攻", "网前/前场", "平抽/推挡"}
NEUTRAL_DEFENSE_MACROS = {"网前/前场", "高远/挑球", "平抽/推挡", "吊球"}
DEFENSE_MACROS = {"高远/挑球"}

STYLE_RADAR = [
    "smash_share",
    "drop_share",
    "net_share",
    "drive_share",
    "backcourt_tendency",
    "rally_tempo",
    "spatial_diversity",
    "sequence_predictability",
]

CAPABILITY_PROXY_RADAR = [
    "attack_build_proxy",
    "attack_finish_proxy",
    "defense_to_attack_proxy",
    "net_engagement",
    "long_rally_share",
]

AUXILIARY_METRICS = [
    "opening_forward_share",
    "initiative_proxy",
    "first_three_bias",
    "net_continuation_proxy",
    "space_coverage_proxy",
    "landing_entropy",
    "line_entropy",
]

CAPABILITY_FULL_AXES = [
    "opening_and_return",
    "attack_build",
    "attack_finish",
    "net_control",
    "defense_to_attack",
    "rally_resilience",
    "error_control",
    "spatial_control",
]

METRIC_LABELS = {
    "smash_share": "杀球倾向",
    "drop_share": "吊球倾向",
    "net_share": "网前倾向",
    "drive_share": "抽挡倾向",
    "backcourt_tendency": "后场倾向",
    "rally_tempo": "回合节奏",
    "spatial_diversity": "空间多样性",
    "sequence_predictability": "序列可预测性",
    "attack_build_proxy": "进攻构建代理",
    "attack_finish_proxy": "进攻终结代理",
    "defense_to_attack_proxy": "防守转攻代理",
    "net_engagement": "网前参与",
    "long_rally_share": "多拍参与",
    "opening_forward_share": "前三拍前场/主动比例",
    "initiative_proxy": "主动性代理",
    "first_three_bias": "前三拍偏好",
    "net_continuation_proxy": "网前连续控制代理",
    "space_coverage_proxy": "落点覆盖代理",
    "landing_entropy": "落点熵",
    "line_entropy": "横向线路熵",
    "opening_and_return": "开局与接发",
    "attack_build": "进攻构建",
    "attack_finish": "进攻终结",
    "net_control": "网前控制",
    "defense_to_attack": "防守转攻",
    "rally_resilience": "多拍韧性",
    "error_control": "失误控制",
    "spatial_control": "空间控制",
}

METRIC_NOTES = {
    "smash_share": "已分类击球中 raw stroke_type 为 smash 或 wrist smash 的比例。",
    "drop_share": "已分类击球中 raw stroke_type 为 drop 或 passive drop 的比例。",
    "net_share": "已分类击球中归入网前/前场宏观类别的比例。",
    "drive_share": "已分类击球中归入平抽/推挡宏观类别的比例。",
    "backcourt_tendency": "有效自身位置中，击球点位于后场的比例。",
    "rally_tempo": "每场比赛中，逐回合（石宇奇/对手全体事件）时间内石宇奇事件数/秒的中位数。",
    "landing_entropy": "九宫格预测落点的归一化 Shannon 熵；越高表示落点使用更分散。",
    "line_entropy": "delta_x = landing_x - own_x 的左偏/中路/右偏三分类熵，是横向线路多样性代理。",
    "spatial_diversity": "落点熵与横向线路熵的平均值；不是空间控制或落点效果。",
    "sequence_predictability": "石宇奇连续已分类击球宏观类别转移的 1 - 归一化条件熵。",
    "initiative_proxy": "已分类击球中后场进攻、网前/前场和平抽/推挡的比例。",
    "opening_forward_share": "前三拍已分类击球中主动/前场宏观类别的比例。",
    "first_three_bias": "前三拍主动/前场比例减去全回合主动性代理；正值表示主动类别更集中于前三拍。",
    "attack_build_proxy": "石宇奇相邻已分类击球中，中性/防守类别后紧接后场进攻的比例。",
    "attack_finish_proxy": "石宇奇每个回合最后一次已分类击球为后场进攻的比例；不等于得分。",
    "defense_to_attack_proxy": "高远/挑球后 1—2 次石宇奇已分类击球内进入后场进攻的比例。",
    "net_engagement": "已分类击球中网前/前场类别的比例。",
    "net_continuation_proxy": "网前/前场击球后，下一次石宇奇已分类击球仍位于前场或中场的比例。",
    "long_rally_share": "石宇奇参与的回合中，全体事件数至少为 10 的回合比例。",
    "space_coverage_proxy": "有效预测落点覆盖的九宫格区域数/9。",
    "opening_and_return": "当前完整效果轴需要发球/接发回合结果；本版本仅用前三拍结构作代理。",
    "attack_build": "当前完整效果轴需要进攻构建后的回合结果；本版本用 attack_build_proxy 作代理。",
    "attack_finish": "当前完整效果轴需要终结性进攻结果；本版本用 attack_finish_proxy 作代理。",
    "net_control": "当前完整效果轴需要网前后回合结果；本版本用 net_engagement 作代理。",
    "defense_to_attack": "当前完整效果轴需要防守转攻后的回合结果；本版本用 defense_to_attack_proxy 作代理。",
    "rally_resilience": "当前完整效果轴需要长回合胜率；本版本用 long_rally_share 作代理。",
    "error_control": "当前数据没有主动/非受迫失误或回合结束方式，不能估计。",
    "spatial_control": "当前数据只有落点使用代理，没有落点效果、对手被动状态或回合结果，不能估计。",
}


def is_finite(value: object) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def clean_value(value: object) -> object:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def mean_or_none(values: list[float]) -> float | None:
    finite = [float(value) for value in values if is_finite(value)]
    return float(np.mean(finite)) if finite else None


def median_or_none(values: list[float]) -> float | None:
    finite = [float(value) for value in values if is_finite(value)]
    return float(np.median(finite)) if finite else None


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def event_sort_key(row: dict) -> tuple[float, float, str]:
    stroke_index = row.get("stroke_index")
    hit_time = row.get("hit_time")
    return (
        float(stroke_index) if is_finite(stroke_index) else 10**12,
        float(hit_time) if is_finite(hit_time) else 10**12,
        str(row.get("event_id") or ""),
    )


def line_class(own_x: float | None, landing_x: float | None) -> str | None:
    if not is_finite(own_x) or not is_finite(landing_x):
        return None
    delta = float(landing_x) - float(own_x)
    threshold = COURT_WIDTH / 6
    if delta < -threshold:
        return "左偏"
    if delta > threshold:
        return "右偏"
    return "中路"


def read_match_events(video_id: str, meta: dict[str, str]) -> list[dict]:
    path = BATCH / video_id / "dataset" / "dataset_rows.csv"
    if not path.exists():
        return []
    shi_side = meta["shi_side"]
    sides = {shi_side, "lower" if shi_side == "upper" else "upper"}
    rows: list[dict] = []
    for raw in base.read_csv(path):
        player_side = (raw.get("player") or "").strip().lower()
        if player_side not in sides:
            continue
        opponent_side = "lower" if player_side == "upper" else "upper"
        own_x = base.as_float(raw.get("player_location_x"))
        own_y = base.as_float(raw.get("player_location_y"))
        opponent_x = base.as_float(raw.get("opponent_location_x"))
        opponent_y = base.as_float(raw.get("opponent_location_y"))
        landing_x = base.as_float(raw.get("landing_x"))
        landing_y = base.as_float(raw.get("landing_y"))
        stroke_type = (raw.get("stroke_type") or "unknown").strip().lower()
        macro_type = base.RAW_TO_MACRO.get(stroke_type, UNCLASSIFIED)
        rally_id = (raw.get("rally_id") or "").strip()
        if not rally_id:
            rally_id = f"event-{raw.get('event_id') or len(rows)}"
        rows.append(
            {
                "video_id": video_id,
                "opponent": meta["opponent"],
                "competition": meta["competition"],
                "shi_side": shi_side,
                "player_side": player_side,
                "player_label": "石宇奇" if player_side == shi_side else "对手",
                "event_id": raw.get("event_id", ""),
                "rally_id": rally_id,
                "stroke_index": base.as_int(raw.get("stroke_index")),
                "hit_time": base.as_float(raw.get("hit_time")),
                "stroke_type": stroke_type,
                "macro_type": macro_type,
                "own_x": own_x,
                "own_y": own_y,
                "opponent_x": opponent_x,
                "opponent_y": opponent_y,
                "landing_x": landing_x,
                "landing_y": landing_y,
                "own_depth": base.depth_zone(own_y, player_side),
                "opponent_depth": base.depth_zone(opponent_y, opponent_side),
                "landing_zone9": base.zone9(base.depth_zone(landing_y, opponent_side), base.width_zone(landing_x)),
                "line_class": line_class(own_x, landing_x),
            }
        )
    return rows


def normalized_entropy(counts: Counter[str], categories: list[str]) -> float | None:
    total = sum(counts.get(category, 0) for category in categories)
    if not total:
        return None
    probabilities = np.array([counts.get(category, 0) / total for category in categories], dtype=float)
    return float(base.entropy(probabilities))


def transition_predictability(sequences: list[list[str]]) -> tuple[float | None, int]:
    index = {category: i for i, category in enumerate(base.MACRO_ORDER)}
    counts = np.zeros((len(base.MACRO_ORDER), len(base.MACRO_ORDER)), dtype=float)
    for sequence in sequences:
        for previous, current in zip(sequence, sequence[1:]):
            if previous in index and current in index:
                counts[index[previous], index[current]] += 1
    total_pairs = int(counts.sum())
    if total_pairs == 0:
        return None, 0
    conditional_entropy = 0.0
    for row in counts:
        row_total = row.sum()
        if row_total == 0:
            continue
        probabilities = row[row > 0] / row_total
        conditional_entropy += (row_total / total_pairs) * float(-(probabilities * np.log(probabilities)).sum())
    maximum_entropy = math.log(len(base.MACRO_ORDER))
    predictability = 1 - conditional_entropy / maximum_entropy if maximum_entropy else None
    return float(predictability), total_pairs


def build_player_metrics(
    video_id: str,
    meta: dict[str, str],
    rows: list[dict],
    player_side: str,
    player_label: str,
) -> dict:
    player_rows = [row for row in rows if row["player_side"] == player_side]
    by_rally: dict[str, list[dict]] = defaultdict(list)
    all_by_rally: dict[str, list[dict]] = defaultdict(list)
    for row in player_rows:
        by_rally[row["rally_id"]].append(row)
    for row in rows:
        all_by_rally[row["rally_id"]].append(row)

    classified_rows = [row for row in player_rows if row["macro_type"] in base.MACRO_ORDER]
    classified_by_rally: dict[str, list[dict]] = defaultdict(list)
    for row in classified_rows:
        classified_by_rally[row["rally_id"]].append(row)
    sequences: list[list[str]] = []
    tempo_values: list[float] = []

    attack_build_opportunities = 0
    attack_build_successes = 0
    attack_finish_rallies = 0
    defense_opportunities = 0
    defense_to_attack_successes = 0
    net_continuation_opportunities = 0
    net_continuation_successes = 0

    long_rally_count = 0
    for rally_id, rally_player_rows in by_rally.items():
        rally_player_rows.sort(key=event_sort_key)
        rally_classified = sorted(classified_by_rally.get(rally_id, []), key=event_sort_key)
        sequence = [row["macro_type"] for row in rally_classified]
        if sequence:
            sequences.append(sequence)
            if sequence[-1] in ATTACK_MACROS:
                attack_finish_rallies += 1
            for previous, current in zip(sequence, sequence[1:]):
                if previous in NEUTRAL_DEFENSE_MACROS:
                    attack_build_opportunities += 1
                    if current in ATTACK_MACROS:
                        attack_build_successes += 1
            for index, current in enumerate(sequence):
                if current in DEFENSE_MACROS:
                    defense_opportunities += 1
                    if any(target in ATTACK_MACROS for target in sequence[index + 1 : index + 3]):
                        defense_to_attack_successes += 1
            for index, row in enumerate(rally_classified[:-1]):
                if row["macro_type"] == "网前/前场":
                    net_continuation_opportunities += 1
                    next_row = rally_classified[index + 1]
                    if next_row.get("own_depth") in {"前场", "中场"}:
                        net_continuation_successes += 1

        all_rows = all_by_rally.get(rally_id, [])
        all_times = [float(row["hit_time"]) for row in all_rows if is_finite(row.get("hit_time"))]
        if len(all_times) >= 2:
            duration = max(all_times) - min(all_times)
            if duration > 0:
                tempo_values.append(len(rally_player_rows) / duration)
        if len(all_rows) >= 10:
            long_rally_count += 1

    classified_count = len(classified_rows)
    classified_rally_count = len(classified_by_rally)
    player_rally_count = len(by_rally)
    valid_position_count = sum(row.get("own_depth") is not None for row in player_rows)
    valid_landing_count = sum(row.get("landing_zone9") is not None for row in player_rows)
    valid_line_count = sum(row.get("line_class") is not None for row in player_rows)

    macro_counts = Counter(row["macro_type"] for row in classified_rows)
    raw_counts = Counter(row["stroke_type"] for row in classified_rows)
    landing_counts = Counter(row["landing_zone9"] for row in player_rows if row.get("landing_zone9"))
    line_counts = Counter(row["line_class"] for row in player_rows if row.get("line_class"))

    active_count = sum(row["macro_type"] in ACTIVE_MACROS for row in classified_rows)
    first_three_rows = [
        row for row in classified_rows if is_finite(row.get("stroke_index")) and int(row["stroke_index"]) <= 3
    ]
    first_three_active_count = sum(row["macro_type"] in ACTIVE_MACROS for row in first_three_rows)
    initiative_proxy = ratio(active_count, classified_count)
    opening_forward_share = ratio(first_three_active_count, len(first_three_rows))
    first_three_bias = (
        opening_forward_share - initiative_proxy
        if is_finite(opening_forward_share) and is_finite(initiative_proxy)
        else None
    )
    landing_entropy = normalized_entropy(landing_counts, ZONE9_ORDER)
    line_entropy = normalized_entropy(line_counts, LINE_ORDER)
    spatial_parts = [value for value in [landing_entropy, line_entropy] if is_finite(value)]
    spatial_diversity = mean_or_none(spatial_parts)
    sequence_predictability, sequence_pairs = transition_predictability(sequences)
    net_engagement = ratio(macro_counts.get("网前/前场", 0), classified_count)

    metrics = {
        "smash_share": ratio(sum(raw_counts.get(name, 0) for name in SMASH_TYPES), classified_count),
        "drop_share": ratio(sum(raw_counts.get(name, 0) for name in DROP_TYPES), classified_count),
        "net_share": net_engagement,
        "drive_share": ratio(macro_counts.get("平抽/推挡", 0), classified_count),
        "backcourt_tendency": ratio(
            sum(row.get("own_depth") == "后场" for row in player_rows),
            valid_position_count,
        ),
        "rally_tempo": median_or_none(tempo_values),
        "landing_entropy": landing_entropy,
        "line_entropy": line_entropy,
        "spatial_diversity": spatial_diversity,
        "sequence_predictability": sequence_predictability,
        "initiative_proxy": initiative_proxy,
        "opening_forward_share": opening_forward_share,
        "first_three_bias": first_three_bias,
        "attack_build_proxy": ratio(attack_build_successes, attack_build_opportunities),
        "attack_finish_proxy": ratio(attack_finish_rallies, classified_rally_count),
        "defense_to_attack_proxy": ratio(defense_to_attack_successes, defense_opportunities),
        "net_engagement": net_engagement,
        "net_continuation_proxy": ratio(net_continuation_successes, net_continuation_opportunities),
        "long_rally_share": ratio(long_rally_count, player_rally_count),
        "space_coverage_proxy": ratio(len(landing_counts), len(ZONE9_ORDER)),
    }

    metric_ns = {
        "smash_share": classified_count,
        "drop_share": classified_count,
        "net_share": classified_count,
        "drive_share": classified_count,
        "backcourt_tendency": valid_position_count,
        "rally_tempo": len(tempo_values),
        "landing_entropy": valid_landing_count,
        "line_entropy": valid_line_count,
        "spatial_diversity": min(valid_landing_count, valid_line_count)
        if valid_landing_count and valid_line_count
        else max(valid_landing_count, valid_line_count),
        "sequence_predictability": sequence_pairs,
        "initiative_proxy": classified_count,
        "opening_forward_share": len(first_three_rows),
        "first_three_bias": len(first_three_rows),
        "attack_build_proxy": attack_build_opportunities,
        "attack_finish_proxy": classified_rally_count,
        "defense_to_attack_proxy": defense_opportunities,
        "net_engagement": classified_count,
        "net_continuation_proxy": net_continuation_opportunities,
        "long_rally_share": player_rally_count,
        "space_coverage_proxy": valid_landing_count,
    }

    record = {
        "video_id": video_id,
        "opponent": meta["opponent"],
        "competition": meta["competition"],
        "athlete": player_label,
        "side": player_side,
        "shi_side": meta["shi_side"],
        "rally_count": player_rally_count,
        "stroke_events": len(player_rows),
        "classified_strokes": classified_count,
        "unclassified_strokes": len(player_rows) - classified_count,
        "classified_rate": ratio(classified_count, len(player_rows)),
        "valid_own_position": valid_position_count,
        "valid_landing": valid_landing_count,
        "valid_line": valid_line_count,
        "attack_build_successes": attack_build_successes,
        "attack_build_opportunities": attack_build_opportunities,
        "defense_to_attack_successes": defense_to_attack_successes,
        "defense_opportunities": defense_opportunities,
        "sequence_pairs": sequence_pairs,
        "landing_zone_count": len(landing_counts),
        "tempo_valid_rallies": len(tempo_values),
    }
    record.update(metrics)
    for metric, n in metric_ns.items():
        record[f"{metric}_n"] = n
    return {key: clean_value(value) for key, value in record.items()}


def bootstrap_mean_ci(values: list[float], seed: int, repetitions: int = 1000) -> tuple[float | None, float | None]:
    finite = np.asarray([float(value) for value in values if is_finite(value)], dtype=float)
    if len(finite) == 0:
        return None, None
    if len(finite) == 1:
        value = float(finite[0])
        return value, value
    rng = np.random.default_rng(seed)
    samples = rng.choice(finite, size=(repetitions, len(finite)), replace=True)
    means = samples.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate_metric(match_rows: list[dict], athlete: str, metric: str, seed: int) -> dict:
    selected = [row for row in match_rows if row.get("athlete") == athlete]
    values = [float(row[metric]) for row in selected if is_finite(row.get(metric))]
    observations = [
        float(row.get(f"{metric}_n"))
        for row in selected
        if is_finite(row.get(f"{metric}_n"))
    ]
    ci_low, ci_high = bootstrap_mean_ci(values, seed)
    return {
        "athlete": athlete,
        "metric": metric,
        "metric_label": METRIC_LABELS.get(metric, metric),
        "raw_mean": mean_or_none(values),
        "raw_median": median_or_none(values),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_total_matches": len(selected),
        "n_valid_matches": len(values),
        "n_observations": int(sum(observations)) if observations else 0,
        "status": "可计算" if values else "无有效数据",
    }


def peer_reference(values: list[float]) -> tuple[float | None, float | None]:
    finite = np.asarray([float(value) for value in values if is_finite(value)], dtype=float)
    if len(finite) == 0:
        return None, None
    return float(np.quantile(finite, 0.05)), float(np.quantile(finite, 0.95))


def display_score(value: float | None, lower: float | None, upper: float | None) -> float | None:
    if not is_finite(value) or not is_finite(lower) or not is_finite(upper):
        return None
    span = float(upper) - float(lower)
    if abs(span) < 1e-12:
        return 50.0
    return float(np.clip((float(value) - float(lower)) / span, 0, 1) * 100)


def build_metric_definitions() -> list[dict]:
    rows: list[dict] = []
    for metric in STYLE_RADAR:
        rows.append(
            {
                "radar": "打法风格主雷达",
                "metric": metric,
                "label": METRIC_LABELS[metric],
                "metric_type": "风格指标",
                "availability": "当前可计算",
                "direction": "高值表示该风格特征更强，不表示效果更好",
                "definition": METRIC_NOTES[metric],
            }
        )
    for metric in CAPABILITY_PROXY_RADAR:
        rows.append(
            {
                "radar": "能力代理辅助雷达",
                "metric": metric,
                "label": METRIC_LABELS[metric],
                "metric_type": "能力代理指标",
                "availability": "当前可计算，但不等于比赛效果",
                "direction": "高值表示代理结构更强，不表示已获得得分",
                "definition": METRIC_NOTES[metric],
            }
        )
    for metric in AUXILIARY_METRICS:
        rows.append(
            {
                "radar": "辅助指标",
                "metric": metric,
                "label": METRIC_LABELS[metric],
                "metric_type": "辅助描述指标",
                "availability": "当前可计算",
                "direction": "按原始值解释",
                "definition": METRIC_NOTES[metric],
            }
        )
    for metric in CAPABILITY_FULL_AXES:
        availability = "当前不可估计" if metric in {"error_control", "spatial_control"} else "当前只支持代理"
        rows.append(
            {
                "radar": "完整比赛效果雷达",
                "metric": metric,
                "label": METRIC_LABELS[metric],
                "metric_type": "结果/能力指标",
                "availability": availability,
                "direction": "需回合结果后定义",
                "definition": METRIC_NOTES[metric],
            }
        )
    return rows


def radar_plot(
    score_rows: list[dict],
    metric_order: list[str],
    title: str,
    output: Path,
    color: str,
) -> None:
    values = [float(next(row["display_score"] for row in score_rows if row["metric"] == metric)) for metric in metric_order]
    labels = [METRIC_LABELS[metric] for metric in metric_order]
    angles = np.linspace(0, 2 * np.pi, len(metric_order), endpoint=False)
    angles_closed = np.concatenate([angles, angles[:1]])
    values_closed = np.concatenate([np.asarray(values, dtype=float), [values[0]]])

    fig, axis = plt.subplots(figsize=(8.2, 8.2), subplot_kw={"polar": True})
    axis.plot(angles_closed, values_closed, color=color, linewidth=2.2)
    axis.fill(angles_closed, values_closed, color=color, alpha=0.22)
    axis.set_ylim(0, 100)
    axis.set_yticks([20, 40, 60, 80, 100])
    axis.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    axis.set_xticks(angles)
    axis.set_xticklabels(labels, fontsize=10)
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_title(title, pad=28, fontsize=14)
    axis.grid(alpha=0.35)
    fig.text(
        0.5,
        0.02,
        "0–100 为相对于同场对手比赛级 P05–P95 分布的显示分数，不代表绝对能力等级。",
        ha="center",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return clean_value(value)


def format_metric_value(metric: str, value: float | None) -> str:
    if not is_finite(value):
        return "NA"
    if metric == "rally_tempo":
        return f"{float(value):.3f} 次/秒"
    if metric == "first_three_bias":
        return f"{float(value) * 100:.1f} 个百分点"
    return f"{float(value) * 100:.1f}%"


def build_report(
    summary: dict,
    raw_rows: list[dict],
    score_rows: list[dict],
) -> str:
    lines = [
        "# 石宇奇球员能力与打法风格雷达：第一版实证结果",
        "",
        "> 研究版本：v1.0；本版严格区分“打法风格指标”和“比赛效果/能力代理指标”。所有 0–100 分均为当前 24 场比赛中相对于同场对手分布的显示分数，不代表绝对能力等级。",
        "",
        "## 1. 数据范围",
        "",
        f"- 纳入比赛：{summary['match_count']} 场；逐拍事件：{summary['event_count']} 次；石宇奇/对手各生成一条比赛级记录。",
        f"- 石宇奇已分类击球：{summary['shi_classified_strokes']} 次；未分类击球：{summary['shi_unclassified_strokes']} 次。",
        f"- 石宇奇参与回合：{summary['shi_rally_count']} 个；有效预测落点：{summary['shi_valid_landing']} 次。",
        "- 分析单位为比赛级指标；Bootstrap 重抽样单位为比赛，不把逐拍事件视作相互独立的运动员样本。",
        "",
        "## 2. 评分方法",
        "",
        "对每个指标，先分别计算每场比赛的原始值，再用石宇奇同场对手的比赛级指标作为参照：",
        "",
        "display_score = 100 × clip((石宇奇原始值 − 对手 P05) / (对手 P95 − 对手 P05), 0, 1)。",
        "",
        "图中分数只用于比较当前样本中的相对位置；论文正文应同时报告原始值、有效比赛数和 95% Bootstrap CI。",
        "",
        "## 3. 打法风格主雷达",
        "",
        "| 指标 | 原始均值 | 95% CI | 显示分数 | 有效比赛 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in STYLE_RADAR:
        score = next(row for row in score_rows if row["radar"] == "style_main" and row["metric"] == metric)
        raw = next(row for row in raw_rows if row["athlete"] == "石宇奇" and row["metric"] == metric)
        ci = f"{format_metric_value(metric, raw['ci_low'])}—{format_metric_value(metric, raw['ci_high'])}"
        raw_value = format_metric_value(metric, raw["raw_mean"])
        display = f"{float(score['display_score']):.1f}" if is_finite(score.get("display_score")) else "NA"
        lines.append(f"| {METRIC_LABELS[metric]} | {raw_value} | {ci} | {display} | {raw['n_valid_matches']} |")
    lines += [
        "",
        "该雷达描述石宇奇“怎么打”，不是“打得是否有效”。例如，杀球倾向更高只表示杀球在已分类击球中的相对占比更高。",
        "",
        "## 4. 能力代理辅助雷达",
        "",
        "| 指标 | 原始均值 | 95% CI | 显示分数 | 有效比赛 |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in CAPABILITY_PROXY_RADAR:
        score = next(row for row in score_rows if row["radar"] == "capability_proxy" and row["metric"] == metric)
        raw = next(row for row in raw_rows if row["athlete"] == "石宇奇" and row["metric"] == metric)
        ci = f"{format_metric_value(metric, raw['ci_low'])}—{format_metric_value(metric, raw['ci_high'])}"
        display = f"{float(score['display_score']):.1f}" if is_finite(score.get("display_score")) else "NA"
        lines.append(
            f"| {METRIC_LABELS[metric]} | {format_metric_value(metric, raw['raw_mean'])} | {ci} | {display} | {raw['n_valid_matches']} |"
        )
    lines += [
        "",
        "这些指标是击球结构代理：进攻构建、进攻终结、防守转攻和多拍参与都没有绑定回合胜负，因此不能写成得分率、成功率或能力的因果证据。",
        "",
        "## 5. 当前不能估计的完整效果轴",
        "",
        "- 失误控制：缺少主动/非受迫失误与回合结束方式。",
        "- 真实空间控制：缺少落点效果、对手被动状态和回合结果。",
        "- 开局与接发的真实效果：前三拍结构可以计算，但缺少发球/接发回合胜率。",
        "- 因此没有用 0 分填充不可估计轴，也没有把代理指标伪装成真实比赛效果。",
        "",
        "## 6. 结果文件",
        "",
        "- player_radar_match_metrics.csv：石宇奇与同场对手的比赛级原始指标。",
        "- player_radar_raw.csv：按运动员和指标汇总的均值、中位数、Bootstrap CI 与有效样本量。",
        "- player_radar_scores.csv：相对于对手 P05–P95 的 0–100 显示分数。",
        "- player_radar_metric_definitions.csv：指标定义、类型和当前可行性边界。",
        "- figure_player_style_radar.png：打法风格主雷达。",
        "- figure_player_capability_proxy_radar.png：能力代理辅助雷达。",
        "",
        "## 7. 论文使用边界",
        "",
        "- 论文正文应将“回合节奏”写成事件频率代理，将“线路熵”写成横向线路多样性代理。",
        "- 雷达图只作综合可视化，统计解释以比赛级原始值、CI、配对差值和样本量为准。",
        "- 下一版如果补充 rally_winner、point_end_type、forced_error/unforced_error 等字段，可把代理雷达升级为真正的比赛效果雷达。",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    base.add_chinese_font()
    mapping = base.load_mapping()
    match_metrics: list[dict] = []
    match_event_count = 0
    for index, (video_id, meta) in enumerate(sorted(mapping.items()), start=1):
        rows = read_match_events(video_id, meta)
        if not rows:
            continue
        match_event_count += len(rows)
        for player_side, player_label in [
            (meta["shi_side"], "石宇奇"),
            ("lower" if meta["shi_side"] == "upper" else "upper", "对手"),
        ]:
            match_metrics.append(build_player_metrics(video_id, meta, rows, player_side, player_label))
        print(f"[{index}/{len(mapping)}] {video_id}: {len(rows)} events")

    if not match_metrics:
        raise SystemExit("No player radar events were found")

    base.write_csv(OUT / "player_radar_match_metrics.csv", match_metrics)

    metric_order = STYLE_RADAR + CAPABILITY_PROXY_RADAR + AUXILIARY_METRICS
    raw_rows: list[dict] = []
    for metric_index, metric in enumerate(metric_order):
        for athlete_index, athlete in enumerate(["石宇奇", "对手"]):
            row = aggregate_metric(match_metrics, athlete, metric, 20260820 + metric_index * 17 + athlete_index)
            row["radar"] = (
                "style_main"
                if metric in STYLE_RADAR
                else "capability_proxy"
                if metric in CAPABILITY_PROXY_RADAR
                else "auxiliary"
            )
            row["direction"] = "higher"
            raw_rows.append(row)
    base.write_csv(OUT / "player_radar_raw.csv", raw_rows)

    score_rows: list[dict] = []
    for radar_name, radar_metrics in [
        ("style_main", STYLE_RADAR),
        ("capability_proxy", CAPABILITY_PROXY_RADAR),
    ]:
        for metric_index, metric in enumerate(radar_metrics):
            shi_raw = next(row for row in raw_rows if row["athlete"] == "石宇奇" and row["metric"] == metric)
            peer_values = [
                float(row[metric])
                for row in match_metrics
                if row["athlete"] == "对手" and is_finite(row.get(metric))
            ]
            peer_p05, peer_p95 = peer_reference(peer_values)
            paired_deltas = []
            for video_id in sorted({row["video_id"] for row in match_metrics}):
                shi_match = next(
                    (row for row in match_metrics if row["video_id"] == video_id and row["athlete"] == "石宇奇"),
                    None,
                )
                peer_match = next(
                    (row for row in match_metrics if row["video_id"] == video_id and row["athlete"] == "对手"),
                    None,
                )
                if shi_match and peer_match and is_finite(shi_match.get(metric)) and is_finite(peer_match.get(metric)):
                    paired_deltas.append(float(shi_match[metric]) - float(peer_match[metric]))
            delta_low, delta_high = bootstrap_mean_ci(paired_deltas, 20300820 + metric_index)
            score_rows.append(
                {
                    "athlete": "石宇奇",
                    "radar": radar_name,
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "raw_value": shi_raw["raw_mean"],
                    "display_score": display_score(shi_raw["raw_mean"], peer_p05, peer_p95),
                    "peer_p05": peer_p05,
                    "peer_p95": peer_p95,
                    "paired_delta_mean": mean_or_none(paired_deltas),
                    "paired_delta_ci_low": delta_low,
                    "paired_delta_ci_high": delta_high,
                    "n_valid_match_metrics": shi_raw["n_valid_matches"],
                    "n_peer_match_metrics": len(peer_values),
                    "status": "可计算",
                    "interpretation": "相对分数越高表示该结构指标在当前样本中高于更多同场对手，不表示比赛效果更好。",
                }
            )

    for metric in ["error_control", "spatial_control"]:
        score_rows.append(
            {
                "athlete": "石宇奇",
                "radar": "capability_full_unavailable",
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "raw_value": None,
                "display_score": None,
                "peer_p05": None,
                "peer_p95": None,
                "paired_delta_mean": None,
                "paired_delta_ci_low": None,
                "paired_delta_ci_high": None,
                "n_valid_match_metrics": 0,
                "n_peer_match_metrics": 0,
                "status": "当前不可估计",
                "interpretation": METRIC_NOTES[metric],
            }
        )
    base.write_csv(OUT / "player_radar_scores.csv", score_rows)

    definitions = build_metric_definitions()
    base.write_csv(OUT / "player_radar_metric_definitions.csv", definitions)

    style_scores = [row for row in score_rows if row["radar"] == "style_main"]
    capability_scores = [row for row in score_rows if row["radar"] == "capability_proxy"]
    radar_plot(
        style_scores,
        STYLE_RADAR,
        "石宇奇打法风格主雷达",
        OUT / "figure_player_style_radar.png",
        "#2c7fb8",
    )
    radar_plot(
        capability_scores,
        CAPABILITY_PROXY_RADAR,
        "石宇奇能力代理辅助雷达",
        OUT / "figure_player_capability_proxy_radar.png",
        "#d95f02",
    )

    shi_rows = [row for row in match_metrics if row["athlete"] == "石宇奇"]
    summary = {
        "version": "player_radar_v1.0",
        "generated_date": date.today().isoformat(),
        "match_count": len({row["video_id"] for row in match_metrics}),
        "event_count": match_event_count,
        "shi_rally_count": int(sum(int(row["rally_count"]) for row in shi_rows)),
        "shi_stroke_events": int(sum(int(row["stroke_events"]) for row in shi_rows)),
        "shi_classified_strokes": int(sum(int(row["classified_strokes"]) for row in shi_rows)),
        "shi_unclassified_strokes": int(sum(int(row["unclassified_strokes"]) for row in shi_rows)),
        "shi_valid_landing": int(sum(int(row["valid_landing"]) for row in shi_rows)),
        "style_metrics": [row for row in score_rows if row["radar"] == "style_main"],
        "capability_proxy_metrics": [row for row in score_rows if row["radar"] == "capability_proxy"],
        "unavailable_capability_axes": ["error_control", "spatial_control"],
        "files": [
            "player_radar_match_metrics.csv",
            "player_radar_raw.csv",
            "player_radar_scores.csv",
            "player_radar_metric_definitions.csv",
            "figure_player_style_radar.png",
            "figure_player_capability_proxy_radar.png",
            "PLAYER_RADAR_RESULTS_v1.md",
        ],
        "notes": [
            "0-100 分数是相对于同场对手 P05-P95 的可视化分数。",
            "Bootstrap 重抽样单位为比赛。",
            "错误控制和真实空间控制没有用 0 分填充。",
        ],
    }
    summary = json_safe(summary)
    (OUT / "player_radar_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report = build_report(summary, raw_rows, score_rows)
    (OUT / "PLAYER_RADAR_RESULTS_v1.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "match_count": summary["match_count"],
                "event_count": summary["event_count"],
                "shi_classified_strokes": summary["shi_classified_strokes"],
                "shi_unclassified_strokes": summary["shi_unclassified_strokes"],
                "shi_valid_landing": summary["shi_valid_landing"],
                "style_scores": [
                    {
                        "metric": row["metric"],
                        "score": row["display_score"],
                    }
                    for row in summary["style_metrics"]
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
