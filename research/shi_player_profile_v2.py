from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Sequence

_RUNTIME_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(_RUNTIME_ROOT / "research_outputs" / ".matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


ROOT = _RUNTIME_ROOT
BATCH = ROOT / "data" / "batch_runs" / "broadcast-uncensored-v1" / "videos"
MAPPING_PATH = ROOT / "research" / "identity_mapping.csv"
PROTOCOL_PATH = (
    ROOT
    / "research_outputs"
    / "shi_tactical_profile_v1"
    / "PLAYER_RADAR_RESEARCH_PROTOCOL_FINAL_v2.md"
)
OUT = ROOT / "research_outputs" / "shi_player_profile_v2"

VERSION = "shi_player_profile_v2.0"
PROTOCOL_VERSION = "shi_player_profile_protocol_v2.0-final"
SEED = 20260822
BOOTSTRAP_REPETITIONS = 10_000
EXPECTED = {
    "matches": 24,
    "match_player_rows": 48,
    "shi_events": 14_157,
    "shi_classified": 10_081,
    "shi_unknown": 4_076,
    "shi_rallies": 1_923,
    "shi_valid_position": 13_630,
    "shi_valid_landing": 6_459,
    "valid_side_events": 26_629,
}

COURT_WIDTH = 6.1
COURT_LENGTH = 13.4
NET_Y = COURT_LENGTH / 2
UNKNOWN = "unknown"
MACRO_ORDER = ["发球", "网前/前场", "后场进攻", "高远/挑球", "平抽/推挡", "吊球"]
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
RAW_CATEGORIES = sorted(RAW_TO_MACRO)
SMASH_TYPES = {"smash", "wrist smash"}
DROP_TYPES = {"drop", "passive drop"}
ACTIVE_MACROS = {"后场进攻", "网前/前场", "平抽/推挡"}
NEUTRAL_DEFENSE_MACROS = {"网前/前场", "高远/挑球", "平抽/推挡", "吊球"}
DEFENSE_MACROS = {"高远/挑球"}
LINE_ORDER = ["左偏", "中路", "右偏"]
DEPTH_ORDER = ["前场", "中场", "后场"]
WIDTH_ORDER = ["左侧", "中路", "右侧"]
ZONE9_ORDER = [f"{depth}-{width}" for depth in DEPTH_ORDER for width in WIDTH_ORDER]

STYLE_METRICS = [
    "smash_share",
    "drop_share",
    "net_share",
    "drive_share",
    "backcourt_tendency",
    "event_rate_proxy",
    "spatial_diversity",
    "sequence_predictability",
]
PROCESS_METRICS = [
    "early_active_share",
    "attack_entry_proxy",
    "terminal_attack_event_share",
    "net_continuation_proxy",
    "defense_to_attack_proxy",
]
AUXILIARY_METRICS = [
    "long_rally_share_10",
    "long_rally_share_8",
    "long_rally_share_12",
    "landing_entropy",
    "line_entropy",
    "landing_coverage",
    "classified_rate",
]
ALL_METRICS = STYLE_METRICS + PROCESS_METRICS + AUXILIARY_METRICS
METRIC_LABELS = {
    "smash_share": "杀球倾向",
    "drop_share": "吊球倾向",
    "net_share": "网前倾向",
    "drive_share": "抽挡倾向",
    "backcourt_tendency": "后场击球倾向",
    "event_rate_proxy": "事件频率代理",
    "spatial_diversity": "空间使用多样性",
    "sequence_predictability": "序列可预测性熵代理",
    "early_active_share": "前三拍主动类别比例",
    "attack_entry_proxy": "进攻进入代理",
    "terminal_attack_event_share": "终局后场进攻事件比例",
    "net_continuation_proxy": "网前连续使用代理",
    "defense_to_attack_proxy": "防守转攻代理",
    "long_rally_share_10": "长回合暴露（≥10拍）",
    "long_rally_share_8": "长回合暴露（≥8拍）",
    "long_rally_share_12": "长回合暴露（≥12拍）",
    "landing_entropy": "落点熵",
    "line_entropy": "横向位移熵",
    "landing_coverage": "落点覆盖代理",
    "classified_rate": "已分类率",
}
METRIC_FAMILY = {
    **{metric: "style" for metric in STYLE_METRICS},
    **{metric: "process_proxy" for metric in PROCESS_METRICS},
    **{metric: "auxiliary" for metric in AUXILIARY_METRICS},
}
LOW_N_THRESHOLDS = {
    "smash_share": 50,
    "drop_share": 50,
    "net_share": 50,
    "drive_share": 50,
    "backcourt_tendency": 50,
    "event_rate_proxy": 30,
    "spatial_diversity": 50,
    "sequence_predictability": 50,
    "early_active_share": 50,
    "attack_entry_proxy": 20,
    "terminal_attack_event_share": 20,
    "net_continuation_proxy": 20,
    "defense_to_attack_proxy": 20,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows([{key: clean_scalar(row.get(key)) for key in fields} for row in materialized])


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_for(label: str) -> int:
    suffix = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return (SEED + suffix) % (2**32 - 1)


def as_float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def as_int(value: object) -> int | None:
    value_float = as_float(value)
    return int(value_float) if value_float is not None else None


def finite(value: object) -> bool:
    return as_float(value) is not None


def clean_scalar(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, (float, np.floating)):
        return "" if not math.isfinite(float(value)) else format(float(value), ".15g")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return value


def ratio(numerator: float | int, denominator: float | int) -> float | None:
    return float(numerator) / float(denominator) if denominator else None


def mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if finite(value)]
    return float(np.mean(clean)) if clean else None


def median(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if finite(value)]
    return float(np.median(clean)) if clean else None


def sign(value: float | None, tolerance: float = 1e-12) -> int:
    if value is None or abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def valid_xy(x: float | None, y: float | None) -> bool:
    return x is not None and y is not None and 0 <= x <= COURT_WIDTH and 0 <= y <= COURT_LENGTH


def valid_x(x: float | None) -> bool:
    return x is not None and 0 <= x <= COURT_WIDTH


def valid_y(y: float | None) -> bool:
    return y is not None and 0 <= y <= COURT_LENGTH


def depth_zone(y: float | None, side: str) -> str | None:
    if y is None or side not in {"upper", "lower"} or not 0 <= y <= COURT_LENGTH:
        return None
    distance = y / NET_Y if side == "upper" else (COURT_LENGTH - y) / NET_Y
    if not 0 <= distance <= 1:
        return None
    if distance < 1 / 3:
        return "后场"
    if distance < 2 / 3:
        return "中场"
    return "前场"


def width_zone(x: float | None) -> str | None:
    if x is None or not 0 <= x <= COURT_WIDTH:
        return None
    if x < COURT_WIDTH / 3:
        return "左侧"
    if x < 2 * COURT_WIDTH / 3:
        return "中路"
    return "右侧"


def line_class(own_x: float | None, landing_x: float | None, divisor: int = 6) -> str | None:
    if own_x is None or landing_x is None:
        return None
    delta = landing_x - own_x
    threshold = COURT_WIDTH / divisor
    if delta < -threshold:
        return "左偏"
    if delta > threshold:
        return "右偏"
    return "中路"


def normalized_entropy(counts: Counter[str], categories: Sequence[str]) -> float | None:
    total = sum(int(counts.get(category, 0)) for category in categories)
    if total <= 0:
        return None
    probabilities = np.asarray([counts.get(category, 0) / total for category in categories], dtype=float)
    nonzero = probabilities[probabilities > 0]
    maximum = math.log(len(categories))
    return float(-(nonzero * np.log(nonzero)).sum() / maximum) if maximum else None


def ordered_rally(rows: Sequence[dict]) -> list[dict] | None:
    if not rows:
        return []
    if all(row.get("stroke_index") is not None for row in rows):
        ordered = sorted(rows, key=lambda row: (row["stroke_index"], row.get("hit_time") or math.inf, row["event_id"]))
        keys = [(row["stroke_index"], row.get("hit_time"), row["event_id"]) for row in ordered]
    elif all(row.get("hit_time") is not None for row in rows):
        ordered = sorted(rows, key=lambda row: (row["hit_time"], row["event_id"]))
        keys = [(row["hit_time"], row["event_id"]) for row in ordered]
    else:
        return None
    if len(keys) != len(set(keys)):
        return None
    return ordered


def transition_predictability(
    ordered_player_rallies: Sequence[Sequence[dict]],
    categories: Sequence[str] = MACRO_ORDER,
    use_raw: bool = False,
    include_unknown: bool = False,
) -> tuple[float | None, int]:
    index = {category: position for position, category in enumerate(categories)}
    matrix = np.zeros((len(categories), len(categories)), dtype=float)
    for rally in ordered_player_rallies:
        prior: str | None = None
        for row in rally:
            value = row["stroke_type"] if use_raw else row["macro_type"]
            if value == UNKNOWN and not include_unknown:
                prior = None
                continue
            if value not in index:
                prior = None
                continue
            if prior is not None:
                matrix[index[prior], index[value]] += 1
            prior = value
    total_pairs = int(matrix.sum())
    if total_pairs == 0:
        return None, 0
    conditional = 0.0
    for row in matrix:
        row_total = row.sum()
        if row_total <= 0:
            continue
        probabilities = row[row > 0] / row_total
        conditional += (row_total / total_pairs) * float(-(probabilities * np.log(probabilities)).sum())
    maximum = math.log(len(categories))
    return float(1 - conditional / maximum), total_pairs


def bootstrap_ci(values: Sequence[float], label: str, statistic: Callable[[np.ndarray], float] | None = None) -> tuple[float | None, float | None]:
    clean = np.asarray([float(value) for value in values if finite(value)], dtype=float)
    if clean.size == 0:
        return None, None
    if clean.size == 1:
        return float(clean[0]), float(clean[0])
    rng = np.random.default_rng(seed_for(label))
    samples = rng.choice(clean, size=(BOOTSTRAP_REPETITIONS, clean.size), replace=True)
    if statistic is None:
        estimates = samples.mean(axis=1)
    else:
        estimates = np.apply_along_axis(statistic, 1, samples)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def tendency_score(values: Sequence[float]) -> float | None:
    clean = np.asarray([float(value) for value in values if finite(value)], dtype=float)
    if clean.size == 0:
        return None
    return float(100 * (np.sum(clean > 0) + 0.5 * np.sum(clean == 0)) / clean.size)


def cluster_bootstrap_ci(values_by_opponent: dict[str, list[float]], label: str) -> tuple[float | None, float | None]:
    opponents = sorted(key for key, values in values_by_opponent.items() if values)
    if not opponents:
        return None, None
    rng = np.random.default_rng(seed_for(label))
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_REPETITIONS):
        selected = rng.choice(opponents, size=len(opponents), replace=True)
        values = [value for opponent in selected for value in values_by_opponent[str(opponent)]]
        estimates.append(float(np.mean(values)))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def load_and_normalize() -> tuple[list[dict], list[dict], dict, list[dict]]:
    mapping_rows = read_csv(MAPPING_PATH)
    if len(mapping_rows) != EXPECTED["matches"]:
        raise RuntimeError(f"identity mapping row count changed: {len(mapping_rows)}")
    video_ids = [row.get("video_id", "").strip() for row in mapping_rows]
    if len(video_ids) != len(set(video_ids)) or any(not value for value in video_ids):
        raise RuntimeError("video_id must be present and unique")

    manifest_entries = []
    for path, role in [(MAPPING_PATH, "identity_mapping"), (PROTOCOL_PATH, "locked_protocol"), (Path(__file__), "analysis_script")]:
        if not path.exists():
            raise FileNotFoundError(path)
        manifest_entries.append({"role": role, "path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})

    events: list[dict] = []
    exclusions = Counter()
    duplicate_ids: dict[str, str] = {}
    quality = Counter()
    per_match = []
    for meta in sorted(mapping_rows, key=lambda row: row["video_id"]):
        video_id = meta["video_id"].strip()
        shi_side = meta.get("shi_side", "").strip().lower()
        if shi_side not in {"upper", "lower"}:
            raise RuntimeError(f"invalid shi_side for {video_id}: {shi_side}")
        source = BATCH / video_id / "dataset" / "dataset_rows.csv"
        if not source.exists() or source.stat().st_size == 0:
            raise FileNotFoundError(source)
        manifest_entries.append({"role": "match_events", "video_id": video_id, "path": str(source), "bytes": source.stat().st_size, "sha256": sha256_file(source)})
        raw_rows = read_csv(source)
        match_valid = 0
        labels = Counter()
        for row_number, raw in enumerate(raw_rows, start=2):
            player_side = (raw.get("player") or "").strip().lower()
            if player_side not in {"upper", "lower"}:
                exclusions["non_upper_lower_player"] += 1
                continue
            event_id = (raw.get("event_id") or "").strip()
            if not event_id:
                quality["missing_event_id"] += 1
                event_id = f"{video_id}:row:{row_number}"
            canonical = json.dumps(raw, sort_keys=True, ensure_ascii=False)
            global_id = f"{video_id}:{event_id}"
            if global_id in duplicate_ids:
                if duplicate_ids[global_id] != canonical:
                    raise RuntimeError(f"inconsistent duplicate event_id: {global_id}")
                exclusions["identical_duplicate_event"] += 1
                continue
            duplicate_ids[global_id] = canonical

            own_x, own_y = as_float(raw.get("player_location_x")), as_float(raw.get("player_location_y"))
            opponent_x, opponent_y = as_float(raw.get("opponent_location_x")), as_float(raw.get("opponent_location_y"))
            landing_x, landing_y = as_float(raw.get("landing_x")), as_float(raw.get("landing_y"))
            if own_x is not None and not valid_x(own_x):
                quality["invalid_own_x"] += 1
                own_x = None
            if own_y is not None and not valid_y(own_y):
                quality["invalid_own_y"] += 1
                own_y = None
            if opponent_x is not None and not valid_x(opponent_x):
                quality["invalid_opponent_x"] += 1
                opponent_x = None
            if opponent_y is not None and not valid_y(opponent_y):
                quality["invalid_opponent_y"] += 1
                opponent_y = None
            if landing_x is not None and not valid_x(landing_x):
                quality["invalid_landing_x"] += 1
                landing_x = None
            if landing_y is not None and not valid_y(landing_y):
                quality["invalid_landing_y"] += 1
                landing_y = None

            stroke_type = (raw.get("stroke_type") or UNKNOWN).strip().lower() or UNKNOWN
            macro_type = RAW_TO_MACRO.get(stroke_type, UNKNOWN)
            rally_id = (raw.get("rally_id") or "").strip()
            if not rally_id:
                quality["missing_rally_id"] += 1
            stroke_index = as_int(raw.get("stroke_index"))
            hit_time = as_float(raw.get("hit_time"))
            if stroke_index is None:
                quality["missing_stroke_index"] += 1
            if hit_time is None:
                quality["missing_hit_time"] += 1
            opponent_side = "lower" if player_side == "upper" else "upper"
            athlete = "石宇奇" if player_side == shi_side else "对手"
            labels[athlete] += 1
            match_valid += 1
            events.append(
                {
                    "video_id": video_id,
                    "opponent": meta.get("opponent", "").strip(),
                    "competition": meta.get("competition", "").strip(),
                    "shi_side": shi_side,
                    "player_side": player_side,
                    "athlete": athlete,
                    "event_id": event_id,
                    "rally_id": rally_id,
                    "rally_key": f"{video_id}::{rally_id}" if rally_id else "",
                    "stroke_index": stroke_index,
                    "hit_time": hit_time,
                    "stroke_type": stroke_type,
                    "macro_type": macro_type,
                    "own_x": own_x,
                    "own_y": own_y,
                    "opponent_x": opponent_x,
                    "opponent_y": opponent_y,
                    "landing_x": landing_x,
                    "landing_y": landing_y,
                    "own_depth": depth_zone(own_y, player_side),
                    "opponent_depth": depth_zone(opponent_y, opponent_side),
                    "own_width": width_zone(own_x),
                    "opponent_width": width_zone(opponent_x),
                    "landing_zone9": (
                        f"{depth_zone(landing_y, opponent_side)}-{width_zone(landing_x)}"
                        if depth_zone(landing_y, opponent_side) and width_zone(landing_x)
                        else None
                    ),
                    "line_class": line_class(own_x, landing_x, 6),
                }
            )
        if not labels["石宇奇"] or not labels["对手"]:
            raise RuntimeError(f"match lacks paired player events: {video_id}")
        per_match.append({"video_id": video_id, "raw_rows": len(raw_rows), "valid_side_events": match_valid, "shi_events": labels["石宇奇"], "opponent_events": labels["对手"]})

    audit = {
        "version": VERSION,
        "mapping_rows": len(mapping_rows),
        "unique_video_ids": len(set(video_ids)),
        "unique_opponents": len(set(row.get("opponent", "") for row in mapping_rows)),
        "raw_rows": sum(row["raw_rows"] for row in per_match),
        "valid_side_events": len(events),
        "excluded": dict(exclusions),
        "quality": dict(quality),
        "per_match": per_match,
    }
    manifest = {"version": VERSION, "protocol": PROTOCOL_VERSION, "entries": manifest_entries}
    return mapping_rows, events, audit, manifest_entries


def process_metrics_for_rally(ordered_player: list[dict], wide_attack: bool = False) -> dict[str, int]:
    def is_attack(row: dict) -> bool:
        return row["macro_type"] == "后场进攻" or (wide_attack and row["stroke_type"] == "rush")

    result = {
        "attack_entry_eligible": 0,
        "attack_entry_success": 0,
        "defense_eligible": 0,
        "defense_success": 0,
        "net_opportunity": 0,
        "net_success": 0,
    }
    attack_eligible = False
    attack_success = False
    defense_eligible = False
    defense_success = False
    for index, anchor in enumerate(ordered_player):
        window = ordered_player[index + 1 : index + 3]
        has_classified = any(row["macro_type"] != UNKNOWN for row in window)
        if anchor["macro_type"] in NEUTRAL_DEFENSE_MACROS and not is_attack(anchor) and has_classified:
            attack_eligible = True
            attack_success = attack_success or any(is_attack(row) for row in window)
        if anchor["macro_type"] in DEFENSE_MACROS and has_classified:
            defense_eligible = True
            defense_success = defense_success or any(is_attack(row) for row in window)
        if anchor["macro_type"] == "网前/前场" and has_classified:
            result["net_opportunity"] += 1
            result["net_success"] += int(any(row["macro_type"] == "网前/前场" for row in window))
    result["attack_entry_eligible"] = int(attack_eligible)
    result["attack_entry_success"] = int(attack_eligible and attack_success)
    result["defense_eligible"] = int(defense_eligible)
    result["defense_success"] = int(defense_eligible and defense_success)
    return result


def build_player_metrics(events: list[dict], athlete: str, wide_attack: bool = False) -> dict:
    player_events = [row for row in events if row["athlete"] == athlete]
    valid_rally_events = [row for row in events if row["rally_key"]]
    valid_player_rally_events = [row for row in player_events if row["rally_key"]]
    all_by_rally: dict[str, list[dict]] = defaultdict(list)
    player_by_rally: dict[str, list[dict]] = defaultdict(list)
    for row in valid_rally_events:
        all_by_rally[row["rally_key"]].append(row)
    for row in valid_player_rally_events:
        player_by_rally[row["rally_key"]].append(row)

    ordered_player_rallies: list[list[dict]] = []
    event_rates: list[float] = []
    long_counts = {8: 0, 10: 0, 12: 0}
    process = Counter()
    terminal_denominator = 0
    terminal_success = 0
    invalid_order_rallies = 0
    nonpositive_duration_rallies = 0
    for rally_key, player_rows in player_by_rally.items():
        ordered_player = ordered_rally(player_rows)
        ordered_all = ordered_rally(all_by_rally[rally_key])
        if ordered_player is None or ordered_all is None:
            invalid_order_rallies += 1
            continue
        ordered_player_rallies.append(ordered_player)
        process.update(process_metrics_for_rally(ordered_player, wide_attack=wide_attack))

        times = [row["hit_time"] for row in ordered_all if row["hit_time"] is not None]
        if len(times) >= 2:
            duration = max(times) - min(times)
            if duration > 0:
                event_rates.append(len(player_rows) / duration)
            else:
                nonpositive_duration_rallies += 1
        for threshold in long_counts:
            long_counts[threshold] += int(len(ordered_all) >= threshold)

        last = ordered_all[-1] if ordered_all else None
        if last is not None and last["athlete"] in {"石宇奇", "对手"} and last["macro_type"] != UNKNOWN:
            terminal_denominator += 1
            is_terminal_attack = last["macro_type"] == "后场进攻" or (wide_attack and last["stroke_type"] == "rush")
            terminal_success += int(last["athlete"] == athlete and is_terminal_attack)

    classified = [row for row in player_events if row["macro_type"] in MACRO_ORDER]
    macro_counts = Counter(row["macro_type"] for row in classified)
    raw_counts = Counter(row["stroke_type"] for row in classified)
    landing_counts = Counter(row["landing_zone9"] for row in player_events if row["landing_zone9"])
    line_counts = Counter(row["line_class"] for row in player_events if row["line_class"])
    valid_positions = sum(row["own_depth"] is not None for row in player_events)
    valid_landings = sum(row["landing_zone9"] is not None for row in player_events)
    valid_lines = sum(row["line_class"] is not None for row in player_events)
    early = [row for row in classified if row["stroke_index"] is not None and row["stroke_index"] <= 3]
    landing_entropy = normalized_entropy(landing_counts, ZONE9_ORDER)
    line_entropy = normalized_entropy(line_counts, LINE_ORDER)
    spatial = mean([landing_entropy, line_entropy]) if landing_entropy is not None and line_entropy is not None else None
    sequence, sequence_pairs = transition_predictability(ordered_player_rallies)
    rally_count = len(player_by_rally)

    metrics = {
        "smash_share": ratio(sum(raw_counts[name] for name in SMASH_TYPES), len(classified)),
        "drop_share": ratio(sum(raw_counts[name] for name in DROP_TYPES), len(classified)),
        "net_share": ratio(macro_counts["网前/前场"], len(classified)),
        "drive_share": ratio(macro_counts["平抽/推挡"], len(classified)),
        "backcourt_tendency": ratio(sum(row["own_depth"] == "后场" for row in player_events), valid_positions),
        "event_rate_proxy": median(event_rates),
        "spatial_diversity": spatial,
        "sequence_predictability": sequence,
        "early_active_share": ratio(sum(row["macro_type"] in ACTIVE_MACROS for row in early), len(early)),
        "attack_entry_proxy": ratio(process["attack_entry_success"], process["attack_entry_eligible"]),
        "terminal_attack_event_share": ratio(terminal_success, terminal_denominator),
        "net_continuation_proxy": ratio(process["net_success"], process["net_opportunity"]),
        "defense_to_attack_proxy": ratio(process["defense_success"], process["defense_eligible"]),
        "long_rally_share_10": ratio(long_counts[10], rally_count),
        "long_rally_share_8": ratio(long_counts[8], rally_count),
        "long_rally_share_12": ratio(long_counts[12], rally_count),
        "landing_entropy": landing_entropy,
        "line_entropy": line_entropy,
        "landing_coverage": ratio(len(landing_counts), len(ZONE9_ORDER)),
        "classified_rate": ratio(len(classified), len(player_events)),
    }
    metric_ns = {
        "smash_share": len(classified),
        "drop_share": len(classified),
        "net_share": len(classified),
        "drive_share": len(classified),
        "backcourt_tendency": valid_positions,
        "event_rate_proxy": len(event_rates),
        "spatial_diversity": min(valid_landings, valid_lines),
        "sequence_predictability": sequence_pairs,
        "early_active_share": len(early),
        "attack_entry_proxy": process["attack_entry_eligible"],
        "terminal_attack_event_share": terminal_denominator,
        "net_continuation_proxy": process["net_opportunity"],
        "defense_to_attack_proxy": process["defense_eligible"],
        "long_rally_share_10": rally_count,
        "long_rally_share_8": rally_count,
        "long_rally_share_12": rally_count,
        "landing_entropy": valid_landings,
        "line_entropy": valid_lines,
        "landing_coverage": valid_landings,
        "classified_rate": len(player_events),
    }
    record = {
        "athlete": athlete,
        "rally_count": rally_count,
        "stroke_events": len(player_events),
        "classified_strokes": len(classified),
        "unknown_strokes": len(player_events) - len(classified),
        "valid_own_position": valid_positions,
        "valid_landing": valid_landings,
        "valid_line": valid_lines,
        "invalid_order_rallies": invalid_order_rallies,
        "nonpositive_duration_rallies": nonpositive_duration_rallies,
        "attack_entry_successes": process["attack_entry_success"],
        "defense_to_attack_successes": process["defense_success"],
        "net_continuation_successes": process["net_success"],
        "terminal_attack_events": terminal_success,
        **{f"macro_{index + 1}_{category}": macro_counts[category] for index, category in enumerate(MACRO_ORDER)},
    }
    for metric in ALL_METRICS:
        record[metric] = metrics[metric]
        record[f"{metric}_n"] = metric_ns[metric]
        threshold = LOW_N_THRESHOLDS.get(metric)
        record[f"{metric}_low_n"] = bool(threshold and metric_ns[metric] < threshold)
    return record


def build_match_metrics(mapping_rows: list[dict], events: list[dict]) -> list[dict]:
    by_video: dict[str, list[dict]] = defaultdict(list)
    for row in events:
        by_video[row["video_id"]].append(row)
    output = []
    for meta in sorted(mapping_rows, key=lambda row: row["video_id"]):
        match_events = by_video[meta["video_id"]]
        for athlete in ["石宇奇", "对手"]:
            record = build_player_metrics(match_events, athlete)
            record = {
                "video_id": meta["video_id"],
                "opponent": meta.get("opponent", ""),
                "competition": meta.get("competition", ""),
                "athlete": athlete,
                "side": meta["shi_side"] if athlete == "石宇奇" else ("lower" if meta["shi_side"] == "upper" else "upper"),
                **{key: value for key, value in record.items() if key != "athlete"},
            }
            output.append(record)
    return output


def summarize_metrics(match_metrics: list[dict]) -> list[dict]:
    rows = []
    for metric in ALL_METRICS:
        for athlete in ["石宇奇", "对手"]:
            selected = [row for row in match_metrics if row["athlete"] == athlete and finite(row.get(metric))]
            values = np.asarray([float(row[metric]) for row in selected], dtype=float)
            low, high = bootstrap_ci(values, f"summary:{metric}:{athlete}")
            rows.append(
                {
                    "family": METRIC_FAMILY[metric],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "athlete": athlete,
                    "mean": float(values.mean()) if values.size else None,
                    "median": float(np.median(values)) if values.size else None,
                    "sd": float(values.std(ddof=1)) if values.size > 1 else None,
                    "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)) if values.size else None,
                    "min": float(values.min()) if values.size else None,
                    "max": float(values.max()) if values.size else None,
                    "ci_low": low,
                    "ci_high": high,
                    "n_valid_matches": int(values.size),
                    "n_observations": int(sum(int(row.get(f"{metric}_n") or 0) for row in selected)),
                    "low_n_matches": int(sum(bool(row.get(f"{metric}_low_n")) for row in selected)),
                    "main_eligible": bool(values.size >= 18),
                }
            )
    return rows


def paired_rows(match_metrics: list[dict]) -> tuple[list[dict], dict[str, list[float]]]:
    by_video_athlete = {(row["video_id"], row["athlete"]): row for row in match_metrics}
    videos = sorted({row["video_id"] for row in match_metrics})
    output = []
    deltas_by_metric: dict[str, list[float]] = {}
    for metric in STYLE_METRICS + PROCESS_METRICS:
        deltas = []
        opponent_groups: dict[str, list[float]] = defaultdict(list)
        for video_id in videos:
            shi = by_video_athlete[(video_id, "石宇奇")]
            opponent = by_video_athlete[(video_id, "对手")]
            if finite(shi.get(metric)) and finite(opponent.get(metric)):
                delta = float(shi[metric]) - float(opponent[metric])
                deltas.append(delta)
                opponent_groups[str(shi["opponent"])].append(delta)
        deltas_by_metric[metric] = deltas
        low, high = bootstrap_ci(deltas, f"paired:{metric}")
        score_low, score_high = bootstrap_ci(deltas, f"score:{metric}", tendency_score)
        cluster_low, cluster_high = cluster_bootstrap_ci(opponent_groups, f"cluster:{metric}")
        output.append(
            {
                "family": METRIC_FAMILY[metric],
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "paired_delta_mean": mean(deltas),
                "paired_delta_median": median(deltas),
                "paired_delta_ci_low": low,
                "paired_delta_ci_high": high,
                "tendency_score": tendency_score(deltas),
                "tendency_score_ci_low": score_low,
                "tendency_score_ci_high": score_high,
                "n_positive": sum(value > 0 for value in deltas),
                "n_equal": sum(value == 0 for value in deltas),
                "n_negative": sum(value < 0 for value in deltas),
                "n_valid_matches": len(deltas),
                "cluster_ci_low": cluster_low,
                "cluster_ci_high": cluster_high,
                "main_eligible": len(deltas) >= 18,
            }
        )
    return output, deltas_by_metric


def paired_delta_for_records(records: list[dict], metric: str, allowed_videos: set[str] | None = None) -> float | None:
    indexed = {(row["video_id"], row["athlete"]): row for row in records}
    deltas = []
    for video_id in sorted({row["video_id"] for row in records}):
        if allowed_videos is not None and video_id not in allowed_videos:
            continue
        shi = indexed.get((video_id, "石宇奇"))
        opponent = indexed.get((video_id, "对手"))
        if shi and opponent and finite(shi.get(metric)) and finite(opponent.get(metric)):
            deltas.append(float(shi[metric]) - float(opponent[metric]))
    return mean(deltas)


def metric_variant_records(mapping_rows: list[dict], events: list[dict], variant: str) -> list[dict]:
    base_records = build_match_metrics(mapping_rows, events)
    if variant == "wide_attack":
        by_video: dict[str, list[dict]] = defaultdict(list)
        for row in events:
            by_video[row["video_id"]].append(row)
        output = []
        for meta in sorted(mapping_rows, key=lambda row: row["video_id"]):
            for athlete in ["石宇奇", "对手"]:
                values = build_player_metrics(by_video[meta["video_id"]], athlete, wide_attack=True)
                output.append({"video_id": meta["video_id"], "opponent": meta["opponent"], "athlete": athlete, **values})
        return output
    if variant.startswith("line_divisor_"):
        divisor = int(variant.rsplit("_", 1)[1])
        output = []
        for record in base_records:
            selected = [row for row in events if row["video_id"] == record["video_id"] and row["athlete"] == record["athlete"]]
            counts = Counter(line_class(row["own_x"], row["landing_x"], divisor) for row in selected)
            counts.pop(None, None)
            line_value = normalized_entropy(counts, LINE_ORDER)
            spatial = mean([record["landing_entropy"], line_value]) if finite(record["landing_entropy"]) and line_value is not None else None
            output.append({**record, "line_entropy": line_value, "spatial_diversity": spatial})
        return output
    if variant == "unknown_seventh":
        output = []
        by_video: dict[str, list[dict]] = defaultdict(list)
        for row in events:
            by_video[row["video_id"]].append(row)
        for record in base_records:
            selected = [row for row in by_video[record["video_id"]] if row["athlete"] == record["athlete"]]
            all_count = len(selected)
            ordered_groups = []
            by_rally: dict[str, list[dict]] = defaultdict(list)
            for row in selected:
                if row["rally_key"]:
                    by_rally[row["rally_key"]].append(row)
            for rows in by_rally.values():
                ordered = ordered_rally(rows)
                if ordered is not None:
                    ordered_groups.append(ordered)
            sequence, _ = transition_predictability(ordered_groups, MACRO_ORDER + [UNKNOWN], include_unknown=True)
            output.append(
                {
                    **record,
                    "smash_share": ratio(sum(row["stroke_type"] in SMASH_TYPES for row in selected), all_count),
                    "drop_share": ratio(sum(row["stroke_type"] in DROP_TYPES for row in selected), all_count),
                    "net_share": ratio(sum(row["macro_type"] == "网前/前场" for row in selected), all_count),
                    "drive_share": ratio(sum(row["macro_type"] == "平抽/推挡" for row in selected), all_count),
                    "sequence_predictability": sequence,
                }
            )
        return output
    if variant == "raw_sequence":
        output = []
        for record in base_records:
            selected = [row for row in events if row["video_id"] == record["video_id"] and row["athlete"] == record["athlete"]]
            by_rally: dict[str, list[dict]] = defaultdict(list)
            for row in selected:
                if row["rally_key"]:
                    by_rally[row["rally_key"]].append(row)
            ordered_groups = [ordered for rows in by_rally.values() if (ordered := ordered_rally(rows)) is not None]
            sequence, _ = transition_predictability(ordered_groups, RAW_CATEGORIES, use_raw=True)
            output.append({**record, "sequence_predictability": sequence})
        return output
    raise ValueError(variant)


def lomo_next_class_records(events: list[dict]) -> list[dict]:
    pairs_by_match: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    grouped: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in events:
        if row["rally_key"]:
            grouped[(row["video_id"], row["athlete"], row["rally_key"])].append(row)
    for (video_id, athlete, _), rows in grouped.items():
        ordered = ordered_rally(rows)
        if ordered is None:
            continue
        prior: str | None = None
        for row in ordered:
            current = row["macro_type"]
            if current == UNKNOWN:
                prior = None
                continue
            if prior is not None:
                pairs_by_match[(video_id, athlete)].append((prior, current))
            prior = current

    videos = sorted({row["video_id"] for row in events})
    opponents = {row["video_id"]: row["opponent"] for row in events}
    output = []
    for athlete in ["石宇奇", "对手"]:
        for held_video in videos:
            train = Counter(
                pair
                for video_id in videos
                if video_id != held_video
                for pair in pairs_by_match.get((video_id, athlete), [])
            )
            prediction: dict[str, str] = {}
            for source in MACRO_ORDER:
                candidates = [(train[(source, target)], target) for target in MACRO_ORDER]
                best_count = max(count for count, _ in candidates)
                if best_count > 0:
                    prediction[source] = sorted(target for count, target in candidates if count == best_count)[0]
            test_pairs = [pair for pair in pairs_by_match.get((held_video, athlete), []) if pair[0] in prediction]
            correct = sum(prediction[source] == target for source, target in test_pairs)
            output.append(
                {
                    "video_id": held_video,
                    "opponent": opponents[held_video],
                    "athlete": athlete,
                    "sequence_predictability": ratio(correct, len(test_pairs)),
                    "sequence_predictability_n": len(test_pairs),
                }
            )
    return output


def sensitivity_analysis(mapping_rows: list[dict], events: list[dict], match_metrics: list[dict], paired: list[dict]) -> tuple[list[dict], dict[str, str]]:
    base_deltas = {row["metric"]: row["paired_delta_mean"] for row in paired}
    rows: list[dict] = []
    failures: dict[str, list[str]] = defaultdict(list)

    def add(test: str, metric: str, variant: str, value: float | None, threshold_ok: bool, detail: str) -> None:
        direction_ok = sign(value) == sign(base_deltas.get(metric))
        passed = bool(threshold_ok and direction_ok)
        rows.append({"test": test, "metric": metric, "metric_label": METRIC_LABELS.get(metric, metric), "variant": variant, "base_paired_delta": base_deltas.get(metric), "variant_paired_delta": value, "direction_consistent": direction_ok, "threshold_ok": threshold_ok, "status": "PASS" if passed else "FAIL", "detail": detail})
        if not passed:
            failures[metric].append(f"{test}:{variant}")

    pooled_records = []
    for athlete in ["石宇奇", "对手"]:
        pooled_records.append({"athlete": athlete, **build_player_metrics(events, athlete)})
    for metric in STYLE_METRICS + PROCESS_METRICS:
        shi = next(row for row in pooled_records if row["athlete"] == "石宇奇")[metric]
        opp = next(row for row in pooled_records if row["athlete"] == "对手")[metric]
        variant_delta = float(shi) - float(opp) if finite(shi) and finite(opp) else None
        base_value = base_deltas.get(metric)
        if metric == "event_rate_proxy":
            denominator = max(abs(float(base_value or 0)), abs(float(variant_delta or 0)), 1e-12)
            threshold_ok = base_value is not None and variant_delta is not None and abs(variant_delta - base_value) / denominator <= 0.20
        else:
            threshold_ok = base_value is not None and variant_delta is not None and abs(variant_delta - base_value) <= 0.10
        add("S1", metric, "pooled_events", variant_delta, threshold_ok, "比赛等权与合并事件口径")

    unknown_records = metric_variant_records(mapping_rows, events, "unknown_seventh")
    for metric in ["smash_share", "drop_share", "net_share", "drive_share", "sequence_predictability"]:
        value = paired_delta_for_records(unknown_records, metric)
        threshold_ok = value is not None and base_deltas.get(metric) is not None and abs(value - float(base_deltas[metric])) <= 0.10
        add("S2", metric, "unknown_as_seventh", value, threshold_ok, "unknown保留为第七类")

    base_long = paired_delta_for_records(match_metrics, "long_rally_share_10")
    for metric, threshold in [("long_rally_share_8", 8), ("long_rally_share_12", 12)]:
        value = paired_delta_for_records(match_metrics, metric)
        rows.append({"test": "S3", "metric": "long_rally_share_10", "metric_label": METRIC_LABELS["long_rally_share_10"], "variant": f"threshold_{threshold}", "base_paired_delta": base_long, "variant_paired_delta": value, "direction_consistent": sign(value) == sign(base_long), "threshold_ok": True, "status": "PASS", "detail": "辅助环境指标，不进入能力结论"})

    for divisor in [8, 4]:
        records = metric_variant_records(mapping_rows, events, f"line_divisor_{divisor}")
        value = paired_delta_for_records(records, "spatial_diversity")
        threshold_ok = value is not None and base_deltas.get("spatial_diversity") is not None and abs(value - float(base_deltas["spatial_diversity"])) <= 0.10
        add("S4", "spatial_diversity", f"court_width_div_{divisor}", value, threshold_ok, "横向分箱阈值")

    raw_records = metric_variant_records(mapping_rows, events, "raw_sequence")
    value = paired_delta_for_records(raw_records, "sequence_predictability")
    threshold_ok = value is not None and base_deltas.get("sequence_predictability") is not None and abs(value - float(base_deltas["sequence_predictability"])) <= 0.10
    add("S5", "sequence_predictability", "raw_stroke_sequence", value, threshold_ok, "原始球种序列")
    prediction_records = lomo_next_class_records(events)
    value = paired_delta_for_records(prediction_records, "sequence_predictability")
    threshold_ok = value is not None
    add("S5", "sequence_predictability", "leave_one_match_out_next_class_accuracy", value, threshold_ok, "以其余比赛训练最频繁转移规则并预测留出比赛")

    wide_records = metric_variant_records(mapping_rows, events, "wide_attack")
    for metric in ["attack_entry_proxy", "terminal_attack_event_share", "defense_to_attack_proxy"]:
        value = paired_delta_for_records(wide_records, metric)
        threshold_ok = value is not None and base_deltas.get(metric) is not None and abs(value - float(base_deltas[metric])) <= 0.10
        add("S6", metric, "attack_plus_rush", value, threshold_ok, "宽口径进攻状态")

    valid_videos = set()
    for video_id in sorted({row["video_id"] for row in match_metrics}):
        paired_rows_for_video = [row for row in match_metrics if row["video_id"] == video_id]
        if len(paired_rows_for_video) == 2 and all(row["valid_landing"] >= 50 and row["valid_landing"] / max(row["stroke_events"], 1) >= 0.30 for row in paired_rows_for_video):
            valid_videos.add(video_id)
    value = paired_delta_for_records(match_metrics, "spatial_diversity", valid_videos)
    threshold_ok = value is not None and base_deltas.get("spatial_diversity") is not None and abs(value - float(base_deltas["spatial_diversity"])) <= 0.10
    add("S7", "spatial_diversity", f"complete_case_n={len(valid_videos)}", value, threshold_ok, "双方每场有效率≥30%且有效落点≥50")

    for row in paired:
        metric = row["metric"]
        direction_ok = sign(row["cluster_ci_low"]) == sign(row["cluster_ci_high"]) if row["cluster_ci_low"] is not None and row["cluster_ci_high"] is not None and sign(row["cluster_ci_low"]) != 0 and sign(row["cluster_ci_high"]) != 0 else True
        rows.append({"test": "S8", "metric": metric, "metric_label": row["metric_label"], "variant": "opponent_cluster_bootstrap", "base_paired_delta": row["paired_delta_mean"], "variant_paired_delta": row["paired_delta_mean"], "direction_consistent": True, "threshold_ok": direction_ok, "status": "PASS" if direction_ok else "CAUTION", "detail": f"cluster_CI=[{row['cluster_ci_low']}, {row['cluster_ci_high']}]"})

    videos = sorted({row["video_id"] for row in match_metrics})
    for metric in STYLE_METRICS + PROCESS_METRICS:
        loo_values = [paired_delta_for_records(match_metrics, metric, set(videos) - {video}) for video in videos]
        finite_loo = [float(value) for value in loo_values if value is not None]
        direction_ok = bool(finite_loo) and all(sign(value) == sign(base_deltas.get(metric)) for value in finite_loo)
        range_text = f"range=[{min(finite_loo):.15g}, {max(finite_loo):.15g}]" if finite_loo else "range=NA"
        rows.append({"test": "S9", "metric": metric, "metric_label": METRIC_LABELS[metric], "variant": "leave_one_match_out", "base_paired_delta": base_deltas.get(metric), "variant_paired_delta": mean(finite_loo), "direction_consistent": direction_ok, "threshold_ok": direction_ok, "status": "PASS" if direction_ok else "FAIL", "detail": range_text})
        if not direction_ok:
            failures[metric].append("S9:leave_one_match_out")

    statuses = {metric: ("UNSTABLE" if failures.get(metric) else "STABLE") for metric in STYLE_METRICS + PROCESS_METRICS}
    return rows, statuses


def composition_outputs(match_metrics: list[dict]) -> tuple[list[dict], list[dict]]:
    rows = []
    clr_rows = []
    for athlete in ["石宇奇", "对手"]:
        selected = [row for row in match_metrics if row["athlete"] == athlete]
        totals = np.asarray([sum(int(row[f"macro_{i + 1}_{category}"]) for row in selected) for i, category in enumerate(MACRO_ORDER)], dtype=float)
        overall = totals / totals.sum()
        for row in selected:
            counts = np.asarray([int(row[f"macro_{i + 1}_{category}"]) for i, category in enumerate(MACRO_ORDER)], dtype=float)
            profile = counts / counts.sum()
            midpoint = 0.5 * (profile + overall)
            jsd = 0.5 * np.sum(np.where(profile > 0, profile * np.log(profile / midpoint), 0)) + 0.5 * np.sum(np.where(overall > 0, overall * np.log(overall / midpoint), 0))
            rows.append({"video_id": row["video_id"], "opponent": row["opponent"], "athlete": athlete, "jsd_from_athlete_overall": float(jsd), **{f"share_{category}": profile[i] for i, category in enumerate(MACRO_ORDER)}})
            adjusted = counts + 0.5
            composition = adjusted / adjusted.sum()
            clr = np.log(composition) - np.log(composition).mean()
            clr_rows.append({"video_id": row["video_id"], "opponent": row["opponent"], "athlete": athlete, **{f"clr_{category}": clr[i] for i, category in enumerate(MACRO_ORDER)}})
    return rows, clr_rows


def metric_definitions() -> list[dict]:
    definitions = {
        "smash_share": "(smash+wrist smash)/六类已分类击球",
        "drop_share": "(drop+passive drop)/六类已分类击球",
        "net_share": "网前/前场宏观类别/六类已分类击球",
        "drive_share": "平抽/推挡宏观类别/六类已分类击球",
        "backcourt_tendency": "后场有效击球位置/全部有效自身位置",
        "event_rate_proxy": "逐回合自有事件数/全回合持续时间的比赛级中位数",
        "spatial_diversity": "0.5×九宫格落点熵+0.5×横向位移熵",
        "sequence_predictability": "1-六类自有击球转移归一化条件熵；unknown断开序列",
        "early_active_share": "全回合stroke_index≤3的自有已分类击球中主动类别比例",
        "attack_entry_proxy": "合格中性/防守锚点后1–2次自有原始事件出现后场进攻的回合比例",
        "terminal_attack_event_share": "全回合最后可分类事件属于目标运动员且为后场进攻的回合比例",
        "net_continuation_proxy": "合格网前锚点后1–2次自有原始事件再次网前的机会比例",
        "defense_to_attack_proxy": "合格高远/挑球锚点后1–2次自有原始事件出现后场进攻的回合比例",
        "long_rally_share_10": "双方有效事件数≥10的回合/有效回合",
        "long_rally_share_8": "双方有效事件数≥8的回合/有效回合",
        "long_rally_share_12": "双方有效事件数≥12的回合/有效回合",
        "landing_entropy": "九宫格预测落点归一化Shannon熵",
        "line_entropy": "横向位移三分类归一化Shannon熵",
        "landing_coverage": "使用过的九宫格数量/9",
        "classified_rate": "六类已分类击球/全部自有事件",
    }
    return [{"family": METRIC_FAMILY[metric], "metric": metric, "metric_label": METRIC_LABELS[metric], "definition": definitions[metric], "interpretation": "风格/过程结构，不代表得分效果" if metric in STYLE_METRICS + PROCESS_METRICS else "辅助描述"} for metric in ALL_METRICS]


def add_chinese_font() -> None:
    for path in [Path(r"C:\Windows\Fonts\msyh.ttc"), Path(r"C:\Windows\Fonts\simhei.ttf")]:
        if path.exists():
            font_manager.fontManager.addfont(str(path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(path)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


def radar_plot(rows: list[dict], metrics: list[str], title: str, output: Path, color: str) -> None:
    selected = {row["metric"]: row for row in rows}
    values = [float(selected[metric]["tendency_score"]) if selected[metric]["main_eligible"] else np.nan for metric in metrics]
    labels = [f"{METRIC_LABELS[metric]}{'*' if selected[metric].get('sensitivity_status') == 'UNSTABLE' else ''}\n(n={selected[metric]['n_valid_matches']})" for metric in metrics]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    closed_values = np.r_[values, values[0]]
    fig, axis = plt.subplots(figsize=(9, 9), subplot_kw={"polar": True})
    axis.plot(closed_angles, closed_values, color=color, linewidth=2.2, marker="o", markersize=5)
    unstable_indices = [index for index, metric in enumerate(metrics) if selected[metric].get("sensitivity_status") == "UNSTABLE" and np.isfinite(values[index])]
    if unstable_indices:
        axis.scatter(angles[unstable_indices], np.asarray(values)[unstable_indices], marker="x", s=70, color="#b2182b", linewidths=2, zorder=5)
    axis.plot(np.linspace(0, 2 * np.pi, 361), np.full(361, 50), color="#777777", linewidth=1.2, linestyle="--")
    axis.set_ylim(0, 100)
    axis.set_yticks([0, 25, 50, 75, 100])
    axis.set_xticks(angles)
    axis.set_xticklabels(labels, fontsize=9)
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_title(title, pad=30, fontsize=14)
    axis.grid(alpha=0.35)
    fig.text(0.5, 0.02, "分数=该指标高于同场对手的比赛比例（平局计0.5）；*及红色×表示敏感性分析不稳定；不是绝对能力。", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def interval_plot(rows: list[dict], output: Path) -> None:
    ordered = [next(row for row in rows if row["metric"] == metric) for metric in STYLE_METRICS + PROCESS_METRICS]
    labels = [row["metric_label"] for row in ordered]
    estimates = np.asarray([row["paired_delta_mean"] for row in ordered], dtype=float)
    lows = np.asarray([row["paired_delta_ci_low"] for row in ordered], dtype=float)
    highs = np.asarray([row["paired_delta_ci_high"] for row in ordered], dtype=float)
    tempo_index = (STYLE_METRICS + PROCESS_METRICS).index("event_rate_proxy")
    mask = np.ones(len(ordered), dtype=bool)
    mask[tempo_index] = False
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw={"width_ratios": [4, 1.4]})
    y = np.arange(mask.sum())
    axes[0].errorbar(estimates[mask], y, xerr=[estimates[mask] - lows[mask], highs[mask] - estimates[mask]], fmt="o", color="#2c7fb8", capsize=3)
    axes[0].axvline(0, color="#777777", linestyle="--", linewidth=1)
    axes[0].set_yticks(y, [labels[index] for index in np.where(mask)[0]])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("石宇奇 − 同场对手（比例/熵单位）")
    axes[0].grid(axis="x", alpha=0.25)
    idx = tempo_index
    axes[1].errorbar([estimates[idx]], [0], xerr=[[estimates[idx] - lows[idx]], [highs[idx] - estimates[idx]]], fmt="o", color="#d95f02", capsize=3)
    axes[1].axvline(0, color="#777777", linestyle="--", linewidth=1)
    axes[1].set_yticks([0], [labels[idx]])
    axes[1].set_xlabel("事件/秒")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("同场配对差及比赛级Bootstrap 95%区间")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def variation_plot(match_metrics: list[dict], output: Path) -> None:
    rows = sorted([row for row in match_metrics if row["athlete"] == "石宇奇"], key=lambda row: row["video_id"])
    matrix = np.asarray([[float(row[metric]) for metric in STYLE_METRICS] for row in rows], dtype=float)
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0, ddof=1)
    z = (matrix - means) / np.where(stds > 0, stds, 1)
    fig, axis = plt.subplots(figsize=(12, 9))
    image = axis.imshow(z, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    axis.set_xticks(range(len(STYLE_METRICS)), [METRIC_LABELS[metric] for metric in STYLE_METRICS], rotation=35, ha="right")
    axis.set_yticks(range(len(rows)), [f"{row['opponent']} | {row['competition']}" for row in rows], fontsize=7)
    axis.set_title("石宇奇比赛级打法风格变异（列内z分数）")
    fig.colorbar(image, ax=axis, label="相对个人24场均值的标准差")
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_report(audit: dict, summary_rows: list[dict], paired: list[dict], sensitivity_status: dict[str, str]) -> str:
    shi_summary = {(row["metric"]): row for row in summary_rows if row["athlete"] == "石宇奇"}
    paired_index = {row["metric"]: row for row in paired}
    lines = [
        "# 石宇奇打法风格与技战术过程画像：v2研究结果",
        "",
        "## Material Passport",
        "",
        "- Origin Skill: academic-research-suite / experiment-agent",
        "- Origin Mode: run + validate",
        "- Origin Date: 2026-08-22",
        "- Verification Status: UNVERIFIED — 需完成独立复跑哈希比较后更新",
        f"- Version Label: {VERSION}",
        "",
        "## 1. 数据覆盖",
        "",
        f"- 比赛：{audit['mapping_rows']}；对手：{audit['unique_opponents']}；双方有效事件：{audit['valid_side_events']}。",
        f"- 非upper/lower事件排除：{audit['excluded'].get('non_upper_lower_player', 0)}；重复事件排除：{audit['excluded'].get('identical_duplicate_event', 0)}。",
        "- 推断单位为比赛，所有结果均为当前算法标注数据下的有限样本描述。",
        "",
        "## 2. 打法风格主结果",
        "",
        "| 指标 | 石宇奇比赛均值 | 95% CI | 配对差 | 配对差95% CI | 倾向分数 | 稳定性 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for metric in STYLE_METRICS:
        raw = shi_summary[metric]
        pair = paired_index[metric]
        lines.append(f"| {METRIC_LABELS[metric]} | {raw['mean']:.4f} | [{raw['ci_low']:.4f}, {raw['ci_high']:.4f}] | {pair['paired_delta_mean']:.4f} | [{pair['paired_delta_ci_low']:.4f}, {pair['paired_delta_ci_high']:.4f}] | {pair['tendency_score']:.1f} | {sensitivity_status[metric]} |")
    lines += ["", "## 3. 技战术过程代理", "", "| 指标 | 石宇奇比赛均值 | 配对差 | 配对差95% CI | 倾向分数 | 稳定性 |", "|---|---:|---:|---:|---:|---|"]
    for metric in PROCESS_METRICS:
        raw = shi_summary[metric]
        pair = paired_index[metric]
        lines.append(f"| {METRIC_LABELS[metric]} | {raw['mean']:.4f} | {pair['paired_delta_mean']:.4f} | [{pair['paired_delta_ci_low']:.4f}, {pair['paired_delta_ci_high']:.4f}] | {pair['tendency_score']:.1f} | {sensitivity_status[metric]} |")
    lines += [
        "",
        "## 4. 解释边界",
        "",
        "- 倾向分数表示该指标高于同场对手的比赛比例；高值不表示能力更强。",
        "- 事件频率不是移动、反应或真实击球速度。",
        "- 终局后场进攻事件不是杀球得分，网前连续使用不是网前控制。",
        "- 当前数据不支持失误控制、回合胜率、多拍韧性或因果结论。",
        "- 标记为UNSTABLE的指标不得进入摘要主要发现。",
        "",
        "## 5. 文件",
        "",
        "完整数值见 `metric_summary.csv`、`paired_comparisons.csv`、`sensitivity_results.csv` 和 `analysis_summary.json`。",
        "",
    ]
    return "\n".join(lines)


def validate_counts(match_metrics: list[dict], events: list[dict]) -> dict:
    shi_rows = [row for row in match_metrics if row["athlete"] == "石宇奇"]
    actual = {
        "matches": len({row["video_id"] for row in match_metrics}),
        "match_player_rows": len(match_metrics),
        "shi_events": sum(int(row["stroke_events"]) for row in shi_rows),
        "shi_classified": sum(int(row["classified_strokes"]) for row in shi_rows),
        "shi_unknown": sum(int(row["unknown_strokes"]) for row in shi_rows),
        "shi_rallies": sum(int(row["rally_count"]) for row in shi_rows),
        "shi_valid_position": sum(int(row["valid_own_position"]) for row in shi_rows),
        "shi_valid_landing": sum(int(row["valid_landing"]) for row in shi_rows),
        "valid_side_events": len(events),
    }
    mismatches = {key: {"expected": EXPECTED[key], "actual": actual[key]} for key in EXPECTED if actual[key] != EXPECTED[key]}
    if mismatches:
        raise RuntimeError(f"locked data counts changed: {json.dumps(mismatches, ensure_ascii=False)}")
    return actual


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    add_chinese_font()
    mapping_rows, events, audit, manifest_entries = load_and_normalize()
    match_metrics = build_match_metrics(mapping_rows, events)
    locked_counts = validate_counts(match_metrics, events)
    audit["locked_counts"] = locked_counts

    summary_rows = summarize_metrics(match_metrics)
    paired, _ = paired_rows(match_metrics)
    sensitivity_rows, sensitivity_status = sensitivity_analysis(mapping_rows, events, match_metrics, paired)
    for row in paired:
        row["sensitivity_status"] = sensitivity_status[row["metric"]]
        if sensitivity_status[row["metric"]] == "UNSTABLE" or row["n_valid_matches"] < 18:
            row["radar_eligible"] = False
        else:
            row["radar_eligible"] = True
    composition_rows, clr_rows = composition_outputs(match_metrics)

    normalized_fields = [
        "video_id", "opponent", "competition", "shi_side", "player_side", "athlete", "event_id", "rally_id", "stroke_index", "hit_time", "stroke_type", "macro_type", "own_x", "own_y", "opponent_x", "opponent_y", "landing_x", "landing_y", "own_depth", "opponent_depth", "own_width", "opponent_width", "landing_zone9", "line_class"
    ]
    write_csv(OUT / "normalized_events.csv", [{key: row.get(key) for key in normalized_fields} for row in events])
    write_csv(OUT / "match_metrics.csv", match_metrics)
    write_csv(OUT / "metric_summary.csv", summary_rows)
    write_csv(OUT / "paired_comparisons.csv", paired)
    write_csv(OUT / "sensitivity_results.csv", sensitivity_rows)
    write_csv(OUT / "composition_jsd.csv", composition_rows)
    write_csv(OUT / "composition_clr.csv", clr_rows)
    write_csv(OUT / "metric_definitions.csv", metric_definitions())
    write_json(OUT / "data_audit.json", audit)
    write_json(OUT / "input_manifest.json", {"version": VERSION, "protocol": PROTOCOL_VERSION, "entries": manifest_entries})
    write_json(
        OUT / "protocol_lock.json",
        {
            "version": VERSION,
            "protocol": PROTOCOL_VERSION,
            "seed": SEED,
            "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "style_metrics": STYLE_METRICS,
            "process_metrics": PROCESS_METRICS,
            "macro_mapping": RAW_TO_MACRO,
            "low_n_thresholds": LOW_N_THRESHOLDS,
            "sensitivity_tests": [f"S{index}" for index in range(1, 10)],
            "radar_rule": "paired tendency score; line and points only; fixed order; 50 reference ring",
        },
    )
    analysis_summary = {
        "version": VERSION,
        "verification_status": "UNVERIFIED",
        "data_counts": locked_counts,
        "style_results": [row for row in paired if row["metric"] in STYLE_METRICS],
        "process_proxy_results": [row for row in paired if row["metric"] in PROCESS_METRICS],
        "sensitivity_status": sensitivity_status,
        "unavailable_effect_axes": ["opening_and_return_effect", "attack_build_effect", "attack_finish_effect", "net_control", "rally_resilience", "error_control", "spatial_control"],
    }
    write_json(OUT / "analysis_summary.json", analysis_summary)
    radar_plot(paired, STYLE_METRICS, "石宇奇打法风格：同场相对倾向", OUT / "figure_style_radar_v2.png", "#2c7fb8")
    radar_plot(paired, PROCESS_METRICS, "石宇奇技战术过程代理：同场相对倾向", OUT / "figure_process_proxy_v2.png", "#d95f02")
    interval_plot(paired, OUT / "figure_paired_intervals_v2.png")
    variation_plot(match_metrics, OUT / "figure_match_variation_v2.png")
    (OUT / "RESEARCH_REPORT_v2.md").write_text(build_report(audit, summary_rows, paired, sensitivity_status), encoding="utf-8")
    (OUT / "RUN_LOG.md").write_text(
        "\n".join(
            [
                "# v2 Analysis Run Log",
                "",
                f"- Version: {VERSION}",
                f"- Protocol: {PROTOCOL_VERSION}",
                f"- Command: `{sys.executable} {Path(__file__).resolve()}`",
                f"- Python: {platform.python_version()}",
                f"- NumPy: {np.__version__}",
                f"- Matplotlib: {matplotlib.__version__}",
                f"- OS: {platform.platform()}",
                f"- Seed: {SEED}",
                f"- Bootstrap repetitions: {BOOTSTRAP_REPETITIONS}",
                "- Status: COMPLETED",
                "- Reproducibility: pending independent second run",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "COMPLETED", "output": str(OUT), "counts": locked_counts, "sensitivity_status": sensitivity_status}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
