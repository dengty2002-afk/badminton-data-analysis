"""Gold contract and evaluation for ShuttleSet stroke type and destination.

"Landing" follows ShuttleSet's stroke-destination meaning: the court-plane
destination at the opponent's next contact, or the ground contact for the last
stroke.  Automatic outputs remain predictions until evaluated against this
independently annotated Gold contract.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from badminton_pipeline.evaluation.hit_events import GoldHit, _optimal_matches


COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
STROKE_TYPES = (
    "net shot",
    "return net",
    "smash",
    "wrist smash",
    "lob",
    "defensive return lob",
    "clear",
    "drive",
    "driven flight",
    "back-court drive",
    "drop",
    "passive drop",
    "push",
    "rush",
    "defensive return drive",
    "cross-court net shot",
    "short service",
    "long service",
)
LANDING_KINDS = {"next_contact", "ground_contact"}
VALID_HITTERS = {"upper", "lower", "unknown"}
SEMANTIC_FIELDS = (
    "video_id",
    "rally_id",
    "hit_frame",
    "hitter",
    "stroke_type",
    "landing_x",
    "landing_y",
    "landing_frame",
    "landing_kind",
    "confidence",
    "uncertainty_frames",
    "notes",
)


@dataclass(frozen=True)
class SemanticGold:
    video_id: str
    rally_id: str
    hit_frame: int
    hitter: str
    stroke_type: str
    landing_x: float
    landing_y: float
    landing_frame: int | None
    landing_kind: str
    confidence: str
    uncertainty_frames: int
    notes: str


def _integer(value: object, label: str, *, optional: bool = False) -> int | None:
    text = "" if value is None else str(value).strip()
    if optional and not text:
        return None
    try:
        result = int(text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc
    return result


def _number(value: object, label: str) -> float:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def read_semantic_gold(path: Path) -> list[SemanticGold]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "video_id", "rally_id", "hit_frame", "hitter", "stroke_type",
            "landing_x", "landing_y", "landing_kind",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"semantic Gold CSV is missing columns: {sorted(required)}")
        rows: list[SemanticGold] = []
        for line_number, row in enumerate(reader, start=2):
            prefix = f"semantic Gold line {line_number}"
            video_id = str(row.get("video_id") or "").strip()
            rally_id = str(row.get("rally_id") or "unknown").strip() or "unknown"
            hit_frame = _integer(row.get("hit_frame"), f"{prefix} hit_frame")
            uncertainty = _integer(
                row.get("uncertainty_frames") or 0, f"{prefix} uncertainty_frames"
            )
            landing_frame = _integer(
                row.get("landing_frame"), f"{prefix} landing_frame", optional=True
            )
            hitter = str(row.get("hitter") or "unknown").strip().lower()
            stroke_type = str(row.get("stroke_type") or "").strip().lower()
            landing_kind = str(row.get("landing_kind") or "").strip().lower()
            landing_x = _number(row.get("landing_x"), f"{prefix} landing_x")
            landing_y = _number(row.get("landing_y"), f"{prefix} landing_y")
            if not video_id or hit_frame is None or hit_frame < 0:
                raise ValueError(f"{prefix} has invalid video_id/hit_frame")
            if uncertainty is None or uncertainty < 0:
                raise ValueError(f"{prefix} has negative uncertainty")
            if hitter not in VALID_HITTERS:
                raise ValueError(f"{prefix} has invalid hitter {hitter!r}")
            if stroke_type not in STROKE_TYPES:
                raise ValueError(f"{prefix} has unsupported stroke_type {stroke_type!r}")
            if landing_kind not in LANDING_KINDS:
                raise ValueError(f"{prefix} has invalid landing_kind {landing_kind!r}")
            if not 0 <= landing_x <= COURT_WIDTH_M or not 0 <= landing_y <= COURT_LENGTH_M:
                raise ValueError(f"{prefix} destination is outside the standard singles court plane")
            if landing_frame is not None and landing_frame < hit_frame:
                raise ValueError(f"{prefix} landing_frame precedes hit_frame")
            rows.append(SemanticGold(
                video_id=video_id,
                rally_id=rally_id,
                hit_frame=hit_frame,
                hitter=hitter,
                stroke_type=stroke_type,
                landing_x=landing_x,
                landing_y=landing_y,
                landing_frame=landing_frame,
                landing_kind=landing_kind,
                confidence=str(row.get("confidence") or "unknown").strip() or "unknown",
                uncertainty_frames=uncertainty,
                notes=str(row.get("notes") or "").strip(),
            ))
    if not rows:
        raise ValueError("semantic Gold CSV contains no rows")
    keys = [(row.video_id, row.hit_frame) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("semantic Gold CSV contains duplicate video_id/hit_frame rows")
    return sorted(rows, key=lambda row: (row.video_id, row.hit_frame))


def create_semantic_template(hits_gold_path: Path, output_path: Path) -> dict[str, Any]:
    """Carry existing hit Gold into a semantic annotation template."""
    with hits_gold_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"video_id", "rally_id", "hit_frame", "hitter"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("hit Gold CSV does not contain the required columns")
        rows = list(reader)
    if not rows:
        raise ValueError("hit Gold CSV contains no rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SEMANTIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "video_id": row.get("video_id", ""),
                "rally_id": row.get("rally_id", "unknown"),
                "hit_frame": row.get("hit_frame", ""),
                "hitter": row.get("hitter", "unknown"),
                "stroke_type": "",
                "landing_x": "",
                "landing_y": "",
                "landing_frame": "",
                "landing_kind": "",
                "confidence": row.get("confidence", "unknown"),
                "uncertainty_frames": row.get("uncertainty_frames", "0"),
                "notes": row.get("notes", ""),
            })
    return {
        "status": "created",
        "source": str(hits_gold_path.resolve()),
        "output": str(output_path.resolve()),
        "row_count": len(rows),
        "stroke_types": list(STROKE_TYPES),
        "landing_policy": "next_contact for returned strokes; ground_contact for terminal strokes",
    }


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            raw = list(csv.DictReader(stream))
    else:
        with path.open("r", encoding="utf-8") as stream:
            raw = [json.loads(line) for line in stream if line.strip()]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(raw, start=1):
        frame = row.get("hit_frame", row.get("candidate_frame"))
        try:
            frame = int(frame)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"prediction row {index} has invalid hit frame") from exc
        def optional_float(*keys: str) -> float | None:
            value = next((row.get(key) for key in keys if row.get(key) not in {None, ""}), None)
            return None if value is None else float(value)
        rows.append({
            **row,
            "video_id": str(row.get("video_id") or ""),
            "candidate_frame": frame,
            "predicted_hitter": str(row.get("predicted_hitter") or row.get("player") or "unknown"),
            "predicted_stroke_type": str(
                row.get("predicted_stroke_type") or row.get("stroke_type") or "unknown"
            ).strip().lower(),
            "predicted_landing_x": optional_float("predicted_landing_x", "landing_x"),
            "predicted_landing_y": optional_float("predicted_landing_y", "landing_y"),
        })
    if not rows:
        raise ValueError("prediction file contains no rows")
    return sorted(rows, key=lambda row: (row["video_id"], row["candidate_frame"]))


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[round((len(ordered) - 1) * fraction)], 6)


def evaluate_semantics(
    predictions: list[dict[str, Any]],
    gold: list[SemanticGold],
    *,
    tolerance_frames: int = 5,
) -> dict[str, Any]:
    timing_gold = [
        GoldHit(
            video_id=row.video_id,
            hit_frame=row.hit_frame,
            hitter=row.hitter,
            rally_id=row.rally_id,
            confidence=row.confidence,
            uncertainty_frames=row.uncertainty_frames,
            notes=row.notes,
        )
        for row in gold
    ]
    matches = _optimal_matches(predictions, timing_gold, tolerance_frames)
    type_pairs: list[tuple[str, str]] = []
    landing_errors: list[float] = []
    landing_scored = 0
    joint_05 = 0
    joint_10 = 0
    matched_rows: list[dict[str, Any]] = []
    for prediction_index, gold_index in matches:
        prediction, truth = predictions[prediction_index], gold[gold_index]
        predicted_type = str(prediction.get("predicted_stroke_type") or "unknown")
        type_pairs.append((truth.stroke_type, predicted_type))
        x = prediction.get("predicted_landing_x")
        y = prediction.get("predicted_landing_y")
        error = None
        if x is not None and y is not None:
            error = math.hypot(float(x) - truth.landing_x, float(y) - truth.landing_y)
            landing_errors.append(error)
            landing_scored += 1
            type_correct = predicted_type == truth.stroke_type
            joint_05 += type_correct and error <= 0.5
            joint_10 += type_correct and error <= 1.0
        matched_rows.append({
            "video_id": truth.video_id,
            "gold_frame": truth.hit_frame,
            "predicted_frame": prediction["candidate_frame"],
            "gold_stroke_type": truth.stroke_type,
            "predicted_stroke_type": predicted_type,
            "stroke_type_correct": predicted_type == truth.stroke_type,
            "gold_landing_x": truth.landing_x,
            "gold_landing_y": truth.landing_y,
            "predicted_landing_x": x,
            "predicted_landing_y": y,
            "landing_error_m": None if error is None else round(error, 6),
        })

    labels = sorted({truth for truth, _prediction in type_pairs})
    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = sum(truth == label and prediction == label for truth, prediction in type_pairs)
        fp = sum(truth != label and prediction == label for truth, prediction in type_pairs)
        fn = sum(truth == label and prediction != label for truth, prediction in type_pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[label] = {
            "support": sum(truth == label for truth, _prediction in type_pairs),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        }
    correct_types = sum(truth == prediction for truth, prediction in type_pairs)
    matched_count = len(matches)
    return {
        "schema_version": "stroke-semantics-evaluation-1.0",
        "scope": "stroke type and ShuttleSet destination against independent human Gold",
        "tolerance_frames": tolerance_frames,
        "prediction_count": len(predictions),
        "gold_count": len(gold),
        "matched_count": matched_count,
        "type": {
            "coverage": 0.0 if not matched_count else round(
                sum(prediction != "unknown" for _truth, prediction in type_pairs) / matched_count, 6
            ),
            "accuracy": None if not matched_count else round(correct_types / matched_count, 6),
            "macro_f1": None if not f1_values else round(sum(f1_values) / len(f1_values), 6),
            "gold_distribution": dict(Counter(truth for truth, _prediction in type_pairs)),
            "per_class": per_class,
        },
        "landing": {
            "coverage": 0.0 if not matched_count else round(landing_scored / matched_count, 6),
            "mean_error_m": None if not landing_errors else round(sum(landing_errors) / len(landing_errors), 6),
            "median_error_m": _percentile(landing_errors, 0.5),
            "p90_error_m": _percentile(landing_errors, 0.9),
            "within_0_5m": None if not landing_errors else round(sum(v <= 0.5 for v in landing_errors) / len(landing_errors), 6),
            "within_1_0m": None if not landing_errors else round(sum(v <= 1.0 for v in landing_errors) / len(landing_errors), 6),
        },
        "joint": {
            "type_and_landing_within_0_5m": None if not matched_count else round(joint_05 / matched_count, 6),
            "type_and_landing_within_1_0m": None if not matched_count else round(joint_10 / matched_count, 6),
        },
        "matches": matched_rows,
    }


def evaluate_files(
    predictions_path: Path,
    gold_path: Path,
    output_dir: Path,
    *,
    tolerance_frames: int = 5,
) -> dict[str, Any]:
    result = evaluate_semantics(
        _read_predictions(predictions_path),
        read_semantic_gold(gold_path),
        tolerance_frames=tolerance_frames,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {key: value for key, value in result.items() if key != "matches"}
    (output_dir / "semantic_evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "semantic_matches.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fieldnames = list(result["matches"][0]) if result["matches"] else [
            "video_id", "gold_frame", "predicted_frame", "gold_stroke_type",
            "predicted_stroke_type", "stroke_type_correct", "gold_landing_x",
            "gold_landing_y", "predicted_landing_x", "predicted_landing_y", "landing_error_m",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["matches"])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    template = subparsers.add_parser("template", help="Create semantic Gold template from hit Gold")
    template.add_argument("--hits-gold", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    evaluate = subparsers.add_parser("evaluate", help="Evaluate automatic semantic predictions")
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--gold", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--tolerance-frames", type=int, default=5)
    args = parser.parse_args()
    try:
        if args.command == "template":
            result = create_semantic_template(args.hits_gold, args.output)
        else:
            result = evaluate_files(
                args.predictions, args.gold, args.output_dir,
                tolerance_frames=args.tolerance_frames,
            )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
