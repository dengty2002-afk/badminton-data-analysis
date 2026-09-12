"""Evaluate machine hit candidates against independently marked Gold frames."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from badminton_pipeline.tracking.postprocess import _video_ids_match


VALID_HITTERS = {"upper", "lower", "unknown"}


@dataclass(frozen=True)
class GoldHit:
    video_id: str
    hit_frame: int
    hitter: str
    rally_id: str
    confidence: str
    uncertainty_frames: int
    notes: str


def read_gold(path: Path) -> list[GoldHit]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"video_id", "hit_frame", "hitter"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Gold CSV must contain columns: {', '.join(sorted(required))}")
        rows: list[GoldHit] = []
        for line_number, row in enumerate(reader, start=2):
            try:
                frame = int(row["hit_frame"])
                uncertainty = int(row.get("uncertainty_frames") or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid frame value on Gold CSV line {line_number}") from exc
            hitter = (row.get("hitter") or "unknown").strip().lower()
            if frame < 0 or uncertainty < 0:
                raise ValueError(f"negative frame value on Gold CSV line {line_number}")
            if hitter not in VALID_HITTERS:
                raise ValueError(f"invalid hitter {hitter!r} on Gold CSV line {line_number}")
            rows.append(GoldHit(
                video_id=(row.get("video_id") or "").strip(),
                hit_frame=frame,
                hitter=hitter,
                rally_id=(row.get("rally_id") or "unknown").strip() or "unknown",
                confidence=(row.get("confidence") or "unknown").strip() or "unknown",
                uncertainty_frames=uncertainty,
                notes=(row.get("notes") or "").strip(),
            ))
    if not rows:
        raise ValueError("Gold CSV contains no annotated hits")
    if any(not row.video_id for row in rows):
        raise ValueError("Gold CSV contains an empty video_id")
    if len({(row.video_id, row.hit_frame) for row in rows}) != len(rows):
        raise ValueError("Gold CSV contains duplicate video_id/hit_frame rows")
    return sorted(rows, key=lambda row: (row.video_id, row.hit_frame))


def read_candidates(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError("candidate events JSONL is empty")
    return sorted(rows, key=lambda row: (str(row.get("video_id", "")), int(row["candidate_frame"])))


def _optimal_matches(
    candidates: list[dict[str, Any]],
    gold: list[GoldHit],
    tolerance: int,
    reaction_allowance_frames: int = 0,
) -> list[tuple[int, int]]:
    """Maximize one-to-one matches, then minimize total absolute frame error."""
    if tolerance < 0:
        raise ValueError("tolerance cannot be negative")
    if reaction_allowance_frames < 0:
        raise ValueError("reaction allowance cannot be negative")
    rows, columns = len(candidates), len(gold)
    scores = [[(0, 0) for _ in range(columns + 1)] for _ in range(rows + 1)]
    actions = [["" for _ in range(columns + 1)] for _ in range(rows + 1)]
    for index in range(1, rows + 1):
        actions[index][0] = "skip_candidate"
    for index in range(1, columns + 1):
        actions[0][index] = "skip_gold"
    priority = {"match": 2, "skip_candidate": 1, "skip_gold": 0}
    for candidate_index in range(1, rows + 1):
        for gold_index in range(1, columns + 1):
            options = [
                (scores[candidate_index - 1][gold_index], "skip_candidate"),
                (scores[candidate_index][gold_index - 1], "skip_gold"),
            ]
            candidate = candidates[candidate_index - 1]
            truth = gold[gold_index - 1]
            frame_error = abs(int(candidate["candidate_frame"]) - truth.hit_frame)
            same_video = _video_ids_match(str(candidate.get("video_id", "")), truth.video_id)
            effective_tolerance = tolerance + truth.uncertainty_frames + reaction_allowance_frames
            if same_video and frame_error <= effective_tolerance:
                previous = scores[candidate_index - 1][gold_index - 1]
                options.append(((previous[0] + 1, previous[1] - frame_error), "match"))
            best_score, best_action = max(options, key=lambda item: (item[0][0], item[0][1], priority[item[1]]))
            scores[candidate_index][gold_index] = best_score
            actions[candidate_index][gold_index] = best_action

    result: list[tuple[int, int]] = []
    candidate_index, gold_index = rows, columns
    while candidate_index > 0 or gold_index > 0:
        action = actions[candidate_index][gold_index]
        if action == "match":
            result.append((candidate_index - 1, gold_index - 1))
            candidate_index -= 1
            gold_index -= 1
        elif action == "skip_candidate":
            candidate_index -= 1
        elif action == "skip_gold":
            gold_index -= 1
        else:
            break
    return list(reversed(result))


def evaluate_candidates(
    candidates: list[dict[str, Any]],
    gold: list[GoldHit],
    *,
    tolerance: int = 5,
    reaction_allowance_frames: int = 0,
) -> dict[str, Any]:
    matches = _optimal_matches(candidates, gold, tolerance, reaction_allowance_frames)
    matched_candidate_indexes = {item[0] for item in matches}
    matched_gold_indexes = {item[1] for item in matches}
    matched_rows: list[dict[str, Any]] = []
    hitter_correct = 0
    hitter_scored = 0
    rally_correct = 0
    rally_scored = 0
    for candidate_index, gold_index in matches:
        candidate, truth = candidates[candidate_index], gold[gold_index]
        predicted_hitter = str(candidate.get("predicted_hitter") or "unknown")
        if truth.hitter != "unknown":
            hitter_scored += 1
            hitter_correct += predicted_hitter == truth.hitter
        predicted_rally = str(candidate.get("rally_id") or "unknown")
        if truth.rally_id != "unknown":
            rally_scored += 1
            rally_correct += predicted_rally == truth.rally_id
        matched_rows.append({
            "event_id": candidate.get("event_id"),
            "candidate_frame": int(candidate["candidate_frame"]),
            "gold_frame": truth.hit_frame,
            "frame_error": int(candidate["candidate_frame"]) - truth.hit_frame,
            "absolute_frame_error": abs(int(candidate["candidate_frame"]) - truth.hit_frame),
            "predicted_hitter": predicted_hitter,
            "gold_hitter": truth.hitter,
            "predicted_rally_id": predicted_rally,
            "gold_rally_id": truth.rally_id,
            "event_confidence": candidate.get("event_confidence"),
            "gold_confidence": truth.confidence,
            "gold_uncertainty_frames": truth.uncertainty_frames,
            "gold_notes": truth.notes,
        })
    false_positives = [candidates[index] for index in range(len(candidates)) if index not in matched_candidate_indexes]
    false_negatives = [gold[index] for index in range(len(gold)) if index not in matched_gold_indexes]
    matched_count = len(matches)
    precision = matched_count / len(candidates) if candidates else 0.0
    recall = matched_count / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = sorted(row["absolute_frame_error"] for row in matched_rows)
    signed_errors = sorted(int(row["frame_error"]) for row in matched_rows)
    median_error = None
    if errors:
        middle = len(errors) // 2
        median_error = errors[middle] if len(errors) % 2 else (errors[middle - 1] + errors[middle]) / 2
    signed_median = None
    signed_mad = None
    residual_p90 = None
    if signed_errors:
        middle = len(signed_errors) // 2
        signed_median = (
            signed_errors[middle]
            if len(signed_errors) % 2
            else (signed_errors[middle - 1] + signed_errors[middle]) / 2
        )
        residuals = sorted(abs(value - signed_median) for value in signed_errors)
        residual_middle = len(residuals) // 2
        signed_mad = (
            residuals[residual_middle]
            if len(residuals) % 2
            else (residuals[residual_middle - 1] + residuals[residual_middle]) / 2
        )
        residual_p90 = residuals[round((len(residuals) - 1) * 0.9)]

    ambiguous_gold = 0
    for truth in gold:
        effective_tolerance = tolerance + truth.uncertainty_frames + reaction_allowance_frames
        nearby = sum(
            _video_ids_match(str(candidate.get("video_id", "")), truth.video_id)
            and abs(int(candidate["candidate_frame"]) - truth.hit_frame) <= effective_tolerance
            for candidate in candidates
        )
        ambiguous_gold += nearby > 1

    return {
        "tolerance_frames": tolerance,
        "reaction_allowance_frames": reaction_allowance_frames,
        "effective_window_policy": "base tolerance + per-label uncertainty + reaction allowance",
        "candidate_count": len(candidates),
        "gold_count": len(gold),
        "matched_count": matched_count,
        "false_positive_count": len(false_positives),
        "false_negative_count": len(false_negatives),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "median_absolute_frame_error": median_error,
        "median_signed_frame_error": signed_median,
        "signed_frame_error_mad": signed_mad,
        "offset_normalized_p90_residual_frames": residual_p90,
        "ambiguous_gold_count": ambiguous_gold,
        "hitter_accuracy": None if not hitter_scored else round(hitter_correct / hitter_scored, 6),
        "hitter_scored_count": hitter_scored,
        "rally_id_accuracy": None if not rally_scored else round(rally_correct / rally_scored, 6),
        "rally_scored_count": rally_scored,
        "matches": matched_rows,
        "false_positives": false_positives,
        "false_negatives": [truth.__dict__ for truth in false_negatives],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def evaluate_files(
    candidates_path: Path,
    gold_path: Path,
    output_dir: Path,
    *,
    tolerances: tuple[int, ...] = (3, 5, 8, 10, 12, 15),
    primary_tolerance: int = 5,
    reaction_allowance_frames: int = 0,
) -> dict[str, Any]:
    candidates, gold = read_candidates(candidates_path), read_gold(gold_path)
    if primary_tolerance not in tolerances:
        tolerances = tuple(sorted({*tolerances, primary_tolerance}))
    raw_evaluations = {
        str(value): evaluate_candidates(candidates, gold, tolerance=value)
        for value in tolerances
    }
    robust_evaluations = {
        str(value): evaluate_candidates(
            candidates, gold, tolerance=value,
            reaction_allowance_frames=reaction_allowance_frames,
        )
        for value in tolerances
    }
    primary = robust_evaluations[str(primary_tolerance)]

    def compact(evaluations: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {
            key: {name: value for name, value in result.items()
                  if name not in {"matches", "false_positives", "false_negatives"}}
            for key, result in evaluations.items()
        }

    summary = {
        "schema_version": "1.0",
        "scope": "candidate event evaluation against independently marked human Gold",
        "candidates": str(candidates_path.resolve()),
        "gold": str(gold_path.resolve()),
        "primary_tolerance_frames": primary_tolerance,
        "reaction_allowance_frames": reaction_allowance_frames,
        "interpretation": {
            "raw": "No global reaction allowance; use only as a strict timing diagnostic.",
            "robust_presence": "Adds the declared reaction allowance; use for hit-presence recall and error review.",
            "offset": "Signed offset is reported but Gold frames are never automatically shifted.",
        },
        "raw_metrics_by_tolerance": compact(raw_evaluations),
        "robust_presence_metrics_by_tolerance": compact(robust_evaluations),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "hit_evaluation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(output_dir / "matched_events.csv", primary["matches"], [
        "event_id", "candidate_frame", "gold_frame", "frame_error", "absolute_frame_error",
        "predicted_hitter", "gold_hitter", "predicted_rally_id", "gold_rally_id",
        "event_confidence", "gold_confidence", "gold_uncertainty_frames", "gold_notes",
    ])
    _write_csv(output_dir / "false_positives.csv", primary["false_positives"], [
        "event_id", "video_id", "candidate_frame", "candidate_time", "predicted_hitter",
        "rally_id", "event_confidence", "trajectory_score", "hand_distance_score", "pose_score",
    ])
    _write_csv(output_dir / "false_negatives.csv", primary["false_negatives"], [
        "video_id", "hit_frame", "hitter", "rally_id", "confidence", "uncertainty_frames", "notes",
    ])
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tolerances", default="3,5,8,10,12,15")
    parser.add_argument("--primary-tolerance", type=int, default=5)
    parser.add_argument(
        "--reaction-allowance-frames", type=int, default=6,
        help="Extra symmetric matching allowance for live human keypress delay; Gold is not shifted",
    )
    args = parser.parse_args()
    try:
        tolerances = tuple(sorted({int(value.strip()) for value in args.tolerances.split(",") if value.strip()}))
        if not tolerances or any(value < 0 for value in tolerances):
            raise ValueError("tolerances must contain non-negative integers")
        result = evaluate_files(
            args.candidates, args.gold, args.output_dir,
            tolerances=tolerances, primary_tolerance=args.primary_tolerance,
            reaction_allowance_frames=args.reaction_allowance_frames,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
