"""Join semantic Gold to enriched frames and export supervised BST samples."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from badminton_pipeline.evaluation.stroke_semantics import read_semantic_gold
from badminton_pipeline.stroke_classification.features import export_bst_samples


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def export_semantic_samples(
    frames_path: Path,
    semantic_gold_path: Path,
    output_path: Path,
    *,
    window_size: int = 30,
    keypoint_threshold: float = 0.2,
) -> dict[str, Any]:
    frames = _read_jsonl(frames_path)
    gold = read_semantic_gold(semantic_gold_path)
    frame_video_ids = {str(row.get("video_id") or "") for row in frames}
    gold_video_ids = {row.video_id for row in gold}
    if len(frame_video_ids) != 1 or frame_video_ids != gold_video_ids:
        raise ValueError("frames and semantic Gold must belong to exactly one matching video_id")
    events = []
    for index, row in enumerate(gold, start=1):
        events.append({
            **asdict(row),
            "event_id": f"gold_semantic_{index:06d}",
            "candidate_frame": row.hit_frame,
            "predicted_hitter": row.hitter,
            "human_hitter": row.hitter,
            "human_stroke_type": row.stroke_type,
            "label_source": "human_semantic_gold",
        })
    temporary_events = output_path.with_suffix(output_path.suffix + ".semantic_events.tmp.jsonl")
    temporary_events.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary_events.open("w", encoding="utf-8", newline="\n") as stream:
            for row in events:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        result = export_bst_samples(
            frames_path,
            temporary_events,
            output_path,
            window_size=window_size,
            keypoint_threshold=keypoint_threshold,
        )
    finally:
        temporary_events.unlink(missing_ok=True)
    metadata_path = Path(str(output_path) + ".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["semantic_gold"] = str(semantic_gold_path.resolve())
    metadata["task"] = "18-class ShuttleSet stroke classification"
    metadata["label_source"] = "human_semantic_gold"
    metadata["destination_targets"] = {
        "landing_x_m": [row.landing_x for row in gold],
        "landing_y_m": [row.landing_y for row in gold],
        "landing_kind": [row.landing_kind for row in gold],
        "landing_frame": [row.landing_frame for row in gold],
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {**result, "semantic_gold": str(semantic_gold_path.resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--semantic-gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--keypoint-threshold", type=float, default=0.2)
    args = parser.parse_args()
    try:
        result = export_semantic_samples(
            args.frames,
            args.semantic_gold,
            args.output,
            window_size=args.window_size,
            keypoint_threshold=args.keypoint_threshold,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
