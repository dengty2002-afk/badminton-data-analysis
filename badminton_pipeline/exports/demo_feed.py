"""Convert hit-candidate JSONL into the review Demo's compact JSON feed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def build_demo_feed(events: list[dict[str, Any]], *, rally: str = "Model pilot") -> dict[str, Any]:
    mapped = []
    for event in events:
        frame = int(event["candidate_frame"])
        mapped.append(
            {
                "id": f"EVT-M{frame:05d}",
                "rally": event.get("rally_id", rally),
                "time": float(event["candidate_time"]),
                "hitter": "A" if event.get("predicted_hitter") == "upper" else "B",
                "hitter_side": event.get("predicted_hitter"),
                "stroke": "待标注",
                "confidence": round(float(event["event_confidence"]), 3),
                "trajectory": round(float(event["trajectory_score"]), 3),
                "wrist": round(float(event["hand_distance_score"]), 3),
                "pose": round(float(event["pose_score"]), 3),
                "status": "unreviewed",
            }
        )
    first_frame = min((int(event["window_start_frame"]) for event in events), default=0)
    last_frame = max((int(event["window_end_frame"]) for event in events), default=0)
    first = events[0] if events else {}
    return {
        "schema_version": "1.0",
        "pipeline_version": first.get("model_version", "hit-rules-0.2.0"),
        "source_frames": {"start": first_frame, "end_exclusive": last_frame + 1},
        "calibration_version": first.get("calibration_version"),
        "events": mapped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rally", default="Model pilot")
    args = parser.parse_args()
    with args.events.open("r", encoding="utf-8") as stream:
        events = [json.loads(line) for line in stream if line.strip()]
    feed = build_demo_feed(events, rally=args.rally)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(feed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"events": len(feed["events"]), "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
