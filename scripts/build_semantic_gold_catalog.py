"""Build the browser-safe semantic Gold catalog from locked hit Gold files."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    videos: list[dict[str, object]] = []
    for gold_path in sorted((ROOT / "data" / "gold").rglob("*.hits_gold.csv")):
        with gold_path.open("r", encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            continue
        video_id = str(rows[0]["video_id"]).strip()
        record_path = ROOT / "data" / "videos" / f"{video_id}.json"
        if not record_path.is_file():
            continue
        record = json.loads(record_path.read_text(encoding="utf-8"))
        anchors = []
        for row in rows:
            frame = int(row["hit_frame"])
            anchors.append({
                "anchorId": f"{video_id}:{frame}",
                "rallyId": str(row.get("rally_id") or "unknown"),
                "hitFrame": frame,
                "hitter": str(row.get("hitter") or "unknown"),
                "confidence": str(row.get("confidence") or "unknown"),
                "uncertaintyFrames": int(row.get("uncertainty_frames") or 0),
                "notes": str(row.get("notes") or ""),
            })
        anchors.sort(key=lambda item: int(item["hitFrame"]))
        videos.append({
            "videoId": video_id,
            "fileName": str(record.get("file_name") or Path(str(record.get("path") or "video.mp4")).name),
            "fps": float(record.get("fps") or 30.0),
            "frameCount": int(record.get("frame_count") or 0),
            "durationSec": float(record.get("duration_sec") or 0.0),
            "anchorCount": len(anchors),
            "anchors": anchors,
        })
    output = ROOT / "demo" / "public" / "gold" / "catalog.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schemaVersion": "shuttlelab-semantic-catalog-1",
        "generatedFrom": "locked hit Gold; model predictions excluded",
        "videoCount": len(videos),
        "anchorCount": sum(int(video["anchorCount"]) for video in videos),
        "videos": videos,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "videos": len(videos), "anchors": payload["anchorCount"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
