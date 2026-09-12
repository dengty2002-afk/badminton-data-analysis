"""Validate a browser Gold bundle and materialize per-video CSV files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


VALID_HITTERS = {"upper", "lower", "unknown"}
VALID_CONFIDENCE = {"high", "medium", "low"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def extract_bundle(
    bundle_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if payload.get("schemaVersion") != "shuttlelab-gold-bundle-1":
        raise ValueError("unsupported Gold bundle schema")
    batch_id = str(payload.get("batchId") or "")
    if not batch_id:
        raise ValueError("Gold bundle has no batchId")
    videos = payload.get("videos")
    if not isinstance(videos, list) or not videos:
        raise ValueError("Gold bundle contains no videos")

    output_dir.mkdir(parents=True, exist_ok=True)
    seen_video_ids: set[str] = set()
    summaries: list[dict[str, Any]] = []
    total_hits = 0

    for video in videos:
        if not isinstance(video, dict):
            raise ValueError("Gold bundle video entry must be an object")
        video_id = str(video.get("videoId") or "")
        if not video_id or video_id in seen_video_ids:
            raise ValueError(f"invalid or duplicate videoId {video_id!r}")
        seen_video_ids.add(video_id)
        frame_count = _integer(video.get("frameCount"), f"{video_id} frameCount")
        if frame_count <= 0:
            raise ValueError(f"{video_id} frameCount must be positive")

        raw_hits = video.get("hits")
        if not isinstance(raw_hits, list):
            raise ValueError(f"{video_id} hits must be an array")
        if not raw_hits:
            raise ValueError(f"{video_id} contains no hit annotations")

        hits: list[dict[str, Any]] = []
        hit_frames: set[int] = set()
        for index, hit in enumerate(raw_hits, start=1):
            if not isinstance(hit, dict) or str(hit.get("videoId") or "") != video_id:
                raise ValueError(f"{video_id} hit {index} has a mismatched videoId")
            frame = _integer(hit.get("hitFrame"), f"{video_id} hit {index} frame")
            uncertainty = _integer(hit.get("uncertaintyFrames"), f"{video_id} hit {index} uncertainty")
            hitter = str(hit.get("hitter") or "")
            confidence = str(hit.get("confidence") or "")
            if not 0 <= frame < frame_count or frame in hit_frames:
                raise ValueError(f"{video_id} hit {index} has an invalid or duplicate frame")
            if uncertainty < 0 or hitter not in VALID_HITTERS or confidence not in VALID_CONFIDENCE:
                raise ValueError(f"{video_id} hit {index} has invalid categorical data")
            hit_frames.add(frame)
            hits.append({
                "video_id": video_id,
                "rally_id": str(hit.get("rallyId") or "unknown"),
                "hit_frame": frame,
                "hitter": hitter,
                "confidence": confidence,
                "uncertainty_frames": uncertainty,
                "notes": str(hit.get("notes") or ""),
            })
        hits.sort(key=lambda row: int(row["hit_frame"]))

        with (output_dir / f"{video_id}.hits_gold.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=[
                "video_id", "rally_id", "hit_frame", "hitter", "confidence", "uncertainty_frames", "notes",
            ])
            writer.writeheader()
            writer.writerows(hits)
        total_hits += len(hits)
        summaries.append({
            "video_id": video_id,
            "frame_count": frame_count,
            "hit_count": len(hits),
            "first_hit_frame": hits[0]["hit_frame"],
            "last_hit_frame": hits[-1]["hit_frame"],
        })

    report = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "source_bundle": bundle_path.name,
        "video_count": len(summaries),
        "total_hits": total_hits,
        "rally_activity_policy": "inferred automatically from shuttle speed, trajectory continuity, and hit sequence; no manual phase classification",
        "complete_for_locked_evaluation": all(row["hit_count"] > 0 for row in summaries),
        "videos": summaries,
    }
    (output_dir / "bundle_validation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    locked_paths = [bundle_path, *sorted(output_dir.glob("vid_*.hits_gold.csv"))]
    lock = {
        "schema_version": "1.0",
        "batch_id": batch_id,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "scope": "human hit-event Gold only",
        "manual_phase_classification": False,
        "rally_activity": "automatic shuttle-speed, trajectory-continuity, and hit-sequence inference",
        "files": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in locked_paths
        ],
    }
    lock_path = output_dir / "gold_lock.json"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        previous_hashes = {row["path"]: row["sha256"] for row in previous.get("files", [])}
        current_hashes = {row["path"]: row["sha256"] for row in lock["files"]}
        if previous_hashes != current_hashes:
            raise ValueError("Gold lock already exists and file hashes have changed")
    else:
        lock_path.write_text(json.dumps(lock, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = extract_bundle(args.bundle, args.output_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
