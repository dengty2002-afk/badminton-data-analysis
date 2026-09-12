"""Rebuild the canonical video catalog from per-video metadata records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from badminton_pipeline.ingest.register_video import SCHEMA_VERSION, read_json, utc_now, write_json


def rebuild_video_catalog(data_dir: Path) -> dict[str, Any]:
    records_dir = data_dir / "videos"
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_checksums: set[str] = set()
    for path in sorted(records_dir.glob("vid_*.json")):
        record = read_json(path, None)
        if not isinstance(record, dict):
            raise ValueError(f"invalid video record: {path}")
        video_id = str(record.get("video_id") or "")
        checksum = str(record.get("checksum") or "")
        if not video_id or path.stem != video_id or not checksum:
            raise ValueError(f"video record identity/checksum mismatch: {path}")
        if video_id in seen_ids or checksum in seen_checksums:
            raise ValueError(f"duplicate video id or checksum: {path}")
        seen_ids.add(video_id)
        seen_checksums.add(checksum)
        records.append(record)
    records.sort(key=lambda row: (str(row.get("registered_at") or ""), str(row["video_id"])))
    catalog = {
        "schema_version": SCHEMA_VERSION,
        "videos": records,
        "updated_at": utc_now(),
        "source": "rebuilt_from_data/videos/vid_*.json",
    }
    write_json(data_dir / "videos.json", catalog)
    return {"status": "rebuilt", "video_count": len(records), "path": str((data_dir / "videos.json").resolve())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    try:
        result = rebuild_video_catalog(args.data_dir.resolve())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
