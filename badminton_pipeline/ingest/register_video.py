"""Register an MP4 video without heavyweight runtime dependencies.

This first vertical slice deliberately uses only the Python standard library.
It calculates a stable checksum, reads core MP4 container metadata, creates a
traceable run record, and publishes a small manifest for the annotation UI.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


PIPELINE_VERSION = "ingest-0.1.0"
SCHEMA_VERSION = "1.0"
CHUNK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class Box:
    kind: str
    payload_start: int
    end: int


@dataclass(frozen=True)
class MediaMetadata:
    duration_sec: float | None
    width: int | None
    height: int | None
    fps: float | None
    frame_count: int | None
    has_audio: bool
    container: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def iter_boxes(data: bytes, start: int = 0, end: int | None = None) -> Iterator[Box]:
    limit = len(data) if end is None else min(end, len(data))
    offset = start
    while offset + 8 <= limit:
        size = struct.unpack_from(">I", data, offset)[0]
        kind = data[offset + 4 : offset + 8].decode("latin-1")
        header_size = 8
        if size == 1:
            if offset + 16 > limit:
                return
            size = struct.unpack_from(">Q", data, offset + 8)[0]
            header_size = 16
        elif size == 0:
            size = limit - offset

        if size < header_size or offset + size > limit:
            return

        yield Box(kind=kind, payload_start=offset + header_size, end=offset + size)
        offset += size


def find_box(data: bytes, path: tuple[str, ...], start: int = 0, end: int | None = None) -> bytes | None:
    if not path:
        return None
    for box in iter_boxes(data, start, end):
        if box.kind != path[0]:
            continue
        if len(path) == 1:
            return data[box.payload_start : box.end]
        nested = find_box(data, path[1:], box.payload_start, box.end)
        if nested is not None:
            return nested
    return None


def read_top_level_box(path: Path, target: str) -> bytes | None:
    file_size = path.stat().st_size
    with path.open("rb") as stream:
        offset = 0
        while offset + 8 <= file_size:
            stream.seek(offset)
            header = stream.read(16)
            if len(header) < 8:
                return None
            size = struct.unpack_from(">I", header, 0)[0]
            kind = header[4:8].decode("latin-1")
            header_size = 8
            if size == 1:
                if len(header) < 16:
                    return None
                size = struct.unpack_from(">Q", header, 8)[0]
                header_size = 16
            elif size == 0:
                size = file_size - offset
            if size < header_size or offset + size > file_size:
                return None
            if kind == target:
                stream.seek(offset + header_size)
                return stream.read(size - header_size)
            offset += size
    return None


def parse_duration(payload: bytes | None) -> tuple[int | None, int | None]:
    if not payload or len(payload) < 20:
        return None, None
    version = payload[0]
    if version == 1 and len(payload) >= 32:
        timescale = struct.unpack_from(">I", payload, 20)[0]
        duration = struct.unpack_from(">Q", payload, 24)[0]
    else:
        timescale = struct.unpack_from(">I", payload, 12)[0]
        duration = struct.unpack_from(">I", payload, 16)[0]
    return timescale or None, duration


def parse_mp4(path: Path) -> MediaMetadata:
    moov = read_top_level_box(path, "moov")
    if moov is None:
        return MediaMetadata(None, None, None, None, None, False, path.suffix.lower().lstrip("."))

    movie_scale, movie_duration = parse_duration(find_box(moov, ("mvhd",)))
    duration_sec = movie_duration / movie_scale if movie_scale and movie_duration is not None else None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None
    has_audio = False

    for track in iter_boxes(moov):
        if track.kind != "trak":
            continue
        track_data = moov[track.payload_start : track.end]
        handler = find_box(track_data, ("mdia", "hdlr"))
        handler_type = handler[8:12].decode("latin-1") if handler and len(handler) >= 12 else ""
        if handler_type == "soun":
            has_audio = True
            continue
        if handler_type != "vide":
            continue

        tkhd = find_box(track_data, ("tkhd",))
        if tkhd and len(tkhd) >= 8:
            width_fixed, height_fixed = struct.unpack_from(">II", tkhd, len(tkhd) - 8)
            width = round(width_fixed / 65536)
            height = round(height_fixed / 65536)

        media_scale, media_duration = parse_duration(find_box(track_data, ("mdia", "mdhd")))
        stsz = find_box(track_data, ("mdia", "minf", "stbl", "stsz"))
        if stsz and len(stsz) >= 12:
            frame_count = struct.unpack_from(">I", stsz, 8)[0]
        media_seconds = media_duration / media_scale if media_scale and media_duration is not None else duration_sec
        if frame_count is not None and media_seconds:
            fps = frame_count / media_seconds
        if duration_sec is None:
            duration_sec = media_seconds
        break

    return MediaMetadata(
        duration_sec=round(duration_sec, 6) if duration_sec is not None else None,
        width=width,
        height=height,
        fps=round(fps, 6) if fps is not None else None,
        frame_count=frame_count,
        has_audio=has_audio,
        container=path.suffix.lower().lstrip("."),
    )


def read_json(path: Path, fallback: object) -> object:
    if not path.exists():
        return fallback
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Register one source video in the ShuttleLab dataset pipeline.")
    parser.add_argument("video", type=Path, help="Source MP4/MOV/MKV path")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--public-manifest", type=Path, default=None)
    parser.add_argument("--match-id", default="pilot-001")
    parser.add_argument("--source", default="local")
    parser.add_argument("--domain", default="fixed-camera-singles")
    parser.add_argument("--redistribution", choices=("allowed", "private", "unknown"), default="private")
    return parser


def register_video(
    video: Path,
    *,
    data_dir: Path = Path("data"),
    public_manifest: Path | None = None,
    match_id: str = "pilot-001",
    source: str = "local",
    domain: str = "fixed-camera-singles",
    redistribution: str = "private",
) -> dict[str, object]:
    """Register one video and return the same structured result as the CLI.

    Registration is idempotent by SHA-256: a repeated call refreshes the
    canonical per-video record instead of duplicating the catalog entry.
    """
    video = video.expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"Video does not exist: {video}")
    if video.suffix.lower() not in {".mp4", ".mov", ".mkv"}:
        raise ValueError(f"Unsupported video extension: {video.suffix}")

    started_at = utc_now()
    checksum = sha256_file(video)
    metadata = parse_mp4(video) if video.suffix.lower() in {".mp4", ".mov"} else MediaMetadata(None, None, None, None, None, False, "mkv")
    video_id = f"vid_{checksum[:12]}"
    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid.uuid4().hex[:6]}"
    file_size = video.stat().st_size

    record = {
        "video_id": video_id,
        "path": str(video),
        "file_name": video.name,
        "checksum": checksum,
        "checksum_algorithm": "sha256",
        "file_size_bytes": file_size,
        **asdict(metadata),
        "match_id": match_id,
        "players": {"A": "unknown", "B": "unknown"},
        "source": source,
        "domain": domain,
        "redistribution": redistribution,
        "registered_at": started_at,
        "status": "registered",
    }

    data_dir = data_dir.resolve()
    catalog_path = data_dir / "videos.json"
    catalog = read_json(catalog_path, {"schema_version": SCHEMA_VERSION, "videos": []})
    if not isinstance(catalog, dict) or not isinstance(catalog.get("videos"), list):
        raise ValueError(f"Invalid video catalog: {catalog_path}")
    retained = [item for item in catalog["videos"] if item.get("checksum") != checksum]
    catalog["videos"] = [*retained, record]
    catalog["updated_at"] = utc_now()

    artifact_path = data_dir / "videos" / f"{video_id}.json"
    run_path = data_dir / "runs" / f"{run_id}.json"
    run_record = {
        "run_id": run_id,
        "pipeline_stage": "video_ingest",
        "pipeline_version": PIPELINE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "status": "completed",
        "started_at": started_at,
        "completed_at": utc_now(),
        "input": {"path": str(video), "checksum": checksum, "size_bytes": file_size},
        "output": {"video_id": video_id, "record": str(artifact_path)},
        "runtime": {"python": sys.version.split()[0], "platform": sys.platform},
    }

    write_json(artifact_path, record)
    write_json(catalog_path, catalog)
    write_json(run_path, run_record)

    if public_manifest:
        public_manifest = public_manifest.resolve()
        write_json(public_manifest, {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now(),
            "pipeline_version": PIPELINE_VERSION,
            "video": record,
            "latest_run": run_record,
        })

    return {
        "status": "registered",
        "video_id": video_id,
        "checksum": checksum,
        "metadata": asdict(metadata),
        "record": str(artifact_path),
        "run": str(run_path),
    }


def main() -> int:
    args = build_parser().parse_args()
    try:
        result = register_video(
            args.video,
            data_dir=args.data_dir,
            public_manifest=args.public_manifest,
            match_id=args.match_id,
            source=args.source,
            domain=args.domain,
            redistribution=args.redistribution,
        )
    except (OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
