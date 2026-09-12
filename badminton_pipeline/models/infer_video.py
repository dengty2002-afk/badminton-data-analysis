"""Run Good-Badminton pose and shuttle models and write normalized JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

from .good_badminton import (
    RTMPoseAdapter,
    ShuttleAdapter,
    load_and_verify_manifest,
    model_metadata,
)
from .records import FrameDetections, ShuttleDetection


def video_id_for_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    # Keep one canonical identifier across ingest, inference, calibration and
    # dataset export.  Older smoke artifacts used the first 24 checksum
    # characters without a prefix; _video_ids_match remains backward
    # compatible with those files.
    return f"vid_{digest.hexdigest()[:12]}"


def _load_segments(path: Path | None) -> list[tuple[int, int]]:
    if path is None:
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    segments: list[tuple[int, int]] = []
    for row in value.get("segments", []):
        start, end = int(row["start_frame"]), int(row["end_frame"])
        if start < 0 or end < start:
            raise ValueError(f"invalid main-view segment in {path}: {start}..{end}")
        segments.append((start, end))
    if not segments:
        raise ValueError(f"main-view manifest has no segments: {path}")
    return segments


def _selected_frame_count(
    segments: list[tuple[int, int]], stride: int, max_frames: int,
) -> int | None:
    """Return the exact selected-frame count when bounded segments are used."""
    if not segments:
        return None
    total = 0
    for start, end in segments:
        first = start + (-start % stride)
        if first <= end:
            total += ((end - first) // stride) + 1
    return min(total, max_frames) if max_frames else total


def _remaining_segment_ranges(
    segments: list[tuple[int, int]], stride: int, resumed_frames: int,
) -> list[tuple[int, int]]:
    """Seek directly to the first not-yet-inferred selected source frame."""
    remaining = resumed_frames
    result: list[tuple[int, int]] = []
    for start, end in segments:
        first = start + (-start % stride)
        selected = 0 if first > end else ((end - first) // stride) + 1
        if remaining >= selected:
            remaining -= selected
            continue
        next_selected = first + remaining * stride
        result.append((next_selected, end))
        remaining = 0
    return result


def _resume_prefix(
    output_path: Path,
    *,
    video_id: str,
    segments: list[tuple[int, int]],
    stride: int,
    max_frames: int,
) -> int:
    """Validate an interrupted JSONL as an exact source-frame prefix.

    A process can be killed while writing its final line.  Only that trailing
    partial line is discarded; any earlier mismatch makes the cache unsafe and
    causes a clean restart.
    """
    if not output_path.is_file() or not segments:
        return 0
    expected = (
        frame_idx
        for start, end in segments
        for frame_idx in range(start, end + 1)
        if frame_idx % stride == 0
    )
    valid_rows = 0
    valid_bytes = 0
    file_size = output_path.stat().st_size
    try:
        with output_path.open("rb") as stream:
            while True:
                line_start = stream.tell()
                line = stream.readline()
                if not line:
                    break
                # Never append directly after a non-newline-terminated record:
                # it may be an interrupted write even when its JSON happens to
                # parse successfully.
                if not line.endswith(b"\n"):
                    valid_bytes = line_start
                    break
                if not line.strip():
                    return 0
                try:
                    row = json.loads(line)
                    expected_frame = next(expected)
                except StopIteration:
                    return 0
                except (UnicodeDecodeError, json.JSONDecodeError):
                    if stream.tell() == file_size:
                        valid_bytes = line_start
                        break
                    return 0
                if (
                    str(row.get("video_id")) != video_id
                    or int(row.get("frame_idx", -1)) != expected_frame
                ):
                    return 0
                valid_rows += 1
                valid_bytes = stream.tell()
                if max_frames and valid_rows >= max_frames:
                    break
    except (OSError, TypeError, ValueError):
        return 0
    if valid_bytes != file_size:
        with output_path.open("r+b") as stream:
            stream.truncate(valid_bytes)
    return valid_rows


def run(args: argparse.Namespace) -> dict[str, object]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is not installed") from exc

    manifest = load_and_verify_manifest(args.manifest, args.models_dir)
    detector_spec = manifest["models"]["person_detector"]
    pose_spec = manifest["models"]["pose"]
    shuttle_spec = manifest["models"]["shuttle"]
    pose = RTMPoseAdapter(
        args.models_dir / detector_spec["filename"],
        args.models_dir / pose_spec["filename"],
        device=args.device,
        confidence_threshold=args.pose_confidence,
    )
    shuttle = None
    if not args.skip_shuttle:
        shuttle = ShuttleAdapter(
            args.models_dir / shuttle_spec["filename"],
            device=args.device,
            confidence=args.shuttle_confidence,
            top_k=int(getattr(args, "shuttle_top_k", 5)),
        )

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.input}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_id = video_id_for_path(args.input)
    models = model_metadata(manifest, args.device)
    models["shuttle"]["confidence_threshold"] = float(args.shuttle_confidence)
    models["shuttle"]["top_k"] = int(getattr(args, "shuttle_top_k", 5))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    segments = _load_segments(getattr(args, "segments_manifest", None))
    if segments and args.start_frame:
        raise ValueError("--start-frame cannot be combined with --segments-manifest")
    if args.start_frame:
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
    started = perf_counter()
    resume_requested = bool(getattr(args, "resume_existing_raw", False))
    resumed_frames = (
        _resume_prefix(
            args.output,
            video_id=video_id,
            segments=segments,
            stride=args.stride,
            max_frames=args.max_frames,
        )
        if resume_requested else 0
    )
    expected_frames = _selected_frame_count(segments, args.stride, args.max_frames)
    if expected_frames is not None and resumed_frames > expected_frames:
        resumed_frames = 0
    processed = resumed_frames
    seen = args.start_frame
    output_mode = "a" if resumed_frames else "w"
    with args.output.open(output_mode, encoding="utf-8", newline="\n") as output:
        ranges = (
            _remaining_segment_ranges(segments, args.stride, resumed_frames)
            if segments else [(args.start_frame, None)]
        )
        stop = False
        if expected_frames is not None and processed >= expected_frames:
            stop = True
        for range_start, range_end in ranges:
            if stop:
                break
            capture.set(cv2.CAP_PROP_POS_FRAMES, range_start)
            frame_idx = range_start
            while range_end is None or frame_idx <= range_end:
                ok, frame = capture.read()
                if not ok:
                    break
                seen += 1
                current_frame = frame_idx
                frame_idx += 1
                if current_frame % args.stride:
                    continue
                players = pose.infer(frame)
                shuttle_detection = (
                    shuttle.infer(frame)
                    if shuttle is not None
                    else ShuttleDetection(raw_xy=None, confidence=None, state="not_run")
                )
                record = FrameDetections(
                    schema_version="1.0", video_id=video_id, frame_idx=current_frame,
                    timestamp_sec=current_frame / fps, width=width, height=height,
                    players=players, shuttle=shuttle_detection, models=models,
                )
                output.write(json.dumps(record.to_dict(), ensure_ascii=False, separators=(",", ":")) + "\n")
                processed += 1
                if args.max_frames and processed >= args.max_frames:
                    stop = True
                    break
            if stop:
                break
    capture.release()

    summary = {
        "schema_version": "1.0",
        "video_id": video_id,
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "processed_frames": processed,
        "resumed_frames": resumed_frames,
        "resumed_existing_raw": bool(resumed_frames),
        "start_frame": args.start_frame,
        "source_frames_read": seen if segments else seen - args.start_frame,
        "end_frame_exclusive": None if segments else seen,
        "segments_manifest": None if not segments else str(args.segments_manifest.resolve()),
        "segment_count": len(segments),
        "stride": args.stride,
        "fps": fps,
        "elapsed_sec": round(perf_counter() - started, 3),
        "inference_fps": round(processed / max(1e-9, perf_counter() - started), 3),
        "python": platform.python_version(),
        "models": models,
        "shuttle_top_k": int(getattr(args, "shuttle_top_k", 5)),
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=root / "models" / "good-badminton")
    parser.add_argument("--manifest", type=Path, default=root / "configs" / "models.good-badminton.json")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--segments-manifest", type=Path)
    parser.add_argument(
        "--resume-existing-raw", action="store_true",
        help="validate and append to an interrupted source-frame JSONL prefix",
    )
    parser.add_argument("--pose-confidence", type=float, default=0.2)
    parser.add_argument("--shuttle-confidence", type=float, default=0.18)
    parser.add_argument("--shuttle-top-k", type=int, default=5)
    parser.add_argument(
        "--skip-shuttle",
        action="store_true",
        help="run the smaller RTMPose-only slice without installing PyTorch/Ultralytics",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.stride < 1:
        raise SystemExit("--stride must be at least 1")
    if args.start_frame < 0:
        raise SystemExit("--start-frame cannot be negative")
    if args.shuttle_top_k < 1:
        raise SystemExit("--shuttle-top-k must be at least 1")
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
