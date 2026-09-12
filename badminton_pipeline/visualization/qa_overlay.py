"""Render a lightweight local QA video over pipeline detections."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any


COCO_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
PLAYER_COLORS = {"upper": (255, 210, 40), "lower": (40, 150, 255)}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _point(value: object) -> tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return round(float(value[0])), round(float(value[1]))


def _draw_pose(cv2: Any, image: Any, player: dict[str, Any]) -> None:
    identity = str(player.get("identity", "unknown"))
    color = PLAYER_COLORS.get(identity, (220, 220, 220))
    points = player.get("keypoints_xy", [])
    missing = player.get("keypoint_missing", [])
    for first, second in COCO_EDGES:
        if first >= len(points) or second >= len(points):
            continue
        if (first < len(missing) and missing[first]) or (second < len(missing) and missing[second]):
            continue
        p1, p2 = _point(points[first]), _point(points[second])
        if p1 is not None and p2 is not None:
            cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
    for index, value in enumerate(points):
        if index < len(missing) and missing[index]:
            continue
        point = _point(value)
        if point is not None:
            cv2.circle(image, point, 3, color, -1, cv2.LINE_AA)
    bbox = player.get("bbox_xyxy")
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        left_top = (round(float(bbox[0])), round(float(bbox[1])))
        right_bottom = (round(float(bbox[2])), round(float(bbox[3])))
        cv2.rectangle(image, left_top, right_bottom, color, 2)
        court = player.get("court_xy_m")
        court_text = ""
        if isinstance(court, (list, tuple)) and len(court) >= 2:
            court_text = f"  court=({float(court[0]):.2f},{float(court[1]):.2f})m"
        cv2.putText(image, identity + court_text, (left_top[0], max(18, left_top[1] - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def render_overlay(
    video_path: Path,
    frames_path: Path,
    events_path: Path,
    calibration_path: Path,
    output_path: Path,
    *,
    start_frame: int = 0,
    max_frames: int = 0,
    scale: float = 1.0,
    trail_frames: int = 20,
    mode: str = "qa",
) -> dict[str, Any]:
    if start_frame < 0:
        raise ValueError("start_frame cannot be negative")
    if max_frames < 0:
        raise ValueError("max_frames cannot be negative")
    if not 0.1 <= scale <= 1.0:
        raise ValueError("scale must be between 0.1 and 1.0")
    if trail_frames < 1:
        raise ValueError("trail_frames must be positive")
    if mode not in {"qa", "gold"}:
        raise ValueError("mode must be 'qa' or 'gold'")
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required for QA video rendering") from exc

    records = {int(item["frame_idx"]): item for item in _read_jsonl(frames_path)}
    events = _read_jsonl(events_path)
    events_by_frame = {int(item["candidate_frame"]): item for item in events}
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    corners = [_point([item["x"], item["y"]]) for item in calibration.get("corners", [])]
    corners = [point for point in corners if point is not None]

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_width = max(2, round(source_width * scale))
    output_height = max(2, round(source_height * scale))
    output_width -= output_width % 2
    output_height -= output_height % 2
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (output_width, output_height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"cannot create QA video: {output_path}")

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    written = 0
    trail: deque[tuple[int, int]] = deque(maxlen=trail_frames)
    started = perf_counter()
    while True:
        ok, image = capture.read()
        if not ok:
            break
        record = records.get(frame_idx)
        if mode == "qa" and len(corners) == 4:
            for index in range(4):
                cv2.line(image, corners[index], corners[(index + 1) % 4], (80, 240, 90), 2, cv2.LINE_AA)
                cv2.circle(image, corners[index], 6, (80, 240, 90), -1, cv2.LINE_AA)
                cv2.putText(image, f"C{index + 1}", (corners[index][0] + 6, corners[index][1] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 240, 90), 2, cv2.LINE_AA)
        if mode == "qa" and record is not None:
            for player in record.get("players", []):
                _draw_pose(cv2, image, player)
            shuttle = record.get("shuttle", {})
            raw = _point(shuttle.get("raw_xy"))
            smooth = _point(shuttle.get("smoothed_xy"))
            if smooth is not None:
                trail.append(smooth)
            if len(trail) > 1:
                for index in range(1, len(trail)):
                    intensity = max(60, round(255 * index / len(trail)))
                    cv2.line(image, trail[index - 1], trail[index], (intensity, 40, 255), 2, cv2.LINE_AA)
            if raw is not None:
                cv2.circle(image, raw, 7, (40, 40, 255), 2, cv2.LINE_AA)
            if smooth is not None:
                cv2.circle(image, smooth, 5, (255, 40, 255), -1, cv2.LINE_AA)
                cv2.putText(image, str(shuttle.get("tracking_state", "unknown")),
                            (smooth[0] + 8, smooth[1] - 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 40, 255), 2, cv2.LINE_AA)

        cv2.rectangle(image, (0, 0), (source_width, 45), (20, 20, 20), -1)
        header = "QA" if mode == "qa" else "GOLD REFERENCE"
        cv2.putText(image, f"{header}  frame={frame_idx}  time={frame_idx / fps:.3f}s",
                    (14, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
        event = events_by_frame.get(frame_idx) if mode == "qa" else None
        if event is not None:
            text = (f"HIT CANDIDATE  {event.get('predicted_hitter', 'unknown')}  "
                    f"confidence={float(event.get('event_confidence', 0.0)):.3f}")
            cv2.rectangle(image, (0, source_height - 52), (source_width, source_height), (15, 15, 210), -1)
            cv2.putText(image, text, (14, source_height - 18), cv2.FONT_HERSHEY_SIMPLEX,
                        0.78, (255, 255, 255), 2, cv2.LINE_AA)
        if scale != 1.0:
            image = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)
        writer.write(image)
        written += 1
        frame_idx += 1
        if max_frames and written >= max_frames:
            break

    capture.release()
    writer.release()
    if written == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError("no video frames were written")
    summary = {
        "schema_version": "1.0",
        "video": str(video_path.resolve()),
        "frames": str(frames_path.resolve()),
        "events": str(events_path.resolve()),
        "calibration": str(calibration_path.resolve()),
        "output": str(output_path.resolve()),
        "start_frame": start_frame,
        "written_frames": written,
        "end_frame_exclusive": start_frame + written,
        "fps": fps,
        "width": output_width,
        "height": output_height,
        "source_event_candidates": sum(start_frame <= frame < start_frame + written for frame in events_by_frame),
        "visible_event_candidates": (
            sum(start_frame <= frame < start_frame + written for frame in events_by_frame) if mode == "qa" else 0
        ),
        "mode": mode,
        "machine_overlays_visible": mode == "qa",
        "elapsed_sec": round(perf_counter() - started, 3),
        "size_bytes": output_path.stat().st_size,
    }
    output_path.with_suffix(".metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--trail-frames", type=int, default=20)
    parser.add_argument(
        "--mode", choices=("qa", "gold"), default="qa",
        help="qa shows model overlays; gold only shows frame/time to avoid annotation bias",
    )
    args = parser.parse_args()
    try:
        result = render_overlay(
            args.video, args.frames, args.events, args.calibration, args.output,
            start_frame=args.start_frame, max_frames=args.max_frames,
            scale=args.scale, trail_frames=args.trail_frames, mode=args.mode,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
