"""Extract or index the stable court camera using Good-Badminton's template rule.

Good-Badminton identifies a court view with normalized grayscale template
matching and a default score threshold of 0.75.  This lightweight adapter
keeps that rule, applies the project's five-frame entry/exit hysteresis, and
writes one silent video containing only stable main-camera intervals, or only
the source-frame manifest when ``--manifest-only`` is selected.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any


def adaptive_threshold(scores: list[float], configured: float, margin: float = 0.012) -> float:
    """Lower a cross-event template threshold only for a tight high-score mode.

    Broadcast color grading and overlays shift absolute correlation even when
    the camera geometry is unchanged. A concentrated score distribution above
    0.65 is treated as a new event's main-camera mode; diverse distributions
    keep the configured threshold to avoid admitting replays and close-ups.
    """
    if not scores:
        return configured
    ordered = sorted(scores)
    p10 = ordered[int(0.10 * (len(ordered) - 1))]
    p90 = ordered[int(0.90 * (len(ordered) - 1))]
    if p10 >= 0.65 and p90 - p10 <= 0.12 and p90 < configured:
        return max(0.65, p10 - margin)
    # A diverse broadcast contains close-ups/replays, so it should not use the
    # tight-mode rule. If its best court-like frames narrowly miss a stricter
    # cross-event configuration, fall back only to Good-Badminton's documented
    # default boundary instead of returning an empty manifest.
    if ordered[-1] < configured and ordered[-1] >= 0.75:
        return 0.75
    # A cross-event template can still produce a handful of scores just above
    # a strict configured threshold while rejecting most of the same stable
    # camera view.  That case used to bypass the near-miss fallback solely
    # because ``max(scores)`` crossed the boundary.  Fall back to the upstream
    # 0.75 rule only when strict matches are sparse and the default boundary
    # recovers a materially large, at least three-times-bigger population.  A
    # normal diverse broadcast with a healthy strict match population remains
    # unchanged.
    if configured > 0.75:
        strict_ratio = sum(score >= configured for score in ordered) / len(ordered)
        default_ratio = sum(score >= 0.75 for score in ordered) / len(ordered)
        if strict_ratio < 0.10 and default_ratio >= 0.15 and default_ratio >= strict_ratio * 3.0:
            return 0.75
    return configured


def _runs(flags: list[bool], value: bool) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, flag in enumerate(flags):
        if flag == value and start is None:
            start = index
        elif flag != value and start is not None:
            runs.append((start, index - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return runs


def stabilize_views(
    raw_flags: list[bool],
    *,
    court_frames_threshold: int = 5,
    non_court_frames_threshold: int = 5,
) -> list[bool]:
    """Apply Good-Badminton's consecutive-frame entry/exit behavior."""
    if court_frames_threshold < 1 or non_court_frames_threshold < 1:
        raise ValueError("frame thresholds must be positive")

    stable = list(raw_flags)
    # A shorter miss does not end the active court view.
    for start, end in _runs(stable, False):
        if start > 0 and end < len(stable) - 1 and end - start + 1 < non_court_frames_threshold:
            stable[start : end + 1] = [True] * (end - start + 1)
    # A shorter match does not start a court view.
    for start, end in _runs(stable, True):
        if end - start + 1 < court_frames_threshold:
            stable[start : end + 1] = [False] * (end - start + 1)
    return stable


def segments_from_flags(flags: list[bool], fps: float) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for number, (start, end) in enumerate(_runs(flags, True), 1):
        segments.append(
            {
                "segment_id": f"main_{number:03d}",
                "start_frame": start,
                "end_frame": end,
                "frame_count": end - start + 1,
                "start_sec": round(start / fps, 3),
                "end_sec": round((end + 1) / fps, 3),
                "duration_sec": round((end - start + 1) / fps, 3),
            }
        )
    return segments


def _read_template(cv2: Any, path: Path, width: int, height: int) -> Any:
    # imdecode supports non-ASCII Windows paths more reliably than imread.
    import numpy as np

    payload = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(payload, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read template image: {path}")
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def run(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python-headless is required") from exc

    capture = cv2.VideoCapture(str(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {args.input}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or total_frames <= 0 or source_width <= 0 or source_height <= 0:
        raise RuntimeError("video metadata is incomplete")

    analysis_width = min(source_width, int(args.analysis_width))
    analysis_height = max(1, round(source_height * analysis_width / source_width))
    template = _read_template(cv2, args.template, analysis_width, analysis_height)

    started = perf_counter()
    scores: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.resize(frame, (analysis_width, analysis_height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        # This is the same rule as Good-Badminton.system.is_court_view.
        score = float(cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED).max())
        scores.append(score)
    capture.release()

    effective_threshold = adaptive_threshold(scores, args.threshold, args.adaptive_margin)
    raw_flags = [score >= effective_threshold for score in scores]
    stable_flags = stabilize_views(
        raw_flags,
        court_frames_threshold=args.court_frames_threshold,
        non_court_frames_threshold=args.non_court_frames_threshold,
    )
    segments = segments_from_flags(stable_flags, fps)
    selected_frames = sum(stable_flags)

    written = 0
    if not args.manifest_only:
        if args.output_video is None:
            raise ValueError("--output-video is required unless --manifest-only is used")
        args.output_video.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (source_width, source_height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot create output video: {args.output_video}")
        capture = cv2.VideoCapture(str(args.input))
        for keep in stable_flags:
            ok, frame = capture.read()
            if not ok:
                break
            if keep:
                writer.write(frame)
                written += 1
        capture.release()
        writer.release()
        if written != selected_frames:
            raise RuntimeError(f"expected to write {selected_frames} frames, wrote {written}")

    elapsed = perf_counter() - started
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "method": {
            "source": "Good-Badminton",
            "rule": "cv2.TM_CCOEFF_NORMED grayscale template matching",
            "threshold": args.threshold,
            "effective_threshold": effective_threshold,
            "adaptive_threshold": effective_threshold != args.threshold,
            "court_frames_threshold": args.court_frames_threshold,
            "non_court_frames_threshold": args.non_court_frames_threshold,
            "analysis_size": [analysis_width, analysis_height],
        },
        "input": str(args.input.resolve()),
        "template": str(args.template.resolve()),
        "output_video": None if args.manifest_only else str(args.output_video.resolve()),
        "output_mode": "source_frame_manifest" if args.manifest_only else "copied_silent_video",
        "source_video_is_not_copied": bool(args.manifest_only),
        "audio": "source_reference_only" if args.manifest_only else "not_included",
        "fps": fps,
        "source_frame_count": len(scores),
        "source_duration_sec": round(len(scores) / fps, 3),
        "selected_frame_count": selected_frames,
        "selected_duration_sec": round(selected_frames / fps, 3),
        "selected_ratio": round(selected_frames / max(1, len(scores)), 6),
        "segment_count": len(segments),
        "segments": segments,
        "score_summary": {
            "min": round(min(scores), 6) if scores else None,
            "max": round(max(scores), 6) if scores else None,
            "mean": round(sum(scores) / len(scores), 6) if scores else None,
        },
        "elapsed_sec": round(elapsed, 3),
        "processing_fps": round(len(scores) / max(elapsed, 1e-9), 3),
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--output-video", type=Path)
    parser.add_argument("--output-manifest", required=True, type=Path)
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--adaptive-margin", type=float, default=0.012)
    parser.add_argument("--analysis-width", type=int, default=320)
    parser.add_argument("--court-frames-threshold", type=int, default=5)
    parser.add_argument("--non-court-frames-threshold", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("--threshold must be in [0, 1]")
    if args.analysis_width < 32:
        raise SystemExit("--analysis-width must be at least 32")
    print(json.dumps(run(args), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
