"""Propose and optionally accept a four-corner court calibration.

The automatic detector is loaded from the vendored Apache-2.0 Good-Badminton
implementation.  Automatic output is a proposal by default: a formal versioned
calibration is written only with ``--accept``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .homography import compute_homography, parse_corners, validate_corners, write_json


Point = tuple[float, float]


def _load_cv2():
    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("opencv and numpy are required for court detection") from exc
    return cv2, np


def _read_image(path: Path):
    cv2, np = _load_cv2()
    image = cv2.imread(str(path))
    if image is None and path.is_file():
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return image


def extract_reference_frame(input_path: Path, frame_time_ms: int = 0):
    """Read an image or seek a video to a representative frame."""
    cv2, _np = _load_cv2()
    input_path = input_path.resolve()
    if input_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = _read_image(input_path)
        if image is None:
            raise ValueError(f"cannot read image: {input_path}")
        return image, 0
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {input_path}")
    requested = max(0, int(frame_time_ms))
    capture.set(cv2.CAP_PROP_POS_MSEC, requested)
    ok, image = capture.read()
    actual = int(round(capture.get(cv2.CAP_PROP_POS_MSEC)))
    if not ok and requested:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, image = capture.read()
        actual = 0
    capture.release()
    if not ok or image is None:
        raise ValueError(f"cannot decode a frame from: {input_path}")
    return image, actual


def _detector(vendor_root: Path | None = None):
    root = (vendor_root or Path("vendor/Good-Badminton")).resolve()
    if not (root / "badminton_analysis" / "court" / "mapper.py").is_file():
        raise ValueError(f"Good-Badminton court detector not found under: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    module = importlib.import_module("badminton_analysis.court.mapper")
    return module.auto_detect_preview, root


def _expand_singles_quad_to_doubles(corners: list[Point]) -> list[Point]:
    """Treat a quad as the 5.18 m singles sidelines and extrapolate 6.10 m corners."""
    cv2, np = _load_cv2()
    width, length, margin = 6.1, 13.4, 0.46
    singles = np.array(
        [[margin, 0], [width - margin, 0], [width - margin, length], [margin, length]],
        dtype=np.float32,
    )
    doubles = np.array([[0, 0], [width, 0], [width, length], [0, length]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(singles, np.asarray(corners, dtype=np.float32))
    expanded = cv2.perspectiveTransform(doubles.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    return [(float(x), float(y)) for x, y in expanded]


def _boundary_semantics_audit(
    image, corners: list[Point], vendor_root: Path,
) -> dict[str, Any]:
    """Compare an outer-court interpretation with a singles-line interpretation.

    Good-Badminton can occasionally return the visually strong singles
    sidelines as the four court edges.  Since our canonical plane is 6.1 m
    wide, score the same pixels again after extrapolating those lines to the
    doubles boundary and conservatively flag a materially better fit.
    """
    root_text = str(vendor_root.resolve())
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    detector_module = importlib.import_module("badminton_analysis.court.detector")
    reference_module = importlib.import_module("badminton_analysis.court.reference")
    reference = reference_module.BadmintonCourtReference()
    line_mask = detector_module.build_reference_line_mask(image)
    support = reference.prepare_line_support(line_mask, image.shape)
    expanded = _expand_singles_quad_to_doubles(corners)
    outer_score, _outer_details = reference.score_line_support(corners, support, image.shape)
    expanded_score, _expanded_details = reference.score_line_support(expanded, support, image.shape)
    delta = float(expanded_score - outer_score)
    likely_singles = expanded_score >= 0.42 and delta >= 0.06
    return {
        "canonical_width_m": 6.1,
        "outer_interpretation_score": round(float(outer_score), 4),
        "expanded_from_singles_score": round(float(expanded_score), 4),
        "expanded_score_delta": round(delta, 4),
        "likely_singles_sidelines": bool(likely_singles),
        "expanded_doubles_corners": [{"x": round(x, 3), "y": round(y, 3)} for x, y in expanded],
    }


def _write_image(path: Path, image) -> None:
    cv2, _np = _load_cv2()
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"failed to encode preview: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded.tofile(str(temporary))
    os.replace(temporary, path)


def _render_preview(image, corners: list[Point] | None, status: str):
    cv2, np = _load_cv2()
    preview = image.copy()
    if corners:
        points = np.array(corners, dtype=np.int32)
        cv2.polylines(preview, [points], True, (0, 255, 0), 3, cv2.LINE_AA)
        for index, point in enumerate(points, start=1):
            xy = tuple(int(value) for value in point)
            cv2.circle(preview, xy, 7, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                preview, str(index), (xy[0] + 10, xy[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA,
            )
    cv2.rectangle(preview, (0, 0), (preview.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(
        preview, status, (14, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
        (80, 230, 255), 2, cv2.LINE_AA,
    )
    return preview


def _save_calibration(
    video_id: str,
    corners: list[Point],
    width: int,
    height: int,
    frame_time_ms: int,
    data_dir: Path,
    source: str,
    proposal_path: Path,
) -> Path:
    valid, area_ratio, message = validate_corners(corners, width, height)
    if not valid:
        raise ValueError(message)
    output_dir = data_dir.resolve() / "calibrations" / video_id
    existing = sorted(output_dir.glob("v*.json")) if output_dir.exists() else []
    version = max((int(path.stem.removeprefix("v")) for path in existing), default=0) + 1
    record = {
        "calibration_id": f"cal_{uuid.uuid4()}",
        "video_id": video_id,
        "version": version,
        "frame_time_ms": max(0, frame_time_ms),
        "image_width": width,
        "image_height": height,
        "corners": [{"x": x, "y": y} for x, y in corners],
        "homography": compute_homography(corners),
        "court_orientation": "upper_is_far_side",
        "quality_status": "accepted",
        "quality": {"area_ratio": area_ratio, "human_confirmed": source == "manual_correction"},
        "source": source,
        "proposal": str(proposal_path.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    output = output_dir / f"v{version}.json"
    write_json(output, record)
    return output


def propose_court(
    input_path: Path,
    video_id: str,
    output_dir: Path,
    *,
    frame_time_ms: int = 0,
    manual_corners: list[Point] | None = None,
    accept: bool = False,
    data_dir: Path = Path("data"),
    vendor_root: Path | None = None,
) -> dict[str, Any]:
    """Create an auditable proposal and optionally save an accepted calibration."""
    image, actual_time_ms = extract_reference_frame(input_path, frame_time_ms)
    height, width = image.shape[:2]
    detector_root: Path | None = None
    if manual_corners is not None:
        corners = [(float(x), float(y)) for x, y in manual_corners]
        method = "manual_correction"
    else:
        detect, detector_root = _detector(vendor_root)
        detected, _vendor_preview = detect(image)
        corners = None if not detected else [(float(x), float(y)) for x, y in detected]
        method = "good_badminton_auto"

    boundary_audit = None
    if corners:
        valid, area_ratio, message = validate_corners(corners, width, height)
        if valid and method == "good_badminton_auto" and detector_root is not None:
            boundary_audit = _boundary_semantics_audit(image, corners, detector_root)
            if boundary_audit["likely_singles_sidelines"]:
                valid = False
                message = (
                    "automatic quadrilateral likely follows singles sidelines; "
                    "review or correct to the 6.1 m doubles boundary"
                )
    else:
        valid, area_ratio, message = False, 0.0, "no reliable court quadrilateral detected"
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_path = output_dir / f"{video_id}.court_preview.png"
    proposal_path = output_dir / f"{video_id}.court_proposal.json"
    _write_image(
        preview_path,
        _render_preview(image, corners, f"{method}: {'VALID' if valid else 'REVIEW REQUIRED'}"),
    )
    proposal: dict[str, Any] = {
        "proposal_version": "court-proposal-v0.1",
        "video_id": video_id,
        "input": str(input_path.resolve()),
        "frame_time_ms": actual_time_ms,
        "image_width": width,
        "image_height": height,
        "method": method,
        "detector_source": None if detector_root is None else str(detector_root),
        "corners": None if corners is None else [{"x": x, "y": y} for x, y in corners],
        "validation": {"valid": valid, "area_ratio": area_ratio, "message": message},
        "boundary_audit": boundary_audit,
        "preview": str(preview_path),
        "accepted": False,
        "calibration": None,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_json(proposal_path, proposal)
    if accept:
        if not valid or corners is None:
            raise ValueError(f"cannot accept proposal: {message}")
        calibration_path = _save_calibration(
            video_id, corners, width, height, actual_time_ms, data_dir,
            "manual_correction" if manual_corners is not None else "good_badminton_auto_accepted",
            proposal_path,
        )
        proposal["accepted"] = True
        proposal["calibration"] = str(calibration_path)
        proposal["accepted_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_json(proposal_path, proposal)
    proposal["proposal"] = str(proposal_path)
    return proposal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Video or reference image")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--frame-time-ms", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, default=Path("data/court_proposals"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--vendor-root", type=Path)
    parser.add_argument("--manual-corners", help="ULx,ULy;URx,URy;LRx,LRy;LLx,LLy")
    parser.add_argument(
        "--accept", action="store_true",
        help="Write a versioned calibration after explicit human review of the preview.",
    )
    args = parser.parse_args()
    try:
        result = propose_court(
            args.input, args.video_id, args.output_dir,
            frame_time_ms=args.frame_time_ms,
            manual_corners=None if args.manual_corners is None else parse_corners(args.manual_corners),
            accept=args.accept, data_dir=args.data_dir, vendor_root=args.vendor_root,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
