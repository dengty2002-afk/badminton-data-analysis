"""Versioned four-corner court calibration using only the standard library."""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


Point = tuple[float, float]
COURT_POINTS: tuple[Point, ...] = ((0.0, 0.0), (6.1, 0.0), (6.1, 13.4), (0.0, 13.4))


def solve_linear_system(rows: Sequence[Sequence[float]]) -> list[float]:
    size = len(rows)
    matrix = [list(row) for row in rows]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(matrix[row][column]))
        if abs(matrix[pivot][column]) < 1e-10:
            raise ValueError("Degenerate corner geometry; homography is not solvable")
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        divisor = matrix[column][column]
        matrix[column] = [value / divisor for value in matrix[column]]
        for row in range(size):
            if row == column:
                continue
            factor = matrix[row][column]
            for index in range(column, size + 1):
                matrix[row][index] -= factor * matrix[column][index]
    return [matrix[row][size] for row in range(size)]


def compute_homography(source: Sequence[Point]) -> list[list[float]]:
    if len(source) != 4:
        raise ValueError("Exactly four court corners are required")
    rows: list[list[float]] = []
    for (x, y), (u, v) in zip(source, COURT_POINTS, strict=True):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, u])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, v])
    values = solve_linear_system(rows)
    return [values[0:3], values[3:6], [values[6], values[7], 1.0]]


def project_point(matrix: Sequence[Sequence[float]], point: Point) -> Point:
    x, y = point
    denominator = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if abs(denominator) < 1e-12:
        raise ValueError("Point projects to infinity")
    return (
        (matrix[0][0] * x + matrix[0][1] * y + matrix[0][2]) / denominator,
        (matrix[1][0] * x + matrix[1][1] * y + matrix[1][2]) / denominator,
    )


def validate_corners(points: Sequence[Point], width: int, height: int) -> tuple[bool, float, str]:
    if len(points) != 4:
        return False, 0.0, "Exactly four court corners are required"
    if width <= 0 or height <= 0:
        return False, 0.0, "Image dimensions must be positive"
    if any(not math.isfinite(x) or not math.isfinite(y) or x < 0 or y < 0 or x > width or y > height for x, y in points):
        return False, 0.0, "All corners must be inside the image"

    signed_area = sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[(index + 1) % 4][0] * points[index][1]
        for index in range(4)
    ) / 2.0
    area_ratio = abs(signed_area) / (width * height)
    crosses = []
    for index in range(4):
        point, nxt, after = points[index], points[(index + 1) % 4], points[(index + 2) % 4]
        crosses.append((nxt[0] - point[0]) * (after[1] - nxt[1]) - (nxt[1] - point[1]) * (after[0] - nxt[0]))
    if not (all(value > 0 for value in crosses) or all(value < 0 for value in crosses)):
        return False, area_ratio, "Court corners cross or form a concave quadrilateral"
    if area_ratio < 0.015:
        return False, area_ratio, "Court quadrilateral is too small"
    if (points[0][1] + points[1][1]) / 2 >= (points[2][1] + points[3][1]) / 2:
        return False, area_ratio, "Upper corners must be above lower corners"
    return True, area_ratio, "accepted"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def parse_corners(value: str) -> list[Point]:
    points: list[Point] = []
    for item in value.split(";"):
        x, y = item.split(",", 1)
        points.append((float(x), float(y)))
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description="Save a versioned manual court calibration.")
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--corners", required=True, help="ULx,ULy;URx,URy;LRx,LRy;LLx,LLy")
    parser.add_argument("--width", required=True, type=int)
    parser.add_argument("--height", required=True, type=int)
    parser.add_argument("--frame-time-ms", default=0, type=int)
    parser.add_argument("--data-dir", default=Path("data"), type=Path)
    args = parser.parse_args()

    corners = parse_corners(args.corners)
    valid, area_ratio, message = validate_corners(corners, args.width, args.height)
    if not valid:
        parser.error(message)
    homography = compute_homography(corners)
    output_dir = args.data_dir.resolve() / "calibrations" / args.video_id
    existing = sorted(output_dir.glob("v*.json")) if output_dir.exists() else []
    version = max((int(path.stem.removeprefix("v")) for path in existing), default=0) + 1
    record = {
        "calibration_id": f"cal_{uuid.uuid4()}",
        "video_id": args.video_id,
        "version": version,
        "frame_time_ms": max(0, args.frame_time_ms),
        "image_width": args.width,
        "image_height": args.height,
        "corners": [{"x": x, "y": y} for x, y in corners],
        "homography": homography,
        "court_orientation": "upper_is_far_side",
        "quality_status": "accepted",
        "quality": {"area_ratio": area_ratio},
        "source": "manual",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    output = output_dir / f"v{version}.json"
    write_json(output, record)
    print(json.dumps({"status": "saved", "version": version, "path": str(output), "homography": homography}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
