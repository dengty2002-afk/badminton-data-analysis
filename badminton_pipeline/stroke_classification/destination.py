"""Conservative automatic stroke-destination prelabels.

For a returned stroke, ShuttleSet's destination is approximated by the next
contact's shuttle observation projected to the court plane. This is usable for
screening and annotation assistance but remains a monocular approximation:
the shuttle is normally above the floor at contact. Terminal ground contacts
are emitted only as low-confidence experimental prelabels when a reliable
measured track visibly ends inside the calibrated court.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from badminton_pipeline.calibration.homography import project_point


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _frame_point(frame: dict[str, Any]) -> tuple[list[float] | None, float]:
    shuttle = frame.get("shuttle", {})
    point = shuttle.get("associated_xy") or shuttle.get("smoothed_xy")
    if not isinstance(point, list) or len(point) != 2:
        return None, 0.0
    measured = shuttle.get("tracking_state") == "measured"
    reliability = float(shuttle.get("trajectory_reliability") or 0.0)
    innovation = float(shuttle.get("innovation_ratio") or 0.0)
    confidence = max(0.0, min(1.0, reliability * math.exp(-innovation) * (1.0 if measured else 0.55)))
    return [float(point[0]), float(point[1])], confidence


def _receiver_position(frame: dict[str, Any], identity: str) -> tuple[list[float] | None, float]:
    if identity not in {"upper", "lower"}:
        return None, 0.0
    track = frame.get("player_tracks", {}).get(identity, {})
    point = track.get("court_xy_m")
    state = str(track.get("state") or "")
    if point is None:
        person = next(
            (row for row in frame.get("players", []) if row.get("identity") == identity), None
        )
        point = None if person is None else person.get("court_xy_m")
        state = "measured" if point is not None else state
    if not isinstance(point, list) or len(point) != 2:
        return None, 0.0
    confidence = 0.62 if state == "measured" else 0.42
    return [float(point[0]), float(point[1])], confidence


def _terminal_ground_contact(
    ordered_frames: list[dict[str, Any]],
    event: dict[str, Any],
    homography: list[list[float]],
    *,
    search_frames: int,
) -> dict[str, Any] | None:
    """Return a conservative track-end proxy, never an asserted truth label."""
    hit_frame = int(event["candidate_frame"])
    segment_id = event.get("play_state_segment_id")
    candidates: list[tuple[int, list[float], float, float, float]] = []
    later_frames: list[dict[str, Any]] = []
    entered_segment = False
    segment_ended = False
    for frame in ordered_frames:
        frame_idx = int(frame["frame_idx"])
        if frame_idx <= hit_frame or frame_idx > hit_frame + search_frames:
            continue
        frame_segment = frame.get("play_state_segment_id")
        if segment_id is not None:
            if frame_segment == segment_id:
                entered_segment = True
            elif entered_segment:
                segment_ended = True
                break
            else:
                continue
        later_frames.append(frame)
        if frame_idx < hit_frame + 4:
            continue
        point, confidence = _frame_point(frame)
        if point is None or confidence < 0.35:
            continue
        x, y = project_point(homography, (point[0], point[1]))
        if 0.0 <= x <= 6.1 and 0.0 <= y <= 13.4:
            candidates.append((frame_idx, point, confidence, x, y))
    if len(candidates) < 2:
        return None
    last_frame, last_point, reliability, x, y = candidates[-1]
    tail = [frame for frame in later_frames if int(frame["frame_idx"]) > last_frame]
    missing_tail = 0
    for frame in tail[:8]:
        point, _ = _frame_point(frame)
        if point is None:
            missing_tail += 1
        else:
            break
    # A clip ending is not evidence that the shuttle hit the floor. Require an
    # explicit active-play boundary or a short run of missing measurements.
    if not segment_ended and missing_tail < 3:
        return None
    recent_image_y = [item[1][1] for item in candidates[-5:]]
    if last_point[1] + 5.0 < max(recent_image_y):
        return None
    confidence = min(0.5, 0.2 + 0.25 * reliability + (0.08 if segment_ended else 0.0))
    return {
        "predicted_landing_x": round(x, 6),
        "predicted_landing_y": round(y, 6),
        "landing_frame": last_frame,
        "landing_kind": "ground_contact",
        "landing_confidence": round(confidence, 6),
        "landing_status": "experimental_terminal_track_end",
        "landing_source": "automatic_terminal_track_end_v1",
        "landing_proxy_kind": "shuttle_track_end_projection",
    }


def add_destination_predictions(
    frames: Iterable[dict[str, Any]],
    events: Iterable[dict[str, Any]],
    homography: list[list[float]],
    *,
    search_radius_frames: int = 3,
    terminal_search_frames: int = 75,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ordered_frames = sorted((dict(row) for row in frames), key=lambda row: int(row["frame_idx"]))
    frame_by_index = {int(row["frame_idx"]): row for row in ordered_frames}
    rows = sorted((dict(row) for row in events), key=lambda row: int(row["candidate_frame"]))
    predicted = 0
    terminal_unavailable = 0
    terminal_predicted = 0
    out_of_bounds = 0
    for index, row in enumerate(rows):
        rally = str(row.get("rally_id") or "")
        following = next(
            (
                item for item in rows[index + 1 :]
                if str(item.get("rally_id") or "") == rally
            ),
            None,
        )
        if following is None:
            terminal = _terminal_ground_contact(
                ordered_frames, row, homography, search_frames=terminal_search_frames
            )
            if terminal is not None:
                row.update(terminal)
                terminal_predicted += 1
            else:
                row.update({
                    "predicted_landing_x": None,
                    "predicted_landing_y": None,
                    "landing_frame": None,
                    "landing_kind": "ground_contact",
                    "landing_confidence": 0.0,
                    "landing_status": "unavailable_terminal_ground_contact",
                    "landing_source": "automatic_destination_v1",
                    "landing_proxy_kind": "unavailable",
                })
                terminal_unavailable += 1
            continue
        target = int(following["candidate_frame"])
        candidates: list[tuple[float, int, list[float]]] = []
        for offset in range(-search_radius_frames, search_radius_frames + 1):
            frame = frame_by_index.get(target + offset)
            if frame is None:
                continue
            point, confidence = _frame_point(frame)
            if point is not None:
                candidates.append((confidence - 0.02 * abs(offset), target + offset, point))
        receiver_identity = str(
            following.get("human_hitter") or following.get("predicted_hitter") or ""
        )
        contact_frame = frame_by_index.get(target, {})
        receiver_point, receiver_confidence = _receiver_position(contact_frame, receiver_identity)
        if not candidates and receiver_point is None:
            row.update(
                {
                    "predicted_landing_x": None,
                    "predicted_landing_y": None,
                    "landing_frame": target,
                    "landing_kind": "next_contact",
                    "landing_confidence": 0.0,
                    "landing_status": "unavailable_missing_shuttle",
                    "landing_source": "automatic_destination_v1",
                }
            )
            continue
        score, landing_frame, point = max(candidates, key=lambda item: item[0]) if candidates else (0.0, target, [0.0, 0.0])
        x, y = project_point(homography, (point[0], point[1])) if candidates else (math.nan, math.nan)
        ball_in_bounds = 0.0 <= x <= 6.1 and 0.0 <= y <= 13.4
        proxy_kind = "shuttle_contact_projection"
        if receiver_point is not None:
            ball_receiver_distance = math.hypot(x - receiver_point[0], y - receiver_point[1]) if ball_in_bounds else math.inf
            if ball_receiver_distance <= 1.8:
                # The shuttle projection is height-biased; the receiver's floor
                # position stabilizes it while retaining the observed direction.
                x = 0.6 * x + 0.4 * receiver_point[0]
                y = 0.6 * y + 0.4 * receiver_point[1]
                confidence = min(max(0.0, score) * 0.65, receiver_confidence)
                proxy_kind = "shuttle_receiver_fusion"
            else:
                x, y = receiver_point
                confidence = receiver_confidence * 0.7
                proxy_kind = "receiver_position_fallback"
        else:
            confidence = max(0.0, min(1.0, score)) * (0.55 if ball_in_bounds else 0.0)
        in_bounds = 0.0 <= x <= 6.1 and 0.0 <= y <= 13.4
        row.update(
            {
                "predicted_landing_x": round(x, 6) if in_bounds else None,
                "predicted_landing_y": round(y, 6) if in_bounds else None,
                "landing_frame": landing_frame,
                "landing_kind": "next_contact",
                "landing_confidence": round(confidence, 6),
                "landing_status": "predicted_monocular_proxy" if in_bounds else "rejected_out_of_bounds",
                "landing_source": "automatic_next_contact_projection_v1",
                "landing_proxy_kind": proxy_kind,
            }
        )
        if in_bounds:
            predicted += 1
        else:
            out_of_bounds += 1
    summary = {
        "schema_version": "destination-prediction-1.0",
        "event_count": len(rows),
        "predicted_next_contact": predicted,
        "predicted_terminal_ground_contact": terminal_predicted,
        "terminal_ground_contact_unavailable": terminal_unavailable,
        "rejected_out_of_bounds": out_of_bounds,
        "truth_status": "automatic monocular prelabels, not ground truth",
        "terminal_policy": "low-confidence track-end proxy only when measured termination evidence exists; otherwise null",
    }
    return rows, summary


def annotate_destinations(
    frames_path: Path,
    events_path: Path,
    calibration_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    rows, summary = add_destination_predictions(
        _read_jsonl(frames_path), _read_jsonl(events_path), calibration["homography"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return {**summary, "output": str(output_path.resolve())}
