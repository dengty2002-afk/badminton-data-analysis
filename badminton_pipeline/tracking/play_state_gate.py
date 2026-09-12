"""Build auditable active-play intervals from alternating hit chains and shuttle flight."""

from __future__ import annotations

import math
from typing import Any


GATE_VERSION = "play-state-gate-0.1.0-dev"


def _point(record: dict[str, Any]) -> tuple[float, float] | None:
    value = record.get("shuttle", {}).get("smoothed_xy")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return float(value[0]), float(value[1])


def _flight_evidence(
    records: list[dict[str, Any]],
    start_frame: int,
    end_frame: int,
) -> dict[str, float | int]:
    window = [
        row for row in records
        if start_frame <= int(row["frame_idx"]) <= end_frame
    ]
    if not window:
        return {"frame_count": 0, "tracked_ratio": 0.0, "moving_ratio": 0.0}
    tracked = sum(
        row.get("shuttle", {}).get("tracking_state") in {"measured", "predicted"}
        for row in window
    )
    valid_pairs = 0
    moving_pairs = 0
    for previous, following in zip(window, window[1:]):
        p0, p1 = _point(previous), _point(following)
        if p0 is None or p1 is None:
            continue
        valid_pairs += 1
        diagonal = max(
            1.0,
            math.hypot(float(following.get("width") or 0.0), float(following.get("height") or 0.0)),
        )
        moving_pairs += math.hypot(p1[0] - p0[0], p1[1] - p0[1]) >= diagonal * 0.0015
    return {
        "frame_count": len(window),
        "tracked_ratio": tracked / len(window),
        "moving_ratio": 0.0 if valid_pairs == 0 else moving_pairs / valid_pairs,
    }


def _merge_intervals(intervals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for interval in sorted(intervals, key=lambda item: (item["start_frame"], item["end_frame"])):
        if not merged or interval["start_frame"] > merged[-1]["end_frame"] + 1:
            merged.append({
                **interval,
                "event_ids": list(interval["event_ids"]),
                "link_count": 1,
            })
            continue
        current = merged[-1]
        current["end_frame"] = max(current["end_frame"], interval["end_frame"])
        current["event_ids"] = sorted(set(current["event_ids"] + interval["event_ids"]))
        current["link_count"] += 1
        current["tracked_ratio_min"] = min(current["tracked_ratio_min"], interval["tracked_ratio_min"])
        current["moving_ratio_min"] = min(current["moving_ratio_min"], interval["moving_ratio_min"])
    for index, interval in enumerate(merged, 1):
        interval["segment_id"] = f"Active {index:02d}"
    return merged


def apply_play_state_gate(
    records: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    fps: float,
    enabled: bool = True,
    min_link_seconds: float = 0.25,
    max_link_seconds: float = 4.0,
    pre_roll_seconds: float = 0.15,
    post_roll_seconds: float = 0.35,
    min_tracked_ratio: float = 0.55,
    min_moving_ratio: float = 0.25,
    min_sequence_support: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Gate provisional candidates; suppressed candidates remain available for audit."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if min_link_seconds <= 0 or max_link_seconds <= min_link_seconds:
        raise ValueError("link seconds must satisfy 0 < min < max")
    if pre_roll_seconds < 0 or post_roll_seconds < 0:
        raise ValueError("roll seconds cannot be negative")
    if not all(0 <= value <= 1 for value in (min_tracked_ratio, min_moving_ratio, min_sequence_support)):
        raise ValueError("flight and sequence ratios must be between 0 and 1")

    ordered_records = sorted(records, key=lambda row: int(row["frame_idx"]))
    ordered_events = sorted(events, key=lambda row: int(row["candidate_frame"]))
    if not enabled:
        for row in ordered_records:
            row.update({
                "play_state_gate_version": GATE_VERSION,
                "play_state": "not_gated",
                "play_state_segment_id": None,
            })
        for event in ordered_events:
            event.update({
                "play_state_gate_version": GATE_VERSION,
                "play_state_gate_applied": False,
                "play_state_gate_decision": "gate_disabled",
                "play_state_segment_id": None,
            })
        return ordered_events, [], {
            "version": GATE_VERSION,
            "enabled": False,
            "provisional_count": len(ordered_events),
            "kept_count": len(ordered_events),
            "suppressed_count": 0,
            "segments": [],
        }

    min_gap = max(1, round(fps * min_link_seconds))
    max_gap = max(min_gap + 1, round(fps * max_link_seconds))
    pre_roll = round(fps * pre_roll_seconds)
    post_roll = round(fps * post_roll_seconds)
    intervals: list[dict[str, Any]] = []
    links: list[dict[str, Any]] = []
    for left_index, left in enumerate(ordered_events):
        left_frame = int(left["candidate_frame"])
        left_hitter = left.get("predicted_hitter")
        left_support = float(left.get("play_state_sequence_support", 1.0))
        if left_hitter not in {"upper", "lower"} or left_support < min_sequence_support:
            continue
        for right in ordered_events[left_index + 1:]:
            right_frame = int(right["candidate_frame"])
            gap = right_frame - left_frame
            if gap > max_gap:
                break
            right_support = float(right.get("play_state_sequence_support", 1.0))
            if (
                gap < min_gap
                or right.get("predicted_hitter") == left_hitter
                or right_support < min_sequence_support
            ):
                continue
            evidence = _flight_evidence(ordered_records, left_frame, right_frame)
            if (
                float(evidence["tracked_ratio"]) < min_tracked_ratio
                or float(evidence["moving_ratio"]) < min_moving_ratio
            ):
                continue
            link = {
                "left_event_id": left.get("event_id"),
                "right_event_id": right.get("event_id"),
                "start_frame": left_frame,
                "end_frame": right_frame,
                "gap_frames": gap,
                **evidence,
            }
            links.append(link)
            intervals.append({
                "start_frame": max(0, left_frame - pre_roll),
                "end_frame": right_frame + post_roll,
                "event_ids": [str(left.get("event_id")), str(right.get("event_id"))],
                "tracked_ratio_min": float(evidence["tracked_ratio"]),
                "moving_ratio_min": float(evidence["moving_ratio"]),
            })

    segments = _merge_intervals(intervals)

    def segment_for_frame(frame: int) -> dict[str, Any] | None:
        return next(
            (segment for segment in segments if segment["start_frame"] <= frame <= segment["end_frame"]),
            None,
        )

    for row in ordered_records:
        segment = segment_for_frame(int(row["frame_idx"]))
        row.update({
            "play_state_gate_version": GATE_VERSION,
            "play_state": "active_play" if segment else "inactive_wait",
            "play_state_segment_id": None if segment is None else segment["segment_id"],
        })

    kept: list[dict[str, Any]] = []
    suppressed: list[dict[str, Any]] = []
    for event in ordered_events:
        segment = segment_for_frame(int(event["candidate_frame"]))
        event.update({
            "play_state_gate_version": GATE_VERSION,
            "play_state_gate_applied": True,
            "play_state_gate_decision": "kept_active_play" if segment else "suppressed_inactive_wait",
            "play_state_segment_id": None if segment is None else segment["segment_id"],
        })
        (kept if segment else suppressed).append(event)

    return kept, suppressed, {
        "version": GATE_VERSION,
        "enabled": True,
        "policy": "sequence_supported_opposite_hitter_links_with_continuous_shuttle_flight",
        "parameters": {
            "min_link_seconds": min_link_seconds,
            "max_link_seconds": max_link_seconds,
            "pre_roll_seconds": pre_roll_seconds,
            "post_roll_seconds": post_roll_seconds,
            "min_tracked_ratio": min_tracked_ratio,
            "min_moving_ratio": min_moving_ratio,
            "min_sequence_support": min_sequence_support,
        },
        "provisional_count": len(ordered_events),
        "kept_count": len(kept),
        "suppressed_count": len(suppressed),
        "link_count": len(links),
        "segments": segments,
    }
