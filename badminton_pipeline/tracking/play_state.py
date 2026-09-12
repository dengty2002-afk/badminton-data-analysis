"""Attach auditable active-play evidence to hit candidates without filtering them."""

from __future__ import annotations

import math
from typing import Any


def _ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else numerator / denominator


def _shuttle_point(record: dict[str, Any]) -> tuple[float, float] | None:
    value = record.get("shuttle", {}).get("smoothed_xy")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return float(value[0]), float(value[1])


def annotate_play_state_evidence(
    records: list[dict[str, Any]],
    events: list[dict[str, Any]],
    *,
    fps: float,
    window_seconds: float = 1.0,
    candidate_support_seconds: float = 3.0,
) -> list[dict[str, Any]]:
    """Add score-only play-state evidence; never remove or relabel an event."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if candidate_support_seconds <= 0:
        raise ValueError("candidate_support_seconds must be positive")
    radius = max(1, round(fps * window_seconds / 2))
    support_radius = max(1, round(fps * candidate_support_seconds))
    ordered_records = sorted(records, key=lambda row: int(row["frame_idx"]))
    event_frames = [int(event["candidate_frame"]) for event in events]

    for event in events:
        frame = int(event["candidate_frame"])
        window = [row for row in ordered_records if abs(int(row["frame_idx"]) - frame) <= radius]
        states = [str(row.get("shuttle", {}).get("tracking_state", "missing")) for row in window]
        measured_ratio = _ratio(states.count("measured"), len(states))
        tracked_ratio = _ratio(sum(state in {"measured", "predicted"} for state in states), len(states))

        valid_pairs = 0
        moving_pairs = 0
        for previous, following in zip(window, window[1:]):
            p0, p1 = _shuttle_point(previous), _shuttle_point(following)
            if p0 is None or p1 is None:
                continue
            valid_pairs += 1
            diagonal = math.hypot(float(following["width"]), float(following["height"]))
            moving_pairs += math.hypot(p1[0] - p0[0], p1[1] - p0[1]) >= diagonal * 0.0015
        moving_ratio = _ratio(moving_pairs, valid_pairs)

        both_players = 0
        for row in window:
            tracks = row.get("player_tracks", {})
            both_players += all(
                tracks.get(identity, {}).get("state") in {"measured", "predicted"}
                for identity in ("upper", "lower")
            )
        both_player_ratio = _ratio(both_players, len(window))
        nearby_candidates = sum(abs(other - frame) <= support_radius for other in event_frames)
        sequence_support = min(1.0, max(0, nearby_candidates - 1) / 2.0)

        score = (
            0.45 * measured_ratio
            + 0.20 * tracked_ratio
            + 0.20 * moving_ratio
            + 0.10 * both_player_ratio
            + 0.05 * sequence_support
        )
        event.update({
            "play_state_version": "play-state-evidence-0.1.0",
            "play_state_score": round(score, 6),
            "play_state_filter_applied": False,
            "play_state_decision": "score_only_unreviewed",
            "play_state_window_seconds": float(window_seconds),
            "play_state_window_frames": len(window),
            "play_state_window_radius_frames": radius,
            "play_state_support_seconds": float(candidate_support_seconds),
            "play_state_support_radius_frames": support_radius,
            "play_state_shuttle_measured_ratio": round(measured_ratio, 6),
            "play_state_shuttle_tracked_ratio": round(tracked_ratio, 6),
            "play_state_shuttle_moving_ratio": round(moving_ratio, 6),
            "play_state_both_player_tracking_ratio": round(both_player_ratio, 6),
            "play_state_nearby_candidate_count": nearby_candidates,
            "play_state_sequence_support": round(sequence_support, 6),
        })
    return events
