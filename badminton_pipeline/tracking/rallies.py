"""Propose rally boundaries from the temporal spacing of hit candidates."""

from __future__ import annotations

from typing import Any


def assign_rally_ids(
    events: list[dict[str, Any]],
    *,
    gap_seconds: float = 6.0,
) -> list[dict[str, Any]]:
    """Assign stable candidate rally IDs without claiming Gold boundaries.

    A new rally candidate starts after a long event-free gap. The result is a
    proposal for annotation and intentionally records the rule that created it.
    """
    if gap_seconds <= 0:
        raise ValueError("gap_seconds must be positive")
    ordered = sorted(events, key=lambda event: (float(event["candidate_time"]), int(event["candidate_frame"])))
    rally_number = 0
    previous_time: float | None = None
    for event in ordered:
        current_time = float(event["candidate_time"])
        gap = None if previous_time is None else current_time - previous_time
        starts_rally = previous_time is None or (gap is not None and gap > gap_seconds)
        if starts_rally:
            rally_number += 1
        event["rally_id"] = f"Rally {rally_number:02d}"
        event["rally_boundary_source"] = "first_event" if previous_time is None else ("time_gap" if starts_rally else "continuation")
        event["previous_hit_gap_sec"] = None if gap is None else round(gap, 3)
        event["rally_boundary_status"] = "candidate"
        previous_time = current_time
    return events
