"""Project players to court coordinates and propose likely hit frames."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

from badminton_pipeline.calibration.homography import project_point
from badminton_pipeline.tracking.play_state import annotate_play_state_evidence
from badminton_pipeline.tracking.play_state_gate import apply_play_state_gate
from badminton_pipeline.tracking.rallies import assign_rally_ids


Point = tuple[float, float]
SHUTTLE_TRACK_VERSION = "shuttle-track-0.2.0-dev"


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _valid_keypoint(player: dict[str, Any], index: int, threshold: float = 0.2) -> Point | None:
    scores = player.get("keypoint_scores", [])
    points = player.get("keypoints_xy", [])
    missing = player.get("keypoint_missing", [])
    if index >= len(scores) or index >= len(points) or index >= len(missing):
        return None
    if missing[index] or float(scores[index]) < threshold:
        return None
    return float(points[index][0]), float(points[index][1])


def player_foot_point(player: dict[str, Any]) -> Point:
    ankles = [point for index in (15, 16) if (point := _valid_keypoint(player, index)) is not None]
    if ankles:
        return sum(point[0] for point in ankles) / len(ankles), sum(point[1] for point in ankles) / len(ankles)
    x1, _, x2, y2 = (float(value) for value in player["bbox_xyxy"])
    return (x1 + x2) / 2.0, y2


def _shuttle_measurements(shuttle: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ranked detector candidates, falling back to legacy raw_xy records."""
    measurements: list[dict[str, Any]] = []
    candidates = shuttle.get("candidates")
    if isinstance(candidates, list):
        for rank, candidate in enumerate(candidates):
            point = candidate.get("raw_xy") if isinstance(candidate, dict) else None
            confidence = candidate.get("confidence") if isinstance(candidate, dict) else None
            if not isinstance(point, (list, tuple)) or len(point) < 2 or confidence is None:
                continue
            measurements.append({
                "rank": rank,
                "point": (float(point[0]), float(point[1])),
                "confidence": float(confidence),
                "bbox_xyxy": candidate.get("bbox_xyxy"),
                "detector_index": candidate.get("detector_index"),
            })
    if measurements:
        return measurements
    raw = shuttle.get("raw_xy")
    confidence = shuttle.get("confidence")
    if isinstance(raw, (list, tuple)) and len(raw) >= 2 and confidence is not None:
        return [{
            "rank": 0,
            "point": (float(raw[0]), float(raw[1])),
            "confidence": float(confidence),
            "bbox_xyxy": shuttle.get("bbox_xyxy"),
            "detector_index": 0,
        }]
    return []


def _smooth_shuttle(records: list[dict[str, Any]], fps: float, max_missing: int = 5) -> None:
    """Associate top-k detections and apply an auditable, hit-aware soft innovation gate."""
    position: Point | None = None
    velocity: Point = (0.0, 0.0)
    previous_frame: int | None = None
    missing_count = 0
    uncertainty_px: float | None = None
    alpha, beta = 0.55, 0.16

    for index, record in enumerate(records):
        frame_idx = int(record["frame_idx"])
        shuttle = record["shuttle"]
        diagonal = max(
            1.0,
            math.hypot(float(record.get("width") or 0.0), float(record.get("height") or 0.0)),
        )
        measurements = [item for item in _shuttle_measurements(shuttle) if item["confidence"] >= 0.18]
        shuttle["track_version"] = SHUTTLE_TRACK_VERSION
        shuttle["detector_candidate_count"] = len(measurements)
        frame_delta = 1 if previous_frame is None else max(1, frame_idx - previous_frame)
        if previous_frame is not None and frame_delta > max(3, round(fps * 0.5)):
            # Disjoint no-copy main-view segments must not share a ballistic
            # track across broadcast gaps.
            position = None
            velocity = (0.0, 0.0)
            missing_count = 0
            uncertainty_px = None
        dt = frame_delta / fps
        previous_frame = frame_idx

        if position is None:
            if not measurements:
                shuttle["smoothed_xy"] = None
                shuttle["tracking_state"] = "missing"
                shuttle["associated_xy"] = None
                shuttle["selected_candidate_rank"] = None
                shuttle["predicted_xy"] = None
                shuttle["innovation_px"] = None
                shuttle["innovation_ratio"] = None
                shuttle["adaptive_gate_px"] = None
                shuttle["continuity_score"] = None
                shuttle["hit_support_score"] = None
                shuttle["hand_support_score"] = None
                shuttle["pose_support_score"] = None
                shuttle["measurement_weight"] = 0.0
                shuttle["measurement_status"] = "no_measurement"
                shuttle["tracking_uncertainty_px"] = None
                shuttle["trajectory_reliability"] = 0.0
                continue
            chosen = max(measurements, key=lambda item: item["confidence"])
            position = chosen["point"]
            velocity = (0.0, 0.0)
            missing_count = 0
            uncertainty_px = diagonal * 0.02
            shuttle["smoothed_xy"] = [position[0], position[1]]
            shuttle["tracking_state"] = "measured"
            shuttle["associated_xy"] = [position[0], position[1]]
            shuttle["selected_candidate_rank"] = chosen["rank"]
            shuttle["predicted_xy"] = [position[0], position[1]]
            shuttle["innovation_px"] = 0.0
            shuttle["innovation_ratio"] = 0.0
            shuttle["adaptive_gate_px"] = None
            shuttle["continuity_score"] = 1.0
            shuttle["hit_support_score"] = 0.0
            shuttle["hand_support_score"] = 0.0
            shuttle["pose_support_score"] = 0.0
            shuttle["measurement_weight"] = 1.0
            shuttle["measurement_status"] = "initialized"
            shuttle["tracking_uncertainty_px"] = uncertainty_px
            shuttle["trajectory_reliability"] = 1.0
            continue

        predicted = (position[0] + velocity[0] * dt, position[1] + velocity[1] * dt)
        uncertainty_px = diagonal * 0.02 if uncertainty_px is None else uncertainty_px
        if measurements:
            pose_score = 0.0
            if 0 < index < len(records) - 1:
                pose_score = min(1.0, _pose_motion(records[index - 1], records[index + 1]) / (diagonal * 0.035))
            speed_step_ratio = math.hypot(*velocity) * dt / diagonal
            options: list[dict[str, Any]] = []
            for measurement in measurements:
                point = measurement["point"]
                residual = (point[0] - predicted[0], point[1] - predicted[1])
                innovation_px = math.hypot(*residual)
                hand_distance, _ = _wrist_distance(record, point)
                hand_score = (
                    0.0 if not math.isfinite(hand_distance)
                    else max(0.0, 1.0 - hand_distance / (diagonal * 0.09))
                )
                hit_support = max(hand_score, pose_score)
                gate_ratio = min(
                    0.22,
                    0.025 + 1.5 * speed_step_ratio + 0.08 * hit_support + uncertainty_px / diagonal,
                )
                gate_px = max(3.0, diagonal * gate_ratio)
                continuity = math.exp(-0.5 * (innovation_px / gate_px) ** 2)
                association_score = (
                    0.55 * measurement["confidence"]
                    + 0.35 * continuity
                    + 0.10 * hit_support
                )
                options.append({
                    **measurement,
                    "residual": residual,
                    "innovation_px": innovation_px,
                    "gate_px": gate_px,
                    "continuity": continuity,
                    "hit_support": hit_support,
                    "hand_support": hand_score,
                    "pose_support": pose_score,
                    "association_score": association_score,
                })
            chosen = max(options, key=lambda item: (item["association_score"], -item["rank"]))
            residual = chosen["residual"]
            continuity_weight = 0.05 + 0.95 * chosen["continuity"]
            hit_evidence_floor = min(
                1.0,
                0.05 + 0.75 * chosen["hand_support"] + 0.15 * chosen["pose_support"],
            )
            measurement_weight = max(continuity_weight, hit_evidence_floor)
            effective_alpha = alpha * (0.10 + 0.90 * measurement_weight)
            effective_beta = beta * (0.05 + 0.95 * measurement_weight)
            position = (
                predicted[0] + effective_alpha * residual[0],
                predicted[1] + effective_alpha * residual[1],
            )
            velocity = (
                velocity[0] + effective_beta * residual[0] / dt,
                velocity[1] + effective_beta * residual[1] / dt,
            )
            missing_count = 0
            accepted = measurement_weight >= 0.35
            shuttle["tracking_state"] = "measured" if accepted else "predicted"
            shuttle["associated_xy"] = [chosen["point"][0], chosen["point"][1]]
            shuttle["selected_candidate_rank"] = chosen["rank"]
            shuttle["predicted_xy"] = [predicted[0], predicted[1]]
            shuttle["innovation_px"] = chosen["innovation_px"]
            shuttle["innovation_ratio"] = chosen["innovation_px"] / diagonal
            shuttle["adaptive_gate_px"] = chosen["gate_px"]
            shuttle["continuity_score"] = chosen["continuity"]
            shuttle["hit_support_score"] = chosen["hit_support"]
            shuttle["hand_support_score"] = chosen["hand_support"]
            shuttle["pose_support_score"] = chosen["pose_support"]
            shuttle["association_score"] = chosen["association_score"]
            shuttle["measurement_weight"] = measurement_weight
            shuttle["measurement_status"] = "accepted" if accepted else "soft_suppressed"
            uncertainty_px = max(
                diagonal * 0.005,
                min(
                    diagonal * 0.25,
                    uncertainty_px * (0.90 - 0.35 * measurement_weight)
                    + diagonal * 0.004 * (1.0 - measurement_weight),
                ),
            )
            shuttle["trajectory_reliability"] = measurement_weight
        else:
            missing_count += frame_delta
            if missing_count > max_missing:
                position = None
                velocity = (0.0, 0.0)
                uncertainty_px = None
                shuttle["smoothed_xy"] = None
                shuttle["tracking_state"] = "missing"
                shuttle["associated_xy"] = None
                shuttle["selected_candidate_rank"] = None
                shuttle["predicted_xy"] = [predicted[0], predicted[1]]
                shuttle["innovation_px"] = None
                shuttle["innovation_ratio"] = None
                shuttle["adaptive_gate_px"] = None
                shuttle["continuity_score"] = None
                shuttle["hit_support_score"] = None
                shuttle["hand_support_score"] = None
                shuttle["pose_support_score"] = None
                shuttle["association_score"] = None
                shuttle["measurement_weight"] = 0.0
                shuttle["measurement_status"] = "reset_after_missing"
                shuttle["tracking_uncertainty_px"] = None
                shuttle["trajectory_reliability"] = 0.0
                continue
            position = predicted
            shuttle["tracking_state"] = "predicted"
            uncertainty_px = min(diagonal * 0.25, uncertainty_px + diagonal * 0.008 * frame_delta)
            shuttle["associated_xy"] = None
            shuttle["selected_candidate_rank"] = None
            shuttle["predicted_xy"] = [predicted[0], predicted[1]]
            shuttle["innovation_px"] = None
            shuttle["innovation_ratio"] = None
            shuttle["adaptive_gate_px"] = None
            shuttle["continuity_score"] = None
            shuttle["hit_support_score"] = None
            shuttle["hand_support_score"] = None
            shuttle["pose_support_score"] = None
            shuttle["association_score"] = None
            shuttle["measurement_weight"] = 0.0
            shuttle["measurement_status"] = "predicted_missing"
            shuttle["trajectory_reliability"] = max(0.2, 1.0 - missing_count / (max_missing + 1.0))
        shuttle["smoothed_xy"] = [position[0], position[1]]
        shuttle["tracking_uncertainty_px"] = uncertainty_px


def _track_players(
    records: list[dict[str, Any]],
    *,
    smoothing_alpha: float = 0.65,
    max_missing_frames: int = 5,
) -> None:
    tracks: dict[str, dict[str, Any]] = {}
    for record in records:
        frame_idx = int(record["frame_idx"])
        measured = {player["identity"]: player for player in record.get("players", [])}
        frame_tracks: dict[str, dict[str, Any]] = {}
        for identity in ("upper", "lower"):
            player = measured.get(identity)
            previous = tracks.get(identity)
            if player is not None:
                raw = player["court_xy_m"]
                position = (float(raw[0]), float(raw[1]))
                if previous is not None:
                    old = previous["position"]
                    position = (
                        smoothing_alpha * position[0] + (1.0 - smoothing_alpha) * old[0],
                        smoothing_alpha * position[1] + (1.0 - smoothing_alpha) * old[1],
                    )
                tracks[identity] = {
                    "position": position,
                    "source_frame": frame_idx,
                }
                frame_tracks[identity] = {
                    "court_xy_m": [position[0], position[1]],
                    "state": "measured",
                    "source_frame": frame_idx,
                }
                continue

            if previous is not None and frame_idx - int(previous["source_frame"]) <= max_missing_frames:
                position = previous["position"]
                frame_tracks[identity] = {
                    "court_xy_m": [position[0], position[1]],
                    "state": "predicted",
                    "source_frame": int(previous["source_frame"]),
                }
            else:
                frame_tracks[identity] = {
                    "court_xy_m": None,
                    "state": "missing",
                    "source_frame": None,
                }
        record["player_tracks"] = frame_tracks


def enrich_records(
    records: list[dict[str, Any]],
    homography: Sequence[Sequence[float]],
    *,
    fps: float,
) -> list[dict[str, Any]]:
    for record in records:
        candidates = record.get("players", [])
        for player in candidates:
            image_point = player_foot_point(player)
            court_point = project_point(homography, image_point)
            player["foot_image_xy"] = [image_point[0], image_point[1]]
            player["court_xy_m"] = [court_point[0], court_point[1]]
            player["court_in_bounds"] = 0.0 <= court_point[0] <= 6.1 and 0.0 <= court_point[1] <= 13.4
        upper = [
            player for player in candidates
            if -0.4 <= player["court_xy_m"][0] <= 6.5 and -0.8 <= player["court_xy_m"][1] <= 6.7
        ]
        lower = [
            player for player in candidates
            if -0.4 <= player["court_xy_m"][0] <= 6.5 and 6.7 < player["court_xy_m"][1] <= 14.2
        ]

        def player_quality(player: dict[str, Any], target_y: float) -> tuple[float, float]:
            ankle_scores = [
                float(player["keypoint_scores"][index])
                for index in (15, 16)
                if not player["keypoint_missing"][index]
            ]
            confidence = sum(ankle_scores) / len(ankle_scores) if ankle_scores else 0.0
            x, y = player["court_xy_m"]
            return confidence, -abs(x - 3.05) - 0.08 * abs(y - target_y)

        selected: list[dict[str, Any]] = []
        if upper:
            chosen = max(upper, key=lambda player: player_quality(player, 3.35))
            chosen["identity"] = "upper"
            selected.append(chosen)
        if lower:
            chosen = max(lower, key=lambda player: player_quality(player, 10.05))
            chosen["identity"] = "lower"
            selected.append(chosen)
        record["pose_candidate_count"] = len(candidates)
        record["players"] = selected
    _track_players(records)
    _smooth_shuttle(records, fps)
    return records


def _shuttle_point(record: dict[str, Any]) -> Point | None:
    point = record.get("shuttle", {}).get("smoothed_xy")
    return None if point is None else (float(point[0]), float(point[1]))


def _shuttle_contact_point(record: dict[str, Any], fallback: Point) -> Point:
    """Use an accepted detector association for contact evidence, not for trajectory curvature."""
    shuttle = record.get("shuttle", {})
    associated = shuttle.get("associated_xy")
    if (
        shuttle.get("measurement_status") in {"initialized", "accepted"}
        and isinstance(associated, (list, tuple))
        and len(associated) >= 2
    ):
        return float(associated[0]), float(associated[1])
    return fallback


def _wrist_distance(record: dict[str, Any], shuttle: Point) -> tuple[float, str | None]:
    best = math.inf
    hitter = None
    for player in record.get("players", []):
        for index in (9, 10):
            wrist = _valid_keypoint(player, index)
            if wrist is None:
                continue
            distance = _distance(wrist, shuttle)
            if distance < best:
                best = distance
                hitter = player.get("identity")
    return best, hitter


def _pose_motion(previous: dict[str, Any], following: dict[str, Any]) -> float:
    before = {player.get("identity"): player for player in previous.get("players", [])}
    after = {player.get("identity"): player for player in following.get("players", [])}
    maximum = 0.0
    for identity in before.keys() & after.keys():
        for index in (9, 10):
            point_before = _valid_keypoint(before[identity], index)
            point_after = _valid_keypoint(after[identity], index)
            if point_before and point_after:
                maximum = max(maximum, _distance(point_before, point_after))
    return maximum


def _windowed_hitter_assignment(
    scored: list[dict[str, Any]],
    index: int,
    window_frames: int,
) -> tuple[str | None, int, float]:
    """Choose hitter evidence near an event peak without moving the event frame."""
    base_frame = int(scored[index]["frame_idx"])
    nearby = [
        item for item in scored
        if abs(int(item["frame_idx"]) - base_frame) <= window_frames
        and item.get("predicted_hitter") is not None
        and float(item.get("hand_distance_score", 0.0)) > 0.0
    ]
    if not nearby:
        return scored[index].get("predicted_hitter"), base_frame, float(scored[index]["hand_distance_score"])
    chosen = max(
        nearby,
        key=lambda item: (
            float(item["hand_distance_score"]),
            -abs(int(item["frame_idx"]) - base_frame),
            float(item["event_confidence"]),
        ),
    )
    return (
        chosen.get("predicted_hitter"),
        int(chosen["frame_idx"]),
        float(chosen["hand_distance_score"]),
    )


def score_hit_candidates(
    records: list[dict[str, Any]],
    *,
    fps: float,
    threshold: float = 0.38,
    min_separation_frames: int | None = None,
    min_separation_seconds: float = 0.4,
    hitter_window_frames: int | None = None,
    hitter_window_seconds: float = 2 / 30,
) -> list[dict[str, Any]]:
    if len(records) < 3:
        return []
    if fps <= 0:
        raise ValueError("fps must be positive")
    if min_separation_frames is None:
        if min_separation_seconds <= 0:
            raise ValueError("min_separation_seconds must be positive")
        min_separation_frames = max(1, round(fps * min_separation_seconds))
    elif min_separation_frames < 1:
        raise ValueError("min_separation_frames must be at least 1")
    if hitter_window_frames is None:
        if hitter_window_seconds < 0:
            raise ValueError("hitter_window_seconds cannot be negative")
        hitter_window_frames = max(0, round(fps * hitter_window_seconds))
    elif hitter_window_frames < 0:
        raise ValueError("hitter_window_frames cannot be negative")
    scored: list[dict[str, Any]] = []
    for index in range(1, len(records) - 1):
        previous, current, following = records[index - 1], records[index], records[index + 1]
        if (
            int(current["frame_idx"]) - int(previous["frame_idx"]) > max(3, round(fps * 0.5))
            or int(following["frame_idx"]) - int(current["frame_idx"]) > max(3, round(fps * 0.5))
        ):
            continue
        p0, p1, p2 = _shuttle_point(previous), _shuttle_point(current), _shuttle_point(following)
        if p0 is None or p1 is None or p2 is None:
            continue
        v1, v2 = (p1[0] - p0[0], p1[1] - p0[1]), (p2[0] - p1[0], p2[1] - p1[1])
        speed1, speed2 = math.hypot(*v1), math.hypot(*v2)
        if speed1 < 0.3 or speed2 < 0.3:
            angle_score = 0.0
        else:
            cosine = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (speed1 * speed2)))
            angle_score = math.acos(cosine) / math.pi
        speed_change = abs(speed2 - speed1) / max(speed1, speed2, 1.0)
        trajectory_score_raw = min(1.0, 0.7 * angle_score + 0.3 * speed_change)
        trajectory_tracking_reliability = min(
            float(row.get("shuttle", {}).get("trajectory_reliability", 1.0))
            for row in (previous, current, following)
        )
        trajectory_score = trajectory_score_raw * (0.35 + 0.65 * trajectory_tracking_reliability)

        diagonal = math.hypot(float(current["width"]), float(current["height"]))
        contact_point = _shuttle_contact_point(current, p1)
        hand_distance, hitter = _wrist_distance(current, contact_point)
        hand_score = 0.0 if not math.isfinite(hand_distance) else max(0.0, 1.0 - hand_distance / (diagonal * 0.09))
        pose_score = min(1.0, _pose_motion(previous, following) / (diagonal * 0.035))
        confidence = 0.52 * trajectory_score + 0.33 * hand_score + 0.15 * pose_score
        scored.append(
            {
                "frame_idx": int(current["frame_idx"]),
                "timestamp_sec": float(current["timestamp_sec"]),
                "predicted_hitter": hitter,
                "trajectory_score": trajectory_score,
                "trajectory_score_raw": trajectory_score_raw,
                "trajectory_tracking_reliability": trajectory_tracking_reliability,
                "hand_distance_score": hand_score,
                "pose_score": pose_score,
                "event_confidence": confidence,
                "player_locations": copy.deepcopy(current.get("player_tracks", {})),
                "shuttle_image_xy": [contact_point[0], contact_point[1]],
                "shuttle_track_xy": [p1[0], p1[1]],
                "shuttle_tracking_state": current.get("shuttle", {}).get("tracking_state", "unknown"),
                "shuttle_measurement_status": current.get("shuttle", {}).get("measurement_status", "unknown"),
                "shuttle_measurement_weight": current.get("shuttle", {}).get("measurement_weight"),
                "shuttle_innovation_ratio": current.get("shuttle", {}).get("innovation_ratio"),
                # A flying shuttle is not on the court plane, so Homography is invalid here.
                "shuttle_court_xy_m": None,
            }
        )

    candidates: list[dict[str, Any]] = []
    for index, item in enumerate(scored):
        if item["event_confidence"] < threshold:
            continue
        neighbours = scored[max(0, index - 2): min(len(scored), index + 3)]
        if item["event_confidence"] < max(other["event_confidence"] for other in neighbours):
            continue
        if candidates and item["frame_idx"] - candidates[-1]["candidate_frame"] < min_separation_frames:
            if item["event_confidence"] <= candidates[-1]["event_confidence"]:
                continue
            candidates.pop()
        frame_idx = int(item["frame_idx"])
        timestamp_sec = float(item["timestamp_sec"])
        predicted_hitter, hitter_frame, hitter_hand_score = _windowed_hitter_assignment(
            scored,
            index,
            hitter_window_frames,
        )
        candidates.append(
            {
                "schema_version": "1.0",
                "event_id": f"evt_{frame_idx:08d}",
                "candidate_frame": frame_idx,
                "candidate_time": timestamp_sec,
                "window_start_frame": max(0, frame_idx - round(fps * 1.5)),
                "window_end_frame": frame_idx + round(fps * 1.5),
                **{
                    key: value for key, value in item.items()
                    if key not in {"frame_idx", "timestamp_sec", "predicted_hitter"}
                },
                "predicted_hitter": predicted_hitter,
                "hitter_assignment_method": "max_hand_proximity_window",
                "hitter_assignment_frame": hitter_frame,
                "hitter_assignment_offset_frames": hitter_frame - frame_idx,
                "hitter_assignment_hand_score": hitter_hand_score,
                "review_status": "unreviewed",
                "label_source": "machine_candidate",
            }
        )
    return candidates


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _video_ids_match(frame_video_id: str, calibration_video_id: str) -> bool:
    frame_id = frame_video_id.removeprefix("vid_")
    calibration_id = calibration_video_id.removeprefix("vid_")
    return frame_id.startswith(calibration_id) or calibration_id.startswith(frame_id)


def process_files(
    frames_path: Path,
    calibration_path: Path,
    output_frames_path: Path,
    output_events_path: Path,
    *,
    fps: float = 0.0,
    event_threshold: float = 0.38,
    min_hit_separation_sec: float = 0.4,
    hitter_window_sec: float = 2 / 30,
    play_state_window_sec: float = 1.0,
    play_state_support_sec: float = 3.0,
    play_state_gate_mode: str = "mechanics",
    play_state_gate_min_link_sec: float = 0.25,
    play_state_gate_max_link_sec: float = 4.0,
    play_state_gate_pre_roll_sec: float = 0.15,
    play_state_gate_post_roll_sec: float = 0.35,
    play_state_gate_min_tracked_ratio: float = 0.55,
    play_state_gate_min_moving_ratio: float = 0.25,
    play_state_gate_min_sequence_support: float = 0.75,
    output_suppressed_events_path: Path | None = None,
    output_provisional_events_path: Path | None = None,
) -> dict[str, Any]:
    if play_state_gate_mode not in {"off", "mechanics"}:
        raise ValueError("play_state_gate_mode must be 'off' or 'mechanics'")
    records = _read_jsonl(frames_path)
    if not records:
        raise ValueError("frames JSONL is empty")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration.get("quality_status") != "accepted":
        raise ValueError("calibration is not accepted")
    frame_video_id = str(records[0].get("video_id", ""))
    calibration_video_id = str(calibration.get("video_id", ""))
    if not frame_video_id or not calibration_video_id or not _video_ids_match(frame_video_id, calibration_video_id):
        raise ValueError(
            f"calibration video_id {calibration_video_id!r} does not match frame video_id {frame_video_id!r}"
        )
    if not fps:
        if len(records) < 2:
            raise ValueError("--fps is required when fewer than two frame records are present")
        frame_delta = float(records[1]["frame_idx"] - records[0]["frame_idx"])
        time_delta = float(records[1]["timestamp_sec"] - records[0]["timestamp_sec"])
        if frame_delta <= 0.0 or time_delta <= 0.0:
            raise ValueError("frame records must have increasing frame indexes and timestamps")
        fps = frame_delta / time_delta

    enriched = enrich_records(records, calibration["homography"], fps=fps)
    provisional_events = score_hit_candidates(
        enriched,
        fps=fps,
        threshold=event_threshold,
        min_separation_seconds=min_hit_separation_sec,
        hitter_window_seconds=hitter_window_sec,
    )
    annotate_play_state_evidence(
        enriched,
        provisional_events,
        fps=fps,
        window_seconds=play_state_window_sec,
        candidate_support_seconds=play_state_support_sec,
    )
    events, suppressed_events, gate_summary = apply_play_state_gate(
        enriched,
        provisional_events,
        fps=fps,
        enabled=play_state_gate_mode == "mechanics",
        min_link_seconds=play_state_gate_min_link_sec,
        max_link_seconds=play_state_gate_max_link_sec,
        pre_roll_seconds=play_state_gate_pre_roll_sec,
        post_roll_seconds=play_state_gate_post_roll_sec,
        min_tracked_ratio=play_state_gate_min_tracked_ratio,
        min_moving_ratio=play_state_gate_min_moving_ratio,
        min_sequence_support=play_state_gate_min_sequence_support,
    )
    assign_rally_ids(events)
    model_version = "hit-rules-0.6.0-dev" if play_state_gate_mode == "mechanics" else "hit-rules-0.5.0-dev"
    for event in events + suppressed_events:
        event["video_id"] = frame_video_id
        event["model_version"] = model_version
        event["calibration_version"] = calibration.get("version")
    if output_suppressed_events_path is None:
        output_suppressed_events_path = output_events_path.with_name(
            f"{output_events_path.stem}.suppressed{output_events_path.suffix}"
        )
    if output_provisional_events_path is None:
        output_provisional_events_path = output_events_path.with_name(
            f"{output_events_path.stem}.provisional{output_events_path.suffix}"
        )
    _write_jsonl(output_frames_path, enriched)
    _write_jsonl(output_events_path, events)
    _write_jsonl(output_suppressed_events_path, suppressed_events)
    _write_jsonl(
        output_provisional_events_path,
        sorted(events + suppressed_events, key=lambda row: int(row["candidate_frame"])),
    )
    return {
        "schema_version": "1.0",
        "video_id": frame_video_id,
        "frames": len(enriched),
        "events": len(events),
        "provisional_events": len(provisional_events),
        "suppressed_events": len(suppressed_events),
        "fps": fps,
        "output_frames": str(output_frames_path.resolve()),
        "output_events": str(output_events_path.resolve()),
        "output_suppressed_events": str(output_suppressed_events_path.resolve()),
        "output_provisional_events": str(output_provisional_events_path.resolve()),
        "calibration": str(calibration_path.resolve()),
        "calibration_version": calibration.get("version"),
        "hit_rule_parameters": {
            "event_threshold": event_threshold,
            "min_hit_separation_sec": min_hit_separation_sec,
            "hitter_window_sec": hitter_window_sec,
            "play_state_window_sec": play_state_window_sec,
            "play_state_support_sec": play_state_support_sec,
            "play_state_filter_applied": play_state_gate_mode == "mechanics",
            "play_state_gate_mode": play_state_gate_mode,
            "shuttle_track_version": SHUTTLE_TRACK_VERSION,
            "shuttle_track_policy": "top_k_temporal_association_plus_hit_aware_soft_innovation_gate",
            "interpolation_applied": False,
        },
        "play_state_gate": gate_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-frames", type=Path, required=True)
    parser.add_argument("--output-events", type=Path, required=True)
    parser.add_argument("--output-suppressed-events", type=Path)
    parser.add_argument("--output-provisional-events", type=Path)
    parser.add_argument("--output-summary", type=Path)
    parser.add_argument("--fps", type=float, default=0.0)
    parser.add_argument("--event-threshold", type=float, default=0.38)
    parser.add_argument("--min-hit-separation-sec", type=float, default=0.4)
    parser.add_argument("--hitter-window-sec", type=float, default=2 / 30)
    parser.add_argument("--play-state-window-sec", type=float, default=1.0)
    parser.add_argument("--play-state-support-sec", type=float, default=3.0)
    parser.add_argument("--play-state-gate-mode", choices=("off", "mechanics"), default="mechanics")
    parser.add_argument("--play-state-gate-min-link-sec", type=float, default=0.25)
    parser.add_argument("--play-state-gate-max-link-sec", type=float, default=4.0)
    parser.add_argument("--play-state-gate-pre-roll-sec", type=float, default=0.15)
    parser.add_argument("--play-state-gate-post-roll-sec", type=float, default=0.35)
    parser.add_argument("--play-state-gate-min-tracked-ratio", type=float, default=0.55)
    parser.add_argument("--play-state-gate-min-moving-ratio", type=float, default=0.25)
    parser.add_argument("--play-state-gate-min-sequence-support", type=float, default=0.75)
    args = parser.parse_args()

    try:
        summary = process_files(
            args.frames,
            args.calibration,
            args.output_frames,
            args.output_events,
            fps=args.fps,
            event_threshold=args.event_threshold,
            min_hit_separation_sec=args.min_hit_separation_sec,
            hitter_window_sec=args.hitter_window_sec,
            play_state_window_sec=args.play_state_window_sec,
            play_state_support_sec=args.play_state_support_sec,
            play_state_gate_mode=args.play_state_gate_mode,
            play_state_gate_min_link_sec=args.play_state_gate_min_link_sec,
            play_state_gate_max_link_sec=args.play_state_gate_max_link_sec,
            play_state_gate_pre_roll_sec=args.play_state_gate_pre_roll_sec,
            play_state_gate_post_roll_sec=args.play_state_gate_post_roll_sec,
            play_state_gate_min_tracked_ratio=args.play_state_gate_min_tracked_ratio,
            play_state_gate_min_moving_ratio=args.play_state_gate_min_moving_ratio,
            play_state_gate_min_sequence_support=args.play_state_gate_min_sequence_support,
            output_suppressed_events_path=args.output_suppressed_events,
            output_provisional_events_path=args.output_provisional_events,
        )
    except ValueError as exc:
        parser.error(str(exc))
    if args.output_summary is not None:
        _write_json(args.output_summary, summary)
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
