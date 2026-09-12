"""Export local pipeline artifacts as a compact ShuttleSet-like dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "1.0"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _player(frame: dict[str, Any], identity: str) -> dict[str, Any] | None:
    return next((item for item in frame.get("players", []) if item.get("identity") == identity), None)


def _xy(value: object) -> tuple[float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None, None
    return float(value[0]), float(value[1])


def _video_record(path: Path | None, frames: list[dict[str, Any]], input_video: Path | None) -> dict[str, Any]:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    first = frames[0]
    return {
        "video_id": first["video_id"],
        "path": str(input_video.resolve()) if input_video is not None else None,
        "file_name": input_video.name if input_video is not None else None,
        "checksum": None,
        "fps": None,
        "frame_count": None,
        "width": first.get("width"),
        "height": first.get("height"),
        "duration_sec": None,
        "match_id": "unknown",
        "players": {"A": "unknown", "B": "unknown"},
        "domain": "fixed-camera-singles",
        "redistribution": "unknown",
    }


def _frame_row(frame: dict[str, Any]) -> dict[str, Any]:
    upper = _player(frame, "upper")
    lower = _player(frame, "lower")
    tracks = frame.get("player_tracks", {})
    shuttle = frame.get("shuttle", {})
    models = frame.get("models", {})

    def player_value(player: dict[str, Any] | None, key: str) -> object:
        return None if player is None else player.get(key)

    upper_xy = _xy(tracks.get("upper", {}).get("court_xy_m"))
    lower_xy = _xy(tracks.get("lower", {}).get("court_xy_m"))
    raw_xy = _xy(shuttle.get("raw_xy"))
    associated_xy = _xy(shuttle.get("associated_xy"))
    predicted_xy = _xy(shuttle.get("predicted_xy"))
    smooth_xy = _xy(shuttle.get("smoothed_xy"))
    return {
        "video_id": str(frame["video_id"]),
        "frame_idx": int(frame["frame_idx"]),
        "timestamp_sec": float(frame["timestamp_sec"]),
        "width": int(frame["width"]),
        "height": int(frame["height"]),
        "is_court_view": bool(upper is not None or lower is not None),
        "upper_bbox": player_value(upper, "bbox_xyxy"),
        "upper_keypoints": player_value(upper, "keypoints_xy"),
        "upper_keypoint_scores": player_value(upper, "keypoint_scores"),
        "upper_keypoint_missing": player_value(upper, "keypoint_missing"),
        "upper_court_x": upper_xy[0],
        "upper_court_y": upper_xy[1],
        "upper_tracking_state": tracks.get("upper", {}).get("state", "missing"),
        "lower_bbox": player_value(lower, "bbox_xyxy"),
        "lower_keypoints": player_value(lower, "keypoints_xy"),
        "lower_keypoint_scores": player_value(lower, "keypoint_scores"),
        "lower_keypoint_missing": player_value(lower, "keypoint_missing"),
        "lower_court_x": lower_xy[0],
        "lower_court_y": lower_xy[1],
        "lower_tracking_state": tracks.get("lower", {}).get("state", "missing"),
        "shuttle_raw_x": raw_xy[0],
        "shuttle_raw_y": raw_xy[1],
        "shuttle_smoothed_x": smooth_xy[0],
        "shuttle_smoothed_y": smooth_xy[1],
        "shuttle_confidence": shuttle.get("confidence"),
        "shuttle_state": shuttle.get("tracking_state", shuttle.get("state", "unknown")),
        "shuttle_track_version": shuttle.get("track_version"),
        "shuttle_candidates_json": json.dumps(
            shuttle.get("candidates", []), ensure_ascii=False, separators=(",", ":"),
        ),
        "shuttle_detector_candidate_count": shuttle.get("detector_candidate_count", 0),
        "shuttle_selected_candidate_rank": shuttle.get("selected_candidate_rank"),
        "shuttle_associated_x": associated_xy[0],
        "shuttle_associated_y": associated_xy[1],
        "shuttle_predicted_x": predicted_xy[0],
        "shuttle_predicted_y": predicted_xy[1],
        "shuttle_innovation_px": shuttle.get("innovation_px"),
        "shuttle_innovation_ratio": shuttle.get("innovation_ratio"),
        "shuttle_adaptive_gate_px": shuttle.get("adaptive_gate_px"),
        "shuttle_continuity_score": shuttle.get("continuity_score"),
        "shuttle_hit_support_score": shuttle.get("hit_support_score"),
        "shuttle_hand_support_score": shuttle.get("hand_support_score"),
        "shuttle_pose_support_score": shuttle.get("pose_support_score"),
        "shuttle_association_score": shuttle.get("association_score"),
        "shuttle_measurement_weight": shuttle.get("measurement_weight"),
        "shuttle_measurement_status": shuttle.get("measurement_status"),
        "shuttle_tracking_uncertainty_px": shuttle.get("tracking_uncertainty_px"),
        "shuttle_trajectory_reliability": shuttle.get("trajectory_reliability"),
        "play_state_gate_version": str(frame.get("play_state_gate_version") or "unknown"),
        "play_state": str(frame.get("play_state") or "unknown"),
        "play_state_segment_id": frame.get("play_state_segment_id"),
        "pose_model_version": models.get("pose", {}).get("family"),
        "pose_weights_sha256": models.get("pose", {}).get("weights_sha256"),
        "shuttle_model_version": models.get("shuttle", {}).get("family"),
        "shuttle_weights_sha256": models.get("shuttle", {}).get("weights_sha256"),
    }


def _event_rows(events: list[dict[str, Any]], video: dict[str, Any]) -> list[dict[str, Any]]:
    counters: defaultdict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda item: (float(item["candidate_time"]), int(item["candidate_frame"]))):
        rally_id = str(event.get("rally_id") or "unknown")
        counters[rally_id] += 1
        side = event.get("predicted_hitter")
        opponent = "lower" if side == "upper" else ("upper" if side == "lower" else None)
        locations = event.get("player_locations", {})
        player_xy = _xy(locations.get(side, {}).get("court_xy_m")) if side else (None, None)
        opponent_xy = _xy(locations.get(opponent, {}).get("court_xy_m")) if opponent else (None, None)
        shuttle_xy = _xy(event.get("shuttle_image_xy"))
        stroke_type = str(
            event.get("human_stroke_type")
            or event.get("predicted_stroke_type")
            or event.get("stroke_type")
            or "unknown"
        )
        stroke_type_confidence = event.get("stroke_type_confidence")
        landing_x = event.get("human_landing_x", event.get("predicted_landing_x", event.get("landing_x")))
        landing_y = event.get("human_landing_y", event.get("predicted_landing_y", event.get("landing_y")))
        landing_source = str(event.get("landing_source") or "unavailable")
        semantic_label_source = str(event.get("semantic_label_source") or event.get("label_source") or "machine_candidate")
        semantic_review_status = str(event.get("semantic_review_status") or event.get("review_status") or "unreviewed")
        rows.append(
            {
                "event_id": str(event["event_id"]),
                "video_id": str(event["video_id"]),
                "match_id": str(video.get("match_id") or "unknown"),
                "set_id": "unknown",
                "rally_id": rally_id,
                "stroke_index": counters[rally_id],
                "hit_frame": int(event["candidate_frame"]),
                "hit_time": float(event["candidate_time"]),
                # Named player identity is unavailable before annotation.  The
                # reliable fixed-camera court-side identity is retained.
                "player": side or "unknown",
                "stroke_type": stroke_type,
                "stroke_type_confidence": stroke_type_confidence,
                "stroke_type_status": str(event.get("stroke_type_status") or "unknown"),
                "stroke_type_input_quality": event.get("stroke_type_input_quality"),
                "stroke_type_source": str(event.get("stroke_type_source") or "unavailable"),
                "stroke_type_raw_class": event.get("stroke_type_raw_class"),
                "stroke_type_top3_json": json.dumps(
                    event.get("stroke_type_top3", []), ensure_ascii=False, separators=(",", ":"),
                ),
                "aroundhead": event.get("aroundhead"),
                "backhand": event.get("backhand"),
                "player_location_x": player_xy[0],
                "player_location_y": player_xy[1],
                "opponent_location_x": opponent_xy[0],
                "opponent_location_y": opponent_xy[1],
                "hit_x": None,
                "hit_y": None,
                "landing_x": landing_x,
                "landing_y": landing_y,
                "landing_frame": event.get("landing_frame"),
                "landing_kind": str(event.get("landing_kind") or "unknown"),
                "landing_confidence": event.get("landing_confidence"),
                "landing_source": landing_source,
                "landing_status": str(event.get("landing_status") or "unknown"),
                "landing_proxy_kind": str(event.get("landing_proxy_kind") or "unavailable"),
                "shuttle_image_x": shuttle_xy[0],
                "shuttle_image_y": shuttle_xy[1],
                "shuttle_tracking_state": event.get("shuttle_tracking_state", "unknown"),
                "window_start_frame": int(event["window_start_frame"]),
                "window_end_frame": int(event["window_end_frame"]),
                "trajectory_score": float(event["trajectory_score"]),
                "trajectory_score_raw": event.get("trajectory_score_raw"),
                "trajectory_tracking_reliability": event.get("trajectory_tracking_reliability"),
                "hand_distance_score": float(event["hand_distance_score"]),
                "pose_score": float(event["pose_score"]),
                "event_confidence": float(event["event_confidence"]),
                "shuttle_measurement_status": str(event.get("shuttle_measurement_status") or "unknown"),
                "shuttle_measurement_weight": event.get("shuttle_measurement_weight"),
                "shuttle_innovation_ratio": event.get("shuttle_innovation_ratio"),
                "play_state_version": str(event.get("play_state_version") or "unknown"),
                "play_state_score": event.get("play_state_score"),
                "play_state_filter_applied": bool(event.get("play_state_filter_applied", False)),
                "play_state_decision": str(event.get("play_state_decision") or "not_available"),
                "play_state_window_seconds": event.get("play_state_window_seconds"),
                "play_state_window_frames": event.get("play_state_window_frames"),
                "play_state_window_radius_frames": event.get("play_state_window_radius_frames"),
                "play_state_support_seconds": event.get("play_state_support_seconds"),
                "play_state_support_radius_frames": event.get("play_state_support_radius_frames"),
                "play_state_shuttle_measured_ratio": event.get("play_state_shuttle_measured_ratio"),
                "play_state_shuttle_tracked_ratio": event.get("play_state_shuttle_tracked_ratio"),
                "play_state_shuttle_moving_ratio": event.get("play_state_shuttle_moving_ratio"),
                "play_state_both_player_tracking_ratio": event.get("play_state_both_player_tracking_ratio"),
                "play_state_nearby_candidate_count": event.get("play_state_nearby_candidate_count"),
                "play_state_sequence_support": event.get("play_state_sequence_support"),
                "play_state_gate_version": str(event.get("play_state_gate_version") or "unknown"),
                "play_state_gate_applied": bool(event.get("play_state_gate_applied", False)),
                "play_state_gate_decision": str(event.get("play_state_gate_decision") or "not_available"),
                "play_state_segment_id": event.get("play_state_segment_id"),
                "label_source": str(event.get("label_source") or "machine_candidate"),
                "annotation_status": str(event.get("review_status") or "unreviewed"),
                "semantic_label_source": semantic_label_source,
                "semantic_review_status": semantic_review_status,
                "model_version": str(event.get("model_version") or "unknown"),
                "calibration_version": event.get("calibration_version"),
            }
        )
    return rows


def _quality_report(
    frames: list[dict[str, Any]],
    frame_rows: list[dict[str, Any]],
    event_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(frame_rows)

    def ratio(value: int, denominator: int = total) -> float | None:
        return None if denominator == 0 else round(value / denominator, 6)

    def player_metrics(identity: str) -> dict[str, Any]:
        states = [str(row[f"{identity}_tracking_state"]) for row in frame_rows]
        measured = states.count("measured")
        predicted = states.count("predicted")
        missing = states.count("missing")
        selected = [_player(frame, identity) for frame in frames]
        selected = [player for player in selected if player is not None]
        keypoints = sum(len(player.get("keypoint_missing", [])) for player in selected)
        valid_keypoints = sum(
            sum(not bool(value) for value in player.get("keypoint_missing", []))
            for player in selected
        )
        in_bounds_values = [bool(player.get("court_in_bounds")) for player in selected if "court_in_bounds" in player]
        return {
            "measured_frames": measured,
            "predicted_frames": predicted,
            "missing_frames": missing,
            "tracking_coverage": ratio(measured + predicted),
            "measured_rate": ratio(measured),
            "selected_pose_rate": ratio(len(selected)),
            "valid_keypoint_rate": ratio(valid_keypoints, keypoints),
            "court_in_bounds_rate": ratio(sum(in_bounds_values), len(in_bounds_values)),
        }

    shuttle_states = [str(row["shuttle_state"]) for row in frame_rows]
    shuttle_measured = shuttle_states.count("measured")
    shuttle_predicted = shuttle_states.count("predicted")
    shuttle_missing = sum(state in {"missing", "not_run", "unknown"} for state in shuttle_states)
    measurement_statuses = [str(row.get("shuttle_measurement_status") or "unknown") for row in frame_rows]
    innovations = sorted(
        float(row["shuttle_innovation_ratio"])
        for row in frame_rows
        if row.get("shuttle_innovation_ratio") is not None
    )
    uncertainties = sorted(
        float(row["shuttle_tracking_uncertainty_px"])
        for row in frame_rows
        if row.get("shuttle_tracking_uncertainty_px") is not None
    )
    timestamps = [float(row["timestamp_sec"]) for row in frame_rows]
    if len(timestamps) > 1:
        duration = max(timestamps) - min(timestamps) + (timestamps[-1] - timestamps[-2])
    else:
        duration = 0.0
    confidences = sorted(float(row["event_confidence"]) for row in event_rows)
    play_state_scores = sorted(
        float(row["play_state_score"])
        for row in event_rows
        if row.get("play_state_score") is not None
    )
    play_states = [str(row.get("play_state") or "unknown") for row in frame_rows]

    def percentile(values: list[float], fraction: float) -> float | None:
        if not values:
            return None
        index = round((len(values) - 1) * fraction)
        return round(values[index], 6)

    warnings: list[str] = []
    player_results = {identity: player_metrics(identity) for identity in ("upper", "lower")}
    for identity, metrics in player_results.items():
        if metrics["tracking_coverage"] is not None and metrics["tracking_coverage"] < 0.9:
            warnings.append(f"{identity} player tracking coverage is below 90%")
        if metrics["court_in_bounds_rate"] is not None and metrics["court_in_bounds_rate"] < 0.9:
            warnings.append(f"{identity} player court in-bounds rate is below 90%")
    shuttle_coverage = ratio(shuttle_measured + shuttle_predicted)
    if shuttle_coverage is not None and shuttle_coverage < 0.8:
        warnings.append("shuttle track coverage is below 80%")
    if not event_rows:
        warnings.append("no hit candidates were generated")
    semantic_predicted = sum(row.get("stroke_type") not in {None, "", "unknown"} for row in event_rows)
    semantic_confirmed = sum(
        row.get("semantic_label_source") == "human_semantic_gold"
        and row.get("semantic_review_status") == "confirmed"
        for row in event_rows
    )
    landing_predicted = sum(row.get("landing_x") is not None and row.get("landing_y") is not None for row in event_rows)
    if event_rows and semantic_predicted < len(event_rows):
        warnings.append("stroke-type prediction coverage is incomplete")
    if event_rows and landing_predicted < len(event_rows):
        warnings.append("destination prediction coverage is incomplete, including terminal strokes")
    if event_rows and semantic_confirmed < len(event_rows):
        warnings.append("semantic rows are automatic prelabels, not confirmed ground truth")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": "warning" if warnings else "ok",
        "scope": "automatic coverage diagnostics; not a Gold accuracy evaluation",
        "frames": total,
        "processed_duration_sec": round(duration, 3),
        "players": player_results,
        "shuttle": {
            "measured_frames": shuttle_measured,
            "predicted_frames": shuttle_predicted,
            "missing_frames": shuttle_missing,
            "raw_detection_rate": ratio(sum(row["shuttle_raw_x"] is not None for row in frame_rows)),
            "track_coverage": shuttle_coverage,
            "track_version": next((row.get("shuttle_track_version") for row in frame_rows if row.get("shuttle_track_version")), None),
            "soft_suppressed_frames": measurement_statuses.count("soft_suppressed"),
            "selected_non_primary_frames": sum(
                row.get("shuttle_selected_candidate_rank") not in {None, 0}
                for row in frame_rows
            ),
            "multi_candidate_frames": sum(
                int(row.get("shuttle_detector_candidate_count") or 0) > 1
                for row in frame_rows
            ),
            "innovation_ratio_p95": percentile(innovations, 0.95),
            "innovation_ratio_p99": percentile(innovations, 0.99),
            "tracking_uncertainty_px_median": percentile(uncertainties, 0.5),
            "interpolation_applied": False,
        },
        "events": {
            "candidate_count": len(event_rows),
            "candidate_rate_per_min": None if duration <= 0 else round(len(event_rows) * 60 / duration, 3),
            "rally_candidate_count": len({row["rally_id"] for row in event_rows}),
            "confidence_p25": percentile(confidences, 0.25),
            "confidence_median": percentile(confidences, 0.5),
            "confidence_p75": percentile(confidences, 0.75),
            "play_state_filter_applied_count": sum(bool(row["play_state_filter_applied"]) for row in event_rows),
            "play_state_score_p25": percentile(play_state_scores, 0.25),
            "play_state_score_median": percentile(play_state_scores, 0.5),
            "play_state_score_p75": percentile(play_state_scores, 0.75),
            "play_state_gate_applied_count": sum(bool(row["play_state_gate_applied"]) for row in event_rows),
            "stroke_type_prediction_coverage": None if not event_rows else round(semantic_predicted / len(event_rows), 6),
            "destination_prediction_coverage": None if not event_rows else round(landing_predicted / len(event_rows), 6),
            "semantic_gold_confirmation_coverage": None if not event_rows else round(semantic_confirmed / len(event_rows), 6),
        },
        "play_state_gate": {
            "active_play_frames": play_states.count("active_play"),
            "inactive_wait_frames": play_states.count("inactive_wait"),
            "not_gated_frames": play_states.count("not_gated"),
            "active_play_frame_rate": ratio(play_states.count("active_play")),
        },
        "warnings": warnings,
    }


def export_dataset(
    frames_path: Path,
    events_path: Path,
    calibration_path: Path,
    output_dir: Path,
    *,
    video_record_path: Path | None = None,
    input_video: Path | None = None,
) -> dict[str, Any]:
    """Write Parquet frame tables and a scalar ShuttleSet-like event table."""
    try:
        import polars as pl
    except ImportError as exc:
        raise RuntimeError("polars is required for Parquet dataset export") from exc

    frames = _read_jsonl(frames_path)
    events = _read_jsonl(events_path)
    if not frames:
        raise ValueError("frames JSONL is empty")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    video = _video_record(video_record_path, frames, input_video)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_rows = [_frame_row(frame) for frame in frames]
    event_rows = _event_rows(events, video)
    video_row = {
        "video_id": str(video.get("video_id") or frames[0]["video_id"]),
        "path": video.get("path"),
        "proxy_path": video.get("proxy_path"),
        "checksum": video.get("checksum"),
        "fps": video.get("fps"),
        "frame_count": video.get("frame_count"),
        "width": video.get("width"),
        "height": video.get("height"),
        "duration_sec": video.get("duration_sec"),
        "match_id": str(video.get("match_id") or "unknown"),
        "players_json": json.dumps(video.get("players", {}), ensure_ascii=False, separators=(",", ":")),
        "domain": str(video.get("domain") or "unknown"),
        "redistribution": str(video.get("redistribution") or "unknown"),
    }
    calibration_row = {
        "calibration_id": calibration.get("calibration_id"),
        "video_id": calibration.get("video_id"),
        "version": calibration.get("version"),
        "frame_time_ms": calibration.get("frame_time_ms"),
        "corners": [[float(item["x"]), float(item["y"])] for item in calibration.get("corners", [])],
        "homography": calibration.get("homography"),
        "court_orientation": calibration.get("court_orientation"),
        "quality_status": calibration.get("quality_status"),
        "source": calibration.get("source"),
    }

    frame_schema = {
        "video_id": pl.String, "frame_idx": pl.Int64, "timestamp_sec": pl.Float64,
        "width": pl.Int64, "height": pl.Int64, "is_court_view": pl.Boolean,
        "upper_bbox": pl.List(pl.Float64), "upper_keypoints": pl.List(pl.List(pl.Float64)),
        "upper_keypoint_scores": pl.List(pl.Float64), "upper_keypoint_missing": pl.List(pl.Boolean),
        "upper_court_x": pl.Float64, "upper_court_y": pl.Float64, "upper_tracking_state": pl.String,
        "lower_bbox": pl.List(pl.Float64), "lower_keypoints": pl.List(pl.List(pl.Float64)),
        "lower_keypoint_scores": pl.List(pl.Float64), "lower_keypoint_missing": pl.List(pl.Boolean),
        "lower_court_x": pl.Float64, "lower_court_y": pl.Float64, "lower_tracking_state": pl.String,
        "shuttle_raw_x": pl.Float64, "shuttle_raw_y": pl.Float64,
        "shuttle_smoothed_x": pl.Float64, "shuttle_smoothed_y": pl.Float64,
        "shuttle_confidence": pl.Float64, "shuttle_state": pl.String,
        "shuttle_track_version": pl.String, "shuttle_candidates_json": pl.String,
        "shuttle_detector_candidate_count": pl.Int64, "shuttle_selected_candidate_rank": pl.Int64,
        "shuttle_associated_x": pl.Float64, "shuttle_associated_y": pl.Float64,
        "shuttle_predicted_x": pl.Float64, "shuttle_predicted_y": pl.Float64,
        "shuttle_innovation_px": pl.Float64, "shuttle_innovation_ratio": pl.Float64,
        "shuttle_adaptive_gate_px": pl.Float64, "shuttle_continuity_score": pl.Float64,
        "shuttle_hit_support_score": pl.Float64, "shuttle_association_score": pl.Float64,
        "shuttle_hand_support_score": pl.Float64, "shuttle_pose_support_score": pl.Float64,
        "shuttle_measurement_weight": pl.Float64, "shuttle_measurement_status": pl.String,
        "shuttle_tracking_uncertainty_px": pl.Float64, "shuttle_trajectory_reliability": pl.Float64,
        "play_state_gate_version": pl.String, "play_state": pl.String,
        "play_state_segment_id": pl.String,
        "pose_model_version": pl.String, "pose_weights_sha256": pl.String,
        "shuttle_model_version": pl.String, "shuttle_weights_sha256": pl.String,
    }
    event_schema = {
        "event_id": pl.String, "video_id": pl.String, "match_id": pl.String, "set_id": pl.String,
        "rally_id": pl.String, "stroke_index": pl.Int64, "hit_frame": pl.Int64, "hit_time": pl.Float64,
        "player": pl.String, "stroke_type": pl.String, "stroke_type_confidence": pl.Float64,
        "stroke_type_status": pl.String, "stroke_type_input_quality": pl.Float64,
        "stroke_type_source": pl.String, "stroke_type_raw_class": pl.String,
        "stroke_type_top3_json": pl.String,
        "aroundhead": pl.Boolean, "backhand": pl.Boolean,
        "player_location_x": pl.Float64, "player_location_y": pl.Float64,
        "opponent_location_x": pl.Float64, "opponent_location_y": pl.Float64,
        "hit_x": pl.Float64, "hit_y": pl.Float64, "landing_x": pl.Float64, "landing_y": pl.Float64,
        "landing_frame": pl.Int64, "landing_kind": pl.String,
        "landing_confidence": pl.Float64, "landing_source": pl.String,
        "landing_status": pl.String, "landing_proxy_kind": pl.String,
        "shuttle_image_x": pl.Float64, "shuttle_image_y": pl.Float64, "shuttle_tracking_state": pl.String,
        "window_start_frame": pl.Int64, "window_end_frame": pl.Int64,
        "trajectory_score": pl.Float64, "trajectory_score_raw": pl.Float64,
        "trajectory_tracking_reliability": pl.Float64, "hand_distance_score": pl.Float64,
        "pose_score": pl.Float64, "event_confidence": pl.Float64,
        "shuttle_measurement_status": pl.String, "shuttle_measurement_weight": pl.Float64,
        "shuttle_innovation_ratio": pl.Float64,
        "play_state_version": pl.String, "play_state_score": pl.Float64,
        "play_state_filter_applied": pl.Boolean, "play_state_decision": pl.String,
        "play_state_window_seconds": pl.Float64, "play_state_window_frames": pl.Int64,
        "play_state_window_radius_frames": pl.Int64,
        "play_state_support_seconds": pl.Float64, "play_state_support_radius_frames": pl.Int64,
        "play_state_shuttle_measured_ratio": pl.Float64,
        "play_state_shuttle_tracked_ratio": pl.Float64,
        "play_state_shuttle_moving_ratio": pl.Float64,
        "play_state_both_player_tracking_ratio": pl.Float64,
        "play_state_nearby_candidate_count": pl.Int64,
        "play_state_sequence_support": pl.Float64,
        "play_state_gate_version": pl.String, "play_state_gate_applied": pl.Boolean,
        "play_state_gate_decision": pl.String, "play_state_segment_id": pl.String,
        "label_source": pl.String, "annotation_status": pl.String,
        "semantic_label_source": pl.String, "semantic_review_status": pl.String,
        "model_version": pl.String, "calibration_version": pl.Int64,
    }
    paths = {
        "videos": output_dir / "videos.parquet",
        "calibrations": output_dir / "calibrations.parquet",
        "frames": output_dir / "frames.parquet",
        "dataset_rows": output_dir / "dataset_rows.parquet",
        "dataset_csv": output_dir / "dataset_rows.csv",
        "dataset_jsonl": output_dir / "dataset_rows.jsonl",
        "dictionary": output_dir / "data_dictionary.json",
        "quality": output_dir / "quality.json",
    }
    pl.DataFrame([video_row], strict=False).write_parquet(paths["videos"], compression="zstd")
    pl.DataFrame([calibration_row], strict=False).write_parquet(paths["calibrations"], compression="zstd")
    pl.DataFrame(frame_rows, schema=frame_schema, strict=False).write_parquet(paths["frames"], compression="zstd")
    event_frame = pl.DataFrame(event_rows, schema=event_schema, strict=False)
    event_frame.write_parquet(paths["dataset_rows"], compression="zstd")
    event_frame.write_csv(paths["dataset_csv"])
    _write_jsonl(paths["dataset_jsonl"], event_rows)

    dictionary = {
        "schema_version": SCHEMA_VERSION,
        "coordinate_systems": {
            "image": "pixels, origin at top-left",
            "court": "metres on a 6.1m x 13.4m court plane",
        },
        "null_policy": "Unknown or unreliable values are null; categorical unknowns use the literal 'unknown'.",
        "identity_policy": "player is upper/lower court-side identity until a named-player mapping is supplied.",
        "play_state_policy": "The mechanics gate keeps candidates inside active-play segments supported by opposite-hitter links and continuous shuttle flight. Suppressed candidates remain in events.suppressed.jsonl outside this exported event table.",
        "semantic_truth_policy": "stroke_type and landing are predictions unless semantic_label_source is human_semantic_gold and semantic_review_status is confirmed. Landing means the opponent's next contact destination, or ground contact for the terminal stroke.",
        "shuttle_track_policy": "Top-k detections are associated over time; innovation residuals are soft-weighted with hit evidence. No interpolation is applied.",
        "tables": {
            "videos.parquet": "One source-video metadata row.",
            "calibrations.parquet": "Accepted four-corner calibration and Homography.",
            "frames.parquet": "One processed frame per row with COCO-17 pose, player court positions and shuttle track.",
            "dataset_rows.parquet": "One machine hit candidate per row in a ShuttleSet-like scalar schema.",
            "quality.json": "Automatic coverage diagnostics; it does not measure accuracy against human Gold labels.",
        },
    }
    _write_json(paths["dictionary"], dictionary)
    quality = _quality_report(frames, frame_rows, event_rows)
    _write_json(paths["quality"], quality)

    artifact_info = {
        key: {
            "path": str(path.resolve()),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for key, path in paths.items()
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "dataset_version": "local-visual-v0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video_id": frames[0]["video_id"],
        "rows": {"videos": 1, "calibrations": 1, "frames": len(frame_rows), "dataset_rows": len(event_rows)},
        "quality": quality,
        "sources": {
            "frames_jsonl": str(frames_path.resolve()),
            "events_jsonl": str(events_path.resolve()),
            "calibration_json": str(calibration_path.resolve()),
        },
        "artifacts": artifact_info,
    }
    manifest_path = output_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path.resolve())
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--video-record", type=Path)
    parser.add_argument("--input-video", type=Path)
    args = parser.parse_args()
    try:
        result = export_dataset(
            args.frames, args.events, args.calibration, args.output_dir,
            video_record_path=args.video_record, input_video=args.input_video,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
