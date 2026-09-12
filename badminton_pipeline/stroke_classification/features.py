"""Convert enriched frame/event JSONL into fixed-window BST training inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SKELETON_EDGES = (
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from error
    return records


def _player(frame: dict[str, Any], identity: str) -> dict[str, Any] | None:
    return next((item for item in frame.get("players", []) if item.get("identity") == identity), None)


def _court_position(frame: dict[str, Any], identity: str) -> tuple[list[float] | None, bool]:
    track = frame.get("player_tracks", {}).get(identity, {})
    point = track.get("court_xy_m")
    if point is None:
        player = _player(frame, identity)
        point = None if player is None else player.get("court_xy_m")
    if not isinstance(point, list) or len(point) != 2:
        return None, False
    return [float(point[0]) / 6.1, float(point[1]) / 13.4], True


def _label(event: dict[str, Any]) -> tuple[str, str]:
    for key in ("human_stroke_type", "stroke_type", "predicted_stroke_type"):
        value = event.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), str(event.get("label_source") or "unknown")
    return "unknown", str(event.get("label_source") or "machine_candidate")


def build_bst_samples(
    frames: Iterable[dict[str, Any]],
    events: Iterable[dict[str, Any]],
    *,
    window_size: int = 30,
    keypoint_threshold: float = 0.2,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build normalized multimodal arrays without inventing missing observations."""
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    frame_list = sorted(frames, key=lambda item: int(item["frame_idx"]))
    event_list = sorted(events, key=lambda item: int(item["candidate_frame"]))
    if not frame_list:
        raise ValueError("at least one enriched frame is required")
    if not event_list:
        raise ValueError("at least one event is required")
    video_ids = {str(item.get("video_id")) for item in frame_list + event_list if item.get("video_id")}
    if len(video_ids) > 1:
        raise ValueError("frames and events must belong to one video_id")

    frame_by_index = {int(item["frame_idx"]): item for item in frame_list}
    sample_count = len(event_list)
    pose_xy = np.zeros((sample_count, window_size, 17, 2), dtype=np.float32)
    pose_confidence = np.zeros((sample_count, window_size, 17), dtype=np.float32)
    pose_mask = np.zeros((sample_count, window_size, 17), dtype=np.bool_)
    skeleton_vectors = np.zeros((sample_count, window_size, len(SKELETON_EDGES), 2), dtype=np.float32)
    skeleton_mask = np.zeros((sample_count, window_size, len(SKELETON_EDGES)), dtype=np.bool_)
    shuttle_xy = np.zeros((sample_count, window_size, 2), dtype=np.float32)
    shuttle_confidence = np.zeros((sample_count, window_size), dtype=np.float32)
    shuttle_mask = np.zeros((sample_count, window_size), dtype=np.bool_)
    shuttle_measured_mask = np.zeros((sample_count, window_size), dtype=np.bool_)
    hitter_court_xy = np.zeros((sample_count, window_size, 2), dtype=np.float32)
    opponent_court_xy = np.zeros((sample_count, window_size, 2), dtype=np.float32)
    hitter_position_mask = np.zeros((sample_count, window_size), dtype=np.bool_)
    opponent_position_mask = np.zeros((sample_count, window_size), dtype=np.bool_)
    frame_mask = np.zeros((sample_count, window_size), dtype=np.bool_)
    frame_indices = np.zeros((sample_count, window_size), dtype=np.int32)
    candidate_frames = np.zeros(sample_count, dtype=np.int32)
    event_ids: list[str] = []
    hitter_sides: list[str] = []
    labels: list[str] = []
    label_sources: list[str] = []
    landing_xy_m = np.zeros((sample_count, 2), dtype=np.float32)
    landing_mask = np.zeros(sample_count, dtype=np.bool_)
    landing_kinds: list[str] = []

    for sample_index, event in enumerate(event_list):
        candidate = int(event["candidate_frame"])
        hitter = str(event.get("human_hitter") or event.get("predicted_hitter") or "")
        if hitter not in {"upper", "lower"}:
            raise ValueError(f"event {event.get('event_id')} must identify upper or lower hitter")
        opponent = "lower" if hitter == "upper" else "upper"
        start = candidate - window_size // 2
        candidate_frames[sample_index] = candidate
        event_ids.append(str(event.get("event_id") or f"event_{candidate}"))
        hitter_sides.append(hitter)
        label, source = _label(event)
        labels.append(label)
        label_sources.append(source)
        landing_x = event.get("human_landing_x", event.get("landing_x"))
        landing_y = event.get("human_landing_y", event.get("landing_y"))
        if landing_x is not None and landing_y is not None:
            landing_xy_m[sample_index] = (float(landing_x), float(landing_y))
            landing_mask[sample_index] = True
        landing_kinds.append(str(event.get("landing_kind") or "unknown"))

        for offset in range(window_size):
            frame_index = start + offset
            frame_indices[sample_index, offset] = frame_index
            frame = frame_by_index.get(frame_index)
            if frame is None:
                continue
            frame_mask[sample_index, offset] = True
            width = max(1.0, float(frame.get("width", 1)))
            height = max(1.0, float(frame.get("height", 1)))
            person = _player(frame, hitter)
            if person is not None:
                points = person.get("keypoints_xy", [])
                scores = person.get("keypoint_scores", [])
                missing = person.get("keypoint_missing", [])
                if len(points) != 17 or len(scores) != 17 or len(missing) != 17:
                    raise ValueError(f"frame {frame_index} hitter pose must contain 17 COCO keypoints")
                for keypoint_index in range(17):
                    score = float(scores[keypoint_index])
                    valid = not bool(missing[keypoint_index]) and score >= keypoint_threshold
                    pose_confidence[sample_index, offset, keypoint_index] = score
                    if valid:
                        pose_xy[sample_index, offset, keypoint_index] = (
                            float(points[keypoint_index][0]) / width,
                            float(points[keypoint_index][1]) / height,
                        )
                        pose_mask[sample_index, offset, keypoint_index] = True
                for edge_index, (source_joint, target_joint) in enumerate(SKELETON_EDGES):
                    if pose_mask[sample_index, offset, source_joint] and pose_mask[sample_index, offset, target_joint]:
                        skeleton_vectors[sample_index, offset, edge_index] = (
                            pose_xy[sample_index, offset, target_joint] - pose_xy[sample_index, offset, source_joint]
                        )
                        skeleton_mask[sample_index, offset, edge_index] = True

            shuttle = frame.get("shuttle", {})
            shuttle_point = shuttle.get("smoothed_xy")
            if isinstance(shuttle_point, list) and len(shuttle_point) == 2:
                shuttle_xy[sample_index, offset] = (float(shuttle_point[0]) / width, float(shuttle_point[1]) / height)
                shuttle_mask[sample_index, offset] = True
                shuttle_measured_mask[sample_index, offset] = shuttle.get("tracking_state") == "measured"
            confidence = shuttle.get("confidence")
            if confidence is not None:
                shuttle_confidence[sample_index, offset] = float(confidence)

            hitter_position, hitter_valid = _court_position(frame, hitter)
            opponent_position, opponent_valid = _court_position(frame, opponent)
            if hitter_valid and hitter_position is not None:
                hitter_court_xy[sample_index, offset] = hitter_position
                hitter_position_mask[sample_index, offset] = True
            if opponent_valid and opponent_position is not None:
                opponent_court_xy[sample_index, offset] = opponent_position
                opponent_position_mask[sample_index, offset] = True

    arrays = {
        "pose_xy": pose_xy,
        "pose_confidence": pose_confidence,
        "pose_mask": pose_mask,
        "skeleton_vectors": skeleton_vectors,
        "skeleton_mask": skeleton_mask,
        "shuttle_xy": shuttle_xy,
        "shuttle_confidence": shuttle_confidence,
        "shuttle_mask": shuttle_mask,
        "shuttle_measured_mask": shuttle_measured_mask,
        "hitter_court_xy": hitter_court_xy,
        "opponent_court_xy": opponent_court_xy,
        "hitter_position_mask": hitter_position_mask,
        "opponent_position_mask": opponent_position_mask,
        "frame_mask": frame_mask,
        "frame_indices": frame_indices,
        "candidate_frames": candidate_frames,
        "event_ids": np.asarray(event_ids),
        "hitter_sides": np.asarray(hitter_sides),
        "labels": np.asarray(labels),
        "label_sources": np.asarray(label_sources),
        "landing_xy_m": landing_xy_m,
        "landing_mask": landing_mask,
        "landing_kinds": np.asarray(landing_kinds),
    }
    metadata = {
        "schema_version": "bst-input-1.0",
        "video_id": next(iter(video_ids), None),
        "sample_count": sample_count,
        "window_size": window_size,
        "candidate_offset": window_size // 2,
        "keypoint_threshold": keypoint_threshold,
        "skeleton_edges": [list(edge) for edge in SKELETON_EDGES],
        "normalization": {
            "pose_xy": "image width/height",
            "shuttle_xy": "image width/height",
            "court_xy": "6.1m x 13.4m",
        },
        "missing_policy": "zero-filled values are valid only where the corresponding mask is true",
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "array_dtypes": {name: str(value.dtype) for name, value in arrays.items()},
        "coverage": {
            "frame": float(frame_mask.mean()),
            "pose_keypoint": float(pose_mask.mean()),
            "shuttle": float(shuttle_mask.mean()),
            "shuttle_measured": float(shuttle_measured_mask.mean()),
            "hitter_position": float(hitter_position_mask.mean()),
            "opponent_position": float(opponent_position_mask.mean()),
            "landing_target": float(landing_mask.mean()),
        },
        "label_counts": {label: labels.count(label) for label in sorted(set(labels))},
    }
    return arrays, metadata


def export_bst_samples(
    frames_path: Path,
    events_path: Path,
    output_path: Path,
    *,
    window_size: int = 30,
    keypoint_threshold: float = 0.2,
    metadata_path: Path | None = None,
) -> dict[str, Any]:
    arrays, metadata = build_bst_samples(
        _read_jsonl(frames_path),
        _read_jsonl(events_path),
        window_size=window_size,
        keypoint_threshold=keypoint_threshold,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    resolved_metadata = metadata_path or Path(f"{output_path}.metadata.json")
    resolved_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**metadata, "output": str(output_path.resolve()), "metadata": str(resolved_metadata.resolve())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=Path, required=True, help="Enriched frame JSONL")
    parser.add_argument("--events", type=Path, required=True, help="Gold or candidate event JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Compressed .npz output")
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--keypoint-threshold", type=float, default=0.2)
    args = parser.parse_args()
    summary = export_bst_samples(
        args.frames,
        args.events,
        args.output,
        window_size=args.window_size,
        keypoint_threshold=args.keypoint_threshold,
        metadata_path=args.metadata,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
