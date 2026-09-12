"""Adapter from ShuttleLab frame records to the official BST inference contract.

The upstream model predicts 17 ShuttleSet stroke types separately for the far
and near players plus an unknown class. This adapter preserves that provenance,
collapses the player-side prefix, and deliberately leaves ShuttleSet's missing
``driven flight`` class unsupported instead of silently inventing a label.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


COURT_WIDTH_M = 6.1
COURT_LENGTH_M = 13.4
SEQUENCE_LENGTH = 100
BONE_PAIRS = (
    (0, 1), (0, 2), (1, 2), (1, 3), (2, 4),
    (3, 5), (4, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)

# Exact upstream order, translated to the official ShuttleSet English labels.
BASE_TYPES = (
    "net shot", "return net", "smash", "wrist smash", "lob",
    "defensive return lob", "clear", "drive", "back-court drive", "drop",
    "passive drop", "push", "rush", "defensive return drive",
    "cross-court net shot", "short service", "long service",
)
OFFICIAL_18_UNSUPPORTED = ("driven flight",)
FINE_CLASSES = tuple(f"Top_{name}" for name in BASE_TYPES) + tuple(
    f"Bottom_{name}" for name in BASE_TYPES
) + ("unknown",)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _person(frame: dict[str, Any], identity: str) -> dict[str, Any] | None:
    return next((row for row in frame.get("players", []) if row.get("identity") == identity), None)


def _normalize_pose(person: dict[str, Any] | None, threshold: float) -> tuple[np.ndarray, float]:
    pose = np.zeros((17, 2), dtype=np.float32)
    if person is None:
        return pose, 0.0
    bbox = person.get("bbox_xyxy")
    points = person.get("keypoints_xy")
    scores = person.get("keypoint_scores")
    missing = person.get("keypoint_missing")
    if not (
        isinstance(bbox, list) and len(bbox) == 4 and isinstance(points, list)
        and isinstance(scores, list) and isinstance(missing, list)
        and len(points) == len(scores) == len(missing) == 17
    ):
        return pose, 0.0
    diagonal = math.hypot(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))
    if diagonal <= 1e-6:
        return pose, 0.0
    valid = 0
    for index, point in enumerate(points):
        if bool(missing[index]) or float(scores[index]) < threshold:
            continue
        pose[index] = (
            (float(point[0]) - float(bbox[0])) / diagonal,
            (float(point[1]) - float(bbox[1])) / diagonal,
        )
        valid += 1
    return pose, valid / 17.0


def _court_position(frame: dict[str, Any], identity: str) -> tuple[np.ndarray, bool]:
    track = frame.get("player_tracks", {}).get(identity, {})
    point = track.get("court_xy_m")
    if point is None:
        row = _person(frame, identity)
        point = None if row is None else row.get("court_xy_m")
    if not isinstance(point, list) or len(point) != 2:
        return np.zeros(2, dtype=np.float32), False
    return np.asarray(
        [float(point[0]) / COURT_WIDTH_M, float(point[1]) / COURT_LENGTH_M],
        dtype=np.float32,
    ), True


def _segment_bounds(
    event_index: int, events: Sequence[dict[str, Any]], available: set[int]
) -> tuple[int, int]:
    event = events[event_index]
    current = int(event["candidate_frame"])
    rally = str(event.get("rally_id") or "")
    previous = next(
        (
            int(events[index]["candidate_frame"])
            for index in range(event_index - 1, -1, -1)
            if str(events[index].get("rally_id") or "") == rally
        ),
        None,
    )
    following = next(
        (
            int(events[index]["candidate_frame"])
            for index in range(event_index + 1, len(events))
            if str(events[index].get("rally_id") or "") == rally
        ),
        None,
    )
    # Upstream seq100 clips span the interval between adjacent contacts. When
    # either neighbor is missing, use a conservative one-second shoulder.
    start = (previous + current) // 2 if previous is not None else current - 30
    end = (current + following) // 2 if following is not None else current + 30
    if available:
        start = max(start, min(available))
        end = min(end, max(available))
    return start, max(start, end)


def _resample_indices(start: int, end: int, target: int = SEQUENCE_LENGTH) -> tuple[list[int], int]:
    length = max(1, end - start + 1)
    if length <= target:
        return list(range(start, end + 1)), length
    stride = length // target + int(length % target > target // 2)
    indices = list(range(start, end + 1, stride))[:target]
    return indices, len(indices)


def build_official_bst_inputs(
    frames: Iterable[dict[str, Any]],
    events: Iterable[dict[str, Any]],
    *,
    keypoint_threshold: float = 0.2,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Create upstream-compatible J+B, position, shuttle, and length arrays."""
    frame_list = sorted(frames, key=lambda row: int(row["frame_idx"]))
    event_list = sorted(events, key=lambda row: int(row["candidate_frame"]))
    if not frame_list or not event_list:
        raise ValueError("BST inference requires both enriched frames and events")
    frame_by_index = {int(row["frame_idx"]): row for row in frame_list}
    available = set(frame_by_index)
    sample_count = len(event_list)
    joints = np.zeros((sample_count, SEQUENCE_LENGTH, 2, 17, 2), dtype=np.float32)
    positions = np.zeros((sample_count, SEQUENCE_LENGTH, 2, 2), dtype=np.float32)
    shuttle = np.zeros((sample_count, SEQUENCE_LENGTH, 2), dtype=np.float32)
    video_lengths = np.ones(sample_count, dtype=np.int64)
    sample_quality = np.zeros(sample_count, dtype=np.float32)
    event_ids: list[str] = []
    candidate_frames: list[int] = []

    for sample, event in enumerate(event_list):
        start, end = _segment_bounds(sample, event_list, available)
        indices, video_length = _resample_indices(start, end)
        video_lengths[sample] = video_length
        event_ids.append(str(event.get("event_id") or f"event_{event['candidate_frame']}"))
        candidate_frames.append(int(event["candidate_frame"]))
        qualities: list[float] = []
        for offset, frame_index in enumerate(indices):
            frame = frame_by_index.get(frame_index)
            if frame is None:
                continue
            width = max(1.0, float(frame.get("width", 1)))
            height = max(1.0, float(frame.get("height", 1)))
            pose_coverage = []
            for person_index, identity in enumerate(("upper", "lower")):
                normalized, coverage = _normalize_pose(_person(frame, identity), keypoint_threshold)
                joints[sample, offset, person_index] = normalized
                pose_coverage.append(coverage)
                position, valid = _court_position(frame, identity)
                if valid:
                    positions[sample, offset, person_index] = position
            shuttle_point = frame.get("shuttle", {}).get("smoothed_xy")
            shuttle_valid = isinstance(shuttle_point, list) and len(shuttle_point) == 2
            if shuttle_valid:
                shuttle[sample, offset] = (
                    float(shuttle_point[0]) / width,
                    float(shuttle_point[1]) / height,
                )
            qualities.append(min(pose_coverage, default=0.0) * (1.0 if shuttle_valid else 0.0))
        sample_quality[sample] = float(sum(qualities) / max(1, len(indices)))

    bones = np.zeros((sample_count, SEQUENCE_LENGTH, 2, len(BONE_PAIRS), 2), dtype=np.float32)
    for bone_index, (source, target) in enumerate(BONE_PAIRS):
        source_joint = joints[:, :, :, source]
        target_joint = joints[:, :, :, target]
        valid = (source_joint != 0.0) & (target_joint != 0.0)
        bones[:, :, :, bone_index] = np.where(valid, target_joint - source_joint, 0.0)
    jnb = np.concatenate((joints, bones), axis=3).reshape(
        sample_count, SEQUENCE_LENGTH, 2, -1
    )
    arrays = {
        "jnb": jnb,
        "positions": positions,
        "shuttle": shuttle,
        "video_lengths": video_lengths,
        "sample_quality": sample_quality,
        "event_ids": np.asarray(event_ids),
        "candidate_frames": np.asarray(candidate_frames, dtype=np.int32),
    }
    metadata = {
        "schema_version": "bst-official-input-1.0",
        "sample_count": sample_count,
        "sequence_length": SEQUENCE_LENGTH,
        "pose_normalization": "bbox origin divided by bbox diagonal",
        "position_normalization": "6.1m x 13.4m court",
        "shuttle_normalization": "image width x height",
        "window_policy": "midpoints between adjacent contacts within rally, one-second shoulder at ends",
        "mean_input_quality": float(sample_quality.mean()),
    }
    return arrays, metadata


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_upstream_model(weight_path: Path, device: str) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for BST inference") from exc
    root = Path(__file__).resolve().parents[2]
    source = root / "vendor" / "BST-Badminton-Stroke-type-Transformer" / "stroke_classification"
    if not source.is_dir():
        raise RuntimeError(f"vendored BST source is unavailable: {source}")
    sys.path.insert(0, str(source))
    try:
        module = importlib.import_module("model.bst")
        network = module.BST_0(in_dim=72, seq_len=SEQUENCE_LENGTH, n_class=len(FINE_CLASSES))
    finally:
        try:
            sys.path.remove(str(source))
        except ValueError:
            pass
    resolved = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    state = torch.load(str(weight_path), map_location=resolved, weights_only=True)
    network.load_state_dict(state)
    network.to(resolved).eval()
    return network, torch, resolved


def predict_arrays(
    arrays: dict[str, np.ndarray], weight_path: Path, *, device: str = "cuda", batch_size: int = 64
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    network, torch, resolved_device = _load_upstream_model(weight_path, device)
    count = int(arrays["jnb"].shape[0])
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            jnb = torch.from_numpy(arrays["jnb"][start:end]).to(resolved_device)
            shuttle = torch.from_numpy(arrays["shuttle"][start:end]).to(resolved_device)
            positions = torch.from_numpy(arrays["positions"][start:end]).to(resolved_device)
            lengths = torch.from_numpy(arrays["video_lengths"][start:end]).to(resolved_device)
            # The selected BST-0 checkpoint uses pose + shuttle. Position
            # arrays are still prepared and audited so a position-fusion BST
            # checkpoint can be substituted without changing the data contract.
            probabilities = torch.softmax(network(jnb, shuttle, lengths), dim=1).cpu().numpy()
            for local_index, probability in enumerate(probabilities):
                index = start + local_index
                order = np.argsort(probability)[::-1]
                best = int(order[0])
                raw_class = FINE_CLASSES[best]
                label = raw_class.split("_", 1)[1] if "_" in raw_class else "unknown"
                confidence = float(probability[best])
                quality = float(arrays["sample_quality"][index])
                status = "unknown" if label == "unknown" else "predicted"
                if status == "predicted" and (confidence < 0.35 or quality < 0.55):
                    status = "low_confidence"
                predictions.append(
                    {
                        "event_id": str(arrays["event_ids"][index]),
                        "candidate_frame": int(arrays["candidate_frames"][index]),
                        "predicted_stroke_type": label,
                        "stroke_type_confidence": round(confidence, 6),
                        "stroke_type_input_quality": round(quality, 6),
                        "stroke_type_status": status,
                        "stroke_type_source": "bst_shuttleset_35_side_collapsed",
                        "stroke_type_raw_class": raw_class,
                        "stroke_type_top3": [
                            {"raw_class": FINE_CLASSES[int(rank)], "probability": round(float(probability[int(rank)]), 6)}
                            for rank in order[:3]
                        ],
                        "semantic_review_status": "unreviewed",
                        "semantic_label_source": "automatic_prelabel",
                    }
                )
    summary = {
        "schema_version": "bst-prediction-1.0",
        "model": "BST_0 seq100 fine35 serial2",
        "device": resolved_device,
        "weight": str(weight_path.resolve()),
        "weight_sha256": _sha256(weight_path),
        "prediction_count": len(predictions),
        "unsupported_official_types": list(OFFICIAL_18_UNSUPPORTED),
        "upstream_reported_accuracy": 0.760,
        "upstream_reported_macro_f1": 0.691,
        "upstream_reported_top2_accuracy": 0.930,
        "local_semantic_gold_evaluated": False,
        "truth_status": "automatic prelabels, not ground truth",
    }
    return predictions, summary


def annotate_event_rows(
    frames_path: Path,
    events_path: Path,
    output_path: Path,
    weight_path: Path,
    *,
    device: str = "cuda",
    keypoint_threshold: float = 0.2,
) -> dict[str, Any]:
    frames = _read_jsonl(frames_path)
    events = _read_jsonl(events_path)
    arrays, input_summary = build_official_bst_inputs(
        frames, events, keypoint_threshold=keypoint_threshold
    )
    predictions, model_summary = predict_arrays(arrays, weight_path, device=device)
    by_id = {row["event_id"]: row for row in predictions}
    output_rows: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        prediction = by_id.get(str(row.get("event_id")))
        if prediction:
            row.update(prediction)
        output_rows.append(row)
    _write_jsonl(output_path, output_rows)
    return {**model_summary, "input": input_summary, "output": str(output_path.resolve())}
