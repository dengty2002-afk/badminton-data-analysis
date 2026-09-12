"""Adapters for the Good-Badminton RTMPose and shuttle models.

Heavy dependencies are imported lazily so metadata and schema tests stay fast.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from .records import PersonPose, ShuttleCandidate, ShuttleDetection


class ModelSetupError(RuntimeError):
    """Raised when model weights or runtime dependencies are unavailable."""


def load_and_verify_manifest(manifest_path: Path, models_dir: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for model_name, spec in manifest["models"].items():
        model_path = models_dir / spec["filename"]
        if not model_path.is_file():
            raise ModelSetupError(f"missing {model_name} weight: {model_path}")
        if model_path.stat().st_size != spec["size_bytes"]:
            raise ModelSetupError(f"unexpected size for {model_name} weight: {model_path}")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if digest != spec["sha256"]:
            raise ModelSetupError(f"SHA-256 mismatch for {model_name} weight: {model_path}")
    return manifest


def _bbox_from_keypoints(points: list[list[float]], missing: list[bool]) -> list[float]:
    visible = [point for point, is_missing in zip(points, missing) if not is_missing]
    if not visible:
        return [0.0, 0.0, 0.0, 0.0]
    xs = [point[0] for point in visible]
    ys = [point[1] for point in visible]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_poses(
    keypoints: Any,
    scores: Any,
    *,
    confidence_threshold: float = 0.2,
    max_players: int = 12,
) -> list[PersonPose]:
    """Convert RTMLib output to full COCO-17 records and assign court-side IDs."""
    if keypoints is None or scores is None:
        return []

    normalized: list[tuple[float, list[float], list[list[float]], list[float], list[bool]]] = []
    for person_points, person_scores in zip(keypoints, scores):
        points = [[float(point[0]), float(point[1])] for point in person_points[:17]]
        confidences = [float(value) for value in person_scores[:17]]
        if len(points) != 17 or len(confidences) != 17:
            continue
        missing = [confidence < confidence_threshold for confidence in confidences]
        bbox = _bbox_from_keypoints(points, missing)
        visible_scores = [score for score, absent in zip(confidences, missing) if not absent]
        if not visible_scores:
            continue
        rank_y = bbox[3]
        normalized.append((rank_y, bbox, points, confidences, missing))

    # Keep multiple people here. Court calibration selects the actual two players
    # later; early image-y filtering can confuse officials or spectators for players.
    normalized.sort(key=lambda item: item[0])
    normalized = normalized[:max_players]
    return [
        PersonPose(
            identity=f"candidate_{index}",
            bbox_xyxy=bbox,
            keypoints_xy=points,
            keypoint_scores=confidences,
            keypoint_missing=missing,
        )
        for index, (_, bbox, points, confidences, missing) in enumerate(normalized)
    ]


class RTMPoseAdapter:
    def __init__(
        self,
        detector_path: Path,
        pose_path: Path,
        *,
        device: str = "cuda",
        confidence_threshold: float = 0.2,
    ) -> None:
        if device == "cuda":
            try:
                import torch
            except ImportError as exc:
                raise ModelSetupError("CUDA mode requires the pinned PyTorch CUDA runtime") from exc
            if not torch.cuda.is_available():
                raise ModelSetupError("PyTorch cannot access CUDA on this machine")

        try:
            from rtmlib import Body
        except ImportError as exc:
            raise ModelSetupError(f"RTMPose runtime dependency is unavailable: {exc}") from exc

        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = Body(
            det=str(detector_path),
            det_input_size=(416, 416),
            pose=str(pose_path),
            pose_input_size=(192, 256),
            backend="onnxruntime",
            device=device,
        )
        if device == "cuda":
            sessions = (self.model.det_model.session, self.model.pose_model.session)
            active = [session.get_providers()[0] for session in sessions]
            if active != ["CUDAExecutionProvider", "CUDAExecutionProvider"]:
                raise ModelSetupError(
                    "CUDA was requested but ONNX Runtime fell back to CPU; "
                    f"active providers: {active}. Install the CUDA 13 and cuDNN 9 runtime DLLs."
                )

    def infer(self, frame: Any) -> list[PersonPose]:
        keypoints, scores = self.model(frame)
        return normalize_poses(
            keypoints,
            scores,
            confidence_threshold=self.confidence_threshold,
        )


def normalize_shuttle_detections(
    confidences: list[float],
    xyxy: list[list[float]],
    *,
    top_k: int = 5,
) -> ShuttleDetection:
    """Keep detector-ranked candidates while preserving the legacy best-box fields."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    ranked = sorted(
        range(min(len(confidences), len(xyxy))),
        key=lambda index: float(confidences[index]),
        reverse=True,
    )[:top_k]
    candidates: list[ShuttleCandidate] = []
    for detector_index in ranked:
        x1, y1, x2, y2 = (float(value) for value in xyxy[detector_index])
        candidates.append(ShuttleCandidate(
            raw_xy=[(x1 + x2) / 2.0, (y1 + y2) / 2.0],
            confidence=float(confidences[detector_index]),
            bbox_xyxy=[x1, y1, x2, y2],
            detector_index=detector_index,
        ))
    if not candidates:
        return ShuttleDetection(raw_xy=None, confidence=None, state="missing")
    best = candidates[0]
    return ShuttleDetection(
        raw_xy=best.raw_xy,
        confidence=best.confidence,
        state="visible",
        bbox_xyxy=best.bbox_xyxy,
        candidates=candidates,
    )


class ShuttleAdapter:
    def __init__(
        self,
        weight_path: Path,
        *,
        device: str = "cuda",
        confidence: float = 0.18,
        top_k: int = 5,
    ) -> None:
        settings_dir = weight_path.parent / ".ultralytics"
        settings_dir.mkdir(parents=True, exist_ok=True)
        matplotlib_dir = settings_dir / "matplotlib"
        matplotlib_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(settings_dir.resolve()))
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir.resolve()))
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelSetupError("ultralytics is not installed; install the model runtime first") from exc
        self.model = YOLO(str(weight_path))
        self.device = 0 if device == "cuda" else "cpu"
        self.confidence = confidence
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.top_k = top_k

    def infer(self, frame: Any) -> ShuttleDetection:
        result = self.model(frame, conf=self.confidence, device=self.device, verbose=False)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return ShuttleDetection(raw_xy=None, confidence=None, state="missing")

        return normalize_shuttle_detections(
            boxes.conf.detach().cpu().tolist(),
            boxes.xyxy.detach().cpu().tolist(),
            top_k=self.top_k,
        )


def model_metadata(manifest: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        "source": manifest["source"]["name"],
        "release": manifest["source"]["release"],
        "device": device,
        "pose": {
            "family": "RTMPose",
            "mode": "balanced",
            "weights_sha256": manifest["models"]["pose"]["sha256"],
        },
        "person_detector": {
            "family": "YOLOX nano",
            "weights_sha256": manifest["models"]["person_detector"]["sha256"],
        },
        "shuttle": {
            "family": "YOLO11s ball",
            "weights_sha256": manifest["models"]["shuttle"]["sha256"],
        },
    }
