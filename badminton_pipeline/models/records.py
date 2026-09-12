"""Dependency-free normalized records for model inference output."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


@dataclass(frozen=True)
class PersonPose:
    identity: str
    bbox_xyxy: list[float]
    keypoints_xy: list[list[float]]
    keypoint_scores: list[float]
    keypoint_missing: list[bool]

    def __post_init__(self) -> None:
        if len(self.keypoints_xy) != 17:
            raise ValueError("COCO pose must contain exactly 17 keypoints")
        if len(self.keypoint_scores) != 17 or len(self.keypoint_missing) != 17:
            raise ValueError("pose score and missing arrays must contain 17 values")


@dataclass(frozen=True)
class ShuttleCandidate:
    raw_xy: list[float]
    confidence: float
    bbox_xyxy: list[float]
    detector_index: int


@dataclass(frozen=True)
class ShuttleDetection:
    raw_xy: list[float] | None
    confidence: float | None
    state: str
    bbox_xyxy: list[float] | None = None
    smoothed_xy: list[float] | None = None
    candidates: list[ShuttleCandidate] = field(default_factory=list)


@dataclass(frozen=True)
class FrameDetections:
    schema_version: str
    video_id: str
    frame_idx: int
    timestamp_sec: float
    width: int
    height: int
    players: list[PersonPose]
    shuttle: ShuttleDetection
    models: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
