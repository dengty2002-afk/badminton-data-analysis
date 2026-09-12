"""Model adapters and normalized frame-level detection records."""

from .records import COCO_KEYPOINT_NAMES, FrameDetections, PersonPose, ShuttleDetection

__all__ = [
    "COCO_KEYPOINT_NAMES",
    "FrameDetections",
    "PersonPose",
    "ShuttleDetection",
]
