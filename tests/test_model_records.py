import json
from pathlib import Path
import unittest

from badminton_pipeline.models.good_badminton import (
    load_and_verify_manifest,
    normalize_poses,
    normalize_shuttle_detections,
)


ROOT = Path(__file__).resolve().parents[1]


class ModelRecordTests(unittest.TestCase):
    @unittest.skipUnless(all((ROOT / "models" / "good-badminton" / s["filename"]).is_file() for s in json.loads((ROOT / "configs" / "models.good-badminton.json").read_text(encoding="utf8"))["models"].values()), "Optional downloaded weights absent; run scripts/download_models.py for integration verification")
    def test_manifest_weights_are_present_and_verified(self):
        manifest = load_and_verify_manifest(
            ROOT / "configs" / "models.good-badminton.json",
            ROOT / "models" / "good-badminton",
        )
        self.assertEqual(manifest["source"]["release"], "v0.1.0")

    def test_pose_keeps_all_coco_keypoints_and_assigns_sides(self):
        upper = [[float(index), 100.0 + index] for index in range(17)]
        lower = [[float(index), 500.0 + index] for index in range(17)]
        poses = normalize_poses([lower, upper], [[0.9] * 17, [0.8] * 17])
        self.assertEqual([pose.identity for pose in poses], ["candidate_0", "candidate_1"])
        self.assertEqual(len(poses[0].keypoints_xy), 17)
        self.assertFalse(any(poses[0].keypoint_missing))

    def test_low_confidence_point_is_marked_missing_without_erasing_coordinates(self):
        points = [[float(index), float(index)] for index in range(17)]
        scores = [0.9] * 17
        scores[9] = 0.1
        pose = normalize_poses([points], [scores], confidence_threshold=0.2)[0]
        self.assertTrue(pose.keypoint_missing[9])
        self.assertEqual(pose.keypoints_xy[9], [9.0, 9.0])

    def test_shuttle_normalization_keeps_ranked_top_k_candidates(self):
        detection = normalize_shuttle_detections(
            [0.4, 0.9, 0.6],
            [[0, 0, 2, 2], [10, 20, 14, 24], [30, 40, 32, 42]],
            top_k=2,
        )
        self.assertEqual(detection.raw_xy, [12.0, 22.0])
        self.assertEqual([item.confidence for item in detection.candidates], [0.9, 0.6])
        self.assertEqual([item.detector_index for item in detection.candidates], [1, 2])

    def test_shuttle_normalization_rejects_invalid_top_k(self):
        with self.assertRaisesRegex(ValueError, "top_k"):
            normalize_shuttle_detections([], [], top_k=0)


if __name__ == "__main__":
    unittest.main()
