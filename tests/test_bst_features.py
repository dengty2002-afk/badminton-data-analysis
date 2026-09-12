import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from badminton_pipeline.stroke_classification.features import build_bst_samples, export_bst_samples


def player(identity: str, *, missing_wrist: bool = False):
    missing = [False] * 17
    missing[9] = missing_wrist
    return {
        "identity": identity,
        "keypoints_xy": [[20.0 + index, 40.0 + index] for index in range(17)],
        "keypoint_scores": [0.9] * 17,
        "keypoint_missing": missing,
        "court_xy_m": [3.05, 2.0 if identity == "upper" else 11.4],
    }


def frame(index: int, *, missing_wrist: bool = False):
    return {
        "video_id": "video-1",
        "frame_idx": index,
        "width": 100,
        "height": 200,
        "players": [player("upper", missing_wrist=missing_wrist), player("lower")],
        "player_tracks": {
            "upper": {"court_xy_m": [3.05, 2.0], "state": "measured"},
            "lower": {"court_xy_m": [3.05, 11.4], "state": "measured"},
        },
        "shuttle": {
            "smoothed_xy": [50.0, 100.0],
            "confidence": 0.8,
            "tracking_state": "measured",
        },
    }


def event():
    return {
        "video_id": "video-1",
        "event_id": "event-11",
        "candidate_frame": 11,
        "predicted_hitter": "upper",
        "human_stroke_type": "杀球",
        "human_landing_x": 2.5,
        "human_landing_y": 10.0,
        "landing_kind": "next_contact",
        "label_source": "gold",
    }


class BstFeatureTests(unittest.TestCase):
    def test_builds_multimodal_window_with_masks(self):
        arrays, metadata = build_bst_samples(
            [frame(10), frame(11, missing_wrist=True), frame(12)],
            [event()],
            window_size=3,
        )

        self.assertEqual(arrays["pose_xy"].shape, (1, 3, 17, 2))
        self.assertEqual(arrays["skeleton_vectors"].shape, (1, 3, 16, 2))
        np.testing.assert_allclose(arrays["shuttle_xy"][0, 1], [0.5, 0.5])
        np.testing.assert_allclose(arrays["hitter_court_xy"][0, 1], [0.5, 2.0 / 13.4])
        np.testing.assert_allclose(arrays["opponent_court_xy"][0, 1], [0.5, 11.4 / 13.4])
        self.assertFalse(arrays["pose_mask"][0, 1, 9])
        self.assertTrue(arrays["frame_mask"].all())
        self.assertEqual(arrays["labels"].tolist(), ["杀球"])
        np.testing.assert_allclose(arrays["landing_xy_m"], [[2.5, 10.0]])
        self.assertTrue(arrays["landing_mask"].all())
        self.assertEqual(arrays["landing_kinds"].tolist(), ["next_contact"])
        self.assertEqual(metadata["candidate_offset"], 1)
        self.assertIn("corresponding mask", metadata["missing_policy"])

    def test_missing_frames_remain_zero_and_masked(self):
        arrays, _ = build_bst_samples([frame(11)], [event()], window_size=3)
        self.assertEqual(arrays["frame_mask"].tolist(), [[False, True, False]])
        self.assertEqual(float(arrays["pose_xy"][0, 0].sum()), 0.0)
        self.assertFalse(arrays["shuttle_mask"][0, 0])

    def test_rejects_mixed_video_inputs(self):
        other = event()
        other["video_id"] = "video-2"
        with self.assertRaisesRegex(ValueError, "one video_id"):
            build_bst_samples([frame(11)], [other], window_size=3)

    def test_exports_npz_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames_path = root / "frames.jsonl"
            events_path = root / "events.jsonl"
            output_path = root / "samples.npz"
            frames_path.write_text("\n".join(json.dumps(frame(index)) for index in (10, 11, 12)), encoding="utf-8")
            events_path.write_text(json.dumps(event()), encoding="utf-8")

            summary = export_bst_samples(frames_path, events_path, output_path, window_size=3)

            self.assertTrue(output_path.exists())
            self.assertTrue(Path(summary["metadata"]).exists())
            with np.load(output_path) as bundle:
                self.assertEqual(bundle["event_ids"].tolist(), ["event-11"])
                self.assertEqual(bundle["pose_xy"].shape, (1, 3, 17, 2))
                np.testing.assert_allclose(bundle["landing_xy_m"], [[2.5, 10.0]])


if __name__ == "__main__":
    unittest.main()
