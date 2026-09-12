import unittest

import numpy as np

from badminton_pipeline.stroke_classification.bst_adapter import (
    BASE_TYPES,
    FINE_CLASSES,
    OFFICIAL_18_UNSUPPORTED,
    build_official_bst_inputs,
)


def person(identity: str, y: float):
    return {
        "identity": identity,
        "bbox_xyxy": [10.0, 20.0, 60.0, 120.0],
        "keypoints_xy": [[20.0 + i, 40.0 + i] for i in range(17)],
        "keypoint_scores": [0.9] * 17,
        "keypoint_missing": [False] * 17,
        "court_xy_m": [3.05, y],
    }


def frame(index: int):
    return {
        "video_id": "v",
        "frame_idx": index,
        "width": 100,
        "height": 200,
        "players": [person("upper", 2.0), person("lower", 11.0)],
        "player_tracks": {
            "upper": {"court_xy_m": [3.05, 2.0]},
            "lower": {"court_xy_m": [3.05, 11.0]},
        },
        "shuttle": {"smoothed_xy": [50.0, 100.0]},
    }


class BstAdapterTests(unittest.TestCase):
    def test_official_shape_and_normalization(self):
        frames = [frame(index) for index in range(0, 31)]
        events = [{"event_id": "e", "candidate_frame": 15, "rally_id": "Rally 01"}]
        arrays, metadata = build_official_bst_inputs(frames, events)
        self.assertEqual(arrays["jnb"].shape, (1, 100, 2, 72))
        self.assertEqual(arrays["positions"].shape, (1, 100, 2, 2))
        self.assertEqual(arrays["shuttle"].shape, (1, 100, 2))
        np.testing.assert_allclose(arrays["shuttle"][0, 0], [0.5, 0.5])
        np.testing.assert_allclose(arrays["positions"][0, 0, 0], [0.5, 2 / 13.4])
        self.assertEqual(int(arrays["video_lengths"][0]), 31)
        self.assertGreater(metadata["mean_input_quality"], 0.9)

    def test_vocabulary_gap_is_explicit(self):
        self.assertEqual(len(BASE_TYPES), 17)
        self.assertEqual(len(FINE_CLASSES), 35)
        self.assertEqual(OFFICIAL_18_UNSUPPORTED, ("driven flight",))


if __name__ == "__main__":
    unittest.main()
