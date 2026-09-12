import unittest

from badminton_pipeline.exports.demo_feed import build_demo_feed


class DemoExportTests(unittest.TestCase):
    def test_maps_pipeline_event_to_review_feed(self):
        event = {
            "candidate_frame": 347,
            "candidate_time": 11.566,
            "window_start_frame": 302,
            "window_end_frame": 392,
            "predicted_hitter": "upper",
            "event_confidence": 0.8274,
            "trajectory_score": 0.88,
            "hand_distance_score": 0.93,
            "pose_score": 0.42,
            "model_version": "hit-rules-0.2.0",
            "calibration_version": 1,
            "rally_id": "Rally 07",
        }
        feed = build_demo_feed([event])
        self.assertEqual(feed["pipeline_version"], "hit-rules-0.2.0")
        self.assertEqual(feed["events"][0]["id"], "EVT-M00347")
        self.assertEqual(feed["events"][0]["hitter"], "A")
        self.assertEqual(feed["events"][0]["hitter_side"], "upper")
        self.assertEqual(feed["events"][0]["stroke"], "待标注")
        self.assertEqual(feed["events"][0]["rally"], "Rally 07")
        self.assertEqual(feed["events"][0]["confidence"], 0.827)

    def test_lower_maps_to_player_b(self):
        event = {
            "candidate_frame": 1,
            "candidate_time": 0.1,
            "window_start_frame": 0,
            "window_end_frame": 2,
            "predicted_hitter": "lower",
            "event_confidence": 0.5,
            "trajectory_score": 0.5,
            "hand_distance_score": 0.5,
            "pose_score": 0.5,
        }
        self.assertEqual(build_demo_feed([event])["events"][0]["hitter"], "B")


if __name__ == "__main__":
    unittest.main()
