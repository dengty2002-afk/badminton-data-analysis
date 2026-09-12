import unittest

from badminton_pipeline.tracking.play_state import annotate_play_state_evidence


class PlayStateEvidenceTests(unittest.TestCase):
    def test_attaches_auditable_score_without_filtering_event(self):
        records = []
        for frame in range(11):
            records.append({
                "frame_idx": frame,
                "width": 100,
                "height": 100,
                "shuttle": {
                    "smoothed_xy": [float(frame * 2), 20.0],
                    "tracking_state": "measured",
                },
                "player_tracks": {
                    "upper": {"state": "measured"},
                    "lower": {"state": "measured"},
                },
            })
        events = [{"candidate_frame": 5}, {"candidate_frame": 9}]
        result = annotate_play_state_evidence(
            records, events, fps=10, window_seconds=1.0, candidate_support_seconds=1.0,
        )
        self.assertEqual(len(result), 2)
        self.assertFalse(result[0]["play_state_filter_applied"])
        self.assertEqual(result[0]["play_state_decision"], "score_only_unreviewed")
        self.assertEqual(result[0]["play_state_shuttle_measured_ratio"], 1.0)
        self.assertEqual(result[0]["play_state_shuttle_tracked_ratio"], 1.0)
        self.assertEqual(result[0]["play_state_both_player_tracking_ratio"], 1.0)
        self.assertEqual(result[0]["play_state_nearby_candidate_count"], 2)
        self.assertEqual(result[0]["play_state_window_seconds"], 1.0)
        self.assertEqual(result[0]["play_state_window_frames"], 11)
        self.assertEqual(result[0]["play_state_window_radius_frames"], 5)
        self.assertEqual(result[0]["play_state_support_seconds"], 1.0)
        self.assertEqual(result[0]["play_state_support_radius_frames"], 10)
        self.assertEqual(result[0]["play_state_sequence_support"], 0.5)
        self.assertGreater(result[0]["play_state_score"], 0.9)

    def test_rejects_invalid_windows(self):
        with self.assertRaisesRegex(ValueError, "window_seconds"):
            annotate_play_state_evidence([], [], fps=30, window_seconds=0)


if __name__ == "__main__":
    unittest.main()
