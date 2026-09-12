import unittest

from badminton_pipeline.stroke_classification.destination import add_destination_predictions


class DestinationTests(unittest.TestCase):
    def test_next_contact_and_terminal_policy(self):
        frames = [
            {
                "frame_idx": 20,
                "player_tracks": {"lower": {"court_xy_m": [2.2, 3.2], "state": "measured"}},
                "shuttle": {
                    "associated_xy": [2.0, 3.0],
                    "tracking_state": "measured",
                    "trajectory_reliability": 1.0,
                    "innovation_ratio": 0.0,
                },
            }
        ]
        events = [
            {"event_id": "e1", "candidate_frame": 10, "rally_id": "R"},
            {"event_id": "e2", "candidate_frame": 20, "rally_id": "R", "predicted_hitter": "lower"},
        ]
        rows, summary = add_destination_predictions(
            frames, events, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        self.assertEqual(rows[0]["landing_kind"], "next_contact")
        self.assertEqual(rows[0]["predicted_landing_x"], 2.08)
        self.assertEqual(rows[0]["predicted_landing_y"], 3.08)
        self.assertEqual(rows[0]["landing_proxy_kind"], "shuttle_receiver_fusion")
        self.assertEqual(rows[1]["landing_status"], "unavailable_terminal_ground_contact")
        self.assertIsNone(rows[1]["predicted_landing_x"])
        self.assertEqual(summary["predicted_next_contact"], 1)

    def test_terminal_ground_contact_requires_track_end_evidence(self):
        frames = []
        for frame_idx, point in ((14, [2.0, 2.0]), (15, [2.2, 2.5]), (16, [2.4, 3.0])):
            frames.append({
                "frame_idx": frame_idx,
                "play_state_segment_id": "P1",
                "shuttle": {
                    "associated_xy": point,
                    "tracking_state": "measured",
                    "trajectory_reliability": 0.9,
                    "innovation_ratio": 0.0,
                },
            })
        for frame_idx in (17, 18, 19):
            frames.append({
                "frame_idx": frame_idx,
                "play_state_segment_id": "P1",
                "shuttle": {"tracking_state": "missing"},
            })
        events = [{"event_id": "e1", "candidate_frame": 10, "rally_id": "R",
                   "play_state_segment_id": "P1"}]
        rows, summary = add_destination_predictions(
            frames, events, [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        )
        self.assertEqual(rows[0]["landing_status"], "experimental_terminal_track_end")
        self.assertEqual(rows[0]["predicted_landing_x"], 2.4)
        self.assertLessEqual(rows[0]["landing_confidence"], 0.5)
        self.assertEqual(summary["predicted_terminal_ground_contact"], 1)


if __name__ == "__main__":
    unittest.main()
