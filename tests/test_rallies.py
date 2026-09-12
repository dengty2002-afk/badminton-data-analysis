import unittest

from badminton_pipeline.tracking.rallies import assign_rally_ids


class RallySegmentationTests(unittest.TestCase):
    def test_long_gap_starts_new_rally_candidate(self):
        events = [
            {"candidate_frame": 10, "candidate_time": 1.0},
            {"candidate_frame": 25, "candidate_time": 1.5},
            {"candidate_frame": 300, "candidate_time": 10.0},
        ]
        assign_rally_ids(events, gap_seconds=6.0)
        self.assertEqual([event["rally_id"] for event in events], ["Rally 01", "Rally 01", "Rally 02"])
        self.assertEqual(events[2]["rally_boundary_source"], "time_gap")
        self.assertEqual(events[2]["rally_boundary_status"], "candidate")

    def test_invalid_gap_is_rejected(self):
        with self.assertRaises(ValueError):
            assign_rally_ids([], gap_seconds=0)


if __name__ == "__main__":
    unittest.main()
