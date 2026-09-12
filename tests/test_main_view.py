import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from badminton_pipeline.video.main_view import adaptive_threshold, segments_from_flags, stabilize_views
from badminton_pipeline.video.main_view import run


class MainViewTests(unittest.TestCase):
    def test_adapts_tight_cross_event_score_mode(self):
        self.assertAlmostEqual(adaptive_threshold([0.73, 0.75, 0.77, 0.78], 0.8), 0.718)

    def test_does_not_adapt_diverse_camera_scores(self):
        self.assertEqual(adaptive_threshold([0.1, 0.4, 0.75, 0.82], 0.8), 0.8)

    def test_falls_back_to_good_badminton_default_for_near_miss(self):
        self.assertEqual(adaptive_threshold([0.1, 0.4, 0.76, 0.798], 0.8), 0.75)

    def test_falls_back_when_strict_cross_event_matches_are_sparse(self):
        scores = [0.50] * 60 + [0.76] * 35 + [0.805] * 5
        self.assertEqual(adaptive_threshold(scores, 0.8), 0.75)

    def test_keeps_strict_threshold_when_match_population_is_healthy(self):
        scores = [0.40] * 50 + [0.76] * 20 + [0.81] * 30
        self.assertEqual(adaptive_threshold(scores, 0.8), 0.8)

    def test_short_non_court_gap_is_bridged(self):
        flags = [True] * 5 + [False] * 4 + [True] * 5
        self.assertEqual(stabilize_views(flags), [True] * len(flags))

    def test_five_non_court_frames_split_segments(self):
        flags = [True] * 5 + [False] * 5 + [True] * 5
        stable = stabilize_views(flags)
        self.assertEqual(segments_from_flags(stable, 5.0), [
            {
                "segment_id": "main_001",
                "start_frame": 0,
                "end_frame": 4,
                "frame_count": 5,
                "start_sec": 0.0,
                "end_sec": 1.0,
                "duration_sec": 1.0,
            },
            {
                "segment_id": "main_002",
                "start_frame": 10,
                "end_frame": 14,
                "frame_count": 5,
                "start_sec": 2.0,
                "end_sec": 3.0,
                "duration_sec": 1.0,
            },
        ])

    def test_short_court_run_is_removed(self):
        self.assertEqual(stabilize_views([False, True, True, True, True, False]), [False] * 6)


if __name__ == "__main__":
    unittest.main()
