import unittest

from badminton_pipeline.tracking.postprocess import (
    _windowed_hitter_assignment,
    enrich_records,
    player_foot_point,
    score_hit_candidates,
)


def player(identity, x, y, wrist_x=None, wrist_y=None):
    points = [[x, y] for _ in range(17)]
    points[15] = [x - 1, y]
    points[16] = [x + 1, y]
    if wrist_x is not None:
        points[9] = [wrist_x, wrist_y]
        points[10] = [wrist_x, wrist_y]
    return {
        "identity": identity,
        "bbox_xyxy": [x - 5, y - 20, x + 5, y],
        "keypoints_xy": points,
        "keypoint_scores": [0.9] * 17,
        "keypoint_missing": [False] * 17,
    }


class TrackingTests(unittest.TestCase):
    def test_foot_point_averages_ankles(self):
        self.assertEqual(player_foot_point(player("upper", 10, 20)), (10.0, 20.0))

    def test_identity_homography_projection(self):
        records = [{"frame_idx": 0, "shuttle": {"raw_xy": None}, "players": [player("upper", 3, 4)]}]
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        enrich_records(records, identity, fps=30.0)
        self.assertEqual(records[0]["players"][0]["court_xy_m"], [3.0, 4.0])
        self.assertEqual(records[0]["players"][0]["identity"], "upper")
        self.assertTrue(records[0]["players"][0]["court_in_bounds"])

    def test_player_track_carries_position_across_short_detection_gap(self):
        records = [
            {"frame_idx": 10, "shuttle": {"raw_xy": None}, "players": [player("upper", 3, 4)]},
            {"frame_idx": 11, "shuttle": {"raw_xy": None}, "players": []},
            {"frame_idx": 17, "shuttle": {"raw_xy": None}, "players": []},
        ]
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        enrich_records(records, identity, fps=30.0)
        self.assertEqual(records[0]["player_tracks"]["upper"]["state"], "measured")
        self.assertEqual(records[1]["player_tracks"]["upper"]["state"], "predicted")
        self.assertEqual(records[1]["player_tracks"]["upper"]["court_xy_m"], [3.0, 4.0])
        self.assertEqual(records[2]["player_tracks"]["upper"]["state"], "missing")
        self.assertIsNone(records[2]["player_tracks"]["upper"]["court_xy_m"])

    def test_direction_change_near_wrist_proposes_event(self):
        points = [(40, 50), (50, 50), (40, 50), (30, 50), (20, 50)]
        records = []
        for index, point in enumerate(points):
            records.append({
                "frame_idx": index,
                "timestamp_sec": index / 30,
                "width": 100,
                "height": 100,
                "players": [player("upper", 50, 80, 50, 50)],
                "shuttle": {"smoothed_xy": list(point)},
            })
        events = score_hit_candidates(records, fps=30, threshold=0.3, min_separation_frames=1)
        self.assertEqual(events[0]["candidate_frame"], 1)
        self.assertEqual(events[0]["predicted_hitter"], "upper")
        self.assertEqual(events[0]["shuttle_image_xy"], [50.0, 50.0])
        self.assertIsNone(events[0]["shuttle_court_xy_m"])

    def test_event_includes_player_court_context(self):
        points = [(4, 4), (5, 4), (4, 4)]
        records = []
        for index, point in enumerate(points):
            records.append({
                "frame_idx": index,
                "timestamp_sec": index / 30,
                "width": 20,
                "height": 20,
                "players": [player("upper", 3, 4, 5, 4), player("lower", 3, 10, 5, 4)],
                "shuttle": {"raw_xy": list(point), "confidence": 0.9},
            })
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        enrich_records(records, identity, fps=30.0)
        events = score_hit_candidates(records, fps=30, threshold=0.0, min_separation_frames=1)
        self.assertEqual(events[0]["player_locations"]["upper"]["state"], "measured")
        self.assertEqual(events[0]["player_locations"]["lower"]["court_xy_m"], [3.0, 10.0])
        self.assertEqual(events[0]["shuttle_tracking_state"], "measured")

    def test_default_hit_separation_scales_with_fps(self):
        records = []
        points = [(40, 50), (50, 50), (40, 50)] * 6
        for index, point in enumerate(points):
            records.append({
                "frame_idx": index,
                "timestamp_sec": index / 30,
                "width": 100,
                "height": 100,
                "players": [player("upper", 50, 80, 50, 50)],
                "shuttle": {"smoothed_xy": list(point)},
            })
        events = score_hit_candidates(records, fps=30, threshold=0.3)
        self.assertTrue(all(
            following["candidate_frame"] - previous["candidate_frame"] >= 12
            for previous, following in zip(events, events[1:])
        ))

    def test_windowed_hitter_uses_nearby_hand_evidence_without_moving_event(self):
        scored = [
            {"frame_idx": 99, "predicted_hitter": "lower", "hand_distance_score": 0.4, "event_confidence": 0.5},
            {"frame_idx": 100, "predicted_hitter": "lower", "hand_distance_score": 0.5, "event_confidence": 0.8},
            {"frame_idx": 102, "predicted_hitter": "upper", "hand_distance_score": 0.9, "event_confidence": 0.6},
            {"frame_idx": 103, "predicted_hitter": "lower", "hand_distance_score": 1.0, "event_confidence": 0.9},
        ]
        hitter, evidence_frame, score = _windowed_hitter_assignment(scored, 1, 2)
        self.assertEqual(hitter, "upper")
        self.assertEqual(evidence_frame, 102)
        self.assertEqual(score, 0.9)

    def test_top_k_association_can_choose_continuous_lower_confidence_candidate(self):
        records = [
            {
                "frame_idx": 0, "width": 100, "height": 100, "players": [],
                "shuttle": {
                    "raw_xy": [10, 10], "confidence": 0.8,
                    "candidates": [{"raw_xy": [10, 10], "confidence": 0.8, "bbox_xyxy": [9, 9, 11, 11]}],
                },
            },
            {
                "frame_idx": 1, "width": 100, "height": 100, "players": [],
                "shuttle": {
                    "raw_xy": [90, 90], "confidence": 0.9,
                    "candidates": [
                        {"raw_xy": [90, 90], "confidence": 0.9, "bbox_xyxy": [89, 89, 91, 91]},
                        {"raw_xy": [12, 10], "confidence": 0.3, "bbox_xyxy": [11, 9, 13, 11]},
                    ],
                },
            },
            {
                "frame_idx": 2, "width": 100, "height": 100, "players": [],
                "shuttle": {"raw_xy": [14, 10], "confidence": 0.8},
            },
        ]
        enrich_records(records, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], fps=30)
        self.assertEqual(records[1]["shuttle"]["detector_candidate_count"], 2)
        self.assertEqual(records[1]["shuttle"]["selected_candidate_rank"], 1)
        self.assertEqual(records[1]["shuttle"]["associated_xy"], [12.0, 10.0])

    def test_large_unsupported_innovation_is_soft_suppressed_not_hard_deleted(self):
        records = [
            {"frame_idx": 0, "width": 100, "height": 100, "players": [],
             "shuttle": {"raw_xy": [10, 10], "confidence": 0.9}},
            {"frame_idx": 1, "width": 100, "height": 100, "players": [],
             "shuttle": {"raw_xy": [90, 90], "confidence": 0.9}},
        ]
        enrich_records(records, [[1, 0, 0], [0, 1, 0], [0, 0, 1]], fps=30)
        audit = records[1]["shuttle"]
        self.assertEqual(audit["measurement_status"], "soft_suppressed")
        self.assertEqual(audit["associated_xy"], [90.0, 90.0])
        self.assertLess(audit["smoothed_xy"][0], 25.0)
        self.assertGreater(audit["innovation_px"], 100.0)

    def test_wrist_evidence_softens_gate_for_plausible_hit_turn(self):
        def make_records(with_player):
            players = [player("candidate_0", 3, 4, 10, 1)] if with_player else []
            return [
                {"frame_idx": 0, "width": 100, "height": 100, "players": players,
                 "shuttle": {"raw_xy": [1, 1], "confidence": 0.9}},
                {"frame_idx": 1, "width": 100, "height": 100, "players": players,
                 "shuttle": {"raw_xy": [10, 1], "confidence": 0.9}},
                {"frame_idx": 2, "width": 100, "height": 100, "players": players,
                 "shuttle": {"raw_xy": [11, 1], "confidence": 0.9}},
            ]

        no_hit = make_records(False)
        supported_hit = make_records(True)
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        enrich_records(no_hit, identity, fps=30)
        enrich_records(supported_hit, identity, fps=30)
        self.assertGreater(
            supported_hit[1]["shuttle"]["measurement_weight"],
            no_hit[1]["shuttle"]["measurement_weight"],
        )
        self.assertGreater(supported_hit[1]["shuttle"]["hit_support_score"], 0.9)
        self.assertGreater(supported_hit[1]["shuttle"]["hand_support_score"], 0.9)


if __name__ == "__main__":
    unittest.main()
