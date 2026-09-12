import csv
import json
import tempfile
import unittest
from pathlib import Path

from badminton_pipeline.evaluation.hit_events import GoldHit, evaluate_candidates, evaluate_files, read_gold


def gold(frame, hitter="upper", uncertainty=0):
    return GoldHit("vid_test", frame, hitter, "Rally 01", "high", uncertainty, "")


def candidate(frame, hitter="upper"):
    return {
        "event_id": f"evt_{frame}", "video_id": "vid_test", "candidate_frame": frame,
        "candidate_time": frame / 30, "predicted_hitter": hitter, "rally_id": "Rally 01",
        "event_confidence": 0.7, "trajectory_score": 0.6,
        "hand_distance_score": 0.8, "pose_score": 0.5,
    }


class HitEventEvaluationTests(unittest.TestCase):
    def test_one_to_one_matching_prefers_closest_candidate(self):
        result = evaluate_candidates(
            [candidate(98), candidate(101), candidate(150), candidate(201)],
            [gold(100), gold(130), gold(200, "lower")], tolerance=3,
        )
        self.assertEqual(result["matched_count"], 2)
        self.assertEqual(result["false_positive_count"], 2)
        self.assertEqual(result["false_negative_count"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.666667)
        self.assertEqual(result["matches"][0]["candidate_frame"], 101)
        self.assertEqual(result["median_absolute_frame_error"], 1.0)
        self.assertEqual(result["hitter_accuracy"], 0.5)

    def test_gold_uncertainty_expands_matching_window(self):
        result = evaluate_candidates([candidate(106)], [gold(100, uncertainty=2)], tolerance=4)
        self.assertEqual(result["matched_count"], 1)

    def test_file_evaluation_writes_reports_for_multiple_tolerances(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = root / "events.jsonl"
            gold_path = root / "gold.csv"
            candidates.write_text(json.dumps(candidate(101)) + "\n", encoding="utf-8")
            with gold_path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["video_id", "rally_id", "hit_frame", "hitter", "confidence", "uncertainty_frames", "notes"])
                writer.writerow(["vid_test", "Rally 01", 100, "upper", "high", 0, ""])
            result = evaluate_files(candidates, gold_path, root / "report", tolerances=(0, 3), primary_tolerance=3)
            self.assertEqual(result["raw_metrics_by_tolerance"]["0"]["recall"], 0.0)
            self.assertEqual(result["robust_presence_metrics_by_tolerance"]["3"]["recall"], 1.0)
            for name in ("hit_evaluation.json", "matched_events.csv", "false_positives.csv", "false_negatives.csv"):
                self.assertTrue((root / "report" / name).is_file())

    def test_reaction_allowance_is_reported_without_shifting_gold(self):
        strict = evaluate_candidates([candidate(108)], [gold(100)], tolerance=3)
        robust = evaluate_candidates(
            [candidate(108)], [gold(100)], tolerance=3, reaction_allowance_frames=6,
        )
        self.assertEqual(strict["recall"], 0.0)
        self.assertEqual(robust["recall"], 1.0)
        self.assertEqual(robust["matches"][0]["frame_error"], 8)
        self.assertEqual(robust["median_signed_frame_error"], 8)

    def test_reports_ambiguous_gold_when_multiple_candidates_are_nearby(self):
        result = evaluate_candidates([candidate(98), candidate(102)], [gold(100)], tolerance=3)
        self.assertEqual(result["matched_count"], 1)
        self.assertEqual(result["ambiguous_gold_count"], 1)

    def test_rejects_invalid_hitter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.csv"
            path.write_text("video_id,hit_frame,hitter\nvid_test,1,A\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid hitter"):
                read_gold(path)


if __name__ == "__main__":
    unittest.main()
