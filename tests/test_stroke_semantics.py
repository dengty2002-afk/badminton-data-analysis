import csv
import json
import tempfile
import unittest
from pathlib import Path

from badminton_pipeline.evaluation.stroke_semantics import (
    create_semantic_template,
    evaluate_semantics,
    read_semantic_gold,
)


class StrokeSemanticTests(unittest.TestCase):
    @staticmethod
    def _gold(path: Path):
        path.write_text(
            "video_id,rally_id,hit_frame,hitter,stroke_type,landing_x,landing_y,landing_frame,landing_kind,confidence,uncertainty_frames,notes\n"
            "vid_test,Rally 01,10,upper,smash,2.0,11.0,20,next_contact,high,1,\n"
            "vid_test,Rally 01,30,lower,net shot,3.0,1.0,40,ground_contact,high,1,\n",
            encoding="utf-8",
        )

    def test_reads_18_class_semantic_gold_and_rejects_invalid_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.csv"
            self._gold(path)
            rows = read_semantic_gold(path)
            self.assertEqual([row.stroke_type for row in rows], ["smash", "net shot"])
            invalid = path.read_text(encoding="utf-8").replace("2.0,11.0", "7.0,11.0")
            path.write_text(invalid, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                read_semantic_gold(path)

    def test_creates_template_without_inventing_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hits = root / "hits.csv"
            hits.write_text(
                "video_id,rally_id,hit_frame,hitter,confidence,uncertainty_frames,notes\n"
                "vid_test,Rally 01,10,upper,high,1,\n",
                encoding="utf-8",
            )
            output = root / "semantic.csv"
            result = create_semantic_template(hits, output)
            with output.open("r", encoding="utf-8-sig", newline="") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(result["row_count"], 1)
            self.assertEqual(row["hit_frame"], "10")
            self.assertEqual(row["stroke_type"], "")
            self.assertEqual(row["landing_x"], "")

    def test_evaluates_type_landing_and_joint_accuracy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gold.csv"
            self._gold(path)
            gold = read_semantic_gold(path)
            predictions = [
                {
                    "video_id": "vid_test", "candidate_frame": 11,
                    "predicted_stroke_type": "smash",
                    "predicted_landing_x": 2.2, "predicted_landing_y": 11.3,
                },
                {
                    "video_id": "vid_test", "candidate_frame": 31,
                    "predicted_stroke_type": "drop",
                    "predicted_landing_x": 4.0, "predicted_landing_y": 1.0,
                },
            ]
            result = evaluate_semantics(predictions, gold, tolerance_frames=2)
            self.assertEqual(result["matched_count"], 2)
            self.assertEqual(result["type"]["accuracy"], 0.5)
            self.assertEqual(result["landing"]["within_0_5m"], 0.5)
            self.assertEqual(result["joint"]["type_and_landing_within_0_5m"], 0.5)


if __name__ == "__main__":
    unittest.main()
