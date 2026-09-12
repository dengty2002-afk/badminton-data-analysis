import json
import tempfile
import unittest
from pathlib import Path

from badminton_pipeline.exports.batch_dataset import build_batch_dataset, discover_dataset_dirs
from badminton_pipeline.exports.dataset import export_dataset


class BatchDatasetTests(unittest.TestCase):
    @staticmethod
    def _export(root: Path, video_id: str, frame_idx: int, event_ids=()) -> Path:
        source = root / video_id
        source.mkdir(parents=True)
        frames = source / "frames.jsonl"
        events = source / "events.jsonl"
        calibration = source / "calibration.json"
        frames.write_text(json.dumps({
            "video_id": video_id,
            "frame_idx": frame_idx,
            "timestamp_sec": frame_idx / 30,
            "width": 1280,
            "height": 720,
            "players": [],
            "player_tracks": {},
            "shuttle": {},
            "models": {},
        }) + "\n", encoding="utf-8")
        event_rows = []
        for event_id in event_ids:
            event_rows.append({
                "event_id": event_id,
                "video_id": video_id,
                "rally_id": "Rally 01",
                "candidate_frame": frame_idx,
                "candidate_time": frame_idx / 30,
                "window_start_frame": frame_idx,
                "window_end_frame": frame_idx,
                "predicted_hitter": "upper",
                "player_locations": {},
                "shuttle_image_xy": None,
                "trajectory_score": 0.5,
                "hand_distance_score": 0.5,
                "pose_score": 0.5,
                "event_confidence": 0.5,
            })
        events.write_text(
            "".join(json.dumps(row) + "\n" for row in event_rows), encoding="utf-8"
        )
        calibration.write_text(json.dumps({
            "calibration_id": f"cal_{video_id}",
            "video_id": video_id,
            "version": 1,
            "corners": [],
            "homography": [],
            "quality_status": "accepted",
            "source": "test",
        }), encoding="utf-8")
        output = source / "dataset"
        export_dataset(frames, events, calibration, output)
        return output

    def test_builds_deterministic_multi_video_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            second = self._export(root, "vid_b", 20, ["evt_shared"])
            first = self._export(root, "vid_a", 10, ["evt_shared"])
            found = discover_dataset_dirs([root])
            self.assertEqual(found, [first.resolve(), second.resolve()])

            result = build_batch_dataset(found, root / "batch")
            self.assertEqual(result["rows"]["videos"], 2)
            self.assertEqual(result["rows"]["frames"], 2)
            self.assertEqual(result["rows"]["dataset_rows"], 2)
            self.assertEqual(result["video_ids"], ["vid_a", "vid_b"])
            self.assertGreater(result["total_size_bytes"], 0)
            import polars as pl
            frames = pl.read_parquet(root / "batch" / "frames.parquet")
            self.assertEqual(frames.get_column("video_id").to_list(), ["vid_a", "vid_b"])

    def test_rejects_tampered_input_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = self._export(root, "vid_a", 10)
            with (dataset / "quality.json").open("a", encoding="utf-8") as stream:
                stream.write(" ")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                build_batch_dataset([dataset], root / "batch")

    def test_rejects_duplicate_video_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._export(root / "one", "vid_same", 10)
            second = self._export(root / "two", "vid_same", 20)
            with self.assertRaisesRegex(ValueError, "duplicate video_id"):
                build_batch_dataset([first, second], root / "batch")


if __name__ == "__main__":
    unittest.main()
