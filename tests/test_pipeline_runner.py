import argparse
import json
import tempfile
import unittest
from pathlib import Path

from badminton_pipeline.run_pipeline import _load_reusable_inference, run_pipeline


class PipelineRunnerTests(unittest.TestCase):
    def test_reuses_only_complete_matching_raw_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"video")
            segments = root / "segments.json"
            segments.write_text("{}", encoding="utf-8")
            raw = root / "frames.raw.jsonl"
            raw.write_text("{}\n{}\n", encoding="utf-8")
            raw.with_suffix(".metadata.json").write_text(
                json.dumps({
                    "input": str(video.resolve()),
                    "segments_manifest": str(segments.resolve()),
                    "processed_frames": 2,
                    "fps": 25.0,
                }),
                encoding="utf-8",
            )
            result = _load_reusable_inference(raw, video, segments)
            self.assertTrue(result["reused_existing_raw"])
            raw.write_text("{}\n", encoding="utf-8")
            self.assertIsNone(_load_reusable_inference(raw, video, segments))

    def test_rejects_calibration_for_different_video_before_model_load(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video.mp4"
            video.write_bytes(b"not a real video")
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps({"video_id": "vid_different", "quality_status": "accepted"}),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                input=video,
                calibration=calibration,
                output_dir=root / "run",
                models_dir=root / "models",
                manifest=root / "models.json",
                device="cpu",
                start_frame=0,
                max_frames=1,
                stride=1,
                pose_confidence=0.2,
                shuttle_confidence=0.18,
                shuttle_top_k=5,
                event_threshold=0.48,
                min_hit_separation_sec=0.4,
                hitter_window_sec=2 / 30,
                play_state_window_sec=1.0,
                play_state_support_sec=3.0,
                skip_shuttle=True,
            )
            with self.assertRaisesRegex(ValueError, "does not match"):
                run_pipeline(args)
            self.assertFalse(args.output_dir.exists())


if __name__ == "__main__":
    unittest.main()
