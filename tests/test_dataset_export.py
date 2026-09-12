import json
import tempfile
import unittest
from pathlib import Path

from badminton_pipeline.exports.dataset import export_dataset


class DatasetExportTests(unittest.TestCase):
    def test_writes_shuttleset_like_parquet_csv_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames.jsonl"
            events = root / "events.jsonl"
            calibration = root / "calibration.json"
            video = root / "video.json"
            output = root / "dataset"
            frame = {
                "video_id": "vid_test12345678", "frame_idx": 10, "timestamp_sec": 1.0,
                "width": 1280, "height": 720,
                "players": [
                    {"identity": "upper", "bbox_xyxy": [1, 2, 3, 4],
                     "keypoints_xy": [[1, 2]] * 17, "keypoint_scores": [0.9] * 17,
                     "keypoint_missing": [False] * 17, "court_in_bounds": False},
                    {"identity": "lower", "bbox_xyxy": [5, 6, 7, 8],
                     "keypoints_xy": [[5, 6]] * 17, "keypoint_scores": [0.8] * 17,
                     "keypoint_missing": [False] * 17, "court_in_bounds": True},
                ],
                "player_tracks": {
                    "upper": {"court_xy_m": [2.0, 3.0], "state": "measured"},
                    "lower": {"court_xy_m": [4.0, 10.0], "state": "measured"},
                },
                "shuttle": {"raw_xy": [100, 200], "smoothed_xy": [101, 201],
                            "confidence": 0.9, "tracking_state": "measured"},
                "models": {"pose": {"family": "RTMPose", "weights_sha256": "pose"},
                           "shuttle": {"family": "YOLO", "weights_sha256": "ball"}},
            }
            event = {
                "event_id": "evt_10", "video_id": "vid_test12345678", "rally_id": "Rally 01",
                "candidate_frame": 10, "candidate_time": 1.0, "window_start_frame": 0,
                "window_end_frame": 20, "predicted_hitter": "upper",
                "player_locations": frame["player_tracks"], "shuttle_image_xy": [101, 201],
                "shuttle_tracking_state": "measured", "trajectory_score": 0.7,
                "hand_distance_score": 0.8, "pose_score": 0.6, "event_confidence": 0.72,
                "label_source": "machine_candidate", "review_status": "unreviewed",
                "model_version": "hit-rules-test", "calibration_version": 1,
                "play_state_version": "play-state-evidence-test", "play_state_score": 0.8,
                "play_state_filter_applied": False, "play_state_decision": "score_only_unreviewed",
                "play_state_window_seconds": 1.0, "play_state_window_frames": 31,
                "play_state_window_radius_frames": 15, "play_state_support_seconds": 3.0,
                "play_state_support_radius_frames": 90,
                "play_state_shuttle_measured_ratio": 0.9, "play_state_shuttle_tracked_ratio": 1.0,
                "play_state_shuttle_moving_ratio": 0.8, "play_state_both_player_tracking_ratio": 1.0,
                "play_state_nearby_candidate_count": 3, "play_state_sequence_support": 1.0,
                "predicted_stroke_type": "smash", "stroke_type_confidence": 0.81,
                "stroke_type_status": "predicted", "stroke_type_input_quality": 0.92,
                "stroke_type_source": "bst_shuttleset_35_side_collapsed",
                "stroke_type_raw_class": "top_smash",
                "stroke_type_top3": [{"raw_class": "top_smash", "probability": 0.81}],
                "predicted_landing_x": 2.5, "predicted_landing_y": 10.2,
                "landing_frame": 24, "landing_kind": "next_contact",
                "landing_confidence": 0.73, "landing_source": "automatic_destination_model",
                "landing_status": "predicted_monocular_proxy",
                "landing_proxy_kind": "shuttle_receiver_fusion",
                "semantic_label_source": "model_prediction", "semantic_review_status": "unreviewed",
            }
            frames.write_text(json.dumps(frame) + "\n", encoding="utf-8")
            events.write_text(json.dumps(event) + "\n", encoding="utf-8")
            calibration.write_text(json.dumps({
                "calibration_id": "cal_test", "video_id": "vid_test12345678", "version": 1,
                "corners": [{"x": 0, "y": 0}, {"x": 1, "y": 0}, {"x": 1, "y": 1}, {"x": 0, "y": 1}],
                "homography": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "quality_status": "accepted", "source": "manual",
            }), encoding="utf-8")
            video.write_text(json.dumps({
                "video_id": "vid_test12345678", "match_id": "match-1", "players": {},
                "domain": "fixed-camera-singles", "redistribution": "private",
            }), encoding="utf-8")

            result = export_dataset(frames, events, calibration, output, video_record_path=video)

            self.assertEqual(result["rows"], {"videos": 1, "calibrations": 1, "frames": 1, "dataset_rows": 1})
            for name in ("videos.parquet", "calibrations.parquet", "frames.parquet",
                         "dataset_rows.parquet", "dataset_rows.csv", "dataset_rows.jsonl",
                         "data_dictionary.json", "quality.json", "manifest.json"):
                self.assertTrue((output / name).is_file(), name)
            csv_text = (output / "dataset_rows.csv").read_text(encoding="utf-8")
            self.assertIn("player_location_x", csv_text)
            self.assertIn("upper", csv_text)
            self.assertIn("unknown", csv_text)
            self.assertIn("play_state_score", csv_text)
            self.assertIn("score_only_unreviewed", csv_text)
            self.assertIn("play_state_window_frames", csv_text)
            self.assertIn("play_state_sequence_support", csv_text)
            self.assertIn("play_state_gate_decision", csv_text)
            self.assertIn("smash", csv_text)
            self.assertIn("top_smash", csv_text)
            self.assertIn("stroke_type_input_quality", csv_text)
            self.assertIn("shuttle_receiver_fusion", csv_text)
            self.assertIn("automatic_destination_model", csv_text)
            frame_columns = (output / "frames.parquet").read_bytes()
            self.assertGreater(len(frame_columns), 0)
            quality = json.loads((output / "quality.json").read_text(encoding="utf-8"))
            self.assertEqual(quality["status"], "warning")
            self.assertIn("automatic prelabels", " ".join(quality["warnings"]))
            self.assertIn("upper player court in-bounds", " ".join(quality["warnings"]))
            self.assertEqual(quality["shuttle"]["track_coverage"], 1.0)
            self.assertFalse(quality["shuttle"]["interpolation_applied"])
            self.assertEqual(quality["play_state_gate"]["active_play_frames"], 0)

    def test_supports_zero_event_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frames = root / "frames.jsonl"
            events = root / "events.jsonl"
            calibration = root / "calibration.json"
            frame = {
                "video_id": "vid_empty000000", "frame_idx": 0, "timestamp_sec": 0.0,
                "width": 10, "height": 10, "players": [], "player_tracks": {},
                "shuttle": {}, "models": {},
            }
            frames.write_text(json.dumps(frame) + "\n", encoding="utf-8")
            events.write_text("", encoding="utf-8")
            calibration.write_text(json.dumps({"video_id": "vid_empty000000", "version": 1,
                "corners": [], "homography": [], "quality_status": "accepted"}), encoding="utf-8")
            result = export_dataset(frames, events, calibration, root / "out")
            self.assertEqual(result["rows"]["dataset_rows"], 0)


if __name__ == "__main__":
    unittest.main()
