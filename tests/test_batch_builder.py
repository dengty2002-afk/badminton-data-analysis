import argparse
import json
import tempfile
import unittest
from pathlib import Path

from badminton_pipeline.batch_builder import (
    BatchBuilder,
    _batch_lock,
    _court_sample_times,
    _mark_derived_source_variants,
    discover_videos,
)
from badminton_pipeline.ingest.register_video import sha256_file


class BatchBuilderTests(unittest.TestCase):
    def test_marks_legacy_main_view_duplicate_but_keeps_inventory(self):
        state = {
            "videos": [
                {
                    "source_path": r"D:\input\match.mp4",
                    "file_name": "match.mp4",
                    "status": "pending",
                    "quality_warnings": [],
                },
                {
                    "source_path": r"D:\input\main_view_outputs\match_主视角.mp4",
                    "file_name": "match_主视角.mp4",
                    "status": "pending",
                    "quality_warnings": [],
                },
            ]
        }
        self.assertEqual(_mark_derived_source_variants(state), 1)
        self.assertEqual(len(state["videos"]), 2)
        derived = state["videos"][1]
        self.assertEqual(derived["status"], "excluded_source_variant")
        self.assertEqual(derived["canonical_file_name"], "match.mp4")

    def test_batch_lock_rejects_second_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "batch.lock"
            with _batch_lock(path):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with _batch_lock(path):
                        pass

    def test_court_samples_prefer_long_main_view_segments(self):
        result = {
            "fps": 25.0,
            "segments": [
                {"start_frame": 100, "end_frame": 199, "frame_count": 100},
                {"start_frame": 1000, "end_frame": 1499, "frame_count": 500},
            ],
        }
        self.assertEqual(_court_sample_times(result, 10000), [49980, 5980, 10000])

    def test_discovers_supported_videos_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            (root / "b.mov").write_bytes(b"b")
            (root / "a.mp4").write_bytes(b"a")
            (nested / "c.mkv").write_bytes(b"c")
            (root / "ignore.txt").write_text("x", encoding="utf-8")

            shallow = discover_videos([root, root / "a.mp4"])
            self.assertEqual([path.name for path in shallow], ["a.mp4", "b.mov"])
            recursive = discover_videos([root], recursive=True)
            self.assertEqual([path.name for path in recursive], ["a.mp4", "b.mov", "c.mkv"])

    def test_waits_for_court_review_then_resumes_without_repeating_completed_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            work = root / "work"
            output = root / "dataset"
            video = root / "match.mp4"
            video.write_bytes(b"test-video")
            video_id = "vid_" + sha256_file(video)[:12]
            calls = {"register": 0, "court": 0, "infer": 0, "validate": 0, "merge": 0}

            def register_fn(path, **_kwargs):
                calls["register"] += 1
                return {
                    "status": "registered",
                    "video_id": video_id,
                    "checksum": sha256_file(path),
                    "record": str(data / "videos" / f"{video_id}.json"),
                }

            def court_fn(path, supplied_video_id, proposal_dir, **_kwargs):
                calls["court"] += 1
                self.assertEqual(supplied_video_id, video_id)
                proposal_dir.mkdir(parents=True, exist_ok=True)
                proposal = proposal_dir / f"{video_id}.court_proposal.json"
                value = {
                    "video_id": video_id,
                    "input": str(path),
                    "accepted": False,
                    "calibration": None,
                    "proposal": str(proposal),
                }
                proposal.write_text(json.dumps(value), encoding="utf-8")
                return value

            def pipeline_fn(args: argparse.Namespace):
                calls["infer"] += 1
                args.output_dir.mkdir(parents=True, exist_ok=True)
                (args.output_dir / "run.json").write_text("{}", encoding="utf-8")
                dataset = args.output_dir / "dataset"
                dataset.mkdir(exist_ok=True)
                (dataset / "manifest.json").write_text("{}", encoding="utf-8")
                return {
                    "status": "completed", "video_id": video_id,
                    "postprocess": {"calibration": str(args.calibration)},
                }

            def validate_fn(dataset_dir):
                calls["validate"] += 1
                return {
                    "status": "validated",
                    "video_id": video_id,
                    "quality_status": "warning",
                    "warnings": ["shuttle track coverage is below 80%"],
                    "rows": {"frames": 10, "dataset_rows": 1},
                    "manifest": str(dataset_dir / "manifest.json"),
                    "manifest_sha256": "test",
                }

            def merge_fn(dataset_dirs, output_dir):
                calls["merge"] += 1
                self.assertEqual(len(list(dataset_dirs)), 1)
                output_dir.mkdir(parents=True, exist_ok=True)
                manifest = output_dir / "manifest.json"
                manifest.write_text("{}", encoding="utf-8")
                return {"status": "completed", "manifest": str(manifest)}

            def builder():
                return BatchBuilder(
                    batch_id="test-batch",
                    inputs=[video],
                    data_dir=data,
                    work_dir=work,
                    output_dir=output,
                    device="cpu",
                    register_fn=register_fn,
                    court_fn=court_fn,
                    pipeline_fn=pipeline_fn,
                    validate_fn=validate_fn,
                    merge_fn=merge_fn,
                )

            first = builder().run()
            self.assertEqual(first["status"], "waiting_court_review")
            self.assertTrue(first["storage"]["source_videos_are_not_copied"])
            self.assertEqual(first["storage"]["source_video_bytes"], video.stat().st_size)
            self.assertEqual(calls, {"register": 1, "court": 1, "infer": 0, "validate": 0, "merge": 0})

            calibration = data / "calibrations" / video_id / "v1.json"
            calibration.parent.mkdir(parents=True)
            calibration.write_text(
                json.dumps({"video_id": video_id, "version": 1, "quality_status": "accepted"}),
                encoding="utf-8",
            )
            proposal = data / "court_proposals" / f"{video_id}.court_proposal.json"
            proposal.write_text(
                json.dumps({
                    "video_id": video_id,
                    "accepted": True,
                    "calibration": str(calibration),
                }),
                encoding="utf-8",
            )

            second = builder().run()
            self.assertEqual(second["status"], "completed_with_warning")
            self.assertEqual(second["counts"]["warnings"], 1)
            self.assertEqual(calls, {"register": 1, "court": 1, "infer": 1, "validate": 1, "merge": 1})

            third = builder().run()
            self.assertEqual(third["status"], "completed_with_warning")
            self.assertEqual(calls, {"register": 1, "court": 1, "infer": 1, "validate": 1, "merge": 1})

            calibration_v2 = data / "calibrations" / video_id / "v2.json"
            calibration_v2.write_text(
                json.dumps({"video_id": video_id, "version": 2, "quality_status": "accepted"}),
                encoding="utf-8",
            )
            proposal.write_text(
                json.dumps({
                    "video_id": video_id, "accepted": True,
                    "calibration": str(calibration_v2),
                }),
                encoding="utf-8",
            )
            fourth = builder().run()
            self.assertEqual(fourth["status"], "completed_with_warning")
            self.assertEqual(calls, {"register": 1, "court": 1, "infer": 2, "validate": 2, "merge": 2})

    def test_failed_inference_is_retried_without_repeating_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            work = root / "work"
            output = root / "dataset"
            video = root / "match.mp4"
            video.write_bytes(b"retry-video")
            video_id = "vid_" + sha256_file(video)[:12]
            calibration = data / "calibrations" / video_id / "v1.json"
            calibration.parent.mkdir(parents=True)
            calibration.write_text(
                json.dumps({"video_id": video_id, "version": 1, "quality_status": "accepted"}),
                encoding="utf-8",
            )
            calls = {"register": 0, "infer": 0}

            def register_fn(path, **_kwargs):
                calls["register"] += 1
                return {"video_id": video_id, "checksum": sha256_file(path)}

            def pipeline_fn(args):
                calls["infer"] += 1
                if calls["infer"] == 1:
                    raise RuntimeError("synthetic GPU failure")
                args.output_dir.mkdir(parents=True, exist_ok=True)
                (args.output_dir / "run.json").write_text("{}", encoding="utf-8")
                (args.output_dir / "dataset").mkdir()
                (args.output_dir / "dataset" / "manifest.json").write_text("{}", encoding="utf-8")
                return {"status": "completed", "postprocess": {"calibration": str(args.calibration)}}

            def validate_fn(dataset_dir):
                return {
                    "quality_status": "ok", "warnings": [], "video_id": video_id,
                    "status": "validated", "rows": {},
                    "manifest": str(dataset_dir / "manifest.json"), "manifest_sha256": "test",
                }

            def merge_fn(_datasets, output_dir):
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
                return {"manifest": str(output_dir / "manifest.json")}

            def run_once():
                return BatchBuilder(
                    batch_id="retry", inputs=[video], data_dir=data, work_dir=work,
                    output_dir=output, device="cpu", register_fn=register_fn,
                    pipeline_fn=pipeline_fn, validate_fn=validate_fn, merge_fn=merge_fn,
                ).run()

            self.assertEqual(run_once()["status"], "partial_failure")
            self.assertEqual(run_once()["status"], "completed")
            self.assertEqual(calls, {"register": 1, "infer": 2})

    def test_resume_rejects_changed_pipeline_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "match.mp4"
            video.write_bytes(b"config-video")
            common = {
                "batch_id": "config", "inputs": [video], "data_dir": root / "data",
                "work_dir": root / "work", "output_dir": root / "output", "device": "cpu",
                "register_fn": lambda path, **_kwargs: {
                    "video_id": "vid_" + sha256_file(path)[:12], "checksum": sha256_file(path)
                },
                "court_fn": lambda *_args, **_kwargs: {"accepted": False},
            }
            BatchBuilder(**common, pipeline_options={"max_frames": 1}).run()
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                BatchBuilder(**common, pipeline_options={"max_frames": 2}).run()


if __name__ == "__main__":
    unittest.main()
