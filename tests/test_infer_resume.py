import json
import tempfile
import unittest
from pathlib import Path

from badminton_pipeline.models.infer_video import (
    _remaining_segment_ranges,
    _resume_prefix,
    _selected_frame_count,
)


class InferResumeTests(unittest.TestCase):
    def test_counts_strided_segment_frames(self):
        self.assertEqual(_selected_frame_count([(1, 5), (10, 13)], 2, 0), 4)
        self.assertEqual(_selected_frame_count([(1, 5), (10, 13)], 2, 3), 3)
        self.assertIsNone(_selected_frame_count([], 1, 0))

    def test_seeks_to_first_unfinished_selected_frame(self):
        segments = [(1, 5), (10, 16)]  # selected at 2,4 and 10,12,14,16
        self.assertEqual(_remaining_segment_ranges(segments, 2, 0), [(2, 5), (10, 16)])
        self.assertEqual(_remaining_segment_ranges(segments, 2, 1), [(4, 5), (10, 16)])
        self.assertEqual(_remaining_segment_ranges(segments, 2, 2), [(10, 16)])
        self.assertEqual(_remaining_segment_ranges(segments, 2, 4), [(14, 16)])
        self.assertEqual(_remaining_segment_ranges(segments, 2, 6), [])

    def test_accepts_exact_prefix_and_truncates_partial_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frames.raw.jsonl"
            rows = [
                {"video_id": "vid_test", "frame_idx": 2},
                {"video_id": "vid_test", "frame_idx": 4},
            ]
            output.write_bytes(
                b"".join((json.dumps(row) + "\n").encode() for row in rows)
                + b'{"video_id":"vid_test"'
            )
            count = _resume_prefix(
                output, video_id="vid_test", segments=[(1, 8)], stride=2, max_frames=0,
            )
            self.assertEqual(count, 2)
            self.assertEqual(len(output.read_text().splitlines()), 2)

    def test_rejects_non_prefix_without_mutating_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frames.raw.jsonl"
            output.write_text(
                json.dumps({"video_id": "vid_test", "frame_idx": 4}) + "\n",
                encoding="utf-8",
            )
            before = output.read_bytes()
            count = _resume_prefix(
                output, video_id="vid_test", segments=[(1, 8)], stride=2, max_frames=0,
            )
            self.assertEqual(count, 0)
            self.assertEqual(output.read_bytes(), before)

    def test_discards_valid_json_without_final_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "frames.raw.jsonl"
            first = json.dumps({"video_id": "vid_test", "frame_idx": 2}) + "\n"
            second = json.dumps({"video_id": "vid_test", "frame_idx": 4})
            output.write_text(first + second, encoding="utf-8")
            count = _resume_prefix(
                output, video_id="vid_test", segments=[(1, 8)], stride=2, max_frames=0,
            )
            self.assertEqual(count, 1)
            self.assertEqual(output.read_text(encoding="utf-8"), first)


if __name__ == "__main__":
    unittest.main()
