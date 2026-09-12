import json
import tempfile
import unittest
from pathlib import Path

from badminton_pipeline.ingest.catalog import rebuild_video_catalog


class VideoCatalogTests(unittest.TestCase):
    def test_rebuilds_catalog_from_individual_records(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            records_dir = data_dir / "videos"
            records_dir.mkdir()
            for video_id, checksum in (("vid_b", "bbb"), ("vid_a", "aaa")):
                (records_dir / f"{video_id}.json").write_text(
                    json.dumps({"video_id": video_id, "checksum": checksum, "registered_at": video_id}),
                    encoding="utf-8",
                )
            result = rebuild_video_catalog(data_dir)
            catalog = json.loads((data_dir / "videos.json").read_text(encoding="utf-8"))
            self.assertEqual(result["video_count"], 2)
            self.assertEqual({row["video_id"] for row in catalog["videos"]}, {"vid_a", "vid_b"})

    def test_rejects_duplicate_checksums(self):
        with tempfile.TemporaryDirectory() as directory:
            data_dir = Path(directory)
            records_dir = data_dir / "videos"
            records_dir.mkdir()
            for video_id in ("vid_a", "vid_b"):
                (records_dir / f"{video_id}.json").write_text(
                    json.dumps({"video_id": video_id, "checksum": "same"}), encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                rebuild_video_catalog(data_dir)
