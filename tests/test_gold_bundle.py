import json
import tempfile
import unittest
from pathlib import Path

from badminton_pipeline.evaluation.gold_bundle import extract_bundle


class GoldBundleTests(unittest.TestCase):
    def _bundle(self) -> dict:
        return {
            "schemaVersion": "shuttlelab-gold-bundle-1",
            "batchId": "batch-1",
            "videos": [{
                "videoId": "vid_abc123",
                "fileName": "clip.mp4",
                "frameCount": 100,
                "durationSec": 3.3,
                "rallyNumber": 1,
                "hits": [{
                    "videoId": "vid_abc123",
                    "rallyId": "Rally 01",
                    "hitFrame": 20,
                    "hitter": "upper",
                    "confidence": "high",
                    "uncertaintyFrames": 1,
                    "notes": "",
                }],
            }],
        }

    def test_extracts_valid_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "bundle.json"
            source.write_text(json.dumps(self._bundle()), encoding="utf-8")
            report = extract_bundle(source, root / "gold")
            self.assertEqual(report["total_hits"], 1)
            self.assertTrue(report["complete_for_locked_evaluation"])
            self.assertTrue((root / "gold" / "vid_abc123.hits_gold.csv").exists())


if __name__ == "__main__":
    unittest.main()
