import json
import tempfile
import unittest
from pathlib import Path

from badminton_pipeline.calibration.auto_court import (
    _boundary_semantics_audit,
    _expand_singles_quad_to_doubles,
    propose_court,
)


class AutoCourtTests(unittest.TestCase):
    @staticmethod
    def _image(path: Path) -> None:
        import cv2
        import numpy as np
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.imwrite(str(path), image)

    def test_manual_correction_stays_proposal_without_accept(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            self._image(image)
            result = propose_court(
                image, "vid_test", root / "proposals",
                manual_corners=[(390, 210), (890, 210), (1120, 680), (160, 680)],
                data_dir=root / "data",
            )
            self.assertFalse(result["accepted"])
            self.assertIsNone(result["calibration"])
            self.assertTrue(Path(result["preview"]).is_file())
            self.assertFalse((root / "data" / "calibrations").exists())

    def test_explicit_accept_saves_versioned_homography(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "frame.png"
            self._image(image)
            result = propose_court(
                image, "vid_test", root / "proposals", accept=True,
                manual_corners=[(390, 210), (890, 210), (1120, 680), (160, 680)],
                data_dir=root / "data",
            )
            self.assertTrue(result["accepted"])
            calibration = json.loads(Path(result["calibration"]).read_text(encoding="utf-8"))
            self.assertEqual(calibration["version"], 1)
            self.assertEqual(calibration["source"], "manual_correction")
            self.assertTrue(calibration["quality"]["human_confirmed"])
            self.assertEqual(len(calibration["homography"]), 3)

    def test_expands_singles_sidelines_to_doubles_boundary(self):
        expanded = _expand_singles_quad_to_doubles(
            [(460, 250), (820, 250), (940, 650), (340, 650)]
        )
        self.assertLess(expanded[0][0], 460)
        self.assertGreater(expanded[1][0], 820)
        self.assertGreater(expanded[2][0], 940)
        self.assertLess(expanded[3][0], 340)

    def test_flags_synthetic_singles_boundary_interpretation(self):
        import cv2
        import numpy as np

        vendor = Path(__file__).resolve().parents[1] / "vendor" / "Good-Badminton"
        import sys
        sys.path.insert(0, str(vendor))
        from badminton_analysis.court.reference import BadmintonCourtReference

        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        outer = [(410, 250), (870, 250), (1040, 660), (240, 660)]
        cv2.fillConvexPoly(image, np.asarray(outer, dtype=np.int32), (55, 145, 55))
        reference = BadmintonCourtReference()
        for start, end, _weight in reference.project_lines(outer):
            cv2.line(
                image, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)),
                (255, 255, 255), 4, cv2.LINE_AA,
            )
        matrix = cv2.getPerspectiveTransform(
            np.array([[0, 0], [6.1, 0], [6.1, 13.4], [0, 13.4]], dtype=np.float32),
            np.asarray(outer, dtype=np.float32),
        )
        singles_world = np.array(
            [[0.46, 0], [5.64, 0], [5.64, 13.4], [0.46, 13.4]], dtype=np.float32,
        )
        singles = cv2.perspectiveTransform(singles_world.reshape(-1, 1, 2), matrix).reshape(-1, 2)
        singles_audit = _boundary_semantics_audit(
            image, [(float(x), float(y)) for x, y in singles], vendor,
        )
        outer_audit = _boundary_semantics_audit(image, outer, vendor)
        self.assertTrue(singles_audit["likely_singles_sidelines"])
        self.assertFalse(outer_audit["likely_singles_sidelines"])


if __name__ == "__main__":
    unittest.main()
