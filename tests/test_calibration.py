import unittest

from badminton_pipeline.calibration.homography import (
    COURT_POINTS,
    compute_homography,
    project_point,
    validate_corners,
)


class HomographyTests(unittest.TestCase):
    def setUp(self):
        self.corners = [(390.0, 210.0), (890.0, 210.0), (1120.0, 680.0), (160.0, 680.0)]

    def test_maps_all_four_corners_to_standard_court(self):
        matrix = compute_homography(self.corners)
        for source, expected in zip(self.corners, COURT_POINTS, strict=True):
            actual = project_point(matrix, source)
            self.assertAlmostEqual(actual[0], expected[0], places=8)
            self.assertAlmostEqual(actual[1], expected[1], places=8)

    def test_accepts_convex_court_area(self):
        valid, area_ratio, message = validate_corners(self.corners, 1280, 720)
        self.assertTrue(valid, message)
        self.assertGreater(area_ratio, 0.1)

    def test_rejects_crossed_corner_order(self):
        crossed = [self.corners[0], self.corners[2], self.corners[1], self.corners[3]]
        valid, _, _ = validate_corners(crossed, 1280, 720)
        self.assertFalse(valid)


if __name__ == "__main__":
    unittest.main()
