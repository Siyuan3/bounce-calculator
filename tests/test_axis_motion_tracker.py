import unittest

import numpy as np

from axis_motion_tracker import (
    detect_axis_motion,
    detect_ball_positions_dual,
    locate_ball_template_box,
)
from restitution_math import axis_coordinate


class AxisMotionTrackerTests(unittest.TestCase):
    def test_ball_silhouette_rejects_connected_narrow_track_reflection(self) -> None:
        height, width = 150, 120
        background = np.full((height, width), 185, dtype=np.uint8)
        yy, xx = np.mgrid[:height, :width]
        positions = [(60, 55), (60, 78), (60, 101)]
        frames = []
        for x, y in positions:
            image = background.copy()
            image[(xx - x) ** 2 + (yy - y) ** 2 <= 10**2] = 30
            # A shiny rail creates a narrow difference streak connected to the ball.
            image[y - 42 : y - 8, x - 2 : x + 3] = 55
            frames.append(image)
        template = np.zeros((20, 20), dtype=np.uint8)
        template[(xx[:20, :20] - 10) ** 2 + (yy[:20, :20] - 10) ** 2 <= 10**2] = 155

        measurements = detect_ball_positions_dual(
            np.stack(frames),
            [0.0, 0.01, 0.02],
            background,
            roi=(20, 5, 80, 135),
            template_frame_index=0,
            ball_box=(50, 45, 20, 20),
            wall_line_points=((10, 5), (110, 5)),
            template_difference_frame=template,
        )

        for measurement, (expected_x, expected_y) in zip(measurements, positions):
            self.assertAlmostEqual(measurement.x, expected_x, delta=2.5)
            self.assertAlmostEqual(measurement.y, expected_y, delta=2.5)

    def test_ball_silhouette_rejects_connected_broad_track_reflection(self) -> None:
        height, width = 150, 120
        background = np.full((height, width), 185, dtype=np.uint8)
        yy, xx = np.mgrid[:height, :width]
        positions = [(60, 55), (60, 78), (60, 101)]
        frames = []
        for x, y in positions:
            image = background.copy()
            image[(xx - x) ** 2 + (yy - y) ** 2 <= 10**2] = 30
            # This reflection is wide enough to survive a density-only filter.
            image[y - 42 : y - 8, x - 4 : x + 5] = 55
            frames.append(image)
        template = np.zeros((20, 20), dtype=np.uint8)
        template[(xx[:20, :20] - 10) ** 2 + (yy[:20, :20] - 10) ** 2 <= 10**2] = 155

        measurements = detect_ball_positions_dual(
            np.stack(frames),
            [0.0, 0.01, 0.02],
            background,
            roi=(20, 5, 80, 135),
            template_frame_index=0,
            ball_box=(50, 45, 20, 20),
            wall_line_points=((10, 5), (110, 5)),
            template_difference_frame=template,
        )

        for measurement, (expected_x, expected_y) in zip(measurements, positions):
            self.assertAlmostEqual(measurement.x, expected_x, delta=2.5)
            self.assertAlmostEqual(measurement.y, expected_y, delta=2.5)

    def test_dual_detector_uses_empty_background_and_agrees_on_ball_position(self) -> None:
        height, width = 90, 130
        background = np.full((height, width), 185, dtype=np.uint8)
        positions = [(28, 45), (44, 45), (61, 45), (79, 45), (96, 45)]
        yy, xx = np.mgrid[:height, :width]
        frames = []
        for x, y in positions:
            image = background.copy()
            image[(xx - x) ** 2 + (yy - y) ** 2 <= 7**2] = 35
            # A weak cast shadow must not pull the full-silhouette midpoint.
            image[(xx - (x + 7)) ** 2 + (yy - (y + 5)) ** 2 <= 6**2] = 165
            frames.append(image)

        measurements = detect_ball_positions_dual(
            np.stack(frames),
            [0.000, 0.004, 0.0085, 0.0125, 0.017],
            background,
            roi=(10, 25, 110, 42),
            template_frame_index=0,
            ball_box=(21, 38, 14, 14),
            wall_line_points=((115, 10), (115, 80)),
        )

        for measurement, (expected_x, expected_y) in zip(measurements, positions):
            self.assertAlmostEqual(measurement.x, expected_x, delta=1.5)
            self.assertAlmostEqual(measurement.y, expected_y, delta=1.5)
            self.assertAlmostEqual(measurement.template_x, expected_x, delta=2.0)
            self.assertLess(measurement.detector_disagreement_px, 2.5)
            self.assertGreater(measurement.confidence, 0.5)

        located_box = locate_ball_template_box(
            frames[2],
            background,
            roi=(10, 25, 110, 42),
            ball_diameter_px=14,
        )
        self.assertAlmostEqual(located_box[0] + located_box[2] / 2, 61, delta=1.5)
        self.assertAlmostEqual(located_box[1] + located_box[3] / 2, 45, delta=1.5)

    def test_template_search_associates_silhouette_with_ball_despite_distractor(self) -> None:
        height, width = 80, 130
        background = np.full((height, width), 190, dtype=np.uint8)
        yy, xx = np.mgrid[:height, :width]
        template_frame = background.copy()
        template_frame[(xx - 25) ** 2 + (yy - 40) ** 2 <= 7**2] = 35
        challenged_frame = background.copy()
        challenged_frame[(xx - 45) ** 2 + (yy - 40) ** 2 <= 7**2] = 35
        challenged_frame[31:50, 88:107] = 5

        measurements = detect_ball_positions_dual(
            np.stack([template_frame, challenged_frame]),
            [0.0, 0.0042],
            background,
            roi=(10, 20, 110, 40),
            template_frame_index=0,
            ball_box=(18, 33, 14, 14),
            wall_line_points=((118, 10), (118, 70)),
        )

        self.assertAlmostEqual(measurements[1].x, 45, delta=2.5)
        self.assertAlmostEqual(measurements[1].template_x, 45, delta=2.5)
        self.assertLess(measurements[1].detector_disagreement_px, 2.5)

    def test_neighbouring_motion_prevents_one_frame_association_jump(self) -> None:
        height, width = 90, 130
        background = np.full((height, width), 190, dtype=np.uint8)
        yy, xx = np.mgrid[:height, :width]
        positions = [(30, 45), (45, 45), (60, 45), (75, 45), (90, 45)]
        frames = []
        for index, (x, y) in enumerate(positions):
            image = background.copy()
            image[(xx - x) ** 2 + (yy - y) ** 2 <= 7**2] = 35
            if index == 2:
                # An identical foreground patch wins the raw template tie.
                image[(xx - 20) ** 2 + (yy - y) ** 2 <= 7**2] = 35
            frames.append(image)

        measurements = detect_ball_positions_dual(
            np.stack(frames),
            [0.000, 0.004, 0.008, 0.012, 0.016],
            background,
            roi=(10, 25, 110, 42),
            template_frame_index=0,
            ball_box=(23, 38, 14, 14),
            wall_line_points=((115, 10), (115, 80)),
        )

        self.assertAlmostEqual(measurements[2].template_x, 20, delta=2.5)
        self.assertAlmostEqual(measurements[2].x, 60, delta=2.5)
        self.assertGreater(measurements[2].detector_disagreement_px, 25)

    def test_manual_boundary_anchor_overrides_persistent_template_drift(self) -> None:
        height, width = 135, 120
        background = np.full((height, width), 190, dtype=np.uint8)
        yy, xx = np.mgrid[:height, :width]
        positions = [(60, 40), (60, 55), (60, 70), (60, 85), (60, 100)]
        frames = []
        for x, y in positions:
            image = background.copy()
            image[(xx - x) ** 2 + (yy - y) ** 2 <= 10**2] = 55
            # A stationary rail reflection looks exactly like the reusable
            # appearance template, so unconstrained global matching always
            # associates the wrong object.
            image[6:25, 25] = 35
            image[15, 16:35] = 35
            frames.append(image)
        misleading_template = np.zeros((20, 20), dtype=np.uint8)
        misleading_template[1:20, 10] = 155
        misleading_template[10, 1:20] = 155

        measurements = detect_ball_positions_dual(
            np.stack(frames),
            [0.000, 0.004, 0.008, 0.012, 0.016],
            background,
            roi=(5, 2, 105, 128),
            template_frame_index=4,
            ball_box=(50, 90, 20, 20),
            wall_line_points=((5, 125), (110, 125)),
            template_difference_frame=misleading_template,
        )

        for measurement, (expected_x, expected_y) in zip(measurements, positions):
            self.assertTrue(measurement.contour_detected)
            self.assertAlmostEqual(measurement.x, expected_x, delta=2.5)
            self.assertAlmostEqual(measurement.y, expected_y, delta=2.5)

    def test_post_collision_branch_bootstraps_from_pre_collision_anchor(self) -> None:
        height, width = 135, 120
        background = np.full((height, width), 190, dtype=np.uint8)
        yy, xx = np.mgrid[:height, :width]
        positions = [
            (60, 100),
            (60, 85),
            (60, 70),
            (60, 72),
            (60, 84),
            (60, 96),
            (60, 108),
        ]
        frames = []
        for x, y in positions:
            image = background.copy()
            image[(xx - x) ** 2 + (yy - y) ** 2 <= 10**2] = 55
            # The reusable template consistently chooses this stationary,
            # non-ball reflection on both free-motion branches.
            image[6:25, 25] = 35
            image[15, 16:35] = 35
            frames.append(image)
        misleading_template = np.zeros((20, 20), dtype=np.uint8)
        misleading_template[1:20, 10] = 155
        misleading_template[10, 1:20] = 155

        measurements = detect_ball_positions_dual(
            np.stack(frames),
            [0.000, 0.004, 0.008, 0.012, 0.016, 0.020, 0.024],
            background,
            roi=(5, 2, 105, 128),
            template_frame_index=2,
            ball_box=(50, 60, 20, 20),
            wall_line_points=((5, 58), (110, 58)),
            template_difference_frame=misleading_template,
            trajectory_break_indices=(3,),
        )

        for measurement, (expected_x, expected_y) in zip(measurements, positions):
            self.assertTrue(measurement.contour_detected)
            self.assertAlmostEqual(measurement.x, expected_x, delta=2.5)
            self.assertAlmostEqual(measurement.y, expected_y, delta=2.5)

    def test_collision_branch_boundary_prevents_cross_bounce_association(self) -> None:
        height, width = 90, 130
        background = np.full((height, width), 190, dtype=np.uint8)
        yy, xx = np.mgrid[:height, :width]
        positions = [(30, 45), (45, 45), (60, 45), (60, 45), (45, 45), (30, 45)]
        frames = []
        for x, y in positions:
            image = background.copy()
            image[(xx - x) ** 2 + (yy - y) ** 2 <= 7**2] = 35
            frames.append(image)

        measurements = detect_ball_positions_dual(
            np.stack(frames),
            [0.000, 0.004, 0.008, 0.012, 0.016, 0.020],
            background,
            roi=(10, 25, 110, 42),
            template_frame_index=0,
            ball_box=(23, 38, 14, 14),
            wall_line_points=((115, 10), (115, 80)),
            trajectory_break_indices=(3,),
        )

        for measurement, (expected_x, expected_y) in zip(measurements, positions):
            self.assertAlmostEqual(measurement.x, expected_x, delta=2.5)
            self.assertAlmostEqual(measurement.y, expected_y, delta=2.5)

    def test_sparse_transparent_ball_frame_is_invalid_instead_of_aborting(self) -> None:
        height, width = 90, 130
        background = np.full((height, width), 185, dtype=np.uint8)
        yy, xx = np.mgrid[:height, :width]
        positions = [(30, 45), (45, 45), (60, 45)]
        frames = []
        for index, (x, y) in enumerate(positions):
            image = background.copy()
            radius_squared = (xx - x) ** 2 + (yy - y) ** 2
            if index == 1:
                # A transparent reflective ball can leave only sparse highlights in
                # background difference, too sparse for a physical midpoint.
                image[y - 10 : y + 11, x : x + 1] = 30
                image[y : y + 1, x - 10 : x + 11] = 30
            else:
                image[radius_squared <= 10**2] = 30
            frames.append(image)
        template = np.zeros((20, 20), dtype=np.uint8)
        template[(xx[:20, :20] - 10) ** 2 + (yy[:20, :20] - 10) ** 2 <= 10**2] = 155

        measurements = detect_ball_positions_dual(
            np.stack(frames),
            [0.0, 0.01, 0.02],
            background,
            roi=(10, 20, 110, 50),
            template_frame_index=0,
            ball_box=(20, 35, 20, 20),
            wall_line_points=((118, 10), (118, 80)),
            template_difference_frame=template,
        )

        self.assertEqual(len(measurements), 3)
        self.assertFalse(measurements[1].contour_detected)
        self.assertEqual(measurements[1].confidence, 0.0)
        self.assertAlmostEqual(measurements[1].x, measurements[1].template_x)
        self.assertAlmostEqual(measurements[1].y, measurements[1].template_y)
        self.assertTrue(measurements[0].contour_detected)
        self.assertTrue(measurements[2].contour_detected)

    def test_broad_sparse_ball_uses_foreground_midpoint_without_template_position(self) -> None:
        height, width = 90, 130
        background = np.full((height, width), 185, dtype=np.uint8)
        yy, xx = np.mgrid[:height, :width]
        positions = [(30, 45), (45, 45), (60, 45)]
        frames = []
        for index, (x, y) in enumerate(positions):
            image = background.copy()
            radius_squared = (xx - x) ** 2 + (yy - y) ** 2
            if index == 1:
                sparse_disc = (radius_squared <= 10**2) & (
                    ((xx - x) + (yy - y)) % 5 == 0
                )
                image[sparse_disc] = 30
            else:
                image[radius_squared <= 10**2] = 30
            frames.append(image)
        template = np.zeros((20, 20), dtype=np.uint8)
        template[(xx[:20, :20] - 10) ** 2 + (yy[:20, :20] - 10) ** 2 <= 10**2] = 155

        measurements = detect_ball_positions_dual(
            np.stack(frames),
            [0.0, 0.01, 0.02],
            background,
            roi=(10, 20, 110, 50),
            template_frame_index=0,
            ball_box=(20, 35, 20, 20),
            wall_line_points=((118, 10), (118, 80)),
            template_difference_frame=template,
        )

        sparse = measurements[1]
        self.assertTrue(sparse.contour_detected)
        self.assertEqual(sparse.contour_mode, "sparse_foreground")
        self.assertAlmostEqual(sparse.x, positions[1][0], delta=2.5)
        self.assertAlmostEqual(sparse.y, positions[1][1], delta=2.5)
        self.assertGreater(sparse.confidence, 0.35)

    def test_tracks_fast_ball_that_does_not_overlap_between_frames(self) -> None:
        height, width = 100, 80
        y_positions = [80, 60, 40, 20, 40, 60, 80]
        frames = []
        yy, xx = np.mgrid[:height, :width]
        for y in y_positions:
            image = np.full((height, width), 180, dtype=np.uint8)
            image[(xx - 40) ** 2 + (yy - y) ** 2 <= 5**2] = 20
            frames.append(image)

        points = detect_axis_motion(
            np.stack(frames),
            [index / 30 for index in range(len(frames))],
            (40, 90),
            (40, 10),
            (35, 75, 10, 10),
        )
        coordinates = [
            axis_coordinate((point.x, point.y), (40, 90), (40, 10))
            for point in points
        ]

        expected = [10, 30, 50, 70, 50, 30, 10]
        for actual, target in zip(coordinates, expected):
            self.assertAlmostEqual(actual, target, delta=3)


if __name__ == "__main__":
    unittest.main()
