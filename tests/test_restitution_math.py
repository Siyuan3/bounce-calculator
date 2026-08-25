import unittest

from restitution_math import (
    TrackPoint,
    candidate_impact_index,
    event_window_indices,
    estimate_restitution,
    estimate_fixed_wall_restitution_multiwindow,
    estimate_single_body_restitution_along_axis,
    estimate_two_body_restitution,
    estimate_restitution_from_events,
    estimate_restitution_from_height_events,
    estimate_restitution_from_heights,
    normal_coordinate,
    upward_floor_normal,
)


class RestitutionMathTests(unittest.TestCase):
    def test_multiwindow_fixed_wall_uses_wall_normal_and_real_timestamps(self) -> None:
        contact_time = 1.0
        separation_time = 1.025
        timestamps = [
            0.954,
            0.959,
            0.963,
            0.968,
            0.972,
            0.977,
            0.981,
            0.986,
            0.990,
            0.995,
            1.000,
            1.008,
            1.017,
            1.025,
            1.030,
            1.034,
            1.039,
            1.043,
            1.048,
            1.052,
            1.057,
            1.061,
            1.066,
        ]
        points = []
        for timestamp in timestamps:
            if timestamp < contact_time:
                x = 100.0 + 30.0 * (timestamp - contact_time)
            elif timestamp >= separation_time:
                x = 100.0 - 21.0 * (timestamp - separation_time)
            else:
                x = 500.0
            # Large tangential movement must not enter the normal COR.
            y = 40.0 + 80.0 * timestamp
            points.append(TrackPoint(timestamp, x, y))

        result = estimate_fixed_wall_restitution_multiwindow(
            points,
            wall_point_1=(100.0, 0.0),
            wall_point_2=(100.0, 200.0),
            contact_time=contact_time,
            separation_time=separation_time,
            window_sizes=(5, 7, 9),
        )

        self.assertAlmostEqual(result["coefficient_of_restitution"], 0.7, places=6)
        self.assertEqual(
            [item["fit_frames"] for item in result["window_results"]],
            [5, 7, 9],
        )
        self.assertTrue(
            all(abs(item["coefficient_of_restitution"] - 0.7) < 1e-6
                for item in result["window_results"])
        )
        self.assertAlmostEqual(result["uncertainty"], 0.0, places=6)
        self.assertEqual(result["quality_status"], "ok")
        self.assertGreater(result["incidence_angle_degrees"], 5.0)
        self.assertEqual(result["collision_geometry_status"], "non_normal_collision")

    def test_floor_normal_points_up_regardless_of_click_order(self) -> None:
        self.assertEqual(upward_floor_normal((0, 100), (100, 100)), (0.0, -1.0))
        self.assertEqual(upward_floor_normal((100, 100), (0, 100)), (0.0, -1.0))

    def test_normal_coordinate_is_height_above_horizontal_floor(self) -> None:
        normal = upward_floor_normal((0, 100), (100, 100))
        self.assertAlmostEqual(normal_coordinate((40, 70), (0, 100), normal), 30)

    def test_estimates_known_velocity_ratio(self) -> None:
        points = []
        for index in range(-30, 31):
            timestamp = index / 30
            height = 50 - 10 * timestamp if timestamp <= 0 else 50 + 7 * timestamp
            points.append(TrackPoint(timestamp, 100, 200 - height))

        restitution, _, velocity_before, velocity_after = estimate_restitution(
            points,
            (0, 200),
            (300, 200),
            fit_frames=10,
            contact_padding_frames=3,
        )
        self.assertAlmostEqual(velocity_before, -10, places=6)
        self.assertAlmostEqual(velocity_after, 7, places=6)
        self.assertAlmostEqual(restitution, 0.7, places=6)

    def test_ignores_deformed_contact_centers(self) -> None:
        points = []
        for index in range(-30, 46):
            timestamp = index / 30
            acceleration = -2.0
            if index <= 0:
                height = 50 - 10 * timestamp + 0.5 * acceleration * timestamp**2
            else:
                height = 50 + 8 * timestamp + 0.5 * acceleration * timestamp**2
            if -3 <= index <= 8:
                height += (index % 3 - 1) * 12
            points.append(TrackPoint(timestamp, 100, 200 - height))

        restitution, _, _, _ = estimate_restitution(
            points,
            (0, 200),
            (300, 200),
            fit_frames=10,
            contact_padding_frames=4,
            post_contact_padding_frames=10,
        )
        self.assertAlmostEqual(restitution, 0.8, places=6)

    def test_candidate_impact_is_only_lowest_tracked_center(self) -> None:
        points = [
            TrackPoint(0.0, 10, 20),
            TrackPoint(0.1, 10, 50),
            TrackPoint(0.2, 10, 30),
        ]
        self.assertEqual(candidate_impact_index(points, (0, 100), (100, 100)), 1)

    def test_candidate_impact_prefers_first_bounce_over_later_resting_ball(self) -> None:
        # The first impact is the first clear falling-to-rising turn.  A ball
        # resting slightly lower later in the clip must not move the annotation
        # window away from that first bounce.
        heights = [120, 100, 70, 35, 10, 30, 55, 35, 12, 5, 4, 3]
        points = [
            TrackPoint(index / 120, 10, 200 - height)
            for index, height in enumerate(heights)
        ]

        self.assertEqual(
            candidate_impact_index(points, (0, 200), (100, 200)),
            4,
        )

    def test_event_window_uses_timestamps_instead_of_fixed_frame_count(self) -> None:
        points = [TrackPoint(index / 120, 0, 0) for index in range(241)]

        start, end = event_window_indices(points, 120, seconds_each_side=0.5)

        self.assertEqual((start, end), (60, 181))

    def test_event_fit_excludes_human_labelled_contact_interval(self) -> None:
        points = []
        acceleration = -2.0
        contact_time = 0.0
        separation_time = 0.1
        for index in range(-20, 21):
            timestamp = index / 100
            if timestamp < contact_time:
                height = 50 - 10 * timestamp + 0.5 * acceleration * timestamp**2
            elif timestamp >= separation_time:
                dt = timestamp - separation_time
                height = 45 + 8 * dt + 0.5 * acceleration * dt**2
            else:
                height = 500 + index * 37
            points.append(TrackPoint(timestamp, 100, 200 - height))

        restitution, velocity_before, velocity_after = (
            estimate_restitution_from_events(
                points,
                (0, 200),
                (300, 200),
                contact_time,
                separation_time,
                fit_frames=8,
            )
        )
        self.assertAlmostEqual(velocity_before, -10, places=6)
        self.assertAlmostEqual(velocity_after, 8, places=6)
        self.assertAlmostEqual(restitution, 0.8, places=6)

    def test_event_fit_rejects_reversed_event_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "完全离地时间必须晚于首次接触时间"):
            estimate_restitution_from_events(
                [TrackPoint(index / 30, 0, index) for index in range(30)],
                (0, 100),
                (100, 100),
                0.5,
                0.4,
            )

    def test_two_body_restitution_uses_relative_separation_speed(self) -> None:
        contact_time = 0.0
        separation_time = 0.05
        ball_1 = []
        ball_2 = []
        for index in range(-12, 13):
            timestamp = index / 30
            if timestamp < contact_time:
                position_1 = 100 + 30 * timestamp
                position_2 = 140 + 0 * timestamp
            elif timestamp >= separation_time:
                dt = timestamp - separation_time
                position_1 = 120 + 6 * dt
                position_2 = 120 + 21 * dt
            else:
                position_1 = position_2 = 120
            ball_1.append(TrackPoint(timestamp, 0, position_1))
            ball_2.append(TrackPoint(timestamp, 0, position_2))

        result = estimate_two_body_restitution(
            ball_1,
            ball_2,
            axis_point_1=(0, 0),
            axis_point_2=(0, 100),
            contact_time=contact_time,
            separation_time=separation_time,
            fit_frames=8,
        )

        self.assertAlmostEqual(result[0], 0.5, places=6)
        self.assertAlmostEqual(result[1], 30.0, places=6)
        self.assertAlmostEqual(result[2], 0.0, places=6)
        self.assertAlmostEqual(result[3], 6.0, places=6)
        self.assertAlmostEqual(result[4], 21.0, places=6)

    def test_single_ball_fixed_end_restitution_uses_axis_speed_ratio(self) -> None:
        contact_time = 0.0
        separation_time = 0.04
        points = []
        for index in range(-12, 13):
            timestamp = index / 30
            if timestamp < contact_time:
                position = 100 + 30 * timestamp
            elif timestamp >= separation_time:
                position = 100 - 21 * (timestamp - separation_time)
            else:
                position = 100
            points.append(TrackPoint(timestamp, 0, position))

        restitution, velocity_before, velocity_after = (
            estimate_single_body_restitution_along_axis(
                points,
                axis_point_1=(0, 0),
                axis_point_2=(0, 100),
                contact_time=contact_time,
                separation_time=separation_time,
                fit_frames=8,
            )
        )

        self.assertAlmostEqual(restitution, 0.7, places=6)
        self.assertAlmostEqual(velocity_before, 30.0, places=6)
        self.assertAlmostEqual(velocity_after, -21.0, places=6)

    def test_height_method_uses_release_and_first_rebound_heights(self) -> None:
        heights = [100, 100, 100, 80, 40, 0, -10, -20, 0, 30, 64, 64, 64, 40]
        points = [
            TrackPoint(index / 100, 100, 200 - height)
            for index, height in enumerate(heights)
        ]
        restitution, drop_height, rebound_height, release_index, apex_index = (
            estimate_restitution_from_heights(
                points,
                (0, 200),
                (300, 200),
                contact_time=0.05,
                separation_time=0.08,
            )
        )
        self.assertAlmostEqual(drop_height, 100, places=6)
        self.assertAlmostEqual(rebound_height, 64, places=6)
        self.assertAlmostEqual(restitution, 0.8, places=6)
        self.assertIn(release_index, {0, 1, 2})
        self.assertIn(apex_index, {10, 11, 12})

    def test_height_method_rejects_window_that_ends_during_ascent(self) -> None:
        heights = [100, 100, 80, 40, 0, -10, -20, 0, 20, 40, 60, 80]
        points = [
            TrackPoint(index / 100, 100, 200 - height)
            for index, height in enumerate(heights)
        ]
        with self.assertRaisesRegex(ValueError, "没有覆盖第一次反弹最高点"):
            estimate_restitution_from_heights(
                points,
                (0, 200),
                (300, 200),
                contact_time=0.04,
                separation_time=0.07,
            )

    def test_height_event_method_uses_four_confirmed_frames(self) -> None:
        heights = [100, 90, 60, 20, 0, -10, -20, 0, 30, 64, 50]
        points = [
            TrackPoint(index / 100, 100, 200 - height)
            for index, height in enumerate(heights)
        ]
        restitution, drop_height, rebound_height, release_index, apex_index = (
            estimate_restitution_from_height_events(
                points,
                (0, 200),
                (300, 200),
                release_time=0.0,
                contact_time=0.04,
                separation_time=0.07,
                apex_time=0.09,
            )
        )
        self.assertAlmostEqual(drop_height, 100, places=6)
        self.assertAlmostEqual(rebound_height, 64, places=6)
        self.assertAlmostEqual(restitution, 0.8, places=6)
        self.assertEqual(release_index, 0)
        self.assertEqual(apex_index, 9)


if __name__ == "__main__":
    unittest.main()
