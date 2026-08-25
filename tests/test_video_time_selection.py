import unittest
import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from axis_motion_tracker import BallPositionMeasurement
from calculate_track_collision import (
    build_event_frames,
    build_video_preview_times,
    clear_derived_collision_results,
    deduplicate_extracted_frame_manifest,
    load_ball_setup,
    load_geometry_annotation,
    nearest_timestamp_index,
    load_setup_annotation,
    save_setup_annotation,
    save_ball_setup,
    save_geometry_annotation,
    record_split_setup_provenance,
    select_event_browse_timestamps,
    select_local_tracking_timestamps,
    validate_requested_setup_id,
    validate_setup_mode,
    validate_annotation_video,
    validate_extracted_frame_manifest,
    video_identity,
    write_track_points,
)
from restitution_math import TrackPoint


class VideoTimeSelectionTests(unittest.TestCase):
    def test_preview_times_cover_the_whole_video(self) -> None:
        source = [index * 0.1 for index in range(121)]
        times = build_video_preview_times(source, maximum_frames=7)

        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 12.0)
        self.assertEqual(len(times), 7)

    def test_preview_times_do_not_invent_end_boundary_timestamp(self) -> None:
        source = [0.0, 0.033, 0.067, 0.1]
        times = build_video_preview_times(source, maximum_frames=121)

        self.assertEqual(times[0], 0.0)
        self.assertEqual(times[-1], 0.1)
        self.assertEqual(times, source)

    def test_preview_times_reject_empty_video_timestamp_list(self) -> None:
        with self.assertRaisesRegex(ValueError, "真实帧时间戳"):
            build_video_preview_times([])

    def test_nearest_timestamp_index(self) -> None:
        timestamps = [1.0, 1.2, 1.4, 1.6]

        self.assertEqual(nearest_timestamp_index(timestamps, 1.31), 2)
        self.assertEqual(nearest_timestamp_index(timestamps, -5.0), 0)
        self.assertEqual(nearest_timestamp_index(timestamps, 9.0), 3)

    def test_event_browse_timestamps_are_local_to_selected_second(self) -> None:
        timestamps = [index / 10 for index in range(101)]

        selected = select_event_browse_timestamps(
            timestamps,
            reference_time=5.0,
            seconds_each_side=0.6,
        )

        self.assertAlmostEqual(selected[0], 4.4)
        self.assertAlmostEqual(selected[-1], 5.6)
        self.assertLess(len(selected), len(timestamps))

    def test_local_tracking_uses_only_pre_and_post_collision_frames(self) -> None:
        timestamps = [index / 100 for index in range(200)]

        pre, post = select_local_tracking_timestamps(
            timestamps,
            contact_time=0.90,
            separation_time=1.00,
            fit_frames=16,
            margin_frames=4,
        )

        self.assertEqual(len(pre), 20)
        self.assertEqual(len(post), 20)
        self.assertLess(pre[-1], 0.90)
        self.assertGreaterEqual(post[0], 1.00)
        self.assertNotIn(0.95, pre + post)

    def test_local_tracking_rejects_insufficient_pre_collision_frames(self) -> None:
        timestamps = [index / 100 for index in range(50)]

        with self.assertRaisesRegex(ValueError, "碰撞前帧不足"):
            select_local_tracking_timestamps(
                timestamps,
                contact_time=0.10,
                separation_time=0.20,
                fit_frames=16,
                margin_frames=4,
            )

    def test_extracted_frame_manifest_rejects_duplicate_actual_pts(self) -> None:
        rows = [
            {
                "selection_index": "0",
                "requested_timestamp": "21.500000000",
                "actual_timestamp": "21.500000000",
                "filename": "frame_0000.jpg",
            },
            {
                "selection_index": "1",
                "requested_timestamp": "21.533333333",
                "actual_timestamp": "21.500000000",
                "filename": "frame_0001.jpg",
            },
        ]

        with self.assertRaisesRegex(ValueError, "重复.*PTS"):
            validate_extracted_frame_manifest(rows, requested_count=2)

        unique_rows = deduplicate_extracted_frame_manifest(
            rows,
            requested_count=2,
        )

        self.assertEqual(len(unique_rows), 1)
        self.assertEqual(unique_rows[0]["filename"], "frame_0000.jpg")

    def test_local_tracking_can_overfetch_candidates_for_pts_deduplication(self) -> None:
        timestamps = [index / 100 for index in range(200)]

        pre, post = select_local_tracking_timestamps(
            timestamps,
            contact_time=0.90,
            separation_time=1.00,
            fit_frames=16,
            margin_frames=4,
            candidate_multiplier=2,
        )

        self.assertEqual(len(pre), 40)
        self.assertEqual(len(post), 40)

    def test_local_track_csv_labels_the_two_disconnected_phases(self) -> None:
        points = [
            TrackPoint(timestamp=float(index), x=float(index), y=0.0)
            for index in range(4)
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "track.csv"

            write_track_points(path, points, pre_collision_count=2)

            rows = path.read_text().splitlines()
        self.assertIn("phase", rows[0])
        self.assertIn("before_collision", rows[1])
        self.assertIn("before_collision", rows[2])
        self.assertIn("after_separation", rows[3])
        self.assertIn("after_separation", rows[4])

    def test_local_track_csv_rejects_duplicate_pts(self) -> None:
        points = [
            TrackPoint(timestamp=1.0, x=0.0, y=0.0),
            TrackPoint(timestamp=1.0, x=1.0, y=0.0),
            TrackPoint(timestamp=2.0, x=2.0, y=0.0),
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "track.csv"

            with self.assertRaisesRegex(ValueError, "严格递增"):
                write_track_points(path, points, pre_collision_count=2)

    def test_track_csv_distinguishes_eligibility_from_window_membership(self) -> None:
        points = [
            TrackPoint(timestamp=float(index), x=float(index), y=0.0)
            for index in range(12)
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "track.csv"

            write_track_points(
                path,
                points,
                pre_collision_count=6,
                excluded_timestamps={0.0},
                fit_windows=(3, 5),
            )

            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(rows[0]["eligible_for_fitting"], "0")
        self.assertEqual(rows[0]["fit_windows"], "")
        self.assertEqual(rows[1]["eligible_for_fitting"], "1")
        self.assertEqual(rows[1]["fit_windows"], "5")
        self.assertEqual(rows[3]["fit_windows"], "3|5")
        self.assertEqual(rows[6]["fit_windows"], "3|5")
        self.assertEqual(rows[11]["fit_windows"], "")

    def test_track_csv_marks_missing_contour_as_ineligible(self) -> None:
        points = [
            BallPositionMeasurement(
                timestamp=float(index),
                x=float(index),
                y=0.0,
                template_x=float(index),
                template_y=0.0,
                detector_disagreement_px=0.0,
                confidence=0.0 if index == 1 else 0.9,
                contour_detected=index != 1,
            )
            for index in range(6)
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "track.csv"

            write_track_points(
                path,
                points,
                pre_collision_count=3,
                fit_windows=(2,),
            )

            with path.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
        self.assertEqual(rows[1]["contour_detected"], "0")
        self.assertEqual(rows[1]["eligible_for_fitting"], "0")
        self.assertEqual(rows[0]["contour_detected"], "1")
        self.assertEqual(rows[0]["eligible_for_fitting"], "1")

    def test_batch_setup_persists_and_resolves_difference_template(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            background_path = root / "background.png"
            frame_path = root / "frame.png"
            setup_path = root / "batch_setup.json"
            background = np.full((40, 60), 180, dtype=np.uint8)
            frame = background.copy()
            frame[12:22, 20:30] = 30
            Image.fromarray(background).save(background_path)
            Image.fromarray(frame).save(frame_path)
            annotation = {
                "collision_roi": [5, 5, 50, 30],
                "wall_line_points": [[50, 5], [50, 35]],
                "ball_box": [20, 12, 10, 10],
                "image_size": [60, 40],
            }
            video = root / "source.mov"
            video.write_bytes(b"video")

            template_path = save_setup_annotation(
                setup_path,
                annotation,
                video,
                template_frame_path=frame_path,
                background_path=background_path,
            )
            loaded = load_setup_annotation(setup_path, (60, 40))

            self.assertTrue(template_path.is_file())
            self.assertEqual(Path(loaded["template_difference_path"]), template_path)
            self.assertEqual(np.asarray(Image.open(template_path)).shape, (10, 10))
            payload = json.loads(setup_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["setup_version"], 2)
            self.assertEqual(payload["template_difference_path"], template_path.name)

    def test_geometry_annotation_is_independent_of_ball_setup(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            geometry_path = root / "track_geometry.json"
            video = root / "source.mov"
            video.write_bytes(b"video")
            annotation = {
                "collision_roi": [5, 5, 50, 30],
                "wall_line_points": [[50, 5], [50, 35]],
                "image_size": [60, 40],
            }

            save_geometry_annotation(
                geometry_path,
                annotation,
                video,
                geometry_id="track_01",
            )
            loaded = load_geometry_annotation(geometry_path, (60, 40))

            self.assertEqual(loaded["geometry_id"], "track_01")
            self.assertEqual(loaded["collision_roi"], [5, 5, 50, 30])
            self.assertEqual(loaded["wall_line_points"], [[50, 5], [50, 35]])
            self.assertNotIn("ball_diameter_pixel", loaded)
            self.assertNotIn("template_difference_path", loaded)

    def test_legacy_combined_setup_can_be_loaded_as_shared_geometry(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_batch.json"
            path.write_text(
                json.dumps(
                    {
                        "setup_version": 2,
                        "collision_roi": [5, 5, 50, 30],
                        "wall_line_points": [[50, 5], [50, 35]],
                        "ball_diameter_pixel": 10,
                        "image_size": [60, 40],
                        "template_difference_path": "old_template.png",
                    }
                ),
                encoding="utf-8",
            )

            loaded = load_geometry_annotation(path, (60, 40))

            self.assertEqual(loaded["geometry_id"], "legacy_batch")
            self.assertTrue(loaded["loaded_from_legacy_combined_setup"])

    def test_two_balls_share_geometry_but_keep_distinct_templates(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            background_path = root / "background.png"
            frame_1_path = root / "ball_01.png"
            frame_2_path = root / "ball_02.png"
            background = np.full((40, 60), 180, dtype=np.uint8)
            frame_1 = background.copy()
            frame_1[12:22, 20:30] = 30
            frame_2 = background.copy()
            frame_2[10:24, 18:32] = 70
            Image.fromarray(background).save(background_path)
            Image.fromarray(frame_1).save(frame_1_path)
            Image.fromarray(frame_2).save(frame_2_path)
            video = root / "source.mov"
            video.write_bytes(b"video")

            first_path = root / "ball_01_setup.json"
            second_path = root / "ball_02_setup.json"
            save_ball_setup(
                first_path,
                {"ball_box": [20, 12, 10, 10], "image_size": [60, 40]},
                video,
                template_frame_path=frame_1_path,
                background_path=background_path,
                geometry_id="track_01",
                ball_id="ball_01",
            )
            save_ball_setup(
                second_path,
                {"ball_box": [18, 10, 14, 14], "image_size": [60, 40]},
                video,
                template_frame_path=frame_2_path,
                background_path=background_path,
                geometry_id="track_01",
                ball_id="ball_02",
            )

            first = load_ball_setup(
                first_path,
                (60, 40),
                expected_geometry_id="track_01",
            )
            second = load_ball_setup(
                second_path,
                (60, 40),
                expected_geometry_id="track_01",
            )

            self.assertEqual(first["ball_diameter_pixel"], 10)
            self.assertEqual(second["ball_diameter_pixel"], 14)
            self.assertNotEqual(
                first["template_difference_path"],
                second["template_difference_path"],
            )
            self.assertEqual(first["compatible_geometry_id"], "track_01")
            self.assertEqual(second["compatible_geometry_id"], "track_01")

    def test_ball_setup_rejects_a_different_geometry(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "ball_template.png"
            Image.fromarray(np.full((10, 10), 100, dtype=np.uint8)).save(template)
            path = root / "ball_setup.json"
            path.write_text(
                json.dumps(
                    {
                        "ball_setup_version": 1,
                        "ball_id": "ball_02",
                        "ball_diameter_pixel": 10,
                        "image_size": [60, 40],
                        "compatible_geometry_id": "track_01",
                        "template_difference_path": template.name,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "几何.*不一致"):
                load_ball_setup(
                    path,
                    (60, 40),
                    expected_geometry_id="track_02",
                )

    def test_ball_setup_rejects_an_empty_ball_id(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            template = root / "ball_template.png"
            Image.fromarray(np.full((10, 10), 100, dtype=np.uint8)).save(template)
            path = root / "ball_setup.json"
            path.write_text(
                json.dumps(
                    {
                        "ball_setup_version": 1,
                        "ball_id": "   ",
                        "ball_diameter_pixel": 10,
                        "image_size": [60, 40],
                        "compatible_geometry_id": "track_01",
                        "template_difference_path": template.name,
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "ball_id"):
                load_ball_setup(
                    path,
                    (60, 40),
                    expected_geometry_id="track_01",
                )

    def test_explicit_ball_id_must_match_existing_ball_setup(self) -> None:
        with self.assertRaisesRegex(ValueError, "ball_02.*ball_01"):
            validate_requested_setup_id("ball_02", "ball_01", "ball_id")

    def test_split_trial_cannot_fall_back_to_default_setup_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "拆分配置"):
            validate_setup_mode(
                setup_path=None,
                geometry_path=None,
                ball_setup_path=None,
                annotation={"geometry_id": "track_01", "ball_id": "ball_02"},
            )

    def test_event_draft_records_split_mode_before_ball_annotation(self) -> None:
        annotation: dict[str, object] = {"annotation_status": "events_selected"}
        record_split_setup_provenance(
            annotation,
            Path("geometry.json"),
            Path("ball_02.json"),
        )

        self.assertEqual(annotation["setup_mode"], "split_geometry_and_ball")
        self.assertTrue(str(annotation["geometry_annotation"]).endswith("geometry.json"))
        self.assertTrue(str(annotation["ball_setup_annotation"]).endswith("ball_02.json"))
        with self.assertRaisesRegex(ValueError, "拆分配置"):
            validate_setup_mode(
                setup_path=None,
                geometry_path=None,
                ball_setup_path=None,
                annotation=annotation,
            )

    def test_event_browser_receives_the_complete_tracking_interval(self) -> None:
        timestamps = [index * 0.1 for index in range(5)]
        paths = [Path(f"frame_{index}.jpg") for index in range(5)]

        frames = build_event_frames(timestamps, paths)

        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[0]["timestamp"], 0.0)
        self.assertEqual(frames[-1]["timestamp"], 0.4)

    def test_event_browser_rejects_single_frame_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "至少需要 2"):
            build_event_frames([1.0], [Path("frame.jpg")])

    def test_draft_keeps_manual_labels_but_removes_stale_results(self) -> None:
        annotation = {
            "ball_box": [1, 2, 3, 4],
            "first_contact_timestamp_s": 1.0,
            "track_csv": "old.csv",
            "coefficient_of_restitution": 0.5,
        }

        clear_derived_collision_results(annotation)

        self.assertIn("ball_box", annotation)
        self.assertIn("first_contact_timestamp_s", annotation)
        self.assertNotIn("track_csv", annotation)
        self.assertNotIn("coefficient_of_restitution", annotation)

    def test_annotation_identity_detects_replaced_video(self) -> None:
        with TemporaryDirectory() as directory:
            video = Path(directory) / "video.mov"
            video.write_bytes(b"original")
            annotation = {"video_identity": video_identity(video)}
            validate_annotation_video(annotation, video)

            video.write_bytes(b"replacement-is-different")

            with self.assertRaisesRegex(ValueError, "已被替换"):
                validate_annotation_video(annotation, video)

    def test_legacy_annotation_is_rejected_even_at_matching_path(self) -> None:
        with TemporaryDirectory() as directory:
            video = Path(directory) / "video.mov"
            video.write_bytes(b"video")
            annotation = {"video": str(video)}

            with self.assertRaisesRegex(ValueError, "缺少视频身份信息"):
                validate_annotation_video(annotation, video)


if __name__ == "__main__":
    unittest.main()
