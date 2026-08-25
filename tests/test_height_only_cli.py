import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import calculate_restitution_from_heights as height_cli
from restitution_math import TrackPoint


class HeightOnlyCliTests(unittest.TestCase):
    def test_video_first_frame_is_release_and_output_contains_only_height_method(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "drop.mov"
            video.write_bytes(b"video")
            output = root / "drop_height.json"
            points = [
                TrackPoint(0.0, 100, 100),
                TrackPoint(0.2, 100, 200),
                TrackPoint(0.3, 100, 200),
                TrackPoint(0.5, 100, 164),
            ]
            annotation = {
                "ball_box": [80, 80, 40, 40],
                "floor_points": [[0, 200], [300, 200]],
                "release_timestamp_s": 0.1,
                "frame_velocity_method": {"stale": True},
            }
            events = {
                "first_contact_timestamp_s": 0.2,
                "first_separated_timestamp_s": 0.3,
            }
            apex = {
                "first_rebound_apex_analysis_frame": 3,
                "first_rebound_apex_timestamp_s": 0.5,
            }

            def bridge(arguments: list[str]) -> subprocess.CompletedProcess[str]:
                stdout = '{"duration_s":0.6}' if arguments[0] == "info" else ""
                return subprocess.CompletedProcess(arguments, 0, stdout=stdout, stderr="")

            args = argparse.Namespace(
                video=video,
                end_time=None,
                save_annotation=output,
                annotation=None,
            )
            captured = io.StringIO()
            with (
                patch.object(height_cli, "parse_args", return_value=args),
                patch.object(height_cli, "run_bridge", side_effect=bridge),
                patch.object(height_cli, "AnnotationWindow") as annotation_window,
                patch.object(height_cli, "read_track", return_value=points),
                patch.object(height_cli, "extract_event_frames", return_value=([{}], 0)),
                patch.object(height_cli, "EventSelectionWindow") as event_window,
                patch.object(height_cli, "extract_all_frames", return_value=[{}]),
                patch.object(height_cli, "HeightFrameSelectionWindow") as height_window,
                contextlib.redirect_stdout(captured),
            ):
                annotation_window.return_value.run.return_value = annotation
                event_window.return_value.run.return_value = events
                height_window.return_value.run.return_value = apex
                return_code = height_cli.main()

            self.assertEqual(return_code, 0)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved["release_analysis_frame"], 0)
            self.assertEqual(saved["release_timestamp_s"], 0.0)
            self.assertEqual(saved["height_method"]["release_analysis_frame"], 0)
            self.assertAlmostEqual(
                saved["height_method"]["coefficient_of_restitution"], 0.6
            )
            self.assertNotIn("frame_velocity_method", saved)
            self.assertEqual(captured.getvalue(), "高度比法恢复系数 e_height = 0.600\n")


if __name__ == "__main__":
    unittest.main()
