#!/usr/bin/env python3
"""Calculate football-floor restitution from drop and first-rebound heights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

from calculate_restitution import (
    AnnotationWindow,
    EventSelectionWindow,
    extract_event_frames,
    read_track,
    run_bridge,
)
from restitution_math import (
    estimate_restitution_from_events,
    estimate_restitution_from_height_events,
)


def extract_all_frames(
    video: Path, points: list, work: Path
) -> list[dict[str, object]]:
    timestamps_path = work / "height_event_timestamps.txt"
    timestamps_path.write_text(
        "\n".join(f"{point.timestamp:.9f}" for point in points) + "\n",
        encoding="utf-8",
    )
    frames_dir = work / "height_event_frames"
    run_bridge(["frames", str(video), str(timestamps_path), str(frames_dir)])
    return [
        {
            "analysis_frame_index": index,
            "timestamp": point.timestamp,
            "image_path": frames_dir / f"frame_{index:04d}.jpg",
        }
        for index, point in enumerate(points)
    ]


class HeightFrameSelectionWindow:
    """Select release and first-rebound apex frames from the full clip."""

    def __init__(
        self,
        frames: list[dict[str, object]],
        contact_time: float,
        separation_time: float,
    ) -> None:
        if not frames:
            raise ValueError("没有可用于高度事件标注的视频帧")
        self.frames = frames
        self.contact_time = contact_time
        self.separation_time = separation_time
        self.position = 0
        self.release_position: int | None = None
        self.apex_position: int | None = None
        self.result: dict[str, object] | None = None

        self.root = tk.Tk()
        self.root.title("高度法恢复系数：选择释放帧和第一次反弹最高点")
        tk.Label(
            self.root,
            text="逐帧浏览：选择释放帧和第一次反弹达到最高点的帧",
            font=("Arial", 15),
        ).pack(pady=(10, 6))
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10)
        self.frame_status = tk.StringVar()
        tk.Label(self.root, textvariable=self.frame_status, font=("Menlo", 12)).pack(
            pady=(6, 4)
        )

        navigation = tk.Frame(self.root)
        navigation.pack(fill="x", padx=10, pady=4)
        tk.Button(navigation, text="◀ 5帧", command=lambda: self.move(-5)).pack(
            side="left"
        )
        tk.Button(navigation, text="◀ 上一帧", command=lambda: self.move(-1)).pack(
            side="left", padx=6
        )
        tk.Button(navigation, text="下一帧 ▶", command=lambda: self.move(1)).pack(
            side="right", padx=6
        )
        tk.Button(navigation, text="5帧 ▶", command=lambda: self.move(5)).pack(
            side="right"
        )

        event_controls = tk.Frame(self.root)
        event_controls.pack(fill="x", padx=10, pady=6)
        tk.Button(
            event_controls,
            text="设为释放帧",
            command=self.mark_release,
            bg="#ffd166",
        ).pack(side="left")
        tk.Button(
            event_controls,
            text="设为第一次反弹最高点",
            command=self.mark_apex,
            bg="#8ecae6",
        ).pack(side="right")

        self.selection_status = tk.StringVar(value="释放：未选择｜反弹最高点：未选择")
        tk.Label(self.root, textvariable=self.selection_status, font=("Arial", 12)).pack(
            pady=4
        )
        footer = tk.Frame(self.root)
        footer.pack(fill="x", padx=10, pady=(4, 10))
        tk.Button(footer, text="重置选择", command=self.reset).pack(side="left")
        self.confirm_button = tk.Button(
            footer, text="确定", command=self.confirm, state="disabled"
        )
        self.confirm_button.pack(side="right")

        self.root.bind("<Left>", lambda _: self.move(-1))
        self.root.bind("<Right>", lambda _: self.move(1))
        self.root.bind("<Shift-Left>", lambda _: self.move(-5))
        self.root.bind("<Shift-Right>", lambda _: self.move(5))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.photo: ImageTk.PhotoImage | None = None
        self.update_frame()

    def current(self) -> dict[str, object]:
        return self.frames[self.position]

    def move(self, offset: int) -> None:
        self.position = max(0, min(self.position + offset, len(self.frames) - 1))
        self.update_frame()

    def update_frame(self) -> None:
        frame = self.current()
        source = Image.open(Path(frame["image_path"])).convert("RGB")
        scale = min(720 / source.width, 760 / source.height, 1)
        display = source.resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
        self.photo = ImageTk.PhotoImage(display)
        self.image_label.configure(image=self.photo)
        self.frame_status.set(
            f"帧 {self.position + 1}/{len(self.frames)}｜"
            f"分析帧 {frame['analysis_frame_index']}｜"
            f"真实时间 {float(frame['timestamp']):.6f} s"
        )

    def mark_release(self) -> None:
        self.release_position = self.position
        self.update_selection_status()

    def mark_apex(self) -> None:
        self.apex_position = self.position
        self.update_selection_status()

    def reset(self) -> None:
        self.release_position = None
        self.apex_position = None
        self.update_selection_status()

    def update_selection_status(self) -> None:
        release = (
            "未选择"
            if self.release_position is None
            else f"{float(self.frames[self.release_position]['timestamp']):.6f} s"
        )
        apex = (
            "未选择"
            if self.apex_position is None
            else f"{float(self.frames[self.apex_position]['timestamp']):.6f} s"
        )
        self.selection_status.set(f"释放：{release}｜反弹最高点：{apex}")
        valid = False
        if self.release_position is not None and self.apex_position is not None:
            release_time = float(self.frames[self.release_position]["timestamp"])
            apex_time = float(self.frames[self.apex_position]["timestamp"])
            valid = (
                release_time < self.contact_time < self.separation_time < apex_time
            )
        self.confirm_button.configure(state="normal" if valid else "disabled")

    def confirm(self) -> None:
        if self.release_position is None or self.apex_position is None:
            return
        release = self.frames[self.release_position]
        apex = self.frames[self.apex_position]
        release_time = float(release["timestamp"])
        apex_time = float(apex["timestamp"])
        if not release_time < self.contact_time < self.separation_time < apex_time:
            messagebox.showwarning(
                "事件顺序错误",
                "必须满足：释放帧 < 首次接触帧 < 完全离地帧 < 反弹最高点帧",
            )
            return
        self.result = {
            "release_analysis_frame": int(release["analysis_frame_index"]),
            "release_timestamp_s": release_time,
            "first_rebound_apex_analysis_frame": int(apex["analysis_frame_index"]),
            "first_rebound_apex_timestamp_s": apex_time,
        }
        self.root.destroy()

    def close(self) -> None:
        self.result = None
        self.root.destroy()

    def run(self) -> dict[str, object]:
        self.root.mainloop()
        if self.result is None:
            raise RuntimeError("用户取消了释放与反弹最高点标注")
        return self.result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过释放高度和第一次反弹高度计算恢复系数"
    )
    parser.add_argument("video", type=Path, help="原始 MOV/MP4 视频")
    parser.add_argument(
        "--start-time", type=float, default=14.0, help="跟踪开始时间（秒）"
    )
    parser.add_argument(
        "--end-time", type=float, default=16.5, help="跟踪结束时间（秒）"
    )
    parser.add_argument(
        "--save-annotation", type=Path, help="可选：保存人工标注和高度法结果 JSON"
    )
    parser.add_argument(
        "--annotation", type=Path, help="可选：复用已有标注 JSON"
    )
    return parser.parse_args()


def save_annotation(
    path: Path,
    annotation: dict[str, object],
    start_time: float,
    end_time: float,
) -> None:
    """Persist human selections before any derived-value validation."""
    annotation["tracking_start_time_s"] = start_time
    annotation["tracking_end_time_s"] = end_time
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    video = args.video.expanduser().resolve()
    if not video.is_file():
        print(f"错误：找不到视频：{video}", file=sys.stderr)
        return 2
    if args.end_time <= args.start_time:
        print("错误：--end-time 必须大于 --start-time", file=sys.stderr)
        return 2

    try:
        with tempfile.TemporaryDirectory(prefix="bounce_height_restitution_") as directory:
            work = Path(directory)
            frame_path = work / "annotation_frame.png"
            track_path = work / "track.csv"

            if args.annotation:
                annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
                saved_start = annotation.get("tracking_start_time_s")
                if saved_start is None:
                    raise ValueError(
                        "该标注缺少 tracking_start_time_s；高度法请删除 --annotation，"
                        "从释放前重新框选足球"
                    )
                if abs(float(saved_start) - args.start_time) > 1e-6:
                    raise ValueError(
                        "标注中的足球框来自不同的开始时间，不能用于当前高度法窗口；"
                        "请删除 --annotation 并重新框选足球"
                    )
            else:
                run_bridge(
                    ["frame", str(video), str(args.start_time), str(frame_path)]
                )
                annotation = AnnotationWindow(frame_path).run()

            box = annotation["ball_box"]
            run_bridge(
                [
                    "track",
                    str(video),
                    str(args.start_time),
                    str(args.end_time),
                    *(str(value) for value in box),
                    str(track_path),
                ]
            )
            points = read_track(track_path)
            floor_1, floor_2 = annotation["floor_points"]

            event_keys = {
                "first_contact_timestamp_s",
                "first_separated_timestamp_s",
            }
            if event_keys.issubset(annotation):
                contact_time = float(annotation["first_contact_timestamp_s"])
                separation_time = float(annotation["first_separated_timestamp_s"])
            else:
                frames, initial_position = extract_event_frames(
                    video, points, annotation["floor_points"], work
                )
                events = EventSelectionWindow(frames, initial_position).run()
                annotation.update(events)
                contact_time = float(events["first_contact_timestamp_s"])
                separation_time = float(events["first_separated_timestamp_s"])

            height_event_keys = {
                "release_timestamp_s",
                "first_rebound_apex_timestamp_s",
            }
            if height_event_keys.issubset(annotation):
                release_time = float(annotation["release_timestamp_s"])
                apex_time = float(annotation["first_rebound_apex_timestamp_s"])
            else:
                height_frames = extract_all_frames(video, points, work)
                height_events = HeightFrameSelectionWindow(
                    height_frames, contact_time, separation_time
                ).run()
                annotation.update(height_events)
                release_time = float(height_events["release_timestamp_s"])
                apex_time = float(height_events["first_rebound_apex_timestamp_s"])

            # Save the four human-confirmed events immediately.  Derived
            # height validation may still fail, but the operator should never
            # have to repeat annotations merely because a calculation failed.
            if args.save_annotation:
                save_annotation(
                    args.save_annotation,
                    annotation,
                    args.start_time,
                    args.end_time,
                )

            restitution, drop_height, rebound_height, release_index, apex_index = (
                estimate_restitution_from_height_events(
                    points,
                    tuple(floor_1),
                    tuple(floor_2),
                    release_time,
                    contact_time,
                    separation_time,
                    apex_time,
                )
            )
            frame_restitution, velocity_before, velocity_after = (
                estimate_restitution_from_events(
                    points,
                    tuple(floor_1),
                    tuple(floor_2),
                    contact_time,
                    separation_time,
                )
            )
            annotation["height_method"] = {
                "release_analysis_frame": release_index,
                "release_timestamp_s": points[release_index].timestamp,
                "drop_height_pixel": drop_height,
                "first_rebound_apex_analysis_frame": apex_index,
                "first_rebound_apex_timestamp_s": points[apex_index].timestamp,
                "first_rebound_height_pixel": rebound_height,
                "coefficient_of_restitution": restitution,
                "formula": "sqrt(first_rebound_height_pixel / drop_height_pixel)",
            }
            annotation["frame_velocity_method"] = {
                "first_contact_timestamp_s": contact_time,
                "first_separated_timestamp_s": separation_time,
                "normal_velocity_before_pixel_per_s": velocity_before,
                "normal_velocity_after_pixel_per_s": velocity_after,
                "coefficient_of_restitution": frame_restitution,
                "formula": "abs(normal_velocity_after / normal_velocity_before)",
                "fit_frames_each_side": 10,
                "timestamp_basis": "video_presentation_timestamp",
            }

            if args.save_annotation:
                save_annotation(
                    args.save_annotation,
                    annotation,
                    args.start_time,
                    args.end_time,
                )

            print(f"高度比法恢复系数 e_height = {restitution:.3f}")
            print(f"轨迹帧速度比法恢复系数 e_frame = {frame_restitution:.3f}")
            return 0
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        print(f"错误：视频处理失败\n{detail}", file=sys.stderr)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
