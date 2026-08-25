#!/usr/bin/env python3
"""Interactively calculate a ball-floor coefficient of restitution."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageTk

from restitution_math import (
    TrackPoint,
    candidate_impact_index,
    event_window_indices,
    estimate_restitution_from_events,
)


SCRIPT_DIR = Path(__file__).resolve().parent
SWIFT_BRIDGE = SCRIPT_DIR / "video_bridge.swift"


def run_bridge(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.setdefault(
        "CLANG_MODULE_CACHE_PATH", "/private/tmp/bounce_restitution_clang_cache"
    )
    environment.setdefault(
        "SWIFT_MODULECACHE_PATH", "/private/tmp/bounce_restitution_swift_cache"
    )
    return subprocess.run(
        ["/usr/bin/swift", str(SWIFT_BRIDGE), *arguments],
        check=True,
        text=True,
        capture_output=True,
        env=environment,
    )


class AnnotationWindow:
    def __init__(self, image_path: Path) -> None:
        self.root = tk.Tk()
        self.root.title("恢复系数计算：标注足球和地板")
        self.source = Image.open(image_path).convert("RGB")
        max_width, max_height = 720, 820
        self.scale = min(max_width / self.source.width, max_height / self.source.height, 1)
        display_size = (
            round(self.source.width * self.scale),
            round(self.source.height * self.scale),
        )
        self.display = self.source.resize(display_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display)

        self.instructions = tk.StringVar(
            value="第 1 步：用鼠标拖出一个刚好包住足球的矩形框"
        )
        tk.Label(self.root, textvariable=self.instructions, font=("Arial", 15)).pack(
            pady=(10, 6)
        )
        self.canvas = tk.Canvas(
            self.root,
            width=self.display.width,
            height=self.display.height,
            cursor="crosshair",
            highlightthickness=0,
        )
        self.canvas.pack(padx=10)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")

        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=10, pady=10)
        tk.Button(controls, text="重置", command=self.reset).pack(side="left")
        self.confirm_button = tk.Button(
            controls, text="确定", command=self.confirm, state="disabled"
        )
        self.confirm_button.pack(side="right")

        self.stage = "ball"
        self.drag_start: tuple[float, float] | None = None
        self.ball_box_display: tuple[float, float, float, float] | None = None
        self.floor_points_display: list[tuple[float, float]] = []
        self.result: dict[str, object] | None = None
        self.canvas.bind("<Button-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def reset(self) -> None:
        self.canvas.delete("annotation")
        self.stage = "ball"
        self.drag_start = None
        self.ball_box_display = None
        self.floor_points_display = []
        self.confirm_button.configure(state="disabled")
        self.instructions.set("第 1 步：用鼠标拖出一个刚好包住足球的矩形框")

    def mouse_down(self, event: tk.Event) -> None:
        if self.stage == "ball":
            self.drag_start = (event.x, event.y)
        elif self.stage == "floor":
            self.add_floor_point(event.x, event.y)

    def mouse_drag(self, event: tk.Event) -> None:
        if self.stage != "ball" or self.drag_start is None:
            return
        self.canvas.delete("ball_preview")
        self.canvas.create_rectangle(
            self.drag_start[0],
            self.drag_start[1],
            event.x,
            event.y,
            outline="#00ff66",
            width=3,
            tags=("annotation", "ball_preview"),
        )

    def mouse_up(self, event: tk.Event) -> None:
        if self.stage != "ball" or self.drag_start is None:
            return
        x1, y1 = self.drag_start
        x2, y2 = event.x, event.y
        x1, x2 = sorted((max(0, x1), min(self.display.width, x2)))
        y1, y2 = sorted((max(0, y1), min(self.display.height, y2)))
        if x2 - x1 < 20 or y2 - y1 < 20:
            messagebox.showwarning("框选过小", "请拖出一个完整包住足球的矩形框")
            self.canvas.delete("ball_preview")
            return
        self.ball_box_display = (x1, y1, x2 - x1, y2 - y1)
        self.stage = "floor"
        self.drag_start = None
        self.instructions.set(
            "第 2 步：沿与地板平行的直线点击两个远点（建议点击墙地交界线）"
        )

    def add_floor_point(self, x: float, y: float) -> None:
        if len(self.floor_points_display) >= 2:
            return
        self.floor_points_display.append((x, y))
        radius = 5
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="#ffcc00",
            outline="black",
            tags="annotation",
        )
        if len(self.floor_points_display) == 2:
            first, second = self.floor_points_display
            self.canvas.create_line(
                *first, *second, fill="#ffcc00", width=3, tags="annotation"
            )
            self.confirm_button.configure(state="normal")
            self.instructions.set("检查绿色足球框和黄色地板线，正确后点击“确定”")

    def confirm(self) -> None:
        assert self.ball_box_display is not None
        factor = 1.0 / self.scale
        x, y, width, height = self.ball_box_display
        self.result = {
            "ball_box": [x * factor, y * factor, width * factor, height * factor],
            "floor_points": [
                [point[0] * factor, point[1] * factor]
                for point in self.floor_points_display
            ],
            "image_size": [self.source.width, self.source.height],
        }
        self.root.destroy()

    def close(self) -> None:
        self.result = None
        self.root.destroy()

    def run(self) -> dict[str, object]:
        self.root.mainloop()
        if self.result is None:
            raise RuntimeError("用户取消了标注")
        return self.result


class EventSelectionWindow:
    """Let the operator choose first contact and first fully separated frames."""

    def __init__(self, frames: list[dict[str, object]], initial_position: int) -> None:
        if not frames:
            raise ValueError("没有可用于接触事件标注的视频帧")
        self.frames = frames
        self.position = max(0, min(initial_position, len(frames) - 1))
        self.contact_position: int | None = None
        self.separation_position: int | None = None
        self.result: dict[str, object] | None = None

        self.root = tk.Tk()
        self.root.title("恢复系数计算：选择接触与完全分离帧")
        self.instructions = tk.StringVar(
            value="逐帧浏览：先选择物体首次接触碰撞面的帧，再选择完全分离帧"
        )
        tk.Label(self.root, textvariable=self.instructions, font=("Arial", 15)).pack(
            pady=(10, 6)
        )

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
            text="设为首次接触帧",
            command=self.mark_contact,
            bg="#ffd166",
        ).pack(side="left")
        tk.Button(
            event_controls,
            text="设为完全分离帧",
            command=self.mark_separation,
            bg="#8ecae6",
        ).pack(side="right")

        self.selection_status = tk.StringVar(value="首次接触：未选择｜完全分离：未选择")
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
        max_width, max_height = 720, 760
        scale = min(max_width / source.width, max_height / source.height, 1)
        display = source.resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
        self.photo = ImageTk.PhotoImage(display)
        self.image_label.configure(image=self.photo)
        self.frame_status.set(
            "候选帧 "
            f"{self.position + 1}/{len(self.frames)}｜"
            f"分析帧 {frame['analysis_frame_index']}｜"
            f"真实时间 {float(frame['timestamp']):.6f} s"
        )

    def mark_contact(self) -> None:
        self.contact_position = self.position
        self.update_selection_status()

    def mark_separation(self) -> None:
        self.separation_position = self.position
        self.update_selection_status()

    def reset(self) -> None:
        self.contact_position = None
        self.separation_position = None
        self.update_selection_status()

    def update_selection_status(self) -> None:
        contact = (
            "未选择"
            if self.contact_position is None
            else f"{float(self.frames[self.contact_position]['timestamp']):.6f} s"
        )
        separation = (
            "未选择"
            if self.separation_position is None
            else f"{float(self.frames[self.separation_position]['timestamp']):.6f} s"
        )
        self.selection_status.set(f"首次接触：{contact}｜完全分离：{separation}")
        valid = (
            self.contact_position is not None
            and self.separation_position is not None
            and self.contact_position < self.separation_position
        )
        self.confirm_button.configure(state="normal" if valid else "disabled")

    def confirm(self) -> None:
        if self.contact_position is None or self.separation_position is None:
            return
        if self.contact_position >= self.separation_position:
            messagebox.showwarning("事件顺序错误", "完全分离帧必须晚于首次接触帧")
            return
        contact = self.frames[self.contact_position]
        separation = self.frames[self.separation_position]
        self.result = {
            "first_contact_analysis_frame": int(contact["analysis_frame_index"]),
            "first_contact_timestamp_s": float(contact["timestamp"]),
            "first_separated_analysis_frame": int(
                separation["analysis_frame_index"]
            ),
            "first_separated_timestamp_s": float(separation["timestamp"]),
        }
        self.root.destroy()

    def close(self) -> None:
        self.result = None
        self.root.destroy()

    def run(self) -> dict[str, object]:
        self.root.mainloop()
        if self.result is None:
            raise RuntimeError("用户取消了接触事件标注")
        return self.result


def read_track(path: Path) -> list[TrackPoint]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            TrackPoint(
                timestamp=float(row["timestamp"]),
                x=float(row["x_pixel"]),
                y=float(row["y_pixel"]),
                confidence=float(row["confidence"]),
            )
            for row in csv.DictReader(handle)
        ]


def extract_event_frames(
    video: Path,
    points: list[TrackPoint],
    floor_points: list[list[float]],
    work: Path,
    *,
    seconds_each_side: float = 0.5,
) -> tuple[list[dict[str, object]], int]:
    floor_1, floor_2 = floor_points
    hint_index = candidate_impact_index(points, tuple(floor_1), tuple(floor_2))
    start, end = event_window_indices(
        points, hint_index, seconds_each_side=seconds_each_side
    )
    selected = points[start:end]
    timestamps_path = work / "event_timestamps.txt"
    timestamps_path.write_text(
        "\n".join(f"{point.timestamp:.9f}" for point in selected) + "\n",
        encoding="utf-8",
    )
    frames_dir = work / "event_frames"
    run_bridge(["frames", str(video), str(timestamps_path), str(frames_dir)])
    frames = [
        {
            "analysis_frame_index": start + offset,
            "timestamp": point.timestamp,
            "image_path": frames_dir / f"frame_{offset:04d}.jpg",
        }
        for offset, point in enumerate(selected)
    ]
    return frames, hint_index - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="通过一次足球框选和两点地板标注计算恢复系数"
    )
    parser.add_argument("video", type=Path, help="原始 MOV/MP4 视频")
    parser.add_argument(
        "--start-time", type=float, default=14.0, help="标注和跟踪开始时间（秒）"
    )
    parser.add_argument(
        "--end-time", type=float, default=16.5, help="跟踪结束时间（秒）"
    )
    parser.add_argument(
        "--save-annotation", type=Path, help="可选：保存人工标注 JSON"
    )
    parser.add_argument(
        "--annotation", type=Path, help="可选：直接复用已有标注 JSON"
    )
    return parser.parse_args()


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
        with tempfile.TemporaryDirectory(prefix="bounce_restitution_") as directory:
            work = Path(directory)
            frame_path = work / "annotation_frame.png"
            track_path = work / "track.csv"

            if args.annotation:
                annotation = json.loads(args.annotation.read_text(encoding="utf-8"))
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

            if args.save_annotation:
                annotation["tracking_start_time_s"] = args.start_time
                annotation["tracking_end_time_s"] = args.end_time
                args.save_annotation.parent.mkdir(parents=True, exist_ok=True)
                args.save_annotation.write_text(
                    json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

            restitution, _, _ = estimate_restitution_from_events(
                points,
                tuple(floor_1),
                tuple(floor_2),
                contact_time,
                separation_time,
            )
            print(f"恢复系数 e = {restitution:.3f}")
            return 0
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        print(f"错误：视频处理失败\n{detail}", file=sys.stderr)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
