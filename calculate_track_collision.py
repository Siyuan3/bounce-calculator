#!/usr/bin/env python3
"""Calculate restitution for one ball colliding with a fixed track end."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
import json
from math import isfinite
from pathlib import Path
import subprocess
import sys
import tempfile
import tkinter as tk
from tkinter import messagebox
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw, ImageTk

from axis_motion_tracker import (
    BallPositionMeasurement,
    detect_ball_positions_dual_from_files,
    locate_ball_template_box_from_files,
)
from calculate_restitution import EventSelectionWindow, run_bridge
from height_sweep import RELEASE_LEVELS
from restitution_math import (
    TrackPoint,
    estimate_fixed_wall_restitution_multiwindow,
)

MAX_DETECTOR_DISAGREEMENT_DIAMETER_RATIO = 0.15


def build_video_preview_times(
    presentation_timestamps: list[float],
    *,
    maximum_frames: int = 81,
) -> list[float]:
    """Sample real frame timestamps across the complete video."""
    if not presentation_timestamps:
        raise ValueError("视频中没有可用的真实帧时间戳")
    if maximum_frames < 2:
        raise ValueError("整段视频预览至少需要 2 帧")
    timestamps = sorted(set(presentation_timestamps))
    if len(timestamps) <= maximum_frames:
        return timestamps
    indices = [
        round(index * (len(timestamps) - 1) / (maximum_frames - 1))
        for index in range(maximum_frames)
    ]
    return [timestamps[index] for index in indices]


def nearest_timestamp_index(timestamps: list[float], target: float) -> int:
    """Return the index whose timestamp is closest to ``target``."""
    if not timestamps:
        raise ValueError("时间戳列表不能为空")
    position = bisect_left(timestamps, target)
    if position <= 0:
        return 0
    if position >= len(timestamps):
        return len(timestamps) - 1
    before = timestamps[position - 1]
    after = timestamps[position]
    return position - 1 if target - before < after - target else position


def select_event_browse_timestamps(
    presentation_timestamps: list[float],
    *,
    reference_time: float,
    seconds_each_side: float = 0.75,
) -> list[float]:
    """Select a short visual browsing interval around an approximate collision."""
    if seconds_each_side <= 0:
        raise ValueError("碰撞浏览窗口必须大于 0 秒")
    selected = [
        timestamp
        for timestamp in presentation_timestamps
        if reference_time - seconds_each_side
        <= timestamp
        <= reference_time + seconds_each_side
    ]
    if len(selected) < 3:
        raise ValueError("碰撞参考时刻附近的视频帧不足")
    return selected


def select_local_tracking_timestamps(
    presentation_timestamps: list[float],
    *,
    contact_time: float,
    separation_time: float,
    fit_frames: int,
    margin_frames: int = 4,
    candidate_multiplier: int = 1,
) -> tuple[list[float], list[float]]:
    """Return only the local free-motion frames used around a collision."""
    if fit_frames < 3:
        raise ValueError("fit_frames 至少为 3")
    if margin_frames < 0:
        raise ValueError("局部跟踪冗余帧数不能为负数")
    if candidate_multiplier < 1:
        raise ValueError("候选帧倍数至少为 1")
    if separation_time <= contact_time:
        raise ValueError("完全分离时间必须晚于首次接触时间")
    required = fit_frames + margin_frames
    pre_candidates = [
        timestamp
        for timestamp in presentation_timestamps
        if timestamp < contact_time
    ]
    post_candidates = [
        timestamp
        for timestamp in presentation_timestamps
        if timestamp >= separation_time
    ]
    if len(pre_candidates) < required:
        raise ValueError(
            f"碰撞前帧不足：需要 {required} 帧，实际 {len(pre_candidates)} 帧"
        )
    if len(post_candidates) < required:
        raise ValueError(
            f"碰撞后帧不足：需要 {required} 帧，实际 {len(post_candidates)} 帧"
        )
    candidate_count = required * candidate_multiplier
    pre = pre_candidates[-candidate_count:]
    post = post_candidates[:candidate_count]
    return pre, post


def probe_video_duration(video: Path) -> float:
    """Read the presentation duration reported by AVFoundation."""
    result = run_bridge(["info", str(video)])
    metadata = json.loads(result.stdout)
    duration = float(metadata["duration_s"])
    if duration <= 0:
        raise ValueError("无法读取有效的视频时长")
    return duration


def read_video_timestamps(
    video: Path,
    duration: float,
    work: Path,
) -> list[float]:
    """Read all real presentation timestamps, excluding the asset end boundary."""
    all_timestamps_path = work / "whole_video_timestamps.txt"
    run_bridge(
        ["timestamps", str(video), "0.0", f"{duration:.9f}", str(all_timestamps_path)]
    )
    timestamps = [
        float(value)
        for value in all_timestamps_path.read_text(encoding="utf-8").splitlines()
        if value.strip() and float(value) < duration - 1e-6
    ]
    if not timestamps:
        raise ValueError("视频中没有可读取的真实帧时间戳")
    return timestamps


def select_empty_background_timestamps(
    presentation_timestamps: list[float],
    *,
    background_seconds: float = 1.0,
    maximum_frames: int = 21,
) -> list[float]:
    """Sample real timestamps from the initial empty-track interval."""
    if background_seconds <= 0:
        raise ValueError("空背景时长必须大于 0 秒")
    if maximum_frames < 3:
        raise ValueError("空背景至少需要抽取 3 帧")
    start = presentation_timestamps[0]
    candidates = [
        timestamp
        for timestamp in presentation_timestamps
        if timestamp <= start + background_seconds
    ]
    if len(candidates) < 3:
        raise ValueError("视频开头没有足够的空轨道帧")
    return build_video_preview_times(candidates, maximum_frames=maximum_frames)


def create_median_background(frame_paths: list[Path], output_path: Path) -> Path:
    """Combine empty-track frames so sensor noise does not become foreground."""
    if len(frame_paths) < 3:
        raise ValueError("生成空背景至少需要 3 帧")
    frames = np.stack(
        [np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for path in frame_paths]
    )
    background = np.median(frames, axis=0).astype(np.uint8)
    Image.fromarray(background, mode="L").save(output_path)
    return output_path


def extract_frames_at_timestamps(
    video: Path,
    timestamps: list[float],
    work: Path,
    *,
    label: str,
) -> tuple[list[Path], list[float]]:
    """Extract selected frames and return paths with actual presentation times."""
    if not timestamps:
        raise ValueError("抽帧时间戳不能为空")
    timestamps_path = work / f"{label}_timestamps.txt"
    timestamps_path.write_text(
        "\n".join(f"{timestamp:.9f}" for timestamp in timestamps) + "\n",
        encoding="utf-8",
    )
    frames_dir = work / f"{label}_frames"
    run_bridge(["frames", str(video), str(timestamps_path), str(frames_dir)])
    manifest_rows = list(
        csv.DictReader((frames_dir / "manifest.csv").open(encoding="utf-8"))
    )
    unique_rows = deduplicate_extracted_frame_manifest(
        manifest_rows, requested_count=len(timestamps)
    )
    actual_timestamps = validate_extracted_frame_manifest(
        unique_rows,
        requested_count=len(unique_rows),
    )
    paths = [frames_dir / row["filename"] for row in unique_rows]
    return paths, actual_timestamps


def deduplicate_extracted_frame_manifest(
    manifest_rows: list[dict[str, str]],
    *,
    requested_count: int,
) -> list[dict[str, str]]:
    """Keep one decoded image per actual PTS while preserving time order."""
    if len(manifest_rows) != requested_count:
        raise ValueError(
            f"抽帧数量不一致：请求 {requested_count} 帧，实际 {len(manifest_rows)} 帧"
        )
    unique_rows: list[dict[str, str]] = []
    previous: float | None = None
    for row in manifest_rows:
        current = float(row["actual_timestamp"])
        if not isfinite(current):
            raise ValueError("抽帧结果包含无效的实际 PTS")
        if previous is not None and current < previous:
            raise ValueError("抽帧结果的实际 PTS 没有按时间递增")
        if previous is None or current != previous:
            unique_rows.append(row)
        previous = current
    return unique_rows


def validate_extracted_frame_manifest(
    manifest_rows: list[dict[str, str]],
    *,
    requested_count: int,
) -> list[float]:
    """Require one distinct, strictly increasing actual PTS per request."""
    if len(manifest_rows) != requested_count:
        raise ValueError(
            f"抽帧数量不一致：请求 {requested_count} 帧，实际 {len(manifest_rows)} 帧"
        )
    actual_timestamps = [float(row["actual_timestamp"]) for row in manifest_rows]
    if any(not isfinite(timestamp) for timestamp in actual_timestamps):
        raise ValueError("抽帧结果包含无效的实际 PTS")
    for previous, current in zip(actual_timestamps, actual_timestamps[1:]):
        if current == previous:
            raise ValueError(f"抽帧结果含有重复的实际 PTS：{current:.9f} 秒")
        if current < previous:
            raise ValueError("抽帧结果的实际 PTS 没有严格递增")
    return actual_timestamps


def extract_video_preview_frames(
    video: Path,
    presentation_timestamps: list[float],
    work: Path,
) -> list[dict[str, object]]:
    """Extract a bounded overview spanning the complete video."""
    requested_times = build_video_preview_times(presentation_timestamps)
    paths, actual_timestamps = extract_frames_at_timestamps(
        video, requested_times, work, label="whole_video_preview"
    )
    return [
        {
            "timestamp": timestamp,
            "image_path": path,
        }
        for path, timestamp in zip(paths, actual_timestamps)
    ]


def extract_exact_frame(
    video: Path,
    requested_time: float,
    output_path: Path,
) -> tuple[Path, float]:
    """Extract one requested frame and return its actual presentation time."""
    result = run_bridge(
        ["frame", str(video), f"{requested_time:.9f}", str(output_path)]
    )
    try:
        actual_timestamp = float(result.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as error:
        raise ValueError("视频工具没有返回标注帧的实际时间戳") from error
    return output_path, actual_timestamp


def video_identity(video: Path) -> dict[str, object]:
    """Return a lightweight identity that changes when a video is replaced."""
    resolved = video.expanduser().resolve()
    stat = resolved.stat()
    return {
        "resolved_path": str(resolved),
        "file_size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
    }


def validate_annotation_video(annotation: dict[str, object], video: Path) -> None:
    """Reject annotations known to belong to another or replaced video."""
    current = video_identity(video)
    stored_identity = annotation.get("video_identity")
    if stored_identity is not None:
        if stored_identity != current:
            raise ValueError("标注文件对应的视频已被替换或不是当前视频，请重新标注")
        return
    raise ValueError("旧标注缺少视频身份信息，无法确认视频未被替换，请重新标注")


class WholeVideoTimeWindow:
    """Choose one reference time while browsing previews of the full video."""

    def __init__(
        self,
        frames: list[dict[str, object]],
        duration: float,
        exact_frame_loader: Callable[[float], tuple[Path, float]],
    ) -> None:
        if not frames:
            raise ValueError("没有可用于整段视频浏览的预览帧")
        self.frames = frames
        self.timestamps = [float(frame["timestamp"]) for frame in frames]
        self.duration = duration
        self.selectable_start = self.timestamps[0]
        self.selectable_end = self.timestamps[-1]
        self.exact_frame_loader = exact_frame_loader
        self.position = 0
        self.displayed_preview_position: int | None = None
        self.result: float | None = None
        self.photo: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title("小球撞固定端：从整段视频选择参考时间")
        tk.Label(
            self.root,
            text="浏览整段视频，选择大致发生碰撞的时刻",
            font=("Arial", 15),
        ).pack(pady=(10, 4))
        tk.Label(
            self.root,
            text="下一步将精确确认接触与完全分离；不会追踪整段运动",
            font=("Arial", 12),
        ).pack(pady=(0, 6))

        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10)
        self.status = tk.StringVar()
        tk.Label(self.root, textvariable=self.status, font=("Menlo", 12)).pack(
            pady=(6, 2)
        )

        self.time_scale = tk.Scale(
            self.root,
            from_=self.selectable_start,
            to=self.selectable_end,
            resolution=0.001,
            orient="horizontal",
            length=720,
            showvalue=False,
            command=self.on_scale,
        )
        self.time_scale.pack(fill="x", padx=12)

        time_controls = tk.Frame(self.root)
        time_controls.pack(fill="x", padx=10, pady=6)
        for label, offset in (("−1 秒", -1.0), ("−0.1 秒", -0.1)):
            tk.Button(
                time_controls,
                text=label,
                command=lambda value=offset: self.move_seconds(value),
            ).pack(side="left", padx=3)
        for label, offset in (("+0.1 秒", 0.1), ("+1 秒", 1.0)):
            tk.Button(
                time_controls,
                text=label,
                command=lambda value=offset: self.move_seconds(value),
            ).pack(side="right", padx=3)

        exact_controls = tk.Frame(self.root)
        exact_controls.pack(fill="x", padx=10, pady=4)
        tk.Label(exact_controls, text="跳转到秒数：").pack(side="left")
        self.time_entry = tk.StringVar(value="0.000")
        entry = tk.Entry(exact_controls, textvariable=self.time_entry, width=12)
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda _: self.jump_to_entry())
        tk.Button(
            exact_controls,
            text="精确预览该秒",
            command=self.jump_to_entry,
        ).pack(side="left")
        tk.Button(
            exact_controls,
            text="使用当前秒数",
            command=self.confirm,
            bg="#8ecae6",
        ).pack(side="right")

        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_frame(self.selectable_start)

    def on_scale(self, raw_time: str) -> None:
        self.update_frame(float(raw_time))

    def update_frame(self, selected_time: float) -> None:
        selected_time = min(
            max(self.selectable_start, selected_time), self.selectable_end
        )
        self.position = nearest_timestamp_index(self.timestamps, selected_time)
        frame = self.frames[self.position]
        if self.position != self.displayed_preview_position:
            self.display_image(Path(frame["image_path"]))
            self.displayed_preview_position = self.position
        self.time_entry.set(f"{selected_time:.3f}")
        self.status.set(
            f"选择时间 {selected_time:.3f} s｜"
            f"概览缩略图 {float(frame['timestamp']):.3f} s｜"
            f"视频总时长 {self.duration:.3f} s"
        )

    def display_image(self, image_path: Path) -> None:
        source = Image.open(image_path).convert("RGB")
        scale = min(720 / source.width, 700 / source.height, 1)
        display = source.resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
        self.photo = ImageTk.PhotoImage(display)
        self.image_label.configure(image=self.photo)

    def set_time(self, value: float) -> None:
        value = min(max(self.selectable_start, value), self.selectable_end)
        self.time_scale.set(value)
        self.update_frame(value)

    def move_seconds(self, offset: float) -> None:
        self.set_time(float(self.time_scale.get()) + offset)

    def jump_to_entry(self) -> None:
        try:
            value = float(self.time_entry.get())
        except ValueError:
            messagebox.showwarning("时间格式错误", "请输入以秒为单位的数字")
            return
        if not self.selectable_start <= value <= self.selectable_end:
            messagebox.showwarning(
                "时间超出范围",
                f"请输入 {self.selectable_start:.6f} 到 "
                f"{self.selectable_end:.6f} 秒之间的时间",
            )
            return
        self.set_time(value)
        self.root.configure(cursor="watch")
        self.status.set(f"正在提取 {value:.3f} 秒对应的精确视频帧……")
        self.root.update_idletasks()
        try:
            image_path, actual_timestamp = self.exact_frame_loader(value)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
            messagebox.showerror("精确预览失败", str(error))
            return
        finally:
            self.root.configure(cursor="")
        self.display_image(image_path)
        self.displayed_preview_position = None
        self.time_entry.set(f"{value:.9f}")
        self.status.set(
            f"请求时间 {value:.3f} s｜实际视频帧 {actual_timestamp:.6f} s｜"
            f"视频总时长 {self.duration:.3f} s"
        )

    def confirm(self) -> None:
        try:
            value = float(self.time_entry.get())
        except ValueError:
            messagebox.showwarning("时间格式错误", "请输入以秒为单位的数字")
            return
        if not self.selectable_start <= value <= self.selectable_end:
            messagebox.showwarning(
                "时间超出范围",
                f"请输入 {self.selectable_start:.6f} 到 "
                f"{self.selectable_end:.6f} 秒之间的时间",
            )
            return
        self.result = value
        self.root.destroy()

    def close(self) -> None:
        self.result = None
        self.root.destroy()

    def run(self) -> float:
        self.root.mainloop()
        if self.result is None:
            raise RuntimeError("用户取消了整段视频时间选择")
        return self.result


class TrackAnnotationWindow:
    """Select the collision ROI, one clear ball template, and the wall line."""

    def __init__(
        self,
        image_path: Path,
        *,
        stages: tuple[str, ...] = ("roi", "ball", "wall"),
        title: str = "小球撞固定端：批次画面标定",
    ) -> None:
        if not stages or any(stage not in {"roi", "ball", "wall"} for stage in stages):
            raise ValueError("画面标注阶段无效")
        self.root = tk.Tk()
        self.root.title(title)
        self.source = Image.open(image_path).convert("RGB")
        self.scale = min(760 / self.source.width, 820 / self.source.height, 1)
        size = (
            round(self.source.width * self.scale),
            round(self.source.height * self.scale),
        )
        self.display = self.source.resize(size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(self.display)
        self.stages = stages
        self.stage_index = 0
        self.instructions = tk.StringVar(value=self.stage_instruction())
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

        self.stage = self.stages[0]
        self.drag_start: tuple[float, float] | None = None
        self.collision_roi: tuple[float, float, float, float] | None = None
        self.ball_box: tuple[float, float, float, float] | None = None
        self.wall_points: list[tuple[float, float]] = []
        self.result: dict[str, object] | None = None
        self.canvas.bind("<Button-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def reset(self) -> None:
        self.canvas.delete("annotation")
        self.canvas.delete("preview")
        self.stage_index = 0
        self.stage = self.stages[0]
        self.drag_start = None
        self.collision_roi = None
        self.ball_box = None
        self.wall_points = []
        self.confirm_button.configure(state="disabled")
        self.instructions.set(self.stage_instruction())

    def stage_instruction(self) -> str:
        stage = self.stages[self.stage_index]
        prefix = f"第 {self.stage_index + 1} 步："
        if stage == "roi":
            return prefix + "框选包含小球运动和挡墙的碰撞区域"
        if stage == "ball":
            return prefix + "在清晰帧中紧贴轮廓框选当前小球"
        return prefix + "沿挡墙可见边缘点击两个相距较远的点"

    def advance_stage(self) -> None:
        self.stage_index += 1
        if self.stage_index < len(self.stages):
            self.stage = self.stages[self.stage_index]
            self.instructions.set(self.stage_instruction())
            return
        self.stage = "complete"
        self.confirm_button.configure(state="normal")
        self.instructions.set("检查本次标注，确认后保存")

    def mouse_down(self, event: tk.Event) -> None:
        if self.stage == "complete":
            return
        if self.stage in {"roi", "ball"}:
            self.drag_start = (event.x, event.y)
        else:
            self.add_wall_point(event.x, event.y)

    def mouse_drag(self, event: tk.Event) -> None:
        if self.stage not in {"roi", "ball"} or self.drag_start is None:
            return
        self.canvas.delete("preview")
        self.canvas.create_rectangle(
            *self.drag_start,
            event.x,
            event.y,
            outline="#00e5ff" if self.stage == "ball" else "#80ed99",
            width=3,
            tags="preview",
        )

    def mouse_up(self, event: tk.Event) -> None:
        if self.stage not in {"roi", "ball"} or self.drag_start is None:
            return
        x1, y1 = self.drag_start
        x2, y2 = event.x, event.y
        x1, x2 = sorted((max(0, x1), min(self.display.width, x2)))
        y1, y2 = sorted((max(0, y1), min(self.display.height, y2)))
        self.canvas.delete("preview")
        minimum = 30 if self.stage == "roi" else 8
        if x2 - x1 < minimum or y2 - y1 < minimum:
            messagebox.showwarning("框选过小", "请完整框住当前要求的区域")
            self.drag_start = None
            return
        box = (x1, y1, x2 - x1, y2 - y1)
        if self.stage == "roi":
            self.collision_roi = box
            colour = "#80ed99"
        else:
            self.ball_box = box
            colour = "#00e5ff"
        self.canvas.create_rectangle(
            x1, y1, x2, y2, outline=colour, width=3, tags="annotation"
        )
        self.drag_start = None
        self.advance_stage()

    def add_wall_point(self, x: float, y: float) -> None:
        if len(self.wall_points) >= 2:
            return
        self.wall_points.append((x, y))
        radius = 5
        self.canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius,
            fill="#ffd166",
            outline="black",
            tags="annotation",
        )
        if len(self.wall_points) == 2:
            self.canvas.create_line(
                *self.wall_points[0],
                *self.wall_points[1],
                fill="#ffd166",
                width=3,
                tags="annotation",
            )
            self.advance_stage()

    def confirm(self) -> None:
        factor = 1 / self.scale
        result: dict[str, object] = {
            "image_size": [self.source.width, self.source.height],
        }
        if "roi" in self.stages:
            assert self.collision_roi is not None
            result["collision_roi"] = [
                value * factor for value in self.collision_roi
            ]
        if "ball" in self.stages:
            assert self.ball_box is not None
            result["ball_box"] = [value * factor for value in self.ball_box]
        if "wall" in self.stages:
            result["wall_line_points"] = [
                [x * factor, y * factor] for x, y in self.wall_points
            ]
        self.result = result
        self.root.destroy()

    def close(self) -> None:
        self.result = None
        self.root.destroy()

    def run(self) -> dict[str, object]:
        self.root.mainloop()
        if self.result is None:
            raise RuntimeError("用户取消了碰撞区域、小球或挡墙标注")
        return self.result


class GeometryAnnotationWindow(TrackAnnotationWindow):
    """Label camera/track geometry that is shared by multiple balls."""

    def __init__(self, image_path: Path) -> None:
        super().__init__(
            image_path,
            stages=("roi", "wall"),
            title="共享几何标定：碰撞区域与挡墙",
        )


class BallTemplateAnnotationWindow(TrackAnnotationWindow):
    """Label only the current ball without changing shared geometry."""

    def __init__(self, image_path: Path) -> None:
        super().__init__(
            image_path,
            stages=("ball",),
            title="当前小球标定：直径与外观模板",
        )


def build_event_frames(
    timestamps: list[float],
    frame_paths: list[Path],
) -> list[dict[str, object]]:
    if len(timestamps) != len(frame_paths):
        raise ValueError("时间戳与视频帧数量不一致")
    if len(timestamps) < 2:
        raise ValueError("碰撞事件浏览至少需要 2 个不同时间的视频帧")
    return [
        {
            "analysis_frame_index": index,
            "timestamp": timestamp,
            "image_path": frame_path,
        }
        for index, (timestamp, frame_path) in enumerate(
            zip(timestamps, frame_paths)
        )
    ]


class FixedEndEventWindow(EventSelectionWindow):
    def __init__(self, frames: list[dict[str, object]], initial_position: int) -> None:
        super().__init__(frames, initial_position)
        self.root.title("小球撞固定端：选择接触与完全分离帧")
        self.instructions.set(
            "可按秒跳转或逐帧微调：选择首次接触，再选择反向后完全分离"
        )
        self.selection_status.set("首次接触：未选择｜完全分离：未选择")

        seconds_controls = tk.Frame(self.root)
        seconds_controls.pack(fill="x", padx=10, pady=(2, 8))
        tk.Label(seconds_controls, text="按秒跳转：").pack(side="left")
        current_time = float(self.frames[self.position]["timestamp"])
        self.event_time_entry = tk.StringVar(value=f"{current_time:.6f}")
        entry = tk.Entry(
            seconds_controls, textvariable=self.event_time_entry, width=14
        )
        entry.pack(side="left", padx=4)
        entry.bind("<Return>", lambda _: self.jump_to_seconds())
        tk.Button(
            seconds_controls, text="跳转到最近实际帧", command=self.jump_to_seconds
        ).pack(side="left")

        first_time = float(self.frames[0]["timestamp"])
        last_time = float(self.frames[-1]["timestamp"])
        self.event_time_scale = tk.Scale(
            self.root,
            from_=first_time,
            to=last_time,
            resolution=0.001,
            orient="horizontal",
            length=720,
            showvalue=False,
            command=self.on_event_time_scale,
        )
        self.event_time_scale.pack(fill="x", padx=12, pady=(0, 8))
        self.event_time_scale.set(current_time)

    def sync_seconds_controls(self) -> None:
        if not hasattr(self, "event_time_entry"):
            return
        timestamp = float(self.frames[self.position]["timestamp"])
        self.event_time_entry.set(f"{timestamp:.6f}")
        self.event_time_scale.set(timestamp)

    def move(self, offset: int) -> None:
        super().move(offset)
        self.sync_seconds_controls()

    def on_event_time_scale(self, raw_time: str) -> None:
        timestamps = [float(frame["timestamp"]) for frame in self.frames]
        self.position = nearest_timestamp_index(timestamps, float(raw_time))
        self.update_frame()
        if hasattr(self, "event_time_entry"):
            actual_time = float(self.frames[self.position]["timestamp"])
            self.event_time_entry.set(f"{actual_time:.6f}")

    def jump_to_seconds(self) -> None:
        try:
            target = float(self.event_time_entry.get())
        except ValueError:
            messagebox.showwarning("时间格式错误", "请输入以秒为单位的数字")
            return
        first_time = float(self.frames[0]["timestamp"])
        last_time = float(self.frames[-1]["timestamp"])
        if not first_time <= target <= last_time:
            messagebox.showwarning(
                "时间超出候选范围",
                f"请输入 {first_time:.6f} 到 {last_time:.6f} 秒之间的时间",
            )
            return
        timestamps = [float(frame["timestamp"]) for frame in self.frames]
        self.position = nearest_timestamp_index(timestamps, target)
        self.update_frame()
        self.sync_seconds_controls()


class ReleaseLevelWindow:
    """Choose the ordinal H1–H5 release mark for one trial."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("选择释放位置")
        tk.Label(
            self.root,
            text="请选择本 trial 使用的释放位置",
            font=("Arial", 15),
        ).pack(padx=24, pady=(18, 10))
        self.value = tk.StringVar(value="H1")
        choices = tk.Frame(self.root)
        choices.pack(padx=20, pady=6)
        for level in RELEASE_LEVELS:
            tk.Radiobutton(
                choices,
                text=level,
                variable=self.value,
                value=level,
                font=("Arial", 13),
            ).pack(side="left", padx=7)
        self.result: str | None = None
        tk.Button(self.root, text="确定", command=self.confirm).pack(pady=(8, 18))
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def confirm(self) -> None:
        self.result = self.value.get()
        self.root.destroy()

    def close(self) -> None:
        self.result = None
        self.root.destroy()

    def run(self) -> str:
        self.root.mainloop()
        if self.result is None:
            raise RuntimeError("用户取消了释放位置选择")
        return self.result


class TemplateFrameSelectionWindow:
    """Choose one sharp pre-contact frame for the reusable batch template."""

    def __init__(self, frame_paths: list[Path], timestamps: list[float]) -> None:
        if not frame_paths or len(frame_paths) != len(timestamps):
            raise ValueError("批次模板候选帧与时间戳数量不一致")
        self.frame_paths = frame_paths
        self.timestamps = timestamps
        self.position = len(frame_paths) - 1
        self.result: int | None = None
        self.photo: ImageTk.PhotoImage | None = None
        self.root = tk.Tk()
        self.root.title("选择批次小球模板帧")
        tk.Label(
            self.root,
            text="选择轮廓清晰、没有运动模糊且未接触挡墙的小球画面",
            font=("Arial", 14),
        ).pack(pady=(10, 6))
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10)
        self.status = tk.StringVar()
        tk.Label(self.root, textvariable=self.status, font=("Menlo", 11)).pack(
            pady=5
        )
        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=10, pady=(2, 10))
        tk.Button(controls, text="上一帧", command=lambda: self.move(-1)).pack(
            side="left", padx=3
        )
        tk.Button(controls, text="下一帧", command=lambda: self.move(1)).pack(
            side="left", padx=3
        )
        tk.Button(
            controls,
            text="使用此清晰帧",
            command=self.confirm,
            bg="#8ecae6",
        ).pack(side="right", padx=3)
        self.root.bind("<Left>", lambda _: self.move(-1))
        self.root.bind("<Right>", lambda _: self.move(1))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_frame()

    def update_frame(self) -> None:
        with Image.open(self.frame_paths[self.position]) as source_image:
            source = source_image.convert("RGB")
        scale = min(760 / source.width, 740 / source.height, 1)
        display = source.resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
        self.photo = ImageTk.PhotoImage(display)
        self.image_label.configure(image=self.photo)
        self.status.set(
            f"{self.position + 1}/{len(self.frame_paths)}｜"
            f"PTS {self.timestamps[self.position]:.6f} s"
        )

    def move(self, offset: int) -> None:
        self.position = min(max(0, self.position + offset), len(self.frame_paths) - 1)
        self.update_frame()

    def confirm(self) -> None:
        self.result = self.position
        self.root.destroy()

    def close(self) -> None:
        self.result = None
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        if self.result is None:
            raise RuntimeError("用户取消了批次小球模板帧选择")
        return self.result


class DetectionReviewWindow:
    """Review dual-detector overlays and exclude bad free-motion frames."""

    def __init__(
        self,
        frame_paths: list[Path],
        measurements: list[BallPositionMeasurement],
        *,
        pre_collision_count: int,
        minimum_frames_each_branch: int,
        initially_excluded: list[float] | None = None,
    ) -> None:
        if len(frame_paths) != len(measurements):
            raise ValueError("检测复核画面与测量数量不一致")
        self.frame_paths = frame_paths
        self.measurements = measurements
        self.pre_collision_count = pre_collision_count
        self.minimum_frames_each_branch = minimum_frames_each_branch
        self.automatic_invalid = {
            measurement.timestamp
            for measurement in measurements
            if not measurement.contour_detected
        }
        self.excluded = set(float(value) for value in (initially_excluded or []))
        self.excluded.update(self.automatic_invalid)
        self.position = 0
        self.result: list[float] | None = None
        self.photo: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title("检查小球轮廓与模板定位")
        tk.Label(
            self.root,
            text="绿色圆点：轮廓位置｜黄色十字：模板位置｜错误帧可排除",
            font=("Arial", 13),
        ).pack(pady=(10, 5))
        self.image_label = tk.Label(self.root)
        self.image_label.pack(padx=10)
        self.status = tk.StringVar()
        tk.Label(self.root, textvariable=self.status, font=("Menlo", 11)).pack(
            pady=5
        )
        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=10, pady=(2, 10))
        tk.Button(controls, text="上一帧", command=lambda: self.move(-1)).pack(
            side="left", padx=3
        )
        tk.Button(controls, text="下一帧", command=lambda: self.move(1)).pack(
            side="left", padx=3
        )
        self.toggle_button = tk.Button(
            controls, text="排除此帧", command=self.toggle_excluded
        )
        self.toggle_button.pack(side="left", padx=12)
        tk.Button(controls, text="确认全部结果", command=self.confirm).pack(
            side="right", padx=3
        )
        self.root.bind("<Left>", lambda _: self.move(-1))
        self.root.bind("<Right>", lambda _: self.move(1))
        self.root.bind("<space>", lambda _: self.toggle_excluded())
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.update_frame()

    def update_frame(self) -> None:
        source = Image.open(self.frame_paths[self.position]).convert("RGB")
        measurement = self.measurements[self.position]
        draw = ImageDraw.Draw(source)
        radius = max(5, round(min(source.width, source.height) * 0.006))
        if measurement.contour_detected:
            draw.ellipse(
                (
                    measurement.x - radius,
                    measurement.y - radius,
                    measurement.x + radius,
                    measurement.y + radius,
                ),
                outline=(0, 230, 118),
                width=max(2, radius // 3),
            )
        draw.line(
            (
                measurement.template_x - radius,
                measurement.template_y,
                measurement.template_x + radius,
                measurement.template_y,
            ),
            fill=(255, 193, 7),
            width=max(2, radius // 3),
        )
        draw.line(
            (
                measurement.template_x,
                measurement.template_y - radius,
                measurement.template_x,
                measurement.template_y + radius,
            ),
            fill=(255, 193, 7),
            width=max(2, radius // 3),
        )
        scale = min(760 / source.width, 740 / source.height, 1)
        display = source.resize(
            (round(source.width * scale), round(source.height * scale)),
            Image.Resampling.LANCZOS,
        )
        self.photo = ImageTk.PhotoImage(display)
        self.image_label.configure(image=self.photo)
        excluded = measurement.timestamp in self.excluded
        automatic_invalid = measurement.timestamp in self.automatic_invalid
        contour_mode_text = {
            "dense_silhouette": "致密轮廓",
            "sparse_foreground": "稀疏前景轮廓",
            "template_only_invalid": "模板占位（无物理测量）",
        }.get(measurement.contour_mode, measurement.contour_mode)
        phase = (
            "碰撞前"
            if self.position < self.pre_collision_count
            else "完全分离后"
        )
        self.status.set(
            f"{self.position + 1}/{len(self.measurements)}｜{phase}｜"
            f"PTS {measurement.timestamp:.6f} s｜双检测差 {measurement.detector_disagreement_px:.2f} px｜"
            f"置信度 {measurement.confidence:.3f}｜{contour_mode_text}｜"
            f"{'轮廓无效，已自动排除' if automatic_invalid else ('已排除' if excluded else '保留')}"
        )
        self.toggle_button.configure(
            text="轮廓无效" if automatic_invalid else ("恢复此帧" if excluded else "排除此帧"),
            state="disabled" if automatic_invalid else "normal",
        )

    def move(self, offset: int) -> None:
        self.position = min(max(0, self.position + offset), len(self.measurements) - 1)
        self.update_frame()

    def toggle_excluded(self) -> None:
        timestamp = self.measurements[self.position].timestamp
        if timestamp in self.automatic_invalid:
            return
        if timestamp in self.excluded:
            self.excluded.remove(timestamp)
        else:
            self.excluded.add(timestamp)
        self.update_frame()

    def confirm(self) -> None:
        pre_valid = sum(
            measurement.timestamp not in self.excluded
            and measurement.confidence >= 0.35
            for measurement in self.measurements[: self.pre_collision_count]
        )
        post_valid = sum(
            measurement.timestamp not in self.excluded
            and measurement.confidence >= 0.35
            for measurement in self.measurements[self.pre_collision_count :]
        )
        if min(pre_valid, post_valid) < self.minimum_frames_each_branch:
            messagebox.showwarning(
                "有效帧不足",
                f"碰撞前后各至少保留 {self.minimum_frames_each_branch} 帧",
            )
            return
        self.result = sorted(self.excluded)
        self.root.destroy()

    def close(self) -> None:
        self.result = None
        self.root.destroy()

    def run(self) -> list[float]:
        self.root.mainloop()
        if self.result is None:
            raise RuntimeError("用户取消了小球检测结果复核")
        return self.result


class ResultConfirmationWindow:
    """Show the computed physical result before marking the trial complete."""

    def __init__(
        self,
        *,
        release_level: str,
        restitution: float,
        uncertainty: float,
        quality_status: str,
        geometry_status: str,
        incidence_angle_degrees: float,
    ) -> None:
        self.result = False
        self.root = tk.Tk()
        self.root.title("确认恢复系数结果")
        lines = (
            f"释放位置：{release_level}\n\n"
            f"恢复系数 e = {restitution:.4f} ± {uncertainty:.4f}\n"
            f"质量状态：{quality_status}\n"
            f"碰撞几何：{geometry_status}\n"
            f"入射角：{incidence_angle_degrees:.2f}°"
        )
        tk.Label(
            self.root,
            text=lines,
            justify="left",
            font=("Arial", 14),
        ).pack(padx=28, pady=(22, 14))
        tk.Label(
            self.root,
            text="确认后才会把本 trial 标记为 complete。",
            font=("Arial", 11),
        ).pack(padx=24, pady=(0, 12))
        controls = tk.Frame(self.root)
        controls.pack(fill="x", padx=20, pady=(0, 18))
        tk.Button(controls, text="取消，不保存结果", command=self.close).pack(
            side="left"
        )
        tk.Button(
            controls,
            text="确认并保存",
            command=self.confirm,
            bg="#8ecae6",
        ).pack(side="right")
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def confirm(self) -> None:
        self.result = True
        self.root.destroy()

    def close(self) -> None:
        self.result = False
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
        if not self.result:
            raise RuntimeError("用户未确认恢复系数结果")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算俯拍水平轨道小球撞固定端的恢复系数")
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--reference-time",
        type=float,
        help="大致发生碰撞的秒数；省略时打开整段视频时间轴",
    )
    parser.add_argument(
        "--event-window-seconds",
        type=float,
        default=0.75,
        help="参考时刻前后用于人工确认碰撞事件的秒数（默认：0.75）",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        help="兼容旧流程：手动指定分析起点，必须与 --end-time 同时使用",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        help="兼容旧流程：手动指定分析终点，必须与 --start-time 同时使用",
    )
    parser.add_argument(
        "--fit-windows",
        type=int,
        nargs="+",
        default=[5, 7, 9],
        help="碰撞前后多窗口拟合帧数（默认：5 7 9）",
    )
    parser.add_argument(
        "--local-margin-frames",
        type=int,
        default=4,
        help="每侧额外提取的局部跟踪冗余帧数（默认：4）",
    )
    parser.add_argument(
        "--background-seconds",
        type=float,
        default=1.0,
        help="视频开头用于建立空轨道背景的时长（默认：1.0 秒）",
    )
    parser.add_argument(
        "--release-level",
        choices=RELEASE_LEVELS,
        help="本 trial 的释放位置；省略时弹窗选择",
    )
    parser.add_argument(
        "--review-detections",
        action="store_true",
        help="即使标注文件已有复核结果，也重新打开逐帧检测复核",
    )
    parser.add_argument(
        "--setup-annotation",
        type=Path,
        help="兼容旧流程：同时包含共享几何和一种小球模板的批次文件",
    )
    parser.add_argument(
        "--geometry-annotation",
        type=Path,
        help="共享相机/轨道几何文件；不同小球可共同使用",
    )
    parser.add_argument(
        "--ball-setup",
        type=Path,
        help="当前小球的直径与外观模板文件；同一小球的 H1–H5 共用",
    )
    parser.add_argument(
        "--geometry-id",
        help="首次创建共享几何时使用的编号；默认采用几何文件名",
    )
    parser.add_argument(
        "--ball-id",
        help="首次创建小球设置时使用的编号；默认采用小球设置文件名",
    )
    parser.add_argument("--annotation", type=Path)
    parser.add_argument("--save-annotation", type=Path)
    return parser.parse_args()


def validate_setup_mode(
    *,
    setup_path: Path | None,
    geometry_path: Path | None,
    ball_setup_path: Path | None,
    annotation: dict[str, object] | None = None,
) -> bool:
    """Reject incomplete, mixed, or silently downgraded setup modes."""
    split_requested = geometry_path is not None or ball_setup_path is not None
    if split_requested and (geometry_path is None or ball_setup_path is None):
        raise ValueError("--geometry-annotation 与 --ball-setup 必须同时提供")
    if setup_path is not None and split_requested:
        raise ValueError("旧 --setup-annotation 不能与拆分后的几何/小球参数同时使用")
    if annotation is not None and not split_requested:
        split_identity_keys = {
            "setup_mode",
            "geometry_id",
            "ball_id",
            "geometry_annotation",
            "ball_setup_annotation",
        }
        if any(annotation.get(key) is not None for key in split_identity_keys):
            raise ValueError(
                "该 trial 使用拆分配置创建；重新运行时必须同时提供 "
                "--geometry-annotation 与 --ball-setup"
            )
    return split_requested


def record_split_setup_provenance(
    annotation: dict[str, object],
    geometry_path: Path,
    ball_setup_path: Path,
) -> None:
    """Mark even an early event-only draft as belonging to split setup mode."""
    annotation["setup_mode"] = "split_geometry_and_ball"
    annotation["geometry_annotation"] = str(geometry_path.resolve())
    annotation["ball_setup_annotation"] = str(ball_setup_path.resolve())


def validate_requested_setup_id(
    requested_id: str | None,
    loaded_id: str,
    field_name: str,
) -> None:
    """Ensure an explicit CLI identifier cannot silently select another setup."""
    if requested_id is None:
        return
    requested_id = requested_id.strip()
    if not requested_id:
        raise ValueError(f"{field_name} 不能为空")
    if requested_id != loaded_id:
        raise ValueError(
            f"命令行 {field_name} 指定为 {requested_id}，配置文件中为 {loaded_id}"
        )


def save_annotation(
    path: Path,
    annotation: dict[str, object],
    video: Path,
    start_time: float | None = None,
    end_time: float | None = None,
) -> None:
    annotation["video"] = str(video)
    annotation["video_identity"] = video_identity(video)
    if start_time is not None and end_time is not None:
        annotation["tracking_start_time_s"] = start_time
        annotation["tracking_end_time_s"] = end_time
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(annotation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_geometry_annotation(
    path: Path,
    image_size: tuple[int, int],
) -> dict[str, object]:
    """Load camera/track geometry, including geometry from a legacy batch setup."""
    source = json.loads(path.read_text(encoding="utf-8"))
    if source.get("geometry_version") == 1:
        geometry_id = str(source.get("geometry_id", "")).strip()
        loaded_from_legacy = False
    elif source.get("setup_version") == 2:
        geometry_id = path.stem
        loaded_from_legacy = True
    else:
        raise ValueError("几何标定文件版本未知")
    required = {"collision_roi", "wall_line_points", "image_size"}
    if not geometry_id or not required.issubset(source):
        raise ValueError("几何标定缺少 geometry_id、碰撞区域、挡墙线或画面尺寸")
    if tuple(int(value) for value in source["image_size"]) != image_size:
        raise ValueError("当前视频尺寸与共享几何标定不一致")
    return {
        "geometry_version": 1,
        "geometry_id": geometry_id,
        "collision_roi": source["collision_roi"],
        "wall_line_points": source["wall_line_points"],
        "image_size": source["image_size"],
        "loaded_from_legacy_combined_setup": loaded_from_legacy,
    }


def save_geometry_annotation(
    path: Path,
    annotation: dict[str, object],
    video: Path,
    *,
    geometry_id: str,
) -> None:
    """Persist geometry that remains valid when the ball is replaced."""
    geometry_id = geometry_id.strip()
    if not geometry_id:
        raise ValueError("geometry_id 不能为空")
    payload = {
        "geometry_version": 1,
        "geometry_id": geometry_id,
        "collision_roi": annotation["collision_roi"],
        "wall_line_points": annotation["wall_line_points"],
        "image_size": annotation["image_size"],
        "source_video_identity": video_identity(video),
        "scope": "fixed_camera_track_and_wall_geometry",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _save_difference_template(
    template_path: Path,
    annotation: dict[str, object],
    *,
    template_frame_path: Path,
    background_path: Path,
) -> Path:
    ball_box = annotation["ball_box"]
    with Image.open(template_frame_path) as source_image:
        template_frame = np.asarray(source_image.convert("L"), dtype=np.uint8)
    with Image.open(background_path) as source_image:
        background = np.asarray(source_image.convert("L"), dtype=np.uint8)
    if template_frame.shape != background.shape:
        raise ValueError("小球模板帧与空背景尺寸不一致")
    bx, by, bw, bh = (int(round(float(value))) for value in ball_box)
    if bw < 4 or bh < 4:
        raise ValueError("小球模板框过小")
    if (
        bx < 0
        or by < 0
        or bx + bw > template_frame.shape[1]
        or by + bh > template_frame.shape[0]
    ):
        raise ValueError("小球模板框超出画面")
    difference = np.abs(
        template_frame.astype(np.int16) - background.astype(np.int16)
    ).astype(np.uint8)
    template = difference[by : by + bh, bx : bx + bw]
    if int(template.max()) < 8:
        raise ValueError("模板框内没有足够明显的小球前景")
    template_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(template).save(template_path)
    return template_path.resolve()


def load_ball_setup(
    path: Path,
    image_size: tuple[int, int],
    *,
    expected_geometry_id: str,
) -> dict[str, object]:
    """Load one ball's visual/size setup and bind it to shared geometry."""
    setup = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "ball_id",
        "ball_diameter_pixel",
        "image_size",
        "compatible_geometry_id",
        "template_difference_path",
    }
    if setup.get("ball_setup_version") != 1 or not required.issubset(setup):
        raise ValueError("小球设置文件版本未知或字段不完整")
    ball_id = str(setup["ball_id"]).strip()
    if not ball_id:
        raise ValueError("小球设置中的 ball_id 不能为空")
    if tuple(int(value) for value in setup["image_size"]) != image_size:
        raise ValueError("当前视频尺寸与小球设置不一致")
    if str(setup["compatible_geometry_id"]) != expected_geometry_id:
        raise ValueError("小球设置绑定的几何标定与当前几何不一致")
    if float(setup["ball_diameter_pixel"]) <= 0:
        raise ValueError("小球直径必须大于 0")
    template_path = Path(str(setup["template_difference_path"]))
    if not template_path.is_absolute():
        template_path = path.parent / template_path
    template_path = template_path.resolve()
    if not template_path.is_file():
        raise ValueError(f"小球模板不存在：{template_path}")
    setup["ball_id"] = ball_id
    setup["template_difference_path"] = str(template_path)
    return setup


def save_ball_setup(
    path: Path,
    annotation: dict[str, object],
    video: Path,
    *,
    template_frame_path: Path,
    background_path: Path,
    geometry_id: str,
    ball_id: str,
) -> Path:
    """Persist one ball's diameter and reusable difference template."""
    geometry_id = geometry_id.strip()
    ball_id = ball_id.strip()
    if not geometry_id or not ball_id:
        raise ValueError("geometry_id 和 ball_id 不能为空")
    template_path = _save_difference_template(
        path.with_name(f"{path.stem}_template.png"),
        annotation,
        template_frame_path=template_frame_path,
        background_path=background_path,
    )
    ball_box = annotation["ball_box"]
    payload = {
        "ball_setup_version": 1,
        "ball_id": ball_id,
        "ball_diameter_pixel": max(float(ball_box[2]), float(ball_box[3])),
        "image_size": annotation["image_size"],
        "compatible_geometry_id": geometry_id,
        "template_difference_path": template_path.name,
        "source_video_identity": video_identity(video),
        "scope": "one_ball_across_H1_to_H5",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return template_path


def load_setup_annotation(path: Path, image_size: tuple[int, int]) -> dict[str, object]:
    """Load reusable fixed-camera geometry without binding it to one video."""
    setup = json.loads(path.read_text(encoding="utf-8"))
    if setup.get("setup_version") != 2:
        raise ValueError("批次标定版本过旧，请重新创建以保存清晰小球模板")
    required = {
        "collision_roi",
        "wall_line_points",
        "ball_diameter_pixel",
        "image_size",
        "template_difference_path",
    }
    if not required.issubset(setup):
        raise ValueError("批次标定文件缺少碰撞区域、挡墙或小球尺寸")
    if tuple(int(value) for value in setup["image_size"]) != image_size:
        raise ValueError("当前视频尺寸与批次标定文件不一致")
    template_path = Path(str(setup["template_difference_path"]))
    if not template_path.is_absolute():
        template_path = path.parent / template_path
    template_path = template_path.resolve()
    if not template_path.is_file():
        raise ValueError(f"批次小球模板不存在：{template_path}")
    setup["template_difference_path"] = str(template_path)
    return setup


def save_setup_annotation(
    path: Path,
    annotation: dict[str, object],
    video: Path,
    *,
    template_frame_path: Path,
    background_path: Path,
) -> Path:
    """Persist reusable geometry and a clear background-difference ball template."""
    ball_box = annotation["ball_box"]
    template_path = _save_difference_template(
        path.with_name(f"{path.stem}_ball_template.png"),
        annotation,
        template_frame_path=template_frame_path,
        background_path=background_path,
    )
    payload = {
        "setup_version": 2,
        "collision_roi": annotation["collision_roi"],
        "wall_line_points": annotation["wall_line_points"],
        "ball_diameter_pixel": max(float(ball_box[2]), float(ball_box[3])),
        "image_size": annotation["image_size"],
        "template_difference_path": template_path.name,
        "source_video_identity": video_identity(video),
        "scope": "fixed_camera_batch_H1_to_H5",
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return template_path


def clear_derived_collision_results(annotation: dict[str, object]) -> None:
    """Keep manual labels while removing results invalidated by a new run."""
    for key in (
        "track_csv",
        "tracking_frame_count",
        "tracking_confidence",
        "tracking_start_time_s",
        "tracking_end_time_s",
        "method",
        "local_tracking_only",
        "full_motion_tracked",
        "fit_frames_each_branch",
        "fit_window_sizes",
        "fit_window_results",
        "local_margin_frames_each_branch",
        "tracking_candidate_multiplier",
        "duplicate_tracking_frames_removed",
        "tracking_pts_unique_and_strictly_increasing",
        "local_tracking_windows",
        "excluded_contact_interval_s",
        "velocity_before_pixel_per_s",
        "velocity_after_pixel_per_s",
        "coefficient_of_restitution_uncertainty",
        "window_relative_spread",
        "quality_status",
        "quality_reasons",
        "incidence_angle_degrees",
        "collision_geometry_status",
        "detector_diagnostics",
        "automatic_invalid_tracking_timestamps_s",
        "automatic_invalid_tracking_frame_count",
        "sparse_foreground_tracking_timestamps_s",
        "sparse_foreground_tracking_frame_count",
        "excluded_tracking_timestamps_s",
        "detection_review_complete",
        "standard_statistics_eligible",
        "coefficient_of_restitution",
        "formula",
        "timestamp_basis",
    ):
        annotation.pop(key, None)


def save_track_csv(source: Path, annotation_path: Path) -> Path:
    target = annotation_path.with_name(f"{annotation_path.stem}_track.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read_bytes())
    return target


def write_track_points(
    path: Path,
    points: list[TrackPoint] | list[BallPositionMeasurement],
    *,
    pre_collision_count: int,
    excluded_timestamps: set[float] | None = None,
    fit_windows: tuple[int, ...] = (5, 7, 9),
    minimum_confidence: float = 0.35,
) -> None:
    if not 0 < pre_collision_count < len(points):
        raise ValueError("局部轨迹的碰撞前后分段位置无效")
    if any(
        current.timestamp <= previous.timestamp
        for previous, current in zip(points, points[1:])
    ):
        raise ValueError("轨迹 CSV 的实际 PTS 必须严格递增且不能重复")
    excluded = excluded_timestamps or set()
    eligible = [
        point.timestamp not in excluded
        and point.confidence >= minimum_confidence
        and getattr(point, "contour_detected", True)
        for point in points
    ]
    memberships: list[list[int]] = [[] for _ in points]
    pre_valid = [
        index for index in range(pre_collision_count) if eligible[index]
    ]
    post_valid = [
        index for index in range(pre_collision_count, len(points)) if eligible[index]
    ]
    for window in fit_windows:
        if len(pre_valid) >= window and len(post_valid) >= window:
            for index in pre_valid[-window:] + post_valid[:window]:
                memberships[index].append(window)
    rows = [
        "local_frame_index,phase,timestamp,x_pixel,y_pixel,template_x_pixel,"
        "template_y_pixel,detector_disagreement_pixel,contour_detected,confidence,"
        "contour_mode,eligible_for_fitting,fit_windows"
    ]
    for index, point in enumerate(points):
        template_x = getattr(point, "template_x", point.x)
        template_y = getattr(point, "template_y", point.y)
        disagreement = getattr(point, "detector_disagreement_px", 0.0)
        contour_detected = getattr(point, "contour_detected", True)
        contour_mode = getattr(point, "contour_mode", "not_recorded")
        rows.append(
            f"{index},{'before_collision' if index < pre_collision_count else 'after_separation'},"
            f"{point.timestamp:.9f},{point.x:.4f},{point.y:.4f},"
            f"{template_x:.4f},{template_y:.4f},{disagreement:.4f},{int(contour_detected)},"
            f"{point.confidence:.6f},{contour_mode},{int(eligible[index])},"
            f"{'|'.join(str(window) for window in memberships[index])}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    video = args.video.expanduser().resolve()
    save_path = args.save_annotation.expanduser() if args.save_annotation else None
    setup_path = (
        args.setup_annotation.expanduser() if args.setup_annotation else None
    )
    geometry_path = (
        args.geometry_annotation.expanduser()
        if args.geometry_annotation
        else None
    )
    ball_setup_path = args.ball_setup.expanduser() if args.ball_setup else None
    try:
        if not video.is_file():
            raise ValueError(f"找不到视频：{video}")
        split_setup_requested = validate_setup_mode(
            setup_path=setup_path,
            geometry_path=geometry_path,
            ball_setup_path=ball_setup_path,
        )
        with tempfile.TemporaryDirectory(prefix="track_collision_") as directory:
            work = Path(directory)
            duration = probe_video_duration(video)
            if args.event_window_seconds <= 0:
                raise ValueError("event-window-seconds 必须大于 0 秒")
            fit_windows = tuple(int(value) for value in args.fit_windows)
            if not fit_windows or any(value < 3 for value in fit_windows):
                raise ValueError("fit-windows 中每个窗口至少为 3 帧")
            if len(set(fit_windows)) != len(fit_windows):
                raise ValueError("fit-windows 不能包含重复值")
            if args.local_margin_frames < 0:
                raise ValueError("local-margin-frames 不能为负数")
            if args.background_seconds <= 0:
                raise ValueError("background-seconds 必须大于 0 秒")
            all_timestamps = read_video_timestamps(video, duration, work)
            background_timestamps = select_empty_background_timestamps(
                all_timestamps,
                background_seconds=args.background_seconds,
            )
            background_paths, actual_background_timestamps = (
                extract_frames_at_timestamps(
                    video,
                    background_timestamps,
                    work,
                    label="empty_background",
                )
            )
            background_path = create_median_background(
                background_paths, work / "median_empty_background.png"
            )

            explicit_range = args.start_time is not None or args.end_time is not None
            if explicit_range and (
                args.start_time is None or args.end_time is None
            ):
                raise ValueError("start-time 与 end-time 必须同时填写")
            if explicit_range and args.reference_time is not None:
                raise ValueError("reference-time 不能与 start-time/end-time 同时使用")

            if args.annotation:
                annotation = json.loads(
                    args.annotation.expanduser().read_text(encoding="utf-8")
                )
                validate_annotation_video(annotation, video)
            else:
                annotation = {}
            validate_setup_mode(
                setup_path=setup_path,
                geometry_path=geometry_path,
                ball_setup_path=ball_setup_path,
                annotation=annotation,
            )
            if split_setup_requested:
                assert geometry_path is not None
                assert ball_setup_path is not None
                record_split_setup_provenance(
                    annotation,
                    geometry_path,
                    ball_setup_path,
                )

            if args.release_level is not None:
                stored_level = annotation.get("release_level")
                if stored_level is not None and stored_level != args.release_level:
                    raise ValueError(
                        f"标注文件释放位置为 {stored_level}，命令行指定为 {args.release_level}"
                    )
                annotation["release_level"] = args.release_level
            elif "release_level" not in annotation:
                annotation["release_level"] = ReleaseLevelWindow().run()

            event_keys = {"first_contact_timestamp_s", "first_separated_timestamp_s"}
            force_event_reselection = bool(
                args.annotation and (explicit_range or args.reference_time is not None)
            )
            if force_event_reselection:
                for key in (
                    "first_contact_analysis_frame",
                    "first_contact_timestamp_s",
                    "first_separated_analysis_frame",
                    "first_separated_timestamp_s",
                ):
                    annotation.pop(key, None)
            if explicit_range:
                event_start = float(args.start_time)
                event_end = float(args.end_time)
                if not 0 <= event_start < event_end <= duration:
                    raise ValueError("start-time/end-time 超出视频范围")
                requested_reference_time = (event_start + event_end) / 2
                event_timestamps = [
                    timestamp
                    for timestamp in all_timestamps
                    if event_start <= timestamp <= event_end
                ]
                selection_mode = "explicit_event_browse_range"
            else:
                if args.reference_time is not None:
                    requested_reference_time = args.reference_time
                    selection_mode = "reference_time_argument"
                elif event_keys.issubset(annotation):
                    requested_reference_time = float(
                        annotation.get(
                            "requested_reference_time_s",
                            (
                                float(annotation["first_contact_timestamp_s"])
                                + float(annotation["first_separated_timestamp_s"])
                            )
                            / 2,
                        )
                    )
                    selection_mode = str(
                        annotation.get("time_selection_mode", "saved_annotation")
                    )
                else:
                    previews = extract_video_preview_frames(
                        video, all_timestamps, work
                    )
                    exact_preview_path = work / "exact_time_preview.png"
                    requested_reference_time = WholeVideoTimeWindow(
                        previews,
                        duration,
                        lambda selected_time: extract_exact_frame(
                            video, selected_time, exact_preview_path
                        ),
                    ).run()
                    selection_mode = "whole_video_timeline"

                reference_index = nearest_timestamp_index(
                    all_timestamps, requested_reference_time
                )
                actual_reference_time = all_timestamps[reference_index]
                event_timestamps = select_event_browse_timestamps(
                    all_timestamps,
                    reference_time=actual_reference_time,
                    seconds_each_side=args.event_window_seconds,
                )

            if not all_timestamps[0] <= requested_reference_time <= all_timestamps[-1]:
                raise ValueError("选择的碰撞参考时间超出真实视频帧范围")
            actual_reference_time = all_timestamps[
                nearest_timestamp_index(all_timestamps, requested_reference_time)
            ]
            time_metadata: dict[str, object] = {
                "time_selection_mode": selection_mode,
                "requested_reference_time_s": requested_reference_time,
                "reference_time_s": actual_reference_time,
            }
            event_selection_performed = not event_keys.issubset(annotation)
            if event_selection_performed:
                event_paths, actual_event_timestamps = extract_frames_at_timestamps(
                    video, event_timestamps, work, label="event_browse"
                )
                event_frames = build_event_frames(
                    actual_event_timestamps, event_paths
                )
                initial_position = nearest_timestamp_index(
                    actual_event_timestamps, actual_reference_time
                )
                annotation.update(
                    FixedEndEventWindow(event_frames, initial_position).run()
                )
                time_metadata["event_browse_seconds_each_side"] = (
                    args.event_window_seconds
                )
                if explicit_range:
                    time_metadata["event_browse_range_s"] = [
                        event_start,
                        event_end,
                    ]
                else:
                    annotation.pop("event_browse_range_s", None)
                annotation.update(time_metadata)
                if save_path:
                    clear_derived_collision_results(annotation)
                    annotation["annotation_status"] = "events_selected"
                    save_annotation(save_path, annotation, video)

            contact_time = float(annotation["first_contact_timestamp_s"])
            separation_time = float(annotation["first_separated_timestamp_s"])
            pre_timestamps, post_timestamps = select_local_tracking_timestamps(
                all_timestamps,
                contact_time=contact_time,
                separation_time=separation_time,
                fit_frames=max(fit_windows),
                margin_frames=args.local_margin_frames,
                candidate_multiplier=2,
            )
            pre_paths, pre_actual_timestamps = extract_frames_at_timestamps(
                video,
                pre_timestamps,
                work,
                label="local_tracking_before",
            )
            post_paths, post_actual_timestamps = extract_frames_at_timestamps(
                video,
                post_timestamps,
                work,
                label="local_tracking_after",
            )
            required_unique_frames = max(fit_windows) + args.local_margin_frames
            if len(pre_actual_timestamps) < required_unique_frames:
                raise ValueError(
                    "碰撞前唯一实际帧不足："
                    f"需要 {required_unique_frames} 帧，去重后只有 "
                    f"{len(pre_actual_timestamps)} 帧"
                )
            if len(post_actual_timestamps) < required_unique_frames:
                raise ValueError(
                    "碰撞后唯一实际帧不足："
                    f"需要 {required_unique_frames} 帧，去重后只有 "
                    f"{len(post_actual_timestamps)} 帧"
                )
            duplicate_tracking_frames_removed = (
                len(pre_timestamps)
                + len(post_timestamps)
                - len(pre_actual_timestamps)
                - len(post_actual_timestamps)
            )
            pre_paths = pre_paths[-required_unique_frames:]
            pre_actual_timestamps = pre_actual_timestamps[-required_unique_frames:]
            post_paths = post_paths[:required_unique_frames]
            post_actual_timestamps = post_actual_timestamps[:required_unique_frames]
            local_paths = pre_paths + post_paths
            local_timestamps = pre_actual_timestamps + post_actual_timestamps
            pre_count = len(pre_actual_timestamps)

            with Image.open(local_paths[pre_count - 1]) as reference_image:
                current_image_size = reference_image.size
            setup_loaded = False
            geometry_id: str | None = None
            ball_id: str | None = None
            batch_template_path: Path | None = None
            annotation_performed = False

            if split_setup_requested:
                assert geometry_path is not None
                assert ball_setup_path is not None
                if geometry_path.is_file():
                    geometry = load_geometry_annotation(
                        geometry_path,
                        current_image_size,
                    )
                    geometry_id = str(geometry["geometry_id"])
                    validate_requested_setup_id(
                        args.geometry_id,
                        geometry_id,
                        "geometry_id",
                    )
                    stored_geometry_id = annotation.get("geometry_id")
                    if (
                        stored_geometry_id is not None
                        and stored_geometry_id != geometry_id
                    ):
                        raise ValueError(
                            f"trial 标注属于几何 {stored_geometry_id}，"
                            f"当前几何为 {geometry_id}"
                        )
                    annotation["collision_roi"] = geometry["collision_roi"]
                    annotation["wall_line_points"] = geometry["wall_line_points"]
                    annotation["image_size"] = list(current_image_size)
                else:
                    geometry_id = (args.geometry_id or geometry_path.stem).strip()
                    geometry_annotation = GeometryAnnotationWindow(
                        local_paths[pre_count - 1]
                    ).run()
                    annotation.update(geometry_annotation)
                    save_geometry_annotation(
                        geometry_path,
                        annotation,
                        video,
                        geometry_id=geometry_id,
                    )
                    annotation_performed = True

                if ball_setup_path.is_file():
                    ball_setup = load_ball_setup(
                        ball_setup_path,
                        current_image_size,
                        expected_geometry_id=geometry_id,
                    )
                    ball_id = str(ball_setup["ball_id"])
                    validate_requested_setup_id(
                        args.ball_id,
                        ball_id,
                        "ball_id",
                    )
                    stored_ball_id = annotation.get("ball_id")
                    if stored_ball_id is not None and stored_ball_id != ball_id:
                        raise ValueError(
                            f"trial 标注属于 {stored_ball_id}，当前小球为 {ball_id}"
                        )
                    if stored_ball_id != ball_id:
                        annotation.pop("ball_box", None)
                    batch_template_path = Path(
                        str(ball_setup["template_difference_path"])
                    )
                    if "ball_box" not in annotation:
                        annotation["ball_box"] = list(
                            locate_ball_template_box_from_files(
                                local_paths[pre_count - 1],
                                background_path,
                                roi=annotation["collision_roi"],
                                ball_diameter_px=float(
                                    ball_setup["ball_diameter_pixel"]
                                ),
                            )
                        )
                else:
                    ball_id = (args.ball_id or ball_setup_path.stem).strip()
                    template_frame_index = TemplateFrameSelectionWindow(
                        local_paths[:pre_count],
                        local_timestamps[:pre_count],
                    ).run()
                    template_frame_path = local_paths[template_frame_index]
                    annotation.update(
                        BallTemplateAnnotationWindow(template_frame_path).run()
                    )
                    batch_template_path = save_ball_setup(
                        ball_setup_path,
                        annotation,
                        video,
                        template_frame_path=template_frame_path,
                        background_path=background_path,
                        geometry_id=geometry_id,
                        ball_id=ball_id,
                    )
                    time_metadata["ball_template_frame_timestamp_s"] = (
                        local_timestamps[template_frame_index]
                    )
                    annotation_performed = True

                annotation["geometry_id"] = geometry_id
                annotation["ball_id"] = ball_id
                annotation["ball_template_difference_path"] = str(
                    batch_template_path
                )

            elif setup_path and setup_path.is_file():
                setup = load_setup_annotation(setup_path, current_image_size)
                annotation["collision_roi"] = setup["collision_roi"]
                annotation["wall_line_points"] = setup["wall_line_points"]
                annotation["image_size"] = list(current_image_size)
                batch_template_path = Path(str(setup["template_difference_path"]))
                if "ball_box" not in annotation:
                    annotation["ball_box"] = list(
                        locate_ball_template_box_from_files(
                            local_paths[pre_count - 1],
                            background_path,
                            roi=setup["collision_roi"],
                            ball_diameter_px=float(setup["ball_diameter_pixel"]),
                        )
                    )
                setup_loaded = True

            if not split_setup_requested:
                template_frame_index = pre_count - 1
                if setup_path and not setup_loaded:
                    template_frame_index = TemplateFrameSelectionWindow(
                        local_paths[:pre_count],
                        local_timestamps[:pre_count],
                    ).run()
                template_frame_path = local_paths[template_frame_index]
                legacy_annotation_needed = not {
                    "ball_box",
                    "collision_roi",
                    "wall_line_points",
                }.issubset(annotation)
                if legacy_annotation_needed:
                    annotation.update(
                        TrackAnnotationWindow(template_frame_path).run()
                    )
                    time_metadata["annotation_frame_time_s"] = local_timestamps[
                        template_frame_index
                    ]
                    time_metadata["annotation_frame_actual_timestamp_s"] = (
                        local_timestamps[template_frame_index]
                    )
                    annotation_performed = True
                if setup_path and not setup_loaded:
                    batch_template_path = save_setup_annotation(
                        setup_path,
                        annotation,
                        video,
                        template_frame_path=template_frame_path,
                        background_path=background_path,
                    )
                    time_metadata["batch_template_frame_timestamp_s"] = (
                        local_timestamps[template_frame_index]
                    )
                    annotation_performed = True
            time_metadata["current_local_reference_timestamp_s"] = local_timestamps[
                pre_count - 1
            ]
            annotation.update(time_metadata)
            if save_path and (event_selection_performed or annotation_performed):
                clear_derived_collision_results(annotation)
                annotation["annotation_status"] = "manual_annotation_complete"
                save_annotation(save_path, annotation, video)

            measurements = detect_ball_positions_dual_from_files(
                local_paths,
                local_timestamps,
                background_path,
                roi=annotation["collision_roi"],
                template_frame_index=pre_count - 1,
                ball_box=annotation["ball_box"],
                wall_line_points=annotation["wall_line_points"],
                template_difference_path=batch_template_path,
                trajectory_break_indices=(pre_count,),
            )
            automatic_invalid_timestamps = sorted(
                measurement.timestamp
                for measurement in measurements
                if not measurement.contour_detected
            )
            sparse_foreground_timestamps = sorted(
                measurement.timestamp
                for measurement in measurements
                if measurement.contour_mode == "sparse_foreground"
            )
            review_performed = bool(
                args.review_detections
                or not annotation.get("detection_review_complete", False)
                or not annotation.get(
                    "tracking_pts_unique_and_strictly_increasing", False
                )
            )
            if review_performed:
                excluded_timestamps = DetectionReviewWindow(
                    local_paths,
                    measurements,
                    pre_collision_count=pre_count,
                    minimum_frames_each_branch=max(fit_windows),
                    initially_excluded=list(
                        annotation.get("excluded_tracking_timestamps_s", [])
                    ),
                ).run()
                annotation["excluded_tracking_timestamps_s"] = excluded_timestamps
                annotation["detection_review_complete"] = True
            else:
                excluded_timestamps = [
                    float(value)
                    for value in annotation.get("excluded_tracking_timestamps_s", [])
                ]
            excluded_timestamps = sorted(
                set(excluded_timestamps) | set(automatic_invalid_timestamps)
            )
            excluded_set = set(excluded_timestamps)
            points = [
                TrackPoint(
                    measurement.timestamp,
                    measurement.x,
                    measurement.y,
                    0.0
                    if measurement.timestamp in excluded_set
                    else measurement.confidence,
                )
                for measurement in measurements
            ]
            track_path = work / "track.csv"
            write_track_points(
                track_path,
                measurements,
                pre_collision_count=pre_count,
                excluded_timestamps=excluded_set,
                fit_windows=fit_windows,
            )
            annotation["tracking_frame_count"] = len(points)
            annotation["tracking_confidence"] = {
                "minimum": min(point.confidence for point in points),
                "maximum": max(point.confidence for point in points),
                "mean": sum(point.confidence for point in points) / len(points),
                "frames_at_or_above_0_35": sum(
                    point.confidence >= 0.35 for point in points
                ),
                "manually_excluded_frames": len(
                    excluded_set - set(automatic_invalid_timestamps)
                ),
                "automatically_invalid_frames": len(
                    automatic_invalid_timestamps
                ),
            }
            annotation["automatic_invalid_tracking_timestamps_s"] = (
                automatic_invalid_timestamps
            )
            annotation["automatic_invalid_tracking_frame_count"] = len(
                automatic_invalid_timestamps
            )
            annotation["sparse_foreground_tracking_timestamps_s"] = (
                sparse_foreground_timestamps
            )
            annotation["sparse_foreground_tracking_frame_count"] = len(
                sparse_foreground_timestamps
            )
            wall_1, wall_2 = annotation["wall_line_points"]
            result = estimate_fixed_wall_restitution_multiwindow(
                points,
                tuple(wall_1),
                tuple(wall_2),
                contact_time,
                separation_time,
                window_sizes=fit_windows,
            )
            restitution = float(result["coefficient_of_restitution"])
            uncertainty = float(result["uncertainty"])
            window_results = list(result["window_results"])
            velocity_before = float(
                np.median(
                    [
                        item["velocity_before_normal_pixel_per_s"]
                        for item in window_results
                    ]
                )
            )
            velocity_after = float(
                np.median(
                    [
                        item["velocity_after_normal_pixel_per_s"]
                        for item in window_results
                    ]
                )
            )
            retained_measurements = [
                measurement
                for measurement in measurements
                if measurement.timestamp not in excluded_set
                and measurement.contour_detected
            ]
            ball_diameter = max(
                float(annotation["ball_box"][2]),
                float(annotation["ball_box"][3]),
            )
            disagreement_ratios = [
                measurement.detector_disagreement_px / ball_diameter
                for measurement in retained_measurements
            ]
            detector_diagnostics = {
                "ball_diameter_pixel": ball_diameter,
                "mean_disagreement_pixel": float(
                    np.mean(
                        [
                            measurement.detector_disagreement_px
                            for measurement in retained_measurements
                        ]
                    )
                ),
                "maximum_disagreement_pixel": max(
                    measurement.detector_disagreement_px
                    for measurement in retained_measurements
                ),
                "maximum_disagreement_ball_diameter_ratio": max(
                    disagreement_ratios
                ),
                "maximum_allowed_disagreement_ball_diameter_ratio": (
                    MAX_DETECTOR_DISAGREEMENT_DIAMETER_RATIO
                ),
                "contour_position_is_physics_measurement": True,
                "template_position_is_cross_check_only": False,
                "template_position_guides_object_association": True,
                "silhouette_uses_ball_size_and_circular_gate": True,
                "association_rejects_temporal_outliers": True,
                "association_uses_explicit_collision_branch_boundary": True,
                "pre_collision_branch_uses_manual_ball_anchor": True,
                "post_collision_branch_bootstraps_from_pre_collision_boundary": True,
                "anchored_branches_use_foreground_continuity": True,
                "template_fallback_is_never_physics_measurement": True,
                "automatic_invalid_frame_count": len(
                    automatic_invalid_timestamps
                ),
                "sparse_foreground_frame_count": len(
                    sparse_foreground_timestamps
                ),
            }
            quality_reasons: list[str] = []
            if result["quality_status"] != "ok":
                quality_reasons.append("fit_window_results_inconsistent")
            if (
                max(disagreement_ratios)
                > MAX_DETECTOR_DISAGREEMENT_DIAMETER_RATIO
            ):
                quality_reasons.append("contour_template_disagreement")
            if result["collision_geometry_status"] != "normal_collision":
                quality_reasons.append("non_normal_collision")
            quality_status = "ok" if not quality_reasons else "review_required"
            ResultConfirmationWindow(
                release_level=str(annotation["release_level"]),
                restitution=restitution,
                uncertainty=uncertainty,
                quality_status=quality_status,
                geometry_status=str(result["collision_geometry_status"]),
                incidence_angle_degrees=float(result["incidence_angle_degrees"]),
            ).run()
            annotation.update(
                {
                    "video": str(video),
                    "empty_background_interval_s": [
                        actual_background_timestamps[0],
                        actual_background_timestamps[-1],
                    ],
                    "empty_background_frame_count": len(actual_background_timestamps),
                    "tracking_start_time_s": local_timestamps[0],
                    "tracking_end_time_s": local_timestamps[-1],
                    "method": "fixed_wall_dual_detector_multiwindow_normal_velocity",
                    "local_tracking_only": True,
                    "full_motion_tracked": False,
                    "fit_window_sizes": list(fit_windows),
                    "fit_window_results": window_results,
                    "local_margin_frames_each_branch": args.local_margin_frames,
                    "tracking_candidate_multiplier": 2,
                    "duplicate_tracking_frames_removed": (
                        duplicate_tracking_frames_removed
                    ),
                    "tracking_pts_unique_and_strictly_increasing": True,
                    "local_tracking_windows": {
                        "before_collision": {
                            "start_timestamp_s": local_timestamps[0],
                            "end_timestamp_s": local_timestamps[pre_count - 1],
                            "extracted_frame_count": pre_count,
                        },
                        "after_separation": {
                            "start_timestamp_s": local_timestamps[pre_count],
                            "end_timestamp_s": local_timestamps[-1],
                            "extracted_frame_count": len(local_timestamps) - pre_count,
                        },
                    },
                    "excluded_contact_interval_s": [contact_time, separation_time],
                    "velocity_before_normal_pixel_per_s": velocity_before,
                    "velocity_after_normal_pixel_per_s": velocity_after,
                    "coefficient_of_restitution": restitution,
                    "coefficient_of_restitution_uncertainty": uncertainty,
                    "window_relative_spread": result["window_relative_spread"],
                    "incidence_angle_degrees": result["incidence_angle_degrees"],
                    "collision_geometry_status": result[
                        "collision_geometry_status"
                    ],
                    "standard_statistics_eligible": (
                        result["collision_geometry_status"] == "normal_collision"
                        and quality_status == "ok"
                    ),
                    "detector_diagnostics": detector_diagnostics,
                    "quality_status": quality_status,
                    "quality_reasons": quality_reasons,
                    "formula": "median(abs(v_after_normal / v_before_normal))",
                    "timestamp_basis": "video_presentation_timestamp",
                    "spatial_calibration_used": False,
                    "metric_velocity_computed": False,
                    "batch_setup_annotation": (
                        str(setup_path.resolve()) if setup_path else None
                    ),
                    "batch_template_difference_path": (
                        str(batch_template_path)
                        if setup_path and batch_template_path
                        else None
                    ),
                    "annotation_status": "complete",
                }
            )
            if save_path:
                saved_track = save_track_csv(track_path, save_path)
                annotation["track_csv"] = str(saved_track)
                save_annotation(
                    save_path,
                    annotation,
                    video,
                    local_timestamps[0],
                    local_timestamps[-1],
                )
            print(
                f"{annotation['release_level']} 恢复系数 e = "
                f"{restitution:.4f} ± {uncertainty:.4f}"
            )
            print(f"质量状态：{quality_status}")
            print(
                f"碰撞几何：{result['collision_geometry_status']}｜"
                f"入射角 {float(result['incidence_angle_degrees']):.2f}°"
            )
            return 0
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        print(f"错误：视频处理失败\n{detail}", file=sys.stderr)
    except (OSError, RuntimeError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"错误：{error}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
