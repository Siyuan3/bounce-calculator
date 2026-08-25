"""Track a fast ball in a fixed-camera narrow track using background difference."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from restitution_math import TrackPoint, wall_unit_normal


@dataclass(frozen=True)
class BallPositionMeasurement:
    """One frame measured by silhouette midpoint and template registration."""

    timestamp: float
    x: float
    y: float
    template_x: float
    template_y: float
    detector_disagreement_px: float
    confidence: float
    contour_detected: bool = True
    contour_mode: str = "dense_silhouette"

    def as_track_point(self) -> TrackPoint:
        return TrackPoint(self.timestamp, self.x, self.y, self.confidence)


def _window_sums(values: np.ndarray, window_height: int, window_width: int) -> np.ndarray:
    """Return every valid rectangular sum using an integral image."""
    integral = np.pad(values.astype(np.float32), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    return (
        integral[window_height:, window_width:]
        - integral[:-window_height, window_width:]
        - integral[window_height:, :-window_width]
        + integral[:-window_height, :-window_width]
    )


def _match_template_across_roi(
    roi_difference: np.ndarray,
    template_difference: np.ndarray,
    *,
    sample_stride: int,
) -> tuple[float, float, float]:
    """Return an ROI-global template centre and cosine similarity."""
    sampled_roi = roi_difference[::sample_stride, ::sample_stride].astype(np.float32)
    sampled_template = template_difference[
        ::sample_stride, ::sample_stride
    ].astype(np.float32)
    rh, rw = sampled_roi.shape
    th, tw = sampled_template.shape
    if th > rh or tw > rw:
        raise ValueError("小球模板大于碰撞区域")
    windows = np.lib.stride_tricks.sliding_window_view(
        sampled_roi,
        (th, tw),
    )
    valid_correlation = np.einsum(
        "ijkl,kl->ij",
        windows,
        sampled_template,
        optimize=True,
    )
    candidate_energy = np.einsum(
        "ijkl,ijkl->ij",
        windows,
        windows,
        optimize=True,
    )
    template_energy = float(np.sum(sampled_template * sampled_template))
    denominator = np.sqrt(np.maximum(candidate_energy * template_energy, 1e-12))
    similarity = valid_correlation / denominator
    top, left = np.unravel_index(int(np.argmax(similarity)), similarity.shape)
    centre_x = left * sample_stride + template_difference.shape[1] / 2
    centre_y = top * sample_stride + template_difference.shape[0] / 2
    return float(centre_x), float(centre_y), float(similarity[top, left])


def _trajectory_guided_association_centres(
    timestamps: Sequence[float],
    template_centres: Sequence[tuple[float, float]],
    *,
    ball_diameter: float,
    trajectory_break_indices: Sequence[int] = (),
) -> list[tuple[float, float]]:
    """Replace only gross one-frame association jumps with a robust line estimate."""
    if len(timestamps) != len(template_centres):
        raise ValueError("模板位置与时间戳数量不一致")
    if len(timestamps) < 3:
        return list(template_centres)
    time_values = np.asarray(timestamps, dtype=float)
    positive_steps = np.diff(time_values)
    if np.any(positive_steps <= 0):
        raise ValueError("目标关联要求严格递增的时间戳")
    typical_step = float(np.median(positive_steps))
    explicit_breaks = sorted(set(int(value) for value in trajectory_break_indices))
    if any(value <= 0 or value >= len(timestamps) for value in explicit_breaks):
        raise ValueError("轨迹分支边界必须位于帧序列内部")
    inferred_breaks = [
        index + 1
        for index, step in enumerate(positive_steps)
        if step > typical_step * 3.0
    ]
    split_indices = sorted(set([*explicit_breaks, *inferred_breaks]))
    segment_boundaries = [0, *split_indices, len(timestamps)]
    result = list(template_centres)
    maximum_raw_deviation = ball_diameter * 0.35

    for start, end in zip(segment_boundaries, segment_boundaries[1:]):
        if end - start < 3:
            continue
        segment_times = time_values[start:end]
        segment_values = np.asarray(template_centres[start:end], dtype=float)
        predictions = np.empty_like(segment_values)
        for coordinate in range(2):
            values = segment_values[:, coordinate]
            slopes = [
                (values[right] - values[left])
                / (segment_times[right] - segment_times[left])
                for left in range(len(values) - 1)
                for right in range(left + 1, len(values))
            ]
            slope = float(np.median(slopes))
            intercept = float(np.median(values - slope * segment_times))
            predictions[:, coordinate] = slope * segment_times + intercept
        deviations = np.linalg.norm(segment_values - predictions, axis=1)
        for offset, deviation in enumerate(deviations):
            if deviation > maximum_raw_deviation:
                result[start + offset] = tuple(predictions[offset])
    return result


def _foreground_window_centre_near_prediction(
    difference: np.ndarray,
    *,
    roi: tuple[int, int, int, int],
    predicted_centre: tuple[float, float],
    branch_anchor: tuple[float, float],
    motion_normal: tuple[float, float],
    ball_diameter: float,
) -> tuple[float, float] | None:
    """Find ball-sized foreground near a temporally predicted branch position."""
    rx, ry, rw, rh = roi
    diameter = max(4, int(round(ball_diameter)))
    if diameter > rw or diameter > rh:
        return None
    roi_difference = np.minimum(
        difference[ry : ry + rh, rx : rx + rw],
        80.0,
    )
    evidence = _window_sums(roi_difference, diameter, diameter)
    centre_y, centre_x = np.mgrid[: evidence.shape[0], : evidence.shape[1]]
    centre_x = centre_x.astype(float) + rx + diameter / 2
    centre_y = centre_y.astype(float) + ry + diameter / 2

    nx, ny = motion_normal
    tx, ty = -ny, nx
    predicted_x, predicted_y = predicted_centre
    anchor_x, anchor_y = branch_anchor
    normal_offset = (centre_x - predicted_x) * nx + (centre_y - predicted_y) * ny
    tangent_offset = (centre_x - anchor_x) * tx + (centre_y - anchor_y) * ty
    search_radius = ball_diameter * 1.8
    tangent_radius = ball_diameter * 0.65
    eligible = (
        np.abs(normal_offset) <= search_radius
    ) & (np.abs(tangent_offset) <= tangent_radius)
    if not np.any(eligible):
        return None

    eligible_evidence = evidence[eligible]
    strongest = float(np.max(eligible_evidence))
    if strongest < diameter * diameter * 4.0:
        return None
    evidence_score = evidence / strongest
    distance_penalty = 0.18 * (normal_offset / search_radius) ** 2
    tangent_penalty = 0.35 * (tangent_offset / tangent_radius) ** 2
    score = evidence_score - distance_penalty - tangent_penalty
    score[~eligible] = -np.inf
    top, left = np.unravel_index(int(np.argmax(score)), score.shape)
    return (
        float(rx + left + diameter / 2),
        float(ry + top + diameter / 2),
    )


def _anchor_guided_association_centres(
    grayscale_frames: np.ndarray,
    timestamps: Sequence[float],
    background: np.ndarray,
    initial_centres: Sequence[tuple[float, float]],
    *,
    roi: tuple[int, int, int, int],
    ball_diameter: float,
    motion_normal: tuple[float, float],
    anchor_index: int,
    anchor_centre: tuple[float, float],
    trajectory_break_indices: Sequence[int] = (),
) -> list[tuple[float, float]]:
    """Track the anchor's free-motion branch using local foreground continuity."""
    if len(grayscale_frames) != len(timestamps) or len(initial_centres) != len(
        timestamps
    ):
        raise ValueError("锚点关联的帧、时间戳与初始位置数量不一致")
    if not 0 <= anchor_index < len(timestamps):
        raise ValueError("人工小球锚点帧编号超出范围")
    time_values = np.asarray(timestamps, dtype=float)
    positive_steps = np.diff(time_values)
    if np.any(positive_steps <= 0):
        raise ValueError("锚点关联要求严格递增的时间戳")
    explicit_breaks = sorted(set(int(value) for value in trajectory_break_indices))
    if any(value <= 0 or value >= len(timestamps) for value in explicit_breaks):
        raise ValueError("轨迹分支边界必须位于帧序列内部")
    inferred_breaks: list[int] = []
    if len(positive_steps):
        typical_step = float(np.median(positive_steps))
        inferred_breaks = [
            index + 1
            for index, step in enumerate(positive_steps)
            if step > typical_step * 3.0
        ]
    boundaries = [
        0,
        *sorted(set([*explicit_breaks, *inferred_breaks])),
        len(timestamps),
    ]
    segment_start = max(value for value in boundaries if value <= anchor_index)
    segment_end = min(value for value in boundaries if value > anchor_index)
    result = list(initial_centres)
    result[anchor_index] = anchor_centre

    def walk(
        indices: Sequence[int],
        *,
        seed_index: int,
        branch_anchor: tuple[float, float],
    ) -> None:
        previous_index = seed_index
        previous_previous_index: int | None = None
        for index in indices:
            previous_centre = result[previous_index]
            predicted_centre = previous_centre
            if previous_previous_index is not None:
                earlier_centre = result[previous_previous_index]
                elapsed = timestamps[previous_index] - timestamps[previous_previous_index]
                step = timestamps[index] - timestamps[previous_index]
                if elapsed != 0:
                    predicted_centre = (
                        previous_centre[0]
                        + (previous_centre[0] - earlier_centre[0]) * step / elapsed,
                        previous_centre[1]
                        + (previous_centre[1] - earlier_centre[1]) * step / elapsed,
                    )
            difference = np.abs(
                grayscale_frames[index].astype(np.float32) - background
            )
            candidate = _foreground_window_centre_near_prediction(
                difference,
                roi=roi,
                predicted_centre=predicted_centre,
                branch_anchor=branch_anchor,
                motion_normal=motion_normal,
                ball_diameter=ball_diameter,
            )
            result[index] = candidate if candidate is not None else predicted_centre
            previous_previous_index = previous_index
            previous_index = index

    walk(
        list(range(anchor_index - 1, segment_start - 1, -1)),
        seed_index=anchor_index,
        branch_anchor=anchor_centre,
    )
    walk(
        list(range(anchor_index + 1, segment_end)),
        seed_index=anchor_index,
        branch_anchor=anchor_centre,
    )

    # A fixed-wall collision leaves the ball close to its last pre-contact
    # position. Bootstrap the explicitly separated post-collision branch from
    # that trusted boundary instead of from a potentially drifting template.
    if segment_end in explicit_breaks:
        post_start = segment_end
        post_end = min(value for value in boundaries if value > post_start)
        pre_boundary_centre = result[post_start - 1]
        post_difference = np.abs(
            grayscale_frames[post_start].astype(np.float32) - background
        )
        post_anchor = _foreground_window_centre_near_prediction(
            post_difference,
            roi=roi,
            predicted_centre=pre_boundary_centre,
            branch_anchor=pre_boundary_centre,
            motion_normal=motion_normal,
            ball_diameter=ball_diameter,
        )
        result[post_start] = (
            post_anchor if post_anchor is not None else pre_boundary_centre
        )
        walk(
            list(range(post_start + 1, post_end)),
            seed_index=post_start,
            branch_anchor=result[post_start],
        )
    return result


def locate_ball_template_box(
    grayscale_frame: np.ndarray,
    background_frame: np.ndarray,
    *,
    roi: Sequence[float],
    ball_diameter_px: float,
) -> tuple[float, float, float, float]:
    """Locate the strongest ball-sized foreground patch in a fixed collision ROI."""
    if grayscale_frame.ndim != 2 or background_frame.shape != grayscale_frame.shape:
        raise ValueError("模板定位需要尺寸一致的单帧灰度图和空背景")
    height, width = grayscale_frame.shape
    rx, ry, rw, rh = (int(round(value)) for value in roi)
    diameter = int(round(ball_diameter_px))
    if diameter < 4:
        raise ValueError("小球直径像素值过小")
    if rx < 0 or ry < 0 or rx + rw > width or ry + rh > height:
        raise ValueError("碰撞区域超出视频画面")
    if diameter > rw or diameter > rh:
        raise ValueError("小球直径大于碰撞区域")
    difference = np.abs(
        grayscale_frame.astype(np.float32) - background_frame.astype(np.float32)
    )[ry : ry + rh, rx : rx + rw]
    sums = _window_sums(difference, diameter, diameter)
    top, left = np.unravel_index(int(np.argmax(sums)), sums.shape)
    if float(sums[top, left]) < diameter * diameter * 4:
        raise ValueError("清晰帧中没有找到足够明显的小球前景")
    return float(rx + left), float(ry + top), float(diameter), float(diameter)


def locate_ball_template_box_from_files(
    frame_path: Path,
    background_path: Path,
    *,
    roi: Sequence[float],
    ball_diameter_px: float,
) -> tuple[float, float, float, float]:
    frame = np.asarray(Image.open(frame_path).convert("L"), dtype=np.uint8)
    background = np.asarray(Image.open(background_path).convert("L"), dtype=np.uint8)
    return locate_ball_template_box(
        frame,
        background,
        roi=roi,
        ball_diameter_px=ball_diameter_px,
    )


def detect_ball_positions_dual(
    grayscale_frames: np.ndarray,
    timestamps: Sequence[float],
    background_frame: np.ndarray,
    *,
    roi: Sequence[float],
    template_frame_index: int,
    ball_box: Sequence[float],
    wall_line_points: Sequence[Sequence[float]],
    template_difference_frame: np.ndarray | None = None,
    trajectory_break_indices: Sequence[int] = (),
) -> list[BallPositionMeasurement]:
    """Locate one ball using a shape-filtered silhouette and appearance matching.

    The silhouette midpoint is the measurement used for physics. Raw template
    matches supply object association and remain visible as a disagreement check.
    """
    if grayscale_frames.ndim != 3:
        raise ValueError("灰度帧数组必须为 [frame, height, width]")
    if background_frame.shape != grayscale_frames.shape[1:]:
        raise ValueError("空背景尺寸必须与视频帧一致")
    if len(grayscale_frames) != len(timestamps) or not len(timestamps):
        raise ValueError("帧数与时间戳数量不一致，或没有视频帧")
    if not 0 <= template_frame_index < len(grayscale_frames):
        raise ValueError("模板帧编号超出范围")

    frame_height, frame_width = background_frame.shape
    rx, ry, rw, rh = (int(round(value)) for value in roi)
    bx, by, bw, bh = (int(round(value)) for value in ball_box)
    if rw < 8 or rh < 8 or bw < 4 or bh < 4:
        raise ValueError("碰撞区域或小球模板框过小")
    if rx < 0 or ry < 0 or rx + rw > frame_width or ry + rh > frame_height:
        raise ValueError("碰撞区域超出视频画面")
    if bx < 0 or by < 0 or bx + bw > frame_width or by + bh > frame_height:
        raise ValueError("小球模板框超出视频画面")

    background = background_frame.astype(np.float32)
    if len(wall_line_points) != 2:
        raise ValueError("挡墙线必须包含两个点")
    wall_1 = tuple(float(value) for value in wall_line_points[0])
    wall_2 = tuple(float(value) for value in wall_line_points[1])
    nx, ny = wall_unit_normal(wall_1, wall_2)
    tx, ty = -ny, nx
    if template_difference_frame is None:
        template_difference = np.abs(
            grayscale_frames[template_frame_index].astype(np.float32) - background
        )[by : by + bh, bx : bx + bw]
    else:
        if template_difference_frame.ndim != 2:
            raise ValueError("批次小球模板必须是单通道图像")
        template_difference = template_difference_frame.astype(np.float32)
        bh, bw = template_difference.shape
    if float(template_difference.max()) < 8.0:
        raise ValueError("模板框内没有检测到与空背景不同的小球")

    local_half_width = max(bw, round(bw * 0.9))
    local_half_height = max(bh, round(bh * 0.9))
    template_sample_stride = max(1, round(max(bw, bh) / 24))
    ball_diameter = float(max(bw, bh))
    measurements: list[BallPositionMeasurement] = []

    template_matches: list[tuple[float, float, float]] = []
    for frame in grayscale_frames:
        difference = np.abs(frame.astype(np.float32) - background)
        roi_difference = difference[ry : ry + rh, rx : rx + rw]
        template_local_x, template_local_y, best_similarity = (
            _match_template_across_roi(
                roi_difference,
                template_difference,
                sample_stride=template_sample_stride,
            )
        )
        template_x = rx + template_local_x
        template_y = ry + template_local_y
        template_matches.append((template_x, template_y, best_similarity))
    association_centres = _trajectory_guided_association_centres(
        timestamps,
        [(x, y) for x, y, _ in template_matches],
        ball_diameter=ball_diameter,
        trajectory_break_indices=trajectory_break_indices,
    )
    association_centres = _anchor_guided_association_centres(
        grayscale_frames,
        timestamps,
        background,
        association_centres,
        roi=(rx, ry, rw, rh),
        ball_diameter=ball_diameter,
        motion_normal=(nx, ny),
        anchor_index=template_frame_index,
        anchor_centre=(bx + bw / 2, by + bh / 2),
        trajectory_break_indices=trajectory_break_indices,
    )

    for timestamp, frame, template_match, association_centre in zip(
        timestamps,
        grayscale_frames,
        template_matches,
        association_centres,
    ):
        template_x, template_y, best_similarity = template_match
        association_x, association_y = association_centre
        difference = np.abs(frame.astype(np.float32) - background)
        x0 = max(rx, int(round(association_x - local_half_width)))
        x1 = min(rx + rw, int(round(association_x + local_half_width + 1)))
        y0 = max(ry, int(round(association_y - local_half_height)))
        y1 = min(ry + rh, int(round(association_y + local_half_height + 1)))
        local_difference = difference[y0:y1, x0:x1]
        threshold = max(12.0, float(local_difference.max()) * 0.28)
        foreground = local_difference >= threshold
        local_global_y, local_global_x = np.mgrid[y0:y1, x0:x1]
        association_radius = max(3.0, ball_diameter * 0.68)
        circular_ball_gate = (
            (local_global_x - association_x) ** 2
            + (local_global_y - association_y) ** 2
            <= association_radius**2
        )
        foreground &= circular_ball_gate
        minimum_dense_span = max(3, round(ball_diameter * 0.28))
        dense_rows = foreground.sum(axis=1) >= minimum_dense_span
        dense_columns = foreground.sum(axis=0) >= minimum_dense_span
        ball_shaped_foreground = (
            foreground & dense_rows[:, np.newaxis] & dense_columns[np.newaxis, :]
        )
        local_y, local_x = np.nonzero(ball_shaped_foreground)
        contour_mode = "dense_silhouette"
        required_pixels = max(8, round(bw * bh * 0.12))
        if len(local_x) < required_pixels:
            raw_y, raw_x = np.nonzero(foreground)
            raw_x_span = (
                float(np.percentile(raw_x, 95) - np.percentile(raw_x, 5))
                if len(raw_x)
                else 0.0
            )
            raw_y_span = (
                float(np.percentile(raw_y, 95) - np.percentile(raw_y, 5))
                if len(raw_y)
                else 0.0
            )
            minimum_sparse_span = ball_diameter * 0.55
            if (
                len(raw_x) >= required_pixels
                and raw_x_span >= minimum_sparse_span
                and raw_y_span >= minimum_sparse_span
            ):
                local_x = raw_x
                local_y = raw_y
                contour_mode = "sparse_foreground"
            else:
                measurements.append(
                    BallPositionMeasurement(
                        timestamp=float(timestamp),
                        x=float(template_x),
                        y=float(template_y),
                        template_x=float(template_x),
                        template_y=float(template_y),
                        detector_disagreement_px=0.0,
                        confidence=0.0,
                        contour_detected=False,
                        contour_mode="template_only_invalid",
                    )
                )
                continue
        global_x = local_x.astype(float) + x0
        global_y = local_y.astype(float) + y0
        normal_values = global_x * nx + global_y * ny
        tangent_values = global_x * tx + global_y * ty
        normal_midpoint = float(
            (np.percentile(normal_values, 5) + np.percentile(normal_values, 95))
            / 2
        )
        tangent_midpoint = float(
            (np.percentile(tangent_values, 5) + np.percentile(tangent_values, 95))
            / 2
        )
        contour_x = normal_midpoint * nx + tangent_midpoint * tx
        contour_y = normal_midpoint * ny + tangent_midpoint * ty

        disagreement = hypot(contour_x - template_x, contour_y - template_y)
        agreement_score = max(0.0, 1.0 - disagreement / max(ball_diameter, 1.0))
        contrast_score = min(1.0, float(local_difference.max()) / 50.0)
        template_score = max(0.0, min(1.0, best_similarity))
        confidence = float(
            max(0.0, min(1.0, 0.45 * agreement_score + 0.30 * contrast_score + 0.25 * template_score))
        )
        if contour_mode == "sparse_foreground":
            confidence *= 0.85
        measurements.append(
            BallPositionMeasurement(
                timestamp=float(timestamp),
                x=contour_x,
                y=contour_y,
                template_x=float(template_x),
                template_y=float(template_y),
                detector_disagreement_px=float(disagreement),
                confidence=confidence,
                contour_mode=contour_mode,
            )
        )
    return measurements


def detect_ball_positions_dual_from_files(
    frame_paths: Sequence[Path],
    timestamps: Sequence[float],
    background_path: Path,
    *,
    roi: Sequence[float],
    template_frame_index: int,
    ball_box: Sequence[float],
    wall_line_points: Sequence[Sequence[float]],
    template_difference_path: Path | None = None,
    trajectory_break_indices: Sequence[int] = (),
) -> list[BallPositionMeasurement]:
    frames = np.stack(
        [np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for path in frame_paths]
    )
    background = np.asarray(Image.open(background_path).convert("L"), dtype=np.uint8)
    template_difference = (
        np.asarray(
            Image.open(template_difference_path).convert("L"), dtype=np.uint8
        )
        if template_difference_path is not None
        else None
    )
    return detect_ball_positions_dual(
        frames,
        timestamps,
        background,
        roi=roi,
        template_frame_index=template_frame_index,
        ball_box=ball_box,
        wall_line_points=wall_line_points,
        template_difference_frame=template_difference,
        trajectory_break_indices=trajectory_break_indices,
    )


def detect_axis_motion(
    grayscale_frames: np.ndarray,
    timestamps: Sequence[float],
    axis_point_1: tuple[float, float],
    axis_point_2: tuple[float, float],
    ball_box: Sequence[float],
) -> list[TrackPoint]:
    """Detect the moving ball's centre coordinate along a labelled track axis."""
    if grayscale_frames.ndim != 3:
        raise ValueError("灰度帧数组必须为 [frame, height, width]")
    if len(grayscale_frames) != len(timestamps) or len(timestamps) < 5:
        raise ValueError("帧数与时间戳数量不一致，或帧数不足")
    dx = axis_point_2[0] - axis_point_1[0]
    dy = axis_point_2[1] - axis_point_1[1]
    axis_length = hypot(dx, dy)
    if axis_length < 20:
        raise ValueError("轨道方向标注过短")
    ux, uy = dx / axis_length, dy / axis_length
    px, py = -uy, ux
    ball_diameter = max(float(ball_box[2]), float(ball_box[3]))
    corridor_half_width = max(14.0, ball_diameter * 0.85)

    height, width = grayscale_frames.shape[1:]
    yy, xx = np.mgrid[:height, :width]
    relative_x = xx - axis_point_1[0]
    relative_y = yy - axis_point_1[1]
    along = relative_x * ux + relative_y * uy
    across = relative_x * px + relative_y * py
    corridor = (
        (np.abs(across) <= corridor_half_width)
        & (along >= -ball_diameter)
        & (along <= axis_length + ball_diameter)
    )
    corridor_y, corridor_x = np.nonzero(corridor)
    corridor_bins = np.rint(along[corridor]).astype(int)
    minimum_bin = int(corridor_bins.min())
    shifted_bins = corridor_bins - minimum_bin
    bin_count = int(shifted_bins.max()) + 1
    background = np.median(grayscale_frames.astype(np.float32), axis=0)
    top_pixels = max(5, round(ball_diameter * 0.5))
    local_radius = max(3, round(ball_diameter * 0.45))
    points: list[TrackPoint] = []

    for timestamp, frame in zip(timestamps, grayscale_frames):
        difference = np.abs(frame.astype(np.float32) - background)
        values = difference[corridor_y, corridor_x]
        scores = np.zeros(bin_count, dtype=float)
        for bin_index in range(bin_count):
            row_values = values[shifted_bins == bin_index]
            if not len(row_values):
                continue
            count = min(top_pixels, len(row_values))
            scores[bin_index] = np.partition(row_values, -count)[-count:].sum()
        peak = int(np.argmax(scores))
        left = max(0, peak - local_radius)
        right = min(bin_count, peak + local_radius + 1)
        local_scores = scores[left:right]
        baseline = float(np.percentile(scores, 50))
        weights = np.maximum(local_scores - baseline, 0)
        if weights.sum() > 0:
            centre_bin = float(
                np.average(np.arange(left, right, dtype=float), weights=weights)
            )
        else:
            centre_bin = float(peak)
        coordinate = centre_bin + minimum_bin
        peak_score = float(scores[peak])
        confidence = max(0.0, min(1.0, (peak_score - baseline) / (peak_score + 1e-6)))
        x = axis_point_1[0] + coordinate * ux
        y = axis_point_1[1] + coordinate * uy
        points.append(TrackPoint(float(timestamp), x, y, confidence))
    return points


def track_axis_motion_from_files(
    frame_paths: Sequence[Path],
    timestamps: Sequence[float],
    axis_point_1: tuple[float, float],
    axis_point_2: tuple[float, float],
    ball_box: Sequence[float],
) -> list[TrackPoint]:
    frames = np.stack(
        [np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for path in frame_paths]
    )
    return detect_axis_motion(
        frames, timestamps, axis_point_1, axis_point_2, ball_box
    )
