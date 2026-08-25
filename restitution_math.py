"""Pure geometry and fitting helpers for restitution estimation."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, sqrt
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class TrackPoint:
    timestamp: float
    x: float
    y: float
    confidence: float = 1.0


def upward_floor_normal(
    point_1: tuple[float, float], point_2: tuple[float, float]
) -> tuple[float, float]:
    """Return a unit normal that points toward the top of an image."""
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]
    length = hypot(dx, dy)
    if length < 1.0:
        raise ValueError("地板标注的两个点距离太近")

    nx, ny = dy / length, -dx / length
    if ny > 0:
        nx, ny = -nx, -ny
    return nx, ny


def normal_coordinate(
    point: tuple[float, float],
    floor_point: tuple[float, float],
    normal: tuple[float, float],
) -> float:
    """Project an image point onto the floor's upward normal."""
    return (
        (point[0] - floor_point[0]) * normal[0]
        + (point[1] - floor_point[1]) * normal[1]
    )


def axis_coordinate(
    point: tuple[float, float],
    axis_point_1: tuple[float, float],
    axis_point_2: tuple[float, float],
) -> float:
    """Project an image point onto a labelled positive track direction."""
    dx = axis_point_2[0] - axis_point_1[0]
    dy = axis_point_2[1] - axis_point_1[1]
    length = hypot(dx, dy)
    if length < 1.0:
        raise ValueError("轨道方向的两个标注点距离太近")
    ux, uy = dx / length, dy / length
    return (point[0] - axis_point_1[0]) * ux + (point[1] - axis_point_1[1]) * uy


def wall_unit_normal(
    point_1: tuple[float, float], point_2: tuple[float, float]
) -> tuple[float, float]:
    """Return either unit normal of a labelled fixed wall line."""
    dx = point_2[0] - point_1[0]
    dy = point_2[1] - point_1[1]
    length = hypot(dx, dy)
    if length < 1.0:
        raise ValueError("挡墙标注的两个点距离太近")
    return -dy / length, dx / length


def _fit_linear_velocity(
    points: Sequence[TrackPoint],
    axis_point_1: tuple[float, float],
    axis_point_2: tuple[float, float],
) -> float:
    times = np.asarray([point.timestamp for point in points], dtype=float)
    positions = np.asarray(
        [
            axis_coordinate((point.x, point.y), axis_point_1, axis_point_2)
            for point in points
        ],
        dtype=float,
    )
    times = times - times.mean()
    return float(np.polyfit(times, positions, 1)[0])


def estimate_single_body_restitution_along_axis(
    points: Sequence[TrackPoint],
    axis_point_1: tuple[float, float],
    axis_point_2: tuple[float, float],
    contact_time: float,
    separation_time: float,
    *,
    fit_frames: int = 8,
    minimum_confidence: float = 0.35,
) -> tuple[float, float, float]:
    """Estimate restitution for one ball colliding with a fixed end stop."""
    if fit_frames < 3:
        raise ValueError("fit_frames 至少为 3")
    if separation_time <= contact_time:
        raise ValueError("完全分离时间必须晚于首次接触时间")
    valid = [point for point in points if point.confidence >= minimum_confidence]
    pre = [point for point in valid if point.timestamp < contact_time][-fit_frames:]
    post = [point for point in valid if point.timestamp >= separation_time][:fit_frames]
    if len(pre) < fit_frames:
        raise ValueError(
            f"首次接触前有效跟踪帧不足：需要 {fit_frames} 帧，实际 {len(pre)} 帧"
        )
    if len(post) < fit_frames:
        raise ValueError(
            f"完全分离后有效跟踪帧不足：需要 {fit_frames} 帧，实际 {len(post)} 帧"
        )
    velocity_before = _fit_linear_velocity(pre, axis_point_1, axis_point_2)
    velocity_after = _fit_linear_velocity(post, axis_point_1, axis_point_2)
    if velocity_before * velocity_after >= 0:
        raise ValueError("碰撞前后速度没有反向，请检查轨道轴、事件帧或跟踪结果")
    if abs(velocity_before) < 1e-6:
        raise ValueError("碰撞前速度接近 0，无法计算恢复系数")
    restitution = abs(velocity_after / velocity_before)
    if not 0.0 <= restitution <= 1.5:
        raise ValueError(f"恢复系数结果异常：{restitution:.3f}")
    return restitution, velocity_before, velocity_after


def estimate_fixed_wall_restitution_multiwindow(
    points: Sequence[TrackPoint],
    wall_point_1: tuple[float, float],
    wall_point_2: tuple[float, float],
    contact_time: float,
    separation_time: float,
    *,
    window_sizes: Sequence[int] = (5, 7, 9),
    minimum_confidence: float = 0.35,
    maximum_relative_window_spread: float = 0.10,
) -> dict[str, object]:
    """Estimate fixed-wall COR from several wall-normal velocity windows.

    Positions and velocities remain in image pixels.  The common local spatial
    scale cancels in the velocity ratio, while timestamps are the supplied
    presentation timestamps rather than a nominal frame interval.
    """
    sizes = tuple(int(size) for size in window_sizes)
    if not sizes or any(size < 3 for size in sizes):
        raise ValueError("每个拟合窗口至少需要 3 帧")
    if len(set(sizes)) != len(sizes):
        raise ValueError("拟合窗口大小不能重复")
    if maximum_relative_window_spread < 0:
        raise ValueError("窗口稳定性阈值不能为负数")

    nx, ny = wall_unit_normal(wall_point_1, wall_point_2)
    normal_axis_end = (wall_point_1[0] + nx, wall_point_1[1] + ny)
    window_results: list[dict[str, float | int]] = []
    for fit_frames in sizes:
        restitution, velocity_before, velocity_after = (
            estimate_single_body_restitution_along_axis(
                points,
                wall_point_1,
                normal_axis_end,
                contact_time,
                separation_time,
                fit_frames=fit_frames,
                minimum_confidence=minimum_confidence,
            )
        )
        valid = [point for point in points if point.confidence >= minimum_confidence]
        pre = [point for point in valid if point.timestamp < contact_time][-fit_frames:]
        post = [point for point in valid if point.timestamp >= separation_time][
            :fit_frames
        ]
        tangent_before = _fit_linear_velocity(pre, wall_point_1, wall_point_2)
        tangent_after = _fit_linear_velocity(post, wall_point_1, wall_point_2)
        incidence_angle = degrees(
            atan2(abs(tangent_before), max(abs(velocity_before), 1e-12))
        )
        window_results.append(
            {
                "fit_frames": fit_frames,
                "coefficient_of_restitution": restitution,
                "velocity_before_normal_pixel_per_s": velocity_before,
                "velocity_after_normal_pixel_per_s": velocity_after,
                "velocity_before_tangent_pixel_per_s": tangent_before,
                "velocity_after_tangent_pixel_per_s": tangent_after,
                "incidence_angle_degrees": incidence_angle,
            }
        )

    values = np.asarray(
        [item["coefficient_of_restitution"] for item in window_results],
        dtype=float,
    )
    final = float(np.median(values))
    uncertainty = float(1.4826 * np.median(np.abs(values - final)))
    relative_spread = float((values.max() - values.min()) / max(abs(final), 1e-12))
    incidence_angle = float(
        np.median([item["incidence_angle_degrees"] for item in window_results])
    )
    quality_status = (
        "ok"
        if relative_spread <= maximum_relative_window_spread
        else "review_required"
    )
    return {
        "coefficient_of_restitution": final,
        "uncertainty": uncertainty,
        "window_relative_spread": relative_spread,
        "quality_status": quality_status,
        "incidence_angle_degrees": incidence_angle,
        "collision_geometry_status": (
            "normal_collision"
            if incidence_angle <= 5.0
            else "non_normal_collision"
        ),
        "window_results": window_results,
    }


def estimate_two_body_restitution(
    ball_1_points: Sequence[TrackPoint],
    ball_2_points: Sequence[TrackPoint],
    axis_point_1: tuple[float, float],
    axis_point_2: tuple[float, float],
    contact_time: float,
    separation_time: float,
    *,
    fit_frames: int = 8,
    minimum_confidence: float = 0.35,
) -> tuple[float, float, float, float, float]:
    """Estimate 1-D restitution from two balls' relative velocities.

    Returns ``(e, u1, u2, v1, v2)`` in pixels per presentation-timeline
    second along the labelled track axis.  The common spatial and time scales
    cancel in ``abs((v2-v1)/(u1-u2))``.
    """
    if fit_frames < 3:
        raise ValueError("fit_frames 至少为 3")
    if separation_time <= contact_time:
        raise ValueError("完全分离时间必须晚于首次接触时间")

    def branch(
        points: Sequence[TrackPoint], before: bool
    ) -> list[TrackPoint]:
        valid = [point for point in points if point.confidence >= minimum_confidence]
        if before:
            return [point for point in valid if point.timestamp < contact_time][
                -fit_frames:
            ]
        return [point for point in valid if point.timestamp >= separation_time][
            :fit_frames
        ]

    pre_1, pre_2 = branch(ball_1_points, True), branch(ball_2_points, True)
    post_1, post_2 = branch(ball_1_points, False), branch(ball_2_points, False)
    for name, values in (
        ("1号球碰撞前", pre_1),
        ("2号球碰撞前", pre_2),
        ("1号球碰撞后", post_1),
        ("2号球碰撞后", post_2),
    ):
        if len(values) < fit_frames:
            raise ValueError(
                f"{name}有效跟踪帧不足：需要 {fit_frames} 帧，实际 {len(values)} 帧"
            )

    u1 = _fit_linear_velocity(pre_1, axis_point_1, axis_point_2)
    u2 = _fit_linear_velocity(pre_2, axis_point_1, axis_point_2)
    v1 = _fit_linear_velocity(post_1, axis_point_1, axis_point_2)
    v2 = _fit_linear_velocity(post_2, axis_point_1, axis_point_2)
    approach_speed = abs(u1 - u2)
    separation_speed = abs(v2 - v1)
    if approach_speed < 1e-6:
        raise ValueError("碰撞前两球相对速度接近 0，无法计算恢复系数")
    restitution = separation_speed / approach_speed
    if not 0.0 <= restitution <= 1.5:
        raise ValueError(f"恢复系数结果异常：{restitution:.3f}")
    return restitution, u1, u2, v1, v2


def _joint_piecewise_velocities(
    pre_times: np.ndarray,
    pre_positions: np.ndarray,
    post_times: np.ndarray,
    post_positions: np.ndarray,
    impact_time: float,
) -> tuple[float, float]:
    """Fit two trajectories with separate velocities and shared acceleration."""
    times = np.concatenate((pre_times, post_times)) - impact_time
    positions = np.concatenate((pre_positions, post_positions))
    is_pre = np.concatenate(
        (np.ones(len(pre_times), dtype=float), np.zeros(len(post_times), dtype=float))
    )
    is_post = 1.0 - is_pre
    design = np.column_stack(
        (
            is_pre,
            is_post,
            is_pre * times,
            is_post * times,
            0.5 * times * times,
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, positions, rcond=None)
    return float(coefficients[2]), float(coefficients[3])


def _joint_event_velocities(
    pre_times: np.ndarray,
    pre_positions: np.ndarray,
    post_times: np.ndarray,
    post_positions: np.ndarray,
    contact_time: float,
    separation_time: float,
) -> tuple[float, float]:
    """Fit velocities at the two human-labelled contact boundaries.

    The pre-contact velocity is evaluated at ``contact_time`` and the
    post-contact velocity at ``separation_time``. Both free-flight branches
    share one acceleration, while their intercepts remain independent so the
    deformed contact interval never enters the fit.
    """
    pre_dt = pre_times - contact_time
    post_dt = post_times - separation_time
    is_pre = np.concatenate(
        (np.ones(len(pre_times), dtype=float), np.zeros(len(post_times), dtype=float))
    )
    is_post = 1.0 - is_pre
    branch_dt = np.concatenate((pre_dt, post_dt))
    positions = np.concatenate((pre_positions, post_positions))
    design = np.column_stack(
        (
            is_pre,
            is_post,
            is_pre * branch_dt,
            is_post * branch_dt,
            0.5 * branch_dt * branch_dt,
        )
    )
    coefficients, _, _, _ = np.linalg.lstsq(design, positions, rcond=None)
    return float(coefficients[2]), float(coefficients[3])


def candidate_impact_index(
    points: Sequence[TrackPoint],
    floor_point_1: tuple[float, float],
    floor_point_2: tuple[float, float],
    *,
    minimum_confidence: float = 0.35,
) -> int:
    """Return the original point index near the first clear bounce.

    This is only a navigation hint for the annotation UI. It is deliberately
    not treated as an automatically detected contact frame.  The earliest
    clear falling-to-rising turn is preferred over the global lowest centre,
    because a ball resting on the floor later in the video can sit lower than
    its centre during the first impact.
    """
    normal = upward_floor_normal(floor_point_1, floor_point_2)
    candidates = [
        (
            index,
            point.timestamp,
            normal_coordinate((point.x, point.y), floor_point_1, normal),
        )
        for index, point in enumerate(points)
        if point.confidence >= minimum_confidence
    ]
    if not candidates:
        raise ValueError("没有达到置信度要求的足球跟踪点")
    if len(candidates) < 3:
        return min(candidates, key=lambda item: item[2])[0]

    heights = [candidate[2] for candidate in candidates]
    vertical_range = max(heights) - min(heights)
    minimum_excursion = max(2.0, vertical_range * 0.05)
    time_deltas = [
        later[1] - earlier[1]
        for earlier, later in zip(candidates, candidates[1:])
        if later[1] > earlier[1]
    ]
    median_delta = float(np.median(time_deltas)) if time_deltas else 1 / 30
    local_radius = max(1, round(0.02 / median_delta))
    excursion_window_s = 0.25

    for position in range(1, len(candidates) - 1):
        _, timestamp, height = candidates[position]
        local_before = heights[max(0, position - local_radius) : position]
        local_after = heights[
            position + 1 : min(len(heights), position + local_radius + 1)
        ]
        if not local_before or not local_after:
            continue
        if height > min((*local_before, *local_after)):
            continue

        before = [
            candidate_height
            for _, candidate_time, candidate_height in candidates[:position]
            if timestamp - candidate_time <= excursion_window_s
        ]
        after = [
            candidate_height
            for _, candidate_time, candidate_height in candidates[position + 1 :]
            if candidate_time - timestamp <= excursion_window_s
        ]
        if not before or not after:
            continue
        if (
            max(before) - height >= minimum_excursion
            and max(after) - height >= minimum_excursion
        ):
            return candidates[position][0]

    return min(candidates, key=lambda item: item[2])[0]


def event_window_indices(
    points: Sequence[TrackPoint],
    centre_index: int,
    *,
    seconds_each_side: float = 0.5,
) -> tuple[int, int]:
    """Return a timestamp-based ``[start, end)`` event annotation window."""
    if not points:
        raise ValueError("轨迹为空，无法生成接触事件候选范围")
    if not 0 <= centre_index < len(points):
        raise ValueError("候选接触帧超出轨迹范围")
    if seconds_each_side <= 0:
        raise ValueError("候选时间窗口必须大于 0 秒")

    centre_time = points[centre_index].timestamp
    start_time = centre_time - seconds_each_side
    end_time = centre_time + seconds_each_side
    start = next(
        (index for index, point in enumerate(points) if point.timestamp >= start_time),
        len(points) - 1,
    )
    end = next(
        (index for index, point in enumerate(points) if point.timestamp > end_time),
        len(points),
    )
    return start, end


def estimate_restitution_from_events(
    points: Sequence[TrackPoint],
    floor_point_1: tuple[float, float],
    floor_point_2: tuple[float, float],
    contact_time: float,
    separation_time: float,
    *,
    fit_frames: int = 10,
    minimum_confidence: float = 0.35,
) -> tuple[float, float, float]:
    """Estimate restitution from human-labelled contact boundaries.

    Exactly ``fit_frames`` valid free-flight observations immediately before
    first contact and immediately after full separation are used. Frames in
    the labelled contact interval are excluded completely.
    """
    if fit_frames < 3:
        raise ValueError("fit_frames 至少为 3")
    if separation_time <= contact_time:
        raise ValueError("完全离地时间必须晚于首次接触时间")

    valid = [point for point in points if point.confidence >= minimum_confidence]
    pre = [point for point in valid if point.timestamp < contact_time][-fit_frames:]
    post = [point for point in valid if point.timestamp >= separation_time][
        :fit_frames
    ]
    if len(pre) < fit_frames:
        raise ValueError(
            f"首次接触前的有效跟踪帧不足：需要 {fit_frames} 帧，实际 {len(pre)} 帧"
        )
    if len(post) < fit_frames:
        raise ValueError(
            f"完全离地后的有效跟踪帧不足：需要 {fit_frames} 帧，实际 {len(post)} 帧"
        )

    normal = upward_floor_normal(floor_point_1, floor_point_2)
    pre_times = np.asarray([point.timestamp for point in pre], dtype=float)
    post_times = np.asarray([point.timestamp for point in post], dtype=float)
    pre_positions = np.asarray(
        [normal_coordinate((point.x, point.y), floor_point_1, normal) for point in pre],
        dtype=float,
    )
    post_positions = np.asarray(
        [
            normal_coordinate((point.x, point.y), floor_point_1, normal)
            for point in post
        ],
        dtype=float,
    )
    velocity_before, velocity_after = _joint_event_velocities(
        pre_times,
        pre_positions,
        post_times,
        post_positions,
        contact_time,
        separation_time,
    )

    if velocity_before >= 0:
        raise ValueError("触地前速度方向异常，请检查首次接触帧、足球框或地板线")
    if velocity_after <= 0:
        raise ValueError("离地后速度方向异常，请检查完全离地帧或跟踪结果")

    restitution = abs(velocity_after / velocity_before)
    if not 0.0 <= restitution <= 1.5:
        raise ValueError(f"恢复系数结果异常：{restitution:.3f}")
    return restitution, velocity_before, velocity_after


def estimate_restitution_from_heights(
    points: Sequence[TrackPoint],
    floor_point_1: tuple[float, float],
    floor_point_2: tuple[float, float],
    contact_time: float,
    separation_time: float,
    *,
    minimum_confidence: float = 0.35,
    peak_samples: int = 3,
    descent_confirmation_frames: int = 3,
    minimum_descent_pixels: float = 3.0,
) -> tuple[float, float, float, int, int]:
    """Estimate restitution from release and first-rebound heights.

    Heights are measured along the floor normal in pixels. The common spatial
    scale cancels in ``sqrt(rebound_height / drop_height)``. The release level
    is referenced to the manually labelled first-contact centre position; the
    rebound level is referenced to the manually labelled first-separated centre
    position. Frames inside the contact interval are never peak candidates.
    """
    if separation_time <= contact_time:
        raise ValueError("完全离地时间必须晚于首次接触时间")
    if peak_samples < 1:
        raise ValueError("peak_samples 至少为 1")
    if descent_confirmation_frames < 1:
        raise ValueError("descent_confirmation_frames 至少为 1")

    valid = [
        (index, point)
        for index, point in enumerate(points)
        if point.confidence >= minimum_confidence
    ]
    pre = [(index, point) for index, point in valid if point.timestamp < contact_time]
    post = [
        (index, point) for index, point in valid if point.timestamp >= separation_time
    ]
    if len(pre) < peak_samples:
        raise ValueError(
            f"首次接触前的有效跟踪帧不足：至少需要 {peak_samples} 帧"
        )
    if len(post) < peak_samples:
        raise ValueError(
            f"完全离地后的有效跟踪帧不足：至少需要 {peak_samples} 帧"
        )

    normal = upward_floor_normal(floor_point_1, floor_point_2)

    def height(point: TrackPoint) -> float:
        return normal_coordinate((point.x, point.y), floor_point_1, normal)

    release_index, release_point = max(pre, key=lambda item: height(item[1]))
    apex_position = max(range(len(post)), key=lambda position: height(post[position][1]))
    apex_index, apex_point = post[apex_position]
    descent_end = apex_position + descent_confirmation_frames + 1
    if descent_end > len(post):
        raise ValueError("分析范围没有覆盖第一次反弹最高点后的下降过程，请增大 --end-time")
    after_apex_heights = [
        height(point) for _, point in post[apex_position + 1 : descent_end]
    ]
    if height(apex_point) - min(after_apex_heights) < minimum_descent_pixels:
        raise ValueError("尚未确认足球越过第一次反弹最高点，请增大 --end-time")
    contact_index, contact_point = min(
        valid, key=lambda item: abs(item[1].timestamp - contact_time)
    )
    separation_index, separation_point = min(
        valid, key=lambda item: abs(item[1].timestamp - separation_time)
    )
    if contact_index >= separation_index:
        raise ValueError("接触与离地标注没有对应到正确的帧顺序")

    pre_heights = sorted((height(point) for _, point in pre), reverse=True)
    post_heights = sorted((height(point) for _, point in post), reverse=True)
    release_level = float(np.median(pre_heights[:peak_samples]))
    apex_level = float(np.median(post_heights[:peak_samples]))
    drop_height = release_level - height(contact_point)
    rebound_height = apex_level - height(separation_point)
    if drop_height <= 0:
        raise ValueError("释放高度不是正值，请扩大开始时间或检查首次接触帧")
    if rebound_height <= 0:
        raise ValueError("第一次反弹高度不是正值，请扩大结束时间或检查完全离地帧")

    restitution = sqrt(rebound_height / drop_height)
    if not 0.0 <= restitution <= 1.5:
        raise ValueError(f"恢复系数结果异常：{restitution:.3f}")
    return restitution, drop_height, rebound_height, release_index, apex_index


def estimate_restitution_from_height_events(
    points: Sequence[TrackPoint],
    floor_point_1: tuple[float, float],
    floor_point_2: tuple[float, float],
    release_time: float,
    contact_time: float,
    separation_time: float,
    apex_time: float,
    *,
    minimum_confidence: float = 0.35,
) -> tuple[float, float, float, int, int]:
    """Estimate restitution from four human-confirmed event frames."""
    if not release_time < contact_time < separation_time < apex_time:
        raise ValueError("事件顺序必须为：释放、首次接触、完全离地、反弹最高点")
    valid = [
        (index, point)
        for index, point in enumerate(points)
        if point.confidence >= minimum_confidence
    ]
    if len(valid) < 4:
        raise ValueError("有效跟踪帧不足")

    def nearest(timestamp: float) -> tuple[int, TrackPoint]:
        return min(valid, key=lambda item: abs(item[1].timestamp - timestamp))

    release_index, release_point = nearest(release_time)
    contact_index, contact_point = nearest(contact_time)
    separation_index, separation_point = nearest(separation_time)
    apex_index, apex_point = nearest(apex_time)
    if not release_index < contact_index < separation_index < apex_index:
        raise ValueError("人工事件时间没有映射到正确的跟踪帧顺序")

    normal = upward_floor_normal(floor_point_1, floor_point_2)

    def height(point: TrackPoint) -> float:
        return normal_coordinate((point.x, point.y), floor_point_1, normal)

    drop_height = height(release_point) - height(contact_point)
    rebound_height = height(apex_point) - height(separation_point)
    if drop_height <= 0:
        raise ValueError("释放高度不是正值，请检查释放帧或首次接触帧")
    if rebound_height <= 0:
        raise ValueError("第一次反弹高度不是正值，请检查完全离地帧或最高点帧")
    restitution = sqrt(rebound_height / drop_height)
    if not 0.0 <= restitution <= 1.5:
        raise ValueError(f"恢复系数结果异常：{restitution:.3f}")
    return restitution, drop_height, rebound_height, release_index, apex_index


def estimate_restitution(
    points: Sequence[TrackPoint],
    floor_point_1: tuple[float, float],
    floor_point_2: tuple[float, float],
    *,
    fit_frames: int = 10,
    contact_padding_frames: int = 4,
    post_contact_padding_frames: int = 10,
    edge_trim_frames: int = 1,
    minimum_confidence: float = 0.35,
) -> tuple[float, int, float, float]:
    """Estimate e with a physics-constrained piecewise trajectory fit.

    Returns ``(e, impact_index, velocity_before, velocity_after)``. Velocities
    use pixels per presentation-timeline second. Their common spatial and time
    scale cancels in the ratio. Pre- and post-impact trajectories have separate
    intercepts and velocities but share one constant acceleration. Separate
    intercepts absorb the football's visible deformation during contact.
    """
    if fit_frames < 3:
        raise ValueError("fit_frames 至少为 3")
    if contact_padding_frames < 1:
        raise ValueError("contact_padding_frames 至少为 1")
    if post_contact_padding_frames < 1:
        raise ValueError("post_contact_padding_frames 至少为 1")
    if edge_trim_frames < 0:
        raise ValueError("edge_trim_frames 不能为负数")

    valid = [point for point in points if point.confidence >= minimum_confidence]
    required = (
        2 * fit_frames
        + contact_padding_frames
        + post_contact_padding_frames
        + 2 * edge_trim_frames
        + 1
    )
    if len(valid) < required:
        raise ValueError(f"有效跟踪帧不足：需要至少 {required} 帧，实际 {len(valid)} 帧")

    normal = upward_floor_normal(floor_point_1, floor_point_2)
    positions = np.asarray(
        [
            normal_coordinate((point.x, point.y), floor_point_1, normal)
            for point in valid
        ],
        dtype=float,
    )
    times = np.asarray([point.timestamp for point in valid], dtype=float)

    left_margin = fit_frames + contact_padding_frames + edge_trim_frames
    right_margin = fit_frames + post_contact_padding_frames + edge_trim_frames
    candidate_positions = positions[left_margin:-right_margin]
    if len(candidate_positions) == 0:
        raise ValueError("碰撞窗口太短，无法在两侧拟合速度")
    impact_index = int(np.argmin(candidate_positions)) + left_margin
    impact_time = float(times[impact_index])

    pre_end = impact_index - contact_padding_frames
    pre_start = edge_trim_frames
    post_start = impact_index + post_contact_padding_frames + 1
    post_end = len(valid) - edge_trim_frames
    if pre_end - pre_start < fit_frames or post_end - post_start < fit_frames:
        raise ValueError("触地点过于接近分析窗口边缘")

    velocity_before, velocity_after = _joint_piecewise_velocities(
        times[pre_start:pre_end],
        positions[pre_start:pre_end],
        times[post_start:post_end],
        positions[post_start:post_end],
        impact_time,
    )

    if velocity_before >= 0:
        raise ValueError("触地前速度方向异常，请检查足球框或地板线")
    if velocity_after <= 0:
        raise ValueError("触地后速度方向异常，请检查跟踪是否丢失")

    restitution = abs(velocity_after / velocity_before)
    if not 0.0 <= restitution <= 1.5:
        raise ValueError(f"恢复系数结果异常：{restitution:.3f}")
    return restitution, impact_index, velocity_before, velocity_after
