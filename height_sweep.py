#!/usr/bin/env python3
"""Summarize one H1–H5 fixed-wall restitution screening group."""

from __future__ import annotations

import argparse
import json
from math import sqrt
from pathlib import Path
from statistics import mean, variance
from typing import Mapping, Sequence


RELEASE_LEVELS = ("H1", "H2", "H3", "H4", "H5")


def summarize_height_sweep(
    trials: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return descriptive statistics without deciding cross-height uniformity."""
    by_level: dict[str, Mapping[str, object]] = {}
    for trial in trials:
        level = str(trial.get("release_level", ""))
        if level not in RELEASE_LEVELS:
            raise ValueError(f"未知释放等级：{level or '未填写'}")
        if level in by_level:
            raise ValueError(f"释放等级 {level} 出现重复结果")
        by_level[level] = trial
    if set(by_level) != set(RELEASE_LEVELS):
        raise ValueError("一组统计必须包含 H1–H5 各一个有效结果")

    ordered: list[dict[str, object]] = []
    values: list[float] = []
    for level in RELEASE_LEVELS:
        source = by_level[level]
        if (
            "standard_statistics_eligible" in source
            and not bool(source["standard_statistics_eligible"])
        ):
            raise ValueError(f"{level} 已被排除，不能进入标准统计")
        value = float(source["coefficient_of_restitution"])
        if not 0.0 <= value <= 1.5:
            raise ValueError(f"{level} 的恢复系数超出合理检查范围")
        uncertainty = float(source.get("uncertainty", 0.0))
        if uncertainty < 0:
            raise ValueError(f"{level} 的不确定性不能为负数")
        item = dict(source)
        item["release_level"] = level
        item["coefficient_of_restitution"] = value
        item["uncertainty"] = uncertainty
        ordered.append(item)
        values.append(value)

    average = mean(values)
    sample_variance = variance(values)
    standard_deviation = sqrt(sample_variance)
    coefficient_of_variation = standard_deviation / average if average else None
    level_mean = 3.0
    trend_slope = sum(
        (index - level_mean) * (value - average)
        for index, value in enumerate(values, start=1)
    ) / sum((index - level_mean) ** 2 for index in range(1, 6))
    differences = [right - left for left, right in zip(values, values[1:])]
    if all(change > 0 for change in differences):
        monotonic_trend = "increasing"
    elif all(change < 0 for change in differences):
        monotonic_trend = "decreasing"
    else:
        monotonic_trend = "none"

    return {
        "release_levels": list(RELEASE_LEVELS),
        "trials": ordered,
        "representative_coefficient": average,
        "mean": average,
        "variance": sample_variance,
        "variance_definition": "sample_variance_n_minus_1",
        "standard_deviation": standard_deviation,
        "coefficient_of_variation": coefficient_of_variation,
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
        "trend_slope_per_level": trend_slope,
        "monotonic_trend": monotonic_trend,
        "uniformity_decision": None,
        "uniformity_decision_basis": "manual_review_required_for_first_group",
    }


def _trial_from_annotation(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "release_level": payload["release_level"],
        "coefficient_of_restitution": payload["coefficient_of_restitution"],
        "uncertainty": payload.get("coefficient_of_restitution_uncertainty", 0.0),
        "quality_status": payload.get("quality_status", "unknown"),
        "standard_statistics_eligible": payload.get(
            "standard_statistics_eligible", False
        ),
        "annotation_file": str(path.resolve()),
        "video": payload.get("video"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总一组 H1–H5 撞墙恢复系数")
    parser.add_argument("annotations", nargs=5, type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        trials = [_trial_from_annotation(path.expanduser()) for path in args.annotations]
        summary = summarize_height_sweep(trials)
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"代表性恢复系数（待人工确认） = {summary['mean']:.4f}")
        print(f"标准差 = {summary['standard_deviation']:.4f}")
        coefficient_of_variation = summary["coefficient_of_variation"]
        if coefficient_of_variation is None:
            print("变异系数 = 未定义（平均值为 0）")
        else:
            print(f"变异系数 = {coefficient_of_variation:.2%}")
        print(f"H1→H5 单调趋势 = {summary['monotonic_trend']}")
        print(f"统计文件：{output}")
        return 0
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"错误：{error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
