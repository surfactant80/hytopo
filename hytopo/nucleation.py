from __future__ import annotations

import os
from dataclasses import dataclass
import math
from typing import Any, Iterable, Optional

import numpy as np

from .output_layout import NUCLEATION, categorised_path


@dataclass(frozen=True)
class NucleationParameters:
    enabled: bool = True
    method: str = "commitment"
    min_cluster_size: int = 50
    persist_frames: int = 10
    allow_gaps: int = 0
    baseline_frames: int = 100
    baseline_sigma: float = 6.0
    adaptive_min_size: int = 10
    commitment_min_growth: float = 5.0
    commitment_min_slope: float = 0.0
    write_summary: bool = True
    write_timeseries: bool = True
    summary_name: str = "nucleation_summary.txt"
    timeseries_name: str = "nucleation_timeseries.dat"


@dataclass(frozen=True)
class NucleationPoint:
    frame_index: int
    time: float
    nmax_hyd: int


@dataclass
class NucleationResult:
    nucleated: bool
    nucleation_frame: Optional[int]
    nucleation_time: Optional[float]
    nucleation_nmax_hyd: Optional[int]
    min_cluster_size: int
    method: str
    effective_threshold: int
    baseline_center: float
    baseline_noise: float
    persist_frames: int
    allow_gaps: int
    first_above_frame: Optional[int]
    first_above_time: Optional[float]
    max_nmax_hyd: int
    max_nmax_frame: Optional[int]
    max_nmax_time: Optional[float]
    total_frames: int
    above_threshold_frames: int
    longest_persistent_run: int
    timeseries: list[dict[str, Any]]


class NucleationAnalyzer:
    def __init__(self, params: Optional[NucleationParameters] = None) -> None:
        self.params = params if params is not None else NucleationParameters()
        self.points: list[NucleationPoint] = []

    def update(self, frame_index: int, time: float, nmax_hyd: int) -> None:
        self.points.append(
            NucleationPoint(
                frame_index=int(frame_index),
                time=float(time),
                nmax_hyd=int(nmax_hyd),
            )
        )

    def extend(self, points: Iterable[NucleationPoint]) -> None:
        for point in points:
            self.update(point.frame_index, point.time, point.nmax_hyd)

    def analyze(self) -> NucleationResult:
        return analyze_nucleation(self.points, self.params)

    def write_outputs(self, outdir: str) -> NucleationResult:
        os.makedirs(outdir, exist_ok=True)

        result = self.analyze()

        if self.params.write_summary:
            write_nucleation_summary(
                categorised_path(outdir, self.params.summary_name, NUCLEATION),
                result,
            )

        if self.params.write_timeseries:
            write_nucleation_timeseries(
                categorised_path(outdir, self.params.timeseries_name, NUCLEATION),
                result,
            )

        return result


def analyze_nucleation(
    points: list[NucleationPoint],
    params: Optional[NucleationParameters] = None,
) -> NucleationResult:
    params = params if params is not None else NucleationParameters()

    if not points:
        return NucleationResult(
            nucleated=False,
            nucleation_frame=None,
            nucleation_time=None,
            nucleation_nmax_hyd=None,
            min_cluster_size=params.min_cluster_size,
            method=params.method,
            effective_threshold=params.min_cluster_size,
            baseline_center=0.0,
            baseline_noise=0.0,
            persist_frames=params.persist_frames,
            allow_gaps=params.allow_gaps,
            first_above_frame=None,
            first_above_time=None,
            max_nmax_hyd=0,
            max_nmax_frame=None,
            max_nmax_time=None,
            total_frames=0,
            above_threshold_frames=0,
            longest_persistent_run=0,
            timeseries=[],
        )

    sorted_points = sorted(points, key=lambda item: item.frame_index)
    effective_threshold, baseline_center, baseline_noise = resolve_nucleation_threshold(
        sorted_points,
        params,
    )

    first_above_frame = None
    first_above_time = None
    max_point = max(sorted_points, key=lambda item: item.nmax_hyd)

    run_start_index: Optional[int] = None
    run_valid_count = 0
    run_gap_count = 0
    longest_run = 0

    nucleated = False
    nucleation_point: Optional[NucleationPoint] = None
    timeseries: list[dict[str, Any]] = []

    for index, point in enumerate(sorted_points):
        above = point.nmax_hyd >= effective_threshold

        if above and first_above_frame is None:
            first_above_frame = point.frame_index
            first_above_time = point.time

        if above:
            if run_start_index is None:
                run_start_index = index
                run_valid_count = 0
                run_gap_count = 0

            run_valid_count += 1
        else:
            if run_start_index is not None and run_gap_count < params.allow_gaps:
                run_gap_count += 1
            else:
                run_start_index = None
                run_valid_count = 0
                run_gap_count = 0

        longest_run = max(longest_run, run_valid_count)

        if (
            not nucleated
            and run_start_index is not None
            and run_valid_count >= params.persist_frames
            and commitment_satisfied(
                sorted_points,
                run_start_index,
                index,
                params,
            )
        ):
            nucleated = True
            nucleation_point = first_above_point_in_run(
                sorted_points,
                run_start_index,
                effective_threshold,
            )

        timeseries.append(
            {
                "frame_index": point.frame_index,
                "time": point.time,
                "nmax_hyd": point.nmax_hyd,
                "above_threshold": int(above),
                "persistent_count": run_valid_count,
                "gap_count": run_gap_count,
                "nucleated": int(nucleated),
            }
        )

    above_threshold_frames = sum(
        1 for point in sorted_points if point.nmax_hyd >= effective_threshold
    )

    return NucleationResult(
        nucleated=nucleated,
        nucleation_frame=None if nucleation_point is None else nucleation_point.frame_index,
        nucleation_time=None if nucleation_point is None else nucleation_point.time,
        nucleation_nmax_hyd=None if nucleation_point is None else nucleation_point.nmax_hyd,
        min_cluster_size=params.min_cluster_size,
        method=params.method,
        effective_threshold=effective_threshold,
        baseline_center=baseline_center,
        baseline_noise=baseline_noise,
        persist_frames=params.persist_frames,
        allow_gaps=params.allow_gaps,
        first_above_frame=first_above_frame,
        first_above_time=first_above_time,
        max_nmax_hyd=max_point.nmax_hyd,
        max_nmax_frame=max_point.frame_index,
        max_nmax_time=max_point.time,
        total_frames=len(sorted_points),
        above_threshold_frames=above_threshold_frames,
        longest_persistent_run=longest_run,
        timeseries=timeseries,
    )


def first_above_point_in_run(
    points: list[NucleationPoint],
    run_start_index: int,
    min_cluster_size: int,
) -> NucleationPoint:
    for point in points[run_start_index:]:
        if point.nmax_hyd >= min_cluster_size:
            return point

    return points[run_start_index]


def write_nucleation_summary(path: str, result: NucleationResult) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("Hydrate nucleation summary\n")
        handle.write("==========================\n\n")

        handle.write(f"nucleated              : {result.nucleated}\n")
        handle.write(f"nucleation_frame       : {format_optional(result.nucleation_frame)}\n")
        handle.write(f"nucleation_time        : {format_optional_float(result.nucleation_time)}\n")
        handle.write(f"nucleation_Nmax_hyd    : {format_optional(result.nucleation_nmax_hyd)}\n")
        handle.write("\n")

        handle.write(f"min_cluster_size       : {result.min_cluster_size}\n")
        handle.write(f"method                 : {result.method}\n")
        handle.write(f"effective_threshold    : {result.effective_threshold}\n")
        handle.write(f"baseline_center        : {result.baseline_center:.6f}\n")
        handle.write(f"baseline_noise         : {result.baseline_noise:.6f}\n")
        handle.write(f"persist_frames         : {result.persist_frames}\n")
        handle.write(f"allow_gaps             : {result.allow_gaps}\n")
        handle.write("\n")

        handle.write(f"first_above_frame      : {format_optional(result.first_above_frame)}\n")
        handle.write(f"first_above_time       : {format_optional_float(result.first_above_time)}\n")
        handle.write(f"max_Nmax_hyd           : {result.max_nmax_hyd}\n")
        handle.write(f"max_Nmax_frame         : {format_optional(result.max_nmax_frame)}\n")
        handle.write(f"max_Nmax_time          : {format_optional_float(result.max_nmax_time)}\n")
        handle.write("\n")

        handle.write(f"total_frames           : {result.total_frames}\n")
        handle.write(f"above_threshold_frames : {result.above_threshold_frames}\n")
        handle.write(f"longest_persistent_run : {result.longest_persistent_run}\n")


def write_nucleation_timeseries(path: str, result: NucleationResult) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# per-frame nucleation status\n")
        handle.write(
            "frame\ttime\tnmax_hyd\tabove_threshold\t"
            "persistent_count\tgap_count\tnucleated\n"
        )

        for row in result.timeseries:
            handle.write(
                f"{row['frame_index']}\t"
                f"{row['time']:.3f}\t"
                f"{row['nmax_hyd']}\t"
                f"{row['above_threshold']}\t"
                f"{row['persistent_count']}\t"
                f"{row['gap_count']}\t"
                f"{row['nucleated']}\n"
            )


def read_nmax_hyd_from_timeseries(path: str) -> list[NucleationPoint]:
    points: list[NucleationPoint] = []

    with open(path, "r", encoding="utf-8") as handle:
        header: Optional[list[str]] = None

        data_index = 0
        for line in handle:
            stripped = line.strip()

            if not stripped:
                continue

            if stripped.startswith("#"):
                candidate = stripped.lstrip("#").strip().split()

                if "Nmax_hyd" in candidate:
                    header = candidate

                continue

            parts = stripped.split()

            if "Nmax_hyd" in parts:
                header = parts
                continue

            if header is None:
                point = parse_legacy_timeseries_line(parts, fallback_frame=data_index)
            else:
                point = parse_named_timeseries_line(
                    header,
                    parts,
                    fallback_frame=data_index,
                )

            if point is not None:
                points.append(point)
                data_index += 1

    return points


def parse_named_timeseries_line(
    header: list[str],
    parts: list[str],
    fallback_frame: int = 0,
) -> Optional[NucleationPoint]:
    if len(parts) < len(header):
        return None

    names = {normalize_column_name(name): index for index, name in enumerate(header)}

    frame_name = first_existing_name(names, ("frame", "frame_index"))
    time_name = first_existing_name(names, ("time", "time_ps"))
    nmax_name = first_existing_name(names, ("nmax_hyd", "max_hyd_cluster"))

    if time_name is None or nmax_name is None:
        return None

    return NucleationPoint(
        frame_index=(
            int(float(parts[names[frame_name]]))
            if frame_name is not None
            else int(fallback_frame)
        ),
        time=float(parts[names[time_name]]),
        nmax_hyd=int(round(float(parts[names[nmax_name]]))),
    )


def parse_legacy_timeseries_line(
    parts: list[str],
    fallback_frame: int = 0,
) -> Optional[NucleationPoint]:
    if len(parts) < 9:
        return None

    try:
        # hytopo <= 0.1.0 wrote no frame column:
        # Time H_total H_crys H_pre Hyd_order Ice O_total Delta Nmax_hyd Nmax_ice
        time = float(parts[0])
        nmax_hyd = int(round(float(parts[8])))
    except ValueError:
        return None

    return NucleationPoint(
        frame_index=int(fallback_frame),
        time=time,
        nmax_hyd=nmax_hyd,
    )


def resolve_nucleation_threshold(
    points: list[NucleationPoint],
    params: NucleationParameters,
) -> tuple[int, float, float]:
    if params.method == "fixed":
        return params.min_cluster_size, 0.0, 0.0

    n_baseline = min(len(points), max(2, params.baseline_frames))
    values = np.asarray(
        [point.nmax_hyd for point in points[:n_baseline]],
        dtype=np.float64,
    )
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    robust_noise = 1.4826 * mad

    if robust_noise <= 1.0e-12:
        q75, q25 = np.percentile(values, [75.0, 25.0])
        robust_noise = float(max(0.0, (q75 - q25) / 1.349))

    threshold = max(
        params.adaptive_min_size,
        int(math.ceil(center + params.baseline_sigma * max(robust_noise, 1.0))),
    )
    return threshold, center, robust_noise


def commitment_satisfied(
    points: list[NucleationPoint],
    start_index: int,
    end_index: int,
    params: NucleationParameters,
) -> bool:
    if params.method != "commitment":
        return True

    start = points[start_index]
    end = points[end_index]
    growth = float(end.nmax_hyd - start.nmax_hyd)
    delta_time = float(end.time - start.time)
    slope = growth / delta_time if delta_time > 0.0 else growth
    return (
        growth >= params.commitment_min_growth
        and slope >= params.commitment_min_slope
    )


def normalize_column_name(name: str) -> str:
    normalized = name.strip().lstrip("#").lower()
    normalized = normalized.replace("(ps)", "").replace("[ps]", "")
    return normalized


def first_existing_name(names: dict[str, int], candidates: tuple[str, ...]) -> Optional[str]:
    for candidate in candidates:
        if candidate in names:
            return candidate

    return None


def format_optional(value: Optional[int]) -> str:
    if value is None:
        return "None"

    return str(value)


def format_optional_float(value: Optional[float]) -> str:
    if value is None:
        return "None"

    return f"{value:.6f}"


def create_nucleation_analyzer(config: Any) -> Optional[NucleationAnalyzer]:
    nucleation_config = getattr(config, "nucleation", None)

    if nucleation_config is not None and not getattr(nucleation_config, "enabled", True):
        return None

    params = NucleationParameters(
        enabled=get_config_value(nucleation_config, "enabled", True),
        method=get_config_value(nucleation_config, "method", "commitment"),
        min_cluster_size=get_config_value(nucleation_config, "min_cluster_size", 50),
        persist_frames=get_config_value(nucleation_config, "persist_frames", 10),
        allow_gaps=get_config_value(nucleation_config, "allow_gaps", 0),
        baseline_frames=get_config_value(nucleation_config, "baseline_frames", 100),
        baseline_sigma=get_config_value(nucleation_config, "baseline_sigma", 6.0),
        adaptive_min_size=get_config_value(nucleation_config, "adaptive_min_size", 10),
        commitment_min_growth=get_config_value(
            nucleation_config,
            "commitment_min_growth",
            5.0,
        ),
        commitment_min_slope=get_config_value(
            nucleation_config,
            "commitment_min_slope",
            0.0,
        ),
        write_summary=get_config_value(nucleation_config, "write_summary", True),
        write_timeseries=get_config_value(nucleation_config, "write_timeseries", True),
        summary_name=get_config_value(nucleation_config, "summary_name", "nucleation_summary.txt"),
        timeseries_name=get_config_value(nucleation_config, "timeseries_name", "nucleation_timeseries.dat"),
    )

    if not params.enabled:
        return None

    return NucleationAnalyzer(params)


def get_config_value(source: Any, name: str, default: Any) -> Any:
    if source is None:
        return default

    return getattr(source, name, default)
