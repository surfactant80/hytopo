from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np

from .cage import CageAnalysisResult, analyze_cages_for_frame
from .cpus import default_worker_count
from .config import AnalysisConfig, LABEL_UNK
from .hop import (
    HopResult,
    HopStats,
    LocalDescriptors,
    PhaseProbabilities,
    analyze_hop_from_graph,
    build_cage_candidate_mask,
)
from .neighbor import build_cage_oo_graph, build_oo_graph, set_numba_enabled
from .nucleation import (
    NucleationAnalyzer,
    create_nucleation_analyzer,
)
from .output import OutputManager
from .checkpoint import (
    clear_checkpoint,
    load_checkpoint,
    save_checkpoint,
    truncate_outputs,
)
from .progress import ProgressReporter
from .phase import PhaseTracker
from .tracking import CageTracker, TrackingParameters
from .trajectory import (
    create_context,
    init_worker,
    iter_frames,
    print_context_summary,
    read_worker_frame,
)
from .visualize import generate_all_visualizations, print_visualization_summary


@dataclass
class FramePayload:
    frame_index: int
    time_ps: float
    box: np.ndarray
    oxygen_positions: np.ndarray
    guest_positions: np.ndarray


@dataclass
class FrameAnalysisResult:
    frame_index: int
    time_ps: float
    box: np.ndarray
    oxygen_positions: np.ndarray
    guest_positions: np.ndarray
    hop_result: HopResult
    cage_result: Optional[CageAnalysisResult]


def run(config: AnalysisConfig) -> None:
    start_wall_time = time.time()

    context = create_context(config)
    print_context_summary(context, config.runtime.quiet)

    tracker: Optional[CageTracker] = create_cage_tracker(config)
    phase_tracker: Optional[PhaseTracker] = (
        PhaseTracker(config.phase)
        if config.find_cages and config.phase.enabled
        else None
    )
    nucleation: Optional[NucleationAnalyzer] = (
        create_nucleation_analyzer(config) if config.run_hop else None
    )

    checkpoint_state = None

    if getattr(config.runtime, "resume", False):
        checkpoint_state = load_checkpoint(config)

        if checkpoint_state is None:
            if not config.runtime.quiet:
                print("No checkpoint found; starting from the first frame.")
        else:
            if checkpoint_state.tracker is not None:
                tracker = checkpoint_state.tracker
            if checkpoint_state.phase_tracker is not None:
                phase_tracker = checkpoint_state.phase_tracker
            if checkpoint_state.nucleation is not None:
                nucleation = checkpoint_state.nucleation

            context.frames = [
                frame_index
                for frame_index in context.frames
                if frame_index > checkpoint_state.last_frame
            ]

            if not config.runtime.quiet:
                print(
                    "Resuming after frame "
                    f"{checkpoint_state.last_frame} "
                    f"({checkpoint_state.done_frames} frames already analysed, "
                    f"{len(context.frames)} remaining)\n"
                )

    if checkpoint_state is not None:
        truncate_outputs(checkpoint_state.offsets)

    output = OutputManager(
        config,
        context.atom_metadata,
        append=checkpoint_state is not None,
    )

    completed = False
    total_frames_analyzed = (
        checkpoint_state.done_frames if checkpoint_state is not None else 0
    )
    progress = ProgressReporter(
        total=max(
            1,
            len(context.frames) + (checkpoint_state.done_frames if checkpoint_state else 0),
        ),
        enabled=not config.runtime.quiet,
    )
    progress.done = total_frames_analyzed

    try:
        if should_use_parallel(config):
            frame_results = analyze_frames_parallel(config, context)
        else:
            frame_results = analyze_frames_serial(config, context)

        checkpoint_every = int(getattr(config.runtime, "checkpoint_every", 0))

        for result in frame_results:
            total_frames_analyzed += 1
            progress.update()

            if tracker is not None and result.cage_result is not None:
                tracker.update(
                    frame_index=result.frame_index,
                    time=result.time_ps,
                    cages=result.cage_result,
                    guest_positions=result.guest_positions,
                    oxygen_positions=result.oxygen_positions,
                    box=result.box,
                )

            if nucleation is not None:
                nucleation.update(
                    frame_index=result.frame_index,
                    time=result.time_ps,
                    nmax_hyd=int(round(result.hop_result.stats.Nmax_hyd)),
                )

            if phase_tracker is not None and result.cage_result is not None:
                phase_tracker.update(
                    frame=result.frame_index,
                    time_ps=result.time_ps,
                    cages=result.cage_result,
                    oxygen_pos=result.oxygen_positions,
                    box=result.box,
                )

            output.write_frame(
                frame_idx=result.frame_index,
                time_ps=result.time_ps,
                box=result.box,
                pos=result.oxygen_positions,
                hop_res=result.hop_result,
                cage_res=result.cage_result,
            )

            if checkpoint_every and total_frames_analyzed % checkpoint_every == 0:
                save_checkpoint(
                    config,
                    last_frame=result.frame_index,
                    done_frames=total_frames_analyzed,
                    tracker=tracker,
                    phase_tracker=phase_tracker,
                    nucleation=nucleation,
                    offsets={
                        **output.offsets(),
                        **(tracker.stream_offsets() if tracker is not None else {}),
                    },
                )

        if tracker is not None:
            tracker.finalize()
            tracker.write_outputs(config.output.outdir)

            try:
                from .network import generate_network_from_tracker

                generate_network_from_tracker(tracker, config.output.outdir)
            except Exception as e:
                if not config.runtime.quiet:
                    print(f"Note: Could not generate transition network: {e}")

        if phase_tracker is not None:
            phase_tracker.write_outputs(config.output.outdir)

        completed = True

    finally:
        progress.finish()
        output.close()

    if completed:
        clear_checkpoint(config)

    nuc_result = None
    if completed and nucleation is not None:
        nuc_result = nucleation.write_outputs(config.output.outdir)

    elapsed_wall_time = time.time() - start_wall_time
    try:
        from .summary import generate_final_summary

        generate_final_summary(
            config=config,
            tracker=tracker,
            nucleation_res=nuc_result,
            phase_tracker=phase_tracker,
            total_frames=total_frames_analyzed,
            elapsed_time=elapsed_wall_time,
        )
    except Exception as e:
        if not config.runtime.quiet:
            print(f"Note: Could not generate final summary report: {e}")

    make_plots = getattr(config.output, "make_plots", True)
    files = generate_all_visualizations(config, make_plots=make_plots)
    print_visualization_summary(files, config.runtime.quiet)

    if getattr(config.output, "write_report", True):
        try:
            from .report import generate_html_report

            report_path = generate_html_report(
                config,
                tracker=tracker,
                nucleation_res=nuc_result,
                phase_tracker=phase_tracker,
                total_frames=total_frames_analyzed,
                elapsed_time=elapsed_wall_time,
            )

            if report_path and not config.runtime.quiet:
                print(f"HTML report: {report_path}\n")
        except Exception as error:
            if not config.runtime.quiet:
                print(f"Note: HTML report skipped: {error}")


def should_use_parallel(config: AnalysisConfig) -> bool:
    if not getattr(config.runtime, "use_parallel", False):
        return False

    n_processes = resolve_n_processes(config)
    return n_processes > 1


def resolve_n_processes(config: AnalysisConfig, n_items: Optional[int] = None) -> int:
    if hasattr(config.runtime, "resolved_n_processes"):
        n_processes = int(config.runtime.resolved_n_processes())
    else:
        n_processes = default_worker_count()

    if n_items is not None:
        n_processes = min(n_processes, max(1, int(n_items)))

    return max(1, n_processes)


def resolve_chunksize(config: AnalysisConfig) -> int:
    chunksize = getattr(config.runtime, "chunksize", 1)
    if chunksize is None or chunksize <= 0:
        return 1
    return int(chunksize)


def analyze_frames_serial(
    config: AnalysisConfig,
    context: Any,
) -> Iterable[FrameAnalysisResult]:
    for frame in iter_frames(context):
        payload = make_frame_payload(frame)
        yield analyze_frame_payload(payload, config)


def analyze_frames_parallel(
    config: AnalysisConfig,
    context: Any,
) -> Iterable[FrameAnalysisResult]:
    n_processes = resolve_n_processes(config, len(context.frames))
    chunksize = resolve_chunksize(config)

    if not config.runtime.quiet:
        print(
            "Frame-level multiprocessing enabled: "
            f"n_processes={n_processes}, chunksize={chunksize}"
        )

    with ProcessPoolExecutor(
        max_workers=n_processes,
        initializer=init_analysis_worker,
        initargs=(config,),
    ) as executor:
        yield from executor.map(
            analyze_worker_frame,
            context.frames,
            chunksize=chunksize,
        )


_WORKER_CONFIG: Optional[AnalysisConfig] = None


def init_analysis_worker(config: AnalysisConfig) -> None:
    global _WORKER_CONFIG
    _WORKER_CONFIG = config
    init_worker(config)


def analyze_worker_frame(frame_index: int) -> FrameAnalysisResult:
    if _WORKER_CONFIG is None:
        raise RuntimeError("Analysis worker is not initialized.")
    frame = read_worker_frame(int(frame_index))
    return slim_result(analyze_frame_payload(make_frame_payload(frame), _WORKER_CONFIG))


def slim_result(result: FrameAnalysisResult) -> FrameAnalysisResult:
    """Drop arrays the parent does not use before sending a result over IPC.

    The probability spectrum and the local descriptors are only needed inside
    the worker (cage gating happens there); keeping them costs ~0.7 MB of
    pickling per frame on this system and far more for large ones.
    """
    hop = result.hop_result

    slim_hop = HopResult(
        stats=hop.stats,
        labels=hop.labels,
        probabilities=PhaseProbabilities(
            P_liq=EMPTY,
            P_Ih=EMPTY,
            P_Ic=EMPTY,
            P_hyd=EMPTY,
            P_pre=EMPTY,
        ),
        descriptors=LocalDescriptors(
            qtet=EMPTY,
            S=EMPTY,
            E=EMPTY,
            coordination=EMPTY,
            raw_hyd=EMPTY,
            raw_pre=EMPTY,
            hyd_support=EMPTY,
            pre_support=EMPTY,
            ice_support=EMPTY,
        ),
    )

    return FrameAnalysisResult(
        frame_index=result.frame_index,
        time_ps=result.time_ps,
        box=result.box,
        oxygen_positions=result.oxygen_positions,
        guest_positions=result.guest_positions,
        hop_result=slim_hop,
        cage_result=result.cage_result,
    )


EMPTY = np.empty(0, dtype=np.float64)


def make_frame_payload(frame: Any) -> FramePayload:
    guest_positions = frame.guest_positions
    if guest_positions is None:
        guest_positions = np.empty((0, 3), dtype=np.float64)

    return FramePayload(
        frame_index=frame.frame_index,
        time_ps=frame.time_ps,
        box=np.asarray(frame.box).copy(),
        oxygen_positions=np.asarray(frame.oxygen_positions).copy(),
        guest_positions=np.asarray(guest_positions, dtype=np.float64).copy(),
    )


def analyze_frame_payload(
    payload: FramePayload,
    config: AnalysisConfig,
) -> FrameAnalysisResult:
    set_numba_enabled(config.runtime.use_numba)

    if config.run_hop:
        hop_graph = build_oo_graph(
            positions=payload.oxygen_positions,
            box=payload.box,
            cutoff=config.hop.oo_cutoff,
        )
        hop_result = analyze_hop_from_graph(
            hop_graph,
            config.hop,
            use_numba=config.runtime.use_numba,
        )
    else:
        hop_graph = None
        hop_result = make_unclassified_hop_result(len(payload.oxygen_positions))

    cage_result: Optional[CageAnalysisResult] = None

    if config.find_cages:
        if config.run_hop:
            candidate_mask = build_cage_candidate_mask(
                labels=hop_result.labels,
                probabilities=hop_result.probabilities,
                exclude_liquid=config.cage.exclude_liquid,
                include_ice=config.cage.include_ice,
                liquid_exclude_prob=config.cage.liquid_exclude_prob,
                hyd_candidate_prob=config.cage.hyd_candidate_prob,
                ice_candidate_prob=config.cage.ice_candidate_prob,
            )
        else:
            candidate_mask = np.ones(len(payload.oxygen_positions), dtype=np.bool_)

        if hop_graph is not None and abs(config.cage.oo_cutoff - config.hop.oo_cutoff) < 1.0e-12:
            cage_graph = hop_graph
        else:
            cage_graph = build_cage_oo_graph(
                positions=payload.oxygen_positions,
                box=payload.box,
                cutoff=config.cage.oo_cutoff,
            )

        cage_result = analyze_cages_for_frame(
            frame=payload.frame_index,
            oxygen_pos=payload.oxygen_positions,
            guest_pos=payload.guest_positions,
            box=payload.box,
            global_graph=cage_graph,
            candidate_mask=candidate_mask,
            params=config.cage,
            quiet=(config.runtime.quiet or not config.runtime.verbose_guests),
        )

    return FrameAnalysisResult(
        frame_index=payload.frame_index,
        time_ps=payload.time_ps,
        box=payload.box,
        oxygen_positions=payload.oxygen_positions,
        guest_positions=payload.guest_positions,
        hop_result=hop_result,
        cage_result=cage_result,
    )


def make_unclassified_hop_result(n_atoms: int) -> HopResult:
    zeros = np.zeros(n_atoms, dtype=np.float64)
    probabilities = PhaseProbabilities(
        P_liq=zeros.copy(),
        P_Ih=zeros.copy(),
        P_Ic=zeros.copy(),
        P_hyd=zeros.copy(),
        P_pre=zeros.copy(),
    )
    descriptors = LocalDescriptors(
        qtet=zeros.copy(),
        S=zeros.copy(),
        E=zeros.copy(),
        coordination=zeros.copy(),
        raw_hyd=zeros.copy(),
        raw_pre=zeros.copy(),
        hyd_support=zeros.copy(),
        pre_support=zeros.copy(),
        ice_support=zeros.copy(),
    )
    stats = HopStats(
        H_total=0.0,
        H_crys=0.0,
        H_pre=0.0,
        Hyd_order=0.0,
        Ice=0.0,
        O_total=0.0,
        Delta=0.0,
        Nmax_hyd=0.0,
        Nmax_ice=0.0,
    )
    return HopResult(
        stats=stats,
        labels=np.full(n_atoms, LABEL_UNK, dtype=np.int32),
        probabilities=probabilities,
        descriptors=descriptors,
    )


def create_cage_tracker(config: AnalysisConfig) -> Optional[CageTracker]:
    if not config.find_cages:
        return None

    tracking_config = getattr(config, "tracking", None)

    if tracking_config is not None and not getattr(tracking_config, "enabled", True):
        return None

    params = TrackingParameters(
        enabled=get_tracking_value(tracking_config, "enabled", True),
        gap_tolerance=get_tracking_value(tracking_config, "gap_tolerance", 1),
        min_water_overlap=get_tracking_value(tracking_config, "min_water_overlap", 0.50),
        min_same_guest_overlap=get_tracking_value(
            tracking_config,
            "min_same_guest_overlap",
            0.25,
        ),
        dedup_water_overlap=get_tracking_value(
            tracking_config,
            "dedup_water_overlap",
            0.85,
        ),
        occupancy_radius=get_tracking_value(tracking_config, "occupancy_radius", 4.5),
        occupancy_center=get_tracking_value(tracking_config, "occupancy_center", "cage"),
        write_lifetime=get_tracking_value(tracking_config, "write_lifetime", True),
        write_transitions=get_tracking_value(tracking_config, "write_transitions", True),
        write_occupancy=get_tracking_value(tracking_config, "write_occupancy", True),
        lifetime_name=get_tracking_value(
            tracking_config,
            "lifetime_name",
            "cage_lifetime.dat",
        ),
        transitions_name=get_tracking_value(
            tracking_config,
            "transitions_name",
            "cage_transitions.dat",
        ),
        occupancy_name=get_tracking_value(
            tracking_config,
            "occupancy_name",
            "cage_occupancy.dat",
        ),
    )

    if not params.enabled:
        return None

    return CageTracker(params, outdir=config.output.outdir)


def get_tracking_value(source: object, name: str, default):
    if source is None:
        return default

    return getattr(source, name, default)
