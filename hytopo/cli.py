#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Optional

from .config import (
    AnalysisConfig,
    CageParameters,
    HopParameters,
    NucleationParameters,
    OutputConfig,
    RuntimeConfig,
    TrackingConfig,
    TrajectoryConfig,
)
from .phase import PhaseParameters


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive float")
    return parsed


def probability(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hytopo",
        description=(
            "Unified hydrate order-parameter and hydrate cage analyzer. "
            "HOP water classification runs first. Cage search is optional "
            "and can automatically exclude liquid-like water molecules."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_input_arguments(parser)
    add_frame_arguments(parser)
    add_analysis_mode_arguments(parser)
    add_output_arguments(parser)
    add_hop_arguments(parser)
    add_cage_arguments(parser)
    add_tracking_arguments(parser)
    add_phase_arguments(parser)
    add_nucleation_arguments(parser)
    add_performance_arguments(parser)
    add_misc_arguments(parser)

    return parser


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("input")

    group.add_argument(
        "-top",
        "--top",
        required=True,
        help="Topology/structure file or periodic EXTXYZ trajectory (.xyz/.extxyz/.etxyz).",
    )
    group.add_argument(
        "-traj",
        "--traj",
        default=None,
        help="Trajectory file, e.g. .xtc, .trr, or EXTXYZ. If omitted, only topology/structure is loaded.",
    )
    group.add_argument(
        "-water",
        "--water-selection",
        default="resname SOL",
        help="Water selection; for EXTXYZ use element/name/species selections.",
    )
    group.add_argument(
        "-oxygen",
        "--oxygen-selection",
        default="name OW",
        help="Water oxygen atom selection; e.g. 'element O' for EXTXYZ.",
    )
    group.add_argument(
        "-guest",
        "--guest-selection",
        default=None,
        help='Guest atom/molecule selection, e.g. "resname CH4" or EXTXYZ "element C".',
    )
    group.add_argument(
        "-guest-grouping",
        "--guest-grouping",
        choices=("auto", "residue", "bonded", "none"),
        default="auto",
        help=(
            "How selected guest atoms are grouped into molecules: auto uses "
            "residues when the input has topology and covalent bonds otherwise; "
            "none keeps one guest per atom (legacy behaviour)."
        ),
    )
    group.add_argument(
        "--guest-bond-scale",
        type=positive_float,
        default=1.15,
        help=(
            "Bond criterion for topology-free inputs: two guest atoms are "
            "bonded when d <= scale * (r_i + r_j), with covalent radii r."
        ),
    )


def add_frame_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("frames")

    group.add_argument(
        "-start",
        "--start",
        type=nonnegative_int,
        default=0,
        help="First frame index.",
    )
    group.add_argument(
        "-stop",
        "--stop",
        type=nonnegative_int,
        default=None,
        help="Stop frame index, exclusive.",
    )
    group.add_argument(
        "-stride",
        "--stride",
        type=positive_int,
        default=1,
        help="Frame stride for analysis.",
    )


def add_analysis_mode_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("analysis mode")

    group.add_argument(
        "--skip-hop",
        action="store_true",
        help="Skip HOP classification. Mainly for debugging. Liquid filtering for cage search will be unavailable.",
    )
    group.add_argument(
        "--find-cages",
        action="store_true",
        help="Enable hydrate cage search after HOP classification.",
    )
    group.add_argument(
        "--no-exclude-liquid",
        action="store_true",
        help="Do not exclude HOP liquid-like water molecules from cage search.",
    )
    group.add_argument(
        "--include-ice-for-cage",
        action="store_true",
        help="Allow ice-like water molecules to participate in cage search.",
    )
    group.add_argument(
        "--standard-cages-only",
        action="store_true",
        help="Only report complete standard cages and disable partial cage fallback.",
    )


def add_output_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("output")

    group.add_argument(
        "-outdir",
        "--outdir",
        default="hydrate_analysis",
        help="Output directory.",
    )
    group.add_argument(
        "--timeseries",
        default="hydrate_timeseries.dat",
        help="Unified HOP/cage time-series output file name.",
    )
    group.add_argument(
        "--labels",
        default="hydrate_labels.dat",
        help="Per-water HOP label output file name.",
    )
    group.add_argument(
        "--no-labels",
        action="store_true",
        help="Disable per-water text label output.",
    )
    group.add_argument(
        "--label-stride",
        type=positive_int,
        default=100,
        help="Frame interval for per-water text label output.",
    )
    group.add_argument(
        "--hop-pdb",
        default="hydrate_labels.pdb",
        help="HOP PDB trajectory file name. This is one multi-model PDB trajectory.",
    )
    group.add_argument(
        "--no-hop-pdb",
        action="store_true",
        help="Disable HOP label PDB trajectory output.",
    )
    group.add_argument(
        "--hop-pdb-stride",
        type=positive_int,
        default=10,
        help="Frame interval for HOP PDB trajectory MODEL output.",
    )
    group.add_argument(
        "--cage-stat",
        default="cage_stat.dat",
        help="Cage count/statistics output file name.",
    )
    group.add_argument(
        "--shell-stat",
        default="guest_shell.dat",
        help="Guest shell statistics output file name.",
    )
    group.add_argument(
        "--cage-pdb-dir",
        default="",
        help="Directory for one-PDB-per-frame cage visualization files.",
    )
    group.add_argument(
        "--no-cage-pdb",
        action="store_true",
        help="Disable cage PDB output.",
    )
    group.add_argument(
        "--cage-pdb-stride",
        type=positive_int,
        default=1,
        help="Frame interval for cage PDB output. Cage PDB remains one file per selected frame.",
    )
    group.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable PNG plot generation while retaining visualization scripts.",
    )
    group.add_argument(
        "--no-report",
        action="store_true",
        help="Disable the one-page HTML summary report (report.html).",
    )


def add_hop_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("HOP parameters")

    group.add_argument(
        "--oo-cutoff",
        type=positive_float,
        default=3.5,
        help="O-O cutoff for HOP neighbor list.",
    )
    group.add_argument(
        "--dist-smooth",
        type=positive_float,
        default=0.15,
        help="Distance smoothing width for HOP coordination.",
    )

    group.add_argument("--hyd-crys-e-center", type=float, default=3.55)
    group.add_argument("--hyd-crys-e-width", type=positive_float, default=0.28)
    group.add_argument("--hyd-crys-s-max", type=float, default=1.10)
    group.add_argument("--hyd-crys-q-min", type=float, default=0.45)
    group.add_argument("--hyd-crys-raw-min", type=float, default=0.12)
    group.add_argument("--hyd-crys-support-min", type=float, default=0.10)

    group.add_argument("--hyd-pre-e-low", type=float, default=2.45)
    group.add_argument("--hyd-pre-e-high", type=float, default=3.40)
    group.add_argument("--hyd-pre-s-max", type=float, default=1.20)
    group.add_argument("--hyd-pre-q-min", type=float, default=0.55)
    group.add_argument("--hyd-pre-raw-min", type=float, default=0.18)
    group.add_argument("--hyd-pre-support-min", type=float, default=0.12)
    group.add_argument("--hyd-pre-neighbor-fraction-min", type=float, default=0.20)

    group.add_argument("--ice-q-min", type=float, default=0.35)

    group.add_argument("--coord-min", type=float, default=3.25)
    group.add_argument("--coord-max", type=float, default=4.85)

    group.add_argument("--pc-hyd", type=probability, default=0.50)
    group.add_argument("--pc-ice", type=probability, default=0.50)

    group.add_argument("--delta-s", type=positive_float, default=0.06)
    group.add_argument("--c-e-center", type=float, default=-0.05)
    group.add_argument("--sigma-ce", type=positive_float, default=0.22)
    group.add_argument("--sigma-count", type=positive_float, default=0.65)


def add_cage_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("cage parameters")

    group.add_argument(
        "--cage-oo-cutoff",
        type=positive_float,
        default=3.6,
        help="O-O cutoff for cage ring graph.",
    )
    group.add_argument(
        "--guest-radius",
        type=positive_float,
        default=6.2,
        help="Radius for selecting local water molecules around each guest.",
    )
    group.add_argument(
        "--face-radius",
        type=positive_float,
        default=7.2,
        help="Maximum guest-ring center distance.",
    )

    group.add_argument("--min-ring", type=positive_int, default=4)
    group.add_argument("--max-ring", type=positive_int, default=6)
    group.add_argument("--ring-planarity", type=positive_float, default=1.5)
    group.add_argument("--face-normal-cut", type=float, default=0.0)

    group.add_argument(
        "--max-local-cycles",
        type=positive_int,
        default=5000,
        help="Maximum local cycles enumerated per guest.",
    )
    group.add_argument(
        "--max-rings-per-guest",
        type=positive_int,
        default=56,
        help="Maximum scored rings kept for each guest.",
    )
    group.add_argument("--beam-width", type=positive_int, default=64)
    group.add_argument("--candidate-limit", type=positive_int, default=20)
    group.add_argument(
        "--cage-dedup-overlap",
        type=probability,
        default=0.85,
        help="Water-node Jaccard overlap used to merge duplicate cages around nearby guests.",
    )

    group.add_argument("--partial-min-faces", type=positive_int, default=5)
    group.add_argument("--partial-min-closure", type=probability, default=0.25)
    group.add_argument("--half-min-faces", type=positive_int, default=6)
    group.add_argument("--half-min-coverage", type=probability, default=0.30)
    group.add_argument("--half-max-coverage", type=probability, default=0.85)
    group.add_argument("--incipient-min-closure", type=probability, default=0.30)
    group.add_argument("--open-min-boundary", type=positive_int, default=6)

    group.add_argument(
        "--liquid-exclude-prob",
        type=probability,
        default=0.70,
        help="Only waters with liquid label and P_liq >= this value are excluded from cage search.",
    )
    group.add_argument(
        "--hyd-candidate-prob",
        type=probability,
        default=0.25,
        help="Minimum P_hyd + P_pre to keep water as cage candidate.",
    )
    group.add_argument(
        "--ice-candidate-prob",
        type=probability,
        default=0.50,
        help="Minimum P_Ih + P_Ic when --include-ice-for-cage is enabled.",
    )
    group.add_argument(
        "--ring-backend",
        choices=("auto", "python", "numba"),
        default="auto",
        help="Ring enumeration backend.",
    )


def add_tracking_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("cage tracking")
    group.add_argument("--no-tracking", action="store_true", help="Disable cage lifetime and transition tracking.")
    group.add_argument("--tracking-gap", type=nonnegative_int, default=1)
    group.add_argument("--tracking-water-overlap", type=probability, default=0.50)
    group.add_argument("--tracking-same-guest-overlap", type=probability, default=0.25)
    group.add_argument("--tracking-dedup-overlap", type=probability, default=0.85)
    group.add_argument("--occupancy-radius", type=positive_float, default=4.5)
    group.add_argument(
        "--occupancy-center",
        choices=("cage", "water"),
        default="cage",
        help="Center used to count guest occupancy.",
    )


def add_phase_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("hydrate crystal phase")
    group.add_argument(
        "--no-phase-analysis",
        action="store_true",
        help="Disable global sI/sII/sH tendency analysis.",
    )
    group.add_argument(
        "--phase-min-cages",
        type=positive_int,
        default=4,
        help="Minimum number of complete cages before assigning a phase.",
    )
    group.add_argument(
        "--phase-min-confidence",
        type=probability,
        default=0.60,
        help="Minimum confidence used for phase transitions.",
    )
    group.add_argument(
        "--phase-transition-persistence",
        type=positive_int,
        default=3,
        help="Analyzed frames required to confirm a phase transition.",
    )
    group.add_argument(
        "--phase-mixed-margin",
        type=probability,
        default=0.08,
        help="Score margin below which the result is reported as mixed.",
    )
    group.add_argument(
        "--phase-lattice-cutoff",
        type=positive_float,
        default=9.0,
        help="Cage-centre neighbour cut-off used to look for face-sharing cages (A).",
    )
    group.add_argument(
        "--phase-lattice-tolerance",
        type=positive_float,
        default=0.8,
        help=(
            "Tolerance on |d(centre,centre) - (d_a + d_b)| when validating that "
            "two cages really share a face (A)."
        ),
    )
    group.add_argument(
        "--no-ideal-reference",
        action="store_true",
        help=(
            "Do not compare the cage-centre distances with the ideal crystal "
            "structures in hytopo/ideal_crystal (composition only)."
        ),
    )
    group.add_argument(
        "--phase-reference-weight",
        type=probability,
        default=0.4,
        help="Weight of the ideal-structure distance check versus composition.",
    )
    group.add_argument(
        "--phase-min-coherence",
        type=probability,
        default=0.45,
        help=(
            "Minimum fraction of face-sharing candidates that pass the geometric "
            "test before a crystalline phase is reported."
        ),
    )


def add_nucleation_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("nucleation parameters")

    group.add_argument(
        "--no-nucleation",
        action="store_true",
        help="Disable nucleation analysis based on Nmax_hyd from hydrate_timeseries.dat.",
    )
    group.add_argument(
        "--nucleation-method",
        choices=("fixed", "adaptive", "commitment"),
        default="commitment",
        help=(
            "fixed uses a user threshold; adaptive estimates a robust threshold "
            "from the initial liquid baseline; commitment additionally requires growth."
        ),
    )
    group.add_argument(
        "--nucleation-min-size",
        type=positive_int,
        default=50,
        help="Minimum Nmax_hyd required to define a hydrate nucleus.",
    )
    group.add_argument(
        "--nucleation-persist-frames",
        type=positive_int,
        default=10,
        help="Number of consecutive analyzed frames required for nucleation.",
    )
    group.add_argument(
        "--nucleation-allow-gaps",
        type=nonnegative_int,
        default=0,
        help="Allowed below-threshold frames inside a persistent nucleation window.",
    )
    group.add_argument("--nucleation-baseline-frames", type=positive_int, default=100)
    group.add_argument("--nucleation-baseline-sigma", type=positive_float, default=6.0)
    group.add_argument("--nucleation-adaptive-min-size", type=positive_int, default=10)
    group.add_argument("--nucleation-min-growth", type=float, default=5.0)
    group.add_argument("--nucleation-min-slope", type=float, default=0.0)
    group.add_argument(
        "--nucleation-summary",
        default="nucleation_summary.txt",
        help="Nucleation summary output file name.",
    )
    group.add_argument(
        "--nucleation-timeseries",
        default="nucleation_timeseries.dat",
        help="Nucleation time-series output file name.",
    )
    group.add_argument(
        "--no-nucleation-timeseries",
        action="store_true",
        help="Disable nucleation time-series output.",
    )


def add_performance_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("performance")

    group.add_argument(
        "--no-numba",
        action="store_true",
        help="Disable numba kernels even if numba is available.",
    )
    group.add_argument(
        "--serial",
        action="store_true",
        help="Disable frame-level multiprocessing (use only one core).",
    )
    group.add_argument(
        "--n-procs",
        type=positive_int,
        default=0,
        help=(
            "Number of worker processes. 0 means all logical cores; lower it "
            "when memory is the limiting factor."
        ),
    )
    group.add_argument(
        "--chunksize",
        type=positive_int,
        default=1,
        help="Multiprocessing chunksize.",
    )
    group.add_argument(
        "--checkpoint-every",
        type=nonnegative_int,
        default=0,
        help=(
            "Write a resumable checkpoint every N analysed frames. 0 disables "
            "checkpointing."
        ),
    )
    group.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run from its checkpoint.",
    )


def add_misc_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("misc")

    group.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="Reduce terminal output.",
    )
    group.add_argument(
        "--verbose-guests",
        action="store_true",
        help="Print one diagnostic line per guest and frame (can be very verbose).",
    )


def make_config(args: argparse.Namespace) -> AnalysisConfig:
    trajectory = TrajectoryConfig(
        top=args.top,
        traj=args.traj,
        water_selection=args.water_selection,
        oxygen_selection=args.oxygen_selection,
        guest_selection=args.guest_selection,
        guest_grouping=args.guest_grouping,
        guest_bond_scale=args.guest_bond_scale,
        start=args.start,
        stop=args.stop,
        stride=args.stride,
    )

    output = OutputConfig(
        outdir=args.outdir,
        timeseries=args.timeseries,
        labels=args.labels,
        write_labels=not args.no_labels,
        label_stride=args.label_stride,
        write_hop_pdb=not args.no_hop_pdb,
        hop_pdb=args.hop_pdb,
        hop_pdb_stride=args.hop_pdb_stride,
        write_cage_pdb=not args.no_cage_pdb,
        cage_pdb_dir=args.cage_pdb_dir,
        cage_pdb_stride=args.cage_pdb_stride,
        cage_stat=args.cage_stat,
        shell_stat=args.shell_stat,
        make_plots=not args.no_plots,
        write_report=not args.no_report,
    )

    hop = HopParameters(
        oo_cutoff=args.oo_cutoff,
        dist_smooth=args.dist_smooth,
        hyd_crys_e_center=args.hyd_crys_e_center,
        hyd_crys_e_width=args.hyd_crys_e_width,
        hyd_crys_s_max=args.hyd_crys_s_max,
        hyd_crys_q_min=args.hyd_crys_q_min,
        hyd_crys_raw_min=args.hyd_crys_raw_min,
        hyd_crys_support_min=args.hyd_crys_support_min,
        hyd_pre_e_low=args.hyd_pre_e_low,
        hyd_pre_e_high=args.hyd_pre_e_high,
        hyd_pre_s_max=args.hyd_pre_s_max,
        hyd_pre_q_min=args.hyd_pre_q_min,
        hyd_pre_raw_min=args.hyd_pre_raw_min,
        hyd_pre_support_min=args.hyd_pre_support_min,
        hyd_pre_neighbor_fraction_min=args.hyd_pre_neighbor_fraction_min,
        ice_q_min=args.ice_q_min,
        coord_min=args.coord_min,
        coord_max=args.coord_max,
        pc_hyd=args.pc_hyd,
        pc_ice=args.pc_ice,
        delta_s=args.delta_s,
        c_e_center=args.c_e_center,
        sigma_ce=args.sigma_ce,
        sigma_count=args.sigma_count,
    )

    cage = CageParameters(
        oo_cutoff=args.cage_oo_cutoff,
        guest_radius=args.guest_radius,
        face_radius=args.face_radius,
        min_ring=args.min_ring,
        max_ring=args.max_ring,
        ring_planarity=args.ring_planarity,
        face_normal_cut=args.face_normal_cut,
        max_local_cycles=args.max_local_cycles,
        max_rings_per_guest=args.max_rings_per_guest,
        beam_width=args.beam_width,
        candidate_limit=args.candidate_limit,
        dedup_water_overlap=args.cage_dedup_overlap,
        standard_only=args.standard_cages_only,
        partial_min_faces=args.partial_min_faces,
        partial_min_closure=args.partial_min_closure,
        half_min_faces=args.half_min_faces,
        half_min_coverage=args.half_min_coverage,
        half_max_coverage=args.half_max_coverage,
        incipient_min_closure=args.incipient_min_closure,
        open_min_boundary=args.open_min_boundary,
        exclude_liquid=not args.no_exclude_liquid,
        include_ice=args.include_ice_for_cage,
        liquid_exclude_prob=args.liquid_exclude_prob,
        hyd_candidate_prob=args.hyd_candidate_prob,
        ice_candidate_prob=args.ice_candidate_prob,
        ring_backend="python" if args.no_numba else args.ring_backend,
    )

    tracking = TrackingConfig(
        enabled=not args.no_tracking,
        gap_tolerance=args.tracking_gap,
        min_water_overlap=args.tracking_water_overlap,
        min_same_guest_overlap=args.tracking_same_guest_overlap,
        dedup_water_overlap=args.tracking_dedup_overlap,
        occupancy_radius=args.occupancy_radius,
        occupancy_center=args.occupancy_center,
    )

    phase = PhaseParameters(
        enabled=not args.no_phase_analysis,
        min_complete_cages=args.phase_min_cages,
        min_confidence=args.phase_min_confidence,
        transition_persist_frames=args.phase_transition_persistence,
        mixed_margin=args.phase_mixed_margin,
        lattice_cutoff=args.phase_lattice_cutoff,
        lattice_tolerance=args.phase_lattice_tolerance,
        min_coherence=args.phase_min_coherence,
        use_ideal_reference=not args.no_ideal_reference,
        reference_weight=args.phase_reference_weight,
    )

    nucleation = NucleationParameters(
        enabled=not args.no_nucleation,
        method=args.nucleation_method,
        min_cluster_size=args.nucleation_min_size,
        persist_frames=args.nucleation_persist_frames,
        allow_gaps=args.nucleation_allow_gaps,
        baseline_frames=args.nucleation_baseline_frames,
        baseline_sigma=args.nucleation_baseline_sigma,
        adaptive_min_size=args.nucleation_adaptive_min_size,
        commitment_min_growth=args.nucleation_min_growth,
        commitment_min_slope=args.nucleation_min_slope,
        write_summary=True,
        write_timeseries=not args.no_nucleation_timeseries,
        summary_name=args.nucleation_summary,
        timeseries_name=args.nucleation_timeseries,
    )

    runtime = RuntimeConfig(
        use_numba=not args.no_numba,
        use_parallel=not args.serial,
        n_processes=args.n_procs,
        chunksize=args.chunksize,
        quiet=args.quiet,
        verbose_guests=args.verbose_guests,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )

    return AnalysisConfig(
        trajectory=trajectory,
        output=output,
        hop=hop,
        cage=cage,
        runtime=runtime,
        nucleation=nucleation,
        tracking=tracking,
        phase=phase,
        run_hop=not args.skip_hop,
        find_cages=args.find_cages,
    )


def run_analysis(config: AnalysisConfig) -> None:
    from .runner import run

    run(config)


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config = make_config(args)
    config.validate()
    config.ensure_output_dirs()
    config.log()

    run_analysis(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
