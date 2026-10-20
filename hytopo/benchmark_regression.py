from __future__ import annotations

"""Per-frame regression and timing benchmark for the HOP + cage pipeline.

The point of this script is to make performance work safe: it records an exact
fingerprint of the per-frame result (HOP statistics, cage list, guest shell
table) so that a kernel rewrite can be proven to reproduce the reference
output, while the same run also reports the wall time per frame.

Generate a reference fingerprint with the current code::

    python -m hytopo.benchmark_regression \
        --top ARAS1.tpr --traj ARAS1.xtc -guest "resname CH4" \
        --frames 0,500,1000,1500 --write-golden benchmarks/golden_aras1.json

Then verify a change against it::

    python -m hytopo.benchmark_regression \
        --top ARAS1.tpr --traj ARAS1.xtc -guest "resname CH4" \
        --frames 0,500,1000,1500 --golden benchmarks/golden_aras1.json
"""

import argparse
import hashlib
import json
import os
import time
from typing import Any, Optional

import numpy as np

from .cage import analyze_cages_for_frame
from .config import (
    AnalysisConfig,
    CageParameters,
    HopParameters,
    OutputConfig,
    RuntimeConfig,
    TrajectoryConfig,
)
from .hop import analyze_hop_from_graph, build_cage_candidate_mask
from .neighbor import build_cage_oo_graph, build_oo_graph
from .trajectory import create_context, read_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Regression fingerprint + timing benchmark for hytopo.",
    )
    parser.add_argument("-top", "--top", required=True, help="Topology/structure file.")
    parser.add_argument("-traj", "--traj", default=None, help="Trajectory file.")
    parser.add_argument("-guest", "--guest-selection", default=None)
    parser.add_argument("-oxygen", "--oxygen-selection", default="name OW")
    parser.add_argument("-water", "--water-selection", default="resname SOL")
    parser.add_argument(
        "--frames",
        default="0,500,1000,1500",
        help="Comma separated frame indices to fingerprint (comma list or start:stop:step).",
    )
    parser.add_argument("--repeat", type=int, default=1, help="Timed repetitions per frame.")
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--golden", default=None, help="Reference fingerprint JSON.")
    parser.add_argument("--write-golden", action="store_true", help="Write the fingerprint.")
    parser.add_argument("--label", default="", help="Free-form label stored in the fingerprint.")
    parser.add_argument("--oo-cutoff", type=float, default=3.5)
    parser.add_argument("--cage-oo-cutoff", type=float, default=3.6)
    parser.add_argument("--guest-radius", type=float, default=6.2)
    parser.add_argument("--face-radius", type=float, default=7.2)
    parser.add_argument("--max-local-cycles", type=int, default=5000)
    parser.add_argument("--max-rings-per-guest", type=int, default=56)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--candidate-limit", type=int, default=20)
    parser.add_argument("--no-hop", action="store_true", help="Cage-only timing.")
    parser.add_argument("--json-out", default=None, help="Write run summary JSON here.")
    return parser


def parse_frames(text: str) -> list[int]:
    text = text.strip()
    if ":" in text:
        start, stop, step = (int(item) for item in text.split(":"))
        return list(range(start, stop, step))
    return [int(item) for item in text.replace(" ", "").split(",") if item != ""]


def make_config(args: argparse.Namespace) -> AnalysisConfig:
    return AnalysisConfig(
        trajectory=TrajectoryConfig(
            top=args.top,
            traj=args.traj,
            water_selection=args.water_selection,
            oxygen_selection=args.oxygen_selection,
            guest_selection=args.guest_selection,
        ),
        output=OutputConfig(outdir=".", write_labels=False, write_hop_pdb=False,
                            write_cage_pdb=False, make_plots=False),
        hop=HopParameters(oo_cutoff=args.oo_cutoff),
        cage=CageParameters(
            oo_cutoff=args.cage_oo_cutoff,
            guest_radius=args.guest_radius,
            face_radius=args.face_radius,
            max_local_cycles=args.max_local_cycles,
            max_rings_per_guest=args.max_rings_per_guest,
            beam_width=args.beam_width,
            candidate_limit=args.candidate_limit,
        ),
        runtime=RuntimeConfig(quiet=True),
        run_hop=not args.no_hop,
        find_cages=True,
    )


def digest(values: Any) -> str:
    return hashlib.sha256(repr(values).encode("utf-8")).hexdigest()[:16]


def fingerprint_frame(result: Any, hop_result: Any) -> dict:
    cages = getattr(result, "cages", None)
    cage_rows = []
    if cages is not None:
        for cage in cages:
            cage_rows.append(
                (
                    int(cage.guest_id),
                    str(cage.cage_type),
                    tuple(int(node) for node in cage.nodes),
                )
            )

    shell_rows = getattr(result, "shell_rows", []) or []
    stats = getattr(hop_result, "stats", None)

    entry = {
        "frame": int(result.frame),
        "n_cages": int(getattr(result, "n_cages", len(cage_rows))),
        "counts": {str(key): int(value) for key, value in dict(result.counts).items()},
        "cages_digest": digest(cage_rows),
        "n_cage_rows": len(cage_rows),
        "shell_digest": digest(tuple(tuple(row) for row in shell_rows)),
        "n_shell_rows": len(shell_rows),
        "n_local_cycles": int(getattr(result, "n_local_cycles_total", 0)),
        "n_local_rings": int(getattr(result, "n_local_rings_total", 0)),
    }

    if stats is not None:
        entry["hop"] = {
            "Nmax_hyd": round(float(stats.Nmax_hyd), 6),
            "Nmax_ice": round(float(stats.Nmax_ice), 6),
            "H_total": round(float(stats.H_total), 9),
            "Hyd_order": round(float(stats.Hyd_order), 9),
        }
        entry["labels_digest"] = digest(
            np.asarray(hop_result.labels, dtype=np.int64).tobytes()
        )

    return entry


def compare(expected: dict, actual: dict) -> list[str]:
    problems = []

    for key in (
        "n_cages",
        "counts",
        "cages_digest",
        "n_cage_rows",
        "shell_digest",
        "n_shell_rows",
        "n_local_cycles",
        "n_local_rings",
        "hop",
        "labels_digest",
    ):
        if key not in expected:
            continue
        if expected[key] != actual.get(key):
            problems.append(f"frame {expected['frame']}: {key} changed")

    return problems


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = make_config(args)
    config.validate()

    frames = parse_frames(args.frames)
    context = create_context(config)

    total_start = time.perf_counter()
    context_seconds = time.perf_counter() - total_start

    entries = []
    timings = []

    for index, frame_index in enumerate(frames):
        payload = read_frame(context, int(frame_index))

        if args.warmup and index == 0:
            # Warm-up pass: numba compilation and first-touch allocations.
            run_once(payload, config)

        best = None
        hop_result = None
        cage_result = None

        for _ in range(max(1, args.repeat)):
            hop_result, cage_result, seconds = run_once(payload, config)
            best = seconds if best is None else min(best, seconds)

        timings.append(best)
        entry = fingerprint_frame(cage_result, hop_result)
        entry["seconds"] = round(float(best), 4)
        entries.append(entry)

    elapsed = time.perf_counter() - total_start
    payload = {
        "label": args.label,
        "top": os.path.basename(args.top),
        "traj": os.path.basename(args.traj) if args.traj else None,
        "guest_selection": args.guest_selection,
        "frames": entries,
    }

    print(f"context load     : {context_seconds:.2f} s")
    for entry in entries:
        print(
            f"frame {entry['frame']:6d} : {entry['seconds']:8.3f} s   "
            f"cages={entry['n_cages']:4d}  "
            f"cycles={entry['n_local_cycles']:6d}  rings={entry['n_local_rings']:6d}"
        )
    if timings:
        mean_seconds = sum(timings) / len(timings)
        print(f"mean per frame   : {mean_seconds:8.3f} s")
        print(f"total (timed)    : {sum(timings):8.3f} s + context {context_seconds:.2f} s")

    status = 0

    if args.golden:
        if args.write_golden:
            with open(args.golden, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=1, sort_keys=True)
            print(f"golden written   : {args.golden}")
        else:
            with open(args.golden, encoding="utf-8") as handle:
                expected = json.load(handle)

            expected_by_frame = {entry["frame"]: entry for entry in expected["frames"]}
            problems = []

            for entry in entries:
                reference = expected_by_frame.get(entry["frame"])
                if reference is None:
                    print(f"frame {entry['frame']}: not present in golden file, skipped")
                    continue
                problems.extend(compare(reference, entry))

            if problems:
                print("REGRESSION         : FAILED")
                for problem in problems:
                    print(f"  - {problem}")
                status = 1
            else:
                print(f"REGRESSION         : PASSED ({len(entries)} frames identical)")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=1, sort_keys=True)

    return status


def run_once(payload: Any, config: AnalysisConfig):
    start = time.perf_counter()

    hop_result = None
    if config.run_hop:
        graph = build_oo_graph(
            positions=payload.oxygen_positions,
            box=payload.box,
            cutoff=config.hop.oo_cutoff,
        )
        hop_result = analyze_hop_from_graph(graph, config.hop, use_numba=True)
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
        graph = None
        candidate_mask = np.ones(len(payload.oxygen_positions), dtype=np.bool_)

    if graph is not None and abs(config.cage.oo_cutoff - config.hop.oo_cutoff) < 1.0e-12:
        cage_graph = graph
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
        quiet=True,
    )

    return hop_result, cage_result, time.perf_counter() - start


if __name__ == "__main__":
    raise SystemExit(main())
