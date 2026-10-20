from __future__ import annotations

"""Reference cage-centre geometry from the ideal crystal structures.

The ideal structures in ``ideal_crystal/`` are periodic.  Their PDB files are
*not* wrapped into the primary cell (sI and sH have atoms well outside the
box), so the coordinates are folded back before any periodic analysis; without
that the boundary cages are split and never close.

Each reference is analysed with the same ring/cage pipeline as a trajectory, so
the reference cage centres, their types and the face-sharing centre-centre
distances come from the structure itself rather than from literature numbers.
The signatures are cached on disk next to the structures.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

IDEAL_FILES = {"sI": "sI.pdb", "sII": "sII.pdb", "sH": "sH.pdb"}

_CACHE: dict[str, Optional["PhaseReference"]] = {}


@dataclass(frozen=True)
class PhaseReference:
    phase: str
    counts: dict
    fractions: dict
    pair_distances: dict
    n_cages: int


def _reference_dir() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "ideal_crystal")


def _cache_path() -> str:
    return os.path.join(_reference_dir(), "reference_signatures.json")


def _build_signature(phase: str) -> Optional[dict]:
    from MDAnalysis import Universe

    from .cage import analyze_cages_for_frame
    from .cage_lattice import build_cage_lattice
    from .config import CageParameters
    from .geometry import wrap_point
    from .neighbor import build_cage_oo_graph

    path = os.path.join(_reference_dir(), IDEAL_FILES[phase])

    if not os.path.exists(path):
        return None

    universe = Universe(path)

    if universe.dimensions is None:
        return None

    box = np.asarray(universe.dimensions, dtype=np.float64)
    oxygen = universe.select_atoms("name OW or element O")
    guests = universe.select_atoms("element C")

    if len(oxygen) == 0 or len(guests) == 0:
        return None

    # fold the (unwrapped) reference coordinates back into the primary cell
    positions_o = wrap_point(oxygen.positions.astype(np.float64), box)
    positions_c = wrap_point(guests.positions.astype(np.float64), box)

    params = CageParameters()
    graph = build_cage_oo_graph(positions_o, box, params.oo_cutoff)
    result = analyze_cages_for_frame(
        0, positions_o, positions_c, box, graph, None, params, quiet=True
    )

    if result.n_cages == 0:
        return None

    lattice = build_cage_lattice(
        result.cages,
        positions_o,
        box,
        cutoff=params.oo_cutoff + 5.4,
        tolerance=0.8,
    )

    counts = {name: int(value) for name, value in dict(result.counts).items()}
    total = sum(counts.values()) or 1
    fractions = {name: value / total for name, value in counts.items()}

    pairs = {}

    if lattice is not None:
        for key, values in lattice.pair_distances.items():
            if len(values) < 3:
                continue
            pairs["|".join(key)] = [
                float(np.mean(values)),
                float(np.std(values)),
                int(len(values)),
            ]

    return {
        "phase": phase,
        "counts": counts,
        "fractions": fractions,
        "pair_distances": pairs,
        "n_cages": int(result.n_cages),
    }


def load_reference(phase: str) -> Optional[PhaseReference]:
    """Reference signature for one phase, cached on disk after first use."""
    if phase in _CACHE:
        return _CACHE[phase]

    path = _cache_path()
    stored = {}

    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                stored = json.load(handle)
        except (OSError, ValueError):
            stored = {}

    signature = stored.get(phase)

    if not signature:
        signature = _build_signature(phase)

        if signature is None:
            _CACHE[phase] = None
            return None

        stored[phase] = signature

        try:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(stored, handle, indent=1, sort_keys=True)
        except OSError:
            pass

    reference = PhaseReference(
        phase=phase,
        counts=dict(signature["counts"]),
        fractions=dict(signature["fractions"]),
        pair_distances={
            tuple(key.split("|")): tuple(value)
            for key, value in signature["pair_distances"].items()
        },
        n_cages=int(signature["n_cages"]),
    )

    _CACHE[phase] = reference
    return reference


def load_all() -> dict[str, PhaseReference]:
    references = {}

    for phase in IDEAL_FILES:
        reference = load_reference(phase)

        if reference is not None:
            references[phase] = reference

    return references


def distance_match(
    pair_distances: dict,
    reference: PhaseReference,
    n_cages: int,
    sigma: float = 0.45,
) -> float:
    """Abundance-weighted agreement of the face-sharing distance signature.

    Each reference cage-type pair contributes with the *smaller* of its expected
    and observed abundance per cage, and the score is normalised by the total
    expected abundance.  A pair that is essentially absent from the frame
    therefore neither rewards the phase nor is silently skipped: a structure
    with three 5^12 6^4 cages cannot be scored as sII just because those few
    distances happen to match.
    """
    if not reference.pair_distances or n_cages <= 0:
        return 0.0

    total_weight = 0.0
    total_score = 0.0

    for key, value in reference.pair_distances.items():
        expected_mean = float(value[0])
        expected_count = float(value[2]) if len(value) > 2 else 1.0
        expected_rate = expected_count / float(reference.n_cages)
        total_weight += expected_rate

        observed = pair_distances.get(key)

        if observed is None or len(observed) == 0:
            continue

        observed_rate = len(observed) / float(n_cages)
        weight = min(expected_rate, observed_rate)

        if weight <= 0.0:
            continue

        difference = float(np.mean(np.asarray(observed, dtype=np.float64))) - expected_mean
        total_score += weight * float(np.exp(-0.5 * (difference / sigma) ** 2))

    if total_weight <= 0.0:
        return 0.0

    return float(total_score / total_weight)
