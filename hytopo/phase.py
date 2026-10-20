from __future__ import annotations

"""Hydrate phase assignment from cage-centre spatial association.

The phase is decided from the *lattice network* of cage centres rather than from
raw cage counts:

* :mod:`hytopo.cage_lattice` links cages that share a validated face and
  returns the connected components (crystal domains), the per-frame
  centre-centre distance distributions and a lattice coherence.
* Counts are then taken over the largest domain only, so isolated cages in
  liquid water or at the interface no longer move the composition ratio.
* Confidence is scaled by the lattice coherence, so a loose mixture of cages
  cannot be reported as a crystalline phase.

This is deliberately *not* a space-group determination: it reports the
dominant hydrate structure (sI / sII / sH), its domain size and the fraction of
cages sitting outside the largest domain (the interface).
"""

from collections import Counter
from dataclasses import dataclass
import os
from typing import Any, Optional

import numpy as np

from .cage_lattice import (
    FrameLattice,
    _domain_radius,
    build_cage_lattice,
    lattice_counts,
)
from .output_layout import PHASE, categorised_path


PHASE_UNKNOWN = "unknown"
PHASE_SI = "sI"
PHASE_SII = "sII"
PHASE_SH = "sH"
PHASE_MIXED = "mixed"


PHASE_TARGETS: dict[str, dict[str, int]] = {
    PHASE_SI: {
        "5^12": 2,
        "5^12_6^2": 6,
    },
    PHASE_SII: {
        "5^12": 16,
        "5^12_6^4": 8,
    },
    PHASE_SH: {
        "5^12": 3,
        "4^3_5^6_6^3": 2,
        "5^12_6^8": 1,
    },
}

TRACKED_CAGE_TYPES = (
    "5^12",
    "5^12_6^2",
    "5^12_6^4",
    "4^3_5^6_6^3",
    "5^12_6^8",
)


@dataclass(frozen=True)
class PhaseParameters:
    enabled: bool = True
    min_complete_cages: int = 4
    min_confidence: float = 0.60
    transition_persist_frames: int = 3
    mixed_margin: float = 0.08

    #: cage-centre neighbour cut-off for face-sharing candidates (Angstrom)
    lattice_cutoff: float = 9.0
    #: tolerance on |d(centre,centre) - (d_a + d_b)| for a valid shared face
    lattice_tolerance: float = 0.8
    #: minimum lattice coherence for a crystalline phase assignment
    min_coherence: float = 0.45
    #: compare the observed cage-centre distances with the ideal structures
    use_ideal_reference: bool = True
    #: width of the distance agreement window (Angstrom)
    reference_sigma: float = 0.60
    #: weight of the distance check relative to the composition score
    reference_weight: float = 0.4


@dataclass(frozen=True)
class PhaseClassification:
    frame: int
    time_ps: float
    phase: str
    confidence: float
    scores: dict[str, float]
    n_complete_cages: int
    counts: dict[str, int]
    units_estimate: float
    domain_size: int = 0
    domain_fraction: float = 0.0
    domain_radius: float = 0.0
    interface_fraction: float = 0.0
    coherence: float = 0.0
    n_domains: int = 0
    domains: tuple = ()
    reference_scores: dict = None


@dataclass(frozen=True)
class PhaseTransition:
    frame: int
    time_ps: float
    from_phase: str
    to_phase: str
    confidence: float


def _composition_scores(
    counts: Counter,
    params: PhaseParameters,
) -> tuple[dict[str, float], dict[str, float], int]:
    n_complete = sum(counts.get(name, 0) for name in TRACKED_CAGE_TYPES)

    scores: dict[str, float] = {}
    units: dict[str, float] = {}

    for phase, target in PHASE_TARGETS.items():
        target_total = float(sum(target.values()))
        observed_total = float(sum(counts.get(name, 0) for name in target))

        observed_fraction = {
            name: counts.get(name, 0) / max(observed_total, 1.0) for name in target
        }
        target_fraction = {
            name: value / target_total for name, value in target.items()
        }

        composition_similarity = 1.0 - 0.5 * sum(
            abs(observed_fraction[name] - target_fraction[name]) for name in target
        )
        unit_estimate = min(
            counts.get(name, 0) / float(value) for name, value in target.items()
        )
        expected_from_units = unit_estimate * target_total
        coverage = min(1.0, expected_from_units / max(n_complete, 1.0))

        scores[phase] = 0.75 * composition_similarity + 0.25 * coverage
        units[phase] = unit_estimate

    return scores, units, n_complete


def _classify_component(
    lattice: FrameLattice,
    component: list[int],
    params: PhaseParameters,
) -> dict:
    counts = Counter(lattice.types[index] for index in component)
    scores, _, _ = _composition_scores(counts, params)
    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_phase, best_score = ordered[0]
    margin = best_score - ordered[1][1]

    phase = best_phase if (best_score >= params.min_confidence
                           and margin >= params.mixed_margin) else PHASE_MIXED
    radius = _domain_radius(
        lattice.centers,
        component,
        lattice.box,
    )

    return {
        "size": len(component),
        "phase": phase,
        "score": float(best_score),
        "radius": radius,
        "counts": dict(counts),
    }


def classify_phase(
    frame: int,
    time_ps: float,
    cages: Any,
    params: Optional[PhaseParameters] = None,
    oxygen_pos: Optional[np.ndarray] = None,
    box: Optional[np.ndarray] = None,
    lattice: Optional[FrameLattice] = None,
) -> PhaseClassification:
    """Assign the hydrate phase from the cage-centre lattice of one frame."""
    params = params or PhaseParameters()
    cage_list = list(getattr(cages, "cages", cages or []))

    if lattice is None:
        lattice = build_cage_lattice(
            cage_list,
            oxygen_pos,
            box,
            cutoff=params.lattice_cutoff,
            tolerance=params.lattice_tolerance,
        )

    counts = lattice_counts(lattice, cage_list)
    scores, units, n_tracked = _composition_scores(counts, params)

    n_complete = sum(counts.values())
    n_total = len(cage_list)

    domain_size = len(lattice.largest_component) if lattice is not None else 0
    coherence = lattice.coherence if lattice is not None else 0.0

    if n_complete < params.min_complete_cages:
        return PhaseClassification(
            frame=int(frame),
            time_ps=float(time_ps),
            phase=PHASE_UNKNOWN,
            confidence=0.0,
            scores={phase: 0.0 for phase in PHASE_TARGETS},
            n_complete_cages=n_complete,
            counts=dict(counts),
            units_estimate=0.0,
            domain_size=domain_size,
            domain_fraction=0.0,
            domain_radius=lattice.domain_radius if lattice is not None else 0.0,
            interface_fraction=0.0,
            coherence=float(coherence),
            n_domains=0,
        )

    reference_scores: dict[str, float] = {}

    if lattice is not None and params.use_ideal_reference:
        from .ideal_reference import distance_match, load_all

        references = load_all()

        if references:
            for phase_name, reference in references.items():
                reference_scores[phase_name] = distance_match(
                    lattice.pair_distances,
                    reference,
                    n_cages=n_complete,
                    sigma=params.reference_sigma,
                )

            if reference_scores:
                # distances act as an absolute check on top of the composition
                for phase_name in scores:
                    reference_score = reference_scores.get(phase_name, 0.0)
                    scores[phase_name] = (
                        (1.0 - params.reference_weight) * scores[phase_name]
                        + params.reference_weight * reference_score
                    )

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_phase, best_score = ordered[0]
    second_score = ordered[1][1]
    margin = best_score - second_score

    spatial_ok = coherence >= params.min_coherence if lattice is not None else True

    if lattice is not None:
        confidence = best_score * (0.5 + 0.5 * min(1.0, coherence))
    else:
        confidence = best_score

    confidence = max(0.0, min(1.0, confidence))

    if (
        best_score < params.min_confidence
        or margin < params.mixed_margin
        or not spatial_ok
    ):
        phase = PHASE_MIXED
    else:
        phase = best_phase

    domain_fraction = domain_size / n_total if n_total else 0.0
    interface_fraction = 1.0 - domain_fraction

    domains: list[dict] = []

    if lattice is not None:
        for rank, component in enumerate(lattice.components):
            if len(component) < params.min_complete_cages:
                continue
            detail = _classify_component(lattice, component, params)
            detail["rank"] = rank
            domains.append(detail)

    return PhaseClassification(
        frame=int(frame),
        time_ps=float(time_ps),
        phase=phase,
        confidence=float(confidence),
        scores={name: float(value) for name, value in scores.items()},
        n_complete_cages=int(n_complete),
        counts=dict(counts),
        units_estimate=float(units[best_phase]),
        domain_size=int(domain_size),
        domain_fraction=float(domain_fraction),
        domain_radius=float(lattice.domain_radius) if lattice is not None else 0.0,
        interface_fraction=float(interface_fraction),
        coherence=float(coherence),
        n_domains=len(domains),
        domains=tuple(domains),
        reference_scores=reference_scores,
    )


def distance_rows(
    frame: int,
    time_ps: float,
    lattice: Optional[FrameLattice],
) -> list[tuple]:
    """One row per cage-type pair with the centre-centre distance statistics."""
    if lattice is None:
        return []

    rows = []

    for (type_a, type_b), values in sorted(lattice.pair_distances.items()):
        if len(values) == 0:
            continue

        rows.append(
            (
                int(frame),
                float(time_ps),
                type_a,
                type_b,
                int(len(values)),
                float(np.mean(values)),
                float(np.median(values)),
                float(np.min(values)),
                float(np.max(values)),
            )
        )

    return rows


def cage_distance_header() -> tuple[str, ...]:
    return (
        "frame",
        "time_ps",
        "type_a",
        "type_b",
        "count",
        "mean_distance",
        "median_distance",
        "min_distance",
        "max_distance",
    )


class PhaseTracker:
    def __init__(self, params: Optional[PhaseParameters] = None) -> None:
        self.params = params or PhaseParameters()
        self.points: list[PhaseClassification] = []
        self.transitions: list[PhaseTransition] = []
        self.distance_table: list[tuple] = []
        self.domain_table: list[tuple] = []
        self._stable_phase = PHASE_UNKNOWN
        self._candidate_phase: Optional[str] = None
        self._candidate_count = 0

    def update(
        self,
        frame: int,
        time_ps: float,
        cages: Any,
        oxygen_pos: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
    ) -> PhaseClassification:
        cage_list = list(getattr(cages, "cages", cages or []))
        lattice = build_cage_lattice(
            cage_list,
            oxygen_pos,
            box,
            cutoff=self.params.lattice_cutoff,
            tolerance=self.params.lattice_tolerance,
        )

        point = classify_phase(
            frame,
            time_ps,
            cages,
            self.params,
            oxygen_pos=oxygen_pos,
            box=box,
            lattice=lattice,
        )
        self.points.append(point)
        self.distance_table.extend(distance_rows(frame, time_ps, lattice))

        for domain in point.domains:
            self.domain_table.append(
                (
                    int(frame),
                    float(time_ps),
                    int(domain["rank"]),
                    int(domain["size"]),
                    domain["phase"],
                    float(domain["score"]),
                    float(domain["radius"]),
                )
            )

        candidate = (
            point.phase
            if point.confidence >= self.params.min_confidence
            else PHASE_UNKNOWN
        )

        if candidate == self._candidate_phase:
            self._candidate_count += 1
        else:
            self._candidate_phase = candidate
            self._candidate_count = 1

        if (
            candidate != self._stable_phase
            and self._candidate_count >= self.params.transition_persist_frames
        ):
            self.transitions.append(
                PhaseTransition(
                    frame=point.frame,
                    time_ps=point.time_ps,
                    from_phase=self._stable_phase,
                    to_phase=candidate,
                    confidence=point.confidence,
                )
            )
            self._stable_phase = candidate

        return point

    def write_outputs(self, outdir: str) -> None:
        os.makedirs(outdir, exist_ok=True)

        path = categorised_path(outdir, "hydrate_phase_timeseries.dat", PHASE)
        header = (
            "frame\ttime_ps\tphase\tconfidence\tn_complete_cages\tunits_estimate\t"
            "sI_score\tsII_score\tsH_score\t"
            "n_5^12\tn_5^12_6^2\tn_5^12_6^4\tn_4^3_5^6_6^3\tn_5^12_6^8\t"
            "domain_size\tdomain_fraction\tdomain_radius\tinterface_fraction\t"
            "coherence\tn_domains\tref_sI\tref_sII\tref_sH"
        )

        with open(path, "w", encoding="utf-8") as handle:
            handle.write("# hytopo hydrate phase from cage-centre spatial association\n")
            handle.write(
                "# phase: sI/sII/sH/mixed/unknown; domain_size: cages in the largest\n"
                "# connected cage-centre domain; interface_fraction = 1 - domain_fraction\n"
            )
            handle.write(header + "\n")

            for point in self.points:
                handle.write(
                    "\t".join(
                        (
                            str(point.frame),
                            f"{point.time_ps:.3f}",
                            point.phase,
                            f"{point.confidence:.6f}",
                            str(point.n_complete_cages),
                            f"{point.units_estimate:.6f}",
                            f"{point.scores.get(PHASE_SI, 0.0):.6f}",
                            f"{point.scores.get(PHASE_SII, 0.0):.6f}",
                            f"{point.scores.get(PHASE_SH, 0.0):.6f}",
                            str(point.counts.get("5^12", 0)),
                            str(point.counts.get("5^12_6^2", 0)),
                            str(point.counts.get("5^12_6^4", 0)),
                            str(point.counts.get("4^3_5^6_6^3", 0)),
                            str(point.counts.get("5^12_6^8", 0)),
                            str(point.domain_size),
                            f"{point.domain_fraction:.6f}",
                            f"{point.domain_radius:.3f}",
                            f"{point.interface_fraction:.6f}",
                            f"{point.coherence:.6f}",
                            str(point.n_domains),
                            f"{(point.reference_scores or {}).get('sI', 0.0):.6f}",
                            f"{(point.reference_scores or {}).get('sII', 0.0):.6f}",
                            f"{(point.reference_scores or {}).get('sH', 0.0):.6f}",
                        )
                    )
                    + "\n"
                )

        transition_path = categorised_path(outdir, "hydrate_phase_transitions.dat", PHASE)

        with open(transition_path, "w", encoding="utf-8") as handle:
            handle.write("# confirmed phase transitions\n")
            handle.write("frame\ttime_ps\tfrom_phase\tto_phase\tconfidence\n")

            for transition in self.transitions:
                handle.write(
                    f"{transition.frame}\t{transition.time_ps:.3f}\t"
                    f"{transition.from_phase}\t{transition.to_phase}\t"
                    f"{transition.confidence:.6f}\n"
                )

        domain_path = categorised_path(outdir, "phase_domains.dat", PHASE)

        with open(domain_path, "w", encoding="utf-8") as handle:
            handle.write("# per-frame crystal domains from the cage-centre lattice\n")
            handle.write(
                "frame\ttime_ps\tdomain_rank\tn_cages\tphase\tscore\tradius_A\n"
            )

            for row in self.domain_table:
                handle.write(
                    f"{row[0]}\t{row[1]:.3f}\t{row[2]}\t{row[3]}\t{row[4]}\t"
                    f"{row[5]:.6f}\t{row[6]:.3f}\n"
                )

        distance_path = categorised_path(outdir, "cage_center_distances.dat", "02_cages")

        with open(distance_path, "w", encoding="utf-8") as handle:
            handle.write(
                "# face-sharing cage-centre distances per cage-type pair (Angstrom)\n"
            )
            handle.write("\t".join(cage_distance_header()) + "\n")

            for row in self.distance_table:
                handle.write(
                    f"{row[0]}\t{row[1]:.3f}\t{row[2]}\t{row[3]}\t{row[4]}\t"
                    f"{row[5]:.4f}\t{row[6]:.4f}\t{row[7]:.4f}\t{row[8]:.4f}\n"
                )

    @property
    def dominant_phase(self) -> str:
        if not self.points:
            return PHASE_UNKNOWN

        counts = Counter(point.phase for point in self.points)
        return counts.most_common(1)[0][0]
