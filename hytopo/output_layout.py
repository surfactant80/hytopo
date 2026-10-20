from __future__ import annotations

"""One place that decides where every output file lives.

The output directory used to be flat, with files added over several revisions
and therefore inconsistent names and formats.  Outputs are now grouped into
numbered categories, and every writer asks this module for its path.
"""

import os
from typing import Iterable, Optional

TIMESERIES = "01_timeseries"
CAGES = "02_cages"
TRACKING = "03_tracking"
PHASE = "04_phase"
NUCLEATION = "05_nucleation"
LABELS = "06_labels"
CAGE_PDB = "07_cages_pdb"
PLOTS = "08_plots"
VIEWS = "09_views"

CATEGORY_ORDER = (
    TIMESERIES,
    CAGES,
    TRACKING,
    PHASE,
    NUCLEATION,
    LABELS,
    CAGE_PDB,
    PLOTS,
    VIEWS,
)

CATEGORY_TITLES = {
    TIMESERIES: "Time series",
    CAGES: "Cage statistics",
    TRACKING: "Cage tracking",
    PHASE: "Hydrate phase and domains",
    NUCLEATION: "Nucleation",
    LABELS: "Per-water labels",
    CAGE_PDB: "Cage visualisation (PDB)",
    PLOTS: "Plots",
    VIEWS: "Molecular viewer scripts",
}

_FILENAME_CATEGORY = {
    "hydrate_timeseries.dat": TIMESERIES,
    "cage_stat.dat": CAGES,
    "guest_shell.dat": CAGES,
    "cage_center_distances.dat": CAGES,
    "cage_lifetime.dat": TRACKING,
    "cage_transitions.dat": TRACKING,
    "cage_transition_summary.dat": TRACKING,
    "cage_occupancy.dat": TRACKING,
    "cage_quality.dat": TRACKING,
    "guest_events.dat": TRACKING,
    "hydrate_phase_timeseries.dat": PHASE,
    "hydrate_phase_transitions.dat": PHASE,
    "phase_domains.dat": PHASE,
    "nucleation_summary.txt": NUCLEATION,
    "nucleation_timeseries.dat": NUCLEATION,
    "hydrate_labels.dat": LABELS,
    "hydrate_labels.pdb": LABELS,
}


def output_category(name: str) -> str:
    """Category directory for an output file name."""
    base = os.path.basename(name)

    if base in _FILENAME_CATEGORY:
        return _FILENAME_CATEGORY[base]

    if base.startswith("cage_transition_network"):
        return TRACKING

    if base.endswith(".png"):
        return PLOTS

    if base.endswith((".vmd", ".pml")):
        return VIEWS

    if base.startswith("cages_frame_") or base == "cage_pdb":
        return CAGE_PDB

    return CAGES


def categorised_path(outdir: str, name: str, category: Optional[str] = None) -> str:
    """Full path for ``name`` inside its category directory under ``outdir``.

    ``category`` overrides the filename lookup, which keeps user-renamed output
    files in the right folder.
    """
    folder = category or output_category(name)
    return os.path.join(outdir, folder, os.path.basename(name))


def ensure_category_dirs(outdir: str, categories: Iterable[str] = CATEGORY_ORDER) -> None:
    for category in categories:
        os.makedirs(os.path.join(outdir, category), exist_ok=True)


def relpath(outdir: str, name: str) -> str:
    """Category-relative path, for report tables."""
    return os.path.join(output_category(name), os.path.basename(name))
