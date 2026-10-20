from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .output_layout import VIEWS, categorised_path
from .config import (
    AnalysisConfig,
    CAGE_BETA,
    CAGE_RESNAME,
    CAGE_TYPES,
    LABEL_HYD,
    LABEL_IC,
    LABEL_IH,
    LABEL_LIQ,
    LABEL_NAMES,
    LABEL_PRE,
    LABEL_UNK,
)


HOP_LABEL_COLORS = {
    LABEL_LIQ: "blue",
    LABEL_IH: "cyan",
    LABEL_IC: "iceblue",
    LABEL_HYD: "red",
    LABEL_PRE: "orange",
    LABEL_UNK: "gray",
}

HOP_LABEL_RGB = {
    LABEL_LIQ: (0.20, 0.35, 1.00),
    LABEL_IH: (0.00, 0.85, 1.00),
    LABEL_IC: (0.65, 0.95, 1.00),
    LABEL_HYD: (1.00, 0.10, 0.10),
    LABEL_PRE: (1.00, 0.55, 0.00),
    LABEL_UNK: (0.55, 0.55, 0.55),
}

CAGE_TYPE_COLORS = {
    "5^12": "red",
    "5^12_6^2": "orange",
    "5^12_6^3": "yellow",
    "5^12_6^4": "green",
    "4^3_5^6_6^3": "cyan",
    "5^12_6^8": "blue",
    "half_5^12": "magenta",
    "half_5^12_6^2": "purple",
    "half_5^12_6^4": "pink",
    "incipient_cage": "tan",
    "open_cage": "silver",
    "distorted_cage": "gray",
    "partial_5^12": "magenta",
    "partial_5^12_6^2": "purple",
    "partial_5^12_6^4": "pink",
    "partial": "white",
    "other": "gray",
}

CAGE_TYPE_RGB = {
    "5^12": (1.00, 0.00, 0.00),
    "5^12_6^2": (1.00, 0.50, 0.00),
    "5^12_6^3": (1.00, 0.85, 0.00),
    "5^12_6^4": (0.00, 0.75, 0.20),
    "4^3_5^6_6^3": (0.00, 0.90, 0.90),
    "5^12_6^8": (0.10, 0.30, 1.00),
    "half_5^12": (1.00, 0.00, 1.00),
    "half_5^12_6^2": (0.60, 0.20, 1.00),
    "half_5^12_6^4": (1.00, 0.45, 0.75),
    "incipient_cage": (0.75, 0.55, 0.30),
    "open_cage": (0.75, 0.75, 0.75),
    "distorted_cage": (0.45, 0.45, 0.45),
    "partial_5^12": (1.00, 0.00, 1.00),
    "partial_5^12_6^2": (0.60, 0.20, 1.00),
    "partial_5^12_6^4": (1.00, 0.45, 0.75),
    "partial": (1.00, 1.00, 1.00),
    "other": (0.45, 0.45, 0.45),
}


@dataclass(frozen=True)
class VisualizationFiles:
    hop_vmd: Optional[str]
    hop_pymol: Optional[str]
    cage_vmd: Optional[str]
    cage_pymol: Optional[str]
    timeseries_png: Optional[str]
    cage_png: Optional[str]


def write_visualization_scripts(config: AnalysisConfig) -> VisualizationFiles:
    os.makedirs(config.output.outdir, exist_ok=True)

    hop_vmd = None
    hop_pymol = None
    cage_vmd = None
    cage_pymol = None

    view_dir = os.path.join(config.output.outdir, VIEWS)

    if config.output.write_hop_pdb:
        hop_pdb = config.output.hop_pdb_path
        hop_vmd = categorised_path(config.output.outdir, "view_hop.vmd")
        hop_pymol = categorised_path(config.output.outdir, "view_hop.pml")
        rel_pdb = os.path.relpath(hop_pdb, view_dir).replace(os.sep, "/")

        write_hop_vmd_script(hop_vmd, rel_pdb)
        write_hop_pymol_script(hop_pymol, rel_pdb)

    if config.find_cages and config.output.write_cage_pdb:
        cage_dir = config.output.cage_pdb_path
        cage_vmd = categorised_path(config.output.outdir, "view_cages.vmd")
        cage_pymol = categorised_path(config.output.outdir, "view_cages.pml")
        rel_dir = os.path.relpath(cage_dir, view_dir).replace(os.sep, "/")

        write_cage_vmd_script(cage_vmd, rel_dir)
        write_cage_pymol_script(cage_pymol, rel_dir)

    return VisualizationFiles(
        hop_vmd=hop_vmd,
        hop_pymol=hop_pymol,
        cage_vmd=cage_vmd,
        cage_pymol=cage_pymol,
        timeseries_png=None,
        cage_png=None,
    )


def write_hop_vmd_script(path: str, rel_pdb: str) -> None:

    lines = [
        "# VMD script for HOP label trajectory",
        "# Usage: vmd -e view_hop.vmd",
        "",
        f"mol new {rel_pdb} type pdb waitfor all",
        "mol delrep 0 top",
        "display projection Orthographic",
        "axes location Off",
        "color Display Background white",
        "",
    ]

    for label, color in HOP_LABEL_COLORS.items():
        label_name = LABEL_NAMES.get(label, str(label))
        lines.extend(
            [
                f"# {label}: {label_name}",
                "mol representation VDW 0.450 12.000",
                f"mol color ColorID {vmd_color_id(color)}",
                f"mol selection \"beta == {float(label):.1f}\"",
                "mol material Opaque",
                "mol addrep top",
                "",
            ]
        )

    lines.extend(
        [
            "animate goto 0",
            "puts \"HOP labels are stored in beta factor.\"",
            "puts \"Label definition:\"",
        ]
    )

    for label, name in LABEL_NAMES.items():
        lines.append(f"puts \"  beta {label}: {name}\"")

    write_text(path, "\n".join(lines) + "\n")


def write_hop_pymol_script(path: str, rel_pdb: str) -> None:

    lines = [
        "# PyMOL script for HOP label trajectory",
        f"load {rel_pdb}, hop_labels",
        "hide everything, hop_labels",
        "show spheres, hop_labels",
        "set sphere_scale, 0.35, hop_labels",
        "bg_color white",
        "",
    ]

    for label, rgb in HOP_LABEL_RGB.items():
        color_name = f"hop_label_{label}"
        selection_name = f"hop_{label}"
        lines.append(f"set_color {color_name}, [{rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}]")
        lines.append(f"select {selection_name}, hop_labels and b = {float(label):.1f}")
        lines.append(f"color {color_name}, {selection_name}")
        lines.append("")

    lines.extend(
        [
            "set all_states, on",
            "orient hop_labels",
            "",
        ]
    )

    write_text(path, "\n".join(lines))


def write_cage_vmd_script(path: str, rel_dir: str) -> None:

    lines = [
        "# VMD script for cage PDB files",
        "# Usage: vmd -e view_cages.vmd",
        "",
        "display projection Orthographic",
        "axes location Off",
        "color Display Background white",
        "",
        f"set cage_files [lsort [glob -nocomplain {rel_dir}/cages_frame_*.pdb]]",
        "if {[llength $cage_files] == 0} {",
        "    puts \"No cage PDB files found.\"",
        "    return",
        "}",
        "",
        "foreach pdb $cage_files {",
        "    mol new $pdb type pdb waitfor all",
        "    mol delrep 0 top",
        "    mol representation Licorice 0.200 12.000 12.000",
        "    mol color Name",
        "    mol selection all",
        "    mol material Opaque",
        "    mol addrep top",
        "}",
        "",
        "puts \"Cage type encoding uses atom name, resname, element, and beta factor.\"",
        "puts \"Cage beta definition:\"",
    ]

    for cage_type in CAGE_TYPES:
        beta = CAGE_BETA.get(cage_type, 20.0)
        resname = CAGE_RESNAME.get(cage_type, "OTH")
        color = CAGE_TYPE_COLORS.get(cage_type, "gray")
        lines.append(f"puts \"  {cage_type:20s} beta={beta:5.1f} resname={resname} color={color}\"")

    write_text(path, "\n".join(lines) + "\n")


def write_cage_pymol_script(path: str, rel_dir: str) -> None:

    lines = [
        "# PyMOL script for cage PDB files",
        f"python",
        f"import glob",
        f"for pdb in sorted(glob.glob('{rel_dir}/cages_frame_*.pdb')):",
        f"    cmd.load(pdb)",
        f"python end",
        "",
        "hide everything",
        "show sticks",
        "set stick_radius, 0.12",
        "bg_color white",
        "",
    ]

    for cage_type in CAGE_TYPES:
        beta = CAGE_BETA.get(cage_type, 20.0)
        rgb = CAGE_TYPE_RGB.get(cage_type, (0.45, 0.45, 0.45))
        color_name = sanitize_color_name(f"cage_{cage_type}")
        selection_name = sanitize_color_name(f"sel_{cage_type}")

        lines.append(f"set_color {color_name}, [{rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}]")
        lines.append(f"select {selection_name}, b = {beta:.1f}")
        lines.append(f"color {color_name}, {selection_name}")
        lines.append("")

    lines.extend(
        [
            "orient all",
            "set valence, 1",
            "",
        ]
    )

    write_text(path, "\n".join(lines))


def plot_outputs(config: AnalysisConfig) -> VisualizationFiles:
    timeseries_png = None
    cage_png = None

    if os.path.exists(config.output.timeseries_path):
        timeseries_png = categorised_path(config.output.outdir, "hydrate_timeseries.png")
        plot_timeseries(config.output.timeseries_path, timeseries_png)

    if config.find_cages and os.path.exists(config.output.cage_stat_path):
        cage_png = categorised_path(config.output.outdir, "cage_counts.png")
        plot_cage_counts(config.output.cage_stat_path, cage_png)

    return VisualizationFiles(
        hop_vmd=None,
        hop_pymol=None,
        cage_vmd=None,
        cage_pymol=None,
        timeseries_png=timeseries_png,
        cage_png=cage_png,
    )


def plot_timeseries(timeseries_path: str, output_png: str) -> None:
    import matplotlib.pyplot as plt

    names, data = load_table(timeseries_path)
    if data.size == 0:
        return

    time = get_column(names, data, "Time(ps)", fallback_index=0)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    for col in ("H_total", "H_crys", "H_pre", "Hyd_order", "Ice", "O_total"):
        if col in names:
            axes[0].plot(time, get_column(names, data, col), label=col)

    axes[0].set_ylabel("Order parameter")
    axes[0].legend(frameon=False, ncol=3)
    axes[0].grid(alpha=0.25)

    for col in ("Delta", "Nmax_hyd", "Nmax_ice"):
        if col in names:
            axes[1].plot(time, get_column(names, data, col), label=col)

    axes[1].set_xlabel("Time (ps)")
    axes[1].set_ylabel("Cluster / difference")
    axes[1].legend(frameon=False, ncol=3)
    axes[1].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def plot_cage_counts(cage_stat_path: str, output_png: str) -> None:
    import matplotlib.pyplot as plt

    names, data = load_table(cage_stat_path)
    if data.size == 0:
        return

    frame = get_column(names, data, "frame", fallback_index=0)

    fig, ax = plt.subplots(figsize=(11, 5))

    plotted = 0
    for cage_type in CAGE_TYPES:
        if cage_type not in names:
            continue

        values = get_column(names, data, cage_type)
        if np.max(values) <= 0:
            continue

        ax.plot(frame, values, label=cage_type)
        plotted += 1

    if "total" in names:
        ax.plot(frame, get_column(names, data, "total"), color="black", linewidth=2.0, label="total")

    ax.set_xlabel("Frame")
    ax.set_ylabel("Cage count")
    ax.grid(alpha=0.25)

    if plotted > 0:
        ax.legend(frameon=False, ncol=4, fontsize=8)

    fig.tight_layout()
    fig.savefig(output_png, dpi=200)
    plt.close(fig)


def load_table(path: str) -> tuple[list[str], np.ndarray]:
    """Read a hytopo table: optional '#' comments, one header line, TSV rows."""
    names: list[str] = []

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            names = stripped.split()
            break

    if not names:
        return [], np.empty((0, 0), dtype=np.float64)

    rows = []
    header_seen = False

    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if not header_seen:
                header_seen = True
                continue

            numeric = []
            for value in stripped.split()[: len(names)]:
                try:
                    numeric.append(float(value))
                except ValueError:
                    numeric.append(np.nan)

            if numeric:
                rows.append(numeric)

    if not rows:
        return names, np.empty((0, len(names)), dtype=np.float64)

    return names, np.asarray(rows, dtype=np.float64)


def get_column(
    names: list[str],
    data: np.ndarray,
    name: str,
    fallback_index: Optional[int] = None,
) -> np.ndarray:
    if name in names:
        return data[:, names.index(name)]

    if fallback_index is not None and fallback_index < data.shape[1]:
        return data[:, fallback_index]

    raise KeyError(f"Column not found: {name}")


def write_text(path: str, content: str) -> None:
    with open(path, "w") as handle:
        handle.write(content)


def sanitize_color_name(name: str) -> str:
    return (
        name.replace("^", "")
        .replace("-", "_")
        .replace("+", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def vmd_color_id(color: str) -> int:
    color_ids = {
        "blue": 0,
        "red": 1,
        "gray": 2,
        "orange": 3,
        "yellow": 4,
        "tan": 5,
        "silver": 6,
        "green": 7,
        "white": 8,
        "pink": 9,
        "cyan": 10,
        "purple": 11,
        "lime": 12,
        "mauve": 13,
        "ochre": 14,
        "iceblue": 15,
        "black": 16,
    }
    return color_ids.get(color, 2)


def print_visualization_summary(files: VisualizationFiles, quiet: bool = False) -> None:
    if quiet:
        return

    print("Visualization files:")

    for label, path in (
        ("HOP VMD", files.hop_vmd),
        ("HOP PyMOL", files.hop_pymol),
        ("cage VMD", files.cage_vmd),
        ("cage PyMOL", files.cage_pymol),
        ("timeseries plot", files.timeseries_png),
        ("cage plot", files.cage_png),
    ):
        if path:
            print(f"  {label:16s}: {path}")

    print("")


def generate_all_visualizations(config: AnalysisConfig, make_plots: bool = True) -> VisualizationFiles:
    script_files = write_visualization_scripts(config)

    if not make_plots:
        return script_files

    try:
        plot_files = plot_outputs(config)
    except Exception as exc:
        if not config.runtime.quiet:
            print(f"Note: PNG plot generation skipped: {exc}")
        return script_files

    return VisualizationFiles(
        hop_vmd=script_files.hop_vmd,
        hop_pymol=script_files.hop_pymol,
        cage_vmd=script_files.cage_vmd,
        cage_pymol=script_files.cage_pymol,
        timeseries_png=plot_files.timeseries_png,
        cage_png=plot_files.cage_png,
    )
