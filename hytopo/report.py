from __future__ import annotations

"""One-page HTML summary report and output manifest.

The report is a single self-contained file: plots are embedded as base64 PNGs
and every number is read back from the tables that were just written, so the
page can be shared as-is.  The raw data stays in the categorised directories.
"""

import base64
import os
import time
from typing import Any, Optional

import numpy as np

from .output_layout import CATEGORY_ORDER, CATEGORY_TITLES
from .visualize import get_column, load_table


def _read_png_base64(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.exists(path):
        return None

    try:
        with open(path, "rb") as handle:
            return base64.b64encode(handle.read()).decode("ascii")
    except OSError:
        return None


def _table_rows(path: Optional[str], limit: int = 40):
    if not path or not os.path.exists(path):
        return [], np.empty((0, 0))

    try:
        return load_table(path)
    except Exception:
        return [], np.empty((0, 0))


def _format_number(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def _plot_phase(path: str, phase_path: str) -> Optional[str]:
    names, data = _table_rows(phase_path)

    if data.size == 0 or "time_ps" not in names:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    time_axis = get_column(names, data, "time_ps")
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(time_axis, get_column(names, data, "confidence"), color="#2563eb")
    axes[0].set_ylabel("Phase confidence")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.25)

    axes[1].plot(time_axis, get_column(names, data, "domain_size"), color="#059669")
    axes[1].set_ylabel("Domain size (cages)")
    axes[1].grid(alpha=0.25)

    axes[2].plot(time_axis, get_column(names, data, "interface_fraction"), color="#b45309")
    axes[2].set_ylabel("Outside domain")
    axes[2].set_xlabel("Time (ps)")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_nucleation(path: str, nucleation_path: str) -> Optional[str]:
    names, data = _table_rows(nucleation_path)

    if data.size == 0:
        return None

    column = None

    for candidate in ("nmax_hyd", "Nmax_hyd"):
        if candidate in names:
            column = candidate
            break

    if column is None:
        return None

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None

    if "time_ps" in names:
        time_axis = get_column(names, data, "time_ps")
    elif "time" in names:
        time_axis = get_column(names, data, "time")
    else:
        time_axis = np.arange(data.shape[0], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(time_axis, get_column(names, data, column), color="#7c3aed")
    ax.set_xlabel("Time (ps)")
    ax.set_ylabel("Largest hydrate cluster")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _cage_table(cage_stat_path: str) -> tuple[list[str], list[list[str]]]:
    names, data = _table_rows(cage_stat_path)

    if data.size == 0:
        return [], []

    last = data[-1]
    skip = {"frame", "n_oxygen", "n_guest"}
    rows = []

    for index, name in enumerate(names):
        if name in skip:
            continue

        value = float(last[index])

        if name != "total" and value <= 0:
            continue

        rows.append([name, f"{int(round(value))}"])

    rows.sort(key=lambda item: int(item[1]), reverse=True)
    return ["Cage type", "Count (last frame)"], rows


def _lifetime_table(lifetime_path: str) -> tuple[list[str], list[list[str]]]:
    names, data = _table_rows(lifetime_path)

    if data.size == 0 or "dominant_type" not in names:
        return [], []

    # load_table only returns numeric columns; lifetimes are read directly.
    buckets: dict[str, list[float]] = {}
    header: list[str] = []

    with open(lifetime_path, "r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()

            if not stripped or stripped.startswith("#"):
                continue

            if not header:
                header = stripped.split("\t")
                continue

            parts = stripped.split("\t")

            if len(parts) != len(header):
                continue

            row = dict(zip(header, parts))

            try:
                lifetime = float(row.get("lifetime_time", "nan"))
            except ValueError:
                continue

            if np.isfinite(lifetime):
                buckets.setdefault(row.get("dominant_type", "other"), []).append(lifetime)

    rows = []

    for cage_type, values in sorted(buckets.items(), key=lambda item: -len(item[1])):
        rows.append(
            [
                cage_type,
                str(len(values)),
                f"{np.mean(values):.2f}",
                f"{np.max(values):.2f}",
            ]
        )

    return ["Dominant cage type", "Tracks", "Mean lifetime (ps)", "Max lifetime (ps)"], rows


def _file_tree(outdir: str) -> str:
    lines = []

    for category in CATEGORY_ORDER:
        directory = os.path.join(outdir, category)

        if not os.path.isdir(directory):
            continue

        entries = sorted(
            name for name in os.listdir(directory) if not name.startswith(".")
        )

        if not entries:
            continue

        lines.append(f"<h4>{category} &mdash; {CATEGORY_TITLES[category]}</h4>")
        lines.append("<ul>")

        for name in entries[:60]:
            size = os.path.getsize(os.path.join(directory, name))
            lines.append(f"<li><code>{category}/{name}</code> &middot; {size:,} B</li>")

        if len(entries) > 60:
            lines.append(f"<li>{len(entries) - 60} more files</li>")

        lines.append("</ul>")

    return "\n".join(lines)


def _html_table(header: list[str], rows: list[list[str]]) -> str:
    if not header:
        return "<p class='muted'>No data.</p>"

    head = "".join(f"<th>{cell}</th>" for cell in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )

    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"



def _cage_quality_section(outdir: str) -> tuple[str, str]:
    """Mean sphericity per cage type and guest escape/recapture counts."""
    quality_path = os.path.join(outdir, "03_tracking", "cage_quality.dat")
    events_path = os.path.join(outdir, "03_tracking", "guest_events.dat")

    quality_rows: list[list[str]] = []
    buckets: dict[str, list[float]] = {}

    if os.path.exists(quality_path):
        header = None

        with open(quality_path, "r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()

                if not stripped or stripped.startswith("#"):
                    continue

                if header is None:
                    header = stripped.split("\t")
                    continue

                parts = stripped.split("\t")

                if len(parts) != len(header):
                    continue

                row = dict(zip(header, parts))

                try:
                    buckets.setdefault(row["cage_type"], []).append(float(row["sphericity"]))
                    buckets.setdefault(row["cage_type"] + "#radius", []).append(
                        float(row["radius_mean"])
                    )
                except (KeyError, ValueError):
                    continue

    for cage_type, values in sorted(buckets.items(), key=lambda item: -len(item[1])):
        if cage_type.endswith("#radius"):
            continue
        radii = buckets.get(cage_type + "#radius", [])
        quality_rows.append(
            [
                cage_type,
                str(len(values)),
                f"{np.mean(values):.3f}",
                f"{np.mean(radii):.2f}" if radii else "-",
            ]
        )

    escape = 0
    recapture = 0

    if os.path.exists(events_path):
        with open(events_path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("#") or line.startswith("frame"):
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                if parts[3] == "escape":
                    escape += 1
                elif parts[3] == "recapture":
                    recapture += 1

    quality_table = _html_table(
        ["Cage type", "Cages", "Mean sphericity", "Mean radius (A)"],
        quality_rows[:12],
    )

    summary = (
        f"<p class='note'>Guest escape events: <b>{escape}</b> &middot; "
        f"recapture events: <b>{recapture}</b> (a cage reforming around the same "
        "guest within a few frames).</p>"
    )

    return quality_table, summary


def generate_html_report(
    config: Any,
    tracker: Optional[Any] = None,
    nucleation_res: Optional[Any] = None,
    phase_tracker: Optional[Any] = None,
    total_frames: int = 0,
    elapsed_time: float = 0.0,
) -> Optional[str]:
    """Write ``report.html`` and the output manifest; return the report path."""
    outdir = config.output.outdir
    os.makedirs(outdir, exist_ok=True)

    plot_dir = os.path.join(outdir, "08_plots")
    os.makedirs(plot_dir, exist_ok=True)

    phase_png = _plot_phase(
        os.path.join(plot_dir, "hydrate_phase_domains.png"),
        config.output.path("hydrate_phase_timeseries.dat"),
    )
    nucleation_png = _plot_nucleation(
        os.path.join(plot_dir, "nucleation.png"),
        config.output.path("nucleation_timeseries.dat"),
    )

    images = [
        ("Hydrate order parameter", config.output.path("hydrate_timeseries.png")),
        ("Cage population by type", config.output.path("cage_counts.png")),
        ("Phase confidence, domain size and interface fraction", phase_png),
        ("Largest hydrate cluster", nucleation_png),
        ("Cage transition network", config.output.path("cage_transition_network.png")),
    ]

    cards = []

    if phase_tracker is not None and phase_tracker.points:
        last = phase_tracker.points[-1]
        cards.append(("Dominant phase", phase_tracker.dominant_phase.upper()))
        cards.append(("Confidence (last frame)", _format_number(last.confidence)))
        cards.append(("Domain size (cages)", str(last.domain_size)))
        cards.append(("Domain radius (A)", _format_number(last.domain_radius, 1)))
        cards.append(("Outside domain", _format_number(last.interface_fraction)))

    if nucleation_res is not None:
        status = "detected" if nucleation_res.nucleated else "not detected"

        if nucleation_res.nucleated:
            cards.append(("Nucleation", f"{status} @ {nucleation_res.nucleation_time:.0f} ps"))
        else:
            cards.append(("Nucleation", status))

    if tracker is not None and tracker.tracks:
        cards.append(("Tracked cages", str(len(tracker.tracks))))

    cards.append(("Frames analysed", str(total_frames)))
    cards.append(("Wall time", f"{elapsed_time:.1f} s"))

    cage_header, cage_rows = _cage_table(config.output.path("cage_stat.dat"))
    lifetime_header, lifetime_rows = _lifetime_table(
        config.output.path("cage_lifetime.dat")
    )

    quality_table, quality_summary = _cage_quality_section(outdir)

    phase_transitions = []

    if phase_tracker is not None:
        for transition in phase_tracker.transitions:
            phase_transitions.append(
                [
                    str(transition.frame),
                    _format_number(transition.time_ps, 1),
                    transition.from_phase,
                    transition.to_phase,
                    _format_number(transition.confidence),
                ]
            )

    image_blocks = []

    for title, path in images:
        encoded = _read_png_base64(path)

        if encoded:
            image_blocks.append(
                f"<figure><img src='data:image/png;base64,{encoded}' alt='{title}'/>"
                f"<figcaption>{title}</figcaption></figure>"
            )

    card_html = "".join(
        f"<div class='card'><div class='k'>{label}</div><div class='v'>{value}</div></div>"
        for label, value in cards
    )

    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    top_name = os.path.basename(str(config.trajectory.top))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>hytopo report &mdash; {top_name}</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0; padding: 28px; background: #f6f8fb; color: #16202e;
       font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif; line-height: 1.45; }}
h1 {{ font-size: 22px; margin: 0 0 4px; }}
h2 {{ font-size: 16px; margin: 30px 0 10px; padding-bottom: 5px;
      border-bottom: 2px solid #dbe4f0; }}
h4 {{ font-size: 13px; margin: 14px 0 4px; color: #334155; }}
.sub {{ color: #64748b; font-size: 13px; }}
.cards {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0 4px; }}
.card {{ flex: 1 1 150px; background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
         padding: 10px 12px; }}
.card .k {{ font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: #64748b; }}
.card .v {{ font-size: 19px; font-weight: 650; margin-top: 2px; }}
figure {{ margin: 12px 0; background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
          padding: 10px; }}
figure img {{ width: 100%; height: auto; display: block; }}
figcaption {{ font-size: 12px; color: #64748b; margin-top: 6px; text-align: center; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 13px;
         border: 1px solid #e2e8f0; border-radius: 8px; overflow: hidden; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #eef2f7; text-align: left; }}
th {{ background: #eef3fa; font-weight: 600; }}
tr:last-child td {{ border-bottom: none; }}
code {{ font-size: 12px; }}
ul {{ margin: 2px 0 8px; padding-left: 20px; font-size: 12px; color: #475569; }}
.muted {{ color: #94a3b8; font-size: 13px; }}
.note {{ font-size: 12px; color: #64748b; }}
</style>
</head>
<body>
<h1>hytopo &mdash; hydrate topology report</h1>
<div class="sub">{top_name} &middot; generated {generated}</div>

<div class="cards">{card_html}</div>

<h2>1. Time evolution</h2>
{''.join(block for block in image_blocks[:2])}

<h2>2. Hydrate phase and crystal domains</h2>
<p class="note">Phase is assigned from the cage-centre spatial association:
cages are linked when they share a validated face, the largest connected
component is the crystal domain, and the composition is taken over that domain.
Cages outside the domain are reported as interface / amorphous.</p>
{image_blocks[2] if len(image_blocks) > 2 else '<p class="muted">No phase data.</p>'}

<h2>3. Nucleation</h2>
{image_blocks[3] if len(image_blocks) > 3 else '<p class="muted">No nucleation data.</p>'}

<h2>4. Cage population (last frame)</h2>
{_html_table(cage_header, cage_rows)}

<h2>5. Cage lifetimes</h2>
{_html_table(lifetime_header, lifetime_rows)}

<h2>6. Cage shape quality and guest exchange</h2>
{quality_summary}
{quality_table}

<h2>7. Confirmed phase transitions</h2>
{_html_table(["Frame", "Time (ps)", "From", "To", "Confidence"], phase_transitions)}

<h2>8. Output files</h2>
<p class="note">Raw data is kept in the categorised directories below.
Tables are tab separated with <code>#</code> comment lines and one header line.</p>
{_file_tree(outdir)}

<h2>9. Method notes</h2>
<ul>
<li>Guest: one molecule per centre (residue grouping with topology, covalent-bond
grouping without), see <code>-guest-grouping</code>.</li>
<li>HOP: local orientational order with a smoothed six-phase probability spectrum.</li>
<li>Cage: chordless ring perception on the HOP-gated water network, beam-search
assembly, Euler / connectivity closure check.</li>
<li>Phase: cage-centre lattice coherence, domain size and interface fraction.</li>
</ul>
</body>
</html>
"""

    report_path = os.path.join(outdir, "report.html")

    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(html)

    write_manifest(config, outdir, total_frames, elapsed_time)
    return report_path


def write_manifest(
    config: Any,
    outdir: str,
    total_frames: int = 0,
    elapsed_time: float = 0.0,
) -> str:
    lines = [
        "hytopo output",
        "=============",
        f"top            : {config.trajectory.top}",
        f"traj           : {config.trajectory.traj or '(topology only)'}",
        f"frames analysed: {total_frames}",
        f"wall time      : {elapsed_time:.2f} s",
        f"generated      : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Output layout (all tables are tab separated, '#' marks comments):",
        "",
    ]

    for category in CATEGORY_ORDER:
        directory = os.path.join(outdir, category)
        lines.append(f"  {category}/  - {CATEGORY_TITLES[category]}")

        if not os.path.isdir(directory):
            continue

        for name in sorted(os.listdir(directory)):
            if name.startswith("."):
                continue
            lines.append(f"      {name}")

        lines.append("")

    lines.extend(
        [
            "start here: report.html",
            "",
        ]
    )

    path = os.path.join(outdir, "README.txt")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    return path
