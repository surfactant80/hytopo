# hytopo

**hytopo** analyses the structure and evolution of gas hydrates in molecular
dynamics trajectories. It classifies every water molecule by local order
(HOP), recognises the hydrate cages around guest molecules (standard, distorted
and defective), tracks those cages through time, assigns the hydrate phase from
the spatial association of cage centres, detects nucleation, and writes a
one-page summary report.

The name is the package and the command:

```bash
hytopo -top system.tpr -traj traj.xtc -guest "resname CH4" --find-cages
```

Chinese documentation: [README.zh-CN.md](README.zh-CN.md)

Trajectory support comes entirely from MDAnalysis: **every format it can read
works, there is no whitelist**. Point `-top` and `-traj` at the files — GROMACS
(`.tpr`/`.xtc`/`.trr`), Amber, NAMD/CHARMM (DCD), LAMMPS, `.gro`, `.pdb`,
`.dcd`, `.trj`, … all go through the same MDAnalysis readers. Periodic extended
XYZ (`.xyz`, `.extxyz`, `.etxyz`) is handled by hytopo's own lightweight reader
because that format carries no topology.

---

## Table of contents

1. [Features](#features)
2. [Installation](#installation)
3. [Requirements](#requirements)
4. [Quick start](#quick-start)
5. [Input and selections](#input-and-selections)
6. [Command-line options](#command-line-options)
7. [Output layout](#output-layout)
8. [What the analysers do](#what-the-analysers-do)
9. [Performance](#performance)
10. [Parallelism, checkpoints and long runs](#parallelism-checkpoints-and-long-runs)
11. [Verification and benchmarks](#verification-and-benchmarks)
12. [Python API](#python-api)
13. [Troubleshooting](#troubleshooting)
14. [Citation and license](#citation-and-license)

---

## Features

### 1. Hydrate order parameter (HOP)

Every water oxygen is described by a smoothed local environment (tetrahedrality
`qtet`, staggered/eclipsed bond counts `S`/`E`, coordination) and mapped onto a
continuous six-state probability spectrum:

| label | meaning |
|---|---|
| `liquid` | bulk liquid-like |
| `ice_Ih`, `ice_Ic` | hexagonal / cubic ice-like |
| `hydrate_crystal_like` | crystalline hydrate |
| `hydrate_precursor` | amorphous hydrate / precursor |
| `interface_or_uncertain` | interface or ambiguous |

The per-frame statistics (`H_total`, `H_crys`, `H_pre`, `Hyd_order`, `Ice`,
`Delta`, `Nmax_hyd`, `Nmax_ice`) go to the main time series; per-water labels can
be written to a text table and to a multi-model PDB (label in the B-factor) for
VMD/PyMOL.

### 2. Cage recognition

Hydrate cages are detected around each guest from the local water-oxygen
network:

* chordless rings of 4–6 oxygens are perceived once per frame (Numba DFS), with
  a planarity gate on each ring;
* rings facing a guest are assembled into a closed polyhedron by a beam search
  over faces (bitmask state, Numba kernel), validated by the Euler
  characteristic, edge-sharing and connectivity;
* 13 standard topologies are recognised (`5^12`, `5^12 6^2`, `5^12 6^4`,
  `4^3 5^6 6^3`, `5^12 6^8`, …), plus half cages, partial cages, incipient,
  open and distorted cages;
* a HOP-based candidate mask skips water that cannot belong to a cage, which is
  what makes the search affordable on large systems.

### 3. Guest molecule recognition

A guest is a **molecule**, not an atom:

* with topology (`.tpr`, `.gro`, `.pdb`) the atoms selected by `-guest` are
  grouped by residue;
* without topology (extended XYZ) bonds are inferred from covalent radii with
  `d <= 1.15 (r_i + r_j)` (configurable), using a periodic-aware capped-distance
  neighbour list;
* each molecule contributes one periodic-aware centre, so an all-atom CH4 is one
  guest and a CO2 is one guest.

Selecting only the central atom (`-guest "resname CH4 and name C"`) makes each
guest a single atom — the numbers then match a united-atom model.

### 4. Cage tracking

Cages are linked between frames by water-set Jaccard overlap, guest identity and
type continuity, with a tolerance for missed detections. Outputs:

* cage lifetimes per track,
* cage type transition events and their aggregated summary,
* guest occupancy per cage (empty / single / multiple),
* cage shape quality: mean and spread of the centre-to-vertex radius,
  sphericity `1 - std/mean`, and anisotropy from the vertex covariance,
* guest escape (cage lost) and recapture (cage reformed around the same guest).

### 5. Hydrate phase from cage-centre spatial association

Cages are linked when they share a **validated** face: at least three common
water oxygens, the two centres on opposite sides of the shared polygon, and
`|d(centre,centre) - (d_a + d_b)|` within tolerance. The connected components of
that network are the crystal domains. The phase (`sI`, `sII`, `sH`, `mixed`) is
assigned from the composition of the largest domain, scaled by the lattice
coherence, and cross-checked against the ideal structures shipped in
`hytopo/ideal_crystal/`:

* reference cage-centre distance signatures are extracted from `sI.pdb`,
  `sII.pdb` and `sH.pdb` on first use and cached;
* the observed face-sharing distance distribution is compared with each
  reference (abundance-weighted) so a structure with a few stray `5^12 6^4`
  cages is not mistaken for `sII`;
* the report also gives the domain size (cages), domain radius (Å) and the
  fraction of cages outside the largest domain (interface + amorphous).

The old composition-only decision remains available with `--no-ideal-reference`.

### 6. Nucleation

Nucleation is detected from the largest hydrate-like cluster `Nmax_hyd`
(persistent threshold, adaptive baseline, or a commitment criterion with growth)
and written to a summary plus a per-frame status table.

### 7. Pathway network

Cage transitions are aggregated into a directed weighted network with a
publication-style pathway figure, a side ranking of the strongest transitions,
and machine-readable node/edge tables.

The left-most column is the **guest environment**, not a HOP water label. For
every guest the HOP labels of its first hydration shell (the water oxygens
within `--guest-radius`) are counted: `LIQ` when the majority is liquid-like,
`interface` when the shell is ordered (precursor or crystal-like) but no cage is
detected. The state is smoothed over `ENVIRONMENT_WINDOW = 3` frames, so
single-frame label flicker does not create edges. A detected cage is itself a
state, so `LIQ → incipient → … → crystal` and `crystal → LIQ` are genuine
environment changes of the same guest; tracks are linked between frames by the
Jaccard overlap of their water sets (`--tracking-water-overlap`), guest identity
and type continuity, with `--tracking-gap` frames of tolerance for missed
detections. A track that starts or ends is also counted as an **appearance** or
a **disappearance** in `03_tracking/cage_network_events.dat`. The bookkeeping
`Unmatched` node is only used when HOP is unavailable (`--skip-hop`).

### 8. One-page report and categorised outputs

Every run can write a self-contained `report.html` (English, plots embedded) and
a `README.txt` manifest. All tables are tab separated with `#` comment lines and
one header line, grouped into numbered directories.

### 9. Engineering

* all-core multiprocessing by default (CPU-affinity and cgroup aware), with the
  worker count limited to the number of frames;
* checkpoint / resume (`--checkpoint-every`, `--resume`) that reproduces a clean
  run byte for byte;
* progress bar with rate and ETA;
* Numba kernels with pure-Python fallbacks (`--no-numba`);
* regression harness that fingerprints a frame result and fails loudly if a
  kernel change alters the science.

---

## Installation

### From source (recommended)

```bash
git clone <repository-url>
cd hytopo
pip install .
```

or, for development (editable install):

```bash
pip install -e ".[dev]"
```

### From a wheel

```bash
pip install hytopo-1.0.4-py3-none-any.whl
```

### With conda

```bash
conda create -n hytopo python=3.11
conda activate hytopo
conda install -c conda-forge numpy scipy mdanalysis numba matplotlib networkx
pip install .
```

The command is then available as `hytopo`. Without installing:

```bash
python -m hytopo -h
```

---

## Requirements

| package | version | role |
|---|---|---|
| Python | ≥ 3.10 | |
| numpy | ≥ 1.22 | arrays, linear algebra |
| scipy | ≥ 1.8 | spherical harmonics |
| MDAnalysis | ≥ 2.0 | trajectory I/O, periodic distances |
| numba | ≥ 0.56 | ring perception, cage assembly |
| matplotlib | ≥ 3.5 | plots and the report figures |
| networkx | ≥ 2.8 | transition network |

Optional: `pytest` for the test suite.

Numba is used through cached kernels; a system without numba still runs, just
slower (`--no-numba` forces the Python fallbacks).

---

## Quick start

Minimal HOP-only run:

```bash
hytopo -top system.tpr -traj traj.xtc
```

HOP plus cage search:

```bash
hytopo -top system.tpr -traj traj.xtc -guest "resname CH4" --find-cages
```

Production run with sparse visualisation output:

```bash
hytopo -top system.tpr -traj traj.xtc \
  -start 0 -stop 10001 -stride 2 \
  -guest "resname CH4" --find-cages \
  --cage-pdb-stride 100 --hop-pdb-stride 1000 \
  -outdir hydrate_analysis
```

Cage search only (no HOP):

```bash
hytopo -top system.tpr -traj traj.xtc -guest "resname CH4" \
  --find-cages --skip-hop --no-nucleation
```

Periodic extended XYZ:

```bash
hytopo -top trajectory.etxyz -water "element O" -oxygen "element O" \
  -guest "element C" --find-cages
```

After the run, open `hydrate_analysis/report.html`.

---

## Input and selections

| option | meaning | default |
|---|---|---|
| `-top`, `--top` | topology / structure file (required) | – |
| `-traj`, `--traj` | trajectory file | topology only |
| `-water`, `--water-selection` | water molecules | `resname SOL` |
| `-oxygen`, `--oxygen-selection` | water oxygens | `name OW` |
| `-guest`, `--guest-selection` | guest atoms / molecules | none |
| `-start`, `--start` | first frame | 0 |
| `-stop`, `--stop` | stop frame (exclusive) | end |
| `-stride`, `--stride` | frame stride | 1 |

The oxygen selection is applied as `(water) and (oxygen)`. For extended XYZ the
selection grammar is `all`, `element X`, `name X`, `species X`, `resname X`,
`index a:b`.

Guest grouping (`-guest-grouping`): `auto` (residues with topology, bonds
without), `residue`, `bonded`, `none` (one guest per atom, legacy). The bond
criterion is `d <= scale (r_i + r_j)` with `--guest-bond-scale` (default 1.15).

---

## Command-line options

Run `hytopo -h` for the full list. Groups:

**Analysis mode** — `--find-cages`, `--skip-hop`, `--no-exclude-liquid`,
`--include-ice-for-cage`, `--standard-cages-only`.

**HOP parameters** — `--oo-cutoff` (3.5), `--dist-smooth` (0.15), plus the
thresholds `--hyd-crys-*`, `--hyd-pre-*`, `--ice-q-min`, `--coord-min/max`,
`--pc-hyd`, `--pc-ice`, `--delta-s`, `--c-e-center`, `--sigma-ce`,
`--sigma-count`.

**Cage parameters** — `--cage-oo-cutoff` (3.6), `--guest-radius` (6.2),
`--face-radius` (7.2), `--min-ring`/`--max-ring` (4/6), `--ring-planarity`
(1.5 Å), `--max-local-cycles` (5000 per frame), `--max-rings-per-guest` (56),
`--beam-width` (64), `--candidate-limit` (20), `--cage-dedup-overlap` (0.85),
the partial-cage thresholds, and the candidate-mask probabilities
(`--liquid-exclude-prob`, `--hyd-candidate-prob`, `--ice-candidate-prob`),
`--ring-backend {auto,python,numba}`.

**Tracking** — `--no-tracking`, `--tracking-gap`, `--tracking-water-overlap`,
`--tracking-same-guest-overlap`, `--tracking-dedup-overlap`,
`--occupancy-radius`, `--occupancy-center`.

**Phase** — `--no-phase-analysis`, `--phase-min-cages`,
`--phase-min-confidence`, `--phase-transition-persistence`,
`--phase-mixed-margin`, `--phase-lattice-cutoff`,
`--phase-lattice-tolerance`, `--phase-min-coherence`,
`--no-ideal-reference`, `--phase-reference-weight`.

**Nucleation** — `--no-nucleation`, `--nucleation-method
{fixed,adaptive,commitment}`, `--nucleation-min-size`,
`--nucleation-persist-frames`, `--nucleation-allow-gaps`,
`--nucleation-baseline-frames`, `--nucleation-baseline-sigma`,
`--nucleation-adaptive-min-size`, `--nucleation-min-growth`,
`--nucleation-min-slope`.

**Output** — `-outdir`, `--timeseries`, `--labels`, `--no-labels`,
`--label-stride`, `--hop-pdb`, `--no-hop-pdb`, `--hop-pdb-stride`,
`--cage-stat`, `--shell-stat`, `--cage-pdb-dir`, `--no-cage-pdb`,
`--cage-pdb-stride`, `--no-plots`, `--no-report`.

**Performance** — `--no-numba`, `--serial`, `--n-procs`, `--chunksize`,
`--checkpoint-every`, `--resume`.

**Misc** — `-q/--quiet`, `--verbose-guests`.

---

## Output layout

Outputs are grouped into numbered category directories:

```text
hydrate_analysis/
  report.html                     one-page English summary (self-contained)
  README.txt                      manifest of every file
  final_summary.txt               plain-text summary
  01_timeseries/hydrate_timeseries.dat
  02_cages/cage_stat.dat
  02_cages/guest_shell.dat
  02_cages/cage_center_distances.dat
  03_tracking/cage_lifetime.dat
  03_tracking/cage_transitions.dat
  03_tracking/cage_transition_summary.dat
  03_tracking/cage_occupancy.dat
  03_tracking/cage_quality.dat
  03_tracking/guest_events.dat
  03_tracking/cage_transition_network_{nodes,edges}.dat
  03_tracking/cage_network_events.dat
  04_phase/hydrate_phase_timeseries.dat
  04_phase/hydrate_phase_transitions.dat
  04_phase/phase_domains.dat
  05_nucleation/nucleation_summary.txt
  05_nucleation/nucleation_timeseries.dat
  06_labels/hydrate_labels.dat
  06_labels/hydrate_labels.pdb
  07_cages_pdb/cages_frame_*.pdb
  08_plots/*.png
  09_views/view_*.vmd, view_*.pml
```

Format conventions:

* every table is **tab separated**;
* `#` lines carry comments and units;
* exactly one header line follows the comments;
* `frame` and `time_ps` are the standard first two columns where applicable.

Key columns:

* `hydrate_timeseries.dat` — HOP statistics, `Nmax_hyd`/`Nmax_ice`, one column
  per cage type, `total_cages`.
* `hydrate_phase_timeseries.dat` — `phase`, `confidence`, per-phase scores,
  composition, `domain_size`, `domain_radius`, `interface_fraction`,
  `coherence`, `n_domains`, and the ideal-structure reference scores.
* `phase_domains.dat` — one row per crystal domain (rank, size, phase, score,
  radius).
* `cage_center_distances.dat` — face-sharing centre-centre distances per cage
  type pair (count, mean, median, min, max).
* `cage_quality.dat` — per-cage sphericity and anisotropy.
* `guest_events.dat` — escape / recapture events.

---

### Cage PDB residue names

`07_cages_pdb/cages_frame_*.pdb` stores **one HETATM residue per cage**:

* `resid` is the cage index within the frame;
* the first atom of each residue, `CNT` with residue name `CGE` (element `X`),
  marks the cage centre;
* every water oxygen of that cage carries the cage-type residue name, a
  pseudo-element in the `name`/`element` columns, and the same type code in the
  B-factor column, so VMD/PyMOL can colour by type;
* `CONECT` records follow the ring edges of the cage, and coordinates are
  unwrapped around the centre so a cage crossing the periodic boundary stays
  whole.

PDB residue names are limited to three characters, so cage types that share a
code are distinguished by the B-factor:

| residue | cage type | pseudo-element | B-factor |
|---|---|---|---|
| `SML` | `5^12` | C | 1 |
| `LR1` | `5^12 6^2` | N | 2 |
| `LR2` | `5^12 6^4` | S | 3 |
| `SHM` | `4^3 5^6 6^3` | P | 4 |
| `SHL` | `5^12 6^8` | F | 5 |
| `T13` | `5^12 6^1` | O | 6 |
| `T15` | `5^12 6^3` | B | 7 |
| `T17` | `5^12 6^5` | I | 8 |
| `T18` | `5^12 6^6` | K | 9 |
| `A13` | `4^1 5^10 6^2` | V | 10 |
| `A14` | `4^1 5^10 6^3` | Ti | 11 |
| `A15` | `4^1 5^10 6^4` | Cr | 12 |
| `A16` | `4^2 5^8 6^4` | Mn | 13 |
| `HSM` | `half 5^12` | Xe | 21 |
| `HLR` | `half 5^12 6^2` / `half 5^12 6^4` | Xe | 22 / 23 |
| `HSH` | `half 4^3 5^6 6^3` / `half 5^12 6^8` | Xe | 24 / 25 |
| `HT1` | `half 5^12 6^1` / `6^3` / `6^5` / `6^6` | Xe | 26 / 27 / 28 / 29 |
| `HA1` | `half 4^1 5^10 6^2` / `6^3` / `6^4`, `half 4^2 5^8 6^4` | Xe | 30 / 31 / 32 / 33 |
| `P12` | `partial 5^12` | Al | 45 |
| `P14` | `partial 5^12 6^2` | Si | 46 |
| `P16` | `partial 5^12 6^4` | Cl | 47 |
| `INC` | incipient cage | Y | 41 |
| `OPN` | open cage | Zr | 42 |
| `DST` | distorted cage | Nb | 43 |
| `PAR` | generic partial cage | Xe | 44 |
| `OTH` | other | Xe | 50 |

The same codes appear in the `cage_type` column of the tracking and transition
tables, so a residue name maps one-to-one onto the statistics.


## What the analysers do

**HOP.** Each water oxygen is described by the four nearest neighbours: the
tetrahedrality `qtet` and the correlation `q3m` of the local bond-orientation
vectors. Pair correlations are turned into staggered/eclipsed bond counts,
smoothed over the first neighbour shell (CSR graph, mixing 0.15), and passed
through Fermi/Gaussian gates to obtain the phase propensities. The label is the
most confident state that also passes the structural gates.

**Cages.** Rings are perceived once per frame on the HOP-gated water subgraph.
For each guest the facing rings are scored (`2·alignment − 0.15·planarity −
0.025·distance`) and assembled face by face; closure requires Euler χ = 2,
connected faces and every edge shared by exactly two faces. Non-closing shells
fall back to half/partial/incipient/open/distorted classification.

**Phase.** Face-sharing cage centres form a lattice; the largest connected
component is the crystal domain. Composition is taken over that domain and
blended with the ideal-structure distance check, so isolated or defective cages
cannot flip the phase.

**Nucleation.** `Nmax_hyd` must exceed a threshold persistently (with optional
gaps). The `fixed` method uses `--nucleation-min-size`; `adaptive` derives the
threshold from the initial baseline; `commitment` additionally requires growth
and a positive slope.

---

## Performance

Reference system: 5888 water oxygens, 1024 methane guests, 100 ps frames.

| task | cost |
|---|---|
| HOP + cage + phase kernel | ≈ 0.6 s per frame per core |
| full pipeline (tracking, quality, output) | ≈ 0.9 s per frame on one core |
| same, default all-core pool | ≈ 0.4 s per frame wall clock |
| ring scoring / coverage (batched) | a few ms per frame |

A 5001-frame trajectory (500 ns) needs roughly **1.2 h on one core** or
**25–30 min with 12 workers**.

Practical notes:

* `--cage-pdb-stride 1` writes one PDB per frame (≈ 1.4 MB each); use a stride
  for production.
* `guest_shell.dat` has one row per guest per frame; for long runs consider a
  coarser analysis stride.
* the parent process does the tracking and writing; beyond ~12 workers it
  becomes the ceiling.

---

## Parallelism, checkpoints and long runs

Multiprocessing is on by default. The worker count is
`min(usable CPUs, number of frames)`, where "usable" honours
`HYTOPO_NPROCS`, scheduler affinity, cgroup CPU quotas and SLURM/PBS/SGE
variables. Control it with `--n-procs N`, or disable with `--serial`.

Long runs can be interrupted and resumed:

```bash
hytopo ... --checkpoint-every 200 --outdir run
# ... crash or Ctrl-C ...
hytopo ... --checkpoint-every 200 --outdir run --resume
```

The checkpoint stores the tracker state and the byte offsets of every output
file; on resume the files are truncated to those offsets and the run continues.
A resumed run is byte-identical to an uninterrupted one.

---

## Verification and benchmarks

`hytopo.benchmark_regression` fingerprints a frame result (HOP statistics, label
hash, full cage list, cage counts, complete guest-shell table) and compares it
with a stored reference:

```bash
python -m hytopo.benchmark_regression \
  -top system.tpr -traj traj.xtc -guest "resname CH4" \
  --frames 0,500,1000,1500 --golden benchmarks/golden.json
```

It prints the wall time per frame and `REGRESSION : PASSED/FAILED`. Create a
reference with `--golden file --write-golden`.

`hytopo.benchmark_scaling` measures the local-graph construction on synthetic
million-atom systems.

The test suite:

```bash
python -m pytest hytopo/test_core.py      # or: python -m unittest hytopo.test_core
```

---

## Python API

```python
from hytopo.config import AnalysisConfig, TrajectoryConfig, CageParameters
from hytopo.runner import run

config = AnalysisConfig(
    trajectory=TrajectoryConfig(top="system.tpr", traj="traj.xtc",
                                guest_selection="resname CH4"),
    find_cages=True,
)
run(config)
```

The per-frame building blocks are also importable
(`hytopo.cage.analyze_cages_for_frame`, `hytopo.hop.analyze_hop_from_graph`,
`hytopo.cage_lattice.build_cage_lattice`, `hytopo.phase.classify_phase`,
`hytopo.tracking.CageTracker`).

---

## Troubleshooting

**`No water oxygen atoms found`** — the selection does not match. For GROMACS
the usual pair is `-water "resname SOL" -oxygen "name OW"`; check with
`-oxygen "name O"` or `element O`.

**`No cages found`** — did you pass `--find-cages` and `-guest`? Is the guest
selection correct? Try relaxing the filters with `--no-exclude-liquid` and
`--include-ice-for-cage`, and inspect `02_cages/guest_shell.dat` to confirm the
guests have water shells.

**Nucleation is not detected on a growth trajectory** — the `adaptive` and
`commitment` methods assume the run starts from liquid. For systems that begin
with a hydrate seed, use `--nucleation-method fixed --nucleation-min-size ...`
or `--no-nucleation`.

**Very slow or out of memory** — check `--n-procs` (each worker loads the
topology itself), reduce `--max-rings-per-guest`/`--beam-width`, or set
`--cage-pdb-stride`/`--hop-pdb-stride` so fewer PDB frames are written.

**Numba problems** — `--no-numba` falls back to the equivalent Python kernels.

**MDAnalysis cannot read the trajectory** — for GROMACS use `.tpr` + `.xtc`;
the topology and trajectory must describe the same atoms.

**The network figure is crowded** — reduce `min_edge_weight`/`max_edges`, raise
`edge_label_min_weight`, or set `show_transition_panel = True` in
`hytopo/network.py` (all edges stay, the numbers move to the side panel). The
figure can be regenerated without re-running the analysis:

```python
from hytopo.network import generate_network_from_tables
generate_network_from_tables("run/03_tracking", outdir="run")
```

---

## Citation and license

If you use hytopo in published work, please cite the preprint:

> Zherui Chen, Shuquan Huang, Ke Xu, Zhanbo Li, Xinyu Song, Haijun Wang,
> Dingyuan Tang, Qingxia Liu, Yajun Deng. *HyTopo: A High-Performance
> Hierarchical Algorithm for Mapping the Structural Evolution and Transition
> Networks of Gas Hydrates*. arXiv:2609.37036 (2026).
> <https://arxiv.org/abs/2609.37036>

```bibtex
@misc{chen2026hytopo,
  title  = {HyTopo: A High-Performance Hierarchical Algorithm for Mapping the
            Structural Evolution and Transition Networks of Gas Hydrates},
  author = {Chen, Zherui and Huang, Shuquan and Xu, Ke and Li, Zhanbo and
            Song, Xinyu and Wang, Haijun and Tang, Dingyuan and Liu, Qingxia
            and Deng, Yajun},
  year   = {2026},
  eprint = {2609.37036},
  archivePrefix = {arXiv},
  primaryClass = {physics.comp-ph},
  url    = {https://arxiv.org/abs/2609.37036}
}
```

**License.** hytopo is distributed under the **GNU Lesser General Public
License, version 3 or (at your option) any later version** (LGPL-3.0-or-later);
see `LICENSE` (LGPL-3.0) and `COPYING` (GPL-3.0, which the LGPL incorporates by
reference). You may use hytopo as a library in your own code — including
proprietary code — and redistribute modified versions of hytopo under the LGPL.
