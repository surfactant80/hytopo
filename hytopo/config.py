from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .cpus import usable_cpu_count
from .output_layout import (
    CAGE_PDB,
    CAGES,
    LABELS,
    TIMESERIES,
    categorised_path,
    ensure_category_dirs,
)
from .nucleation import NucleationParameters
from .phase import PhaseParameters
from .tracking import TrackingParameters as TrackingConfig


# ============================================================
# Label definition
# ============================================================

LABEL_LIQ = 0
LABEL_IH = 1
LABEL_IC = 2
LABEL_HYD = 3
LABEL_PRE = 4
LABEL_UNK = 5

LABEL_NAMES = {
    LABEL_LIQ: "liquid",
    LABEL_IH: "ice_Ih",
    LABEL_IC: "ice_Ic",
    LABEL_HYD: "hydrate_crystal_like",
    LABEL_PRE: "hydrate_precursor_or_amorphous_hydrate",
    LABEL_UNK: "interface_or_uncertain",
}


# ============================================================
# Cage type definition
# ============================================================


TARGETS = [
    ((0, 12, 0), "5^12"),
    ((0, 12, 2), "5^12_6^2"),
    ((0, 12, 4), "5^12_6^4"),
    ((3, 6, 3), "4^3_5^6_6^3"),
    ((0, 12, 8), "5^12_6^8"),
    ((0, 12, 1), "5^12_6^1"),
    ((0, 12, 3), "5^12_6^3"),
    ((0, 12, 5), "5^12_6^5"),
    ((0, 12, 6), "5^12_6^6"),
    ((1, 10, 2), "4^1_5^10_6^2"),
    ((1, 10, 3), "4^1_5^10_6^3"),
    ((1, 10, 4), "4^1_5^10_6^4"),
    ((2, 8, 4), "4^2_5^8_6^4"),
]

COMPLETE_TYPES = [name for _, name in TARGETS]

# 对应生成的半笼列表
HALF_TYPES = [f"half_{name}" for name in COMPLETE_TYPES]

# 其他特殊不完整笼
SPECIAL_PARTIALS = [
    "partial_5^12",
    "partial_5^12_6^2",
    "partial_5^12_6^4",
    "incipient_cage",
    "open_cage",
    "distorted_cage",
    "partial",
]

CAGE_TYPES = COMPLETE_TYPES + HALF_TYPES + SPECIAL_PARTIALS + ["other"]

# 优先级设定 (确保统计和网络图的逻辑层级)
TYPE_PRIORITY = {}
for name in COMPLETE_TYPES:
    TYPE_PRIORITY[name] = 100
for name in HALF_TYPES:
    TYPE_PRIORITY[name] = 60

TYPE_PRIORITY.update({
    "partial_5^12": 32,
    "partial_5^12_6^2": 31,
    "partial_5^12_6^4": 30,
    "incipient_cage": 45,
    "open_cage": 40,
    "distorted_cage": 35,
    "partial": 20,
    "other": 0,
})

# 可视化 Beta 因子 (用于 VMD 染色)
CAGE_BETA = {}
# 完整笼使用 1-19
for i, name in enumerate(COMPLETE_TYPES):
    CAGE_BETA[name] = float(i + 1)
# 半笼使用 21-39 (与完整笼一一对应，方便记忆)
for i, name in enumerate(HALF_TYPES):
    CAGE_BETA[name] = float(i + 21)

CAGE_BETA.update({
    "partial_5^12": 45.0,
    "partial_5^12_6^2": 46.0,
    "partial_5^12_6^4": 47.0,
    "incipient_cage": 41.0,
    "open_cage": 42.0,
    "distorted_cage": 43.0,
    "partial": 44.0,
    "other": 50.0,
})

# 元素分配 (用于 PDB 渲染)
CAGE_ELEMENT = {
    "5^12": "C", "5^12_6^2": "N", "5^12_6^4": "S", 
    "4^3_5^6_6^3": "P", "5^12_6^8": "F",
    "5^12_6^1": "O", "5^12_6^3": "B", "5^12_6^5": "I", "5^12_6^6": "K",
    "4^1_5^10_6^2": "V", "4^1_5^10_6^3": "Ti", "4^1_5^10_6^4": "Cr", "4^2_5^8_6^4": "Mn",
}
# 半笼统一使用金属元素或稀有气体以便区分
for name in HALF_TYPES:
    CAGE_ELEMENT[name] = "Xe" 

CAGE_ELEMENT.update({
    "partial_5^12": "Al", "partial_5^12_6^2": "Si", "partial_5^12_6^4": "Cl",
    "incipient_cage": "Y", "open_cage": "Zr", "distorted_cage": "Nb", "partial": "Xe", "other": "Xe"
})

# 残基名称 (Resname 限制 3 字符)
CAGE_RESNAME = {}
# 完整笼
res_map = {
    "5^12": "SML", "5^12_6^2": "LR1", "5^12_6^4": "LR2", 
    "4^3_5^6_6^3": "SHM", "5^12_6^8": "SHL",
    "5^12_6^1": "T13", "5^12_6^3": "T15", "5^12_6^5": "T17", "5^12_6^6": "T18",
    "4^1_5^10_6^2": "A13", "4^1_5^10_6^3": "A14", "4^1_5^10_6^4": "A15", "4^2_5^8_6^4": "A16",
}
CAGE_RESNAME.update(res_map)
# 半笼 (前缀加 H)
for name in COMPLETE_TYPES:
    full_res = res_map.get(name, "OTH")
    # 取前两位加 H，如 SML -> HSML -> HSM
    half_res = ("H" + full_res)[:3]
    CAGE_RESNAME[f"half_{name}"] = half_res

CAGE_RESNAME.update({
    "partial_5^12": "P12", "partial_5^12_6^2": "P14", "partial_5^12_6^4": "P16",
    "incipient_cage": "INC", "open_cage": "OPN", "distorted_cage": "DST", "partial": "PAR", "other": "OTH"
})


# ============================================================
# HOP structural parameters
# ============================================================

@dataclass(frozen=True)
class HopParameters:
    oo_cutoff: float = 3.5
    dist_smooth: float = 0.15

    hyd_crys_e_center: float = 3.55
    hyd_crys_e_width: float = 0.28
    hyd_crys_s_max: float = 1.10
    hyd_crys_q_min: float = 0.45
    hyd_crys_raw_min: float = 0.12
    hyd_crys_support_min: float = 0.10

    hyd_pre_e_low: float = 2.45
    hyd_pre_e_high: float = 3.40
    hyd_pre_s_max: float = 1.20
    hyd_pre_q_min: float = 0.55
    hyd_pre_raw_min: float = 0.18
    hyd_pre_support_min: float = 0.12
    hyd_pre_neighbor_fraction_min: float = 0.20

    ice_q_min: float = 0.35

    coord_min: float = 3.25
    coord_max: float = 4.85

    pc_hyd: float = 0.50
    pc_ice: float = 0.50

    delta_s: float = 0.06
    c_e_center: float = -0.05
    sigma_ce: float = 0.22
    sigma_count: float = 0.65
    eps: float = 1.0e-12


# ============================================================
# Cage structural parameters
# ============================================================

@dataclass(frozen=True)
class CageParameters:
    oo_cutoff: float = 3.6
    guest_radius: float = 6.2
    face_radius: float = 7.2

    min_ring: int = 4
    max_ring: int = 6
    ring_planarity: float = 1.5
    face_normal_cut: float = 0.0

    max_local_cycles: int = 5000
    max_rings_per_guest: int = 56
    beam_width: int = 64
    candidate_limit: int = 20
    dedup_water_overlap: float = 0.85

    standard_only: bool = False

    partial_min_faces: int = 5
    partial_min_closure: float = 0.25
    half_min_faces: int = 6
    half_min_coverage: float = 0.30
    half_max_coverage: float = 0.85
    incipient_min_closure: float = 0.30
    open_min_boundary: int = 6

    exclude_liquid: bool = True
    include_ice: bool = False
    liquid_exclude_prob: float = 0.70
    hyd_candidate_prob: float = 0.25
    ice_candidate_prob: float = 0.50

    ring_backend: str = "auto"


# ============================================================
# Output parameters
# ============================================================

@dataclass(frozen=True)
class OutputConfig:
    outdir: str = "hydrate_analysis"

    timeseries: str = "hydrate_timeseries.dat"
    labels: str = "hydrate_labels.dat"

    write_labels: bool = True
    label_stride: int = 100

    write_hop_pdb: bool = True
    hop_pdb: str = "hydrate_labels.pdb"
    hop_pdb_stride: int = 1

    write_cage_pdb: bool = True
    #: sub-directory inside 07_cages_pdb; empty means the category directory
    cage_pdb_dir: str = ""
    cage_pdb_stride: int = 1

    cage_stat: str = "cage_stat.dat"
    shell_stat: str = "guest_shell.dat"
    make_plots: bool = True
    write_report: bool = True

    def path(self, filename: str, category: Optional[str] = None) -> str:
        """Path of an output file inside its category directory."""
        return categorised_path(self.outdir, filename, category)

    @property
    def timeseries_path(self) -> str:
        return self.path(self.timeseries, TIMESERIES)

    @property
    def labels_path(self) -> str:
        return self.path(self.labels, LABELS)

    @property
    def hop_pdb_path(self) -> str:
        return self.path(self.hop_pdb, LABELS)

    @property
    def cage_stat_path(self) -> str:
        return self.path(self.cage_stat, CAGES)

    @property
    def shell_stat_path(self) -> str:
        return self.path(self.shell_stat, CAGES)

    @property
    def cage_pdb_path(self) -> str:
        base = os.path.join(self.outdir, CAGE_PDB)

        if self.cage_pdb_dir and self.cage_pdb_dir != CAGE_PDB:
            return os.path.join(base, self.cage_pdb_dir)

        return base


# ============================================================
# Runtime / trajectory parameters
# ============================================================

@dataclass(frozen=True)
class RuntimeConfig:
    use_numba: bool = True
    #: Frame-level multiprocessing is on by default and uses every logical core
    #: unless ``n_processes`` is set explicitly.
    use_parallel: bool = True
    n_processes: int = 0
    chunksize: int = 1
    quiet: bool = False
    verbose_guests: bool = False
    #: write a resumable checkpoint every N analysed frames (0 disables)
    checkpoint_every: int = 0
    #: continue from an existing checkpoint
    resume: bool = False

    def resolved_n_processes(self) -> int:
        if not self.use_parallel:
            return 1
        if self.n_processes > 0:
            return self.n_processes
        return usable_cpu_count()


@dataclass(frozen=True)
class TrajectoryConfig:
    top: str
    traj: Optional[str] = None

    water_selection: str = "resname SOL"
    oxygen_selection: str = "name OW"
    guest_selection: Optional[str] = None

    # Guest atoms are grouped into molecules before cage analysis:
    #   auto    - residues when the input carries topology, bonds otherwise
    #   residue - always use MD residues (topology)
    #   bonded  - always infer covalent bonds from geometry
    #   none    - every selected atom is its own guest (legacy behaviour)
    guest_grouping: str = "auto"
    guest_bond_scale: float = 1.15

    start: int = 0
    stop: Optional[int] = None
    stride: int = 1

    @property
    def oxygen_query(self) -> str:
        return f"({self.water_selection}) and ({self.oxygen_selection})"


# ============================================================
# Top-level config
# ============================================================

@dataclass(frozen=True)
class AnalysisConfig:
    trajectory: TrajectoryConfig
    output: OutputConfig = OutputConfig()
    hop: HopParameters = HopParameters()
    cage: CageParameters = CageParameters()
    runtime: RuntimeConfig = RuntimeConfig()
    nucleation: NucleationParameters = NucleationParameters()
    tracking: TrackingConfig = TrackingConfig()
    phase: PhaseParameters = PhaseParameters()

    run_hop: bool = True
    find_cages: bool = False

    def validate(self) -> None:
        if self.find_cages and not self.trajectory.guest_selection:
            raise ValueError("--find-cages requires -guest/--guest-selection.")

        if self.trajectory.stop is not None and self.trajectory.stop <= self.trajectory.start:
            raise ValueError("--stop must be larger than --start.")

        if self.trajectory.stride <= 0:
            raise ValueError("--stride must be positive.")

        if self.trajectory.guest_grouping not in (
            "auto",
            "residue",
            "bonded",
            "none",
        ):
            raise ValueError("guest_grouping must be auto/residue/bonded/none.")

        if self.trajectory.guest_bond_scale <= 0.0:
            raise ValueError("guest_bond_scale must be positive.")

        if self.runtime.checkpoint_every < 0:
            raise ValueError("checkpoint_every must be non-negative.")

        if self.output.label_stride <= 0:
            raise ValueError("label_stride must be positive.")

        if self.output.hop_pdb_stride <= 0:
            raise ValueError("hop_pdb_stride must be positive.")

        if self.output.cage_pdb_stride <= 0:
            raise ValueError("cage_pdb_stride must be positive.")

        if self.hop.oo_cutoff <= 0.0:
            raise ValueError("HOP O-O cutoff must be positive.")

        if self.cage.oo_cutoff <= 0.0:
            raise ValueError("cage O-O cutoff must be positive.")

        if self.cage.min_ring <= 0:
            raise ValueError("min_ring must be positive.")

        if self.cage.max_ring <= 0:
            raise ValueError("max_ring must be positive.")

        if self.cage.min_ring > self.cage.max_ring:
            raise ValueError("min_ring cannot be larger than max_ring.")

        if self.cage.ring_backend not in ("auto", "python", "numba"):
            raise ValueError("ring_backend must be one of auto/python/numba.")

        if self.cage.max_local_cycles <= 0:
            raise ValueError("max_local_cycles must be positive.")

        if self.cage.max_rings_per_guest <= 0:
            raise ValueError("max_rings_per_guest must be positive.")

        if self.cage.beam_width <= 0 or self.cage.candidate_limit <= 0:
            raise ValueError("beam_width and candidate_limit must be positive.")

        if not 0.0 <= self.cage.dedup_water_overlap <= 1.0:
            raise ValueError("cage dedup_water_overlap must be between 0 and 1.")

        if not 0.0 <= self.cage.half_min_coverage <= self.cage.half_max_coverage <= 1.0:
            raise ValueError("half coverage thresholds must satisfy 0 <= min <= max <= 1.")

        if self.nucleation.min_cluster_size <= 0:
            raise ValueError("nucleation_min_size must be positive.")

        if self.nucleation.persist_frames <= 0:
            raise ValueError("nucleation_persist_frames must be positive.")

        if self.nucleation.allow_gaps < 0:
            raise ValueError("nucleation_allow_gaps must be non-negative.")

        if self.nucleation.method not in ("fixed", "adaptive", "commitment"):
            raise ValueError("nucleation method must be fixed/adaptive/commitment.")

        if self.nucleation.baseline_frames <= 1:
            raise ValueError("nucleation baseline_frames must be larger than 1.")

        if self.nucleation.baseline_sigma <= 0.0:
            raise ValueError("nucleation baseline_sigma must be positive.")

        if self.nucleation.adaptive_min_size <= 0:
            raise ValueError("nucleation adaptive_min_size must be positive.")

        if self.tracking.gap_tolerance < 0:
            raise ValueError("tracking gap_tolerance must be non-negative.")

        for name, value in (
            ("tracking min_water_overlap", self.tracking.min_water_overlap),
            ("tracking min_same_guest_overlap", self.tracking.min_same_guest_overlap),
            ("tracking dedup_water_overlap", self.tracking.dedup_water_overlap),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if self.tracking.occupancy_radius <= 0.0:
            raise ValueError("tracking occupancy_radius must be positive.")

        if self.tracking.occupancy_center not in ("cage", "water"):
            raise ValueError("tracking occupancy_center must be cage or water.")

        if self.phase.min_complete_cages <= 0:
            raise ValueError("phase min_complete_cages must be positive.")

        if not 0.0 <= self.phase.min_confidence <= 1.0:
            raise ValueError("phase min_confidence must be between 0 and 1.")

        if self.phase.transition_persist_frames <= 0:
            raise ValueError("phase transition_persist_frames must be positive.")

        if not 0.0 <= self.phase.mixed_margin <= 1.0:
            raise ValueError("phase mixed_margin must be between 0 and 1.")

    def ensure_output_dirs(self) -> None:
        os.makedirs(self.output.outdir, exist_ok=True)
        ensure_category_dirs(self.output.outdir)

        if self.find_cages and self.output.write_cage_pdb:
            os.makedirs(self.output.cage_pdb_path, exist_ok=True)

    def log(self) -> None:
        if self.runtime.quiet:
            return

        print("==== hytopo ====")
        print(f"top:                 {self.trajectory.top}")
        print(f"traj:                {self.trajectory.traj if self.trajectory.traj else self.trajectory.top}")
        print(f"oxygen query:        {self.trajectory.oxygen_query}")
        print(f"guest selection:     {self.trajectory.guest_selection or 'none'}")
        print(f"guest grouping:      {self.trajectory.guest_grouping}")
        if self.trajectory.guest_grouping in ("auto", "bonded"):
            print(f"guest bond scale:    {self.trajectory.guest_bond_scale}")
        print(
            "frames:              "
            f"start={self.trajectory.start} "
            f"stop={self.trajectory.stop} "
            f"stride={self.trajectory.stride}"
        )
        print(f"run HOP:             {self.run_hop}")
        print(f"find cages:          {self.find_cages}")
        print(f"exclude liquid:      {self.cage.exclude_liquid}")
        print(f"include ice:         {self.cage.include_ice}")
        print(f"HOP PDB stride:      {self.output.hop_pdb_stride if self.output.write_hop_pdb else 'disabled'}")
        print(f"cage PDB stride:     {self.output.cage_pdb_stride if self.output.write_cage_pdb else 'disabled'}")
        print(f"nucleation:          {self.nucleation.enabled}")
        if self.nucleation.enabled:
            print(f"nuc method:          {self.nucleation.method}")
            print(f"nuc min Nmax_hyd:    {self.nucleation.min_cluster_size}")
            print(f"nuc persist frames:  {self.nucleation.persist_frames}")
            print(f"nuc allow gaps:      {self.nucleation.allow_gaps}")
        print(f"parallel:            {self.runtime.use_parallel}")
        print(f"processes:           {self.runtime.resolved_n_processes()}")
        print(f"numba:               {self.runtime.use_numba}")
        print(f"ring backend:        {self.cage.ring_backend}")
        print(f"tracking:            {self.tracking.enabled and self.find_cages}")
        print(f"plots:               {self.output.make_plots}")
        print(f"phase analysis:      {self.phase.enabled and self.find_cages}")
        print(f"outdir:              {self.output.outdir}")
        print("")
