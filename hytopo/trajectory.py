from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Optional

import numpy as np
import MDAnalysis as mda

from .config import AnalysisConfig, TrajectoryConfig
from .extxyz import ExtXYZReader, select_extxyz_atoms
from .guest import (
    DEFAULT_BOND_SCALE,
    GuestGrouping,
    group_centres,
    grouping_by_bonds,
    grouping_by_residue,
    should_use_bonded,
)


@dataclass(frozen=True)
class AtomMetadata:
    atom_ids: np.ndarray
    atom_names: tuple[str, ...]
    resnames: tuple[str, ...]
    resids: tuple[int, ...]
    elements: tuple[str, ...]


@dataclass(frozen=True)
class FrameData:
    frame_index: int
    time_ps: float
    box: np.ndarray
    oxygen_positions: np.ndarray
    guest_positions: Optional[np.ndarray]


@dataclass
class TrajectoryContext:
    universe: Optional[mda.Universe]
    oxygen_atoms: Any
    guest_atoms: Any
    atom_metadata: AtomMetadata
    frames: list[int]
    extxyz_reader: Optional[ExtXYZReader] = None
    extxyz_oxygen_indices: Optional[np.ndarray] = None
    extxyz_guest_indices: Optional[np.ndarray] = None
    guest_grouping: Optional[GuestGrouping] = None

    @property
    def n_oxygen(self) -> int:
        if self.extxyz_reader is not None:
            return 0 if self.extxyz_oxygen_indices is None else len(self.extxyz_oxygen_indices)
        return len(self.oxygen_atoms)

    @property
    def n_guest(self) -> int:
        """Number of guest molecules (one centre each)."""
        if self.guest_grouping is not None:
            return self.guest_grouping.n_groups
        return self.n_guest_atoms

    @property
    def n_guest_atoms(self) -> int:
        if self.extxyz_reader is not None:
            return 0 if self.extxyz_guest_indices is None else len(self.extxyz_guest_indices)
        if self.guest_atoms is None:
            return 0
        return len(self.guest_atoms)


def safe_int(value, fallback: int = 1) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def safe_string(value, fallback: str = "X") -> str:
    try:
        text = str(value)
        if len(text) == 0:
            return fallback
        return text
    except Exception:
        return fallback


def load_universe(config: TrajectoryConfig) -> mda.Universe:
    if config.traj:
        return mda.Universe(config.top, config.traj)
    return mda.Universe(config.top)


def select_oxygen_atoms(universe: mda.Universe, config: TrajectoryConfig):
    oxygen_atoms = universe.select_atoms(config.oxygen_query)

    if len(oxygen_atoms) == 0:
        raise RuntimeError(
            f"No water oxygen atoms found. Check selection: {config.oxygen_query}"
        )

    return oxygen_atoms


def select_guest_atoms(
    universe: mda.Universe,
    config: TrajectoryConfig,
    require_guest: bool,
):
    if not config.guest_selection:
        if require_guest:
            raise RuntimeError("Guest selection is required when cage search is enabled.")
        return None

    guest_atoms = universe.select_atoms(config.guest_selection)

    if len(guest_atoms) == 0:
        raise RuntimeError(
            f"No guest atoms found. Check selection: {config.guest_selection}"
        )

    return guest_atoms


def guest_symbols(atoms) -> list[str]:
    """Element symbols for a guest AtomGroup, falling back to atom names."""
    symbols = []

    for atom in atoms:
        element = str(getattr(atom, "element", "") or "").strip()
        symbols.append(element if element else str(getattr(atom, "name", "") or ""))

    return symbols


def build_topology_guest_grouping(
    guest_atoms,
    universe: mda.Universe,
    config: AnalysisConfig,
) -> Optional[GuestGrouping]:
    """Group guest atoms into molecules using the input topology."""
    if guest_atoms is None or len(guest_atoms) == 0:
        return None

    method = getattr(config.trajectory, "guest_grouping", "auto")

    if method == "none":
        return None

    scale = getattr(config.trajectory, "guest_bond_scale", DEFAULT_BOND_SCALE)

    if should_use_bonded(method, has_topology=True):
        return grouping_by_bonds(
            guest_atoms.positions,
            guest_symbols(guest_atoms),
            universe.dimensions,
            scale,
        )

    return grouping_by_residue(guest_atoms.resindices)


def build_extxyz_guest_grouping(
    reader: ExtXYZReader,
    first,
    guest_indices: Optional[np.ndarray],
    config: AnalysisConfig,
) -> Optional[GuestGrouping]:
    """Group guest atoms into molecules by inferred bonds (no topology)."""
    if guest_indices is None or len(guest_indices) == 0:
        return None

    method = getattr(config.trajectory, "guest_grouping", "auto")

    if method == "none":
        return None

    if method == "residue" and not getattr(config.runtime, "quiet", False):
        print(
            "Note: EXTXYZ input carries no residue information; "
            "using bond-based guest grouping.",
            flush=True,
        )

    scale = getattr(config.trajectory, "guest_bond_scale", DEFAULT_BOND_SCALE)
    positions = first.positions[np.asarray(guest_indices, dtype=np.int64)]
    symbols = [first.symbols[int(index)] for index in guest_indices]

    return grouping_by_bonds(positions, symbols, first.box, scale)


def build_frame_list(universe: mda.Universe, config: TrajectoryConfig) -> list[int]:
    return build_frame_list_from_count(len(universe.trajectory), config)


def build_frame_list_from_count(
    n_frames: int,
    config: TrajectoryConfig,
) -> list[int]:

    start = min(config.start, n_frames)
    stop = n_frames if config.stop is None else min(config.stop, n_frames)

    if stop <= start:
        return []

    return list(range(start, stop, config.stride))


def collect_atom_metadata(oxygen_atoms) -> AtomMetadata:
    atom_ids = oxygen_atoms.ids.copy()

    atom_names = []
    resnames = []
    resids = []
    elements = []

    for local_index, atom in enumerate(oxygen_atoms):
        atom_name = safe_string(getattr(atom, "name", ""), "OW")
        resname = safe_string(getattr(atom, "resname", ""), "SOL")
        resid = safe_int(getattr(atom, "resid", local_index + 1), local_index + 1)

        element = safe_string(getattr(atom, "element", ""), "")
        if len(element.strip()) == 0:
            stripped = atom_name.strip()
            element = stripped[0] if stripped else "O"

        atom_names.append(atom_name)
        resnames.append(resname)
        resids.append(resid)
        elements.append(element)

    return AtomMetadata(
        atom_ids=atom_ids,
        atom_names=tuple(atom_names),
        resnames=tuple(resnames),
        resids=tuple(resids),
        elements=tuple(elements),
    )


def is_extxyz_input(config: TrajectoryConfig) -> bool:
    extensions = (".xyz", ".extxyz", ".etxyz")
    path = config.traj if config.traj else config.top
    return isinstance(path, str) and path.lower().endswith(extensions)


def create_extxyz_context(config: AnalysisConfig) -> TrajectoryContext:
    extensions = (".xyz", ".extxyz", ".etxyz")
    if config.trajectory.traj and config.trajectory.traj.lower().endswith(extensions):
        path = config.trajectory.traj
    else:
        path = config.trajectory.top

    reader = ExtXYZReader(path)
    first = reader.read(0)
    if np.any(first.box[:3] <= 0.0):
        raise RuntimeError(
            "Periodic EXTXYZ input requires a valid 3x3 Lattice field."
        )
    oxygen_indices = select_extxyz_atoms(
        config.trajectory.oxygen_query,
        first.symbols,
        first.properties,
        oxygen_fallback=True,
    )

    if len(oxygen_indices) == 0:
        raise RuntimeError(
            f"No oxygen atoms found in EXTXYZ selection: "
            f"{config.trajectory.oxygen_query}"
        )

    if config.trajectory.guest_selection:
        guest_indices = select_extxyz_atoms(
            config.trajectory.guest_selection,
            first.symbols,
            first.properties,
        )
        if len(guest_indices) == 0:
            raise RuntimeError(
                f"No guest atoms found in EXTXYZ selection: "
                f"{config.trajectory.guest_selection}. "
                "Use e.g. 'element C' or 'index 0:2'."
            )
    else:
        guest_indices = None

    frames = build_frame_list_from_count(len(reader), config.trajectory)
    if not frames:
        raise RuntimeError("No frames selected. Check start/stop/stride.")

    guest_grouping = build_extxyz_guest_grouping(
        reader=reader,
        first=first,
        guest_indices=guest_indices,
        config=config,
    )

    selected_symbols = tuple(first.symbols[int(index)] for index in oxygen_indices)
    atom_metadata = AtomMetadata(
        atom_ids=np.arange(1, len(oxygen_indices) + 1, dtype=np.int64),
        atom_names=selected_symbols,
        resnames=selected_symbols,
        resids=tuple(range(1, len(oxygen_indices) + 1)),
        elements=selected_symbols,
    )

    return TrajectoryContext(
        universe=None,
        oxygen_atoms=None,
        guest_atoms=None,
        atom_metadata=atom_metadata,
        frames=frames,
        extxyz_reader=reader,
        extxyz_oxygen_indices=oxygen_indices,
        extxyz_guest_indices=guest_indices,
        guest_grouping=guest_grouping,
    )


def create_context(config: AnalysisConfig) -> TrajectoryContext:
    if is_extxyz_input(config.trajectory):
        return create_extxyz_context(config)

    universe = load_universe(config.trajectory)

    oxygen_atoms = select_oxygen_atoms(universe, config.trajectory)
    guest_atoms = select_guest_atoms(
        universe=universe,
        config=config.trajectory,
        require_guest=config.find_cages,
    )

    frames = build_frame_list(universe, config.trajectory)
    if len(frames) == 0:
        raise RuntimeError("No frames selected. Check start/stop/stride.")

    atom_metadata = collect_atom_metadata(oxygen_atoms)
    guest_grouping = build_topology_guest_grouping(guest_atoms, universe, config)

    return TrajectoryContext(
        universe=universe,
        oxygen_atoms=oxygen_atoms,
        guest_atoms=guest_atoms,
        atom_metadata=atom_metadata,
        frames=frames,
        guest_grouping=guest_grouping,
    )


def read_frame(context: TrajectoryContext, frame_index: int) -> FrameData:
    if context.extxyz_reader is not None:
        frame = context.extxyz_reader.read(int(frame_index))
        oxygen_indices = context.extxyz_oxygen_indices
        guest_indices = context.extxyz_guest_indices

        if oxygen_indices is None:
            raise RuntimeError("EXTXYZ oxygen selection is not initialized.")

        if guest_indices is None:
            guest_positions = None
        elif context.guest_grouping is not None:
            guest_positions = group_centres(
                frame.positions[guest_indices],
                context.guest_grouping,
                frame.box,
            )
        else:
            guest_positions = frame.positions[guest_indices].astype(np.float64, copy=True)

        return FrameData(
            frame_index=int(frame.frame_index),
            time_ps=float(frame.time_ps),
            box=frame.box.copy(),
            oxygen_positions=frame.positions[oxygen_indices].astype(np.float64, copy=True),
            guest_positions=guest_positions,
        )

    if context.universe is None:
        raise RuntimeError("Trajectory context has no active reader.")

    context.universe.trajectory[frame_index]
    ts = context.universe.trajectory.ts

    oxygen_positions = context.oxygen_atoms.positions.astype(np.float64, copy=True)

    if context.guest_atoms is None:
        guest_positions = None
    elif context.guest_grouping is not None:
        guest_positions = group_centres(
            context.guest_atoms.positions,
            context.guest_grouping,
            np.asarray(ts.dimensions, dtype=np.float64),
        )
    else:
        guest_positions = context.guest_atoms.positions.astype(np.float64, copy=True)

    return FrameData(
        frame_index=int(ts.frame),
        time_ps=float(ts.time),
        box=np.asarray(ts.dimensions, dtype=np.float64).copy(),
        oxygen_positions=oxygen_positions,
        guest_positions=guest_positions,
    )


def iter_frames(context: TrajectoryContext) -> Iterator[FrameData]:
    for frame_index in context.frames:
        yield read_frame(context, frame_index)


# ============================================================
# Multiprocessing worker support
# ============================================================

_WORKER_CONTEXT: Optional[TrajectoryContext] = None


def init_worker(config: AnalysisConfig) -> None:
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = create_context(config)


def get_worker_context() -> TrajectoryContext:
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Worker context is not initialized.")
    return _WORKER_CONTEXT


def read_worker_frame(frame_index: int) -> FrameData:
    context = get_worker_context()
    return read_frame(context, frame_index)


def summarize_context(context: TrajectoryContext) -> dict:
    return {
        "n_frames_selected": len(context.frames),
        "first_frame": context.frames[0] if context.frames else None,
        "last_frame": context.frames[-1] if context.frames else None,
        "n_oxygen": context.n_oxygen,
        "n_guest": context.n_guest,
        "n_guest_atoms": context.n_guest_atoms,
        "guest_grouping": (
            context.guest_grouping.method if context.guest_grouping is not None else "none"
        ),
    }


def print_context_summary(context: TrajectoryContext, quiet: bool = False) -> None:
    if quiet:
        return

    summary = summarize_context(context)

    print(f"selected frames:     {summary['n_frames_selected']}")
    print(f"first frame:         {summary['first_frame']}")
    print(f"last frame:          {summary['last_frame']}")
    print(f"oxygen atoms:        {summary['n_oxygen']}")
    print(f"guest atoms:         {summary['n_guest_atoms']}")
    print(
        f"guest molecules:     {summary['n_guest']} "
        f"(grouping: {summary['guest_grouping']})"
    )
    print("")
