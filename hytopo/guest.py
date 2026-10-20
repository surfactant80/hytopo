from __future__ import annotations

"""Guest molecule recognition.

A guest is a molecule, not an atom.  In an all-atom force field one CH4 is five
atoms and one CO2 is three atoms; treating every selected atom as an independent
guest centre both inflates the guest count and detaches the geometric centre
from the molecule.  This module groups the selected guest atoms into molecules
and returns one centre per molecule.

Two sources of information are supported:

``residue``
    The topology (``.tpr``, ``.gro``, ``.pdb``, ...) already says which atoms
    belong together: MD residues are molecules for the species that end up as
    guests.  No geometry is involved, so it is cheap and exact.

``bonded``
    For files without topology (extended XYZ) bonds are inferred from
    interatomic distances with covalent radii, honouring periodic boundary
    conditions and using a capped-distance neighbour list so the cost stays
    linear in the number of guest atoms.

Both paths return the same :class:`GuestGrouping`, whose centres are computed
with periodic-aware unwrapping, so a molecule straddling the box boundary still
gets a physical centre.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from MDAnalysis.lib.distances import self_capped_distance

from .geometry import pbc_delta


# Cordero et al., Dalton Trans. 2008, 2832-2838 (covalent radii, Angstrom).
COVALENT_RADII: dict[str, float] = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Sc": 1.70, "Ti": 1.60, "V": 1.53, "Cr": 1.39,
    "Mn": 1.39, "Fe": 1.32, "Co": 1.26, "Ni": 1.24, "Cu": 1.32, "Zn": 1.22,
    "Ga": 1.22, "Ge": 1.20, "As": 1.19, "Se": 1.20, "Br": 1.20, "Kr": 1.16,
    "Rb": 2.20, "Sr": 1.95, "Y": 1.90, "Zr": 1.75, "Nb": 1.64, "Mo": 1.54,
    "Tc": 1.47, "Ru": 1.46, "Rh": 1.42, "Pd": 1.39, "Ag": 1.45, "Cd": 1.44,
    "In": 1.42, "Sn": 1.39, "Sb": 1.39, "Te": 1.38, "I": 1.39, "Xe": 1.40,
    "Cs": 2.44, "Ba": 2.15, "La": 2.07, "Ce": 2.04, "Hf": 1.75, "Ta": 1.70,
    "W": 1.62, "Re": 1.51, "Os": 1.44, "Ir": 1.41, "Pt": 1.36, "Au": 1.36,
    "Hg": 1.32, "Tl": 1.45, "Pb": 1.46, "Bi": 1.48, "Po": 1.40, "At": 1.50,
    "Rn": 1.50,
}

DEFAULT_COVALENT_RADIUS = 1.50

#: A bond is inferred when ``d <= DEFAULT_BOND_SCALE * (r_i + r_j)``.
#: Typical single bonds (C-C 1.54 A vs 1.15*1.52 = 1.75 A, C-H 1.09 A vs
#: 1.15*1.07 = 1.23 A, C-O 1.43 A vs 1.63 A) are captured, while hydrogen
#: bonds (O...H ~1.8 A against a 1.12 A limit) and non-bonded contacts are not.
DEFAULT_BOND_SCALE = 1.15

GUEST_GROUPING_CHOICES = ("auto", "residue", "bonded", "none")


@dataclass(frozen=True)
class GuestGrouping:
    """Atoms of one guest molecule grouped together.

    ``order`` sorts the selected atoms so that each molecule occupies a
    contiguous slice, ``offsets[i]:offsets[i+1]`` being the slice of molecule
    ``i``.  ``group_index`` is the molecule id of every selected atom in the
    original (selection) order.
    """

    group_index: np.ndarray
    order: np.ndarray
    offsets: np.ndarray
    method: str
    n_groups: int

    @property
    def n_atoms(self) -> int:
        return int(len(self.group_index))


def normalise_symbol(symbol: str) -> str:
    text = str(symbol or "").strip()
    if not text:
        return ""
    letters = "".join(character for character in text if character.isalpha())
    if not letters:
        return ""
    return letters[0].upper() + letters[1:].lower()


def covalent_radius(symbol: str) -> float:
    return COVALENT_RADII.get(normalise_symbol(symbol), DEFAULT_COVALENT_RADIUS)


def grouping_from_labels(
    labels: Sequence[int] | np.ndarray,
    method: str,
) -> Optional[GuestGrouping]:
    """Build a grouping from an arbitrary per-atom molecule label."""
    labels = np.asarray(labels, dtype=np.int64)

    if labels.size == 0:
        return None

    _, group_index = np.unique(labels, return_inverse=True)
    group_index = group_index.astype(np.int64, copy=False)
    n_groups = int(group_index.max()) + 1

    order = np.argsort(group_index, kind="stable")
    counts = np.bincount(group_index, minlength=n_groups)
    offsets = np.zeros(n_groups + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])

    return GuestGrouping(
        group_index=group_index,
        order=order,
        offsets=offsets,
        method=method,
        n_groups=n_groups,
    )


def grouping_by_residue(resindices: Sequence[int] | np.ndarray) -> Optional[GuestGrouping]:
    """Group selected atoms by MD residue (one residue = one molecule)."""
    return grouping_from_labels(resindices, "residue")


def infer_bond_components(
    positions: np.ndarray,
    symbols: Sequence[str],
    box: Optional[np.ndarray],
    scale: float = DEFAULT_BOND_SCALE,
) -> np.ndarray:
    """Connected components of the covalent bond graph, as per-atom labels.

    Two atoms are bonded when their minimum-image distance is at most
    ``scale * (r_i + r_j)``, i.e. ``scale`` times the sum of the covalent
    radii.  The neighbour list is built with a capped distance query at the
    largest possible pair cut-off, so the cost scales with the number of
    neighbouring pairs rather than the square of the atom count, and periodic
    images are handled by the box.
    """
    n_atoms = len(positions)
    labels = np.arange(n_atoms, dtype=np.int64)

    if n_atoms < 2:
        return labels

    positions = np.asarray(positions, dtype=np.float64)
    radii = np.asarray([covalent_radius(symbol) for symbol in symbols], dtype=np.float64)

    max_cutoff = float(scale * radii.max() * 2.0)

    if box is not None and np.asarray(box).size >= 3 and np.all(np.asarray(box)[:3] > 0.0):
        pairs, distances = self_capped_distance(
            positions,
            max_cutoff=max_cutoff,
            box=np.asarray(box, dtype=np.float64),
            return_distances=True,
        )
    else:
        pairs, distances = self_capped_distance(
            positions,
            max_cutoff=max_cutoff,
            return_distances=True,
        )

    if len(pairs) == 0:
        return labels

    limit = scale * (radii[pairs[:, 0]] + radii[pairs[:, 1]])
    bonded = pairs[distances <= limit]

    if len(bonded) == 0:
        return labels

    parent = list(range(n_atoms))

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for left, right in bonded:
        left_root = find(int(left))
        right_root = find(int(right))

        if left_root != right_root:
            parent[right_root] = left_root

    return np.asarray([find(index) for index in range(n_atoms)], dtype=np.int64)


def grouping_by_bonds(
    positions: np.ndarray,
    symbols: Sequence[str],
    box: Optional[np.ndarray],
    scale: float = DEFAULT_BOND_SCALE,
) -> Optional[GuestGrouping]:
    """Group selected atoms into molecules by inferred covalent bonds."""
    return grouping_from_labels(
        infer_bond_components(positions, symbols, box, scale),
        "bonded",
    )


def group_centres(
    positions: np.ndarray,
    grouping: Optional[GuestGrouping],
    box: Optional[np.ndarray],
) -> np.ndarray:
    """One geometric centre per guest molecule.

    The centre is the centroid of the molecule's selected atoms, unwrapped with
    the minimum image convention relative to the molecule's first atom so that
    molecules crossing a periodic boundary are handled correctly.
    """
    positions = np.asarray(positions, dtype=np.float64)

    if grouping is None or grouping.n_groups == 0:
        return positions.copy()

    counts = np.diff(grouping.offsets)

    if counts.max() == 1:
        # Every molecule contributes one atom: no unwrapping is needed and
        # the atom order is simply permuted.
        return positions[grouping.order].copy()

    sorted_positions = positions[grouping.order]
    references = sorted_positions[grouping.offsets[:-1]]

    if box is None:
        sums = np.add.reduceat(sorted_positions, grouping.offsets[:-1], axis=0)
        return sums / counts[:, None]

    reference_per_atom = np.repeat(references, counts, axis=0)
    delta = pbc_delta(sorted_positions - reference_per_atom, box)
    sums = np.add.reduceat(delta, grouping.offsets[:-1], axis=0)

    return references + sums / counts[:, None]


def should_use_bonded(method: str, has_topology: bool) -> bool:
    if method == "bonded":
        return True
    if method == "residue":
        return False
    if method == "none":
        return False
    return not has_topology
