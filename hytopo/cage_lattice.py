from __future__ import annotations

"""Cage-centre spatial association.

The old phase decision only looked at the *composition* of the detected cages
(counts of 5^12 versus 5^12 6^2 ...).  That is fragile: a handful of isolated
or boundary cages can move the ratio, and a well-ordered sI domain surrounded by
liquid water looks the same as a random mixture of the same cage types.

This module builds the spatial association of cage centres instead:

* two cages are linked when they share a real face (>= 3 common water nodes
  *and* the shared polygon lies between them geometrically);
* the link is validated with the centre-to-face distance, so a pair that merely
  shares three waters by coincidence is rejected;
* the linked cages form a lattice network whose connected components are the
  crystal domains.

The face-sharing geometry also gives the centre-centre distances that are
reported per cage-type pair (5^12-5^12, 5^12-5^12 6^2, ...).
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np
from MDAnalysis.lib.distances import capped_distance

from .geometry import pbc_delta

# Standard complete cage types.  Kept local to avoid a circular import
# (config -> phase -> cage_lattice); a unit test asserts it matches
# config.COMPLETE_TYPES.
COMPLETE_CAGE_TYPES = (
    "5^12", "5^12_6^2", "5^12_6^4", "4^3_5^6_6^3", "5^12_6^8",
    "5^12_6^1", "5^12_6^3", "5^12_6^5", "5^12_6^6",
    "4^1_5^10_6^2", "4^1_5^10_6^3", "4^1_5^10_6^4", "4^2_5^8_6^4",
)

_COMPLETE_SET = frozenset(COMPLETE_CAGE_TYPES)


@dataclass
class FaceEdge:
    i: int
    j: int
    distance: float
    shared_nodes: int
    expected: float
    valid: bool


@dataclass
class FrameLattice:
    """Cage-centre network of one frame."""

    indices: list[int]
    types: list[str]
    centers: np.ndarray
    edges: list[FaceEdge]
    neighbors: list[Counter]
    components: list[list[int]]
    largest_component: list[int]
    coherence: float
    pair_distances: dict[tuple[str, str], np.ndarray]
    domain_radius: float
    candidates: int
    box: Optional[np.ndarray] = None


def build_cage_lattice(
    cages: Sequence[Any],
    oxygen_pos: Optional[np.ndarray],
    box: Optional[np.ndarray],
    cutoff: float = 9.0,
    tolerance: float = 0.8,
    min_shared_nodes: int = 3,
) -> Optional[FrameLattice]:
    """Build the validated cage-centre network of one frame.

    Returns ``None`` when there are fewer than two complete cages, when the
    positions are unavailable, or when the box is not usable; callers then fall
    back to a composition-only classification.
    """
    complete = [cage for cage in cages if getattr(cage, "cage_type", "") in _COMPLETE_SET]

    if len(complete) < 2 or oxygen_pos is None:
        return None

    oxygen_pos = np.asarray(oxygen_pos, dtype=np.float64)
    centers = np.asarray([cage.center for cage in complete], dtype=np.float64)

    if centers.shape != (len(complete), 3) or not np.isfinite(centers).all():
        return None

    box_array: Optional[np.ndarray] = None

    if box is not None:
        candidate = np.asarray(box, dtype=np.float64)

        if candidate.size >= 3 and np.all(candidate[:3] > 0.0):
            box_array = candidate

    if box_array is not None:
        pairs, distances = capped_distance(
            centers,
            centers,
            max_cutoff=cutoff,
            box=box_array,
            return_distances=True,
        )
    else:
        pairs, distances = capped_distance(
            centers,
            centers,
            max_cutoff=cutoff,
            return_distances=True,
        )

    node_sets = [frozenset(int(node) for node in cage.nodes) for cage in complete]
    types = [str(cage.cage_type) for cage in complete]
    orthogonal = (
        box_array is None
        or np.allclose(np.asarray(box_array[3:6], dtype=np.float64), 90.0)
    )

    candidate_i: list[int] = []
    candidate_j: list[int] = []
    candidate_d: list[float] = []
    candidate_nodes: list[tuple[int, ...]] = []

    for (left, right), distance in zip(pairs, distances):
        i = int(left)
        j = int(right)

        if i >= j:
            continue

        shared = node_sets[i] & node_sets[j]

        if len(shared) < min_shared_nodes:
            continue

        candidate_i.append(i)
        candidate_j.append(j)
        candidate_d.append(float(distance))
        candidate_nodes.append(tuple(shared))

    edges: list[FaceEdge] = []
    candidates = len(candidate_nodes)

    if candidates:
        face_geometry = _batch_face_geometry(
            candidate_nodes,
            oxygen_pos,
            box_array,
            centers,
            candidate_i,
            candidate_j,
            orthogonal,
        )

        for position in range(candidates):
            side_a, side_b = face_geometry[position]

            if side_a is None:
                continue

            if side_a * side_b >= 0.0:
                # Both centres on the same side: the shared water ring is not a
                # face between them.
                edges.append(
                    FaceEdge(
                        candidate_i[position],
                        candidate_j[position],
                        candidate_d[position],
                        len(candidate_nodes[position]),
                        abs(side_a) + abs(side_b),
                        False,
                    )
                )
                continue

            expected = abs(side_a) + abs(side_b)
            valid = abs(candidate_d[position] - expected) <= tolerance
            edges.append(
                FaceEdge(
                    candidate_i[position],
                    candidate_j[position],
                    candidate_d[position],
                    len(candidate_nodes[position]),
                    expected,
                    bool(valid),
                )
            )

    valid_edges = [edge for edge in edges if edge.valid]
    coherence = len(valid_edges) / candidates if candidates else 0.0

    neighbors: list[Counter] = [Counter() for _ in complete]

    for edge in valid_edges:
        neighbors[edge.i][types[edge.j]] += 1
        neighbors[edge.j][types[edge.i]] += 1

    parent = list(range(len(complete)))

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    for edge in valid_edges:
        left_root = find(edge.i)
        right_root = find(edge.j)

        if left_root != right_root:
            parent[right_root] = left_root

    grouped: dict[int, list[int]] = {}

    for index in range(len(complete)):
        grouped.setdefault(find(index), []).append(index)

    components = sorted(grouped.values(), key=len, reverse=True)
    largest = components[0] if components else []

    pair_distances: dict[tuple[str, str], list[float]] = {}

    for edge in valid_edges:
        key = tuple(sorted((types[edge.i], types[edge.j])))
        pair_distances.setdefault(key, []).append(edge.distance)

    domain_radius = _domain_radius(centers, largest, box_array)

    return FrameLattice(
        indices=[int(cage.guest_id) for cage in complete],
        types=types,
        centers=centers,
        edges=edges,
        neighbors=neighbors,
        components=components,
        largest_component=largest,
        coherence=float(coherence),
        pair_distances={
            key: np.asarray(values, dtype=np.float64)
            for key, values in pair_distances.items()
        },
        domain_radius=float(domain_radius),
        candidates=int(candidates),
        box=box_array,
    )


def _domain_radius(
    centers: np.ndarray,
    component: Sequence[int],
    box: Optional[np.ndarray],
) -> float:
    if len(component) < 2:
        return 0.0

    indices = np.asarray(component, dtype=np.int64)
    coordinates = centers[indices]
    centroid = coordinates.mean(axis=0)
    delta = coordinates - centroid

    if box is not None:
        delta = pbc_delta(delta, box)

    return float(np.max(np.linalg.norm(delta, axis=1)))


def lattice_counts(lattice: Optional[FrameLattice], cages: Sequence[Any]) -> Counter:
    """Cage-type counts, restricted to the largest domain when one exists."""
    if lattice is not None and lattice.largest_component:
        return Counter(lattice.types[index] for index in lattice.largest_component)

    return Counter(
        str(cage.cage_type)
        for cage in cages
        if getattr(cage, "cage_type", "") in _COMPLETE_SET
    )


def _batch_face_geometry(
    node_groups: Sequence[Sequence[int]],
    oxygen_pos: np.ndarray,
    box: Optional[np.ndarray],
    centers: np.ndarray,
    index_i: Sequence[int],
    index_j: Sequence[int],
    orthogonal: bool,
):
    """Vectorised Newell-normal geometry for many shared faces at once."""
    n_groups = len(node_groups)
    max_len = max(len(group) for group in node_groups)

    node_array = np.zeros((n_groups, max_len), dtype=np.int64)
    valid_mask = np.zeros((n_groups, max_len), dtype=bool)

    for position, group in enumerate(node_groups):
        size = len(group)
        node_array[position, :size] = group
        valid_mask[position, :size] = True

    coords = oxygen_pos[node_array]
    reference = coords[:, 0:1, :]
    delta = coords - reference

    if box is not None:
        if orthogonal:
            lengths = np.asarray(box[:3], dtype=np.float64)
            delta = delta - lengths * np.rint(delta / lengths)
        else:
            flat = pbc_delta(delta.reshape(-1, 3), box)
            delta = flat.reshape(n_groups, max_len, 3)

    delta[~valid_mask] = 0.0
    coords = reference + delta

    weights = valid_mask.astype(np.float64)
    counts = weights.sum(axis=1)
    centroid = (coords * weights[:, :, None]).sum(axis=1) / counts[:, None]

    shifted = coords - centroid[:, None, :]
    shifted[~valid_mask] = 0.0
    rolled = np.roll(shifted, -1, axis=1)
    rolled[~valid_mask] = 0.0

    normals = np.cross(shifted, rolled).sum(axis=1)
    norms = np.linalg.norm(normals, axis=1)
    good = norms > 1.0e-9
    normals[good] /= norms[good, None]

    center_a = centers[np.asarray(index_i, dtype=np.int64)]
    center_b = centers[np.asarray(index_j, dtype=np.int64)]

    delta_a = center_a - centroid
    delta_b = center_b - centroid

    if box is not None:
        if orthogonal:
            lengths = np.asarray(box[:3], dtype=np.float64)
            delta_a = delta_a - lengths * np.rint(delta_a / lengths)
            delta_b = delta_b - lengths * np.rint(delta_b / lengths)
        else:
            delta_a = pbc_delta(delta_a, box)
            delta_b = pbc_delta(delta_b, box)

    side_a = np.einsum("ij,ij->i", delta_a, normals)
    side_b = np.einsum("ij,ij->i", delta_b, normals)

    output = []
    for position in range(n_groups):
        if not good[position]:
            output.append((None, None))
        else:
            output.append((float(side_a[position]), float(side_b[position])))

    return output
