from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Iterator, Optional, Sequence

import numpy as np

from .config import TARGETS, TYPE_PRIORITY, CageParameters
from .neighbor import (
    NeighborGraph,
    build_local_graph_from_nodes,
    prune_local_graph_to_two_core,
    select_nodes_within_radius,
)
from . import neighbor as _neighbor
from .beam import NUMBA_BEAM_AVAILABLE as BEAM_AVAILABLE, beam_search
from .rings import (
    GuestRing,
    RingRecord,
    RingTable,
    build_ring_table,
    enumerate_local_cycles,
    estimate_directional_coverage,
    fit_many_rings,
    ring_edges,
    score_guest_rings,
)


@dataclass
class GuestShell:
    guest_id: int
    guest_center: tuple[float, float, float]
    ring_ids: tuple[int, ...]
    n4: int
    n5: int
    n6: int
    n_faces: int
    closure: float
    boundary_edges: int
    euler: int
    connected: bool
    coverage: float
    nearest_target: str
    target_distance: int
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]


@dataclass
class Cage:
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    rings: tuple[int, ...]
    cage_type: str
    center: tuple[float, float, float]
    guest_id: int
    frame: int
    euler: int
    closure: float


@dataclass
class GuestCageResult:
    guest_id: int
    cage: Optional[Cage]
    shell: Optional[GuestShell]
    assigned: str
    status: str
    n_local_nodes: int
    n_local_cycles: int
    n_local_rings: int


ENV_LIQUID = "LIQ"
ENV_INTERFACE = "interface"
ENV_PRECURSOR = "precursor"
ENV_CRYSTAL = "crystal"
ENV_UNKNOWN = "unknown"

#: states that are not a detected cage (used by the transition network)
NON_CAGE_STATES = frozenset(
    {ENV_LIQUID, ENV_INTERFACE, ENV_PRECURSOR, ENV_CRYSTAL, ENV_UNKNOWN}
)

def classify_guest_environment(
    hop_labels: Optional[np.ndarray],
    local_nodes_by_guest: list,
    min_waters: int = 4,
) -> list[str]:
    """Guest environment from the HOP label of its first hydration shell.

    The shell is the set of water oxygens within ``guest_radius`` of the
    guest centre (the same neighbour list the cage search uses, but
    *before* the HOP candidate mask removes liquid water).  The node is a
    property of the guest environment, not of an individual water molecule.
    """
    from .config import LABEL_HYD, LABEL_IC, LABEL_IH, LABEL_LIQ, LABEL_PRE, LABEL_UNK

    if hop_labels is None:
        return [ENV_UNKNOWN] * len(local_nodes_by_guest)

    environments: list[str] = []

    for nodes in local_nodes_by_guest:
        if nodes is None or len(nodes) < max(1, int(min_waters)):
            environments.append(ENV_UNKNOWN)
            continue

        labels = hop_labels[nodes]
        counts = (
            (int((labels == LABEL_HYD).sum()) + int((labels == LABEL_IC).sum())
             + int((labels == LABEL_IH).sum()), ENV_CRYSTAL),
            (int((labels == LABEL_PRE).sum()), ENV_PRECURSOR),
            (int((labels == LABEL_UNK).sum()), ENV_INTERFACE),
            (int((labels == LABEL_LIQ).sum()), ENV_LIQUID),
        )

        best_count, best_state = max(counts, key=lambda item: item[0])

        if best_count <= 0:
            environments.append(ENV_UNKNOWN)
        elif best_state in (ENV_PRECURSOR, ENV_CRYSTAL):
            # without a detected cage an ordered shell means the guest sits at
            # an interface, not that it is inside a crystal cavity
            environments.append(ENV_INTERFACE)
        else:
            environments.append(best_state)

    return environments

@dataclass
class CageAnalysisResult:
    frame: int
    cages: list[Cage]
    guest_results: list[GuestCageResult]
    shell_rows: list[tuple]
    counts: Counter
    n_guest: int
    n_cages: int
    n_local_cycles_total: int
    n_local_rings_total: int
    guest_environments: list[str] = field(default_factory=list)


def signature(ring_ids: tuple[int, ...], rings: list[GuestRing]) -> tuple[int, int, int]:
    n4 = 0
    n5 = 0
    n6 = 0

    for ring_id in ring_ids:
        size = rings[int(ring_id)].size

        if size == 4:
            n4 += 1
        elif size == 5:
            n5 += 1
        elif size == 6:
            n6 += 1

    return n4, n5, n6


def topology_from_edge_count(
    ring_ids: tuple[int, ...] | frozenset[int],
    rings: list[GuestRing],
    edge_count: dict[tuple[int, int], int],
) -> Optional[dict]:
    if not edge_count:
        return None

    nodes = set()

    for ring_id in ring_ids:
        nodes.update(rings[int(ring_id)].nodes)

    vertex_count = len(nodes)
    edge_count_len = len(edge_count)
    face_count = len(ring_ids)

    euler = vertex_count - edge_count_len + face_count

    total_face_edges = sum(edge_count.values())
    double_edges = sum(1 for count in edge_count.values() if count == 2)
    boundary_edges = sum(1 for count in edge_count.values() if count == 1)

    closure = 2.0 * double_edges / total_face_edges if total_face_edges else 0.0
    closed = all(count == 2 for count in edge_count.values())

    return {
        "nodes": tuple(sorted(int(node) for node in nodes)),
        "edges": tuple(sorted(edge_count.keys())),
        "euler": int(euler),
        "closure": float(closure),
        "closed": bool(closed),
        "boundary_edges": int(boundary_edges),
    }


def topology(
    ring_ids: tuple[int, ...] | range,
    rings: list[GuestRing],
) -> Optional[dict]:
    ring_id_tuple = tuple(int(ring_id) for ring_id in ring_ids)
    edge_count: dict[tuple[int, int], int] = defaultdict(int)

    for ring_id in ring_id_tuple:
        for edge in rings[ring_id].edges:
            edge_count[edge] += 1

    return topology_from_edge_count(ring_id_tuple, rings, edge_count)


def connected_ring_set(
    ring_ids: tuple[int, ...] | frozenset[int],
    rings: list[GuestRing],
) -> bool:
    """True when the selected rings form one edge-connected cluster."""
    ring_id_list = [int(ring_id) for ring_id in ring_ids]

    if not ring_id_list:
        return False

    count = len(ring_id_list)

    if count == 1:
        return True

    parent = list(range(count))

    def find(node: int) -> int:
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    first_ring_of_edge: dict[tuple[int, int], int] = {}

    for local_index, ring_id in enumerate(ring_id_list):
        for edge in rings[ring_id].edges:
            other = first_ring_of_edge.get(edge)

            if other is None:
                first_ring_of_edge[edge] = local_index
                continue

            left_root = find(other)
            right_root = find(local_index)

            if left_root != right_root:
                parent[right_root] = left_root

    root = find(0)

    return all(find(index) == root for index in range(1, count))


def counts_add(counts: tuple[int, int, int], ring_size: int) -> tuple[int, int, int]:
    n4, n5, n6 = counts

    if ring_size == 4:
        return n4 + 1, n5, n6

    if ring_size == 5:
        return n4, n5 + 1, n6

    if ring_size == 6:
        return n4, n5, n6 + 1

    return counts


def counts_exceed(
    counts: tuple[int, int, int],
    target: tuple[int, int, int],
) -> bool:
    return counts[0] > target[0] or counts[1] > target[1] or counts[2] > target[2]


def target_face_number(target: tuple[int, int, int]) -> int:
    return target[0] + target[1] + target[2]


def nearest_cage_target(sig: tuple[int, int, int]) -> tuple[str, int]:
    best_name = "none"
    best_distance = 10**9

    for target, name in TARGETS:
        distance = (
            abs(sig[0] - target[0])
            + abs(sig[1] - target[1])
            + abs(sig[2] - target[2])
        )

        if distance < best_distance:
            best_distance = distance
            best_name = name

    return best_name, best_distance


def possible_targets(shell: GuestShell) -> list[tuple[tuple[int, int, int], str]]:
    candidates = []
    sig = (shell.n4, shell.n5, shell.n6)

    for target, cage_type in TARGETS:
        if sig[0] < target[0]:
            continue
        if sig[1] < target[1]:
            continue
        if sig[2] < target[2]:
            continue

        candidates.append((target, cage_type))

    candidates.sort(
        key=lambda item: (
            abs(sig[0] - item[0][0])
            + abs(sig[1] - item[0][1])
            + abs(sig[2] - item[0][2]),
            target_face_number(item[0]),
        )
    )

    return candidates


def _decode_bits(mask: int) -> Iterator[int]:
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _to_words(value: int, n_words: int) -> np.ndarray:
    out = np.zeros(n_words, dtype=np.uint64)
    for index in range(n_words):
        out[index] = np.uint64((value >> (64 * index)) & 0xFFFFFFFFFFFFFFFF)
    return out


class RingAssembly:
    """Bitmask form of one guest's candidate rings.

    The beam search needs the edge multiplicity of the partially assembled
    shell at every candidate extension.  Storing it as a ``dict`` costs
    O(number of edges) per transition and dominated the whole runtime; storing
    it as two bitmasks (edges covered once / twice) turns each transition into
    a handful of integer operations.

    Two engines are provided and must produce identical results:

    * :meth:`_search_python` -- Python integers, always available;
    * :meth:`_search_numba`  -- the same search in numba (see
      :mod:`hytopo.beam`), used when numba is importable.

    The engines are chosen once per call; a failing numba build simply falls
    back to the Python path.
    """

    __slots__ = (
        "rings",
        "ring_masks",
        "ring_sizes",
        "ring_scores",
        "edge_keys",
        "edge_ring_masks",
        "n_edge_words",
        "n_sel_words",
        "_numba_cache",
    )

    def __init__(self, rings: Sequence[GuestRing]) -> None:
        self.rings = rings

        edge_index: dict[tuple[int, int], int] = {}
        ring_masks: list[int] = []
        ring_sizes: list[int] = []
        ring_scores: list[float] = []

        for ring in rings:
            mask = 0
            for edge in ring.edges:
                edge_id = edge_index.get(edge)
                if edge_id is None:
                    edge_id = len(edge_index)
                    edge_index[edge] = edge_id
                mask |= 1 << edge_id
            ring_masks.append(mask)
            ring_sizes.append(int(ring.size))
            ring_scores.append(float(ring.score))

        edge_ring_masks = [0] * len(edge_index)
        for ring_id, mask in enumerate(ring_masks):
            bit = 1 << ring_id
            for edge_id in _decode_bits(mask):
                edge_ring_masks[edge_id] |= bit

        self.ring_masks = ring_masks
        self.ring_sizes = ring_sizes
        self.ring_scores = ring_scores
        self.edge_keys = list(edge_index.keys())
        self.edge_ring_masks = edge_ring_masks
        self.n_edge_words = max(1, (len(edge_index) + 63) // 64)
        self.n_sel_words = max(1, (len(rings) + 63) // 64)
        self._numba_cache: Optional[tuple] = None

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------

    def grow(
        self,
        target: tuple[int, int, int],
        cage_type: str,
        params: CageParameters,
    ) -> Optional[tuple[tuple[int, ...], str, dict]]:
        if not self.rings:
            return None

        candidates = None

        if BEAM_AVAILABLE and _neighbor.NUMBA_RUNTIME_ENABLED:
            candidates = self._search_numba(target, params)

        if candidates is None:
            candidates = self._search_python(target, params)

        if not candidates:
            return None

        best = None

        for selected, edge_count, score in candidates:
            topo = topology_from_edge_count(selected, self.rings, edge_count)

            if topo is None:
                continue

            if not topo["closed"]:
                continue

            if topo["euler"] != 2:
                continue

            if not connected_ring_set(selected, self.rings):
                continue

            final_score = score + 100.0 * topo["closure"]

            if best is None or final_score > best[0]:
                best = (final_score, selected, topo)

        if best is None:
            return None

        _, selected, topo = best

        return tuple(sorted(int(ring_id) for ring_id in selected)), cage_type, topo

    # ------------------------------------------------------------------
    # numba engine
    # ------------------------------------------------------------------

    def _numba_arrays(self) -> tuple:
        if self._numba_cache is not None:
            return self._numba_cache

        n_rings = len(self.rings)
        ring_edge = np.zeros((n_rings, self.n_edge_words), dtype=np.uint64)
        ring_sel = np.zeros((n_rings, self.n_sel_words), dtype=np.uint64)

        for ring_id in range(n_rings):
            ring_edge[ring_id, :] = _to_words(
                self.ring_masks[ring_id], self.n_edge_words
            )
            ring_sel[ring_id, :] = _to_words(1 << ring_id, self.n_sel_words)

        edge_ring = np.zeros((len(self.edge_keys), self.n_sel_words), dtype=np.uint64)

        for edge_id, mask in enumerate(self.edge_ring_masks):
            edge_ring[edge_id, :] = _to_words(mask, self.n_sel_words)

        self._numba_cache = (
            ring_edge,
            ring_sel,
            np.asarray(self.ring_sizes, dtype=np.int64),
            np.asarray(self.ring_scores, dtype=np.float64),
            edge_ring,
        )

        return self._numba_cache

    def _search_numba(
        self,
        target: tuple[int, int, int],
        params: CageParameters,
    ) -> Optional[list[tuple[frozenset[int], dict, float]]]:
        ring_edge, ring_sel, ring_sizes, ring_scores, edge_ring = self._numba_arrays()

        try:
            result = beam_search(
                ring_edge,
                ring_sel,
                ring_sizes,
                ring_scores,
                edge_ring,
                target,
                int(params.beam_width),
                int(params.candidate_limit),
            )
        except Exception:
            # A broken numba path must never change the answer; fall back.
            return None

        if result is None:
            return None

        n_final, sel, once, twice, _counts, scores = result

        candidates = []

        for index in range(int(n_final)):
            selected = frozenset(
                _decode_words(sel[index], self.n_sel_words)
            )
            edge_count: dict[tuple[int, int], int] = {}

            for edge_id in _decode_words(once[index], self.n_edge_words):
                edge_count[self.edge_keys[edge_id]] = 1
            for edge_id in _decode_words(twice[index], self.n_edge_words):
                edge_count[self.edge_keys[edge_id]] = 2

            candidates.append((selected, edge_count, float(scores[index])))

        return candidates

    # ------------------------------------------------------------------
    # reference engine
    # ------------------------------------------------------------------

    def _search_python(
        self,
        target: tuple[int, int, int],
        params: CageParameters,
    ) -> list[tuple[frozenset[int], dict, float]]:
        rings = self.rings

        if not rings:
            return []

        target_faces = target_face_number(target)
        beam_width = params.beam_width
        candidate_limit = params.candidate_limit

        ring_masks = self.ring_masks
        ring_sizes = self.ring_sizes
        ring_scores = self.ring_scores
        edge_ring_masks = self.edge_ring_masks
        edge_keys = self.edge_keys

        states = []

        for ring_id, ring in enumerate(rings):
            counts = counts_add((0, 0, 0), ring.size)

            if counts_exceed(counts, target):
                continue

            states.append(
                (1 << ring_id, ring_masks[ring_id], 0, counts, ring_scores[ring_id])
            )

        if not states:
            return []

        states.sort(key=lambda state: state[4], reverse=True)
        states = states[:beam_width]

        for _ in range(1, target_faces):
            new_states = []

            for selected, edge_once, edge_twice, counts, score in states:
                if not edge_once:
                    continue

                ring_candidates = 0

                for edge_id in _decode_bits(edge_once):
                    ring_candidates |= edge_ring_masks[edge_id]

                ring_candidates &= ~selected

                if not ring_candidates:
                    continue

                scored_candidates = []

                for ring_id in _decode_bits(ring_candidates):
                    shared = (ring_masks[ring_id] & edge_once).bit_count()

                    if shared == 0:
                        continue

                    scored_candidates.append((-shared, -ring_scores[ring_id], ring_id))

                scored_candidates.sort()

                for _, _, ring_id in scored_candidates[:candidate_limit]:
                    new_counts = counts_add(counts, ring_sizes[ring_id])

                    if counts_exceed(new_counts, target):
                        continue

                    face_mask = ring_masks[ring_id]

                    if edge_twice & face_mask:
                        continue

                    new_twice = edge_twice | (edge_once & face_mask)
                    new_once = (edge_once ^ face_mask) & ~new_twice

                    double_edges = new_twice.bit_count()
                    total_face_edges = new_once.bit_count() + 2 * double_edges
                    closure = (
                        2.0 * double_edges / total_face_edges
                        if total_face_edges
                        else 0.0
                    )

                    new_states.append(
                        (
                            selected | (1 << ring_id),
                            new_once,
                            new_twice,
                            new_counts,
                            score + ring_scores[ring_id] + 3.0 * closure,
                        )
                    )

            if not new_states:
                return []

            best_by_selected: dict[int, tuple] = {}

            for state in new_states:
                current = best_by_selected.get(state[0])

                if current is None or state[4] > current[4]:
                    best_by_selected[state[0]] = state

            states = sorted(
                best_by_selected.values(),
                key=lambda state: state[4],
                reverse=True,
            )
            states = states[:beam_width]

        candidates = []

        for selected, edge_once, edge_twice, counts, score in states:
            if counts != target:
                continue

            edge_count: dict[tuple[int, int], int] = {}

            for edge_id in _decode_bits(edge_once):
                edge_count[edge_keys[edge_id]] = 1
            for edge_id in _decode_bits(edge_twice):
                edge_count[edge_keys[edge_id]] = 2

            candidates.append((frozenset(_decode_bits(selected)), edge_count, score))

        return candidates


def _decode_words(words: np.ndarray, n_words: int) -> Iterator[int]:
    for index in range(n_words):
        value = int(words[index])
        while value:
            low = value & -value
            yield index * 64 + low.bit_length() - 1
            value ^= low




def build_guest_shell(
    guest_id: int,
    guest_center: np.ndarray,
    rings: list[GuestRing],
    box: np.ndarray,
) -> Optional[GuestShell]:
    if not rings:
        return None

    local_ring_ids = tuple(range(len(rings)))
    topo = topology(local_ring_ids, rings)

    if topo is None:
        return None

    n4, n5, n6 = signature(local_ring_ids, rings)
    nearest_target, target_distance = nearest_cage_target((n4, n5, n6))

    return GuestShell(
        guest_id=int(guest_id),
        guest_center=tuple(float(value) for value in guest_center),
        ring_ids=tuple(int(ring.ring_id) for ring in rings),
        n4=int(n4),
        n5=int(n5),
        n6=int(n6),
        n_faces=len(rings),
        closure=float(topo["closure"]),
        boundary_edges=int(topo["boundary_edges"]),
        euler=int(topo["euler"]),
        connected=connected_ring_set(local_ring_ids, rings),
        coverage=float(estimate_directional_coverage(rings, guest_center, box)),
        nearest_target=nearest_target,
        target_distance=int(target_distance),
        nodes=topo["nodes"],
        edges=topo["edges"],
    )


def classify_partial_shell(shell: Optional[GuestShell], params: CageParameters) -> str:
    if shell is None:
        return "other"

    if shell.n_faces < params.partial_min_faces:
        return "other"

    if shell.closure < params.partial_min_closure:
        return "partial"

    if not shell.connected:
        return "partial"

    half_coverage = params.half_min_coverage <= shell.coverage <= params.half_max_coverage

    if (
        shell.n_faces >= params.half_min_faces
        and shell.n5 >= 6
        and shell.n6 == 0
        and half_coverage
    ):
        return "half_5^12"

    if (
        shell.n_faces >= params.half_min_faces
        and shell.n5 >= 6
        and 1 <= shell.n6 <= 2
        and half_coverage
    ):
        return "half_5^12_6^2"

    if (
        shell.n_faces >= params.half_min_faces
        and shell.n5 >= 6
        and 3 <= shell.n6 <= 4
        and half_coverage
    ):
        return "half_5^12_6^4"

    if 5 <= shell.n_faces <= 8 and shell.closure >= params.incipient_min_closure:
        return "incipient_cage"

    if shell.nearest_target == "5^12" and shell.target_distance <= 4 and shell.closure >= 0.55:
        return "partial_5^12"

    if shell.nearest_target == "5^12_6^2" and shell.target_distance <= 4 and shell.closure >= 0.55:
        return "partial_5^12_6^2"

    if shell.nearest_target == "5^12_6^4" and shell.target_distance <= 5 and shell.closure >= 0.55:
        return "partial_5^12_6^4"

    if shell.n_faces >= 8 and shell.boundary_edges >= params.open_min_boundary and shell.closure >= 0.35:
        return "open_cage"

    if shell.n_faces >= 8 and shell.closure >= 0.45:
        return "distorted_cage"

    return "partial"


def build_cage_from_topology(
    topo: dict,
    local_ring_ids: tuple[int, ...],
    rings: list[GuestRing],
    cage_type: str,
    guest_center: np.ndarray,
    guest_id: int,
    frame: int,
) -> Cage:
    global_ring_ids = tuple(int(rings[ring_id].ring_id) for ring_id in local_ring_ids)

    return Cage(
        nodes=topo["nodes"],
        edges=topo["edges"],
        rings=global_ring_ids,
        cage_type=cage_type,
        center=tuple(float(value) for value in guest_center),
        guest_id=int(guest_id),
        frame=int(frame),
        euler=int(topo["euler"]),
        closure=float(topo["closure"]),
    )


def build_partial_cage(
    shell: GuestShell,
    cage_type: str,
    guest_center: np.ndarray,
    guest_id: int,
    frame: int,
) -> Cage:
    return Cage(
        nodes=shell.nodes,
        edges=shell.edges,
        rings=shell.ring_ids,
        cage_type=cage_type,
        center=tuple(float(value) for value in guest_center),
        guest_id=int(guest_id),
        frame=int(frame),
        euler=int(shell.euler),
        closure=float(shell.closure),
    )


def analyze_guest_with_rings(
    guest_id: int,
    guest_center: np.ndarray,
    frame: int,
    box: np.ndarray,
    params: CageParameters,
    ring_table: RingTable,
    ring_indices: np.ndarray,
    n_local_nodes: int,
    n_local_cycles: int,
) -> GuestCageResult:
    """Classify one guest from its pre-computed candidate rings.

    The rings are perceived once per frame by
    :func:`build_frame_ring_table`; this function only performs the cheap
    per-guest part of the analysis.
    """
    n_local_rings = int(len(ring_indices))

    if n_local_nodes < params.min_ring:
        return GuestCageResult(
            guest_id=guest_id,
            cage=None,
            shell=None,
            assigned="none",
            status="too_few_local_nodes",
            n_local_nodes=n_local_nodes,
            n_local_cycles=0,
            n_local_rings=0,
        )

    if n_local_cycles == 0:
        return GuestCageResult(
            guest_id=guest_id,
            cage=None,
            shell=None,
            assigned="none",
            status="no_cycle",
            n_local_nodes=n_local_nodes,
            n_local_cycles=n_local_cycles,
            n_local_rings=n_local_rings,
        )

    if n_local_rings == 0:
        return GuestCageResult(
            guest_id=guest_id,
            cage=None,
            shell=None,
            assigned="none",
            status="no_valid_ring",
            n_local_nodes=n_local_nodes,
            n_local_cycles=n_local_cycles,
            n_local_rings=n_local_rings,
        )

    guest_rings = score_guest_rings(
        candidate_rings=ring_table,
        guest_center=guest_center,
        box=box,
        params=params,
        indices=ring_indices,
    )

    if not guest_rings:
        return GuestCageResult(
            guest_id=guest_id,
            cage=None,
            shell=None,
            assigned="none",
            status="no_guest_facing_ring",
            n_local_nodes=n_local_nodes,
            n_local_cycles=n_local_cycles,
            n_local_rings=n_local_rings,
        )

    shell = build_guest_shell(
        guest_id=guest_id,
        guest_center=guest_center,
        rings=guest_rings,
        box=box,
    )

    if shell is None:
        return GuestCageResult(
            guest_id=guest_id,
            cage=None,
            shell=None,
            assigned="none",
            status="no_shell",
            n_local_nodes=n_local_nodes,
            n_local_cycles=n_local_cycles,
            n_local_rings=n_local_rings,
        )

    assembly = RingAssembly(guest_rings)

    for target, cage_type in possible_targets(shell):
        result = assembly.grow(target, cage_type, params)

        if result is None:
            continue

        local_ring_ids, assigned_type, topo = result

        cage = build_cage_from_topology(
            topo=topo,
            local_ring_ids=local_ring_ids,
            rings=guest_rings,
            cage_type=assigned_type,
            guest_center=guest_center,
            guest_id=guest_id,
            frame=frame,
        )

        return GuestCageResult(
            guest_id=guest_id,
            cage=cage,
            shell=shell,
            assigned=assigned_type,
            status="complete",
            n_local_nodes=n_local_nodes,
            n_local_cycles=n_local_cycles,
            n_local_rings=n_local_rings,
        )

    if not params.standard_only:
        subtype = classify_partial_shell(shell, params)

        if subtype not in ("other", "none"):
            cage = build_partial_cage(
                shell=shell,
                cage_type=subtype,
                guest_center=guest_center,
                guest_id=guest_id,
                frame=frame,
            )

            return GuestCageResult(
                guest_id=guest_id,
                cage=cage,
                shell=shell,
                assigned=subtype,
                status="partial",
                n_local_nodes=n_local_nodes,
                n_local_cycles=n_local_cycles,
                n_local_rings=n_local_rings,
            )

    return GuestCageResult(
        guest_id=guest_id,
        cage=None,
        shell=shell,
        assigned="none",
        status="unassigned",
        n_local_nodes=n_local_nodes,
        n_local_cycles=n_local_cycles,
        n_local_rings=n_local_rings,
    )






def build_frame_ring_table(
    oxygen_pos: np.ndarray,
    box: np.ndarray,
    global_graph: NeighborGraph,
    candidate_mask: Optional[np.ndarray],
    params: CageParameters,
    quiet: bool = False,
) -> tuple[RingTable, np.ndarray, list[tuple[int, ...]]]:
    """Perceive the rings of one frame once, for all guests.

    A ring is a property of the water network, not of a guest: neighbouring
    cages share faces, so the per-guest depth-first search and planarity fit
    were repeated for the same water ring up to three times.  Here the
    candidate subgraph is enumerated once, every ring is fitted by one batched
    eigen solve, and :func:`assign_cycles_to_guests` attaches each ring to the
    guests whose ``guest_radius`` sphere contains all of its vertices.

    Returns ``(records, record_of_cycle, cycles_global)`` where
    ``record_of_cycle[i]`` is the index into ``records`` for cycle ``i`` or -1
    when the ring failed the planarity gate.
    """
    if candidate_mask is None:
        nodes = np.arange(len(oxygen_pos), dtype=np.int64)
    else:
        nodes = np.flatnonzero(candidate_mask).astype(np.int64)

    if len(nodes) < params.min_ring:
        return build_ring_table([]), np.empty(0, dtype=np.int64), []

    local_graph = prune_local_graph_to_two_core(
        build_local_graph_from_nodes(global_graph, nodes)
    )

    if len(local_graph.global_nodes) < params.min_ring:
        return build_ring_table([]), np.empty(0, dtype=np.int64), []

    # Frame-level enumeration cap.  The per-guest cap cannot be reused here:
    # it limits the cycles of one guest, while a frame has to hold the union of
    # the rings of all guests.
    cap = max(
        params.max_local_cycles,
        min(500_000, 8 * len(local_graph.global_nodes)),
    )
    cycle_params = params if cap == params.max_local_cycles else replace(
        params, max_local_cycles=cap
    )

    local_cycles = enumerate_local_cycles(local_graph, cycle_params)

    if not local_cycles:
        return build_ring_table([]), np.empty(0, dtype=np.int64), []

    if len(local_cycles) >= cycle_params.max_local_cycles:
        # The enumeration hit the safety cap; results would silently depend on
        # traversal order, which is worse than being loud about it.
        if not quiet:
            print(
                "Warning: frame ring enumeration reached max_local_cycles="
                f"{cycle_params.max_local_cycles}; results may be incomplete.",
                flush=True,
            )

    local_to_global = local_graph.local_to_global
    cycles_global = [
        tuple(int(local_to_global[int(node)]) for node in cycle)
        for cycle in local_cycles
    ]

    centers, normals, planarity = fit_many_rings(cycles_global, oxygen_pos, box)

    records: list[RingRecord] = []
    record_of_cycle = np.full(len(cycles_global), -1, dtype=np.int64)

    for cycle_id, cycle in enumerate(cycles_global):
        if planarity[cycle_id] > params.ring_planarity:
            continue

        record_of_cycle[cycle_id] = len(records)
        records.append(
            RingRecord(
                ring_id=len(records),
                nodes=cycle,
                size=len(cycle),
                center=(
                    float(centers[cycle_id][0]),
                    float(centers[cycle_id][1]),
                    float(centers[cycle_id][2]),
                ),
                normal=(
                    float(normals[cycle_id][0]),
                    float(normals[cycle_id][1]),
                    float(normals[cycle_id][2]),
                ),
                planarity=float(planarity[cycle_id]),
                edges=ring_edges(cycle),
            )
        )

    return build_ring_table(records), record_of_cycle, cycles_global


def assign_cycles_to_guests(
    cycles_global: Sequence[tuple[int, ...]],
    local_nodes_by_guest: Sequence[np.ndarray],
) -> list[list[int]]:
    """Attach each ring to every guest that contains all of its vertices.

    Each water node stores the bitmask of guests whose ``guest_radius`` sphere
    contains it; intersecting the masks of a ring's vertices yields exactly the
    guests that see the whole ring, which is the same ring set the per-guest
    local graph would have produced.
    """
    node_guest_bits: dict[int, int] = {}

    for guest_id, local_nodes in enumerate(local_nodes_by_guest):
        bit = 1 << guest_id

        for node in local_nodes:
            key = int(node)
            node_guest_bits[key] = node_guest_bits.get(key, 0) | bit

    cycle_ids_by_guest: list[list[int]] = [[] for _ in range(len(local_nodes_by_guest))]

    for cycle_id, cycle in enumerate(cycles_global):
        mask = -1

        for node in cycle:
            mask &= node_guest_bits.get(node, 0)

            if mask == 0:
                break

        while mask:
            low = mask & -mask
            cycle_ids_by_guest[low.bit_length() - 1].append(cycle_id)
            mask ^= low

    return cycle_ids_by_guest


def deduplicate_cages(cages: list[Cage], overlap_threshold: float = 0.85) -> list[Cage]:
    ordered = sorted(
        cages,
        key=lambda cage: (
            TYPE_PRIORITY.get(cage.cage_type, 0),
            float(cage.closure),
            len(cage.rings),
            len(cage.nodes),
        ),
        reverse=True,
    )
    kept: list[Cage] = []
    kept_nodes: list[set[int]] = []

    # Two cages can only be duplicates when they share almost every water node,
    # so a node index avoids the quadratic all-pairs Jaccard scan.
    node_index: dict[int, list[int]] = {}

    for cage in ordered:
        nodes = set(cage.nodes)
        candidates: set[int] = set()

        for node in nodes:
            for kept_index in node_index.get(node, ()):
                candidates.add(kept_index)

        duplicate = False

        for kept_index in candidates:
            other_nodes = kept_nodes[kept_index]
            union_size = len(nodes | other_nodes)
            overlap = len(nodes & other_nodes) / union_size if union_size else 1.0
            if overlap >= overlap_threshold:
                duplicate = True
                break

        if not duplicate:
            kept_index = len(kept)
            kept.append(cage)
            kept_nodes.append(nodes)
            for node in nodes:
                node_index.setdefault(node, []).append(kept_index)

    return sorted(kept, key=lambda cage: (cage.guest_id, cage.cage_type))


def shell_row_from_guest_result(frame: int, result: GuestCageResult) -> tuple:
    shell = result.shell

    if shell is None:
        return (
            frame,
            result.guest_id,
            0,
            0,
            0,
            0,
            0.0,
            0,
            0,
            0,
            0.0,
            "none",
            999,
            result.assigned,
            result.status,
            result.n_local_nodes,
            result.n_local_cycles,
            result.n_local_rings,
        )

    return (
        frame,
        result.guest_id,
        shell.n4,
        shell.n5,
        shell.n6,
        shell.n_faces,
        shell.closure,
        shell.boundary_edges,
        shell.euler,
        int(shell.connected),
        shell.coverage,
        shell.nearest_target,
        shell.target_distance,
        result.assigned,
        result.status,
        result.n_local_nodes,
        result.n_local_cycles,
        result.n_local_rings,
    )


def analyze_cages_for_frame(
    frame: int,
    oxygen_pos: np.ndarray,
    guest_pos: Optional[np.ndarray],
    box: np.ndarray,
    global_graph: NeighborGraph,
    candidate_mask: Optional[np.ndarray],
    params: CageParameters,
    quiet: bool = False,
    hop_labels: Optional[np.ndarray] = None,
) -> CageAnalysisResult:
    if guest_pos is None or len(guest_pos) == 0:
        return CageAnalysisResult(
            frame=int(frame),
            cages=[],
            guest_results=[],
            shell_rows=[],
            counts=Counter(),
            n_guest=0,
            n_cages=0,
            n_local_cycles_total=0,
            n_local_rings_total=0,
            guest_environments=[],
        )

    local_nodes_by_guest = select_nodes_within_radius(
        centers=guest_pos,
        positions=oxygen_pos,
        box=box,
        radius=params.guest_radius,
    )

    local_nodes_all = local_nodes_by_guest

    if candidate_mask is not None:
        local_nodes_by_guest = [
            nodes[candidate_mask[nodes]] if len(nodes) else nodes
            for nodes in local_nodes_by_guest
        ]

    ring_table, record_of_cycle, cycles_global = build_frame_ring_table(
        oxygen_pos=oxygen_pos,
        box=box,
        global_graph=global_graph,
        candidate_mask=candidate_mask,
        params=params,
        quiet=quiet,
    )

    guest_environments = classify_guest_environment(
        hop_labels=hop_labels,
        local_nodes_by_guest=local_nodes_all,
        min_waters=max(4, int(params.min_ring)),
    )

    cycle_ids_by_guest = assign_cycles_to_guests(cycles_global, local_nodes_by_guest)

    guest_results: list[GuestCageResult] = []

    for guest_id, guest_center in enumerate(guest_pos):
        cycle_ids = cycle_ids_by_guest[guest_id]

        if cycle_ids:
            candidates = np.asarray(
                [record_of_cycle[cycle_id] for cycle_id in cycle_ids],
                dtype=np.int64,
            )
            ring_indices = candidates[candidates >= 0]
        else:
            ring_indices = np.empty(0, dtype=np.int64)

        result = analyze_guest_with_rings(
            guest_id=int(guest_id),
            guest_center=np.asarray(guest_center, dtype=np.float64),
            frame=int(frame),
            box=box,
            params=params,
            ring_table=ring_table,
            ring_indices=ring_indices,
            n_local_nodes=len(local_nodes_by_guest[guest_id]),
            n_local_cycles=len(cycle_ids),
        )

        guest_results.append(result)

        if not quiet:
            print_guest_result(result)

    cages = [result.cage for result in guest_results if result.cage is not None]
    cages = deduplicate_cages(cages, params.dedup_water_overlap)

    counts = Counter(cage.cage_type for cage in cages)
    shell_rows = [shell_row_from_guest_result(int(frame), result) for result in guest_results]

    n_local_cycles_total = sum(result.n_local_cycles for result in guest_results)
    n_local_rings_total = sum(result.n_local_rings for result in guest_results)

    return CageAnalysisResult(
        frame=int(frame),
        cages=cages,
        guest_results=guest_results,
        shell_rows=shell_rows,
        counts=counts,
        n_guest=len(guest_pos),
        n_cages=len(cages),
        n_local_cycles_total=int(n_local_cycles_total),
        n_local_rings_total=int(n_local_rings_total),
        guest_environments=guest_environments,
    )




def print_guest_result(result: GuestCageResult) -> None:
    if result.shell is None:
        print(
            f"    guest {result.guest_id:5d}: "
            f"local_nodes={result.n_local_nodes:4d} "
            f"cycles={result.n_local_cycles:5d} "
            f"rings={result.n_local_rings:4d} "
            f"assigned={result.assigned} "
            f"status={result.status}",
            flush=True,
        )
        return

    shell = result.shell

    print(
        f"    guest {result.guest_id:5d}: "
        f"local_nodes={result.n_local_nodes:4d} "
        f"cycles={result.n_local_cycles:5d} "
        f"rings={result.n_local_rings:4d} "
        f"sig=({shell.n4},{shell.n5},{shell.n6}) "
        f"closure={shell.closure:.3f} "
        f"coverage={shell.coverage:.3f} "
        f"assigned={result.assigned} "
        f"status={result.status}",
        flush=True,
    )


