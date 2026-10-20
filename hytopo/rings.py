from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CageParameters
from .geometry import fit_ring, pbc_delta, wrap_point
from .neighbor import LocalGraph


try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func

        return wrapper


@dataclass(frozen=True)
class RingRecord:
    ring_id: int
    nodes: tuple[int, ...]
    size: int
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    planarity: float
    edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class GuestRing:
    ring_id: int
    nodes: tuple[int, ...]
    size: int
    center: tuple[float, float, float]
    normal: tuple[float, float, float]
    planarity: float
    score: float
    edges: tuple[tuple[int, int], ...]


@njit(cache=True)
def _has_path_node(path: np.ndarray, depth: int, node: int) -> bool:
    for index in range(depth):
        if path[index] == node:
            return True
    return False


@njit(cache=True)
def _normalize_cycle_fixed(path: np.ndarray, depth: int, out: np.ndarray) -> None:
    min_pos = 0
    min_value = path[0]

    for index in range(1, depth):
        if path[index] < min_value:
            min_value = path[index]
            min_pos = index

    forward = np.empty(depth, dtype=np.int64)
    reverse = np.empty(depth, dtype=np.int64)

    for index in range(depth):
        forward[index] = path[(min_pos + index) % depth]

    reverse[0] = forward[0]
    for index in range(1, depth):
        reverse[index] = forward[depth - index]

    use_forward = True
    for index in range(depth):
        if forward[index] < reverse[index]:
            use_forward = True
            break
        if forward[index] > reverse[index]:
            use_forward = False
            break

    for index in range(depth):
        out[index] = forward[index] if use_forward else reverse[index]


@njit(cache=True)
def _csr_has_edge(indptr: np.ndarray, indices: np.ndarray, left: int, right: int) -> bool:
    for ptr in range(indptr[left], indptr[left + 1]):
        if indices[ptr] == right:
            return True
    return False


@njit(cache=True)
def _is_chordless_cycle_csr(
    path: np.ndarray,
    depth: int,
    indptr: np.ndarray,
    indices: np.ndarray,
) -> bool:
    for i in range(depth):
        for j in range(i + 1, depth):
            if j == i + 1 or (i == 0 and j == depth - 1):
                continue
            if _csr_has_edge(indptr, indices, path[i], path[j]):
                return False
    return True


@njit(cache=True)
def find_local_cycles_kernel(
    indptr: np.ndarray,
    indices: np.ndarray,
    min_ring: int,
    max_ring: int,
    max_cycles: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    n_nodes = indptr.shape[0] - 1
    cycles = -np.ones((max_cycles, max_ring), dtype=np.int64)
    sizes = np.zeros(max_cycles, dtype=np.int64)

    if n_nodes <= 0 or max_cycles <= 0 or max_ring <= 0:
        return cycles, sizes, 0

    # DFS frontier size is governed by local branching and maximum depth, not
    # by the total number of cycles.  Capping this avoids hundreds of MB of
    # temporary stack storage when max_cycles is large.
    max_stack = max(
        1024,
        min(1_000_000, max_cycles * max_ring * 2 + n_nodes + 128),
    )
    stack_nodes = np.empty(max_stack, dtype=np.int64)
    stack_depths = np.empty(max_stack, dtype=np.int64)
    stack_paths = -np.ones((max_stack, max_ring), dtype=np.int64)

    normalized = np.empty(max_ring, dtype=np.int64)
    n_cycles = 0

    for root in range(n_nodes):
        if n_cycles >= max_cycles:
            break

        top = 0
        stack_nodes[top] = root
        stack_depths[top] = 1

        for node_index in range(max_ring):
            stack_paths[top, node_index] = -1

        stack_paths[top, 0] = root
        top += 1

        while top > 0 and n_cycles < max_cycles:
            top -= 1

            current = stack_nodes[top]
            depth = stack_depths[top]
            path = stack_paths[top].copy()

            for ptr in range(indptr[current], indptr[current + 1]):
                nxt = indices[ptr]

                if nxt < 0 or nxt >= n_nodes:
                    continue

                if nxt == root:
                    if depth >= min_ring:
                        if depth > 2 and path[1] >= path[depth - 1]:
                            continue

                        if not _is_chordless_cycle_csr(
                            path,
                            depth,
                            indptr,
                            indices,
                        ):
                            continue

                        _normalize_cycle_fixed(path, depth, normalized)

                        for node_index in range(depth):
                            cycles[n_cycles, node_index] = normalized[node_index]

                        sizes[n_cycles] = depth
                        n_cycles += 1

                        if n_cycles >= max_cycles:
                            break

                    continue

                if nxt <= root:
                    continue

                if depth >= max_ring:
                    continue

                if _has_path_node(path, depth, nxt):
                    continue

                has_chord = False
                for old_index in range(1, depth - 1):
                    if _csr_has_edge(indptr, indices, nxt, path[old_index]):
                        has_chord = True
                        break
                if has_chord:
                    continue

                if top >= max_stack:
                    continue

                stack_nodes[top] = nxt
                stack_depths[top] = depth + 1

                for node_index in range(max_ring):
                    stack_paths[top, node_index] = path[node_index]

                stack_paths[top, depth] = nxt
                top += 1

    return cycles, sizes, n_cycles


def edge_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def ring_edges(nodes: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    n_nodes = len(nodes)
    return tuple(edge_key(nodes[i], nodes[(i + 1) % n_nodes]) for i in range(n_nodes))


def normalize_cycle(path: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    path = list(path)
    n_nodes = len(path)
    min_index = min(range(n_nodes), key=lambda index: path[index])

    forward = path[min_index:] + path[:min_index]
    reverse = [forward[0]] + list(reversed(forward[1:]))

    forward_tuple = tuple(forward)
    reverse_tuple = tuple(reverse)

    return forward_tuple if forward_tuple < reverse_tuple else reverse_tuple


def is_chordless_cycle(
    cycle: tuple[int, ...],
    adjacency_sets: list[set[int]],
) -> bool:
    n_nodes = len(cycle)

    for i in range(n_nodes):
        atom_i = cycle[i]
        neighbors = adjacency_sets[atom_i]

        for j in range(i + 1, n_nodes):
            if abs(i - j) == 1:
                continue

            if i == 0 and j == n_nodes - 1:
                continue

            atom_j = cycle[j]

            if atom_j in neighbors:
                return False

    return True


def enumerate_local_cycles_python(
    indptr: np.ndarray,
    indices: np.ndarray,
    min_ring: int,
    max_ring: int,
    max_cycles: int,
) -> list[tuple[int, ...]]:
    n_nodes = len(indptr) - 1
    adjacency = []

    for node in range(n_nodes):
        start = int(indptr[node])
        end = int(indptr[node + 1])
        adjacency.append(tuple(int(value) for value in indices[start:end]))

    adjacency_sets = [set(neighbors) for neighbors in adjacency]
    degrees = [len(neighbors) for neighbors in adjacency]
    roots = [node for node, degree in enumerate(degrees) if degree >= 2]

    cycles: dict[tuple[int, ...], tuple[int, ...]] = {}

    def dfs(root: int, current: int, path: list[int], visited: set[int]) -> None:
        if len(cycles) >= max_cycles:
            return

        if len(path) > max_ring:
            return

        for nxt in adjacency[current]:
            if nxt == root:
                if min_ring <= len(path) <= max_ring:
                    if len(path) > 2 and path[1] >= path[-1]:
                        continue
                    cycle = normalize_cycle(path)

                    if cycle not in cycles and is_chordless_cycle(cycle, adjacency_sets):
                        cycles[cycle] = cycle

                continue

            if nxt in visited:
                continue

            if nxt < root:
                continue

            if len(path) >= max_ring:
                continue

            has_chord = False
            for old_node in path[1:-1]:
                if nxt in adjacency_sets[old_node]:
                    has_chord = True
                    break

            if has_chord:
                continue

            visited.add(nxt)
            path.append(nxt)

            dfs(root, nxt, path, visited)

            path.pop()
            visited.remove(nxt)

    for root in roots:
        if len(cycles) >= max_cycles:
            break

        second_nodes = [node for node in adjacency[root] if node > root]
        second_nodes.sort(key=lambda node: -degrees[node])

        for second_node in second_nodes:
            if len(cycles) >= max_cycles:
                break

            dfs(root, second_node, [root, second_node], {root, second_node})

    return list(cycles.values())


def enumerate_local_cycles_numba(
    indptr: np.ndarray,
    indices: np.ndarray,
    min_ring: int,
    max_ring: int,
    max_cycles: int,
) -> list[tuple[int, ...]]:
    cycles, sizes, n_cycles = find_local_cycles_kernel(
        np.asarray(indptr, dtype=np.int64),
        np.asarray(indices, dtype=np.int64),
        int(min_ring),
        int(max_ring),
        int(max_cycles),
    )

    output: list[tuple[int, ...]] = []

    for cycle_index in range(n_cycles):
        size = int(sizes[cycle_index])
        output.append(
            tuple(int(cycles[cycle_index, node_index]) for node_index in range(size))
        )

    return output


def enumerate_local_cycles(
    local_graph: LocalGraph,
    params: CageParameters,
) -> list[tuple[int, ...]]:
    indptr = np.asarray(local_graph.local_indptr, dtype=np.int64)
    indices = np.asarray(local_graph.local_indices, dtype=np.int64)

    backend = params.ring_backend

    if backend == "python":
        return enumerate_local_cycles_python(
            indptr=indptr,
            indices=indices,
            min_ring=params.min_ring,
            max_ring=params.max_ring,
            max_cycles=params.max_local_cycles,
        )

    if backend == "numba":
        if NUMBA_AVAILABLE:
            try:
                return enumerate_local_cycles_numba(
                    indptr=indptr,
                    indices=indices,
                    min_ring=params.min_ring,
                    max_ring=params.max_ring,
                    max_cycles=params.max_local_cycles,
                )
            except Exception:
                pass

        return enumerate_local_cycles_python(
            indptr=indptr,
            indices=indices,
            min_ring=params.min_ring,
            max_ring=params.max_ring,
            max_cycles=params.max_local_cycles,
        )

    if backend == "auto":
        if NUMBA_AVAILABLE:
            try:
                return enumerate_local_cycles_numba(
                    indptr=indptr,
                    indices=indices,
                    min_ring=params.min_ring,
                    max_ring=params.max_ring,
                    max_cycles=params.max_local_cycles,
                )
            except Exception:
                pass

        return enumerate_local_cycles_python(
            indptr=indptr,
            indices=indices,
            min_ring=params.min_ring,
            max_ring=params.max_ring,
            max_cycles=params.max_local_cycles,
        )

    raise ValueError(f"Unsupported ring backend: {backend}")






@dataclass(frozen=True)
class RingTable:
    """All rings of one frame in both object and array form.

    The array form lets the per-guest scoring loop work on the whole candidate
    set at once instead of calling a periodic-distance routine per ring.
    """

    records: list[RingRecord]
    centers: np.ndarray
    normals: np.ndarray
    planarity: np.ndarray
    sizes: np.ndarray

    def subset(self, indices) -> "RingTable":
        indices = np.asarray(indices, dtype=np.int64)
        return RingTable(
            records=[self.records[int(index)] for index in indices],
            centers=self.centers[indices],
            normals=self.normals[indices],
            planarity=self.planarity[indices],
            sizes=self.sizes[indices],
        )


def build_ring_table(records: list[RingRecord]) -> RingTable:
    if not records:
        return RingTable([], np.empty((0, 3)), np.empty((0, 3)),
                         np.empty(0), np.empty(0, dtype=np.int64))

    centers = np.asarray([record.center for record in records], dtype=np.float64)
    normals = np.asarray([record.normal for record in records], dtype=np.float64)
    planarity = np.asarray([record.planarity for record in records], dtype=np.float64)
    sizes = np.asarray([record.size for record in records], dtype=np.int64)

    return RingTable(records=records, centers=centers, normals=normals,
                     planarity=planarity, sizes=sizes)


def score_guest_rings(
    candidate_rings,
    guest_center: np.ndarray,
    box: np.ndarray,
    params: CageParameters,
    indices=None,
) -> list[GuestRing]:
    """Rings facing a guest, scored and truncated, computed in one batch."""
    if isinstance(candidate_rings, RingTable):
        table = candidate_rings

        if indices is not None:
            table = table.subset(indices)
    else:
        table = build_ring_table(list(candidate_rings))

    if len(table.records) == 0:
        return []

    guest = np.asarray(guest_center, dtype=np.float64)
    vector = pbc_delta(guest - table.centers, box)
    distance = np.linalg.norm(vector, axis=1)

    safe_distance = np.where(distance > 1.0e-8, distance, 1.0)
    unit = vector / safe_distance[:, None]
    align = np.abs(np.einsum("ij,ij->i", table.normals, unit))

    keep = (distance > 1.0e-8) & (distance <= params.face_radius)
    keep &= align >= params.face_normal_cut

    if not np.any(keep):
        return []

    score = 2.0 * align - 0.15 * table.planarity - 0.025 * distance
    candidates = np.flatnonzero(keep)
    order = np.argsort(-score[candidates], kind="stable")

    selected = candidates[order][: params.max_rings_per_guest]
    output: list[GuestRing] = []

    for position in selected:
        record = table.records[int(position)]
        output.append(
            GuestRing(
                ring_id=record.ring_id,
                nodes=record.nodes,
                size=record.size,
                center=record.center,
                normal=record.normal,
                planarity=record.planarity,
                score=float(score[int(position)]),
                edges=record.edges,
            )
        )

    return output


def estimate_directional_coverage(
    rings: list[GuestRing],
    guest_center: tuple[float, float, float] | np.ndarray,
    box: np.ndarray,
) -> float:
    if not rings:
        return 0.0

    centers = np.asarray([ring.center for ring in rings], dtype=np.float64)
    guest = np.asarray(guest_center, dtype=np.float64)
    vec = pbc_delta(centers - guest, box)
    norm = np.linalg.norm(vec, axis=1)
    valid = norm > 1.0e-8

    if not np.any(valid):
        return 0.0

    directions = vec[valid] / norm[valid, None]

    axes = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    projections = directions @ axes.T
    occupied = int(np.count_nonzero(projections.max(axis=0) > 0.45))

    return occupied / 6.0


def fit_many_rings(
    cycles,
    oxygen_pos: np.ndarray,
    box: np.ndarray,
):
    """Vectorised :func:`fit_ring` for many cycles at once.

    Returns ``(centers, normals, planarity)`` as arrays, one row per cycle.
    The centre/normal convention is identical to :func:`fit_ring`; the plane
    normal is the smallest eigenvector of the centred coordinate covariance,
    which equals the smallest right singular vector of the SVD.
    """
    n_rings = len(cycles)

    if n_rings == 0:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )

    lengths = np.fromiter(
        (len(cycle) for cycle in cycles), dtype=np.int64, count=n_rings
    )
    max_len = int(lengths.max())
    nodes = np.zeros((n_rings, max_len), dtype=np.int64)
    valid = np.zeros((n_rings, max_len), dtype=bool)

    for index, cycle in enumerate(cycles):
        size = len(cycle)
        nodes[index, :size] = cycle
        valid[index, :size] = True

    positions = np.asarray(oxygen_pos, dtype=np.float64)
    coords = positions[nodes]
    ref = coords[:, 0:1, :].copy()

    delta = pbc_delta(coords - ref, box)
    delta[~valid] = 0.0
    unwrapped = ref + delta

    weights = valid.astype(np.float64)
    counts = weights.sum(axis=1)

    for _ in range(2):
        center = (unwrapped * weights[:, :, None]).sum(axis=1) / counts[:, None]
        delta = pbc_delta(unwrapped - center[:, None, :], box)
        delta[~valid] = 0.0
        unwrapped = center[:, None, :] + delta

    center = (unwrapped * weights[:, :, None]).sum(axis=1) / counts[:, None]
    centered = unwrapped - center[:, None, :]
    centered[~valid] = 0.0

    covariance = np.einsum("nli,nlj->nij", centered, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normals = eigenvectors[:, :, 0]

    distances = np.abs(np.einsum("nli,ni->nl", centered, normals))
    distances[~valid] = 0.0
    planarity = distances.max(axis=1)

    centers = wrap_point(center, box)

    return centers, normals, planarity
