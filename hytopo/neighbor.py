from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from MDAnalysis.lib.distances import capped_distance, minimize_vectors, self_capped_distance

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper


NUMBA_RUNTIME_ENABLED = True


def set_numba_enabled(enabled: bool) -> None:
    global NUMBA_RUNTIME_ENABLED
    NUMBA_RUNTIME_ENABLED = bool(enabled)


@dataclass(frozen=True)
class NeighborGraph:
    n_atoms: int
    cutoff: float
    pairs: np.ndarray
    distances: np.ndarray
    vectors: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    degree: np.ndarray


@dataclass(frozen=True)
class NearestNeighbors:
    indices: np.ndarray
    distances: np.ndarray
    vectors: np.ndarray


@dataclass(frozen=True)
class LocalGraph:
    global_nodes: np.ndarray
    local_pairs: np.ndarray
    local_indptr: np.ndarray
    local_indices: np.ndarray
    local_to_global: np.ndarray
    global_to_local: np.ndarray


def run_kernel(function, *args):
    if not NUMBA_RUNTIME_ENABLED and hasattr(function, "py_func"):
        return function.py_func(*args)
    try:
        return function(*args)
    except Exception:
        if hasattr(function, "py_func"):
            return function.py_func(*args)
        raise


@njit(cache=True)
def build_csr_numba(n_atoms: int, pairs: np.ndarray):
    degree = np.zeros(n_atoms, dtype=np.int64)

    for pair_index in range(pairs.shape[0]):
        i = pairs[pair_index, 0]
        j = pairs[pair_index, 1]
        degree[i] += 1
        degree[j] += 1

    indptr = np.zeros(n_atoms + 1, dtype=np.int64)

    for atom_index in range(n_atoms):
        indptr[atom_index + 1] = indptr[atom_index] + degree[atom_index]

    indices = np.empty(indptr[-1], dtype=np.int64)
    cursor = indptr[:-1].copy()

    for pair_index in range(pairs.shape[0]):
        i = pairs[pair_index, 0]
        j = pairs[pair_index, 1]

        indices[cursor[i]] = j
        cursor[i] += 1

        indices[cursor[j]] = i
        cursor[j] += 1

    return indptr, indices, degree


@njit(cache=True)
def build_nearest_k_numba(
    n_atoms: int,
    pairs: np.ndarray,
    distances: np.ndarray,
    vectors: np.ndarray,
    k_neighbors: int,
):
    nearest_idx = -np.ones((n_atoms, k_neighbors), dtype=np.int64)
    nearest_dist = np.empty((n_atoms, k_neighbors), dtype=np.float64)
    nearest_vec = np.zeros((n_atoms, k_neighbors, 3), dtype=np.float64)

    for atom_index in range(n_atoms):
        for neighbor_slot in range(k_neighbors):
            nearest_dist[atom_index, neighbor_slot] = 1.0e30

    for pair_index in range(pairs.shape[0]):
        i = pairs[pair_index, 0]
        j = pairs[pair_index, 1]
        distance = distances[pair_index]

        max_slot = 0
        max_distance = nearest_dist[i, 0]

        for neighbor_slot in range(1, k_neighbors):
            if nearest_dist[i, neighbor_slot] > max_distance:
                max_distance = nearest_dist[i, neighbor_slot]
                max_slot = neighbor_slot

        if distance < max_distance:
            nearest_idx[i, max_slot] = j
            nearest_dist[i, max_slot] = distance
            nearest_vec[i, max_slot, 0] = vectors[pair_index, 0]
            nearest_vec[i, max_slot, 1] = vectors[pair_index, 1]
            nearest_vec[i, max_slot, 2] = vectors[pair_index, 2]

        max_slot = 0
        max_distance = nearest_dist[j, 0]

        for neighbor_slot in range(1, k_neighbors):
            if nearest_dist[j, neighbor_slot] > max_distance:
                max_distance = nearest_dist[j, neighbor_slot]
                max_slot = neighbor_slot

        if distance < max_distance:
            nearest_idx[j, max_slot] = i
            nearest_dist[j, max_slot] = distance
            nearest_vec[j, max_slot, 0] = -vectors[pair_index, 0]
            nearest_vec[j, max_slot, 1] = -vectors[pair_index, 1]
            nearest_vec[j, max_slot, 2] = -vectors[pair_index, 2]

    return nearest_idx, nearest_dist, nearest_vec


@njit(cache=True)
def build_local_pairs_from_csr_numba(
    global_nodes: np.ndarray,
    global_indptr: np.ndarray,
    global_indices: np.ndarray,
):
    n_edges = 0

    for local_i in range(global_nodes.shape[0]):
        global_i = global_nodes[local_i]
        for ptr in range(global_indptr[global_i], global_indptr[global_i + 1]):
            target = global_indices[ptr]
            left = 0
            right = global_nodes.shape[0] - 1
            local_j = -1
            while left <= right:
                middle = (left + right) // 2
                value = global_nodes[middle]
                if value == target:
                    local_j = middle
                    break
                if value < target:
                    left = middle + 1
                else:
                    right = middle - 1
            if local_j > local_i:
                n_edges += 1

    local_pairs = np.empty((n_edges, 2), dtype=np.int64)
    cursor = 0

    for local_i in range(global_nodes.shape[0]):
        global_i = global_nodes[local_i]
        for ptr in range(global_indptr[global_i], global_indptr[global_i + 1]):
            target = global_indices[ptr]
            left = 0
            right = global_nodes.shape[0] - 1
            local_j = -1
            while left <= right:
                middle = (left + right) // 2
                value = global_nodes[middle]
                if value == target:
                    local_j = middle
                    break
                if value < target:
                    left = middle + 1
                else:
                    right = middle - 1
            if local_j > local_i:
                local_pairs[cursor, 0] = local_i
                local_pairs[cursor, 1] = local_j
                cursor += 1

    return local_pairs


@njit(cache=True)
def two_core_mask_numba(indptr: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n_nodes = indptr.shape[0] - 1
    degree = np.empty(n_nodes, dtype=np.int64)
    active = np.ones(n_nodes, dtype=np.bool_)
    queue = np.empty(n_nodes, dtype=np.int64)
    head = 0
    tail = 0

    for node in range(n_nodes):
        degree[node] = indptr[node + 1] - indptr[node]
        if degree[node] < 2:
            queue[tail] = node
            tail += 1
            active[node] = False

    while head < tail:
        node = queue[head]
        head += 1
        for ptr in range(indptr[node], indptr[node + 1]):
            neighbor = indices[ptr]
            if active[neighbor]:
                degree[neighbor] -= 1
                if degree[neighbor] < 2:
                    active[neighbor] = False
                    queue[tail] = neighbor
                    tail += 1

    return active


def minimum_image_vectors(positions: np.ndarray, pairs: np.ndarray, box: np.ndarray) -> np.ndarray:
    if len(pairs) == 0:
        return np.empty((0, 3), dtype=np.float64)

    vectors = positions[pairs[:, 1]] - positions[pairs[:, 0]]
    return minimize_vectors(vectors, box=box).astype(np.float64, copy=False)


def build_oo_graph(
    positions: np.ndarray,
    box: np.ndarray,
    cutoff: float,
) -> NeighborGraph:
    positions = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)

    pairs, distances = self_capped_distance(
        positions,
        max_cutoff=cutoff,
        box=box,
        return_distances=True,
    )

    pairs = pairs.astype(np.int64, copy=False)
    distances = distances.astype(np.float64, copy=False)

    vectors = minimum_image_vectors(positions, pairs, box)
    indptr, indices, degree = run_kernel(build_csr_numba, len(positions), pairs)

    return NeighborGraph(
        n_atoms=len(positions),
        cutoff=float(cutoff),
        pairs=pairs,
        distances=distances,
        vectors=vectors,
        indptr=indptr,
        indices=indices,
        degree=degree,
    )


def build_cage_oo_graph(
    positions: np.ndarray,
    box: np.ndarray,
    cutoff: float,
) -> NeighborGraph:
    return build_oo_graph(positions=positions, box=box, cutoff=cutoff)


def build_nearest_neighbors(
    graph: NeighborGraph,
    k_neighbors: int = 4,
    use_numba: bool = True,
) -> NearestNeighbors:
    kernel = build_nearest_k_numba
    if not use_numba and hasattr(kernel, "py_func"):
        kernel = kernel.py_func

    nearest_idx, nearest_dist, nearest_vec = run_kernel(
        kernel,
        graph.n_atoms,
        graph.pairs,
        graph.distances,
        graph.vectors,
        k_neighbors,
    )

    return NearestNeighbors(
        indices=nearest_idx,
        distances=nearest_dist,
        vectors=nearest_vec,
    )


def select_nodes_within_radius(
    centers: np.ndarray,
    positions: np.ndarray,
    box: np.ndarray,
    radius: float,
) -> list[np.ndarray]:
    centers = np.asarray(centers, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    box = np.asarray(box, dtype=np.float64)

    if len(centers) == 0 or len(positions) == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(len(centers))]

    pairs = capped_distance(
        centers,
        positions,
        max_cutoff=radius,
        box=box,
        return_distances=False,
    )

    result = [[] for _ in range(len(centers))]

    for center_index, atom_index in pairs:
        result[int(center_index)].append(int(atom_index))

    return [
        np.asarray(sorted(nodes), dtype=np.int64)
        for nodes in result
    ]


def select_guest_local_nodes(
    guest_center: np.ndarray,
    oxygen_positions: np.ndarray,
    box: np.ndarray,
    radius: float,
    candidate_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    guest_center = np.asarray(guest_center, dtype=np.float64).reshape(1, 3)

    pairs = capped_distance(
        guest_center,
        oxygen_positions,
        max_cutoff=radius,
        box=box,
        return_distances=False,
    )

    if len(pairs) == 0:
        return np.empty(0, dtype=np.int64)

    nodes = np.asarray(sorted(set(int(pair[1]) for pair in pairs)), dtype=np.int64)

    if candidate_mask is not None:
        nodes = nodes[candidate_mask[nodes]]

    return nodes


def build_local_graph_from_nodes(
    global_graph: NeighborGraph,
    global_nodes: np.ndarray,
) -> LocalGraph:
    global_nodes = np.asarray(global_nodes, dtype=np.int64)

    if len(global_nodes) == 0:
        return LocalGraph(
            global_nodes=global_nodes,
            local_pairs=np.empty((0, 2), dtype=np.int64),
            local_indptr=np.zeros(1, dtype=np.int64),
            local_indices=np.empty(0, dtype=np.int64),
            local_to_global=global_nodes,
            global_to_local=np.empty(0, dtype=np.int64),
        )

    global_nodes = np.asarray(sorted(set(int(node) for node in global_nodes)), dtype=np.int64)

    local_pairs = run_kernel(
        build_local_pairs_from_csr_numba,
        global_nodes,
        global_graph.indptr,
        global_graph.indices,
    )

    local_indptr, local_indices, _ = run_kernel(
        build_csr_numba,
        len(global_nodes),
        local_pairs,
    )

    return LocalGraph(
        global_nodes=global_nodes,
        local_pairs=local_pairs,
        local_indptr=local_indptr,
        local_indices=local_indices,
        local_to_global=global_nodes,
        global_to_local=np.empty(0, dtype=np.int64),
    )




def prune_local_graph_to_two_core(local_graph: LocalGraph) -> LocalGraph:
    if len(local_graph.global_nodes) == 0:
        return local_graph

    keep = run_kernel(
        two_core_mask_numba,
        local_graph.local_indptr,
        local_graph.local_indices,
    )
    if np.all(keep):
        return local_graph

    kept_old = np.flatnonzero(keep)
    global_nodes = local_graph.local_to_global[kept_old]
    old_to_new = -np.ones(len(local_graph.global_nodes), dtype=np.int64)
    old_to_new[kept_old] = np.arange(len(kept_old), dtype=np.int64)

    if len(local_graph.local_pairs) == 0:
        local_pairs = np.empty((0, 2), dtype=np.int64)
    else:
        pair_keep = keep[local_graph.local_pairs[:, 0]] & keep[local_graph.local_pairs[:, 1]]
        local_pairs = old_to_new[local_graph.local_pairs[pair_keep]]

    local_indptr, local_indices, _ = run_kernel(
        build_csr_numba,
        len(global_nodes),
        local_pairs,
    )
    return LocalGraph(
        global_nodes=global_nodes,
        local_pairs=local_pairs,
        local_indptr=local_indptr,
        local_indices=local_indices,
        local_to_global=global_nodes,
        global_to_local=np.empty(0, dtype=np.int64),
    )


def filter_graph_nodes_by_mask(
    graph: NeighborGraph,
    node_mask: np.ndarray,
) -> LocalGraph:
    node_mask = np.asarray(node_mask, dtype=np.bool_)

    if node_mask.shape[0] != graph.n_atoms:
        raise ValueError("node_mask length does not match graph.n_atoms.")

    global_nodes = np.where(node_mask)[0].astype(np.int64)

    return build_local_graph_from_nodes(
        global_graph=graph,
        global_nodes=global_nodes,
    )


def csr_neighbors(indptr: np.ndarray, indices: np.ndarray, atom_index: int) -> np.ndarray:
    start = indptr[atom_index]
    end = indptr[atom_index + 1]
    return indices[start:end]


def graph_density(graph: NeighborGraph) -> float:
    if graph.n_atoms <= 1:
        return 0.0

    possible_edges = graph.n_atoms * (graph.n_atoms - 1) / 2.0
    return float(len(graph.pairs) / possible_edges)


def local_graph_density(local_graph: LocalGraph) -> float:
    n_nodes = len(local_graph.global_nodes)

    if n_nodes <= 1:
        return 0.0

    possible_edges = n_nodes * (n_nodes - 1) / 2.0
    return float(len(local_graph.local_pairs) / possible_edges)


def remap_local_cycle_to_global(
    cycle: tuple[int, ...] | list[int] | np.ndarray,
    local_to_global: np.ndarray,
) -> tuple[int, ...]:
    return tuple(int(local_to_global[int(node)]) for node in cycle)


def remap_local_cycles_to_global(
    cycles: list[tuple[int, ...]],
    local_to_global: np.ndarray,
) -> list[tuple[int, ...]]:
    return [
        remap_local_cycle_to_global(cycle, local_to_global)
        for cycle in cycles
    ]


def adjacency_tuple_from_csr(indptr: np.ndarray, indices: np.ndarray) -> tuple[tuple[int, ...], ...]:
    n_atoms = len(indptr) - 1
    adjacency = []

    for atom_index in range(n_atoms):
        start = indptr[atom_index]
        end = indptr[atom_index + 1]
        adjacency.append(tuple(int(value) for value in indices[start:end]))

    return tuple(adjacency)


def validate_graph(graph: NeighborGraph) -> None:
    if graph.pairs.ndim != 2 or graph.pairs.shape[1] != 2:
        raise ValueError("pairs must have shape (n_pairs, 2).")

    if graph.distances.shape[0] != graph.pairs.shape[0]:
        raise ValueError("distances length must match pairs length.")

    if graph.vectors.shape != (graph.pairs.shape[0], 3):
        raise ValueError("vectors must have shape (n_pairs, 3).")

    if graph.indptr.shape[0] != graph.n_atoms + 1:
        raise ValueError("indptr length must be n_atoms + 1.")

    if graph.degree.shape[0] != graph.n_atoms:
        raise ValueError("degree length must match n_atoms.")
