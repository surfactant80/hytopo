from __future__ import annotations

"""Micro-benchmark for the large-system local graph path.

Run from the project root with:

    python -m hytopo.benchmark_scaling --atoms 1000000 --guests 200
"""

import argparse
import time

import numpy as np

from .neighbor import (
    NeighborGraph,
    build_csr_numba,
    build_local_graph_from_nodes,
    run_kernel,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--atoms", type=int, default=1_000_000)
    parser.add_argument("--guests", type=int, default=200)
    parser.add_argument("--local-nodes", type=int, default=32)
    return parser


def legacy_global_edge_scan(
    n_atoms: int,
    pairs: np.ndarray,
    nodes: np.ndarray,
) -> np.ndarray:
    mask = np.zeros(n_atoms, dtype=np.bool_)
    mask[nodes] = True
    keep = mask[pairs[:, 0]] & mask[pairs[:, 1]]
    return pairs[keep]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    n_atoms = max(100, int(args.atoms))
    n_guests = max(1, int(args.guests))
    n_local = max(4, int(args.local_nodes))

    first = np.column_stack(
        (
            np.arange(n_atoms - 1, dtype=np.int64),
            np.arange(1, n_atoms, dtype=np.int64),
        )
    )
    second = np.column_stack(
        (
            np.arange(n_atoms - 2, dtype=np.int64),
            np.arange(2, n_atoms, dtype=np.int64),
        )
    )
    pairs = np.vstack((first, second))
    indptr, indices, degree = run_kernel(build_csr_numba, n_atoms, pairs)
    graph = NeighborGraph(
        n_atoms=n_atoms,
        cutoff=1.0,
        pairs=pairs,
        distances=np.empty(len(pairs), dtype=np.float64),
        vectors=np.empty((len(pairs), 3), dtype=np.float64),
        indptr=indptr,
        indices=indices,
        degree=degree,
    )

    max_start = max(1, n_atoms - n_local - 1)
    starts = np.linspace(0, max_start, n_guests, dtype=np.int64)
    node_sets = [
        np.arange(start, start + n_local, dtype=np.int64)
        for start in starts
    ]

    build_local_graph_from_nodes(graph, node_sets[0])
    legacy_global_edge_scan(n_atoms, pairs, node_sets[0])

    start = time.perf_counter()
    for nodes in node_sets:
        legacy_global_edge_scan(n_atoms, pairs, nodes)
    legacy_seconds = time.perf_counter() - start

    start = time.perf_counter()
    for nodes in node_sets:
        build_local_graph_from_nodes(graph, nodes)
    optimized_seconds = time.perf_counter() - start

    speedup = legacy_seconds / max(optimized_seconds, 1.0e-12)
    print(f"atoms              : {n_atoms}")
    print(f"guest neighborhoods : {n_guests}")
    print(f"legacy global scan  : {legacy_seconds:.6f} s")
    print(f"CSR local induction : {optimized_seconds:.6f} s")
    print(f"local-graph speedup : {speedup:.2f}x")
    print(
        "Note: end-to-end speedup also depends on trajectory I/O, ring count, "
        "cage density, output settings, and process count."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
