from __future__ import annotations

"""Numba kernel for the hydrate cage beam search.

The reference implementation in :mod:`hytopo.cage` keeps the partially
assembled shell as two Python integer bitmasks.  That is already much cheaper
than a dict of edge multiplicities, but it still pays interpreter overhead for
every candidate extension (~10^6 per frame on a 5888-water system).

This module ports the same search to numba, holding the edge state as
fixed-width ``uint64`` words.  The traversal order, beam updates,
de-duplication rule and scoring are identical to the Python version, so
results do not change; only the constant factor does.

The module is optional: if numba is unavailable the caller keeps using the
Python implementation.
"""

import numpy as np

try:
    from numba import njit

    NUMBA_BEAM_AVAILABLE = True
except ImportError:  # pragma: no cover - numba is an install requirement
    NUMBA_BEAM_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(function):
            return function

        return wrapper


@njit(cache=True, inline="always")
def _popcount64(value):
    value = value - ((value >> 1) & np.uint64(0x5555555555555555))
    value = (value & np.uint64(0x3333333333333333)) + (
        (value >> 2) & np.uint64(0x3333333333333333)
    )
    value = (value + (value >> 4)) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return int((value * np.uint64(0x0101010101010101)) >> 56)


@njit(cache=True, inline="always")
def _better(shared_a, score_a, ring_a, shared_b, score_b, ring_b):
    """True when candidate A is ranked before candidate B."""
    if shared_a != shared_b:
        return shared_a > shared_b
    if score_a != score_b:
        return score_a > score_b
    return ring_a < ring_b


@njit(cache=True, inline="always")
def _stable_sort_desc(keys, order, count, limit):
    """Stable descending insertion sort of ``order[0:count]`` by ``keys``.

    Returns the number of retained entries (``min(count, limit)``); the full
    range is sorted, the caller slices it.
    """
    for index in range(1, count):
        key = keys[index]
        slot = order[index]
        position = index - 1

        while position >= 0 and keys[position] < key:
            keys[position + 1] = keys[position]
            order[position + 1] = order[position]
            position -= 1

        keys[position + 1] = key
        order[position + 1] = slot

    return min(count, limit)


@njit(cache=True)
def beam_search_numba(
    ring_edge_words,
    ring_sel_words,
    ring_sizes,
    ring_scores,
    edge_ring_words,
    target4,
    target5,
    target6,
    target_faces,
    beam_width,
    candidate_limit,
):
    """Beam search for one guest and one target cage.

    Returns ``(count, sel, once, twice, counts, score)`` describing every
    complete candidate shell, in the original traversal order.
    """
    n_rings = ring_sizes.shape[0]
    n_edge_words = ring_edge_words.shape[1]
    n_sel_words = ring_sel_words.shape[1]

    empty_sel = np.zeros((0, n_sel_words), dtype=np.uint64)
    empty_edge = np.zeros((0, n_edge_words), dtype=np.uint64)
    empty_counts = np.zeros((0, 3), dtype=np.int64)
    empty_score = np.zeros(0, dtype=np.float64)

    s_sel = np.zeros((beam_width, n_sel_words), dtype=np.uint64)
    s_once = np.zeros((beam_width, n_edge_words), dtype=np.uint64)
    s_twice = np.zeros((beam_width, n_edge_words), dtype=np.uint64)
    s_counts = np.zeros((beam_width, 3), dtype=np.int64)
    s_score = np.zeros(beam_width, dtype=np.float64)

    seed_sel = np.zeros((n_rings, n_sel_words), dtype=np.uint64)
    seed_once = np.zeros((n_rings, n_edge_words), dtype=np.uint64)
    seed_counts = np.zeros((n_rings, 3), dtype=np.int64)
    seed_score = np.zeros(n_rings, dtype=np.float64)
    seed_key = np.zeros(n_rings, dtype=np.float64)
    seed_order = np.zeros(n_rings, dtype=np.int64)

    # ---- seed states ----
    n_seed = 0

    for ring_id in range(n_rings):
        size = ring_sizes[ring_id]
        c4 = 1 if size == 4 else 0
        c5 = 1 if size == 5 else 0
        c6 = 1 if size == 6 else 0

        if c4 > target4 or c5 > target5 or c6 > target6:
            continue

        for word in range(n_sel_words):
            seed_sel[n_seed, word] = ring_sel_words[ring_id, word]
        for word in range(n_edge_words):
            seed_once[n_seed, word] = ring_edge_words[ring_id, word]

        seed_counts[n_seed, 0] = c4
        seed_counts[n_seed, 1] = c5
        seed_counts[n_seed, 2] = c6
        seed_score[n_seed] = ring_scores[ring_id]
        seed_key[n_seed] = ring_scores[ring_id]
        seed_order[n_seed] = n_seed
        n_seed += 1

    if n_seed == 0:
        return 0, empty_sel, empty_edge, empty_edge, empty_counts, empty_score

    n_states = _stable_sort_desc(seed_key, seed_order, n_seed, beam_width)

    for index in range(n_states):
        source = seed_order[index]
        for word in range(n_sel_words):
            s_sel[index, word] = seed_sel[source, word]
        for word in range(n_edge_words):
            s_once[index, word] = seed_once[source, word]
            s_twice[index, word] = np.uint64(0)
        for column in range(3):
            s_counts[index, column] = seed_counts[source, column]
        s_score[index] = seed_score[source]

    # ---- grow ----
    max_new = beam_width * candidate_limit + 1

    n_sel_new = np.zeros((max_new, n_sel_words), dtype=np.uint64)
    n_once_new = np.zeros((max_new, n_edge_words), dtype=np.uint64)
    n_twice_new = np.zeros((max_new, n_edge_words), dtype=np.uint64)
    n_counts_new = np.zeros((max_new, 3), dtype=np.int64)
    n_score_new = np.zeros(max_new, dtype=np.float64)

    cand_ring = np.zeros(n_rings, dtype=np.int64)
    cand_shared = np.zeros(n_rings, dtype=np.int64)
    cand_score = np.zeros(n_rings, dtype=np.float64)

    hash_size = 1
    while hash_size < max_new * 2:
        hash_size *= 2

    hash_keys = np.full(hash_size, -1, dtype=np.int64)
    hash_slot = np.zeros(hash_size, dtype=np.int64)
    order = np.zeros(max_new, dtype=np.int64)
    keys = np.zeros(max_new, dtype=np.float64)

    cand = np.zeros(n_sel_words, dtype=np.uint64)

    for _round in range(1, target_faces):
        n_new = 0

        for state in range(n_states):
            for word in range(n_sel_words):
                cand[word] = np.uint64(0)

            has_boundary = False

            for word in range(n_edge_words):
                bits = s_once[state, word]

                while bits != np.uint64(0):
                    low = bits & (np.uint64(0) - bits)
                    bits ^= low
                    edge_id = word * 64 + _popcount64(low - np.uint64(1))
                    for word_sel in range(n_sel_words):
                        cand[word_sel] |= edge_ring_words[edge_id, word_sel]
                    has_boundary = True

            if not has_boundary:
                continue

            any_candidate = False

            for word_sel in range(n_sel_words):
                cand[word_sel] &= ~s_sel[state, word_sel]
                if cand[word_sel] != np.uint64(0):
                    any_candidate = True

            if not any_candidate:
                continue

            # ---- rank the candidate rings ----
            n_cand = 0

            for word_sel in range(n_sel_words):
                bits = cand[word_sel]

                while bits != np.uint64(0):
                    low = bits & (np.uint64(0) - bits)
                    bits ^= low
                    ring_id = word_sel * 64 + _popcount64(low - np.uint64(1))

                    shared = 0
                    for word in range(n_edge_words):
                        shared += _popcount64(
                            s_once[state, word] & ring_edge_words[ring_id, word]
                        )

                    if shared == 0:
                        continue

                    cand_ring[n_cand] = ring_id
                    cand_shared[n_cand] = shared
                    cand_score[n_cand] = ring_scores[ring_id]
                    n_cand += 1

            limit = candidate_limit if candidate_limit < n_cand else n_cand

            for slot in range(limit):
                best = slot

                for other in range(slot + 1, n_cand):
                    if _better(
                        cand_shared[other],
                        cand_score[other],
                        cand_ring[other],
                        cand_shared[best],
                        cand_score[best],
                        cand_ring[best],
                    ):
                        best = other

                if best != slot:
                    tmp_ring = cand_ring[slot]
                    tmp_shared = cand_shared[slot]
                    tmp_score = cand_score[slot]
                    cand_ring[slot] = cand_ring[best]
                    cand_shared[slot] = cand_shared[best]
                    cand_score[slot] = cand_score[best]
                    cand_ring[best] = tmp_ring
                    cand_shared[best] = tmp_shared
                    cand_score[best] = tmp_score

            # ---- extend ----
            for index in range(limit):
                ring_id = cand_ring[index]
                size = ring_sizes[ring_id]

                c4 = s_counts[state, 0]
                c5 = s_counts[state, 1]
                c6 = s_counts[state, 2]

                if size == 4:
                    c4 += 1
                elif size == 5:
                    c5 += 1
                elif size == 6:
                    c6 += 1

                if c4 > target4 or c5 > target5 or c6 > target6:
                    continue

                conflict = False

                for word in range(n_edge_words):
                    if (
                        s_twice[state, word] & ring_edge_words[ring_id, word]
                    ) != np.uint64(0):
                        conflict = True
                        break

                if conflict:
                    continue

                if n_new >= max_new:
                    break

                double_edges = 0
                total_edges = 0

                for word in range(n_edge_words):
                    face_mask = ring_edge_words[ring_id, word]
                    twice = s_twice[state, word] | (s_once[state, word] & face_mask)
                    once = (s_once[state, word] ^ face_mask) & ~twice
                    n_twice_new[n_new, word] = twice
                    n_once_new[n_new, word] = once

                    doubles = _popcount64(twice)
                    double_edges += doubles
                    total_edges += _popcount64(once) + 2 * doubles

                for word_sel in range(n_sel_words):
                    n_sel_new[n_new, word_sel] = (
                        s_sel[state, word_sel] | ring_sel_words[ring_id, word_sel]
                    )

                n_counts_new[n_new, 0] = c4
                n_counts_new[n_new, 1] = c5
                n_counts_new[n_new, 2] = c6

                closure = 2.0 * double_edges / total_edges if total_edges else 0.0
                n_score_new[n_new] = (
                    s_score[state] + ring_scores[ring_id] + 3.0 * closure
                )
                n_new += 1

        if n_new == 0:
            return 0, empty_sel, empty_edge, empty_edge, empty_counts, empty_score

        # ---- de-duplicate by ring selection, keeping the best score ----
        n_unique = 0

        for index in range(hash_size):
            hash_keys[index] = -1

        for index in range(n_new):
            hash_value = np.uint64(1469598103934665603)

            for word_sel in range(n_sel_words):
                hash_value = (hash_value ^ n_sel_new[index, word_sel]) * np.uint64(
                    1099511628211
                )

            slot = int(hash_value) & (hash_size - 1)

            while True:
                stored = hash_keys[slot]

                if stored < 0:
                    hash_keys[slot] = index
                    hash_slot[slot] = n_unique
                    order[n_unique] = index
                    n_unique += 1
                    break

                same = True

                for word_sel in range(n_sel_words):
                    if n_sel_new[index, word_sel] != n_sel_new[stored, word_sel]:
                        same = False
                        break

                if same:
                    if n_score_new[index] > n_score_new[stored]:
                        for word_sel in range(n_sel_words):
                            n_sel_new[stored, word_sel] = n_sel_new[index, word_sel]
                        for word in range(n_edge_words):
                            n_once_new[stored, word] = n_once_new[index, word]
                            n_twice_new[stored, word] = n_twice_new[index, word]
                        for column in range(3):
                            n_counts_new[stored, column] = n_counts_new[index, column]
                        n_score_new[stored] = n_score_new[index]
                    break

                slot = (slot + 1) & (hash_size - 1)

        for index in range(n_unique):
            keys[index] = n_score_new[order[index]]

        n_states = _stable_sort_desc(keys, order, n_unique, beam_width)

        for index in range(n_states):
            source = order[index]
            for word_sel in range(n_sel_words):
                s_sel[index, word_sel] = n_sel_new[source, word_sel]
            for word in range(n_edge_words):
                s_once[index, word] = n_once_new[source, word]
                s_twice[index, word] = n_twice_new[source, word]
            for column in range(3):
                s_counts[index, column] = n_counts_new[source, column]
            s_score[index] = n_score_new[source]

    # ---- collect complete shells ----
    n_final = 0
    final_sel = np.zeros((n_states, n_sel_words), dtype=np.uint64)
    final_once = np.zeros((n_states, n_edge_words), dtype=np.uint64)
    final_twice = np.zeros((n_states, n_edge_words), dtype=np.uint64)
    final_counts = np.zeros((n_states, 3), dtype=np.int64)
    final_score = np.zeros(n_states, dtype=np.float64)

    for index in range(n_states):
        if (
            s_counts[index, 0] != target4
            or s_counts[index, 1] != target5
            or s_counts[index, 2] != target6
        ):
            continue

        for word_sel in range(n_sel_words):
            final_sel[n_final, word_sel] = s_sel[index, word_sel]
        for word in range(n_edge_words):
            final_once[n_final, word] = s_once[index, word]
            final_twice[n_final, word] = s_twice[index, word]
        for column in range(3):
            final_counts[n_final, column] = s_counts[index, column]
        final_score[n_final] = s_score[index]
        n_final += 1

    return (
        n_final,
        final_sel[:n_final].copy(),
        final_once[:n_final].copy(),
        final_twice[:n_final].copy(),
        final_counts[:n_final].copy(),
        final_score[:n_final].copy(),
    )


def beam_search(
    ring_edge_words: np.ndarray,
    ring_sel_words: np.ndarray,
    ring_sizes: np.ndarray,
    ring_scores: np.ndarray,
    edge_ring_words: np.ndarray,
    target: tuple[int, int, int],
    beam_width: int,
    candidate_limit: int,
):
    """Thin wrapper around the numba kernel; returns ``None`` without numba."""
    if not NUMBA_BEAM_AVAILABLE:
        return None

    return beam_search_numba(
        ring_edge_words,
        ring_sel_words,
        ring_sizes,
        ring_scores,
        edge_ring_words,
        int(target[0]),
        int(target[1]),
        int(target[2]),
        int(target[0] + target[1] + target[2]),
        int(beam_width),
        int(candidate_limit),
    )
