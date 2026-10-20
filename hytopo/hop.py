from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper

try:
    from scipy.special import sph_harm_y
    USE_SPH_HARM_Y = True
    USE_SCIPY_SPH_HARM = True
except Exception:
    try:
        from scipy.special import sph_harm
        USE_SPH_HARM_Y = False
        USE_SCIPY_SPH_HARM = True
    except Exception:
        sph_harm = None
        USE_SPH_HARM_Y = False
        USE_SCIPY_SPH_HARM = False

from .config import (
    HopParameters,
    LABEL_HYD,
    LABEL_IC,
    LABEL_IH,
    LABEL_LIQ,
    LABEL_PRE,
    LABEL_UNK,
)
from .neighbor import NeighborGraph, build_nearest_neighbors


@dataclass(frozen=True)
class PhaseProbabilities:
    P_liq: np.ndarray
    P_Ih: np.ndarray
    P_Ic: np.ndarray
    P_hyd: np.ndarray
    P_pre: np.ndarray

    @property
    def P_hyd_total(self) -> np.ndarray:
        return self.P_hyd + self.P_pre

    @property
    def P_ice_total(self) -> np.ndarray:
        return self.P_Ih + self.P_Ic


@dataclass(frozen=True)
class LocalDescriptors:
    qtet: np.ndarray
    S: np.ndarray
    E: np.ndarray
    coordination: np.ndarray
    raw_hyd: np.ndarray
    raw_pre: np.ndarray
    hyd_support: np.ndarray
    pre_support: np.ndarray
    ice_support: np.ndarray


@dataclass(frozen=True)
class HopStats:
    H_total: float
    H_crys: float
    H_pre: float
    Hyd_order: float
    Ice: float
    O_total: float
    Delta: float
    Nmax_hyd: float
    Nmax_ice: float


@dataclass(frozen=True)
class HopResult:
    stats: HopStats
    labels: np.ndarray
    probabilities: PhaseProbabilities
    descriptors: LocalDescriptors


@njit(cache=True)
def calc_qtet_all_numba(nearest_idx: np.ndarray, nearest_vec: np.ndarray, eps: float):
    n_atoms = nearest_idx.shape[0]
    qtet = np.zeros(n_atoms, dtype=np.float64)

    for atom_index in range(n_atoms):
        valid = True
        unit = np.zeros((4, 3), dtype=np.float64)

        for neighbor_slot in range(4):
            if nearest_idx[atom_index, neighbor_slot] < 0:
                valid = False
                break

            vx = nearest_vec[atom_index, neighbor_slot, 0]
            vy = nearest_vec[atom_index, neighbor_slot, 1]
            vz = nearest_vec[atom_index, neighbor_slot, 2]
            norm = np.sqrt(vx * vx + vy * vy + vz * vz)

            if norm < eps:
                valid = False
                break

            unit[neighbor_slot, 0] = vx / norm
            unit[neighbor_slot, 1] = vy / norm
            unit[neighbor_slot, 2] = vz / norm

        if not valid:
            qtet[atom_index] = 0.0
            continue

        add = 0.0

        for a in range(4):
            for b in range(a + 1, 4):
                cosang = (
                    unit[a, 0] * unit[b, 0]
                    + unit[a, 1] * unit[b, 1]
                    + unit[a, 2] * unit[b, 2]
                )
                add += (cosang + 1.0 / 3.0) ** 2

        q = 1.0 - 3.0 / 8.0 * add

        if q > 0.0:
            qtet[atom_index] = q
        else:
            qtet[atom_index] = 0.0

    return qtet


@njit(cache=True)
def calc_coordination_numba(
    n_atoms: int,
    pairs: np.ndarray,
    distances: np.ndarray,
    cutoff: float,
    dist_smooth: float,
):
    coordination = np.zeros(n_atoms, dtype=np.float64)

    for pair_index in range(pairs.shape[0]):
        i = pairs[pair_index, 0]
        j = pairs[pair_index, 1]
        distance = distances[pair_index]
        weight = 1.0 / (1.0 + np.exp((distance - cutoff) / dist_smooth))

        coordination[i] += weight
        coordination[j] += weight

    return coordination


@njit(cache=True)
def accumulate_pair_values_numba(n_atoms: int, pairs: np.ndarray, values: np.ndarray):
    out = np.zeros(n_atoms, dtype=np.float64)

    for pair_index in range(pairs.shape[0]):
        i = pairs[pair_index, 0]
        j = pairs[pair_index, 1]
        value = values[pair_index]

        out[i] += value
        out[j] += value

    return out


@njit(cache=True)
def neighbor_average_numba(values: np.ndarray, indptr: np.ndarray, indices: np.ndarray):
    n_atoms = values.shape[0]
    avg = np.zeros(n_atoms, dtype=np.float64)

    for atom_index in range(n_atoms):
        start = indptr[atom_index]
        end = indptr[atom_index + 1]
        count = end - start

        if count == 0:
            avg[atom_index] = values[atom_index]
        else:
            total = 0.0
            for ptr in range(start, end):
                total += values[indices[ptr]]
            avg[atom_index] = total / count

    return avg


@njit(cache=True)
def neighbor_fraction_from_scores_numba(
    scores: np.ndarray,
    threshold: float,
    indptr: np.ndarray,
    indices: np.ndarray,
):
    n_atoms = scores.shape[0]
    fraction = np.zeros(n_atoms, dtype=np.float64)

    for atom_index in range(n_atoms):
        start = indptr[atom_index]
        end = indptr[atom_index + 1]
        count = end - start

        if count == 0:
            fraction[atom_index] = 0.0
            continue

        hit = 0.0

        for ptr in range(start, end):
            neighbor = indices[ptr]
            if scores[neighbor] >= threshold:
                hit += 1.0

        fraction[atom_index] = hit / count

    return fraction


@njit(cache=True)
def largest_cluster_size_csr_numba(
    mask: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    weights: np.ndarray,
):
    n_atoms = mask.shape[0]
    visited = np.zeros(n_atoms, dtype=np.bool_)
    queue = np.empty(n_atoms, dtype=np.int64)
    max_size = 0.0

    for start_atom in range(n_atoms):
        if visited[start_atom] or not mask[start_atom]:
            continue

        head = 0
        tail = 0
        queue[tail] = start_atom
        tail += 1
        visited[start_atom] = True
        cluster_size = 0.0

        while head < tail:
            current = queue[head]
            head += 1

            cluster_size += weights[current]

            for ptr in range(indptr[current], indptr[current + 1]):
                neighbor = indices[ptr]

                if not visited[neighbor] and mask[neighbor]:
                    visited[neighbor] = True
                    queue[tail] = neighbor
                    tail += 1

        if cluster_size > max_size:
            max_size = cluster_size

    return max_size


@njit(cache=True)
def smooth_step_numba(x: float, center: float, width: float):
    return 1.0 / (1.0 + np.exp(-(x - center) / width))


@njit(cache=True)
def smooth_less_numba(x: float, center: float, width: float):
    return 1.0 / (1.0 + np.exp((x - center) / width))


@njit(cache=True)
def gaussian_count_numba(value: float, target: float, sigma_count: float):
    return np.exp(-((value - target) ** 2) / (2.0 * sigma_count ** 2))


@njit(cache=True)
def calculate_phase_probabilities_numba(
    qtet: np.ndarray,
    staggered_count: np.ndarray,
    eclipsed_count: np.ndarray,
    coordination: np.ndarray,
    qtet_nb: np.ndarray,
    S_nb: np.ndarray,
    E_nb: np.ndarray,
    indptr: np.ndarray,
    indices: np.ndarray,
    hyd_crys_e_center: float,
    hyd_crys_e_width: float,
    hyd_crys_s_max: float,
    hyd_crys_q_min: float,
    hyd_crys_raw_min: float,
    hyd_crys_support_min: float,
    hyd_pre_e_low: float,
    hyd_pre_e_high: float,
    hyd_pre_s_max: float,
    hyd_pre_q_min: float,
    hyd_pre_raw_min: float,
    hyd_pre_support_min: float,
    hyd_pre_neighbor_fraction_min: float,
    ice_q_min: float,
    coord_min: float,
    coord_max: float,
    sigma_count: float,
    eps: float,
):
    n_atoms = qtet.shape[0]

    P_liq = np.zeros(n_atoms, dtype=np.float64)
    P_Ih = np.zeros(n_atoms, dtype=np.float64)
    P_Ic = np.zeros(n_atoms, dtype=np.float64)
    P_hyd = np.zeros(n_atoms, dtype=np.float64)
    P_pre = np.zeros(n_atoms, dtype=np.float64)

    raw_hyd = np.zeros(n_atoms, dtype=np.float64)
    raw_pre = np.zeros(n_atoms, dtype=np.float64)
    raw_ice = np.zeros(n_atoms, dtype=np.float64)

    labels = np.zeros(n_atoms, dtype=np.int32)

    for atom_index in range(n_atoms):
        q = qtet[atom_index]
        S = staggered_count[atom_index]
        E = eclipsed_count[atom_index]
        coord = coordination[atom_index]

        q_env = 0.85 * q + 0.15 * qtet_nb[atom_index]
        S_env = 0.85 * S + 0.15 * S_nb[atom_index]
        E_env = 0.85 * E + 0.15 * E_nb[atom_index]

        coord_good = (
            smooth_step_numba(coord, coord_min, 0.18)
            * smooth_less_numba(coord, coord_max, 0.25)
        )

        phi_ic = (
            q_env
            * coord_good
            * gaussian_count_numba(S_env, 4.0, sigma_count)
            * gaussian_count_numba(E_env, 0.0, sigma_count)
        )

        phi_ih = (
            q_env
            * coord_good
            * gaussian_count_numba(S_env, 3.0, sigma_count)
            * gaussian_count_numba(E_env, 1.0, sigma_count)
        )

        phi_hyd = (
            q_env
            * coord_good
            * np.exp(-((E_env - hyd_crys_e_center) ** 2) / (2.0 * hyd_crys_e_width ** 2))
            * smooth_less_numba(S_env, hyd_crys_s_max, 0.18)
            * smooth_step_numba(q_env, hyd_crys_q_min, 0.08)
        )

        phi_pre = (
            q_env
            * coord_good
            * smooth_step_numba(q_env, hyd_pre_q_min, 0.08)
            * smooth_step_numba(E_env, hyd_pre_e_low, 0.18)
            * smooth_less_numba(E_env, hyd_pre_e_high, 0.20)
            * smooth_less_numba(S_env, hyd_pre_s_max, 0.22)
        )

        raw_hyd[atom_index] = phi_hyd
        raw_pre[atom_index] = phi_pre
        raw_ice[atom_index] = phi_ic + phi_ih

        ordered_sum = phi_ic + phi_ih + phi_hyd + phi_pre
        phi_liq = 1.0 / (1.0 + 8.0 * ordered_sum)
        total = phi_liq + phi_ih + phi_ic + phi_hyd + phi_pre + eps

        P_liq[atom_index] = phi_liq / total
        P_Ih[atom_index] = phi_ih / total
        P_Ic[atom_index] = phi_ic / total
        P_hyd[atom_index] = phi_hyd / total
        P_pre[atom_index] = phi_pre / total

    hyd_support = np.zeros(n_atoms, dtype=np.float64)
    pre_support = np.zeros(n_atoms, dtype=np.float64)
    ice_support = np.zeros(n_atoms, dtype=np.float64)

    for atom_index in range(n_atoms):
        start = indptr[atom_index]
        end = indptr[atom_index + 1]
        count = end - start

        if count == 0:
            continue

        hyd_sum = 0.0
        pre_sum = 0.0
        ice_sum = 0.0

        for ptr in range(start, end):
            neighbor = indices[ptr]
            hyd_sum += raw_hyd[neighbor]
            pre_sum += raw_pre[neighbor]
            ice_sum += raw_ice[neighbor]

        hyd_support[atom_index] = hyd_sum / count
        pre_support[atom_index] = pre_sum / count
        ice_support[atom_index] = ice_sum / count

    pre_neighbor_fraction = neighbor_fraction_from_scores_numba(
        raw_pre,
        hyd_pre_raw_min,
        indptr,
        indices,
    )

    for atom_index in range(n_atoms):
        q = qtet[atom_index]
        S = staggered_count[atom_index]
        E = eclipsed_count[atom_index]
        coord = coordination[atom_index]

        liquid_like = (
            P_liq[atom_index] > 0.35
            and raw_pre[atom_index] < 0.25
            and raw_hyd[atom_index] < 0.10
        )

        hyd_confident = (
            P_hyd[atom_index] > 0.30
            and raw_hyd[atom_index] >= hyd_crys_raw_min
            and E >= 3.25
            and S <= hyd_crys_s_max
            and q >= hyd_crys_q_min
            and coord >= coord_min
            and coord <= coord_max
            and hyd_support[atom_index] >= hyd_crys_support_min
        )

        ice_ic_confident = (
            P_Ic[atom_index] > 0.35
            and q >= ice_q_min
            and S >= 3.35
            and E <= 0.90
            and coord >= coord_min
            and coord <= coord_max
            and ice_support[atom_index] >= 0.12
        )

        ice_ih_confident = (
            P_Ih[atom_index] > 0.35
            and q >= ice_q_min
            and S >= 2.45
            and E <= 1.80
            and coord >= coord_min
            and coord <= coord_max
            and ice_support[atom_index] >= 0.12
        )

        pre_confident = (
            P_pre[atom_index] > 0.38
            and raw_pre[atom_index] >= hyd_pre_raw_min
            and q >= hyd_pre_q_min
            and E >= hyd_pre_e_low
            and E <= hyd_pre_e_high
            and S <= hyd_pre_s_max
            and coord >= coord_min
            and coord <= coord_max
            and pre_support[atom_index] >= hyd_pre_support_min
            and pre_neighbor_fraction[atom_index] >= hyd_pre_neighbor_fraction_min
            and not liquid_like
        )

        if hyd_confident:
            labels[atom_index] = LABEL_HYD
        elif ice_ic_confident:
            labels[atom_index] = LABEL_IC
        elif ice_ih_confident:
            labels[atom_index] = LABEL_IH
        elif pre_confident:
            labels[atom_index] = LABEL_PRE
        elif P_liq[atom_index] > 0.40:
            labels[atom_index] = LABEL_LIQ
        else:
            labels[atom_index] = LABEL_UNK

    return (
        P_liq,
        P_Ih,
        P_Ic,
        P_hyd,
        P_pre,
        labels,
        raw_hyd,
        raw_pre,
        hyd_support,
        pre_support,
        ice_support,
    )


def calc_q3m_all_vectorized(
    nearest_idx: np.ndarray,
    nearest_vec: np.ndarray,
    eps: float,
):
    n_atoms = nearest_idx.shape[0]
    qlm = np.zeros((n_atoms, 7), dtype=np.complex128)

    valid = np.all(nearest_idx >= 0, axis=1)

    if not np.any(valid):
        return qlm, valid

    vectors = nearest_vec[valid].reshape(-1, 3)
    norms = np.linalg.norm(vectors, axis=1)

    good = norms > eps
    unit = np.zeros_like(vectors)
    unit[good] = vectors[good] / norms[good, None]

    theta = np.arccos(np.clip(unit[:, 2], -1.0, 1.0))
    phi = np.arctan2(unit[:, 1], unit[:, 0])

    qlm_flat = np.zeros((vectors.shape[0], 7), dtype=np.complex128)
    l_value = 3

    for col, m_value in enumerate(range(-l_value, l_value + 1)):
        qlm_flat[:, col] = spherical_harmonic_l3(m_value, theta, phi)

    qlm_sum = qlm_flat.reshape(-1, 4, 7).sum(axis=1) / 4.0
    valid_indices = np.where(valid)[0]
    qlm[valid_indices] = qlm_sum

    qnorm = np.sqrt(np.sum(np.abs(qlm) ** 2, axis=1))
    valid = valid & (qnorm > eps)

    return qlm, valid


def spherical_harmonic_l3(
    m_value: int,
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    if USE_SCIPY_SPH_HARM:
        if USE_SPH_HARM_Y:
            return sph_harm_y(3, m_value, theta, phi)
        return sph_harm(m_value, 3, phi, theta)

    if m_value < 0:
        positive = spherical_harmonic_l3(-m_value, theta, phi)
        return ((-1) ** (-m_value)) * np.conjugate(positive)

    x = np.cos(theta)
    one_minus_x2 = np.maximum(0.0, 1.0 - x * x)

    if m_value == 0:
        associated = 0.5 * (5.0 * x**3 - 3.0 * x)
    elif m_value == 1:
        associated = -1.5 * (5.0 * x * x - 1.0) * np.sqrt(one_minus_x2)
    elif m_value == 2:
        associated = 15.0 * x * one_minus_x2
    elif m_value == 3:
        associated = -15.0 * one_minus_x2 ** 1.5
    else:
        raise ValueError("l=3 spherical harmonics require -3 <= m <= 3")

    import math

    normalization = math.sqrt(
        7.0
        / (4.0 * math.pi)
        * math.factorial(3 - m_value)
        / math.factorial(3 + m_value)
    )
    return normalization * associated * np.exp(1j * m_value * phi)


def calc_pair_correlations_vectorized(
    q3m: np.ndarray,
    q3m_valid: np.ndarray,
    pairs: np.ndarray,
    eps: float,
):
    atom_i = pairs[:, 0]
    atom_j = pairs[:, 1]

    valid = q3m_valid[atom_i] & q3m_valid[atom_j]
    correlations = np.full(len(pairs), np.nan, dtype=np.float64)

    if not np.any(valid):
        return correlations

    qi = q3m[atom_i[valid]]
    qj = q3m[atom_j[valid]]

    numerator = np.sum(qi * np.conjugate(qj), axis=1)
    denominator = np.sqrt(
        np.sum(np.abs(qi) ** 2, axis=1)
        * np.sum(np.abs(qj) ** 2, axis=1)
    )

    good = denominator > eps
    values = np.zeros(np.count_nonzero(valid), dtype=np.float64)
    values[good] = np.real(numerator[good] / denominator[good])

    correlations[valid] = values
    return correlations


def calculate_local_descriptors(
    graph: NeighborGraph,
    params: HopParameters,
    use_numba: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nearest4 = build_nearest_neighbors(graph, k_neighbors=4, use_numba=use_numba)

    qtet = call_kernel(calc_qtet_all_numba, use_numba,
        nearest4.indices,
        nearest4.vectors,
        params.eps,
    )

    q3m, q3m_valid = calc_q3m_all_vectorized(
        nearest4.indices,
        nearest4.vectors,
        params.eps,
    )

    correlations = calc_pair_correlations_vectorized(
        q3m,
        q3m_valid,
        graph.pairs,
        params.eps,
    )

    finite = np.isfinite(correlations)
    c_safe = np.zeros_like(correlations)
    c_safe[finite] = correlations[finite]

    weight_r = 1.0 / (
        1.0 + np.exp((graph.distances - params.oo_cutoff) / params.dist_smooth)
    )

    staggered_pair = np.zeros_like(correlations)
    eclipsed_pair = np.zeros_like(correlations)

    staggered_pair[finite] = (
        1.0 / (1.0 + np.exp((c_safe[finite] + 0.8) / params.delta_s))
    ) * weight_r[finite]

    eclipsed_pair[finite] = (
        np.exp(-((c_safe[finite] - params.c_e_center) ** 2) / (2.0 * params.sigma_ce ** 2))
    ) * weight_r[finite]

    S = call_kernel(accumulate_pair_values_numba, use_numba,
        graph.n_atoms,
        graph.pairs,
        staggered_pair,
    )

    E = call_kernel(accumulate_pair_values_numba, use_numba,
        graph.n_atoms,
        graph.pairs,
        eclipsed_pair,
    )

    coordination = call_kernel(calc_coordination_numba, use_numba,
        graph.n_atoms,
        graph.pairs,
        graph.distances,
        params.oo_cutoff,
        params.dist_smooth,
    )

    return qtet, S, E, coordination


def calculate_phase_probabilities(
    qtet: np.ndarray,
    S: np.ndarray,
    E: np.ndarray,
    coordination: np.ndarray,
    graph: NeighborGraph,
    params: HopParameters,
    use_numba: bool = True,
):
    qtet_nb = call_kernel(neighbor_average_numba, use_numba, qtet, graph.indptr, graph.indices)
    S_nb = call_kernel(neighbor_average_numba, use_numba, S, graph.indptr, graph.indices)
    E_nb = call_kernel(neighbor_average_numba, use_numba, E, graph.indptr, graph.indices)

    return call_kernel(calculate_phase_probabilities_numba, use_numba,
        qtet,
        S,
        E,
        coordination,
        qtet_nb,
        S_nb,
        E_nb,
        graph.indptr,
        graph.indices,
        params.hyd_crys_e_center,
        params.hyd_crys_e_width,
        params.hyd_crys_s_max,
        params.hyd_crys_q_min,
        params.hyd_crys_raw_min,
        params.hyd_crys_support_min,
        params.hyd_pre_e_low,
        params.hyd_pre_e_high,
        params.hyd_pre_s_max,
        params.hyd_pre_q_min,
        params.hyd_pre_raw_min,
        params.hyd_pre_support_min,
        params.hyd_pre_neighbor_fraction_min,
        params.ice_q_min,
        params.coord_min,
        params.coord_max,
        params.sigma_count,
        params.eps,
    )


def calculate_hop_stats(
    labels: np.ndarray,
    probabilities: PhaseProbabilities,
    graph: NeighborGraph,
    params: HopParameters,
    use_numba: bool = True,
) -> HopStats:
    P_hyd_total = probabilities.P_hyd_total
    P_ice_total = probabilities.P_ice_total

    H_total = np.mean((labels == LABEL_HYD) | (labels == LABEL_PRE))
    H_crys = np.mean(labels == LABEL_HYD)
    H_pre = np.mean(labels == LABEL_PRE)

    Hyd_order = np.mean(P_hyd_total)
    Ice = np.mean(P_ice_total)
    O_total = np.mean(P_hyd_total + P_ice_total)
    Delta = Hyd_order - Ice

    hyd_mask = P_hyd_total > params.pc_hyd
    ice_mask = P_ice_total > params.pc_ice

    Nmax_hyd = call_kernel(largest_cluster_size_csr_numba, use_numba,
        hyd_mask,
        graph.indptr,
        graph.indices,
        P_hyd_total.astype(np.float64, copy=False),
    )

    Nmax_ice = call_kernel(largest_cluster_size_csr_numba, use_numba,
        ice_mask,
        graph.indptr,
        graph.indices,
        P_ice_total.astype(np.float64, copy=False),
    )

    return HopStats(
        H_total=float(H_total),
        H_crys=float(H_crys),
        H_pre=float(H_pre),
        Hyd_order=float(Hyd_order),
        Ice=float(Ice),
        O_total=float(O_total),
        Delta=float(Delta),
        Nmax_hyd=float(Nmax_hyd),
        Nmax_ice=float(Nmax_ice),
    )


def analyze_hop_from_graph(
    graph: NeighborGraph,
    params: HopParameters,
    use_numba: bool = True,
) -> HopResult:
    qtet, S, E, coordination = calculate_local_descriptors(
        graph,
        params,
        use_numba=use_numba,
    )

    (
        P_liq,
        P_Ih,
        P_Ic,
        P_hyd,
        P_pre,
        labels,
        raw_hyd,
        raw_pre,
        hyd_support,
        pre_support,
        ice_support,
    ) = calculate_phase_probabilities(
        qtet,
        S,
        E,
        coordination,
        graph,
        params,
        use_numba=use_numba,
    )

    probabilities = PhaseProbabilities(
        P_liq=P_liq,
        P_Ih=P_Ih,
        P_Ic=P_Ic,
        P_hyd=P_hyd,
        P_pre=P_pre,
    )

    descriptors = LocalDescriptors(
        qtet=qtet,
        S=S,
        E=E,
        coordination=coordination,
        raw_hyd=raw_hyd,
        raw_pre=raw_pre,
        hyd_support=hyd_support,
        pre_support=pre_support,
        ice_support=ice_support,
    )

    stats = calculate_hop_stats(
        labels=labels,
        probabilities=probabilities,
        graph=graph,
        params=params,
        use_numba=use_numba,
    )

    return HopResult(
        stats=stats,
        labels=labels.astype(np.int32, copy=False),
        probabilities=probabilities,
        descriptors=descriptors,
    )


def build_cage_candidate_mask(
    labels: np.ndarray,
    probabilities: PhaseProbabilities,
    exclude_liquid: bool = True,
    include_ice: bool = False,
    liquid_exclude_prob: float = 0.70,
    hyd_candidate_prob: float = 0.25,
    ice_candidate_prob: float = 0.50,
) -> np.ndarray:
    hyd_like = probabilities.P_hyd_total >= hyd_candidate_prob
    uncertain = labels == LABEL_UNK

    mask = (
        hyd_like
        | uncertain
        | (labels == LABEL_HYD)
        | (labels == LABEL_PRE)
    )

    if include_ice:
        ice_like = probabilities.P_ice_total >= ice_candidate_prob
        mask |= ice_like | (labels == LABEL_IH) | (labels == LABEL_IC)

    if exclude_liquid:
        confident_liquid = (
            (labels == LABEL_LIQ)
            & (probabilities.P_liq >= liquid_exclude_prob)
        )
        mask &= ~confident_liquid

    return mask.astype(np.bool_, copy=False)










def call_kernel(function, use_numba: bool, *args):
    if not use_numba and hasattr(function, "py_func"):
        return function.py_func(*args)
    try:
        return function(*args)
    except Exception:
        if hasattr(function, "py_func"):
            return function.py_func(*args)
        raise
