from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
from MDAnalysis.lib.distances import minimize_vectors

try:
    from numba import njit
except ImportError:
    def njit(*args, **kwargs):
        def wrapper(func):
            return func
        return wrapper


_RIGHT_ANGLES = np.asarray([90.0, 90.0, 90.0], dtype=np.float64)


def pbc_delta(vec: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    根据最小镜像约定（Minimum Image Convention）计算向量在 PBC 下的位移。
    
    Parameters
    ----------
    vec : np.ndarray
        输入的三维相对位移向量，形状为 (3,) 或 (N, 3)
    box : np.ndarray
        系统盒子参数 [a, b, c, alpha, beta, gamma]
        
    Returns
    -------
    np.ndarray
        最小镜像处理后的位移向量
    """
    vec = np.asarray(vec, dtype=np.float64)
    original_shape = vec.shape
    vec = vec.reshape(-1, 3)

    dimensions = np.asarray(box, dtype=np.float64)
    orthogonal = dimensions.size < 6 or bool(
        np.array_equal(dimensions[3:6], _RIGHT_ANGLES)
    )

    if orthogonal:
        # MDAnalysis minimize_vectors re-validates the box on every call, which
        # dominated the per-guest ring scoring loops.  For an orthogonal cell
        # the minimum image is simply the component-wise round-off.
        lengths = dimensions[:3]

        if np.all(lengths > 0.0):
            return (vec - lengths * np.rint(vec / lengths)).reshape(original_shape)

    out = minimize_vectors(vec, box)
    return out.reshape(original_shape)


def wrap_point(point: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    将坐标折叠（wrap）回初级模拟盒子 [0, L] 内。
    
    Parameters
    ----------
    point : np.ndarray
        输入坐标，形状为 (3,) 或 (N, 3)
    box : np.ndarray
        系统盒子参数
        
    Returns
    -------
    np.ndarray
        折叠后的坐标
    """
    point = np.asarray(point, dtype=np.float64)
    dimensions = np.asarray(box, dtype=np.float64)
    lengths = dimensions[:3]

    if dimensions.size < 6 or np.allclose(dimensions[3:6], 90.0):
        return point - lengths * np.floor(point / lengths)

    a, b, c = lengths
    alpha, beta, gamma = np.deg2rad(dimensions[3:6])
    sin_gamma = np.sin(gamma)
    if abs(sin_gamma) < 1.0e-12:
        return point - lengths * np.floor(point / lengths)

    cell = np.asarray(
        [
            [a, 0.0, 0.0],
            [b * np.cos(gamma), b * sin_gamma, 0.0],
            [
                c * np.cos(beta),
                c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma,
                0.0,
            ],
        ],
        dtype=np.float64,
    )
    z_sq = c * c - cell[2, 0] ** 2 - cell[2, 1] ** 2
    cell[2, 2] = np.sqrt(max(0.0, z_sq))

    try:
        fractional = point @ np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        return point - lengths * np.floor(point / lengths)

    fractional -= np.floor(fractional)
    return fractional @ cell


def unwrap_points_around_center(
    points: np.ndarray,
    center: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """
    以指定中心点为基准，对一组点进行 PBC 展开（unwrap）。
    常用于笼型可视化时，消除笼子跨越周期性边界造成的断键。
    
    Parameters
    ----------
    points : np.ndarray
        需要展开的原子坐标，形状为 (N, 3)
    center : np.ndarray
        参考中心点坐标，形状为 (3,)
    box : np.ndarray
        系统盒子参数
        
    Returns
    -------
    np.ndarray
        展开后的绝对坐标
    """
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    return center + pbc_delta(points - center, box)


def unwrap_ring_coords(
    nodes: tuple[int, ...] | list[int] | np.ndarray,
    oxygen_pos: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """
    以环的第一个节点为基准，展开整个环的原子坐标。
    通过迭代质心进一步消除由于环跨边界引起的扭曲。
    
    Parameters
    ----------
    nodes : tuple or list of int
        环中原子的全局索引
    oxygen_pos : np.ndarray
        所有水氧原子的位置矩阵，形状为 (M, 3)
    box : np.ndarray
        系统盒子参数
        
    Returns
    -------
    np.ndarray
        展开后的环坐标，形状为 (len(nodes), 3)
    """
    node_array = np.asarray(nodes, dtype=np.int64)
    coords = oxygen_pos[node_array].astype(np.float64, copy=True)

    # 以第一个原子为参考点进行初步展开
    ref = coords[0]
    unwrapped = ref + pbc_delta(coords - ref, box)

    # 迭代计算质心进行二次展开，确保极度平滑
    for _ in range(2):
        center = unwrapped.mean(axis=0)
        unwrapped = center + pbc_delta(unwrapped - center, box)

    return unwrapped


def fit_ring(
    global_nodes: tuple[int, ...] | list[int] | np.ndarray,
    oxygen_pos: np.ndarray,
    box: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    对展开后的环坐标进行平面拟合（通过 SVD 分解）。
    计算环的几何中心、法向量以及平整度。
    
    Parameters
    ----------
    global_nodes : tuple or list of int
        环的全局原子索引
    oxygen_pos : np.ndarray
        氧原子位置
    box : np.ndarray
        系统盒子参数
        
    Returns
    -------
    Optional[Tuple[np.ndarray, np.ndarray, float]]
        (wrapped_center, normal, planarity)
        若 SVD 求解失败或法向量退化则返回 None。
        planarity 定义为原子到拟合平面最大垂直距离。
    """
    coords = unwrap_ring_coords(global_nodes, oxygen_pos, box)
    center = coords.mean(axis=0)
    centered = coords - center

    try:
        # 使用奇异值分解（SVD）确定最小方差方向（即法向量方向）
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    normal = vh[-1]
    norm = np.linalg.norm(normal)

    if norm < 1.0e-10:
        return None

    normal = normal / norm
    # 计算每个原子到拟合平面的距离，取最大值作为非平整度度量
    planarity = float(np.max(np.abs(centered @ normal)))
    
    # 拟合平面中心折叠回主要盒子
    center_wrapped = wrap_point(center, box)

    return center_wrapped, normal, planarity


def unwrap_cage_positions(
    cage_nodes: tuple[int, ...] | list[int] | np.ndarray,
    cage_center: tuple[float, float, float] | np.ndarray,
    oxygen_pos: np.ndarray,
    box: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    以笼子中心（一般是 guest 分子中心）为参考，展开笼子上所有原子的坐标。
    主要用于最终 PDB 文件输出，确保可视化时笼子呈完整的几何形状。
    
    Parameters
    ----------
    cage_nodes : tuple of int
        笼中所有水氧原子的索引
    cage_center : tuple or np.ndarray
        笼的几何中心或 Guest 分子中心
    oxygen_pos : np.ndarray
        氧原子位置
    box : np.ndarray
        系统盒子参数
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (wrapped_center, unwrapped_positions)
    """
    center = wrap_point(np.asarray(cage_center, dtype=np.float64), box)
    raw_coords = oxygen_pos[np.asarray(cage_nodes, dtype=np.int64)]
    unwrapped = unwrap_points_around_center(raw_coords, center, box)
    return center, unwrapped
