"""
将 NeuS Mesh 平移对齐到 COLMAP 稀疏点云。

两阶段纯平移配准（移植自 SatDN/refine_mesh_nvdiffrast.py）：

  Stage 1 — 3D FFT 体素互相关（全局粗对齐）：
    两点云分别体素化为三维占据格，高斯平滑后做 FFT 互相关，
    峰值位置即为最佳粗平移量（参考 PHASER, Bernreiter et al., IEEE RA-L 2021）。
    优点：全局最优，不受局部极值影响。

  Stage 2 — 迭代 trimmed-median 最近邻精炼（亚米级）：
    对每个 COLMAP 点查找对齐后 Mesh 的最近顶点，丢弃最差 trim_fraction 的对，
    取剩余 delta 的逐元素中值作为增量修正，直至收敛。
"""

import numpy as np


def load_colmap_points3d(
    path: str,
    max_reproj_error: float = 2.0,
    min_track_len: int = 3,
) -> np.ndarray:
    """
    从 COLMAP points3D.txt 加载并过滤 3D 点。

    文件格式（每行）：
        POINT3D_ID  X  Y  Z  R  G  B  ERROR  TRACK[] (IMAGE_ID POINT2D_IDX ...)

    Parameters
    ----------
    path             : points3D.txt 路径
    max_reproj_error : 丢弃重投影误差超过此阈值（像素）的点，默认 2.0
    min_track_len    : 丢弃被观测图像数少于此值的点，默认 3

    Returns
    -------
    (N, 3) float64 点坐标数组
    """
    pts = []
    total = 0
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            total += 1
            error = float(parts[7])
            if error > max_reproj_error:
                continue
            track_len = (len(parts) - 8) // 2
            if track_len < min_track_len:
                continue
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])

    arr = np.array(pts, dtype=np.float64)
    print(
        f"[alignment] 加载 COLMAP 点: {len(arr)}/{total} 个有效点 "
        f"(max_reproj_error={max_reproj_error} px, min_track_len={min_track_len})"
    )
    return arr


def align_mesh_to_pointcloud(
    vertices: np.ndarray,
    colmap_pts: np.ndarray,
    trim_fraction: float = 0.3,
    n_iters: int = 20,
    fft_voxel_size: float = 2.0,
    fft_max_grid: int = 256,
    fft_smooth_sigma: float = 1.5,
) -> np.ndarray:
    """
    将 Mesh 顶点通过纯平移配准到 COLMAP 稀疏点云。

    Parameters
    ----------
    vertices         : (V, 3) float32/float64，Mesh 顶点（本地坐标，米）
    colmap_pts       : (N, 3) float64，COLMAP 过滤后的 3D 点
    trim_fraction    : 每轮丢弃的最差最近邻对比例，默认 0.3
    n_iters          : Stage 2 最大迭代次数，默认 20
    fft_voxel_size   : FFT 粗对齐体素尺寸（米），默认 2.0
    fft_max_grid     : FFT 网格每轴最大体素数（限制内存），默认 256
    fft_smooth_sigma : 体素化后高斯平滑 sigma（体素单位），默认 1.5

    Returns
    -------
    (V, 3) float32，对齐后的 Mesh 顶点
    """
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    verts_d = vertices.astype(np.float64)
    pts_d = colmap_pts.astype(np.float64)

    # ------------------------------------------------------------------
    # Stage 1: 3D FFT 互相关 → 粗平移
    # ------------------------------------------------------------------
    pad = fft_voxel_size * 10
    bbox_min = np.minimum(verts_d.min(0), pts_d.min(0)) - pad
    bbox_max = np.maximum(verts_d.max(0), pts_d.max(0)) + pad

    grid_shape = np.ceil((bbox_max - bbox_min) / fft_voxel_size).astype(int)
    grid_shape = np.minimum(grid_shape, fft_max_grid)
    print(f"[alignment] FFT 网格: {grid_shape}  体素尺寸={fft_voxel_size} m")

    def _voxelize(pts_in: np.ndarray) -> np.ndarray:
        idx = np.floor((pts_in - bbox_min) / fft_voxel_size).astype(int)
        valid = np.all((idx >= 0) & (idx < grid_shape), axis=1)
        idx = idx[valid]
        grid = np.zeros(grid_shape, dtype=np.float64)
        np.add.at(grid, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
        return grid

    mesh_grid = _voxelize(verts_d)
    colmap_grid = _voxelize(pts_d)

    if fft_smooth_sigma > 0:
        mesh_grid = gaussian_filter(mesh_grid, sigma=fft_smooth_sigma)
        colmap_grid = gaussian_filter(colmap_grid, sigma=fft_smooth_sigma)

    cross_corr = np.real(
        np.fft.ifftn(np.fft.fftn(colmap_grid) * np.conj(np.fft.fftn(mesh_grid)))
    )

    peak = np.array(
        np.unravel_index(np.argmax(cross_corr), grid_shape), dtype=np.float64
    )
    for i in range(3):
        if peak[i] > grid_shape[i] / 2:
            peak[i] -= grid_shape[i]

    t = peak * fft_voxel_size
    print(f"[alignment] FFT 粗平移: t={np.round(t, 3)}  ||t||={np.linalg.norm(t):.3f} m")

    # ------------------------------------------------------------------
    # Stage 2: 迭代 trimmed-median 精炼
    # ------------------------------------------------------------------
    for i in range(n_iters):
        tree = cKDTree(verts_d + t)
        dists, nn_idx = tree.query(pts_d, k=1, workers=-1)

        threshold = np.percentile(dists, (1.0 - trim_fraction) * 100.0)
        valid = dists <= threshold
        n_valid = int(valid.sum())
        if n_valid < 10:
            print(f"[alignment] iter {i+1}: 内点过少 ({n_valid})，提前停止")
            break

        delta = pts_d[valid] - (verts_d[nn_idx[valid]] + t)
        increment = np.median(delta, axis=0)
        t += increment

        if (i + 1) % 5 == 0 or np.linalg.norm(increment) < 1e-3:
            print(
                f"[alignment] iter {i+1:2d}/{n_iters}: "
                f"内点={n_valid}/{len(pts_d)}, "
                f"||increment||={np.linalg.norm(increment):.5f} m"
            )

        if np.linalg.norm(increment) < 1e-4:
            print("[alignment] 已收敛")
            break

    print(f"[alignment] 最终平移: t={np.round(t, 4)}  ||t||={np.linalg.norm(t):.4f} m")
    return np.ascontiguousarray((verts_d + t).astype(np.float32))
