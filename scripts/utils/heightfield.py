"""
将 NeuS 重建的全 3D Mesh 转换为含墙面的流形 Mesh。

管线：
  1. 对规则 XY 网格沿 -Z 方向射线采样，得到高度场 z_grid
  2. 将高度场体素化（在 z_grid 以下的体素标记为 solid）
  3. 用 marching cubes 提取流形表面（天然包含屋顶+墙面+底面）
  4. 用 xatlas 自动 UV 展开（各向同性，无投影拉伸）
"""

import os

import numpy as np
import trimesh
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import NearestNDInterpolator


def load_mesh_info(mesh_info_path: str) -> dict:
    """解析 mesh_info.txt，返回 center, range_, val_range。"""
    info = {}
    with open(mesh_info_path, "r") as f:
        content = f.read()

    for line in content.strip().splitlines():
        line = line.strip()
        if line.startswith("center:"):
            vals = line.split("[")[1].split("]")[0].split()
            info["center"] = np.array([float(v) for v in vals])
        elif line.startswith("range:"):
            info["range"] = float(line.split(":")[1].strip())
        elif line.startswith("val_range:"):
            # val_range 跨两行，提取所有浮点数
            rest = content[content.find("val_range:") :]
            nums = []
            for token in rest.replace("[", " ").replace("]", " ").split():
                try:
                    nums.append(float(token))
                    if len(nums) == 6:
                        break
                except ValueError:
                    pass
            info["val_range"] = np.array([nums[:3], nums[3:]])
    return info


def mesh_to_heightfield(
    mesh: trimesh.Trimesh,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    resolution: int = 512,
) -> tuple:
    """
    射线采样，生成高度场。

    Returns
    -------
    z_grid : np.ndarray (resolution, resolution)
        每个 XY 网格位置的最高 Z 值（本地坐标，单位：米）。
    hit_mask : np.ndarray (resolution, resolution) bool
        True 表示该位置有射线命中。
    """
    xs = np.linspace(x_min, x_max, resolution)
    ys = np.linspace(y_min, y_max, resolution)
    xx, yy = np.meshgrid(xs, ys)  # (H, W), 行=y，列=x

    n = resolution * resolution
    ray_origins = np.stack(
        [xx.ravel(), yy.ravel(), np.full(n, z_max + 10.0)], axis=1
    ).astype(np.float64)
    ray_directions = np.tile(np.array([0.0, 0.0, -1.0]), (n, 1))

    locs, idx_ray, _ = mesh.ray.intersects_location(
        ray_origins, ray_directions, multiple_hits=True
    )

    z_grid = np.full((resolution, resolution), z_min - 1.0)
    hit_mask = np.zeros((resolution, resolution), dtype=bool)

    if len(locs) > 0:
        for loc, ri in zip(locs, idx_ray):
            iy, ix = divmod(int(ri), resolution)
            if loc[2] > z_grid[iy, ix]:
                z_grid[iy, ix] = loc[2]
                hit_mask[iy, ix] = True

    return z_grid, hit_mask, xx, yy


def fill_holes(z_grid: np.ndarray, hit_mask: np.ndarray) -> np.ndarray:
    """用最近邻插值填补射线未命中的空洞。"""
    if hit_mask.all():
        return z_grid

    known_ys, known_xs = np.where(hit_mask)
    known_vals = z_grid[hit_mask]

    if len(known_vals) == 0:
        return z_grid

    interp = NearestNDInterpolator(
        np.stack([known_xs, known_ys], axis=1), known_vals
    )
    all_ys, all_xs = np.mgrid[0 : z_grid.shape[0], 0 : z_grid.shape[1]]
    z_grid_filled = interp(
        np.stack([all_xs.ravel(), all_ys.ravel()], axis=1)
    ).reshape(z_grid.shape)
    # 保留已命中位置的原始值
    z_grid_filled[hit_mask] = z_grid[hit_mask]
    return z_grid_filled


# ---------------------------------------------------------------------------
# 体素化管线（替代原来的 grid_to_mesh）
# ---------------------------------------------------------------------------

def heightfield_to_voxels(
    z_grid: np.ndarray,
    z_min: float,
    z_max: float,
    voxel_z_res: int = 128,
) -> tuple:
    """
    将高度场转换为三维体素格。

    体素 solid[k, j, i] = True 表示该体素在地表以下（实心区域）。
    体素坐标约定：k=Z 轴（下→上），j=Y 轴，i=X 轴。

    Returns
    -------
    solid  : np.ndarray (D, H, W) bool
    dz     : float  每个体素在 Z 方向的实际高度（米）
    """
    H, W = z_grid.shape
    # 底部多一层空体素（z_lo < z_min），使 marching cubes 能封闭底面
    z_lo = z_min - (z_max - z_min) / voxel_z_res  # 比 z_min 低一个体素
    dz = (z_max - z_lo) / voxel_z_res
    k_vals = z_lo + np.arange(voxel_z_res) * dz    # (D,) 每层的 z 值

    # solid[k, j, i] = True 当 k_vals[k] <= z_grid[j, i]
    solid = k_vals[:, None, None] <= z_grid[None, :, :]   # (D, H, W) bool

    # 顶部追加一层全空体素：确保最高地形处也有实心→空的过渡，
    # 否则 marching cubes 在数组边界无法生成顶面三角形（产生空洞）
    empty_top = np.zeros((1, H, W), dtype=bool)
    solid = np.concatenate([solid, empty_top], axis=0)   # (D+1, H, W)

    return solid, dz, z_lo


def voxels_to_mesh(
    solid: np.ndarray,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_lo: float,
    dz: float,
) -> tuple:
    """
    用 marching cubes 从体素格提取流形表面。

    Returns
    -------
    vertices : np.ndarray (N, 3) float32，世界坐标（米）
    faces    : np.ndarray (M, 3) int32
    """
    from skimage.measure import marching_cubes

    D, H, W = solid.shape
    # marching cubes 输入为浮点型体积，level=0.5 作为 solid/空 边界
    verts_v, faces, _, _ = marching_cubes(
        solid.astype(np.float32), level=0.5, allow_degenerate=False
    )
    # verts_v: (N, 3)，坐标顺序为 (z_idx, y_idx, x_idx)（skimage 约定）
    verts_world = np.stack([
        x_min + verts_v[:, 2] / W * (x_max - x_min),
        y_min + verts_v[:, 1] / H * (y_max - y_min),
        z_lo  + verts_v[:, 0] * dz,
    ], axis=1).astype(np.float32)

    return verts_world, faces.astype(np.int32)


def simplify_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    target_faces: int,
) -> tuple:
    """
    使用 quadric error metrics 对 Mesh 进行面数简化。

    依赖 fast-simplification（trimesh 后端）。

    Parameters
    ----------
    target_faces : 目标面数；若大于当前面数则直接返回原始 Mesh

    Returns
    -------
    vertices : np.ndarray (N', 3) float32
    faces    : np.ndarray (M', 3) int32
    """
    if target_faces <= 0 or target_faces >= len(faces):
        return vertices, faces

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    simplified = mesh.simplify_quadric_decimation(face_count=target_faces, aggression=0)
    v = np.asarray(simplified.vertices, dtype=np.float32)
    f = np.asarray(simplified.faces,    dtype=np.int32)
    print(f"[heightfield] 网格简化: {len(faces)} → {len(f)} 面, "
          f"{len(vertices)} → {len(v)} 顶点")
    return v, f


# ---------------------------------------------------------------------------
# 单 Island UV（Tutte 调和映射）
# ---------------------------------------------------------------------------

def _order_boundary_loop(boundary_edges: np.ndarray) -> np.ndarray:
    """
    将无序的边界边集合排列成有序顶点回路。

    Parameters
    ----------
    boundary_edges : (B, 2) int，每行为一条只属于一个面的边

    Returns
    -------
    loop : (B,) int，有序顶点序列（首尾不重复）
    """
    from collections import defaultdict
    adj = defaultdict(list)
    for a, b in boundary_edges:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    start = int(boundary_edges[0, 0])
    loop  = [start]
    prev, cur = -1, start
    while True:
        neighbors = [v for v in adj[cur] if v != prev]
        if not neighbors or neighbors[0] == start:
            break
        prev, cur = cur, neighbors[0]
        loop.append(cur)
    return np.array(loop, dtype=np.int32)


def _t_to_square(t: np.ndarray) -> np.ndarray:
    """
    将参数 t ∈ [0, 1) 按周长比例映射到单位正方形边界上。

    正方形顺序：底边 → 右边 → 顶边 → 左边（逆时针）。

    Parameters
    ----------
    t : (B,) float，归一化参数，t[0]=0

    Returns
    -------
    uv : (B, 2) float32
    """
    uv = np.zeros((len(t), 2), dtype=np.float32)
    for k, tk in enumerate(t):
        s = tk * 4.0          # 映射到 [0, 4) 代表 4 段边
        if s < 1.0:           # 底边：(0,0) → (1,0)
            uv[k] = [s, 0.0]
        elif s < 2.0:         # 右边：(1,0) → (1,1)
            uv[k] = [1.0, s - 1.0]
        elif s < 3.0:         # 顶边：(1,1) → (0,1)
            uv[k] = [3.0 - s, 1.0]
        else:                 # 左边：(0,1) → (0,0)
            uv[k] = [0.0, 4.0 - s]
    return uv


def single_island_uv(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """
    Tutte 调和映射：将具有单边界回路（拓扑圆盘）的 Mesh 展开为严格单 island UV。

    算法步骤：
      1. 找到边界边（只属于一个面的边）
      2. 将边界回路按 3D 弧长映射到单位正方形边界
      3. 构建均匀 Laplacian（内部顶点 UV = 邻居均值）
      4. 求解稀疏线性方程组得到内部顶点 UV

    Tutte 定理保证：当边界为凸多边形时，内部不出现 UV 折叠。

    Parameters
    ----------
    vertices : (N, 3) float32，世界坐标
    faces    : (M, 3) int32

    Returns
    -------
    uv : (N, 2) float32，归一化 [0, 1]，**顶点数与输入完全一致（无复制）**
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    N = len(vertices)

    # ------------------------------------------------------------------
    # 1. 找到边界边（仅出现在 1 个面中的边）
    # ------------------------------------------------------------------
    edges_all    = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges_sorted = np.sort(edges_all, axis=1)
    unique_edges, counts = np.unique(edges_sorted, axis=0, return_counts=True)
    boundary_edges = unique_edges[counts == 1]   # (B, 2)

    if len(boundary_edges) == 0:
        raise ValueError(
            "Mesh 没有边界边（封闭流形），Tutte 映射需要拓扑圆盘（单边界回路）。"
            "请检查 marching cubes 体素格设置。"
        )

    # ------------------------------------------------------------------
    # 2. 将边界边排列成有序回路，并映射到单位正方形
    # ------------------------------------------------------------------
    boundary_loop = _order_boundary_loop(boundary_edges)   # (B,)
    bv_pos   = vertices[boundary_loop]                     # (B, 3)
    seg_diff = np.diff(bv_pos, axis=0, prepend=bv_pos[-1:])
    seg_len  = np.linalg.norm(seg_diff, axis=1)            # (B,)
    total_len = seg_len.sum()
    if total_len < 1e-12:
        raise ValueError("边界回路总长度接近零，无法进行 UV 展开。")

    t = np.cumsum(seg_len) / total_len   # (B,)  ∈ (0, 1]
    t = np.roll(t, 1)                    # 使 t[0] = 0
    t[0] = 0.0
    uv_boundary = _t_to_square(t)        # (B, 2)

    # ------------------------------------------------------------------
    # 3. 构建均匀 Laplacian 矩阵（向量化，避免 Python 循环）
    # ------------------------------------------------------------------
    # 每条边 (a,b) 在 L 中贡献：L[a,b] -= 1, L[b,a] -= 1, L[a,a] += 1, L[b,b] += 1
    a0, a1, a2 = faces[:, 0], faces[:, 1], faces[:, 2]
    # 所有有向边对（双向）
    src = np.concatenate([a0, a1, a1, a2, a2, a0])
    dst = np.concatenate([a1, a0, a2, a1, a0, a2])
    # 非对角项：-1
    data_off = np.full(len(src), -1.0)
    L = sp.csr_matrix((data_off, (src, dst)), shape=(N, N))
    # 对角项：每行 = 度数
    deg = np.asarray(-L.sum(axis=1)).ravel()
    L  += sp.diags(deg)

    # ------------------------------------------------------------------
    # 4. 分离内部/边界变量，求解线性方程组（u 和 v 同时求解）
    # ------------------------------------------------------------------
    is_boundary = np.zeros(N, dtype=bool)
    is_boundary[boundary_loop] = True
    interior_idx    = np.where(~is_boundary)[0]
    boundary_idx    = np.where( is_boundary)[0]

    uv_full = np.zeros((N, 2), dtype=np.float64)
    uv_full[boundary_loop] = uv_boundary.astype(np.float64)

    if len(interior_idx) > 0:
        L_ii = L[np.ix_(interior_idx, interior_idx)]
        L_ib = L[np.ix_(interior_idx, boundary_idx)]
        rhs  = -(L_ib @ uv_full[boundary_idx])     # (n_interior, 2)
        uv_full[interior_idx] = spla.spsolve(L_ii.tocsc(), rhs)

    uv_full = np.clip(uv_full, 0.0, 1.0)
    return uv_full.astype(np.float32)


# ---------------------------------------------------------------------------
# 保留原 grid_to_mesh（仅供外部调试使用，主管线不再调用）
# ---------------------------------------------------------------------------

def grid_to_mesh(
    xx: np.ndarray, yy: np.ndarray, z_grid: np.ndarray
) -> tuple:
    """
    将规则 XY 网格 + Z 高度场转换为三角网格（XY 投影 UV）。
    仅供调试/对比，主管线改用 voxel → marching cubes → xatlas。
    """
    H, W = z_grid.shape
    x_min, x_max = xx[0, 0], xx[0, -1]
    y_min, y_max = yy[0, 0], yy[-1, 0]

    vertices = np.stack(
        [xx.ravel(), yy.ravel(), z_grid.ravel()], axis=1
    ).astype(np.float32)

    u = (xx - x_min) / (x_max - x_min + 1e-12)
    v = (yy - y_min) / (y_max - y_min + 1e-12)
    uv = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float32)

    rows, cols = np.mgrid[0 : H - 1, 0 : W - 1]
    rows = rows.ravel()
    cols = cols.ravel()

    idx_tl = rows * W + cols
    idx_tr = rows * W + cols + 1
    idx_bl = (rows + 1) * W + cols
    idx_br = (rows + 1) * W + cols + 1

    tri_upper = np.stack([idx_tl, idx_tr, idx_bl], axis=1)
    tri_lower = np.stack([idx_tr, idx_br, idx_bl], axis=1)
    faces = np.concatenate([tri_upper, tri_lower], axis=0).astype(np.int32)

    return vertices, faces, uv


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def build_heightfield_mesh(
    mesh_ply_path: str,
    mesh_info_path: str,
    resolution: int = 512,
    voxel_z_res: int = 128,
    simplify_faces: int = 0,
    aligned_vertices: np.ndarray = None,
) -> dict:
    """
    完整管线：加载 NeuS Mesh → 射线采样高度场 → 体素化 → marching cubes
    → （可选）quadric 简化 → Tutte 调和映射单 Island UV

    Parameters
    ----------
    mesh_ply_path     : 输入 PLY 路径
    mesh_info_path    : mesh_info.txt 路径
    resolution        : 高度场 XY 采样分辨率（体素 X/Y 方向格数相同）
    voxel_z_res       : 体素 Z 方向层数，越大墙面越精细
    simplify_faces    : 目标面数，0 = 不简化
    aligned_vertices  : (V, 3) float32，若提供则替换 PLY 中的顶点（对齐后）

    Returns
    -------
    dict 包含：
        vertices  : (N, 3) float32，世界坐标（米）
        faces     : (M, 3) int32
        uv        : (N, 2) float32，归一化 [0,1]，严格单 island
        x_min/max, y_min/max, z_min/max : 场景范围（米）
        z_grid    : (H, W) float32，高度场（调试用）
        info      : mesh_info 字典
    """
    mesh_ply_path = os.path.abspath(mesh_ply_path)
    if not os.path.isfile(mesh_ply_path):
        raise FileNotFoundError(
            f"找不到 Mesh 文件: {mesh_ply_path}\n"
            f"请确认文件路径正确，当前工作目录: {os.getcwd()}"
        )
    print(f"[heightfield] 加载 mesh: {mesh_ply_path}")
    mesh = trimesh.load(mesh_ply_path, force="mesh", process=False)
    print(f"[heightfield] 顶点: {len(mesh.vertices)}, 面: {len(mesh.faces)}")

    if aligned_vertices is not None:
        if len(aligned_vertices) != len(mesh.vertices):
            raise ValueError(
                f"aligned_vertices 顶点数 ({len(aligned_vertices)}) "
                f"与 PLY ({len(mesh.vertices)}) 不一致"
            )
        mesh = trimesh.Trimesh(
            vertices=aligned_vertices.astype(np.float32),
            faces=mesh.faces,
            process=False,
        )
        print("[heightfield] 使用已对齐顶点")

    info = load_mesh_info(mesh_info_path)

    if aligned_vertices is not None:
        verts = np.asarray(mesh.vertices, dtype=np.float32)
        x_min = float(verts[:, 0].min())
        x_max = float(verts[:, 0].max())
        y_min = float(verts[:, 1].min())
        y_max = float(verts[:, 1].max())
        z_min = float(verts[:, 2].min())
        z_max = float(verts[:, 2].max())
        print("[heightfield] Bounds 由实际对齐顶点范围确定")
    else:
        range_ = info["range"]
        val_range = info["val_range"]
        x_min = float(val_range[0, 0] * range_)
        x_max = float(val_range[1, 0] * range_)
        y_min = float(val_range[0, 1] * range_)
        y_max = float(val_range[1, 1] * range_)
        z_min = float(val_range[0, 2] * range_)
        z_max = float(val_range[1, 2] * range_)

    print(f"[heightfield] 场景范围 X:[{x_min:.1f}, {x_max:.1f}] "
          f"Y:[{y_min:.1f}, {y_max:.1f}] Z:[{z_min:.1f}, {z_max:.1f}] 米")
    print(f"[heightfield] 射线采样 {resolution}×{resolution} ...")

    z_grid, hit_mask, xx, yy = mesh_to_heightfield(
        mesh, x_min, x_max, y_min, y_max, z_min, z_max, resolution
    )

    hit_rate = hit_mask.sum() / hit_mask.size
    print(f"[heightfield] 射线命中率: {hit_rate:.1%}")
    if hit_rate < 0.5:
        print("[heightfield] 警告：命中率过低，请检查坐标系设置")

    print("[heightfield] 填补空洞 ...")
    z_grid = fill_holes(z_grid, hit_mask)

    # ------------------------------------------------------------------
    # 体素化 → marching cubes → （可选）简化 → Tutte 单 Island UV
    # ------------------------------------------------------------------
    print(f"[heightfield] 体素化 (XY={resolution}, Z={voxel_z_res}) ...")
    solid, dz, z_lo = heightfield_to_voxels(z_grid, z_min, z_max, voxel_z_res)

    print("[heightfield] marching cubes 提取流形 Mesh ...")
    vertices, faces = voxels_to_mesh(solid, x_min, x_max, y_min, y_max, z_lo, dz)
    print(f"[heightfield] 流形 Mesh: {len(vertices)} 顶点, {len(faces)} 面")

    # 保留 marching cubes 原始结果（供外部保存/调试）
    mc_vertices = vertices.copy()
    mc_faces    = faces.copy()

    if simplify_faces > 0:
        print(f"[heightfield] quadric 网格简化 → 目标面数: {simplify_faces} ...")
        vertices, faces = simplify_mesh(vertices, faces, simplify_faces)

    print("[heightfield] Tutte 调和映射 → 单 Island UV ...")
    uv = single_island_uv(vertices, faces)
    print(f"[heightfield] UV 展开完成: {len(vertices)} 顶点（无缝合复制）")

    return {
        "vertices": vertices,
        "faces":    faces,
        "uv":       uv,
        "mc_vertices": mc_vertices,
        "mc_faces":    mc_faces,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "resolution": resolution,
        "z_grid": z_grid,
        "info":   info,
    }


def save_heightfield_mesh(hf: dict, output_path: str):
    """将流形 Mesh 保存为 PLY 文件。"""
    mesh = trimesh.Trimesh(
        vertices=hf["vertices"],
        faces=hf["faces"],
        process=False,
    )
    mesh.export(output_path)
    print(f"[heightfield] Mesh 已保存: {output_path}")
