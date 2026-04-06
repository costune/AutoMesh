"""
将 NeuS 重建的全 3D Mesh 转换为 2.5D 高度场 Mesh。

对每个规则 XY 网格位置沿 -Z 方向发射射线，取与原始 Mesh 的最高交点作为 Z 值，
生成规则网格三角化的高度场 Mesh（天然保证 2-流形）。
UV 坐标直接由 XY 归一化坐标确定，与俯视卫星图像天然对齐。
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


def grid_to_mesh(
    xx: np.ndarray, yy: np.ndarray, z_grid: np.ndarray
) -> tuple:
    """
    将规则 XY 网格 + Z 高度场转换为三角网格。

    Returns
    -------
    vertices : np.ndarray (H*W, 3)
    faces    : np.ndarray (2*(H-1)*(W-1), 3)
    uv       : np.ndarray (H*W, 2)  归一化 [0,1]
    """
    H, W = z_grid.shape
    x_min, x_max = xx[0, 0], xx[0, -1]
    y_min, y_max = yy[0, 0], yy[-1, 0]

    vertices = np.stack(
        [xx.ravel(), yy.ravel(), z_grid.ravel()], axis=1
    ).astype(np.float32)

    # 归一化 UV：u 对应 x，v 对应 y
    u = (xx - x_min) / (x_max - x_min + 1e-12)
    v = (yy - y_min) / (y_max - y_min + 1e-12)
    uv = np.stack([u.ravel(), v.ravel()], axis=1).astype(np.float32)

    # 规则网格三角化：每个格子拆成两个三角形
    rows, cols = np.mgrid[0 : H - 1, 0 : W - 1]
    rows = rows.ravel()
    cols = cols.ravel()

    idx_tl = rows * W + cols
    idx_tr = rows * W + cols + 1
    idx_bl = (rows + 1) * W + cols
    idx_br = (rows + 1) * W + cols + 1

    # 上三角：tl-tr-bl，下三角：tr-br-bl
    tri_upper = np.stack([idx_tl, idx_tr, idx_bl], axis=1)
    tri_lower = np.stack([idx_tr, idx_br, idx_bl], axis=1)
    faces = np.concatenate([tri_upper, tri_lower], axis=0).astype(np.int32)

    return vertices, faces, uv


def build_heightfield_mesh(
    mesh_ply_path: str,
    mesh_info_path: str,
    resolution: int = 512,
    aligned_vertices: np.ndarray = None,
) -> dict:
    """
    完整流程：加载 NeuS Mesh → （可选）替换为对齐顶点 → 射线采样 → 生成 2.5D 高度场 Mesh。

    Parameters
    ----------
    mesh_ply_path     : 输入 PLY 路径
    mesh_info_path    : mesh_info.txt 路径
    resolution        : 高度场网格分辨率
    aligned_vertices  : (V, 3) float32，若提供则替换 PLY 中的顶点（对齐后），
                        拓扑（faces）保持不变；None 表示直接使用原始顶点。

    Returns
    -------
    dict 包含：
        vertices  : (N, 3) float32，本地坐标（米）
        faces     : (M, 3) int32
        uv        : (N, 2) float32，归一化 [0,1]
        x_min/max, y_min/max, z_min/max : 场景范围（米）
        info      : mesh_info 字典
    """
    # 转换为绝对路径，避免 trimesh 因 cwd 不一致无法识别相对路径
    mesh_ply_path = os.path.abspath(mesh_ply_path)
    if not os.path.isfile(mesh_ply_path):
        raise FileNotFoundError(
            f"找不到 Mesh 文件: {mesh_ply_path}\n"
            f"请确认文件路径正确，当前工作目录: {os.getcwd()}"
        )
    print(f"[heightfield] 加载 mesh: {mesh_ply_path}")
    mesh = trimesh.load(mesh_ply_path, force="mesh", process=False)
    print(f"[heightfield] 顶点: {len(mesh.vertices)}, 面: {len(mesh.faces)}")

    # 若提供了对齐后的顶点，替换原始顶点（拓扑不变）
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
        # 对齐后用实际顶点范围重算 bounds，防止平移后超出 val_range 边界
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
        # val_range 是 Marching Cube 归一化坐标，mesh 已恢复真实尺度
        # 场景范围（本地坐标，单位：米）
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

    print("[heightfield] 三角化网格 ...")
    vertices, faces, uv = grid_to_mesh(xx, yy, z_grid)

    print(f"[heightfield] 高度场 Mesh: {len(vertices)} 顶点, {len(faces)} 面")

    return {
        "vertices": vertices,
        "faces": faces,
        "uv": uv,
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "z_min": z_min, "z_max": z_max,
        "resolution": resolution,
        "info": info,
    }


def save_heightfield_mesh(hf: dict, output_path: str):
    """将高度场 Mesh 保存为 PLY 文件（含 UV）。"""
    mesh = trimesh.Trimesh(
        vertices=hf["vertices"],
        faces=hf["faces"],
        process=False,
    )
    mesh.export(output_path)
    print(f"[heightfield] 高度场 Mesh 已保存: {output_path}")
