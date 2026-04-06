"""
模拟 UAV 路径新视角相机生成。

参照论文 2512.07527v2 §3.2 Iterative Texture Refinement：
  - 在 Mesh AABB 外扩 100m 的范围内，以 150m 均匀网格采样相机水平位置
  - 相机高度：450m（相对 Mesh 最高点）
  - 俯仰角：45°（向下看）
  - 朝向：东/西/南/北四个基本方向
  - 渲染分辨率：1024×1024（与 FLUX-Schnell 最优输入尺寸一致）

坐标系与 cameras/*.json 一致（本地坐标，单位：米）。
"""

import numpy as np
from typing import List


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = None) -> np.ndarray:
    """
    构建 look-at W2C 矩阵（4×4）。
    相机 Z 轴指向场景（从 eye 指向 target）。

    Returns
    -------
    W2C : (4,4)
    """
    if up is None:
        up = np.array([0.0, 0.0, 1.0])

    z_axis = target - eye
    z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-12)

    x_axis = np.cross(z_axis, up)
    norm = np.linalg.norm(x_axis)
    if norm < 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        x_axis = np.cross(z_axis, up)
        norm = np.linalg.norm(x_axis)
    x_axis = x_axis / (norm + 1e-12)

    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-12)

    # R: 旋转矩阵（行为相机轴）
    R = np.stack([x_axis, y_axis, z_axis], axis=0)  # (3,3)
    t = -R @ eye  # (3,)

    W2C = np.eye(4, dtype=np.float64)
    W2C[:3, :3] = R
    W2C[:3, 3] = t
    return W2C


def _build_pinhole_K(
    fov_deg: float,
    img_w: int,
    img_h: int,
) -> np.ndarray:
    """
    根据水平 FOV 和图像尺寸构建内参矩阵。

    论文中合成训练数据的 FOV 由高度和 GSD 推导：
        θ = 2 * arctan(W * GSD / (2 * H))
    对于 UAV 450m 高度，假设合理 GSD，使用固定 FOV = 60°。

    Returns
    -------
    K : (3,3)
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = (img_w / 2.0) / np.tan(fov_rad / 2.0)
    fy = (img_h / 2.0) / np.tan(
        2.0 * np.arctan(img_h / img_w * np.tan(fov_rad / 2.0)) / 2.0
    )
    cx = img_w / 2.0
    cy = img_h / 2.0
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )


def generate_novel_cameras(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    z_max: float,
    uav_height: float = 450.0,
    pitch_deg: float = 45.0,
    grid_spacing: float = 150.0,
    aabb_margin: float = 100.0,
    img_w: int = 1024,
    img_h: int = 1024,
    fov_deg: float = 60.0,
) -> List[dict]:
    """
    在 Mesh AABB 外扩 aabb_margin 范围内，均匀采样 UAV 相机。

    Parameters
    ----------
    x_min, x_max, y_min, y_max : Mesh 水平范围（本地坐标，米）
    z_max : Mesh 最高点（本地坐标，米）
    uav_height    : 相机高度（相对 z_max，米）
    pitch_deg     : 俯仰角（正值=向下倾斜，度）
    grid_spacing  : 相机水平网格间距（米）
    aabb_margin   : AABB 外扩边距（米）
    img_w, img_h  : 渲染分辨率
    fov_deg       : 水平视场角（度）

    Returns
    -------
    list of dict，每个 dict 包含：
        K     : (3,3) 内参
        W2C   : (4,4) 世界→相机
        img_w : int
        img_h : int
        name  : str 相机名称
    """
    x0 = x_min - aabb_margin
    x1 = x_max + aabb_margin
    y0 = y_min - aabb_margin
    y1 = y_max + aabb_margin

    cam_z = z_max + uav_height  # 相机绝对高度

    # 均匀网格采样
    xs = np.arange(x0, x1 + grid_spacing, grid_spacing)
    ys = np.arange(y0, y1 + grid_spacing, grid_spacing)

    # 四个基本朝向：东(+X)、西(-X)、南(-Y)、北(+Y)
    directions = {
        "E": np.array([1.0, 0.0, 0.0]),
        "W": np.array([-1.0, 0.0, 0.0]),
        "S": np.array([0.0, -1.0, 0.0]),
        "N": np.array([0.0, 1.0, 0.0]),
    }

    # 场景中心（水平）
    scene_cx = (x_min + x_max) / 2.0
    scene_cy = (y_min + y_max) / 2.0
    scene_cz = z_max  # 俯视目标点在场景最高面

    K = _build_pinhole_K(fov_deg, img_w, img_h)
    cameras = []

    for xi, x in enumerate(xs):
        for yi, y in enumerate(ys):
            for dir_name, forward in directions.items():
                # 45° 俯仰：相机水平偏移 = height * tan(pitch)
                horizontal_offset = uav_height * np.tan(np.deg2rad(pitch_deg))

                # 相机位置：在 forward 方向偏移
                eye = np.array([
                    x - forward[0] * horizontal_offset,
                    y - forward[1] * horizontal_offset,
                    cam_z,
                ], dtype=np.float64)

                # 目标点：沿 forward 方向的地面点
                target = np.array([
                    x + forward[0] * horizontal_offset * 0.5,
                    y + forward[1] * horizontal_offset * 0.5,
                    scene_cz,
                ], dtype=np.float64)

                W2C = _look_at(eye, target)

                name = f"uav_x{xi:03d}_y{yi:03d}_{dir_name}"
                cameras.append({
                    "name": name,
                    "K": K.copy(),
                    "W2C": W2C,
                    "img_w": img_w,
                    "img_h": img_h,
                    "image_path": None,
                })

    return cameras


def subsample_novel_cameras(
    cameras: List[dict],
    max_count: int = 64,
    seed: int = 42,
) -> List[dict]:
    """随机下采样新视角相机，避免内存溢出。"""
    if len(cameras) <= max_count:
        return cameras
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(cameras), max_count, replace=False)
    return [cameras[i] for i in sorted(indices)]
