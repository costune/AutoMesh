"""
相机参数加载与投影工具。

支持两种格式：
  1. cameras/*.json  : K (16元素行主序4×4) + W2C (16元素行主序4×4) + img_size [W,H]
  2. transforms.json : NeRF C2W 格式 (fl_x, fl_y, cx, cy, transform_matrix)

坐标系约定：
  - Mesh 顶点为本地坐标（以 mesh center 为原点，单位：米）
  - cameras/*.json 的 K/W2C 已与该本地坐标系对齐，可直接使用
  - nvdiffrast 需要 clip-space 坐标（NDC [-1,1]）
"""

import json
import os
import glob
import numpy as np
import torch


def load_cameras_from_dir(cameras_dir: str, images_dir: str) -> list:
    """
    从 cameras/*.json 加载所有相机。

    Returns
    -------
    list of dict，每个 dict 包含：
        K        : np.ndarray (3,3)  内参矩阵
        W2C      : np.ndarray (4,4)  世界→相机变换
        img_size : (W, H)
        image_path : str 对应图像路径（若存在）
        name     : str 相机名称
    """
    cameras = []
    json_files = sorted(glob.glob(os.path.join(cameras_dir, "*.json")))

    for json_path in json_files:
        with open(json_path, "r") as f:
            data = json.load(f)

        K_flat = np.array(data["K"], dtype=np.float64).reshape(4, 4)
        W2C = np.array(data["W2C"], dtype=np.float64).reshape(4, 4)
        img_w, img_h = int(data["img_size"][0]), int(data["img_size"][1])

        # 取左上 3×3 作为内参
        K3 = K_flat[:3, :3].copy()

        # 根据 cameras/JAX_068_001_RGB.json 推断对应图像名
        basename = os.path.splitext(os.path.basename(json_path))[0]  # JAX_068_001_RGB
        image_path = None
        for ext in [".png", ".jpg", ".tif"]:
            candidate = os.path.join(images_dir, basename + ext)
            if os.path.exists(candidate):
                image_path = candidate
                break

        cameras.append(
            {
                "name": basename,
                "K": K3,
                "W2C": W2C,
                "img_w": img_w,
                "img_h": img_h,
                "image_path": image_path,
            }
        )

    return cameras


def load_cameras_from_transforms(
    transforms_path: str, images_base: str
) -> list:
    """
    从 transforms.json (NeRF 格式) 加载相机。
    transform_matrix 为 C2W (列主序 4×4)。
    """
    with open(transforms_path, "r") as f:
        data = json.load(f)

    cameras = []
    for frame in data["frames"]:
        C2W = np.array(frame["transform_matrix"], dtype=np.float64)
        W2C = np.linalg.inv(C2W)

        fl_x = float(frame["fl_x"])
        fl_y = float(frame["fl_y"])
        cx = float(frame["cx"])
        cy = float(frame["cy"])

        K3 = np.array(
            [[fl_x, 0.0, cx], [0.0, fl_y, cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        file_path = frame["file_path"]
        image_path = os.path.join(
            os.path.dirname(transforms_path), file_path.lstrip("./")
        )
        if not os.path.exists(image_path):
            image_path = None

        basename = os.path.splitext(os.path.basename(file_path))[0]
        cameras.append(
            {
                "name": basename,
                "K": K3,
                "W2C": W2C,
                "img_w": None,  # 从图像读取
                "img_h": None,
                "image_path": image_path,
            }
        )

    return cameras


def project_vertices(
    vertices: np.ndarray,
    K: np.ndarray,
    W2C: np.ndarray,
) -> tuple:
    """
    将 Mesh 本地坐标顶点投影到像素坐标。

    Parameters
    ----------
    vertices : (N, 3) 本地坐标（米）
    K        : (3, 3) 相机内参
    W2C      : (4, 4) 世界→相机

    Returns
    -------
    uv   : (N, 2) 像素坐标
    depth: (N,)   相机空间深度（Z_cam）
    """
    N = len(vertices)
    v_h = np.concatenate([vertices, np.ones((N, 1))], axis=1)  # (N,4)
    v_cam = (W2C @ v_h.T).T  # (N,4)
    z = v_cam[:, 2]

    valid = z > 0
    u = np.full(N, np.nan)
    v_px = np.full(N, np.nan)
    u[valid] = K[0, 0] * v_cam[valid, 0] / z[valid] + K[0, 2]
    v_px[valid] = K[1, 1] * v_cam[valid, 1] / z[valid] + K[1, 2]

    return np.stack([u, v_px], axis=1), z


def build_mvp_matrix(
    K: np.ndarray,
    W2C: np.ndarray,
    img_w: int,
    img_h: int,
    near: float = 1.0,
    far: float = 2e6,
) -> np.ndarray:
    """
    构建 nvdiffrast 所需的 MVP (Model-View-Projection) 矩阵。

    nvdiffrast 的 clip-space 约定：
      x_clip ∈ [-1, 1]（右为正）
      y_clip ∈ [-1, 1]（上为正，与 OpenGL 相同）
      z_clip ∈ [-1, 1]（近为 -1，远为 +1）

    像素坐标 (u, v) 与 clip-space 的关系：
      x_clip = (u / img_w) * 2 - 1
      y_clip = 1 - (v / img_h) * 2   ← 注意 Y 轴翻转

    Parameters
    ----------
    K     : (3,3) 相机内参（像素坐标）
    W2C   : (4,4) 世界→相机
    img_w : 图像宽（像素）
    img_h : 图像高（像素）
    near  : 近裁剪面距离（米）
    far   : 远裁剪面距离（米）

    Returns
    -------
    mvp : (4,4) float64，将本地坐标变换到 clip-space
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # 卫星相机沿 +Z 方向观察（z_cam > 0 表示场景在相机前方），
    # 与 OpenGL 约定（沿 -Z 观察）相反，因此 w_clip = +z_cam（而非 -z_cam）。
    #
    # NDC 映射关系（先将像素坐标归一化再匹配 nvdiffrast 约定）：
    #   x_ndc = (u_pixel / img_w) * 2 - 1
    #         = 2*fx/img_w * x_cam/z_cam + (2*cx/img_w - 1)
    #   y_ndc = 1 - (v_pixel / img_h) * 2        ← Y 轴翻转（图像 y↓，NDC y↑）
    #         = -2*fy/img_h * y_cam/z_cam + (1 - 2*cy/img_h)
    #
    # 以 w_clip = +z_cam，x_ndc = x_clip / w_clip：
    #   x_clip = a*x_cam + cx_ndc*z_cam      (a = 2*fx/img_w, cx_ndc = 2*cx/img_w - 1)
    #   y_clip = -b*y_cam + cy_ndc*z_cam     (b = 2*fy/img_h, cy_ndc = 1 - 2*cy/img_h)
    #
    # 深度归一化：z ∈ [near, far] → z_ndc ∈ [-1, +1]
    #   z_ndc = c + d/z_cam   其中 c ≈ +1, d = -2*near*far/(far-near)

    a = 2.0 * fx / img_w
    b = 2.0 * fy / img_h
    cx_ndc = 2.0 * cx / img_w - 1.0
    cy_ndc = 1.0 - 2.0 * cy / img_h   # Y 轴翻转

    # 深度归一化系数（+Z convention: near→-1, far→+1）
    nf = far - near
    c =  (far + near) / nf            # ≈ +1 for large far
    d = -2.0 * far * near / nf        # ≈ -2 for near=1

    # 投影矩阵 P（w = +z_cam）
    P = np.array(
        [
            [a,   0.0,  cx_ndc, 0.0],
            [0.0, -b,  cy_ndc, 0.0],
            [0.0,  0.0,  c,     d  ],
            [0.0,  0.0,  1.0,   0.0],   # w_clip = +z_cam
        ],
        dtype=np.float64,
    )

    mvp = P @ W2C
    return mvp


def mvp_to_tensor(mvp: np.ndarray, device="cuda") -> torch.Tensor:
    """将 (4,4) MVP 矩阵转换为 (1,4,4) CUDA Tensor。"""
    return torch.tensor(mvp, dtype=torch.float32, device=device).unsqueeze(0)


def vertices_to_clip(
    vertices: np.ndarray,
    mvp: np.ndarray,
) -> np.ndarray:
    """
    将 Mesh 顶点变换到 clip-space（齐次坐标）。

    Parameters
    ----------
    vertices : (N, 3)
    mvp      : (4, 4)

    Returns
    -------
    clip : (N, 4) 齐次 clip-space 坐标
    """
    N = len(vertices)
    v_h = np.concatenate([vertices, np.ones((N, 1), dtype=np.float64)], axis=1)
    clip = (mvp @ v_h.T).T
    return clip.astype(np.float32)
