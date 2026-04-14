"""
基于 nvdiffrast 的可微分纹理渲染工具。

流程：
  1. rasterize       : 将 Mesh 三角面光栅化，得到逐像素三角形 ID 和重心坐标
  2. interpolate     : 插值 UV 坐标到每个像素，同时计算 UV 屏幕空间微分（用于 mip LOD）
  3. texture_sample  : 从 UV atlas 采样颜色（可微分），支持 mip mapping
  4. antialias       : 抗锯齿（用于梯度回传到顶点）

Mip Mapping 说明：
  - 优化阶段（需要梯度）：auto-mip 模式，nvdiffrast 内部从 base texture 自动生成 mip chain
    并支持梯度反传。通过 max_mip_level 限制最低层级，防止采样到极小 mip 层导致过度模糊。
  - 推理/可视化阶段（无需梯度）：可调用 build_mip_stack() 显式构建 mip pyramid，
    传入 mip_stack 参数跳过每次前向的 mip 重建，提升推理速度。
  - 纹理尺寸必须为 2 的幂（512/1024/2048/4096/8192），否则 nvdiffrast auto-mip 会报错。

nvdiffrast 坐标约定：
  - 输入顶点为 clip-space 齐次坐标 (x, y, z, w)
  - y 轴向上（图像 y 轴翻转）
  - 输出图像 [B, H, W, C]，原点在左上角
"""

import torch
import torch.nn.functional as F
import numpy as np

try:
    import nvdiffrast.torch as dr
except ImportError as e:
    raise ImportError("请安装 nvdiffrast: pip install nvdiffrast") from e


_glctx = None  # 全局 GL 上下文


def get_glctx(device: str = "cuda"):
    """获取（或创建）nvdiffrast GL 上下文。"""
    global _glctx
    if _glctx is None:
        _glctx = dr.RasterizeCudaContext()
    return _glctx


def prepare_mesh_buffers(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    device: str = "cuda",
) -> tuple:
    """
    将 Mesh 数据转换为 nvdiffrast 所需的 Tensor 格式。

    Parameters
    ----------
    vertices : (N, 3) float32 本地坐标
    faces    : (M, 3) int32
    uv       : (N, 2) float32 归一化 UV [0,1]

    Returns
    -------
    verts_t  : (1, N, 3) float32 CUDA Tensor
    faces_t  : (M, 3) int32 CUDA Tensor（nvdiffrast 要求连续）
    uv_t     : (1, N, 2) float32 CUDA Tensor
    """
    verts_t = torch.tensor(vertices, dtype=torch.float32, device=device).unsqueeze(0)
    faces_t = torch.tensor(faces, dtype=torch.int32, device=device).contiguous()
    uv_t = torch.tensor(uv, dtype=torch.float32, device=device).unsqueeze(0)
    return verts_t, faces_t, uv_t


def apply_mvp(
    verts: torch.Tensor,
    mvp: torch.Tensor,
) -> torch.Tensor:
    """
    将顶点坐标变换到 clip-space。

    Parameters
    ----------
    verts : (1, N, 3)
    mvp   : (1, 4, 4)

    Returns
    -------
    clip  : (1, N, 4) 齐次 clip-space 坐标
    """
    N = verts.shape[1]
    ones = torch.ones(1, N, 1, dtype=verts.dtype, device=verts.device)
    v_h = torch.cat([verts, ones], dim=-1)  # (1, N, 4)
    clip = torch.bmm(v_h, mvp.permute(0, 2, 1))  # (1, N, 4)
    return clip


def build_mip_stack(
    texture: torch.Tensor,
    max_levels: int = None,
) -> list:
    """
    显式构建 mip pyramid（各层级纹理列表），供推理时传入 render_texture 以避免
    每次前向重复构建 mip chain。

    纹理尺寸必须是 2 的幂；每层将宽高各减半直至 1x1 或达到 max_levels。

    Parameters
    ----------
    texture   : (1, H, W, C) float32 CUDA Tensor，基础层纹理
    max_levels: 最大 mip 层数（含第 0 层），None 表示生成完整 pyramid

    Returns
    -------
    mip_list  : list of (1, H_i, W_i, C) Tensor，第 0 项为原始分辨率
    """
    h, w = texture.shape[1], texture.shape[2]
    if (h & (h - 1)) != 0 or (w & (w - 1)) != 0:
        raise ValueError(f"build_mip_stack: 纹理尺寸必须是 2 的幂，实际为 {h}x{w}")

    mip_list = [texture]
    cur = texture
    level = 1
    while True:
        nh, nw = cur.shape[1] // 2, cur.shape[2] // 2
        if nh < 1 or nw < 1:
            break
        if max_levels is not None and level >= max_levels:
            break
        # (1, H, W, C) -> NCHW -> bilinear downsample -> NHWC
        t = cur.permute(0, 3, 1, 2)
        t = F.interpolate(t, size=(nh, nw), mode="bilinear", align_corners=False)
        t = t.permute(0, 2, 3, 1).contiguous()
        mip_list.append(t)
        cur = t
        level += 1

    return mip_list


def render_texture(
    verts: torch.Tensor,
    faces: torch.Tensor,
    uv: torch.Tensor,
    texture: torch.Tensor,
    mvp: torch.Tensor,
    img_h: int,
    img_w: int,
    enable_mip: bool = True,
    max_mip_level: int = None,
    mip_stack: list = None,
) -> tuple:
    """
    可微分纹理渲染，支持 mip mapping。

    Parameters
    ----------
    verts         : (1, N, 3) 本地坐标
    faces         : (M, 3) int32
    uv            : (1, N, 2) 归一化 UV
    texture       : (1, H_tex, W_tex, 3) 纹理 atlas，值域 [0,1]（尺寸须为 2 的幂）
    mvp           : (1, 4, 4) MVP 矩阵
    img_h, img_w  : 输出分辨率
    enable_mip    : 是否启用 mip mapping（推荐 True 以减少 aliasing）
    max_mip_level : 最大 mip 层级（0 = 仅用原始分辨率，4 = 最小 1/16 分辨率）
                    None 表示不限制。建议设为 log2(atlas/output)+2 以防止过度模糊。
    mip_stack     : 预构建的 mip pyramid（build_mip_stack() 的返回值）
                    提供时跳过内部 auto-mip 构建；适合推理/无梯度渲染。
                    优化阶段请传 None，让 nvdiffrast 自动构建以支持梯度反传。

    Returns
    -------
    color : (1, img_h, img_w, 3) 渲染颜色，值域 [0,1]
    alpha : (1, img_h, img_w, 1) 可见性 mask（bool-like float）
    """
    glctx = get_glctx(verts.device.type)

    # 1. 变换到 clip-space
    clip = apply_mvp(verts, mvp)  # (1, N, 4)

    # 2. 光栅化（rast_db 包含重心坐标屏幕空间微分，mip 计算必需）
    rast, rast_db = dr.rasterize(glctx, clip, faces, resolution=[img_h, img_w])

    # 3. 插值 UV 并计算 UV 屏幕空间微分 uv_da（用于 mip LOD 估计）
    if enable_mip:
        uv_interp, uv_da = dr.interpolate(uv, rast, faces, rast_db=rast_db, diff_attrs="all")
        # mip_stack 提供时使用显式 pyramid（推理加速），否则 auto-mip（支持梯度）
        tex_src = mip_stack if mip_stack is not None else texture
        color = dr.texture(
            tex_src,
            uv_interp,
            uv_da=uv_da,
            filter_mode="linear-mipmap-linear",
            max_mip_level=max_mip_level,
        )
    else:
        uv_interp, _ = dr.interpolate(uv, rast, faces)
        color = dr.texture(texture, uv_interp, filter_mode="linear")

    # 4. 生成可见性 mask（triangle_id > 0 表示有三角形覆盖）
    alpha = (rast[..., 3:4] > 0).float()  # (1, H, W, 1)

    # 5. 抗锯齿（反向传播时平滑边界梯度）
    color = dr.antialias(color, rast, clip, faces)

    # 6. 垂直翻转：nvdiffrast 输出 row 0 = NDC y=-1 = 图像底部（OpenGL Y-up 规范），
    #    翻转后 row 0 = 图像顶部，与标准图像坐标系一致（参见 nvdiffrast/samples/triangle.py）
    color = color.flip(dims=[1])
    alpha = alpha.flip(dims=[1])

    return color, alpha


def texture_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    alpha: torch.Tensor,
    lambda_mse: float = 0.8,
    lambda_ssim: float = 0.2,
) -> torch.Tensor:
    """
    纹理优化损失 = λ_MSE * L_MSE + λ_SSIM * (1 - SSIM)

    仅在可见像素 (alpha > 0) 上计算。

    Parameters
    ----------
    rendered : (1, H, W, 3) 渲染图，值域 [0,1]
    target   : (1, H, W, 3) 目标图（卫星图像裁剪/调整后），值域 [0,1]
    alpha    : (1, H, W, 1) 可见性 mask
    """
    from pytorch_msssim import ssim

    mask = alpha.squeeze(-1).bool()  # (1, H, W)

    # 转换为 NCHW 格式供 SSIM 计算
    r_nchw = rendered.permute(0, 3, 1, 2)  # (1, 3, H, W)
    t_nchw = target.permute(0, 3, 1, 2)

    # MSE（仅可见区域）
    r_masked = rendered * alpha
    t_masked = target * alpha
    n_visible = alpha.sum().clamp(min=1.0)
    l_mse = ((r_masked - t_masked) ** 2).sum() / n_visible / 3.0

    # SSIM（全图，使用 alpha 加权）
    l_ssim = 1.0 - ssim(
        r_nchw * alpha.permute(0, 3, 1, 2),
        t_nchw * alpha.permute(0, 3, 1, 2),
        data_range=1.0,
        size_average=True,
    )

    return lambda_mse * l_mse + lambda_ssim * l_ssim


def create_texture(
    atlas_size: int = 8192,
    device: str = "cuda",
    init: str = "gray",
) -> torch.nn.Parameter:
    """
    初始化可优化的纹理 atlas。

    Parameters
    ----------
    atlas_size : 纹理分辨率（正方形）
    init : "gray" | "zero" | "random"
    """
    if init == "gray":
        data = torch.full((1, atlas_size, atlas_size, 3), 0.5)
    elif init == "zero":
        data = torch.zeros(1, atlas_size, atlas_size, 3)
    else:
        data = torch.rand(1, atlas_size, atlas_size, 3)

    return torch.nn.Parameter(data.to(device))


def load_image_as_tensor(
    image_path: str,
    device: str = "cuda",
) -> torch.Tensor:
    """
    加载图像为 (1, H, W, 3) float32 CUDA Tensor，值域 [0,1]。
    """
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img, device=device).unsqueeze(0)  # (1, H, W, 3)


def load_mask_as_tensor(
    mask_path: str,
    target_h: int,
    target_w: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    加载图像 mask，返回 (1, H, W, 1) float32 Tensor，值域 [0, 1]。

    支持格式：
      - .npy : 值 0/1 (uint8)
      - .png / .jpg : 灰度图，255 = 有效，0 = 无效
    """
    import cv2
    if mask_path.endswith(".npy"):
        m = np.load(mask_path).astype(np.float32)          # (H, W), 0/1
    else:
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"无法读取 mask: {mask_path}")
        m = m.astype(np.float32) / 255.0                    # (H, W), 0/1

    if m.shape[0] != target_h or m.shape[1] != target_w:
        m = cv2.resize(m, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return torch.tensor(m, device=device).unsqueeze(0).unsqueeze(-1)  # (1, H, W, 1)


def save_texture(texture: torch.Tensor, path: str):
    """将纹理 atlas 保存为 PNG。"""
    import cv2
    t = texture.detach().cpu().squeeze(0).numpy()  # (H, W, 3)
    t = np.clip(t * 255, 0, 255).astype(np.uint8)
    t = cv2.cvtColor(t, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, t)
    print(f"[render] 纹理已保存: {path}")


def render_normals(
    verts: torch.Tensor,
    faces: torch.Tensor,
    normals: torch.Tensor,
    mvp: torch.Tensor,
    img_h: int,
    img_w: int,
) -> tuple:
    """
    渲染世界空间顶点法向量图。

    Parameters
    ----------
    verts   : (1, N, 3) 本地坐标
    faces   : (M, 3) int32
    normals : (1, N, 3) 顶点法向量（单位向量，世界空间）
    mvp     : (1, 4, 4) MVP 矩阵
    img_h, img_w : 输出分辨率

    Returns
    -------
    normal_img : (1, H, W, 3) 逐像素法向量，值域 [-1, 1]，背景为 0
    alpha      : (1, H, W, 1) 可见性 mask
    """
    glctx = get_glctx(verts.device.type)

    clip = apply_mvp(verts, mvp)
    rast, _ = dr.rasterize(glctx, clip, faces, resolution=[img_h, img_w])

    # 插值顶点法向量到每个像素
    n_interp, _ = dr.interpolate(normals, rast, faces)   # (1, H, W, 3)

    alpha = (rast[..., 3:4] > 0).float()   # (1, H, W, 1)

    # 插值后重归一化（三角形内部插值会破坏单位长度）
    n_interp = n_interp / (n_interp.norm(dim=-1, keepdim=True) + 1e-8)

    # 背景像素置零
    n_interp = n_interp * alpha

    # 垂直翻转：与 render_texture 保持一致（nvdiffrast Y-up 规范修正）
    n_interp = n_interp.flip(dims=[1])
    alpha    = alpha.flip(dims=[1])

    return n_interp, alpha


def upsample_texture(
    texture: torch.nn.Parameter,
    new_size: int,
) -> torch.nn.Parameter:
    """
    将纹理 atlas 双线性上采样到新分辨率，返回新的可优化 Parameter。

    Parameters
    ----------
    texture  : (1, H, W, 3) nn.Parameter，值域 [0,1]
    new_size : 目标分辨率（正方形）

    Returns
    -------
    新的 (1, new_size, new_size, 3) nn.Parameter，与原 texture 设备相同
    """
    t = texture.data.permute(0, 3, 1, 2)   # (1, 3, H, W)
    t = F.interpolate(t, size=(new_size, new_size), mode="bilinear", align_corners=False)
    t = t.permute(0, 2, 3, 1).contiguous()  # (1, new_size, new_size, 3)
    return torch.nn.Parameter(t.clamp(0.0, 1.0))
