"""
卫星 Mesh 纹理生成主脚本。

实现论文 "From Orbit to Ground: Generative City Photogrammetry from Extreme
Off-Nadir Satellite Images" (2512.07527v2) 中的两阶段纹理生成方法：

  阶段 1 - 基础纹理 T_basic：
    对每张卫星输入图像做可微分渲染，用 MSE + SSIM 损失优化 UV atlas（100 epochs）

  阶段 2 - 迭代精炼（可选，需要恢复网络 D）：
    生成模拟 UAV 新视角 → 渲染退化图 → 经恢复网络生成伪真值 → 再次优化（2 iter × 20 epochs）

用法（基础，无 FLUX）：
  conda activate dmodel
  python scripts/texture_bake.py \\
    --mesh      data/JAX_068/mesh/00100000.ply \\
    --mesh_info data/JAX_068/mesh/mesh_info.txt \\
    --cameras   data/JAX_068/cameras \\
    --images    data/JAX_068/images \\
    --output    data/JAX_068/textured

用法（含 FLUX-Schnell 恢复网络）：
  conda activate dmodel
  python scripts/texture_bake.py \\
    --mesh      data/JAX_068/mesh/00100000.ply \\
    --mesh_info data/JAX_068/mesh/mesh_info.txt \\
    --cameras   data/JAX_068/cameras \\
    --images    data/JAX_068/images \\
    --output    data/JAX_068/textured \\
    --use_flux \\
    --flux_weights_dir /path/to/weights \\  # 可选，默认使用 HuggingFace 缓存
    --flux_strength 0.3
"""

import os

# 必须在 import torch / CUDA 初始化之前设置，避免显存碎片化导致 OOM
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import cv2

# 将 scripts 目录加入路径
sys.path.insert(0, os.path.dirname(__file__))

from utils.heightfield import build_heightfield_mesh, save_heightfield_mesh
from utils.camera_utils import (
    load_cameras_from_dir,
    build_mvp_matrix,
    mvp_to_tensor,
)
from utils.render_utils import (
    prepare_mesh_buffers,
    render_texture,
    texture_loss,
    create_texture,
    upsample_texture,
    load_image_as_tensor,
    save_texture,
)
from utils.novel_view import generate_novel_cameras, subsample_novel_cameras


# ---------------------------------------------------------------------------
# 图像加载辅助
# ---------------------------------------------------------------------------

def load_and_resize_image(
    image_path: str,
    target_h: int,
    target_w: int,
    device: str = "cuda",
) -> torch.Tensor:
    """
    加载图像并缩放到目标分辨率，返回 (1, H, W, 3) float32 [0,1]。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"无法读取图像: {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img.shape[0] != target_h or img.shape[1] != target_w:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# 导出带纹理 OBJ
# ---------------------------------------------------------------------------

def export_textured_obj(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv: np.ndarray,
    texture_path: str,
    output_dir: str,
    name: str = "mesh_textured",
):
    """
    将带 UV 的 Mesh 导出为 OBJ + MTL 格式。

    Parameters
    ----------
    vertices : (N, 3)
    faces    : (M, 3) 顶点索引（0-based）
    uv       : (N, 2) 归一化 UV（与 vertices 一一对应）
    texture_path : 纹理 PNG 文件路径
    output_dir   : 输出目录
    name         : 文件名前缀
    """
    os.makedirs(output_dir, exist_ok=True)
    obj_path = os.path.join(output_dir, f"{name}.obj")
    mtl_path = os.path.join(output_dir, f"{name}.mtl")
    tex_name = os.path.basename(texture_path)

    # 写 MTL
    with open(mtl_path, "w") as f:
        f.write(f"newmtl material0\n")
        f.write(f"Ka 1.0 1.0 1.0\n")
        f.write(f"Kd 1.0 1.0 1.0\n")
        f.write(f"Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd {tex_name}\n")

    # 写 OBJ
    with open(obj_path, "w") as f:
        f.write(f"mtllib {os.path.basename(mtl_path)}\n")
        f.write(f"usemtl material0\n")
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in uv:
            # OBJ UV：v 轴翻转（OBJ 原点在左下，nvdiffrast 在左上）
            f.write(f"vt {t[0]:.6f} {1.0 - t[1]:.6f}\n")
        # 面（1-based 索引，顶点/UV 相同）
        for tri in faces:
            i0, i1, i2 = tri[0] + 1, tri[1] + 1, tri[2] + 1
            f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2}\n")

    print(f"[export] OBJ 已保存: {obj_path}")
    print(f"[export] MTL 已保存: {mtl_path}")


# ---------------------------------------------------------------------------
# 渐进分辨率调度
# ---------------------------------------------------------------------------

def build_progressive_schedule(
    init_size: int,
    final_size: int,
    total_epochs: int,
) -> list:
    """
    构建从 init_size 到 final_size 的 2× 逐级分辨率调度表。

    Parameters
    ----------
    init_size    : 起始纹理分辨率（2 的幂）
    final_size   : 最终纹理分辨率
    total_epochs : 全部 epoch 数，均匀分配到各级

    Returns
    -------
    [(size, epochs), ...] 从低到高排列
    """
    sizes, s = [], init_size
    while s < final_size:
        sizes.append(s)
        s *= 2
    sizes.append(final_size)

    n = len(sizes)
    epochs_per = max(1, total_epochs // n)
    schedule = [(sz, epochs_per) for sz in sizes[:-1]]
    # 最后一级吸收余数
    schedule.append((final_size, total_epochs - epochs_per * (n - 1)))
    return schedule


# ---------------------------------------------------------------------------
# 核心优化循环
# ---------------------------------------------------------------------------

def optimize_texture(
    verts_t: torch.Tensor,
    faces_t: torch.Tensor,
    uv_t: torch.Tensor,
    texture: torch.nn.Parameter,
    cameras: list,
    n_epochs: int,
    lr: float = 0.01,
    stage_name: str = "T_basic",
    lambda_mse: float = 0.8,
    lambda_ssim: float = 0.2,
    device: str = "cuda",
):
    """
    对给定相机列表执行纹理优化。

    Parameters
    ----------
    cameras : list of dict，每个 dict 需包含 K, W2C, img_w, img_h, image_path
              image_path 可为 None（新视角无参考图像时跳过该相机）
    n_epochs : 优化轮数
    """
    optimizer = torch.optim.Adam([texture], lr=lr)

    # 过滤掉没有参考图像的相机
    valid_cameras = [c for c in cameras if c.get("image_path") is not None]
    if len(valid_cameras) == 0:
        print(f"[{stage_name}] 警告：无有效参考图像，跳过优化")
        return

    print(f"\n[{stage_name}] 开始优化: {n_epochs} epochs × {len(valid_cameras)} 相机")

    for epoch in range(n_epochs):
        t0 = time.time()
        epoch_loss = 0.0
        cam_count = 0

        for cam in valid_cameras:
            img_w = cam["img_w"]
            img_h = cam["img_h"]

            # 读取参考图像
            try:
                gt = load_and_resize_image(
                    cam["image_path"], img_h, img_w, device
                )
            except Exception as e:
                continue

            # 构建 MVP 矩阵
            mvp_np = build_mvp_matrix(
                cam["K"], cam["W2C"], img_w, img_h
            )
            mvp = mvp_to_tensor(mvp_np, device)

            # 可微分渲染
            color, alpha = render_texture(
                verts_t, faces_t, uv_t, texture, mvp, img_h, img_w
            )

            # 检查是否有可见像素
            if alpha.sum() < 10:
                continue

            # 计算损失
            loss = texture_loss(color, gt, alpha, lambda_mse, lambda_ssim)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 将纹理值 clamp 到 [0, 1]
            with torch.no_grad():
                texture.data.clamp_(0.0, 1.0)

            epoch_loss += loss.item()
            cam_count += 1

        elapsed = time.time() - t0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = epoch_loss / max(cam_count, 1)
            print(
                f"  [{stage_name}] epoch {epoch+1:3d}/{n_epochs} | "
                f"loss={avg_loss:.5f} | cams={cam_count} | {elapsed:.1f}s"
            )

    print(f"[{stage_name}] 优化完成")


# ---------------------------------------------------------------------------
# 迭代精炼
# ---------------------------------------------------------------------------

class IdentityRestorer:
    """恢复网络占位符：直接返回输入图像（无增强）。"""
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image


def run_iterative_refinement(
    verts_t: torch.Tensor,
    faces_t: torch.Tensor,
    uv_t: torch.Tensor,
    texture: torch.nn.Parameter,
    novel_cameras: list,
    n_iters: int,
    n_epochs: int,
    restorer,
    output_dir: str,
    device: str = "cuda",
):
    """
    迭代精炼循环（论文 §3.2）。

    每次迭代：
      1. 从当前纹理渲染新视角图像 I_low
      2. 经恢复网络得到伪真值 I_target = D(I_low)
      3. 以 I_target 为监督优化纹理（20 epochs）
    """
    print(f"\n[精炼] 开始迭代精炼: {n_iters} 轮 × {n_epochs} epochs")

    for it in range(n_iters):
        print(f"\n[精炼] === 迭代 {it+1}/{n_iters} ===")

        # 为每个新视角相机渲染并生成伪真值
        refined_cameras = []

        for cam in novel_cameras:
            img_w = cam["img_w"]
            img_h = cam["img_h"]

            mvp_np = build_mvp_matrix(
                cam["K"], cam["W2C"], img_w, img_h
            )
            mvp = mvp_to_tensor(mvp_np, device)

            with torch.no_grad():
                I_low, alpha = render_texture(
                    verts_t, faces_t, uv_t, texture, mvp, img_h, img_w
                )

            if alpha.sum() < 100:
                continue

            # 经恢复网络生成伪真值
            I_target = restorer(I_low)

            # 保存伪真值到临时目录，供 optimize_texture 加载
            tmp_dir = os.path.join(output_dir, "tmp_novel_views")
            os.makedirs(tmp_dir, exist_ok=True)
            tmp_path = os.path.join(tmp_dir, f"iter{it}_{cam['name']}.png")

            # 保存为图像文件
            img_np = I_target.detach().cpu().squeeze(0).numpy()
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(tmp_path, cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))

            cam_with_img = dict(cam)
            cam_with_img["image_path"] = tmp_path
            refined_cameras.append(cam_with_img)

        if len(refined_cameras) == 0:
            print(f"[精炼] 迭代 {it+1}：无有效新视角，跳过")
            continue

        print(f"[精炼] 迭代 {it+1}：{len(refined_cameras)} 个有效新视角")

        optimize_texture(
            verts_t, faces_t, uv_t, texture,
            refined_cameras,
            n_epochs=n_epochs,
            stage_name=f"refine_iter{it+1}",
            device=device,
        )

    print("[精炼] 迭代精炼完成")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="卫星 Mesh 纹理生成")
    p.add_argument("--mesh",        required=True, help="输入 mesh.ply 路径")
    p.add_argument("--mesh_info",   required=True, help="mesh_info.txt 路径")
    p.add_argument("--cameras",     required=True, help="cameras/ 目录路径")
    p.add_argument("--images",      required=True, help="输入卫星图像目录")
    p.add_argument("--output",      required=True, help="输出目录")
    p.add_argument("--hf_res",      type=int, default=512,  help="高度场分辨率")
    p.add_argument("--atlas_size",  type=int, default=8192, help="纹理 atlas 最终分辨率")
    p.add_argument("--init_atlas_size", type=int, default=512,
                   help="渐进分辨率起始尺寸（2 的幂，如 512）；None 表示直接用 atlas_size")
    p.add_argument("--basic_epochs",type=int, default=100,  help="T_basic 优化轮数")
    p.add_argument("--refine_iters",type=int, default=2,    help="精炼迭代次数")
    p.add_argument("--refine_epochs",type=int, default=20,  help="每轮精炼优化步数")
    p.add_argument("--max_novel_cams",type=int,default=64,  help="最大新视角相机数")
    p.add_argument("--lr",          type=float, default=0.01, help="Adam 学习率")
    p.add_argument("--device",      default="cuda", help="计算设备")
    p.add_argument("--skip_refine", action="store_true", help="跳过迭代精炼阶段")
    p.add_argument("--save_hf_mesh",action="store_true", help="保存高度场 PLY Mesh")
    # Mesh 对齐到稀疏点云
    p.add_argument("--points3d", default=None,
                   help="COLMAP points3D.txt 路径；提供后在高度场转换前自动对齐 Mesh 到点云")
    p.add_argument("--align_fft_voxel_size", type=float, default=2.0,
                   help="FFT 粗对齐体素尺寸（米），默认 2.0")
    p.add_argument("--align_trim",  type=float, default=0.3,
                   help="trimmed-median 每轮丢弃的最差对比例，默认 0.3")
    p.add_argument("--align_iters", type=int,   default=20,
                   help="精对齐最大迭代次数，默认 20")
    # FLUX-Schnell 恢复网络参数
    p.add_argument("--use_flux",         action="store_true",
                   help="使用 FLUX-Schnell 图像恢复网络增强新视角（需要 diffusers）")
    p.add_argument("--flux_model",       default="black-forest-labs/FLUX.1-schnell",
                   help="FLUX 模型 ID 或本地路径")
    p.add_argument("--flux_weights_dir", default=None,
                   help="HuggingFace 权重缓存目录（默认使用 HuggingFace 标准缓存）")
    p.add_argument("--flux_strength",    type=float, default=0.3,
                   help="img2img 强度 (0~1)，越低越保留输入结构")
    p.add_argument("--flux_res",         type=int,   default=1024,
                   help="FLUX 内部处理分辨率（正方形，之后缩放回原尺寸）")
    p.add_argument("--flux_prompt",
                   default="high quality aerial urban view, sharp building details, photorealistic",
                   help="FLUX 引导提示词")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = args.device

    print("=" * 60)
    print("卫星 Mesh 纹理生成")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 步骤 0a（可选）：Mesh 对齐到 COLMAP 稀疏点云
    # ------------------------------------------------------------------
    aligned_verts = None
    if args.points3d:
        from utils.alignment import load_colmap_points3d, align_mesh_to_pointcloud
        import trimesh as _trimesh

        points3d_path = os.path.abspath(args.points3d)
        if not os.path.isfile(points3d_path):
            raise FileNotFoundError(f"找不到 points3D.txt: {points3d_path}")

        print("\n[步骤 0a] 加载 COLMAP 稀疏点云 ...")
        colmap_pts = load_colmap_points3d(points3d_path,
                                          max_reproj_error=2.0,
                                          min_track_len=3)
        print(f"  有效点: {len(colmap_pts)}")

        print("\n[步骤 0b] Mesh 对齐到点云 ...")
        raw_mesh_path = os.path.abspath(args.mesh)
        raw_mesh = _trimesh.load(raw_mesh_path, force="mesh", process=False)
        aligned_verts = align_mesh_to_pointcloud(
            np.asarray(raw_mesh.vertices, dtype=np.float32),
            colmap_pts,
            trim_fraction=args.align_trim,
            n_iters=args.align_iters,
            fft_voxel_size=args.align_fft_voxel_size,
        )

        # 保存对齐结果供可视化检验
        aligned_mesh_path = os.path.join(args.output, "aligned_mesh.ply")
        colmap_ply_path   = os.path.join(args.output, "colmap_points.ply")

        aligned_mesh_obj = _trimesh.Trimesh(
            vertices=aligned_verts.astype(np.float64),
            faces=raw_mesh.faces,
            process=False,
        )
        aligned_mesh_obj.export(aligned_mesh_path)
        print(f"  [对齐] 已保存对齐后 Mesh: {aligned_mesh_path}")

        colmap_pcd = _trimesh.PointCloud(vertices=colmap_pts.astype(np.float64))
        colmap_pcd.export(colmap_ply_path)
        print(f"  [对齐] 已保存 COLMAP 点云: {colmap_ply_path}")

    # ------------------------------------------------------------------
    # 步骤 0：NeuS Mesh → 2.5D 高度场 Mesh
    # ------------------------------------------------------------------
    print("\n[步骤 0] 生成 2.5D 高度场 Mesh ...")
    hf = build_heightfield_mesh(
        mesh_ply_path=args.mesh,
        mesh_info_path=args.mesh_info,
        resolution=args.hf_res,
        aligned_vertices=aligned_verts,
    )

    if args.save_hf_mesh:
        hf_ply_path = os.path.join(args.output, "heightfield.ply")
        save_heightfield_mesh(hf, hf_ply_path)

    vertices = hf["vertices"]   # (N, 3) float32
    faces    = hf["faces"]      # (M, 3) int32
    uv       = hf["uv"]         # (N, 2) float32
    x_min, x_max = hf["x_min"], hf["x_max"]
    y_min, y_max = hf["y_min"], hf["y_max"]
    z_min, z_max = hf["z_min"], hf["z_max"]

    # ------------------------------------------------------------------
    # 步骤 1：加载相机
    # ------------------------------------------------------------------
    print("\n[步骤 1] 加载卫星相机 ...")
    cameras = load_cameras_from_dir(args.cameras, args.images)
    valid_cams = [c for c in cameras if c["image_path"] is not None]
    print(f"  加载 {len(cameras)} 个相机，其中 {len(valid_cams)} 个有对应图像")

    if len(valid_cams) == 0:
        print("  错误：未找到任何卫星图像，请检查 --images 目录")
        print("  期望格式：images/JAX_068_001_RGB.png, ...")
        sys.exit(1)

    # 补全 img_w / img_h（从实际图像读取）
    for cam in cameras:
        if cam["image_path"] is not None and (cam["img_w"] is None or cam["img_h"] is None):
            img = cv2.imread(cam["image_path"])
            if img is not None:
                cam["img_h"], cam["img_w"] = img.shape[:2]

    # ------------------------------------------------------------------
    # 步骤 2：准备 GPU Tensor
    # ------------------------------------------------------------------
    print("\n[步骤 2] 上传 Mesh 到 GPU ...")
    verts_t, faces_t, uv_t = prepare_mesh_buffers(vertices, faces, uv, device)
    print(f"  顶点: {verts_t.shape}, 面: {faces_t.shape}")

    # ------------------------------------------------------------------
    # 步骤 3：初始化纹理 atlas（渐进分辨率从 init_atlas_size 开始）
    # ------------------------------------------------------------------
    use_progressive = (
        args.init_atlas_size is not None
        and args.init_atlas_size < args.atlas_size
    )
    init_size = args.init_atlas_size if use_progressive else args.atlas_size
    texture = create_texture(init_size, device, init="gray")
    if use_progressive:
        print(f"\n[步骤 3] 渐进纹理 atlas: {init_size}×{init_size} → {args.atlas_size}×{args.atlas_size}")
    else:
        print(f"\n[步骤 3] 纹理 atlas: {args.atlas_size}×{args.atlas_size}")

    # ------------------------------------------------------------------
    # 步骤 4：T_basic 优化（支持渐进分辨率）
    # ------------------------------------------------------------------
    print("\n[步骤 4] 基础纹理 T_basic 优化 ...")
    if use_progressive:
        schedule = build_progressive_schedule(
            args.init_atlas_size, args.atlas_size, args.basic_epochs
        )
        print(f"  渐进调度: {[(sz, ep) for sz, ep in schedule]}")
        for stage_size, stage_epochs in schedule:
            cur_size = texture.shape[1]
            if cur_size != stage_size:
                texture = upsample_texture(texture, stage_size)
                print(f"  [进度] 上采样纹理: {cur_size}×{cur_size} → {stage_size}×{stage_size}")
            optimize_texture(
                verts_t, faces_t, uv_t, texture,
                valid_cams,
                n_epochs=stage_epochs,
                lr=args.lr,
                stage_name=f"T_basic_{stage_size}",
                device=device,
            )
    else:
        optimize_texture(
            verts_t, faces_t, uv_t, texture,
            valid_cams,
            n_epochs=args.basic_epochs,
            lr=args.lr,
            stage_name="T_basic",
            device=device,
        )

    # 保存 T_basic
    basic_tex_path = os.path.join(args.output, "texture_basic.png")
    save_texture(texture, basic_tex_path)

    # ------------------------------------------------------------------
    # 步骤 5：迭代精炼（可选）
    # ------------------------------------------------------------------
    if not args.skip_refine and args.refine_iters > 0:
        print("\n[步骤 5] 生成模拟 UAV 新视角相机 ...")
        novel_cams = generate_novel_cameras(
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            z_max=z_max,
        )
        novel_cams = subsample_novel_cameras(novel_cams, args.max_novel_cams)
        print(f"  生成 {len(novel_cams)} 个新视角相机")

        # 恢复网络：根据 --use_flux 参数选择
        if args.use_flux:
            from utils.flux_restorer import FluxRestorer
            restorer = FluxRestorer(
                model_id=args.flux_model,
                cache_dir=args.flux_weights_dir,
                flux_resolution=args.flux_res,
                strength=args.flux_strength,
                prompt=args.flux_prompt,
                device=device,
            )
            print(f"  [FLUX] 恢复网络已加载: {args.flux_model}")
            print(f"  [FLUX] strength={args.flux_strength}, resolution={args.flux_res}")
        else:
            restorer = IdentityRestorer()
            print("  [注意] 使用 IdentityRestorer（无图像增强）")
            print("  [提示] 使用 --use_flux 启用 FLUX-Schnell 恢复网络")

        run_iterative_refinement(
            verts_t, faces_t, uv_t, texture,
            novel_cams,
            n_iters=args.refine_iters,
            n_epochs=args.refine_epochs,
            restorer=restorer,
            output_dir=args.output,
            device=device,
        )
    else:
        print("\n[步骤 5] 跳过迭代精炼（--skip_refine 已设置或 refine_iters=0）")

    # ------------------------------------------------------------------
    # 步骤 6：导出带纹理 Mesh
    # ------------------------------------------------------------------
    print("\n[步骤 6] 导出带纹理 Mesh ...")
    final_tex_path = os.path.join(args.output, "texture_final.png")
    save_texture(texture, final_tex_path)

    export_textured_obj(
        vertices=vertices,
        faces=faces,
        uv=uv,
        texture_path=final_tex_path,
        output_dir=args.output,
        name="mesh_textured",
    )

    print("\n" + "=" * 60)
    print(f"完成！输出目录: {args.output}")
    print("  mesh_textured.obj  - 带纹理 Mesh")
    print("  mesh_textured.mtl  - 材质文件")
    print("  texture_final.png  - 最终纹理 atlas")
    print("  texture_basic.png  - T_basic 基础纹理")
    print("=" * 60)


if __name__ == "__main__":
    main()
