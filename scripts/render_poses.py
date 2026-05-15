"""
根据 poses_for_paper.json 中的相机位姿渲染带纹理 Mesh，输出每个视角的 PNG 图像。

用法：
    conda run -n dmodel python scripts/render_poses.py \\
        --obj   data/JAX_068/textured/mesh_textured.obj \\
        --poses data/JAX_068/poses_for_paper.json \\
        --output data/JAX_068/renders_from_poses

输入：
    --obj    : 带纹理 OBJ 文件（MTL 与纹理 PNG 须在同目录）
    --poses  : poses_for_paper.json
    --output : 输出目录
    --device : cuda / cpu（默认 cuda）

输出：
    <output>/<image_name>   按 poses JSON 中的 image_name 命名
"""

import os
import sys
import json
import argparse

import numpy as np
import cv2

# 将 scripts/ 加入路径
sys.path.insert(0, os.path.dirname(__file__))

from utils.render_utils import render_texture, get_glctx
from utils.camera_utils import build_mvp_matrix, mvp_to_tensor


# ---------------------------------------------------------------------------
# OBJ + MTL + 纹理加载
# ---------------------------------------------------------------------------

def load_obj_with_texture(obj_path: str, device: str = "cuda"):
    """
    加载带纹理 OBJ 文件，返回 vertices/faces/uv/texture tensors。

    支持 texture_bake.py 导出的格式：
      - vt 与 v 一一对应（f v/vt v/vt v/vt，且 v_idx == vt_idx）
      - MTL 通过 map_Kd 指定纹理 PNG

    Returns
    -------
    vertices : (1, N, 3) float32 CUDA Tensor
    faces    : (M, 3) int32 CUDA Tensor
    uv       : (1, N, 2) float32 CUDA Tensor
    texture  : (1, H, W, 3) float32 CUDA Tensor，值域 [0, 1]
    """
    import torch

    obj_dir = os.path.dirname(os.path.abspath(obj_path))

    vertices_list = []
    uv_list       = []
    faces_v_list  = []    # 顶点索引（0-based）
    faces_vt_list = []    # UV 索引（0-based）
    tex_path      = None

    # 先扫描 mtllib 引用
    mtl_path = None
    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("mtllib "):
                mtl_path = os.path.join(obj_dir, line.split(None, 1)[1])
                break

    if mtl_path and os.path.isfile(mtl_path):
        with open(mtl_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("map_Kd "):
                    tex_name = line.split(None, 1)[1].strip()
                    tex_path = os.path.join(obj_dir, tex_name)
                    break

    # 解析 OBJ
    with open(obj_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("v ") and not line.startswith("vt"):
                vals = line.split()[1:]
                vertices_list.append([float(v) for v in vals])
            elif line.startswith("vt "):
                vals = line.split()[1:]
                uv_list.append([float(v) for v in vals])
            elif line.startswith("f "):
                tokens = line.split()[1:]
                idxs_v, idxs_vt = [], []
                for tok in tokens:
                    parts = tok.split("/")
                    idxs_v.append(int(parts[0]) - 1)
                    idxs_vt.append(int(parts[1]) - 1 if len(parts) > 1 and parts[1] else int(parts[0]) - 1)
                # 三角化（面可能是 quad）
                for k in range(1, len(idxs_v) - 1):
                    faces_v_list.append([idxs_v[0],  idxs_v[k],  idxs_v[k+1]])
                    faces_vt_list.append([idxs_vt[0], idxs_vt[k], idxs_vt[k+1]])

    vertices_np = np.array(vertices_list, dtype=np.float32)  # (N, 3)
    uv_raw_np   = np.array(uv_list,       dtype=np.float32)  # (N_vt, 2)
    faces_v_np  = np.array(faces_v_list,  dtype=np.int32)    # (M, 3)
    faces_vt_np = np.array(faces_vt_list, dtype=np.int32)    # (M, 3)

    # OBJ V 轴原点在左下，nvdiffrast 原点在左上 → 翻转 V
    uv_raw_np[:, 1] = 1.0 - uv_raw_np[:, 1]

    # 检查 v 与 vt 是否一一对应（texture_bake.py 的 Tutte/xatlas 导出均满足）
    if np.array_equal(faces_v_np, faces_vt_np):
        # 一一对应：直接使用
        uv_np = uv_raw_np  # (N, 2)
    else:
        # 不一一对应（如有缝合复制）：构建统一索引
        # 将 (v_idx, vt_idx) 配对去重，重建新顶点/UV 数组
        print("[render_poses] v / vt 索引不一致，重建统一索引...")
        pair_to_new = {}
        new_vertices = []
        new_uv       = []
        new_faces    = []
        for fi in range(len(faces_v_np)):
            tri_new = []
            for vi, vti in zip(faces_v_np[fi], faces_vt_np[fi]):
                key = (vi, vti)
                if key not in pair_to_new:
                    pair_to_new[key] = len(new_vertices)
                    new_vertices.append(vertices_np[vi])
                    new_uv.append(uv_raw_np[vti])
                tri_new.append(pair_to_new[key])
            new_faces.append(tri_new)
        vertices_np = np.array(new_vertices, dtype=np.float32)
        uv_np       = np.array(new_uv,       dtype=np.float32)
        faces_v_np  = np.array(new_faces,    dtype=np.int32)

    print(f"[render_poses] Mesh: {len(vertices_np)} 顶点, {len(faces_v_np)} 面")

    # 加载纹理
    if tex_path and os.path.isfile(tex_path):
        img = cv2.imread(tex_path)
        if img is None:
            raise FileNotFoundError(f"无法读取纹理: {tex_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tex_np = img.astype(np.float32) / 255.0
        print(f"[render_poses] 纹理: {tex_np.shape[1]}×{tex_np.shape[0]} ({tex_path})")
    else:
        raise FileNotFoundError(
            f"未找到纹理文件（检查 MTL 中的 map_Kd）。\n"
            f"  OBJ: {obj_path}\n  MTL: {mtl_path}\n  纹理: {tex_path}"
        )

    verts_t   = torch.tensor(vertices_np, dtype=torch.float32, device=device).unsqueeze(0)
    faces_t   = torch.tensor(faces_v_np,  dtype=torch.int32,   device=device).contiguous()
    uv_t      = torch.tensor(uv_np,       dtype=torch.float32, device=device).unsqueeze(0)
    texture_t = torch.tensor(tex_np,      dtype=torch.float32, device=device).unsqueeze(0)

    return verts_t, faces_t, uv_t, texture_t


# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="根据 poses_for_paper.json 渲染带纹理 Mesh")
    p.add_argument("--obj",    required=True, help="mesh_textured.obj 路径")
    p.add_argument("--poses",  required=True, help="poses_for_paper.json 路径")
    p.add_argument("--output", required=True, help="输出目录")
    p.add_argument("--device", default="cuda", help="计算设备（cuda/cpu）")
    p.add_argument("--max_mip_level", type=int, default=4,
                   help="mip mapping 最大层级（默认 4）")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    device = args.device

    # 初始化 nvdiffrast 上下文
    get_glctx(device)

    # 加载 Mesh + 纹理
    print(f"\n[render_poses] 加载 OBJ: {args.obj}")
    verts_t, faces_t, uv_t, texture_t = load_obj_with_texture(args.obj, device)

    # 加载 poses
    print(f"[render_poses] 加载 poses: {args.poses}")
    with open(args.poses, "r") as f:
        poses_data = json.load(f)

    views = poses_data["views"]
    print(f"[render_poses] 共 {len(views)} 个视角\n")

    import torch

    for i, view in enumerate(views):
        name       = view["name"]
        image_name = view.get("image_name", f"{name}.png")

        # 提取相机内参（优先使用 view 内的 intrinsics，否则使用全局）
        intr = view.get("intrinsics", poses_data.get("intrinsics", {}))
        K_list = intr["K"]
        K = np.array(K_list, dtype=np.float64)   # (3,3)
        img_w = int(intr["width"])
        img_h = int(intr["height"])

        # w2c：(4,4) 行优先
        W2C = np.array(view["w2c"], dtype=np.float64)

        # 构建 MVP
        mvp_np = build_mvp_matrix(K, W2C, img_w, img_h)
        mvp    = mvp_to_tensor(mvp_np, device)

        # 渲染
        with torch.no_grad():
            color, alpha = render_texture(
                verts_t, faces_t, uv_t, texture_t,
                mvp, img_h, img_w,
                max_mip_level=args.max_mip_level,
            )

        # (1, H, W, 3) → (H, W, 3) uint8 BGR
        color_np = color.cpu().squeeze(0).numpy()
        alpha_np = alpha.cpu().squeeze(0).squeeze(-1).numpy()

        # 背景填白
        bg = np.ones_like(color_np)
        color_np = color_np * alpha_np[..., None] + bg * (1 - alpha_np[..., None])
        color_np = np.clip(color_np * 255, 0, 255).astype(np.uint8)

        out_path = os.path.join(args.output, image_name)
        cv2.imwrite(out_path, cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR))

        # 保存 mask（单通道灰度图，有三角形=255，无=0）
        mask_np = (alpha_np > 0).astype(np.uint8) * 255
        mask_name = os.path.splitext(image_name)[0] + "_mask.png"
        cv2.imwrite(os.path.join(args.output, mask_name), mask_np)
        print(f"  [{i+1:2d}/{len(views)}] {name:20s} → {out_path}")

    print(f"\n[render_poses] 完成，{len(views)} 张图像已保存至: {args.output}")


if __name__ == "__main__":
    main()
