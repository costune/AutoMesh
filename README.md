# AutoMesh

**AutoMesh** 是一个面向卫星多视角图像的 **3D Mesh 自动纹理生成**框架。输入为 NeuS 神经场景重建得到的无纹理 Mesh 及对应卫星图像，通过"几何重建 + 可微分渲染优化 + FLUX 生成式精炼"三阶段管线，生成高质量带纹理 Mesh。

> 实现参考论文：*From Orbit to Ground: Generative City Photogrammetry from Extreme Off-Nadir Satellite Images*（arXiv 2512.07527v2）

---

## 目录

- [方法概述](#方法概述)
- [环境配置](#环境配置)
- [数据文件夹格式](#数据文件夹格式)
- [快速开始](#快速开始)
- [主要脚本说明](#主要脚本说明)
- [完整参数说明](#完整参数说明)
- [输出文件说明](#输出文件说明)
- [详细技术文档](#详细技术文档)

---

## 方法概述

```
NeuS Mesh (untextured)
        │
        ▼ 【可选】Mesh 对齐到 COLMAP 稀疏点云
        │   3D FFT 互相关粗对齐 + Trimmed-Median 精对齐
        │
        ▼ 几何预处理（heightfield.py）
        │   射线采样高度场 → 体素化 → Marching Cubes（流形 Mesh）
        │   → （可选）Quadric 简化 → Tutte / xatlas UV 展开
        │
        ▼ 基础纹理优化 T_basic（nvdiffrast 可微分渲染）
        │   MSE + SSIM 损失，渐进分辨率（512→2048），100 epochs
        │
        ▼ 【可选】几何优化
        │   固定纹理，自由优化顶点 XYZ + Laplacian 平滑正则化
        │
        ▼ 【可选】迭代纹理精炼（FLUX FlowEdit）
        │   模拟 UAV 新视角渲染 → FLUX 图像增强 → 纹理优化
        │
        ▼
Textured Mesh（OBJ + MTL + PNG）
```

---

## 环境配置

```bash
# 创建 conda 环境（需要 CUDA 12.9+）
conda create -n automesh python=3.10
conda activate automesh

# 安装 PyTorch（示例：CUDA 12.9）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129

# 安装 nvdiffrast
pip install nvdiffrast

# 安装其余依赖
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
trimesh>=4.0.0
rtree>=1.0.0
pytorch-msssim>=1.0.0
scipy>=1.10.0
opencv-python>=4.8.0
xatlas>=0.0.8
scikit-image>=0.21.0
fast-simplification>=0.1.0
diffusers>=0.30.0          # 启用 --use_flux 时需要
transformers>=4.40.0
sentencepiece>=0.1.99
accelerate>=0.30.0
```

---

## 数据文件夹格式

每个场景对应一个独立文件夹，期望的完整目录结构如下：

```
data/
└── <SCENE_ID>/                    # 场景根目录，如 JAX_068、OMA_203
    │
    ├── mesh/                      # NeuS 重建输出（必需）
    │   ├── mesh.ply               # 无纹理 Mesh，顶点坐标为本地坐标系（米）
    │   └── mesh_info.txt          # Mesh 空间范围描述文件
    │
    ├── cameras/                   # 卫星相机参数（必需）
    │   ├── <NAME>_RGB.json        # 每张图像对应一个 JSON
    │   └── ...
    │
    ├── images/                    # 卫星 RGB 图像（必需）
    │   ├── <NAME>_RGB.png         # 与 cameras/ 下 JSON 同名（不含扩展名）
    │   └── ...
    │
    ├── masks/                     # 图像有效区域掩码（可选，推荐）
    │   ├── <NAME>_RGB.npy         # 与图像同名，uint8，值域 {0,1}
    │   ├── <NAME>_RGB.png         # 或 PNG 灰度图，255=有效，0=无效
    │   └── ...
    │
    ├── points3D.txt               # COLMAP 稀疏点云（可选，用于 Mesh 对齐）
    │
    └── poses_for_paper.json       # 可视化渲染位姿（可选，供 render_poses.py 使用）
```

### `mesh_info.txt` 格式

```
iter_step: 100000
resolution: 512
threshold: 0.0
vertices: (922367, 3)
triangles: (1840640, 3)
center: [4.3620769e+05  3.3576070e+06  1.5000013e+00]   # UTM 坐标（米）
range: 205.125                                            # 真实物理尺度（米）
val_range: [[-0.627  -0.541  -0.236]                      # 本地坐标 XYZ 最小值
            [ 0.621   0.707   0.236]]                     # 本地坐标 XYZ 最大值
```

- `center`：Mesh 中心的 UTM 坐标，本地坐标 + center = 世界 UTM 坐标
- `range`：归一化尺度因子，本地坐标 = 归一化坐标 × range（mesh.ply 已乘以此因子）
- `val_range`：本地坐标系下的 XYZ 范围 `[min, max]`

### `cameras/*.json` 格式

每张卫星图像对应一个 JSON，包含 4×4 内参矩阵、世界→相机矩阵和图像尺寸：

```json
{
  "K": [fx, 0, cx, 0,  0, fy, cy, 0,  0, 0, 1, 0,  0, 0, 0, 1],
  "W2C": [r00, r01, r02, t0,  r10, r11, r12, t1,  r20, r21, r22, t2,  0, 0, 0, 1],
  "img_size": [W, H]
}
```

- `K`：16 元素行主序（4×4），取左上 3×3 为相机内参，单位为像素
- `W2C`：16 元素行主序（4×4），世界坐标系 → 相机坐标系变换
- 相机坐标系：**+Z 轴朝向场景**（卫星下视约定，与 OpenGL -Z 相反）
- 坐标单位与 mesh.ply 一致（本地坐标，米）

### `masks/` 格式

- `.npy` 文件：`uint8` 数组，形状 `(H, W)`，值 `1` = 有效像素，`0` = 无效
- `.png` 文件：灰度图，`255` = 有效，`0` = 无效
- 文件名须与对应图像文件名相同（仅扩展名不同），如 `JAX_068_001_RGB.npy` 对应 `JAX_068_001_RGB.png`

### `points3D.txt` 格式

COLMAP 标准稀疏点云输出格式：

```
# 3D point list with one line of data per point:
#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
<ID>  <X>  <Y>  <Z>  <R>  <G>  <B>  <ERROR>  <IMAGE_ID> <POINT2D_IDX> ...
```

坐标单位与 mesh.ply 一致（本地坐标，米）。加载时自动过滤重投影误差 > 2.0 px 或轨迹长度 < 3 的点。

---

## 快速开始

### 基础纹理生成（无 FLUX）

```bash
conda activate automesh

python scripts/texture_bake.py \
    --mesh      data/JAX_068/mesh/mesh.ply \
    --mesh_info data/JAX_068/mesh/mesh_info.txt \
    --cameras   data/JAX_068/cameras \
    --images    data/JAX_068/images \
    --masks     data/JAX_068/masks \
    --points3d  data/JAX_068/points3D.txt \
    --output    data/JAX_068/textured \
    --skip_refine
```

### 完整管线（含 FLUX FlowEdit 精炼）

```bash
python scripts/texture_bake.py \
    --mesh      data/JAX_068/mesh/mesh.ply \
    --mesh_info data/JAX_068/mesh/mesh_info.txt \
    --cameras   data/JAX_068/cameras \
    --images    data/JAX_068/images \
    --masks     data/JAX_068/masks \
    --points3d  data/JAX_068/points3D.txt \
    --output    data/JAX_068/textured \
    --use_flux \
    --flux_xformers
```

### 批量处理多场景

```bash
bash run_all.sh
```

### 渲染可视化位姿

```bash
python scripts/render_poses.py \
    --obj    data/JAX_068/textured/mesh_textured.obj \
    --poses  data/JAX_068/poses_for_paper.json \
    --output data/JAX_068/renders_from_poses
```

---

## 主要脚本说明

| 脚本 | 功能 |
|---|---|
| `scripts/texture_bake.py` | 主入口，完整纹理生成管线 |
| `scripts/render_poses.py` | 根据 `poses_for_paper.json` 渲染带纹理 Mesh |
| `scripts/utils/heightfield.py` | NeuS Mesh → 流形 Mesh（体素化 + Marching Cubes + UV 展开） |
| `scripts/utils/render_utils.py` | nvdiffrast 可微分渲染工具（含 Mip Mapping） |
| `scripts/utils/flux_restorer.py` | FLUX FlowEdit 图像编辑封装 |
| `scripts/utils/alignment.py` | Mesh 对齐到 COLMAP 稀疏点云 |
| `scripts/utils/novel_view.py` | 模拟 UAV 新视角相机生成 |
| `scripts/utils/camera_utils.py` | 相机参数加载与 MVP 矩阵构建 |

---

## 完整参数说明

### 几何预处理

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--hf_res` | 256 | 高度场 XY 采样分辨率（同时决定体素 XY 格数） |
| `--voxel_z_res` | 128 | 体素 Z 方向层数，越大墙面越精细 |
| `--simplify_faces` | 0 | Quadric 简化目标面数；0 = 不简化，推荐约为 MC 面数的 20% |
| `--uv_method` | `tutte` | UV 展开：`tutte`（单 island，无顶点复制）或 `xatlas`（多 island） |
| `--save_hf_mesh` | — | 保存流形 Mesh 到 `heightfield.ply` |
| `--save_normals` | — | 渲染并保存所有 COLMAP 视角的法向量图 |

### Mesh 对齐

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--points3d` | None | COLMAP `points3D.txt` 路径；提供后执行 Mesh 对齐 |
| `--align_fft_voxel_size` | 2.0m | FFT 粗对齐体素尺寸 |
| `--align_trim` | 0.3 | Trimmed-Median 每轮丢弃比例 |
| `--align_iters` | 20 | 精对齐最大迭代次数 |

### 纹理优化

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--atlas_size` | 2048 | 最终纹理 atlas 分辨率（须为 2 的幂） |
| `--init_atlas_size` | 512 | 渐进分辨率起始尺寸 |
| `--basic_texture_epochs` | 100 | T_basic 优化总轮数 |
| `--lr` | 0.01 | 纹理优化 Adam 学习率 |
| `--max_mip_level` | 4 | Mip mapping 最大层级 |
| `--masks` | None | 图像掩码目录（.npy 或 .png） |

### 几何优化

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--basic_geometry_epochs` | 0 | 几何优化轮数；0 = 跳过 |
| `--geo_lr` | 1e-3 | 几何优化学习率（米/step） |
| `--lambda_smooth` | 0.01 | Laplacian 平滑权重 |
| `--lambda_reg` | 0.01 | 顶点偏移 L2 正则化权重 |

### UAV 新视角

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--uav_height` | 80m | UAV 相机高度（相对 Mesh 最高点） |
| `--uav_pitch` | 45° | 俯仰角 |
| `--uav_grid` | 60m | 相机水平网格间距 |
| `--uav_margin` | 0m | AABB 外扩边距 |
| `--uav_fov` | 60° | 水平视场角 |
| `--max_novel_cams` | 96 | 最大新视角相机数 |

### FLUX FlowEdit

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--use_flux` | — | 启用 FLUX 精炼 |
| `--flux_model` | `FLUX.1-dev` | HuggingFace 模型 ID 或本地路径 |
| `--flux_weights_dir` | `/dexmal-fa-ltl/weights` | 权重缓存目录 |
| `--flux_res` | 1024 | FLUX 内部处理分辨率 |
| `--flux_T_steps` | 28 | FlowEdit 总时间步数 |
| `--flux_n_max` | 15 | 实际执行的编辑步数 |
| `--flux_n_min` | 0 | 末尾切换 SDEdit 步数（0=全程 ODE） |
| `--flux_n_avg` | 1 | 速度场蒙特卡洛平均次数 |
| `--flux_src_guidance` | 1.5 | 源方向引导强度 |
| `--flux_tar_guidance` | 5.5 | 目标方向引导强度 |
| `--flux_compile` | — | 启用 `torch.compile` 加速 Transformer |
| `--flux_xformers` | — | 启用 xFormers memory-efficient attention |
| `--refine_iters` | 3 | 迭代精炼轮数 |
| `--refine_epochs` | 100 | 每轮精炼优化步数 |

---

## 输出文件说明

```
<output>/
├── aligned_mesh.ply           # 对齐到 COLMAP 点云后的原始 Mesh（--points3d 时生成）
├── colmap_points.ply          # COLMAP 稀疏点云（可视化验证对齐用）
├── marching_cubes_mesh.ply    # Marching Cubes 原始流形 Mesh（简化/UV 之前）
├── heightfield.ply            # 含 UV 的流形 Mesh（--save_hf_mesh 时生成）
├── normal_maps/               # 各视角法向量图（--save_normals 时生成）
│   └── <NAME>_normal.png
├── texture_basic.png          # T_basic 阶段完成的纹理 atlas
├── geometry_optimized.ply     # 几何优化后的 Mesh（--basic_geometry_epochs > 0）
├── texture_final.png          # 最终纹理 atlas
├── mesh_textured.obj          # 最终带纹理 Mesh
├── mesh_textured.mtl          # OBJ 材质文件
├── progress/                  # 各阶段优化进度渲染图（每 20 epoch 一张）
│   ├── T_basic_512/
│   ├── T_basic_1024/
│   ├── T_basic_2048/
│   └── geo_opt/
└── tmp_novel_views/           # 精炼阶段新视角渲染（含 FLUX 前后对比）
    ├── iter1_uav_x000_y000_N.png         # 对比图（左：渲染，右：FLUX 增强）
    ├── iter1_uav_x000_y000_N_before.png  # 精炼前原始渲染
    └── iter1_uav_x000_y000_N_after.png   # FLUX 增强后（用于纹理优化）
```

---

## 详细技术文档

完整的方法描述、公式推导和参数说明见 [docs/pipeline.md](docs/pipeline.md)。
