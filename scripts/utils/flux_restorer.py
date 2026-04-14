"""
基于 FLUX-Schnell 的图像编辑网络封装（FlowEdit 方法）。

方法：FlowEdit（ODE 流编辑）
  输入：从当前纹理渲染的退化视角图 I_low  (1, H, W, 3) float32 [0,1]
  输出：经模型增强后的伪真值图 I_target   (1, H, W, 3) float32 [0,1]

FlowEdit 通过源提示词（描述退化图）与目标提示词（描述清晰图）的差分速度场
引导图像编辑，比 img2img 更精准地保留原始几何结构（尤其是建筑边缘和纹理）。

参考实现：
  FlowEdit: Inversion-Free Text-Based Image Editing using Pre-trained Flow Models
  https://github.com/fallenshock/FlowEdit
"""

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# 默认提示词
# ---------------------------------------------------------------------------

DEFAULT_SRC_PROMPT = (
    "Satellite image of an urban area with modern and older buildings, roads, "
    "green spaces, and a unique white angular structure. Some areas appear "
    "distorted, with blurring and warping artifacts near edges and trees."
)

DEFAULT_TAR_PROMPT = (
    "Clear satellite image of an urban area with sharp buildings, smooth edges, "
    "and no distortions. Roads, green spaces, and the white angular structure "
    "are crisp, with natural lighting and well-defined textures."
)


# ---------------------------------------------------------------------------
# 调度器辅助函数（移植自 diffusers 内部，保证版本兼容）
# ---------------------------------------------------------------------------

def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """计算 FLUX 时间偏移参数 mu（线性插值）。"""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, mu=None):
    """
    从调度器获取时间步序列。

    支持通过 sigmas 数组直接设置（FLUX FlowMatch 调度器所需）。
    """
    kwargs = {}
    if mu is not None:
        kwargs["mu"] = mu
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


def _scale_noise(scheduler, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    """
    流匹配噪声混合：x_t = (1 - sigma) * x_0 + sigma * noise

    sigma 从 scheduler 当前 step_index 读取。
    """
    sigma = scheduler.sigmas[scheduler.step_index]
    return (1.0 - sigma) * x + sigma * noise


# ---------------------------------------------------------------------------
# FLUX Transformer 速度场计算
# ---------------------------------------------------------------------------

@torch.no_grad()
def _calc_v_flux(
    pipe,
    latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor,
    guidance: torch.Tensor,
    text_ids: torch.Tensor,
    latent_image_ids: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    """
    调用 FLUX Transformer 计算当前时刻的速度场 V。

    Parameters
    ----------
    latents          : (B, seq_len, C) packed latents
    t                : scalar 或 (1,) 时间步（来自 scheduler.timesteps，范围 ~[0,1000]）

    Returns
    -------
    (B, seq_len, C) 速度场预测
    """
    t_in = t.expand(latents.shape[0]) if t.ndim == 0 else t
    noise_pred = pipe.transformer(
        hidden_states=latents,
        timestep=t_in / 1000.0,          # FLUX Transformer 接受 [0,1] 范围
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]
    return noise_pred


# ---------------------------------------------------------------------------
# FlowEdit 主循环
# ---------------------------------------------------------------------------

@torch.no_grad()
def _flow_edit_flux(
    pipe,
    scheduler,
    x_src: torch.Tensor,
    src_prompt: str,
    tar_prompt: str,
    T_steps: int = 28,
    n_avg: int = 1,
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 24,
) -> torch.Tensor:
    """
    FlowEdit ODE 流编辑核心函数。

    算法：
      - 在前 (n_max - n_min) 步：以源/目标速度场之差 dV = V_tar - V_src 引导 ODE
      - 在最后 n_min 步：切换为标准 SDEDIT 采样（仅用目标提示词）

    Parameters
    ----------
    x_src            : (B, C, H_lat, W_lat) VAE 编码后的源 latents（未 pack）
    src_prompt       : 描述退化图的提示词
    tar_prompt       : 描述目标清晰图的提示词
    T_steps          : 总时间步数
    n_avg            : 速度场估计的平均次数（n_avg > 1 降低方差，但更慢）
    src_guidance_scale : 源方向的引导强度
    tar_guidance_scale : 目标方向的引导强度
    n_min            : 最后使用普通 SDEDIT 的步数（0 = 全程 ODE）
    n_max            : 开始编辑的步数（前 T_steps-n_max 步跳过）

    Returns
    -------
    (B, C, H_lat, W_lat) 编辑后的 latents（未 pack，与 x_src 形状相同）
    """
    device = x_src.device
    orig_h = x_src.shape[2] * pipe.vae_scale_factor // 2
    orig_w = x_src.shape[3] * pipe.vae_scale_factor // 2
    num_channels_latents = pipe.transformer.config.in_channels // 4

    # --- 1. 准备 latents 和 image_ids ---
    x_src, latent_src_image_ids = pipe.prepare_latents(
        batch_size=x_src.shape[0],
        num_channels_latents=num_channels_latents,
        height=orig_h,
        width=orig_w,
        dtype=x_src.dtype,
        device=device,
        generator=None,
        latents=x_src,
    )
    x_src_packed = pipe._pack_latents(
        x_src, x_src.shape[0], num_channels_latents, x_src.shape[2], x_src.shape[3]
    )
    latent_tar_image_ids = latent_src_image_ids

    # --- 2. 时间步调度 ---
    sigmas = np.linspace(1.0, 1.0 / T_steps, T_steps)
    image_seq_len = x_src_packed.shape[1]
    mu = _calculate_shift(
        image_seq_len,
        scheduler.config.base_image_seq_len,
        scheduler.config.max_image_seq_len,
        scheduler.config.base_shift,
        scheduler.config.max_shift,
    )
    timesteps, T_steps = _retrieve_timesteps(scheduler, T_steps, device, sigmas=sigmas, mu=mu)

    # --- 3. 编码提示词 ---
    src_prompt_embeds, src_pooled_embeds, src_text_ids = pipe.encode_prompt(
        prompt=src_prompt, prompt_2=None, device=device,
    )
    pipe._guidance_scale = tar_guidance_scale
    tar_prompt_embeds, tar_pooled_embeds, tar_text_ids = pipe.encode_prompt(
        prompt=tar_prompt, prompt_2=None, device=device,
    )

    # --- 4. 引导向量（仅 FLUX-dev 系列需要，Schnell 不使用） ---
    if pipe.transformer.config.guidance_embeds:
        src_guidance = torch.tensor([src_guidance_scale], device=device).expand(x_src_packed.shape[0])
        tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(x_src_packed.shape[0])
    else:
        src_guidance = None
        tar_guidance = None

    # --- 5. 初始化编辑轨迹 z_edit = x_src ---
    zt_edit = x_src_packed.clone()
    xt_tar  = None   # SDEDIT 阶段用

    for i, t in enumerate(timesteps):
        if T_steps - i > n_max:
            continue   # 跳过前面几步（热身）

        scheduler._init_step_index(t)
        t_i   = scheduler.sigmas[scheduler.step_index]
        t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) - 1 else t_i

        if T_steps - i > n_min:
            # ── ODE 编辑阶段：差分速度场 ──
            V_delta_avg = torch.zeros_like(x_src_packed)

            for _ in range(n_avg):
                fwd_noise = torch.randn_like(x_src_packed)

                # 对应时刻的带噪 src/tar latents
                zt_src = (1.0 - t_i) * x_src_packed + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src_packed

                Vt_src = _calc_v_flux(
                    pipe, zt_src, src_prompt_embeds, src_pooled_embeds,
                    src_guidance, src_text_ids, latent_src_image_ids, t,
                )
                Vt_tar = _calc_v_flux(
                    pipe, zt_tar, tar_prompt_embeds, tar_pooled_embeds,
                    tar_guidance, tar_text_ids, latent_tar_image_ids, t,
                )
                V_delta_avg += (1.0 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_im1 - t_i) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        else:
            # ── SDEDIT 阶段：纯目标提示词标准采样 ──
            if T_steps - i == n_min:
                # 切换点：以 zt_edit 为基础初始化带噪 latents
                fwd_noise = torch.randn_like(x_src_packed)
                xt_src = _scale_noise(scheduler, x_src_packed, fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            Vt_tar = _calc_v_flux(
                pipe, xt_tar, tar_prompt_embeds, tar_pooled_embeds,
                tar_guidance, tar_text_ids, latent_tar_image_ids, t,
            )
            xt_tar = xt_tar.to(torch.float32)
            xt_tar = xt_tar + (t_im1 - t_i) * Vt_tar
            xt_tar = xt_tar.to(Vt_tar.dtype)

    out_packed = zt_edit if n_min == 0 else xt_tar
    return pipe._unpack_latents(out_packed, orig_h, orig_w, pipe.vae_scale_factor)


# ---------------------------------------------------------------------------
# 公开的 FluxRestorer 类
# ---------------------------------------------------------------------------

class FluxRestorer:
    """
    基于 FlowEdit 的 FLUX-Schnell 图像编辑器。

    使用源提示词（描述退化渲染图）和目标提示词（描述清晰图）的速度场差分，
    以 ODE 流编辑方式将退化图引导至清晰图，同时保留原始几何结构。

    Parameters
    ----------
    model_id         : HuggingFace 模型 ID 或本地权重路径
    cache_dir        : 权重缓存目录
    flux_resolution  : 内部处理分辨率（正方形），处理后缩放回原始尺寸
    src_prompt       : 描述输入退化图的提示词
    tar_prompt       : 描述目标清晰图的提示词
    T_steps          : FlowEdit 总时间步数（默认 28）
    n_avg            : 速度场蒙特卡洛平均次数（1 最快，>1 更稳定）
    src_guidance     : 源方向引导强度（默认 1.5）
    tar_guidance     : 目标方向引导强度（默认 5.5）
    n_min            : 末尾切换为 SDEDIT 的步数（0 = 全程 ODE）
    n_max            : 开始编辑的步数（默认 24）
    seed             : 随机种子（固定结果）
    device           : 计算设备
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        cache_dir: str = "/dexmal-fa-ltl/weights",
        flux_resolution: int = 1024,
        src_prompt: str = DEFAULT_SRC_PROMPT,
        tar_prompt: str = DEFAULT_TAR_PROMPT,
        T_steps: int = 28,
        n_avg: int = 1,
        src_guidance: float = 1.5,
        tar_guidance: float = 5.5,
        n_min: int = 0,
        n_max: int = 24,
        seed: int = 42,
        device: str = "cuda",
    ):
        from diffusers import FluxPipeline
        from diffusers import FlowMatchEulerDiscreteScheduler

        print(f"[FluxRestorer] 加载模型（FlowEdit 模式）: {model_id}")
        print(f"[FluxRestorer] 权重缓存目录: {cache_dir}")

        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)

        # 使用独立的调度器副本（避免影响 pipe 内部状态）
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.flux_resolution = self._align16(flux_resolution)
        self.src_prompt  = src_prompt
        self.tar_prompt  = tar_prompt
        self.T_steps     = T_steps
        self.n_avg       = n_avg
        self.src_guidance = src_guidance
        self.tar_guidance = tar_guidance
        self.n_min       = n_min
        self.n_max       = n_max
        self.seed        = seed
        self.device      = device

        print(
            f"[FluxRestorer] 初始化完成 | resolution={self.flux_resolution} | "
            f"T_steps={T_steps} | n_avg={n_avg} | "
            f"src_guidance={src_guidance} | tar_guidance={tar_guidance}"
        )
        print(f"  src_prompt: {src_prompt[:60]}...")
        print(f"  tar_prompt: {tar_prompt[:60]}...")

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        对退化渲染图进行 FlowEdit 编辑增强。

        Parameters
        ----------
        image : torch.Tensor  (1, H, W, 3) float32 [0, 1]
            nvdiffrast 渲染的退化视角图。

        Returns
        -------
        torch.Tensor  (1, H, W, 3) float32 [0, 1]
            FlowEdit 增强后的伪真值图，与输入保持相同空间分辨率。
        """
        img_np = image.detach().cpu().squeeze(0).numpy()   # (H, W, 3) float32
        orig_h, orig_w = img_np.shape[:2]

        # --- 1. 缩放到 FLUX 处理分辨率 ---
        img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        pil_input = Image.fromarray(img_uint8).resize(
            (self.flux_resolution, self.flux_resolution), Image.LANCZOS
        )

        # --- 2. PIL → VAE 编码 latents ---
        # FLUX VAE 输入需归一化到 [-1, 1]，NCHW 格式
        x_np = np.array(pil_input).astype(np.float32) / 127.5 - 1.0   # (H, W, 3)
        x_t  = torch.tensor(x_np, dtype=torch.bfloat16, device=self.device)
        x_t  = x_t.permute(2, 0, 1).unsqueeze(0)   # (1, 3, H, W)

        # 编码并缩放（FLUX VAE 的 shift/scale 因子）
        vae = self.pipe.vae
        latents = vae.encode(x_t).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

        # --- 3. 固定随机种子 ---
        torch.manual_seed(self.seed)

        # --- 4. FlowEdit 编辑 ---
        edited_latents = _flow_edit_flux(
            pipe=self.pipe,
            scheduler=self.scheduler,
            x_src=latents,
            src_prompt=self.src_prompt,
            tar_prompt=self.tar_prompt,
            T_steps=self.T_steps,
            n_avg=self.n_avg,
            src_guidance_scale=self.src_guidance,
            tar_guidance_scale=self.tar_guidance,
            n_min=self.n_min,
            n_max=self.n_max,
        )

        # --- 5. 解码 latents → 图像 ---
        edited_latents = edited_latents / vae.config.scaling_factor + vae.config.shift_factor
        decoded = vae.decode(edited_latents.to(vae.dtype)).sample   # (1, 3, H, W) ∈ [-1, 1]

        # --- 6. 后处理：[-1,1] → [0,1] → 缩放回原始分辨率 → Tensor ---
        decoded_np = decoded.float().squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        decoded_np = np.clip((decoded_np + 1.0) / 2.0, 0.0, 1.0)
        pil_out = Image.fromarray((decoded_np * 255).astype(np.uint8))

        if (orig_h, orig_w) != (self.flux_resolution, self.flux_resolution):
            pil_out = pil_out.resize((orig_w, orig_h), Image.LANCZOS)

        out_np = np.array(pil_out).astype(np.float32) / 255.0
        return torch.tensor(out_np, device=image.device).unsqueeze(0)   # (1, H, W, 3)

    @staticmethod
    def _align16(size: int) -> int:
        """将分辨率对齐到 16 的倍数（FLUX VAE 及 patchify 要求）。"""
        return (size // 16) * 16
