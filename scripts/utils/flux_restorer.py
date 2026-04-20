"""
基于 FLUX-Schnell 的图像编辑网络封装（FlowEdit 方法）。

方法：FlowEdit（ODE 流编辑）
  输入：从当前纹理渲染的退化视角图 I_low  (1, H, W, 3) float32 [0,1]
  输出：经模型增强后的伪真值图 I_target   (1, H, W, 3) float32 [0,1]

加速优化：
  1. 文本嵌入缓存 ── 模型加载后立即计算 src/tar 嵌入，多张图共享（避免重复调用 T5-XXL）
  2. 时间步缓存  ── 分辨率固定时 timesteps 只计算一次
  3. torch.compile ── 可选，对 Transformer 前向做图优化（首次调用有预热开销）
  4. xFormers     ── 可选，启用 memory-efficient attention 降低显存并加速

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
    "Clear satellite image of an urban area with sharp buildings, smooth edges and no distortions. "
    "Roads, green spaces, and the white angular structure are crisp, with natural lighting, well-defined textures and smooth color transition. "
    "The windows on the side facade of the building have distinct and regular shapes."
)


# ---------------------------------------------------------------------------
# 调度器辅助函数
# ---------------------------------------------------------------------------

def _calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def _retrieve_timesteps(scheduler, num_inference_steps, device, sigmas=None, mu=None):
    kwargs = {}
    if mu is not None:
        kwargs["mu"] = mu
    if sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
    return scheduler.timesteps, len(scheduler.timesteps)


def _scale_noise(scheduler, x: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
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
    t_in = t.expand(latents.shape[0]) if t.ndim == 0 else t
    return pipe.transformer(
        hidden_states=latents,
        timestep=t_in / 1000.0,
        guidance=guidance,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=latent_image_ids,
        return_dict=False,
    )[0]


# ---------------------------------------------------------------------------
# FlowEdit 主循环（接受预计算的嵌入和时间步）
# ---------------------------------------------------------------------------

@torch.no_grad()
def _flow_edit_flux(
    pipe,
    scheduler,
    x_src: torch.Tensor,
    # 预计算的文本嵌入（在 FluxRestorer.__init__ 中一次性计算，多张图复用）
    src_prompt_embeds: torch.Tensor,
    src_pooled_embeds: torch.Tensor,
    src_text_ids: torch.Tensor,
    tar_prompt_embeds: torch.Tensor,
    tar_pooled_embeds: torch.Tensor,
    tar_text_ids: torch.Tensor,
    # 预计算的时间步
    timesteps: torch.Tensor,
    T_steps: int,
    # 超参数
    src_guidance_scale: float = 1.5,
    tar_guidance_scale: float = 5.5,
    n_min: int = 0,
    n_max: int = 15,
    n_avg: int = 1,
) -> torch.Tensor:
    """
    FlowEdit ODE 流编辑。文本嵌入和时间步由调用方预计算传入，多张图可复用。

    Returns
    -------
    (B, C, H_lat, W_lat) 编辑后 latents（未 pack，与 x_src 形状相同）
    """
    device = x_src.device
    actual_vae_scale = 2 ** (len(pipe.vae.config.block_out_channels) - 1)
    orig_h = x_src.shape[2] * actual_vae_scale
    orig_w = x_src.shape[3] * actual_vae_scale
    num_channels_latents = pipe.transformer.config.in_channels // 4

    # --- 1. 准备 latents 和 image_ids ---
    result = pipe.prepare_latents(
        batch_size=x_src.shape[0],
        num_channels_latents=num_channels_latents,
        height=orig_h,
        width=orig_w,
        dtype=x_src.dtype,
        device=device,
        generator=None,
        latents=x_src,
    )
    if isinstance(result, (tuple, list)):
        x_src, latent_image_ids = result[0], result[1]
    else:
        x_src = result
        latent_image_ids = pipe._prepare_latent_image_ids(
            x_src.shape[0], x_src.shape[2] // 2, x_src.shape[3] // 2, device, x_src.dtype,
        )

    h_lat, w_lat = x_src.shape[2], x_src.shape[3]
    x_src_packed = pipe._pack_latents(x_src, x_src.shape[0], num_channels_latents, h_lat, w_lat)

    # --- 2. 引导向量（FLUX-dev 需要；Schnell guidance_embeds=False 则为 None） ---
    if pipe.transformer.config.guidance_embeds:
        B = x_src_packed.shape[0]
        src_guidance = torch.tensor([src_guidance_scale], device=device).expand(B)
        tar_guidance = torch.tensor([tar_guidance_scale], device=device).expand(B)
    else:
        src_guidance = tar_guidance = None

    # --- 3. ODE 主循环 ---
    zt_edit = x_src_packed.clone()
    xt_tar  = None

    for i, t in enumerate(timesteps):
        if T_steps - i > n_max:
            continue

        scheduler._init_step_index(t)
        t_i   = scheduler.sigmas[scheduler.step_index]
        t_im1 = scheduler.sigmas[scheduler.step_index + 1] if i < len(timesteps) - 1 else t_i

        if T_steps - i > n_min:
            V_delta_avg = torch.zeros_like(x_src_packed)
            for _ in range(n_avg):
                fwd_noise = torch.randn_like(x_src_packed)
                zt_src = (1.0 - t_i) * x_src_packed + t_i * fwd_noise
                zt_tar = zt_edit + zt_src - x_src_packed

                Vt_src = _calc_v_flux(pipe, zt_src, src_prompt_embeds, src_pooled_embeds,
                                      src_guidance, src_text_ids, latent_image_ids, t)
                Vt_tar = _calc_v_flux(pipe, zt_tar, tar_prompt_embeds, tar_pooled_embeds,
                                      tar_guidance, tar_text_ids, latent_image_ids, t)
                V_delta_avg += (1.0 / n_avg) * (Vt_tar - Vt_src)

            zt_edit = (zt_edit.float() + (t_im1 - t_i) * V_delta_avg).to(V_delta_avg.dtype)

        else:
            if T_steps - i == n_min:
                fwd_noise = torch.randn_like(x_src_packed)
                xt_src = _scale_noise(scheduler, x_src_packed, fwd_noise)
                xt_tar = zt_edit + xt_src - x_src_packed

            Vt_tar = _calc_v_flux(pipe, xt_tar, tar_prompt_embeds, tar_pooled_embeds,
                                  tar_guidance, tar_text_ids, latent_image_ids, t)
            xt_tar = (xt_tar.float() + (t_im1 - t_i) * Vt_tar).to(Vt_tar.dtype)

    out_packed = zt_edit if n_min == 0 else xt_tar
    return pipe._unpack_latents(out_packed, orig_h, orig_w, pipe.vae_scale_factor)


# ---------------------------------------------------------------------------
# 公开的 FluxRestorer 类
# ---------------------------------------------------------------------------

class FluxRestorer:
    """
    基于 FlowEdit 的 FLUX 图像编辑器。

    加速策略（初始化时自动应用）：
      - 文本嵌入缓存：加载后立即计算 src/tar 嵌入，整个推理过程只算一次
      - 时间步缓存：分辨率固定时 timesteps 复用
      - torch.compile（可选）：对 Transformer 做图优化，首次调用有 ~30s 预热
      - xFormers（可选）：memory-efficient attention

    Parameters
    ----------
    model_id         : HuggingFace 模型 ID 或本地权重路径
    cache_dir        : 权重缓存目录
    flux_resolution  : 内部处理分辨率（正方形）
    src_prompt       : 描述输入退化图的提示词
    tar_prompt       : 描述目标清晰图的提示词
    T_steps          : FlowEdit 总时间步数（默认 28）
    n_avg            : 速度场蒙特卡洛平均次数（1 最快，>1 更稳定）
    src_guidance     : 源方向引导强度（默认 1.5）
    tar_guidance     : 目标方向引导强度（默认 5.5）
    n_min            : 末尾切换为 SDEDIT 的步数（默认 0 = 全程 ODE）
    n_max            : 开始编辑的步数（默认 15）
    use_compile      : 是否用 torch.compile 优化 Transformer（默认 False）
    use_xformers     : 是否启用 xFormers memory-efficient attention（默认 False）
    seed             : 随机种子
    device           : 计算设备
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        cache_dir: str = "/dexmal-fa-ltl/weights",
        flux_resolution: int = 1024,
        src_prompt: str = DEFAULT_SRC_PROMPT,
        tar_prompt: str = DEFAULT_TAR_PROMPT,
        T_steps: int = 28,
        n_avg: int = 1,
        src_guidance: float = 1.5,
        tar_guidance: float = 5.5,
        n_min: int = 0,
        n_max: int = 15,
        use_compile: bool = False,
        use_xformers: bool = False,
        seed: int = 42,
        device: str = "cuda",
    ):
        from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler

        print(f"[FluxRestorer] 加载模型（FlowEdit 模式）: {model_id}")

        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)

        # ── 可选加速：xFormers ──
        if use_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("[FluxRestorer] xFormers memory-efficient attention 已启用")
            except Exception as e:
                print(f"[FluxRestorer] xFormers 不可用，跳过: {e}")

        # ── 可选加速：torch.compile ──
        if use_compile:
            print("[FluxRestorer] torch.compile 编译 Transformer（首次调用约需 30s 预热）...")
            self.pipe.transformer = torch.compile(
                self.pipe.transformer,
                mode="reduce-overhead",
                fullgraph=True,
            )

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
            self.pipe.scheduler.config
        )

        self.flux_resolution = self._align16(flux_resolution)
        self.T_steps      = T_steps
        self.n_avg        = n_avg
        self.src_guidance = src_guidance
        self.tar_guidance = tar_guidance
        self.n_min        = n_min
        self.n_max        = n_max
        self.seed         = seed
        self.device       = device

        # ── 加速①：预计算文本嵌入（T5-XXL 只调用一次） ──
        print("[FluxRestorer] 预计算文本嵌入（src + tar）...")
        with torch.no_grad():
            self._src_embeds, self._src_pooled, self._src_text_ids = \
                self.pipe.encode_prompt(prompt=src_prompt, prompt_2=None, device=device)
            self.pipe._guidance_scale = tar_guidance
            self._tar_embeds, self._tar_pooled, self._tar_text_ids = \
                self.pipe.encode_prompt(prompt=tar_prompt, prompt_2=None, device=device)

        # ── 加速②：预计算时间步（分辨率固定时每次相同） ──
        self._timesteps, self._T_steps_actual = self._precompute_timesteps()

        print(
            f"[FluxRestorer] 初始化完成 | resolution={self.flux_resolution} | "
            f"T_steps={T_steps}(实际 {self.n_max} 步有效) | n_avg={n_avg} | "
            f"compile={use_compile} | xformers={use_xformers}"
        )
        print(f"  src: {src_prompt[:70]}...")
        print(f"  tar: {tar_prompt[:70]}...")

    def _precompute_timesteps(self):
        """预计算固定分辨率下的时间步序列。"""
        # 用虚拟 latent 尺寸推算 image_seq_len
        lat_size = self.flux_resolution // 8         # VAE 8x 压缩
        packed_size = (lat_size // 2) ** 2           # _pack_latents 再 2x 压缩
        mu = _calculate_shift(
            packed_size,
            self.scheduler.config.base_image_seq_len,
            self.scheduler.config.max_image_seq_len,
            self.scheduler.config.base_shift,
            self.scheduler.config.max_shift,
        )
        sigmas = np.linspace(1.0, 1.0 / self.T_steps, self.T_steps)
        timesteps, T_actual = _retrieve_timesteps(
            self.scheduler, self.T_steps, self.device, sigmas=sigmas, mu=mu
        )
        return timesteps, T_actual

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        对退化渲染图进行 FlowEdit 编辑增强。

        Parameters
        ----------
        image : (1, H, W, 3) float32 [0, 1]

        Returns
        -------
        (1, H, W, 3) float32 [0, 1]
        """
        img_np = image.detach().cpu().squeeze(0).numpy()
        orig_h, orig_w = img_np.shape[:2]

        # --- 1. 缩放 + VAE 编码 ---
        img_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        pil_input = Image.fromarray(img_uint8).resize(
            (self.flux_resolution, self.flux_resolution), Image.LANCZOS
        )
        x_np = np.array(pil_input).astype(np.float32) / 127.5 - 1.0
        x_t  = torch.tensor(x_np, dtype=torch.bfloat16, device=self.device)
        x_t  = x_t.permute(2, 0, 1).unsqueeze(0)   # (1, 3, H, W)

        vae     = self.pipe.vae
        latents = vae.encode(x_t).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor

        # --- 2. FlowEdit（复用预计算的嵌入和时间步） ---
        torch.manual_seed(self.seed)
        edited_latents = _flow_edit_flux(
            pipe=self.pipe,
            scheduler=self.scheduler,
            x_src=latents,
            src_prompt_embeds=self._src_embeds,
            src_pooled_embeds=self._src_pooled,
            src_text_ids=self._src_text_ids,
            tar_prompt_embeds=self._tar_embeds,
            tar_pooled_embeds=self._tar_pooled,
            tar_text_ids=self._tar_text_ids,
            timesteps=self._timesteps,
            T_steps=self._T_steps_actual,
            src_guidance_scale=self.src_guidance,
            tar_guidance_scale=self.tar_guidance,
            n_min=self.n_min,
            n_max=self.n_max,
            n_avg=self.n_avg,
        )

        # --- 3. VAE 解码 ---
        edited_latents = edited_latents / vae.config.scaling_factor + vae.config.shift_factor
        decoded = vae.decode(edited_latents.to(vae.dtype)).sample   # (1,3,H,W) ∈[-1,1]

        decoded_np = decoded.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        decoded_np = np.clip((decoded_np + 1.0) / 2.0, 0.0, 1.0)
        pil_out = Image.fromarray((decoded_np * 255).astype(np.uint8))

        if (orig_h, orig_w) != (self.flux_resolution, self.flux_resolution):
            pil_out = pil_out.resize((orig_w, orig_h), Image.LANCZOS)

        out_np = np.array(pil_out).astype(np.float32) / 255.0
        return torch.tensor(out_np, device=image.device).unsqueeze(0)

    @staticmethod
    def _align16(size: int) -> int:
        return (size // 16) * 16
