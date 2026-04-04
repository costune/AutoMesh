"""
基于 FLUX-Schnell 的图像恢复网络封装。

对应论文 2512.07527v2 §3.2 中的图像复原网络 D：
  输入：从当前纹理渲染的退化视角图 I_low  (1, H, W, 3) float32 [0,1]
  输出：经模型增强后的伪真值图 I_target   (1, H, W, 3) float32 [0,1]

实现采用 FluxImg2ImgPipeline（diffusers >= 0.30），在 img2img 模式下
以较低 strength 保留输入几何结构，同时提升纹理质量。

注：论文原始方法使用在合成配对数据上微调的 FLUX-Schnell 确定性映射版本。
若使用该微调权重，将 model_id 指向本地路径即可。
"""

import numpy as np
import torch
from PIL import Image


class FluxRestorer:
    """
    FLUX-Schnell img2img 图像恢复器。

    Parameters
    ----------
    model_id : str
        HuggingFace 模型 ID 或本地权重路径（微调版本）。
    cache_dir : str
        HuggingFace 权重缓存目录。
    flux_resolution : int
        FLUX 处理分辨率（正方形），处理后缩放回原始尺寸。
        较低的值（512/1024）速度更快；论文使用 2048。
    strength : float
        img2img 强度，范围 (0, 1]。
        0.3 → 保留约 70% 输入结构；1.0 → 完全生成。
    num_inference_steps : int
        去噪步数，FLUX-Schnell 推荐 4 步。
    guidance_scale : float
        分类器自由引导强度，FLUX-Schnell 使用 0.0（蒸馏模型）。
    prompt : str
        引导提示词，用于描述目标图像风格。
    seed : int
        固定随机种子，保证每次调用结果一致（确定性输出）。
    device : str
        计算设备（"cuda" 或 "cpu"）。
    """

    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-schnell",
        cache_dir: str = "/dexmal-fa-ltl/weights",
        flux_resolution: int = 1024,
        strength: float = 0.3,
        num_inference_steps: int = 4,
        guidance_scale: float = 0.0,
        prompt: str = "high quality aerial urban view, sharp building details, photorealistic",
        seed: int = 42,
        device: str = "cuda",
    ):
        from diffusers import FluxImg2ImgPipeline

        print(f"[FluxRestorer] 加载模型: {model_id}")
        print(f"[FluxRestorer] 权重缓存目录: {cache_dir}")

        self.pipe = FluxImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            cache_dir=cache_dir,
        ).to(device)
        self.pipe.set_progress_bar_config(disable=True)

        self.flux_resolution = flux_resolution
        self.strength = strength
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.prompt = prompt
        self.seed = seed
        self.device = device

        print(
            f"[FluxRestorer] 初始化完成 | "
            f"resolution={flux_resolution} | strength={strength} | "
            f"steps={num_inference_steps}"
        )

    @torch.no_grad()
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        对退化渲染图进行增强。

        Parameters
        ----------
        image : torch.Tensor  (1, H, W, 3) float32 [0, 1]
            nvdiffrast 渲染的退化视角图。

        Returns
        -------
        torch.Tensor  (1, H, W, 3) float32 [0, 1]
            FLUX 增强后的伪真值图，与输入保持相同空间分辨率。
        """
        # --- 1. Tensor → PIL Image ---
        img_np = image.detach().cpu().squeeze(0).numpy()  # (H, W, 3) float32
        orig_h, orig_w = img_np.shape[:2]
        img_np_uint8 = np.clip(img_np * 255, 0, 255).astype(np.uint8)
        pil_input = Image.fromarray(img_np_uint8)

        # --- 2. 缩放到 FLUX 处理分辨率（正方形，FLUX 偏好 8 的倍数） ---
        flux_size = self._align8(self.flux_resolution)
        pil_resized = pil_input.resize((flux_size, flux_size), Image.LANCZOS)

        # --- 3. FLUX img2img 推理 ---
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        result = self.pipe(
            prompt=self.prompt,
            image=pil_resized,
            strength=self.strength,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            max_sequence_length=256,
            generator=generator,
        )
        pil_output = result.images[0]  # PIL Image，flux_size × flux_size

        # --- 4. 缩放回原始分辨率 ---
        pil_output = pil_output.resize((orig_w, orig_h), Image.LANCZOS)

        # --- 5. PIL Image → Tensor ---
        out_np = np.array(pil_output).astype(np.float32) / 255.0  # (H, W, 3)
        out_tensor = torch.tensor(out_np, device=image.device).unsqueeze(0)  # (1, H, W, 3)

        return out_tensor

    @staticmethod
    def _align8(size: int) -> int:
        """将分辨率对齐到 8 的倍数（FLUX VAE 要求）。"""
        return (size // 8) * 8
