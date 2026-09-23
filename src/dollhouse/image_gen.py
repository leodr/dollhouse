import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

from dollhouse import models

MODEL_ID = "Qwen/Qwen-Image-2.1"

# Resolutions recommended on the model card (the ~4 MP set).
ASPECT_RATIOS = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}


def _load() -> QwenImage21Pipeline:
    pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    # Text encoder (17.5 GB) + transformer (14.2 GB) + VAE exceed 32 GB together,
    # so keep each component on the GPU only while it runs.
    pipe.enable_model_cpu_offload()
    # An untiled ~4 MP decode peaks at ~27 GiB, so tile it. The VAE's default
    # 256 px tiles leave visible seams every 192 px; 1024 px tiles with a
    # 256 px overlap peak at ~7 GiB and blend over a much wider band.
    pipe.vae.enable_tiling(
        tile_sample_min_height=1024,
        tile_sample_min_width=1024,
        tile_sample_stride_height=768,
        tile_sample_stride_width=768,
    )
    return pipe


def get_pipeline() -> QwenImage21Pipeline:
    return models.acquire("qwen-image", _load)


def generate(
    prompt: str,
    images: list[Image.Image],
    size: tuple[int, int] | None,
    output_resolution: int,
    steps: int,
    seed: int,
    on_step: callable,
) -> Image.Image:
    """Text-to-image, or image editing when `images` is non-empty.

    `size` is (width, height); None derives it from the first input image.
    """
    kwargs = {}
    if size is None:
        kwargs["output_resolution"] = output_resolution
    else:
        kwargs["width"], kwargs["height"] = size
    if images:
        kwargs["image"] = images if len(images) > 1 else images[0]

    def on_step_end(_pipe, step, _t, callback_kwargs):
        on_step(step + 1)
        return callback_kwargs

    try:
        return get_pipeline()(
            prompt=prompt,
            num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            callback_on_step_end=on_step_end,
            **kwargs,
        ).images[0]
    finally:
        torch.cuda.empty_cache()
