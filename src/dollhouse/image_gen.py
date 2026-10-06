from typing import NamedTuple

import torch
from diffusers import QwenImage21Pipeline
from PIL import Image

from dollhouse import models

MODEL_ID = "Qwen/Qwen-Image-2.1"
DEFAULT_PROMPT = (
    "Generate a three-quarter view of the couch from this image as a standalone picture "
    "with a transparent background."
)

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


class Encoded(NamedTuple):
    """The text encoder's output for one prompt and its condition images, kept on the CPU."""

    prompt_embeds: torch.Tensor
    prompt_embeds_mask: torch.Tensor | None
    image_pad_mask: torch.Tensor


class _Stop(Exception):
    """Ends a call of the encoder-only pipeline once the prompt is encoded."""

    def __init__(self, encoded: Encoded):
        super().__init__()
        self.encoded = encoded


class _Pipeline(QwenImage21Pipeline):
    """Qwen-Image 2.1 loaded either with only the text encoder or without it.

    The text encoder (16.3 GiB) and the transformer (13.3 GiB) do not fit in VRAM together, and
    CPU offload copies both to the GPU on every call. So encode() loads only the text encoder and
    generate() everything else, and models.acquire() keeps one of the two loaded at a time.
    Both run the library's __call__ unchanged, so the condition images are resized as in one call:
    with the text encoder, the call stops after encode_prompt() and returns its output through
    _Stop; without it, encode_prompt() returns `encoded` instead.
    """

    encoded: Encoded | None = None

    def encode_prompt(self, *args, **kwargs):
        if self.text_encoder is not None:
            outputs = super().encode_prompt(*args, **kwargs)
            raise _Stop(Encoded(*(None if t is None else t.cpu() for t in outputs)))
        if self.encoded is None:
            raise RuntimeError("Loaded without the text encoder: pass encode()'s result to generate()")
        return tuple(None if t is None else t.to(self._execution_device) for t in self.encoded)


def _load_encoder() -> _Pipeline:
    return _Pipeline.from_pretrained(MODEL_ID, transformer=None, vae=None, dtype=torch.bfloat16).to("cuda")


def _load_denoiser() -> _Pipeline:
    pipe = _Pipeline.from_pretrained(MODEL_ID, text_encoder=None, dtype=torch.bfloat16).to("cuda")
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


def _inputs(prompt: str, images: list[Image.Image], size: tuple[int, int] | None, output_resolution: int) -> dict:
    """The pipeline arguments that decide what the text encoder sees."""
    kwargs = {"prompt": prompt}
    if size is None:
        kwargs["output_resolution"] = output_resolution
    else:
        kwargs["width"], kwargs["height"] = size
    if images:
        kwargs["image"] = images if len(images) > 1 else images[0]
    return kwargs


def encode(
    prompt: str,
    images: list[Image.Image],
    size: tuple[int, int] | None,
    output_resolution: int,
) -> Encoded:
    """Runs the text encoder for a later generate() call with the same arguments."""
    pipe = models.acquire("qwen-image-encoder", _load_encoder)
    try:
        pipe(**_inputs(prompt, images, size, output_resolution))
    except _Stop as stop:
        return stop.encoded
    raise RuntimeError("the pipeline returned without encoding the prompt")


def generate(
    prompt: str,
    images: list[Image.Image],
    size: tuple[int, int] | None,
    output_resolution: int,
    steps: int,
    seed: int,
    on_step: callable,
    encoded: Encoded | None = None,
) -> Image.Image:
    """Text-to-image, or image editing when `images` is non-empty.

    `size` is (width, height); None derives it from the first input image.
    `encoded` is encode()'s result for the same prompt, images, size and output_resolution;
    without it, the prompt is encoded first. Encoding many prompts before generating any
    loads each model once instead of once per image.
    """
    if encoded is None:
        encoded = encode(prompt, images, size, output_resolution)

    def on_step_end(_pipe, step, _t, callback_kwargs):
        on_step(step + 1)
        return callback_kwargs

    pipe = models.acquire("qwen-image", _load_denoiser)
    pipe.encoded = encoded
    try:
        return pipe(
            num_inference_steps=steps,
            generator=torch.Generator("cuda").manual_seed(seed),
            callback_on_step_end=on_step_end,
            **_inputs(prompt, images, size, output_resolution),
        ).images[0]
    finally:
        pipe.encoded = None
        torch.cuda.empty_cache()
