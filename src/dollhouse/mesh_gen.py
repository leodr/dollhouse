import os
import sys
from pathlib import Path

import torch
from PIL import Image

from dollhouse import models

# trellis2 is vendored (it has no packaging metadata); see vendor/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor"))
# flash-attn has no wheel for this torch/Python; xformers does. Must be set before
# trellis2 is imported, which reads it at import time.
os.environ.setdefault("ATTN_BACKEND", "xformers")

MODEL_ID = "microsoft/TRELLIS.2-4B"
# The model's pipeline.json names briaai/RMBG-2.0, which is gated. It is a BiRefNet
# fine-tune; the ungated upstream BiRefNet serves the same purpose.
REMBG_MODEL_ID = "ZhengPeng7/BiRefNet"

PIPELINE_TYPES = {"512": "512", "1024": "1024_cascade", "1536": "1536_cascade"}


def _load():
    from trellis2.pipelines import Trellis2ImageTo3DPipeline, rembg

    class OpenBiRefNet(rembg.BiRefNet):
        def __init__(self, model_name: str):
            super().__init__(REMBG_MODEL_ID)
            # Its checkpoint is fp16 and transformers 5 keeps the stored dtype, but
            # trellis2 feeds it fp32 tensors.
            self.model.float()

    rembg.BiRefNet = OpenBiRefNet

    pipe = Trellis2ImageTo3DPipeline.from_pretrained(MODEL_ID)
    # low_vram (the default) keeps weights on the CPU and moves each stage to the
    # GPU only while it runs.
    pipe.cuda()
    return pipe


def get_pipeline():
    return models.acquire("trellis2", _load)


def remove_background(image: Image.Image) -> Image.Image:
    """Cut out and center the subject; images that already have alpha keep it."""
    return get_pipeline().preprocess_image(image)


def generate(
    subject: Image.Image,
    resolution: str,
    seed: int,
    decimation_target: int,
    texture_size: int,
    out_path: Path,
) -> Path:
    """Generate a textured GLB from an image already passed through remove_background."""
    import o_voxel

    pipe = get_pipeline()
    try:
        mesh = pipe.run(
            subject,
            seed=seed,
            preprocess_image=False,
            pipeline_type=PIPELINE_TYPES[resolution],
        )[0]
        mesh.simplify(16777216)  # nvdiffrast's face limit
        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=decimation_target,
            texture_size=texture_size,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            use_tqdm=True,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        glb.export(out_path, extension_webp=True)
        return out_path
    finally:
        torch.cuda.empty_cache()
