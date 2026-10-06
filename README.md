# dollhouse

Gradio app with three tabs:

- **Detection**: Gemma 4 12B. Finds objects in an image and draws their bounding boxes. The prompt is editable.
- **Image**: Qwen-Image 2.1, for text-to-image and image editing with up to 10 input images.
- **3D**: TRELLIS.2-4B, which turns one image into a textured GLB. "Send to 3D tab" moves an Image result over.

Only one model is loaded at a time (they don't fit together in 62 GB RAM / 32 GB VRAM), so the first
generation after switching tabs reloads the model.

## Setup (per machine)

`.venv` and `vendor/ext` are symlinks into `/var/tmp/$USER`, which is local to each host. On a new
`nc-*` machine:

```bash
mkdir -p "$LOCAL_CACHE_ROOT/venvs/dollhouse" && ln -sfn "$LOCAL_CACHE_ROOT/venvs/dollhouse" .venv
./scripts/setup-trellis-ext.sh      # fetch + patch CuMesh, FlexGEMM, o-voxel
source scripts/cuda-env.sh          # CUDA 13 nvcc + CCCL from pip; system nvcc is 12.0
uv sync
uv run hf auth login                # DINOv3 (used by TRELLIS.2) is gated
```

`setup-trellis-ext.sh` makes these patches: C++17 → C++20 (torch 2.14's headers require it), and it
removes o-voxel's unpinned git dependencies on CuMesh/FlexGEMM.

## Run

```bash
uv run dollhouse                    # serves on 127.0.0.1:7860
ssh -L 7860:localhost:7860 <host>   # from your laptop
```

Results are saved to `outputs/` (images) and `outputs/3d/` (GLB files), on the NFS home, which has a quota.

## Photo to 3D scene

```bash
uv run python -m dollhouse.scene PHOTO            # new run in outputs/scene/<timestamp>/
uv run python -m dollhouse.scene PHOTO --run DIR  # resume a run; --refit redoes only placement
```

Gemma 4 lists the objects, SAM 3 masks every instance, Depth Anything 3 estimates metric depth,
Qwen-Image 2.1 extracts each object from a crop with its outline drawn in red (square, 1024 px;
`--rewrite` expands the prompt with Qwen-Image-2.1-PE-T2I and uses its aspect ratio, but its descriptions
override what the crop shows), TRELLIS.2 makes a mesh of each, and each mesh is fitted to its object's depth. The result is
`scene.glb` (Y up, metres, with the photo's camera), and `renders.png` shows it from the photo's camera,
from the side and from above. Every stage caches its output per object in the run folder. A run takes
about an hour for ~25 objects and ~200 MB of quota. Don't use the app's Image or 3D tab during a run:
two copies of Qwen-Image don't fit in RAM.
