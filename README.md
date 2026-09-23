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
