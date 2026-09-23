# Source before `uv sync` when the CUDA extensions need (re)building.
# The system nvcc is CUDA 12.0, which is too old for torch's CUDA 13 and for
# Blackwell (sm_120); use the userspace nvcc wheel on local disk instead.
# nvidia-cuda-cccl supplies the matching CUB/Thrust/libcu++ headers; without it
# nvcc silently falls back to the CUDA 12.0 copies in /usr/include.
source ~/.config/local-caches.sh
if [ ! -d "$LOCAL_CACHE_ROOT/cuda-nvcc/nvidia/cu13/include/cccl" ]; then
    uv pip install -q --python 3.12 --target "$LOCAL_CACHE_ROOT/cuda-nvcc" nvidia-cuda-nvcc nvidia-cuda-cccl
fi
export CUDA_HOME="$LOCAL_CACHE_ROOT/cuda-nvcc/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="12.0"  # SM 12.0 (Blackwell), not a CUDA version
export MAX_JOBS="$(nproc)"
