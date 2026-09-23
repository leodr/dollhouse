#!/usr/bin/env bash
# Fetch TRELLIS.2's CUDA extensions at pinned commits onto machine-local disk,
# patch them for torch >= 2.14, and link them at vendor/ext for `uv sync`.
#
# Run once per machine (/var/tmp is per-host), then build with:
#   source scripts/cuda-env.sh && uv sync
set -euo pipefail

source ~/.config/local-caches.sh
PROJECT="$(cd "$(dirname "$0")/.." && pwd)"
EXT="$LOCAL_CACHE_ROOT/src/trellis-ext"
mkdir -p "$EXT"

fetch() {  # fetch <dir> <url> <commit>
    local dir="$EXT/$1"
    if [ ! -d "$dir/.git" ]; then
        git clone -q "$2" "$dir"
    fi
    git -C "$dir" fetch -q origin "$3" 2>/dev/null || true
    git -C "$dir" checkout -q --force "$3"
    git -C "$dir" submodule update -q --init --recursive
}

fetch CuMesh     https://github.com/JeffreyXiang/CuMesh    12289e1062f0603f2f0d0771b02e1395d247f26f
fetch FlexGEMM   https://github.com/JeffreyXiang/FlexGEMM  6dd94a859c26ee8246888502eada3dd8ad85532e
fetch TRELLIS.2  https://github.com/microsoft/TRELLIS.2    75fbf0183001ed9876c8dbb35de6b68552ee08bd

# torch 2.14's headers refuse to compile below C++20; these pin C++17.
sed -i 's/c++17/c++20/g' \
    "$EXT/CuMesh/setup.py" \
    "$EXT/FlexGEMM/setup.py" \
    "$EXT/TRELLIS.2/o-voxel/setup.py"
sed -i 's/cpp_standard = 17/cpp_standard = 20/' "$EXT/CuMesh/third_party/cubvh/setup.py"

# o-voxel requires CuMesh/FlexGEMM from unpinned git URLs, which conflicts with the
# patched local checkouts above; the project depends on those directly instead.
sed -i '/"cumesh @ git+/d; /"flex_gemm @ git+/d' "$EXT/TRELLIS.2/o-voxel/pyproject.toml"

ln -sfn "$EXT" "$PROJECT/vendor/ext"
echo "Extensions ready in $EXT (linked at vendor/ext)."
