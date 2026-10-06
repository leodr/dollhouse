"""Keeps at most one model in memory at a time.

Qwen-Image (its text encoder ~17 GB or its transformer ~14 GB of VRAM; image_gen
loads them as two models), TRELLIS.2 (~20 GB) and Gemma 4 12B (~24 GB
of VRAM) do not fit together in the workstation's 62 GB of RAM / 32 GB of VRAM, so
switching tabs unloads whichever model was loaded before.
"""

import gc
from typing import Callable

import torch

_name: str | None = None
_pipe = None


def acquire(name: str, load: Callable[[], object]):
    global _name, _pipe
    if _name != name:
        release()
        _pipe = load()
        _name = name
    return _pipe


def release() -> None:
    global _name, _pipe
    if _pipe is None:
        return
    if hasattr(_pipe, "remove_all_hooks"):  # diffusers CPU-offload hooks hold references
        _pipe.remove_all_hooks()
    _name = _pipe = None
    gc.collect()
    torch.cuda.empty_cache()
