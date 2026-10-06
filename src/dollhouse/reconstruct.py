"""Stage 4 prototype: a metric point cloud of a photo, split into objects.

Depth Anything 3 estimates metric depth and the camera intrinsics; SAM 3 finds one
mask per object instance for each text prompt. Writes to outputs/scene/<stamp>/:

  points.ply   every pixel as a 3D point, in the photo's colours
  objects.ply  the same points, coloured by object (grey: no object)
  masks.png    the masks over the photo
  scene.npz    depth, intrinsics, masks and labels, for the placement step

Usage: uv run python -m dollhouse.reconstruct PHOTO couch cabinet table ...
"""

import argparse
import gc
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import trimesh
from PIL import Image

from dollhouse import segment
from dollhouse.segment import Instance

# depth_anything_3 is vendored (its package pins Python <= 3.13); see vendor/.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "vendor"))

DEPTH_MODEL_ID = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "scene"

# Depth models blur across object edges, which leaves points floating between an
# object and whatever is behind it; masks are shrunk by this many pixels (at depth
# resolution) before their points are collected.
MASK_ERODE_PX = 2
MIN_ERODED_FRACTION = 0.5

PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
           "#f032e6", "#bcf60c", "#008080", "#9a6324", "#800000", "#000075"]


@dataclass
class Depth:
    depth: np.ndarray       # H, W in metres
    intrinsics: np.ndarray  # 3, 3 for the H, W grid
    image: np.ndarray       # H, W, 3 uint8, the photo at depth resolution


def estimate_depth(image: Image.Image, process_res: int = 504) -> Depth:
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained(DEPTH_MODEL_ID).to("cuda").eval()
    pred = model.inference([image], process_res=process_res)
    del model
    _free()
    return Depth(pred.depth[0], pred.intrinsics[0], pred.processed_images[0])


def label_map(instances: list[Instance], shape: tuple[int, int]) -> np.ndarray:
    """Per-pixel instance index at `shape` (H, W), -1 for none.

    Where masks overlap (e.g. "couch" and "chair" both claim the same sofa), the
    higher-scoring instance wins.
    """
    from scipy.ndimage import binary_erosion

    labels = np.full(shape, -1, dtype=np.int32)
    best = np.zeros(shape, dtype=np.float32)
    for i, inst in enumerate(instances):
        mask = np.array(Image.fromarray(inst.mask).resize(shape[::-1], Image.NEAREST))
        # Thin objects (a lamp pole) would vanish; erode them less.
        for iterations in range(MASK_ERODE_PX, 0, -1):
            eroded = binary_erosion(mask, iterations=iterations)
            if eroded.sum() >= MIN_ERODED_FRACTION * mask.sum():
                mask = eroded
                break
        take = mask & (inst.score > best)
        labels[take] = i
        best[take] = inst.score
    return labels


def backproject(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """H, W, 3 points in the camera frame (x right, y down, z forward), in metres."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    return np.stack([(u - cx) / fx * depth, (v - cy) / fy * depth, depth], axis=-1)


def fit_plane(points: np.ndarray, iterations: int = 500, tolerance: float = 0.02, seed: int = 0):
    """RANSAC plane fit; returns (unit normal, offset) with normal . p + offset = 0,
    and the normal oriented towards the camera at the origin."""
    rng = np.random.default_rng(seed)
    best_inliers = None
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        if np.linalg.norm(normal) < 1e-9:
            continue
        normal /= np.linalg.norm(normal)
        inliers = np.abs((points - a) @ normal) < tolerance
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers
    # Refine on all inliers: the normal is the direction of least variance.
    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    normal = np.linalg.svd(inlier_points - centroid)[2][-1]
    offset = -normal @ centroid
    if offset < 0:
        normal, offset = -normal, -offset
    return normal, offset


def _free() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def _hex_rgb(colour: str) -> tuple[int, int, int]:
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("photo", type=Path)
    parser.add_argument("prompts", nargs="+", help='objects to segment, e.g. couch "coffee table"')
    parser.add_argument("--floor-prompt", default="floor")
    parser.add_argument("--process-res", type=int, default=504, help="DA3 processing resolution")
    parser.add_argument("--threshold", type=float, default=segment.DEFAULT_THRESHOLD,
                        help="SAM 3 instance score threshold")
    args = parser.parse_args()

    image = Image.open(args.photo).convert("RGB")
    out = OUTPUT_DIR / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True)

    t = time.time()
    depth = estimate_depth(image, args.process_res)
    h, w = depth.depth.shape
    print(f"depth: {w}x{h}, {depth.depth.min():.2f}-{depth.depth.max():.2f} m, "
          f"fx={depth.intrinsics[0, 0]:.0f} px ({time.time() - t:.0f} s)")

    t = time.time()
    instances = segment.run(image, [args.floor_prompt, *args.prompts], args.threshold)
    print(f"segment: {len(instances)} instances ({time.time() - t:.0f} s)")

    points = backproject(depth.depth, depth.intrinsics)
    labels = label_map(instances, (h, w))

    floor = [i for i, inst in enumerate(instances) if inst.label == args.floor_prompt]
    floor_points = points[np.isin(labels, floor)]
    normal = offset = None
    if len(floor_points) >= 100:
        normal, offset = fit_plane(floor_points)
        print(f"floor: {len(floor_points)} points, camera {offset:.2f} m above it")
    else:
        print("floor: not found; no plane fitted")

    object_colours = np.full((h, w, 3), 128, dtype=np.uint8)
    for i, inst in enumerate(instances):
        region = labels == i
        colour = _hex_rgb(PALETTE[i % len(PALETTE)])
        object_colours[region] = colour
        if inst.label == args.floor_prompt or not region.any():
            continue
        extent = ""
        if normal is not None:
            heights = points[region] @ normal + offset
            extent = f", top {heights.max():.2f} m above floor"
        print(f"  [{i}] {inst.label} (score {inst.score:.2f}): {region.sum()} points{extent}")

    flat = points.reshape(-1, 3)
    trimesh.PointCloud(flat, colors=depth.image.reshape(-1, 3)).export(out / "points.ply")
    trimesh.PointCloud(flat, colors=object_colours.reshape(-1, 3)).export(out / "objects.ply")

    overlay = np.array(image.resize((w, h)), dtype=np.float32)
    has_label = labels >= 0
    overlay[has_label] = 0.4 * overlay[has_label] + 0.6 * object_colours[has_label]
    Image.fromarray(overlay.astype(np.uint8)).save(out / "masks.png")

    np.savez_compressed(
        out / "scene.npz",
        depth=depth.depth, intrinsics=depth.intrinsics, image=depth.image, labels=labels,
        instance_labels=np.array([inst.label for inst in instances]),
        instance_scores=np.array([inst.score for inst in instances]),
        floor_normal=normal if normal is not None else np.zeros(3),
        floor_offset=offset if offset is not None else 0.0,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
