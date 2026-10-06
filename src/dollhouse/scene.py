"""Photo to 3D scene, one stage at a time.

  1. inventory  Gemma 4 lists the object types in the photo
  2. masks      SAM 3 masks every instance of each type, and the floor
  3. depth      Depth Anything 3 estimates metric depth and the camera intrinsics
  4. crops      per instance: its bounding box, padded, with the mask outlined in red
  5. prompts    the extraction prompt for each crop; with --rewrite, Qwen-Image-2.1-PE-T2I
                expands it and picks the aspect ratio (off by default: its detailed
                descriptions override what the crop shows)
  6. extract    Qwen-Image 2.1 turns each crop into a standalone RGBA image
  7. meshes     TRELLIS.2 turns each image into a textured GLB
  8. place      each mesh is fitted to its instance's points from the depth map
  9. assemble   the placed meshes, a floor and the camera go into scene.glb

Every stage writes into the run folder and is skipped when its output exists, so
`--run DIR` resumes a run, and deleting a stage's files reruns it. Stages run for
all instances before the next begins: switching models costs 20-36 GB of loading.

Usage: uv run python -m dollhouse.scene PHOTO [--run DIR]
"""

import argparse
import gc
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import trimesh
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion, label
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from dollhouse import detect, image_gen, mesh_gen, models, reconstruct, render, rewrite, segment

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "scene"

FLOOR_PROMPT = "floor"
EXTRACT_PROMPT = (
    "Generate a three-quarter view of the {name} outlined in red in this image "
    "as a standalone picture with a transparent background."
)
EXTRACT_SIZE = (1024, 1024)  # without --rewrite
EXTRACT_PIXELS = 1024 * 1024  # with --rewrite, at the rewriter's aspect ratio
# The rewriter describes the background as a transparency checkerboard and the red
# outline as part of the picture, and Qwen-Image then draws both. Sentences about
# either are dropped, and REWRITE_SUFFIX states them correctly.
REWRITE_DROP = re.compile(r"outline|checker|transparen|backdrop|background|shadow|\bfloor\b|grid", re.IGNORECASE)
REWRITE_SUFFIX = (" It is the {name} outlined in red in the input image, shown on its own without the red outline,"
                  " without any other objects, floor or shadow, on a fully transparent background.")
OUTLINE_RGB = (255, 0, 0)

# Extraction checks: failed extractions are retried with a new seed and this hint.
EXTRACT_ATTEMPTS = 4
RETRY_HINT = (" Show exactly one {name}: only the one inside the red outline, without any neighbouring"
              " objects, and do not draw the red outline.")
MAX_OUTLINE_FRACTION = 0.002  # of the opaque pixels
# From this attempt on, other instances in the crop are faded towards light grey.
ISOLATE_FROM_ATTEMPT = 1
FADE_RGB = (235, 235, 235)
FADE_STRENGTH = 0.75
MULTIPLE_MIN_AREA = 0.3       # instances this large relative to the largest count as separate objects

# Masks from different prompts that overlap this much are the same object claimed
# twice (e.g. "cabinet" and "shelf"); the higher-scoring one is kept.
DUPLICATE_IOU = 0.85
# Instances covering less of the image than this are dropped.
MIN_AREA = 0.0002
# Crops extend this fraction of the box's longer side beyond it on every side.
CROP_PAD = 0.25

# Scene objects need less detail than a standalone model: 100k faces and 1024 px
# textures make a ~2-3 MB GLB instead of ~15 MB. Large flat objects (boxes) exceed
# 32 GB of VRAM at the 1024 cascade, and fall back to 512.
MESH_RESOLUTIONS = ("1024", "512")
MESH_FACES = 100_000
MESH_TEXTURE = 1024

# Placement fit.
SURFACE_SAMPLES = 20000
YAW_STEPS = 24                # initial yaw candidates, every 15 degrees
SCALE_STEPS = 7               # initial scale candidates per yaw
REFINE_STARTS = 4             # candidates refined with Nelder-Mead, from different yaw basins
REFINE_YAW_SEPARATION = 45    # degrees between refined candidates
ICP_SAMPLES = 3000            # surface samples used by ICP (rendering uses all)
# Generated meshes often have other proportions than the real object (a longer,
# flatter couch), so the last refinement may stretch each axis, at most this much
# and with a penalty that keeps it near uniform.
MAX_STRETCH = 1.6
STRETCH_WEIGHT = 0.5
ICP_ITERATIONS = 15
TRIM = 0.8                    # fraction of closest correspondences used per ICP step
TRUNCATE = 0.10               # m; depth differences are capped here in the score
FREE_SPACE_MARGIN = 0.05      # m; mesh this far in front of the observed surface outside the mask is spill
FREE_SPACE_WEIGHT = 1.0
# A covered pixel with the wrong depth costs at most this; an uncovered one costs 1.
# Were both 1, shrinking the mesh to nothing would beat any imperfect fit.
DEPTH_WEIGHT = 0.5
# The observed points are only the visible surface, so their extent is a lower bound
# on the object's; a placed mesh smaller than this fraction of it is penalised. Only
# vertically and across the view: along the view the depth map smears thin objects
# (a lamp pole blends into the wall behind it), which would inflate the extent.
SIZE_SLACK = 0.9
SIZE_WEIGHT = 5.0
# Objects on the floor whose points stay this low are flat (rugs, cardboard sheets).
# TRELLIS.2 often makes a block of them, so their mesh is squashed to the observed height.
FLAT_HEIGHT = 0.06
FLOOR_CONTACT = 0.15          # m; objects whose lowest point is below this stand on the floor
MIN_POINTS = 30

# Appearance check. A box-like object fits the depth about as well turned by 180
# degrees: the depth differences between its front and back are below the depth
# noise. So the refined yaw candidates that score close to the best are rendered
# from the photo's camera and compared with the photo in DINOv3 patch features
# (TRELLIS.2 generated the mesh from the same features).
APPEARANCE_MODEL_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
APPEARANCE_RES = 448          # px; square crops are resized to this (28 x 28 patches)
APPEARANCE_PAD = 0.1          # crop margin, as a fraction of the box's longer side
APPEARANCE_GREY = (128, 128, 128)  # fills everything outside the instance mask, in photo and render
APPEARANCE_MIN_MASK = 0.5     # patches less covered by the instance mask are not compared
APPEARANCE_MARGIN = 0.02      # candidates within best objective * (1 + this) + this are compared
# Another candidate replaces the depth's best only if it is this much more alike;
# symmetric objects (an open shelf) differ by about 0.01, a flipped sideboard by 0.15.
APPEARANCE_MIN_GAIN = 0.05
APPEARANCE_VRAM_GIB = 4       # cap for this process while placing: the GPU may be shared


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


# ---------------------------------------------------------------- 1. inventory

def stage_inventory(run: Path, photo: Image.Image) -> dict:
    path = run / "inventory.json"
    if not path.exists():
        log("inventory: Gemma 4")
        inv, raw = detect.inventory(photo)
        path.write_text(json.dumps({"objects": inv.objects, "background": inv.background, "raw": raw}, indent=2))
    data = json.loads(path.read_text())
    log(f"inventory: {data['objects']}")
    if not data["objects"]:
        sys.exit(f"Gemma listed no objects; see {path}")
    return data


# -------------------------------------------------------------------- 2. masks

def stage_masks(run: Path, photo: Image.Image, inventory: dict) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Returns the instances, their masks (N, H, W) and the floor mask (H, W)."""
    path = run / "masks.npz"
    if not path.exists():
        names = list(inventory["objects"])
        log(f"masks: SAM 3 for {len(names)} object types and the floor")
        found = segment.run(photo, [*names, FLOOR_PROMPT])
        floor = np.zeros(photo.size[::-1], dtype=bool)
        for inst in found:
            if inst.label == FLOOR_PROMPT:
                floor |= inst.mask
        objects = [i for i in found if i.label != FLOOR_PROMPT and i.mask.mean() >= MIN_AREA]
        objects = _drop_duplicates(objects)

        instances = []
        counts = {}
        for inst in objects:
            counts[inst.label] = counts.get(inst.label, 0) + 1
            ys, xs = np.nonzero(inst.mask)
            instances.append({
                "id": f"{len(instances):02d}-{slug(inst.label)}",
                "name": inst.label,
                "score": round(inst.score, 3),
                "box": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                "area": round(float(inst.mask.mean()), 4),
            })
        for name, expected in inventory["objects"].items():
            if counts.get(name, 0) != expected:
                log(f"masks: Gemma counted {expected} x {name!r}, SAM 3 found {counts.get(name, 0)}")
        np.savez_compressed(path, masks=np.array([i.mask for i in objects]).reshape(-1, *floor.shape), floor=floor)
        (run / "instances.json").write_text(json.dumps(instances, indent=2))
        _masks_overlay(photo, objects, floor).save(run / "masks.png")

    data = np.load(path)
    instances = json.loads((run / "instances.json").read_text())
    log(f"masks: {len(instances)} instances")
    return instances, data["masks"], data["floor"]


def _drop_duplicates(instances: list[segment.Instance]) -> list[segment.Instance]:
    kept = []
    for inst in sorted(instances, key=lambda i: -i.score):
        duplicate = False
        for other in kept:
            if other.label == inst.label:
                continue  # SAM 3 already separates instances of one prompt
            inter = np.logical_and(inst.mask, other.mask).sum()
            union = np.logical_or(inst.mask, other.mask).sum()
            if union and inter / union > DUPLICATE_IOU:
                log(f"masks: dropping {inst.label!r} ({inst.score:.2f}), same object as {other.label!r}")
                duplicate = True
                break
        if not duplicate:
            kept.append(inst)
    return kept


def _masks_overlay(photo: Image.Image, instances: list[segment.Instance], floor: np.ndarray) -> Image.Image:
    overlay = np.array(photo, dtype=np.float32)
    layers = [(floor, (160, 160, 160))] + [
        (inst.mask, reconstruct._hex_rgb(reconstruct.PALETTE[i % len(reconstruct.PALETTE)]))
        for i, inst in enumerate(instances)
    ]
    for mask, colour in layers:
        overlay[mask] = 0.45 * overlay[mask] + 0.55 * np.array(colour)
    return Image.fromarray(overlay.astype(np.uint8))


# -------------------------------------------------------------------- 3. depth

def stage_depth(run: Path, photo: Image.Image) -> reconstruct.Depth:
    path = run / "depth.npz"
    if not path.exists():
        log("depth: Depth Anything 3")
        models.release()
        depth = reconstruct.estimate_depth(photo)
        np.savez_compressed(path, depth=depth.depth, intrinsics=depth.intrinsics, image=depth.image)
    data = np.load(path)
    depth = reconstruct.Depth(data["depth"], data["intrinsics"], data["image"])
    log(f"depth: {depth.depth.shape[1]}x{depth.depth.shape[0]}, {depth.depth.min():.2f}-{depth.depth.max():.2f} m")
    return depth


# -------------------------------------------------------------------- 4. crops

def stage_crops(run: Path, photo: Image.Image, instances: list[dict], masks: np.ndarray) -> None:
    """Writes crop.png, and crop-isolated.png for retries: the same crop with every
    other instance faded, for when the outline alone did not single the object out."""
    for index, (inst, mask) in enumerate(zip(instances, masks)):
        folder = run / "objects" / inst["id"]
        folder.mkdir(parents=True, exist_ok=True)
        if not (folder / "crop.png").exists():
            _crop(photo, mask, inst["box"]).save(folder / "crop.png")
        if not (folder / "crop-isolated.png").exists():
            others = np.any(np.delete(masks, index, axis=0), axis=0) & ~mask if len(masks) > 1 else None
            _crop(photo, mask, inst["box"], fade=others).save(folder / "crop-isolated.png")
    log(f"crops: {len(instances)}")


def _crop(photo: Image.Image, mask: np.ndarray, box: list[int], fade: np.ndarray | None = None) -> Image.Image:
    x1, y1, x2, y2 = box
    pad = round(CROP_PAD * max(x2 - x1, y2 - y1))
    w, h = photo.size
    x1, y1, x2, y2 = max(0, x1 - pad), max(0, y1 - pad), min(w, x2 + pad), min(h, y2 + pad)
    crop = np.array(photo.crop((x1, y1, x2, y2)))
    if fade is not None:
        faded = fade[y1:y2, x1:x2]
        grey = crop[faded].mean(axis=1, keepdims=True)
        crop[faded] = (FADE_STRENGTH * np.array(FADE_RGB) + (1 - FADE_STRENGTH) * grey).astype(np.uint8)
    region = mask[y1:y2, x1:x2]
    width = max(2, round(max(crop.shape[:2]) / 200))
    edge = region & ~binary_erosion(region, iterations=1)
    crop[binary_dilation(edge, iterations=width)] = OUTLINE_RGB
    return Image.fromarray(crop)


# ------------------------------------------------------------------ 5. prompts

def stage_prompts(run: Path, instances: list[dict], enabled: bool, seed: int) -> None:
    todo = [i for i in instances if not (run / "objects" / i["id"] / "prompt.json").exists()]
    for n, inst in enumerate(todo):
        folder = run / "objects" / inst["id"]
        instruction = EXTRACT_PROMPT.format(name=inst["name"])
        record = {"instruction": instruction, "prompt": instruction}
        if enabled:
            log(f"prompts: {n + 1}/{len(todo)} {inst['id']} (Qwen-Image-2.1-PE-T2I)")
            prompt, record["ratio"], record["raw"] = rewrite.rewrite(instruction, Image.open(folder / "crop.png"), seed)
            record["rewritten"] = prompt
            record["prompt"] = clean_rewritten(prompt, inst["name"])
        (folder / "prompt.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
    log(f"prompts: {len(instances)} ({'rewritten' if enabled else 'not rewritten'})")


def clean_rewritten(prompt: str, name: str) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", prompt)
    return " ".join(s for s in sentences if not REWRITE_DROP.search(s)) + REWRITE_SUFFIX.format(name=name)


# ------------------------------------------------------------------ 6. extract

def stage_extract(run: Path, instances: list[dict], steps: int, seed: int) -> None:
    """Extracts every instance, then checks the results and retries the failures.

    Qwen-Image sometimes extracts the outlined object's neighbours too (a stack of
    boxes, both floor lamps) or keeps the red outline; see check_extraction().
    """
    for attempt in range(EXTRACT_ATTEMPTS):
        todo = [i for i in instances if not (run / "objects" / i["id"] / "extracted.png").exists()]
        for n, inst in enumerate(todo):
            folder = run / "objects" / inst["id"]
            log(f"extract: {n + 1}/{len(todo)} {inst['id']} (Qwen-Image 2.1)")
            record = json.loads((folder / "prompt.json").read_text())
            prompt = record["prompt"]
            size = rewrite.size_for_ratio(record["ratio"], EXTRACT_PIXELS) if record.get("ratio") else EXTRACT_SIZE
            tries = _read_check(folder).get("attempt", 0)
            crop = folder / "crop.png"
            if tries:
                prompt += RETRY_HINT.format(name=inst["name"])
            if tries >= ISOLATE_FROM_ATTEMPT:
                crop = folder / "crop-isolated.png"
            image = image_gen.generate(
                prompt, [Image.open(crop).convert("RGB")], size,
                output_resolution=1024, steps=steps, seed=seed + 1000 * tries, on_step=lambda step: None,
            )
            image.save(folder / "extracted.png")

        unchecked = [i for i in instances if "problems" not in _read_check(run / "objects" / i["id"])]
        if unchecked:
            log(f"extract: checking {len(unchecked)} (SAM 3)")
        retry = 0
        for inst in unchecked:
            folder = run / "objects" / inst["id"]
            check = _read_check(folder)
            check["problems"] = check_extraction(Image.open(folder / "extracted.png"), inst["name"])
            if check["problems"] and check.get("attempt", 0) + 1 < EXTRACT_ATTEMPTS:
                log(f"extract: {inst['id']}: {', '.join(check['problems'])}; retrying")
                check = {"attempt": check.get("attempt", 0) + 1, "previous": check["problems"]}
                for stale in ("extracted.png", "subject.png", "mesh.glb", "pose.json"):
                    (folder / stale).unlink(missing_ok=True)
                retry += 1
            elif check["problems"]:
                log(f"extract: {inst['id']}: {', '.join(check['problems'])}; keeping the last attempt")
            (folder / "check.json").write_text(json.dumps(check, indent=2))
        if not retry:
            break
    log(f"extract: {len(instances)}")


def _read_check(folder: Path) -> dict:
    path = folder / "check.json"
    return json.loads(path.read_text()) if path.exists() else {}


def check_extraction(image: Image.Image, name: str) -> list[str]:
    """Problems with an extracted image: several objects, or the red outline left in."""
    rgba = np.array(image.convert("RGBA")).astype(np.int16)
    opaque = rgba[..., 3] > 128
    red = opaque & (rgba[..., 0] > 200) & (rgba[..., 1] < 70) & (rgba[..., 2] < 70)
    problems = []
    if opaque.any() and red.sum() / opaque.sum() > MAX_OUTLINE_FRACTION:
        problems.append("red outline kept")
    # Separate objects: separate opaque regions (SAM 3 misses thin objects such as
    # floor lamps out of context), or separate SAM 3 instances (stacked boxes touch).
    regions, n = label(opaque)
    sizes = np.sort(np.bincount(regions.ravel())[1:])[::-1]
    by_alpha = int(np.sum(sizes >= max(MULTIPLE_MIN_AREA * sizes[0], 0.01 * opaque.size))) if n else 0

    white = Image.new("RGB", image.size, "white")
    white.paste(image.convert("RGBA"), mask=image.convert("RGBA").getchannel("A"))
    masks = sorted((inst.mask for inst in segment.run(white, [name])), key=lambda m: -m.sum())
    # Parts SAM 3 splits off (a cushion, a drawer) are small next to the whole, and a
    # mask mostly inside a kept one is the same object again.
    kept = []
    for mask in masks:
        if mask.sum() < MULTIPLE_MIN_AREA * masks[0].sum():
            break
        if all(np.logical_and(mask, other).sum() < 0.5 * mask.sum() for other in kept):
            kept.append(mask)

    count = max(by_alpha, len(kept))
    if count > 1:
        problems.append(f"{count} objects")
    return problems


# ------------------------------------------------------------------- 7. meshes

def stage_meshes(run: Path, instances: list[dict], seed: int) -> None:
    todo = [i for i in instances if not (run / "objects" / i["id"] / "mesh.glb").exists()]
    for n, inst in enumerate(todo):
        folder = run / "objects" / inst["id"]
        log(f"meshes: {n + 1}/{len(todo)} {inst['id']} (TRELLIS.2)")
        record = json.loads((folder / "mesh.json").read_text()) if (folder / "mesh.json").exists() else {}
        failed = record.setdefault("out_of_memory", [])
        untried = [r for r in MESH_RESOLUTIONS if r not in failed]
        if not untried:
            log(f"meshes: {inst['id']}: out of GPU memory at every resolution; skipped")
            continue
        subject = mesh_gen.remove_background(Image.open(folder / "extracted.png"))
        subject.save(folder / "subject.png")
        if _generate_mesh(subject, untried[0], seed, folder / "mesh.glb"):
            record["resolution"] = untried[0]
            (folder / "mesh.json").write_text(json.dumps(record))
        else:
            # Memory lost in a failed attempt (CuMesh's own buffers) is not returned,
            # so this process stops; the next worker starts with a clean GPU.
            failed.append(untried[0])
            (folder / "mesh.json").write_text(json.dumps(record))
            log(f"meshes: {inst['id']}: out of GPU memory at {untried[0]}")
            return


def meshes_to_retry(run: Path, instances: list[dict]) -> list[str]:
    """Instances without a mesh that still have a resolution left to try."""
    pending = []
    for inst in instances:
        folder = run / "objects" / inst["id"]
        if (folder / "mesh.glb").exists():
            continue
        record = json.loads((folder / "mesh.json").read_text()) if (folder / "mesh.json").exists() else {}
        if any(r not in record.get("out_of_memory", []) for r in MESH_RESOLUTIONS):
            pending.append(inst["id"])
    return pending


def _generate_mesh(subject: Image.Image, resolution: str, seed: int, path: Path) -> bool:
    """False if TRELLIS.2 or its mesh extensions run out of GPU memory."""
    try:
        mesh_gen.generate(subject, resolution, seed=seed, decimation_target=MESH_FACES,
                          texture_size=MESH_TEXTURE, out_path=path)
        return True
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if "out of memory" not in str(error):
            raise
    # Outside the except block, so the traceback no longer holds the failed tensors.
    gc.collect()
    torch.cuda.empty_cache()
    return False


# -------------------------------------------------------------------- 8. place

class World:
    """Floor-aligned metric frame: Y up, floor at y = 0, origin below the camera,
    camera looking along -Z (the glTF convention)."""

    def __init__(self, depth: reconstruct.Depth, floor: np.ndarray):
        self.depth = depth
        h, w = depth.depth.shape
        self.points_cam = reconstruct.backproject(depth.depth, depth.intrinsics)
        floor_small = np.array(Image.fromarray(floor).resize((w, h), Image.NEAREST))
        floor_small = binary_erosion(floor_small, iterations=reconstruct.MASK_ERODE_PX)
        floor_points = self.points_cam[floor_small]
        if len(floor_points) >= 100:
            normal, offset = reconstruct.fit_plane(floor_points)
        else:
            log("place: no floor found; assuming a level camera and the lowest points as floor")
            normal = np.array([0.0, -1.0, 0.0])
            offset = np.percentile(self.points_cam.reshape(-1, 3) @ -normal, 99)
        up = normal
        x = np.array([1.0, 0.0, 0.0]) - up[0] * up
        x /= np.linalg.norm(x)
        z = np.cross(x, up)
        # Rows map camera coordinates to world axes; origin is the floor point below the camera.
        self.rotation = np.stack([x, up, z])
        self.origin = -offset * up
        self.camera_height = float(offset)

    def to_world(self, points_cam: np.ndarray) -> np.ndarray:
        return (points_cam - self.origin) @ self.rotation.T

    def to_camera(self, points_world: np.ndarray) -> np.ndarray:
        return points_world @ self.rotation + self.origin

    def camera_pose(self) -> np.ndarray:
        """4x4 camera-to-world transform in the OpenGL convention (x right, y up, -z forward)."""
        pose = np.eye(4)
        pose[:3, :3] = self.rotation @ np.diag([1.0, -1.0, -1.0])
        pose[:3, 3] = self.to_world(np.zeros(3))
        return pose

    def photo_camera(self, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """World-to-camera matrix (OpenCV axes) and intrinsics of the photo's camera at `size` pixels."""
        h, w = self.depth.depth.shape
        view = np.eye(4)
        view[:3, :3] = self.rotation.T
        view[:3, 3] = self.origin
        return view, self.depth.intrinsics * np.array([[size[0] / w], [size[1] / h], [1.0]])

    def project(self, points_world: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pixel column, row (at depth resolution) and camera depth of each point."""
        p = self.to_camera(points_world)
        k = self.depth.intrinsics
        z = p[:, 2]
        safe = np.where(z > 1e-6, z, 1e-6)
        return k[0, 0] * p[:, 0] / safe + k[0, 2], k[1, 1] * p[:, 1] / safe + k[1, 2], z


def yaw_matrix(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class Fit:
    """Places one mesh (Y up) on its instance: p = Ry(yaw) @ (scale * stretch * q) + t.

    Parameters are [yaw, log scale, tx, ty, tz] or, with per-axis stretch,
    [yaw, log scale, tx, ty, tz, sx, sy, sz] where s* are log stretches (mean removed).

    Scored by rendering the placed mesh's depth from the photo's camera and comparing
    it with the estimated depth: inside the instance's mask the depths should agree
    and every pixel should be covered; outside it, the mesh must not appear in front
    of a farther observed surface. Pixels hidden behind a nearer object (the mesh
    behind an occluder) cost nothing, so occluded parts are free.
    """

    def __init__(self, world: World, observed: np.ndarray, surface: np.ndarray,
                 mask: np.ndarray, mask_full: np.ndarray):
        self.world = world
        self.observed = observed
        self.surface = surface
        self.mask = mask            # eroded instance mask at depth resolution: depth is compared here
        self.outside = ~mask_full   # pixels that are clearly not this instance
        self.depth = world.depth.depth
        self.on_floor = observed[:, 1].min() < FLOOR_CONTACT
        low, high = np.percentile(observed, [1, 99], axis=0)
        if self.on_floor:
            low[1] = 0.0
        self.observed_extent = np.maximum(high - low, 0.01)
        forward = world.rotation @ np.array([0.0, 0.0, 1.0])
        forward[1] = 0.0
        across = np.cross([0.0, 1.0, 0.0], forward / np.linalg.norm(forward))
        self.prior_axes = np.stack([across, [0.0, 1.0, 0.0]])
        low, high = np.percentile(observed @ self.prior_axes.T, [1, 99], axis=0)
        if self.on_floor:
            low[1] = 0.0
        self.prior_extent = np.maximum(high - low, 0.01)
        self.flat = self.on_floor and high[1] < FLAT_HEIGHT
        self.flat_scale = max(high[1], 0.01) / max(np.ptp(surface[:, 1]), 1e-6)
        self.icp_surface = surface[np.random.default_rng(0).choice(len(surface), min(ICP_SAMPLES, len(surface)),
                                                                   replace=False)]

    def scales(self, params) -> np.ndarray:
        stretch = np.zeros(3)
        if len(params) > 5:
            stretch = np.clip(np.asarray(params[5:8]) - np.mean(params[5:8]), -np.log(MAX_STRETCH), np.log(MAX_STRETCH))
        scales = np.exp(params[1] + stretch)
        if self.flat:
            scales[1] = self.flat_scale
        return scales

    def transform(self, params, surface: np.ndarray | None = None) -> np.ndarray:
        surface = self.surface if surface is None else surface
        return (surface * self.scales(params)) @ yaw_matrix(params[0]).T + np.asarray(params[2:5])

    def ground(self, params) -> list[float]:
        """Rests an on-floor object's lowest point on the floor (yaw does not change heights)."""
        params = list(params)
        if self.on_floor:
            params[3] = -self.scales(params)[1] * self.surface[:, 1].min()
        return params

    def matrix(self, params) -> np.ndarray:
        m = np.eye(4)
        m[:3, :3] = yaw_matrix(params[0]) @ np.diag(self.scales(params))
        m[:3, 3] = params[2:5]
        return m

    def render(self, placed: np.ndarray) -> np.ndarray:
        """Depth of the nearest surface sample per pixel (inf where none), each sample
        splatted over 2x2 pixels to close the gaps between samples."""
        h, w = self.depth.shape
        u, v, z = self.world.project(placed)
        buffer = np.full(h * w, np.inf)
        for du in (0, 1):
            for dv in (0, 1):
                ui, vi = (u + du - 0.5).astype(int), (v + dv - 0.5).astype(int)
                ok = (z > 0) & (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
                np.minimum.at(buffer, vi[ok] * w + ui[ok], z[ok])
        return buffer.reshape(h, w)

    def score(self, params) -> dict:
        rendered = self.render(self.transform(params))
        covered = np.isfinite(rendered)
        n = self.mask.sum()
        inside = self.mask & covered
        err = np.minimum(np.abs(rendered[inside] - self.depth[inside]), TRUNCATE)
        spill = covered & self.outside & (self.depth > rendered + FREE_SPACE_MARGIN)
        missing = (self.mask & ~covered).sum() / n
        spill_fraction = spill.sum() / n
        total = ((DEPTH_WEIGHT * np.sum(err ** 2) / TRUNCATE ** 2 + (self.mask & ~covered).sum()) / n
                 + FREE_SPACE_WEIGHT * spill_fraction)
        return {"total": float(total), "depth_rms_m": float(np.sqrt(np.mean(err ** 2))) if len(err) else TRUNCATE,
                "missing": float(missing), "spill": float(spill_fraction)}

    def size_penalty(self, params) -> float:
        extent = np.ptp(self.transform(params, self.icp_surface) @ self.prior_axes.T, axis=0)
        short = np.maximum(np.log(SIZE_SLACK * self.prior_extent / np.maximum(extent, 1e-6)), 0.0)
        return SIZE_WEIGHT * float(np.sum(short ** 2))

    def objective(self, params) -> float:
        return self.score(params)["total"] + self.size_penalty(params)

    def initial(self, yaw: float, scale: float) -> list[float]:
        base = 0.0 if self.on_floor else self.observed[:, 1].min()
        centre = self.observed.mean(axis=0)
        return self.ground([yaw, np.log(scale), centre[0], base - scale * self.surface[:, 1].min(), centre[2]])

    def icp(self, params) -> list[float]:
        """Yaw and translation from observed points to mesh surface, at fixed scale.

        Only a starting point for the search: with only the visible side observed,
        a free scale would grow until the mesh covers every point.
        """
        params = list(params)
        scale = np.exp(params[1])
        for _ in range(ICP_ITERATIONS):
            dist, idx = cKDTree(self.transform(params, self.icp_surface)).query(self.observed)
            keep = dist <= np.quantile(dist, TRIM)
            p, q = self.icp_surface[idx[keep]], self.observed[keep]
            pc, qc = p - p.mean(axis=0), q - q.mean(axis=0)
            yaw = np.arctan2(np.sum(qc[:, 0] * pc[:, 2] - qc[:, 2] * pc[:, 0]),
                             np.sum(qc[:, 0] * pc[:, 0] + qc[:, 2] * pc[:, 2]))
            t = q.mean(axis=0) - scale * yaw_matrix(yaw) @ p.mean(axis=0)
            params = self.ground([yaw, params[1], *t])
        return params

    def flipped(self, params) -> list[float]:
        """The same placement turned by 180 degrees about the placed mesh's centre."""
        turned = list(params)
        turned[0] += np.pi
        shift = self.transform(params, self.icp_surface).mean(axis=0) - self.transform(turned, self.icp_surface).mean(axis=0)
        turned[2:5] = np.asarray(turned[2:5]) + shift
        return self.ground(turned)

    def solve(self, appearance: Callable[[list[np.ndarray]], list[float] | None] | None = None) -> dict:
        """`appearance` scores candidate matrices by their likeness to the photo (higher is better)."""
        heights = self.observed[:, 1]
        span = max(heights.max() - (0.0 if self.on_floor else heights.min()), 0.05)
        by_height = span / max(np.ptp(self.surface[:, 1]), 1e-3)
        if self.flat:  # the height says nothing about a rug's size; its footprint does
            by_height = max(self.observed_extent[0], self.observed_extent[2]) / max(np.ptp(self.surface[:, [0, 2]], axis=0))
        # Generated meshes can have other proportions than the real object, so the
        # height alone is not a reliable scale; search well around it.
        scales = by_height * np.geomspace(0.5, 1.4, SCALE_STEPS)
        candidates = []
        for k in range(YAW_STEPS):
            for scale in scales:
                params = self.icp(self.initial(2 * np.pi * k / YAW_STEPS, scale))
                candidates.append((self.objective(params), params))
        candidates.sort(key=lambda c: c[0])

        # Refine the best candidate of each clearly different yaw: a front/back flip
        # often scores close, and one basin must not crowd out the others.
        starts = []
        for _, params in candidates:
            if all(_angle_between(params[0], other[0]) > np.radians(REFINE_YAW_SEPARATION) for other in starts):
                starts.append(params)
            if len(starts) == REFINE_STARTS:
                break
        # The best start's flip is always refined, as the appearance check needs it.
        flip = self.flipped(starts[0])
        if all(_angle_between(flip[0], other[0]) > np.radians(REFINE_YAW_SEPARATION) for other in starts):
            starts.append(flip)
        refined = [self.refine(start) for start in starts]
        lowest = min(score for _, score in refined)
        close = [score <= lowest * (1 + APPEARANCE_MARGIN) + APPEARANCE_MARGIN for _, score in refined]
        # Every candidate is compared, not only the close ones, so pose.json shows how they differ.
        likeness = appearance([self.matrix(params) for params, _ in refined]) if appearance else None
        lowest_index = int(np.argmin([score for _, score in refined]))
        chosen = lowest_index
        if likeness is not None:
            alike = max((i for i in range(len(refined)) if close[i]), key=lambda i: likeness[i])
            if likeness[alike] >= likeness[lowest_index] + APPEARANCE_MIN_GAIN:
                chosen = alike
        # Then let each axis stretch.
        best = self.refine(list(refined[chosen][0]) + [0.0, 0.0, 0.0])[0]

        quality = self.score(best)
        scales = self.scales(best)
        return {
            "matrix": self.matrix(best).tolist(), "yaw_deg": round(float(np.degrees(best[0])) % 360, 1),
            "scale": [round(float(v), 4) for v in scales], "on_floor": bool(self.on_floor), "flat": bool(self.flat),
            "points": len(self.observed), **{k: round(v, 4) for k, v in quality.items()},
            "chosen_by": "depth" if chosen == lowest_index else "appearance",
            "candidates": [{"yaw_deg": round(float(np.degrees(params[0])) % 360, 1), "objective": round(score, 4),
                            "close": close[i], "likeness": None if likeness is None else round(likeness[i], 4)}
                           for i, (params, score) in enumerate(refined)],
        }

    def refine(self, start) -> tuple[list[float], float]:
        start = np.asarray(start, dtype=float)
        free = [i for i in range(len(start)) if not (i == 3 and self.on_floor)]

        def unpack(x):
            params = start.copy()
            params[free] = x
            return self.ground(params)

        def objective(x):
            params = unpack(x)
            penalty = 0.0
            if len(params) > 5:
                stretch = np.asarray(params[5:8]) - np.mean(params[5:8])
                penalty = STRETCH_WEIGHT * float(np.sum(stretch ** 2))
            return self.objective(params) + penalty

        x0 = start[free]
        result = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxfev": 300 * len(free), "xatol": 1e-3, "fatol": 1e-5,
                                   "initial_simplex": _simplex(x0, free)})
        return unpack(result.x), float(result.fun)


def _angle_between(a: float, b: float) -> float:
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def _simplex(x0: np.ndarray, free: list[int]) -> np.ndarray:
    """Initial Nelder-Mead simplex: 10 degrees of yaw, 10% scale, 5 cm translation, 10% stretch."""
    step = {0: np.radians(10), 1: 0.1, 2: 0.05, 3: 0.05, 4: 0.05, 5: 0.1, 6: 0.1, 7: 0.1}
    return np.vstack([x0, x0 + np.diag([step[i] for i in free])])


def load_mesh(path: Path) -> trimesh.Trimesh:
    return trimesh.load(path, force="mesh")


def observed_points(world: World, labels: np.ndarray, index: int) -> np.ndarray:
    points = world.to_world(world.points_cam[labels == index])
    if len(points) < MIN_POINTS:
        return points
    # Drop flying pixels: points far from their neighbours.
    dist = cKDTree(points).query(points, k=9)[0][:, 1:].mean(axis=1)
    return points[dist < dist.mean() + 2 * dist.std()]


class Appearance:
    """Likeness of placed meshes to the photo: each placement is rendered from the
    photo's camera, photo and render are cropped to the instance and greyed outside
    its mask, and their DINOv3 patch features are compared position by position."""

    MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    def __init__(self, photo: Image.Image, world: World):
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(APPEARANCE_MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
        self.registers = self.model.config.num_register_tokens
        self.patch = self.model.config.patch_size
        self.photo = photo
        self.view, self.k = world.photo_camera(photo.size)

    def compare(self, mesh_path: Path, mask: np.ndarray, matrices: list[np.ndarray]) -> list[float] | None:
        """Mean cosine similarity per matrix; None if the mask covers no whole patch or a value is not finite."""
        ys, xs = np.nonzero(mask)
        side = (1 + 2 * APPEARANCE_PAD) * max(xs.max() - xs.min(), ys.max() - ys.min()) + 1
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        box = tuple(round(v) for v in (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2))
        size = (APPEARANCE_RES, APPEARANCE_RES)
        # Area-averaged, so each pixel and later each patch holds the fraction of it inside the mask.
        crop_mask = np.array(Image.fromarray(mask.astype(np.float32)).crop(box).resize(size, Image.BOX))
        grid = APPEARANCE_RES // self.patch
        keep = crop_mask.reshape(grid, self.patch, grid, self.patch).mean(axis=(1, 3)).ravel() >= APPEARANCE_MIN_MASK
        if not keep.any():
            return None

        images = [self.photo] + [render.render(render.parts_from_glb(mesh_path, m), self.view, self.k, self.photo.size,
                                               background=APPEARANCE_GREY)[0] for m in matrices]
        inside = (crop_mask >= 0.5)[..., None]
        crops = [np.where(inside, np.array(image.crop(box).resize(size, Image.BICUBIC)), APPEARANCE_GREY)
                 for image in images]
        x = torch.tensor(np.stack(crops), dtype=torch.float32).permute(0, 3, 1, 2) / 255
        x = ((x - self.MEAN) / self.STD).to("cuda", torch.bfloat16)
        with torch.no_grad():
            tokens = self.model(pixel_values=x).last_hidden_state[:, 1 + self.registers:]
        tokens = torch.nn.functional.normalize(tokens.float(), dim=-1)[:, torch.from_numpy(keep).cuda()]
        likeness = (tokens[1:] * tokens[:1]).sum(dim=-1).mean(dim=-1)
        return likeness.tolist() if torch.isfinite(likeness).all() else None


def stage_place(run: Path, photo: Image.Image, instances: list[dict], masks: np.ndarray, world: World,
                force: bool) -> None:
    h, w = world.depth.depth.shape
    all_instances = [segment.Instance(i["name"], i["score"], m) for i, m in zip(instances, masks)]
    labels = reconstruct.label_map(all_instances, (h, w))
    pending = [index for index, inst in enumerate(instances)
               if (force or not (run / "objects" / inst["id"] / "pose.json").exists())
               and (run / "objects" / inst["id"] / "mesh.glb").exists()]
    if not pending:
        log("place: 0 fitted")
        return
    # Capped so that running out of memory fails this process, not another job on the same GPU.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, APPEARANCE_VRAM_GIB * 2 ** 30 / total))
    appearance = Appearance(photo, world)
    placed = 0
    try:
        for index in pending:
            inst = instances[index]
            folder = run / "objects" / inst["id"]
            observed = observed_points(world, labels, index)
            if len(observed) < MIN_POINTS or (labels == index).sum() < MIN_POINTS:
                log(f"place: {inst['id']}: only {len(observed)} depth points; skipped")
                continue
            mesh = load_mesh(folder / "mesh.glb")
            surface = trimesh.sample.sample_surface(mesh, SURFACE_SAMPLES, seed=0)[0]
            mask_full = np.array(Image.fromarray(masks[index]).resize((w, h), Image.NEAREST))
            pose = Fit(world, observed, surface, labels == index, mask_full).solve(
                lambda matrices: appearance.compare(folder / "mesh.glb", masks[index], matrices))
            (folder / "pose.json").write_text(json.dumps(pose, indent=2))
            placed += 1
            log(f"place: {inst['id']}: yaw {pose['yaw_deg']} deg ({pose['chosen_by']}), "
                f"scale {'/'.join(f'{v:.2f}' for v in pose['scale'])}, depth rms {pose['depth_rms_m'] * 100:.1f} cm, "
                f"uncovered {pose['missing']:.0%}, spill {pose['spill']:.0%}")
    finally:
        appearance = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(1.0)
    log(f"place: {placed} fitted")


# ----------------------------------------------------------------- 9. assemble

def stage_assemble(run: Path, photo: Image.Image, instances: list[dict], world: World) -> None:
    scene = trimesh.Scene()
    parts = []
    for inst in instances:
        folder = run / "objects" / inst["id"]
        if not (folder / "pose.json").exists():
            continue
        matrix = np.array(json.loads((folder / "pose.json").read_text())["matrix"])
        loaded = trimesh.load(folder / "mesh.glb", force="scene")
        for node in loaded.graph.nodes_geometry:
            node_matrix, geometry = loaded.graph[node]
            scene.add_geometry(loaded.geometry[geometry].copy(), node_name=inst["id"], transform=matrix @ node_matrix)
        parts += render.parts_from_glb(folder / "mesh.glb", matrix)

    # Floor: a quad under the floor-level points and every object's footprint.
    points = world.to_world(world.points_cam.reshape(-1, 3))
    low = points[points[:, 1] < 0.05][:, [0, 2]]
    corners = [np.percentile(low, [1, 99], axis=0)] + [p.mesh.bounds[:, [0, 2]] for p in parts]
    (x0, z0), (x1, z1) = np.min([c[0] for c in corners], axis=0) - 0.3, np.max([c[1] for c in corners], axis=0) + 0.3
    floor = trimesh.Trimesh(vertices=[[x0, 0, z0], [x1, 0, z0], [x1, 0, z1], [x0, 0, z1]], faces=[[0, 2, 1], [0, 3, 2]])
    floor.visual.face_colors = [200, 200, 200, 255]
    scene.add_geometry(floor, node_name="floor")

    h, w = world.depth.depth.shape
    k = world.depth.intrinsics
    fov = np.degrees([2 * np.arctan(w / (2 * k[0, 0])), 2 * np.arctan(h / (2 * k[1, 1]))])
    scene.camera = trimesh.scene.Camera(name="photo", resolution=(w, h), fov=fov)
    scene.camera_transform = world.camera_pose()
    scene.export(run / "scene.glb")

    (run / "scene.json").write_text(json.dumps({
        "camera": {"pose": world.camera_pose().tolist(), "fov_deg": fov.tolist(), "resolution": [w, h],
                   "height_m": round(world.camera_height, 3)},
        "objects": {i["id"]: json.loads((run / "objects" / i["id"] / "pose.json").read_text())
                    for i in instances if (run / "objects" / i["id"] / "pose.json").exists()},
    }, indent=2))
    if parts:
        _renders(run, photo, world, parts + [render.Part(floor, None, (170, 170, 170))])
    log(f"assemble: {len(scene.geometry) - 1} objects -> {run / 'scene.glb'}")


def _renders(run: Path, photo: Image.Image, world: World, parts: list) -> None:
    """The scene from the photo's camera (alone and over the photo), from the side and from above."""
    photo_view, k_photo = world.photo_camera(photo.size)
    from_photo, covered = render.render(parts[:-1], photo_view, k_photo, photo.size)
    from_photo.save(run / "render-photo.png")
    base = np.array(photo, dtype=np.float32)
    blended = np.where(covered[..., None], 0.3 * base + 0.7 * np.array(from_photo, dtype=np.float32), 0.6 * base)
    Image.fromarray(blended.astype(np.uint8)).save(run / "render-overlay.png")

    # Orbit views around the objects' centre, with a 60 degree field of view.
    size = (1200, 900)
    focal = size[0] / (2 * np.tan(np.radians(30)))
    k_orbit = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]])
    centre = np.mean([p.mesh.bounds.mean(axis=0) for p in parts[:-1]], axis=0) * [1, 0, 1]
    camera = world.to_world(np.zeros(3))
    back = (camera - centre) * [1, 0, 1]
    distance = np.linalg.norm(back) + 1.0
    back /= np.linalg.norm(back)
    angle = np.radians(70)
    side = np.array([np.cos(angle) * back[0] + np.sin(angle) * back[2], 0, -np.sin(angle) * back[0] + np.cos(angle) * back[2]])
    views = {
        "side": render.look_at(centre + distance * side + [0, 2.5, 0], centre + [0, 0.4, 0]),
        "top": render.look_at(centre + [0, distance + 3.0, 0], centre, up=-back),
    }
    panels = [photo, from_photo]
    for name, view in views.items():
        image, _ = render.render(parts, view, k_orbit, size)
        image.save(run / f"render-{name}.png")
        panels.append(image)
    height = 450
    panels = [p.resize((round(p.size[0] * height / p.size[1]), height)) for p in panels]
    sheet = Image.new("RGB", (sum(p.size[0] for p in panels), height), "white")
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.size[0]
    sheet.save(run / "renders.png")


# --------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("photo", type=Path)
    parser.add_argument("--run", type=Path, help="run folder to resume (default: a new one in outputs/scene/)")
    parser.add_argument("--until", choices=["inventory", "masks", "depth", "crops", "prompts", "extract",
                                            "meshes", "place", "assemble"], default="assemble")
    parser.add_argument("--rewrite", action="store_true", help="expand the extraction prompts with Qwen-Image-2.1-PE-T2I")
    parser.add_argument("--refit", action="store_true", help="redo placement even where pose.json exists")
    parser.add_argument("--steps", type=int, default=40, help="Qwen-Image denoising steps")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mesh-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    run = args.run or OUTPUT_DIR / time.strftime("%Y%m%d-%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    photo_path = run / f"photo{args.photo.suffix.lower()}"
    if not photo_path.exists():
        shutil.copy(args.photo, photo_path)
    photo = Image.open(photo_path).convert("RGB")
    if args.mesh_worker:
        stage_meshes(run, json.loads((run / "instances.json").read_text()), args.seed)
        return
    log(f"run folder: {run}")

    def done(stage: str) -> bool:
        return args.until == stage

    inventory = stage_inventory(run, photo)
    if done("inventory"):
        return
    instances, masks, floor = stage_masks(run, photo, inventory)
    if done("masks"):
        return
    depth = stage_depth(run, photo)
    if done("depth"):
        return
    stage_crops(run, photo, instances, masks)
    if done("crops"):
        return
    stage_prompts(run, instances, args.rewrite, args.seed)
    if done("prompts"):
        return
    stage_extract(run, instances, args.steps, args.seed)
    if done("extract"):
        return
    # Meshes are generated by worker processes, one after another, each until its
    # first out-of-memory failure; see stage_meshes().
    models.release()
    while pending := meshes_to_retry(run, instances):
        log(f"meshes: worker for {len(pending)} instances")
        subprocess.run([sys.executable, "-m", "dollhouse.scene", str(photo_path), "--run", str(run),
                        "--mesh-worker", "--seed", str(args.seed)], check=True)
        if meshes_to_retry(run, instances) == pending and not any(
                (run / "objects" / i / "mesh.json").exists() for i in pending):
            break  # the worker made no progress
    log(f"meshes: {sum((run / 'objects' / i['id'] / 'mesh.glb').exists() for i in instances)}/{len(instances)}")
    if done("meshes"):
        return
    models.release()
    world = World(depth, floor)
    stage_place(run, photo, instances, masks, world, args.refit)
    if done("place"):
        return
    stage_assemble(run, photo, instances, world)


if __name__ == "__main__":
    main()
