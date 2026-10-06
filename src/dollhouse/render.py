"""Renders an assembled scene with nvdiffrast, to check placements by eye.

Cameras use the OpenCV convention: a 4x4 world-to-camera matrix (x right, y down,
z forward) and pixel intrinsics.
"""

from dataclasses import dataclass

import numpy as np
import torch
import trimesh
from PIL import Image

NEAR, FAR = 0.05, 100.0
LIGHT = np.array([0.3, 1.0, 0.5]) / np.linalg.norm([0.3, 1.0, 0.5])  # world direction towards the light


@dataclass
class Part:
    mesh: trimesh.Trimesh      # in world coordinates
    texture: np.ndarray | None  # H, W, 3 uint8 base colour, or None for `colour`
    colour: tuple[int, int, int] = (200, 200, 200)


def parts_from_glb(path, matrix: np.ndarray) -> list[Part]:
    loaded = trimesh.load(path, force="scene")
    parts = []
    for node in loaded.graph.nodes_geometry:
        node_matrix, name = loaded.graph[node]
        mesh = loaded.geometry[name].copy()
        mesh.apply_transform(matrix @ node_matrix)
        texture = None
        material = getattr(mesh.visual, "material", None)
        if getattr(mesh.visual, "uv", None) is not None and getattr(material, "baseColorTexture", None) is not None:
            texture = np.array(material.baseColorTexture.convert("RGB"))
        parts.append(Part(mesh, texture))
    return parts


def look_at(eye, target, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """World-to-camera matrix (OpenCV axes) for a camera at `eye` looking at `target`."""
    eye, target, up = (np.asarray(v, dtype=float) for v in (eye, target, up))
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    view = np.eye(4)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ eye
    return view


def render(parts: list[Part], view: np.ndarray, intrinsics: np.ndarray, size: tuple[int, int],
           background=(255, 255, 255)) -> tuple[Image.Image, np.ndarray]:
    """Shaded colour image and a coverage mask; each part is lit by one directional light."""
    import nvdiffrast.torch as dr

    width, height = size
    fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]
    # Clip space with w = camera z; image row 0 is the top row (ndc y = -1).
    projection = torch.tensor([
        [2 * fx / width, 0, 2 * cx / width - 1, 0],
        [0, 2 * fy / height, 2 * cy / height - 1, 0],
        [0, 0, (FAR + NEAR) / (FAR - NEAR), -2 * FAR * NEAR / (FAR - NEAR)],
        [0, 0, 1, 0],
    ], dtype=torch.float32, device="cuda") @ torch.tensor(view, dtype=torch.float32, device="cuda")

    ctx = dr.RasterizeCudaContext()
    colour = torch.tensor(background, dtype=torch.float32, device="cuda").expand(height, width, 3).clone() / 255
    depth = torch.full((height, width), float("inf"), device="cuda")
    for part in parts:
        mesh = part.mesh
        vertices = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
        faces = torch.tensor(mesh.faces, dtype=torch.int32, device="cuda")
        clip = torch.cat([vertices, torch.ones_like(vertices[:, :1])], dim=1) @ projection.T
        rast, _ = dr.rasterize(ctx, clip[None], faces, resolution=(height, width))
        hit = rast[0, ..., 3] > 0
        z = rast[0, ..., 2]
        closer = hit & (z < depth)
        if not closer.any():
            continue

        if part.texture is not None:
            uv = torch.tensor(mesh.visual.uv, dtype=torch.float32, device="cuda")
            uv = torch.stack([uv[:, 0], 1 - uv[:, 1]], dim=1)  # glTF v runs top-down in the image
            texcoords, _ = dr.interpolate(uv[None], rast, faces)
            texture = torch.tensor(part.texture, dtype=torch.float32, device="cuda")[None] / 255
            base = dr.texture(texture, texcoords, filter_mode="linear")[0]
        else:
            base = torch.tensor(part.colour, dtype=torch.float32, device="cuda").expand(height, width, 3) / 255

        normals = torch.tensor(mesh.face_normals, dtype=torch.float32, device="cuda")
        face = (rast[0, ..., 3].long() - 1).clamp(min=0)
        lambert = (normals[face] @ torch.tensor(LIGHT, dtype=torch.float32, device="cuda")).abs()
        shaded = base * (0.55 + 0.45 * lambert[..., None])
        colour = torch.where(closer[..., None], shaded, colour)
        depth = torch.where(closer, z, depth)

    image = (colour.clamp(0, 1) * 255).byte().cpu().numpy()
    return Image.fromarray(image), torch.isfinite(depth).cpu().numpy()
