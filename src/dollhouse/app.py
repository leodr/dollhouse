import os

# Offloading shuffles multi-GB components on and off the GPU; without this the
# allocator fragments and a full-resolution decode can OOM despite free memory.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import time
from collections import Counter
from pathlib import Path

import gradio as gr
import numpy as np
import torch

from dollhouse import detect, image_gen, mesh_gen, segment

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs"
MAX_SEED = 2**31 - 1
MAX_REFERENCE_IMAGES = 10

AUTO = "Auto (from first input image)"
SIZES = {"Full (~4 MP)": 1.0, "Half (~1 MP, faster)": 0.5}

# All tabs share one GPU and one loaded model; run their jobs one at a time.
GPU_QUEUE = dict(concurrency_id="gpu", concurrency_limit=1)


def _stamp(seed: int) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{seed}"


def detect_objects(image, prompt, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Add an input image.")
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")

    progress(0, desc="Loading Gemma 4")
    try:
        text = detect.run(image, prompt)
    except torch.OutOfMemoryError:
        raise gr.Error("Out of GPU memory. Try a smaller image.")
    detections = detect.parse(text, *image.size)
    if not detections:
        gr.Warning("The model's answer contained no bounding boxes; see the raw output.")
    rows = [[d.label, *d.box] for d in detections]
    return detect.draw(image, detections), rows, text


def segment_objects(image, prompts_text, threshold, progress=gr.Progress()):
    if image is None:
        raise gr.Error("Add an input image.")
    prompts = segment.parse_prompts(prompts_text or "")
    if not prompts:
        raise gr.Error("Enter at least one prompt.")

    progress(0, desc="Loading SAM 3")
    try:
        instances = segment.run(image, prompts, threshold)
    except torch.OutOfMemoryError:
        raise gr.Error("Out of GPU memory. Try a smaller image.")

    counts = Counter()
    annotations, rows = [], []
    for inst in instances:
        counts[inst.label] += 1
        name = f"{inst.label} {counts[inst.label]}"
        annotations.append((inst.mask.astype(np.float32), name))
        rows.append([name, round(inst.score, 2), round(float(100 * inst.mask.mean()), 1)])
    missing = [p for p in prompts if p not in counts]
    if missing:
        gr.Warning(f"Nothing above the threshold for: {', '.join(missing)}")
    return (image, annotations), rows


def generate_image(prompt, gallery, aspect_ratio, size, steps, seed, randomize_seed, progress=gr.Progress()):
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")

    images = [item[0].convert("RGB") for item in (gallery or [])]
    if len(images) > MAX_REFERENCE_IMAGES:
        raise gr.Error(f"At most {MAX_REFERENCE_IMAGES} input images are supported.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)

    scale = SIZES[size]
    if aspect_ratio == AUTO and images:
        target = None
    else:
        width, height = image_gen.ASPECT_RATIOS["1:1" if aspect_ratio == AUTO else aspect_ratio]
        target = (int(width * scale), int(height * scale))

    progress(0, desc="Loading Qwen-Image")
    try:
        result = image_gen.generate(
            prompt,
            images,
            target,
            output_resolution=int(2048 * scale),
            steps=int(steps),
            seed=seed,
            on_step=lambda step: progress((step, int(steps)), desc="Denoising"),
        )
    except torch.OutOfMemoryError:
        raise gr.Error("Out of GPU memory. Try the smaller size or fewer input images.")

    OUTPUT_DIR.mkdir(exist_ok=True)
    result.save(OUTPUT_DIR / f"{_stamp(seed)}.png")
    return result, seed


def generate_mesh(image, resolution, decimation_target, texture_size, seed, randomize_seed,
                  progress=gr.Progress(track_tqdm=True)):
    if image is None:
        raise gr.Error("Add an input image.")

    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    seed = int(seed)

    progress(0, desc="Loading TRELLIS.2")
    try:
        subject = mesh_gen.remove_background(image)
        path = mesh_gen.generate(
            subject,
            resolution,
            seed=seed,
            decimation_target=int(decimation_target),
            texture_size=int(texture_size),
            out_path=OUTPUT_DIR / "3d" / f"{_stamp(seed)}.glb",
        )
    except torch.OutOfMemoryError:
        raise gr.Error("Out of GPU memory. Try a lower resolution.")
    except OSError as e:
        if "gated repo" not in str(e):
            raise
        raise gr.Error(
            "TRELLIS.2 needs the gated DINOv3 model. Request access at "
            "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m, then run "
            "`uv run hf auth login` on this machine and try again.",
            duration=None,
        )
    return subject, str(path), str(path), seed


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Dollhouse") as demo:
        with gr.Tabs() as tabs:
            with gr.Tab("Detection", id="detect"):
                gr.Markdown(
                    "Gemma 4 12B. Finds and boxes objects in an image. "
                    "Change the prompt to look for specific things, e.g. `detect chairs and lamps`."
                )
                with gr.Row():
                    with gr.Column():
                        detect_in = gr.Image(label="Input image", type="pil", image_mode="RGB", height=420)
                        detect_prompt = gr.Textbox(label="Prompt", value=detect.DEFAULT_PROMPT, lines=2)
                        run_detect = gr.Button("Detect", variant="primary")
                    with gr.Column():
                        detect_out = gr.Image(label="Detections", type="pil", format="png", height=420)
                        detect_table = gr.Dataframe(
                            headers=["label", "x1", "y1", "x2", "y2"], label="Boxes (pixels)", interactive=False,
                        )
                        with gr.Accordion("Raw model output", open=False):
                            detect_text = gr.Textbox(show_label=False, lines=8)

            with gr.Tab("Masking", id="mask"):
                gr.Markdown(
                    "SAM 3. Masks every instance of each prompt. Separate prompts with commas; "
                    "short noun phrases work best, e.g. `couch, floor lamp`. "
                    "Hover over a mask or its legend entry to highlight it."
                )
                with gr.Row():
                    with gr.Column():
                        mask_in = gr.Image(label="Input image", type="pil", image_mode="RGB", height=420)
                        mask_prompts = gr.Textbox(label="Prompts", value=segment.DEFAULT_PROMPTS, lines=2)
                        mask_threshold = gr.Slider(
                            0.05, 0.95, value=segment.DEFAULT_THRESHOLD, step=0.05, label="Score threshold",
                            info="Lower finds more instances, including wrong ones.",
                        )
                        run_mask = gr.Button("Segment", variant="primary")
                    with gr.Column():
                        mask_out = gr.AnnotatedImage(label="Masks", height=420)
                        mask_table = gr.Dataframe(
                            headers=["instance", "score", "area %"], label="Instances", interactive=False,
                        )

            with gr.Tab("Image", id="image"):
                gr.Markdown(
                    "Qwen-Image 2.1. Text only generates an image from the prompt. "
                    "With input images, it edits or combines them according to the prompt (up to 10)."
                )
                with gr.Row():
                    with gr.Column():
                        prompt = gr.Textbox(label="Prompt", value=image_gen.DEFAULT_PROMPT, lines=4)
                        gallery = gr.Gallery(
                            label="Input images (optional)",
                            type="pil",
                            interactive=True,
                            columns=4,
                            height=240,
                        )
                        with gr.Row():
                            aspect_ratio = gr.Dropdown(
                                [AUTO, *image_gen.ASPECT_RATIOS], value=AUTO, label="Aspect ratio",
                                info="Auto uses 1:1 when there is no input image.",
                            )
                            size = gr.Radio(list(SIZES), value="Half (~1 MP, faster)", label="Size")
                        steps = gr.Slider(10, 60, value=40, step=1, label="Steps")
                        with gr.Row():
                            image_seed = gr.Number(value=42, precision=0, label="Seed")
                            image_random = gr.Checkbox(value=True, label="Random seed")
                        run_image = gr.Button("Generate", variant="primary")
                    with gr.Column():
                        image_out = gr.Image(label="Result", type="pil", format="png")
                        to_3d = gr.Button("Send to 3D tab →")

            with gr.Tab("3D", id="3d"):
                gr.Markdown(
                    "TRELLIS.2. Turns one image of an object into a textured 3D model (GLB). "
                    "The background is removed automatically unless the image already has transparency."
                )
                with gr.Row():
                    with gr.Column():
                        mesh_in = gr.Image(label="Input image", type="pil", image_mode="RGBA", height=360)
                        resolution = gr.Radio(
                            list(mesh_gen.PIPELINE_TYPES), value="1024", label="Resolution",
                            info="Voxel grid size. 512 is fastest; 1536 is most detailed.",
                        )
                        with gr.Row():
                            decimation_target = gr.Slider(
                                100_000, 1_000_000, value=500_000, step=10_000, label="Target face count",
                            )
                            texture_size = gr.Radio([1024, 2048, 4096], value=2048, label="Texture size")
                        with gr.Row():
                            mesh_seed = gr.Number(value=42, precision=0, label="Seed")
                            mesh_random = gr.Checkbox(value=True, label="Random seed")
                        run_mesh = gr.Button("Generate 3D model", variant="primary")
                    with gr.Column():
                        mesh_out = gr.Model3D(label="Result", height=480)
                        with gr.Row():
                            subject_out = gr.Image(label="Subject as seen by the model", height=200)
                            glb_file = gr.File(label="Download GLB")

        run_detect.click(
            detect_objects,
            inputs=[detect_in, detect_prompt],
            outputs=[detect_out, detect_table, detect_text],
            **GPU_QUEUE,
        )
        run_mask.click(
            segment_objects,
            inputs=[mask_in, mask_prompts, mask_threshold],
            outputs=[mask_out, mask_table],
            **GPU_QUEUE,
        )
        run_image.click(
            generate_image,
            inputs=[prompt, gallery, aspect_ratio, size, steps, image_seed, image_random],
            outputs=[image_out, image_seed],
            **GPU_QUEUE,
        )
        to_3d.click(
            lambda image: (image, gr.Tabs(selected="3d")),
            inputs=image_out,
            outputs=[mesh_in, tabs],
        )
        run_mesh.click(
            generate_mesh,
            inputs=[mesh_in, resolution, decimation_target, texture_size, mesh_seed, mesh_random],
            outputs=[subject_out, mesh_out, glb_file, mesh_seed],
            **GPU_QUEUE,
        )
    return demo


def main() -> None:
    build_ui().queue().launch(server_name="127.0.0.1", server_port=7860)
