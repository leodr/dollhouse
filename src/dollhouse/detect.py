import json
import re
from dataclasses import dataclass

import torch
from PIL import Image, ImageDraw, ImageFont

from dollhouse import models

MODEL_ID = "google/gemma-4-12B-it"
DEFAULT_PROMPT = "detect all objects"
# Image token budget per image (the model supports 70-1120). The default of 280
# is tuned for classification; localization benefits from more detail.
IMAGE_TOKENS = 1120

# Distinct, readable colours; labels are assigned in order of first appearance.
PALETTE = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#46f0f0",
           "#f032e6", "#bcf60c", "#008080", "#9a6324", "#800000", "#000075"]


@dataclass
class Detection:
    label: str
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in pixels


def _load():
    from transformers import AutoModelForMultimodalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    processor.image_processor.max_soft_tokens = IMAGE_TOKENS
    model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    return processor, model


def get_pipeline():
    return models.acquire("gemma4", _load)


def run(image: Image.Image, prompt: str, max_new_tokens: int = 1024) -> str:
    processor, model = get_pipeline()
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": prompt},
    ]}]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    try:
        with torch.inference_mode():
            # Greedy: detection should be reproducible, not creative.
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        torch.cuda.empty_cache()


def parse(text: str, width: int, height: int) -> list[Detection]:
    """Read `[{"box_2d": [y1, x1, y2, x2], "label": ...}, ...]` with 0-1000 coordinates."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    payload = match.group(1) if match else text[text.find("["): text.rfind("]") + 1]
    try:
        items = json.loads(payload)
    except json.JSONDecodeError:
        return []

    detections = []
    for item in items if isinstance(items, list) else []:
        box = item.get("box_2d") if isinstance(item, dict) else None
        if not (isinstance(box, list) and len(box) == 4):
            continue
        y1, x1, y2, x2 = (float(v) / 1000 for v in box)
        detections.append(Detection(
            label=str(item.get("label", "object")),
            box=(round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)),
        ))
    return detections


def draw(image: Image.Image, detections: list[Detection]) -> Image.Image:
    out = image.convert("RGB")
    draw = ImageDraw.Draw(out)
    line = max(2, round(max(out.size) / 400))
    font = ImageFont.load_default(size=max(14, round(max(out.size) / 50)))
    colors: dict[str, str] = {}
    for det in detections:
        color = colors.setdefault(det.label, PALETTE[len(colors) % len(PALETTE)])
        x1, y1, x2, y2 = det.box
        draw.rectangle((x1, y1, x2, y2), outline=color, width=line)
        tx1, ty1, tx2, ty2 = draw.textbbox((x1, y1), det.label, font=font)
        pad = line
        label_top = max(0, y1 - (ty2 - ty1) - 2 * pad)
        draw.rectangle((x1, label_top, x1 + (tx2 - tx1) + 2 * pad, label_top + (ty2 - ty1) + 2 * pad), fill=color)
        draw.text((x1 + pad, label_top + pad - (ty1 - y1)), det.label, fill="white", font=font)
    return out
