from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from dollhouse import models

MODEL_ID = "facebook/sam3"
DEFAULT_PROMPTS = "couch, coffee table, cabinet, floor lamp, cardboard box, floor"
DEFAULT_THRESHOLD = 0.5


@dataclass
class Instance:
    label: str
    score: float
    mask: np.ndarray  # H, W bool at the image's resolution


def _load():
    from transformers import Sam3Model, Sam3Processor

    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model = Sam3Model.from_pretrained(MODEL_ID).to("cuda").eval()
    return processor, model


def get_pipeline():
    return models.acquire("sam3", _load)


def parse_prompts(text: str) -> list[str]:
    """Comma- or newline-separated prompts, without blanks or duplicates."""
    prompts = [p.strip() for line in text.splitlines() for p in line.split(",")]
    return list(dict.fromkeys(p for p in prompts if p))


@torch.no_grad()
def run(image: Image.Image, prompts: list[str], threshold: float = DEFAULT_THRESHOLD) -> list[Instance]:
    """All instances of each prompt, one text prompt per forward pass."""
    processor, model = get_pipeline()
    instances = []
    for prompt in prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
        outputs = model(**inputs)
        result = processor.post_process_instance_segmentation(
            outputs, threshold=threshold, target_sizes=inputs["original_sizes"].tolist()
        )[0]
        for score, mask in zip(result["scores"].tolist(), result["masks"].cpu().numpy()):
            instances.append(Instance(prompt, score, mask.astype(bool)))
    return instances
