"""Prompt rewriting for Qwen-Image, with Qwen-Image-2.1-PE-T2I.

The model turns a short request into the detailed English prompt Qwen-Image-2.1 was
trained on, and picks the aspect ratio to render it at. It reasons in a <think>
block first. It is trained on text-only requests; an image can be passed along with
the text, since the base model (Qwen3.5-VL) reads images.
"""

import json
import re

import torch
from PIL import Image

from dollhouse import models

MODEL_ID = "Qwen/Qwen-Image-2.1-PE-T2I"
# Its reasoning is usually a few hundred to a few thousand tokens; the model card
# allows 16256, which would only matter for runaway generations.
MAX_NEW_TOKENS = 8192
RATIO = re.compile(r"^\s*(\d+)\s*:\s*(\d+)\s*$")


def _load():
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda").eval()
    system_prompt = open(hf_hub_download(MODEL_ID, "system_prompt.txt")).read().strip()
    return processor, model, system_prompt


def get_pipeline():
    return models.acquire("qwen-pe", _load)


def rewrite(request: str, image: Image.Image | None = None, seed: int = 0) -> tuple[str, str | None, str]:
    """The rewritten prompt, the aspect ratio ("w:h") and the model's full output.

    Falls back to `request` unchanged and no ratio if the answer is not the expected JSON.
    """
    processor, model, system_prompt = get_pipeline()
    content = [{"type": "image", "image": image.convert("RGB")}] if image is not None else []
    content.append({"type": "text", "text": request})
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": content},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt", enable_thinking=True,
    ).to(model.device)
    torch.manual_seed(seed)
    try:
        with torch.inference_mode():
            # Sampling settings from the model card.
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
            )
        text = processor.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    finally:
        torch.cuda.empty_cache()
    answer = text.partition("</think>")[2].strip()
    answer = answer[answer.find("{"): answer.rfind("}") + 1]
    try:
        parsed = json.loads(answer)
        rewritten = str(parsed["rewritten_prompt"]).strip()
        ratio = str(parsed.get("wh_ratio", ""))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        rewritten, ratio = "", ""
    return rewritten or request, ratio if RATIO.match(ratio) else None, text


def size_for_ratio(ratio: str, pixels: int) -> tuple[int, int]:
    """(width, height) in multiples of 32 (as Qwen-Image needs) with about `pixels` pixels, for a "w:h" ratio."""
    w, h = (int(v) for v in RATIO.match(ratio).groups())
    width = (pixels * w / h) ** 0.5
    return round(width / 32) * 32, round(width * h / w / 32) * 32
