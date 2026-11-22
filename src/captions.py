"""Optional auto-captioning with Florence-2 (only for images that have no .txt caption)."""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)

FLORENCE_MODEL = os.environ.get("FLORENCE_MODEL", "florence-community/Florence-2-base")


def florence_captioner(cache_dir: str | None = None):
    """Return a callable(paths) -> list[str]. Loads the model lazily; frees GPU memory afterwards."""

    def run(paths: list[str]) -> list[str]:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        processor = AutoProcessor.from_pretrained(FLORENCE_MODEL, cache_dir=cache_dir)
        model = AutoModelForImageTextToText.from_pretrained(FLORENCE_MODEL, dtype=dtype,
                                                            cache_dir=cache_dir).to(device).eval()
        task = "<DETAILED_CAPTION>"
        out = []
        try:
            for p in paths:
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    inputs = processor(text=task, images=im, return_tensors="pt").to(device, dtype)
                with torch.inference_mode():
                    ids = model.generate(**inputs, max_new_tokens=120, num_beams=3, do_sample=False)
                text = processor.batch_decode(ids, skip_special_tokens=False)[0]
                parsed = processor.post_process_generation(text, task=task, image_size=im.size)
                cap = parsed.get(task, "") if isinstance(parsed, dict) else str(parsed)
                cap = cap.replace("<pad>", "").replace("</s>", "").replace("<s>", "").strip()
                out.append(" ".join(cap.split()))
        finally:
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
        return out

    return run
