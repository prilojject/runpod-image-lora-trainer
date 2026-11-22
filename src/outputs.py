"""Deliver the LoRA: HF Hub repo, S3-compatible bucket, presigned PUT URL; encode samples."""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

import requests

from .secret_utils import redact

log = logging.getLogger(__name__)

BUCKET_ENV = ("BUCKET_ENDPOINT_URL", "BUCKET_ACCESS_KEY_ID", "BUCKET_SECRET_ACCESS_KEY")


def bucket_configured() -> bool:
    return all(os.environ.get(k) for k in BUCKET_ENV)


def encode_sample(path: str, max_side: int = 1024, quality: int = 85) -> str:
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def model_card(cfg, output_name: str, prompts: list[str], train: dict[str, Any]) -> str:
    base = {"sdxl": "stabilityai/stable-diffusion-xl-base-1.0", "flux-dev": "black-forest-labs/FLUX.1-dev",
            "flux-schnell": "black-forest-labs/FLUX.1-schnell"}[cfg.base_model]
    lic = {"sdxl": "openrail++", "flux-dev": "other", "flux-schnell": "apache-2.0"}[cfg.base_model]
    lines = [
        "---",
        f"base_model: {base}",
        f"license: {lic}",
    ]
    if cfg.base_model == "flux-dev":
        lines += ["license_name: flux-1-dev-non-commercial-license",
                  "license_link: https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md"]
    lines += ["tags:", "- lora", "- text-to-image", f"- {'flux' if cfg.is_flux else 'stable-diffusion-xl'}",
              "- kohya-ss", "- runpod"]
    if cfg.trigger_word:
        lines.append(f"instance_prompt: {cfg.trigger_word}")
    lines += ["---", "", f"# {output_name}", "",
              f"LoRA for **{base}** trained with [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) "
              "on Runpod Serverless using "
              "[prilojject/runpod-image-lora-trainer](https://github.com/prilojject/runpod-image-lora-trainer).", ""]
    if cfg.trigger_word:
        lines += [f"Trigger word: `{cfg.trigger_word}`", ""]
    lines += [f"- preset: {cfg.preset}", f"- steps: {cfg.steps}", f"- rank/alpha: {cfg.rank}/{cfg.alpha:g}",
              f"- learning rate: {cfg.learning_rate:g}", f"- resolution: {cfg.resolution}",
              f"- final avr_loss: {train.get('final_avr_loss')}", ""]
    if prompts:
        lines += ["Sample prompts:", ""] + [f"- {p}" for p in prompts] + [""]
    if cfg.base_model == "flux-dev":
        lines += ["This LoRA is a derivative of FLUX.1 [dev] and is subject to the "
                  "FLUX.1 [dev] Non-Commercial License.", ""]
    return "\n".join(lines)


def push_to_hf(cfg, token: str, lora_path: str, samples: list[str], card: str) -> str:
    from huggingface_hub import CommitOperationAdd, HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id=cfg.hf_repo_id, repo_type="model", private=cfg.hf_private, exist_ok=True)
    ops = [CommitOperationAdd(path_in_repo=os.path.basename(lora_path), path_or_fileobj=lora_path),
           CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode())]
    for s in samples:
        ops.append(CommitOperationAdd(path_in_repo=f"samples/{os.path.basename(s)}", path_or_fileobj=s))
    api.create_commit(repo_id=cfg.hf_repo_id, repo_type="model", operations=ops,
                      commit_message=f"LoRA {os.path.basename(lora_path)} ({cfg.base_model}, {cfg.steps} steps)")
    return f"https://huggingface.co/{cfg.hf_repo_id}"


def upload_bucket(path: str, prefix: str) -> str:
    from runpod.serverless.utils.rp_upload import upload_file_to_bucket
    return upload_file_to_bucket(file_name=os.path.basename(path), file_location=path,
                                 bucket_name=os.environ.get("BUCKET_NAME") or None, prefix=prefix)


def put_presigned(url: str, path: str) -> int:
    with open(path, "rb") as f:
        r = requests.put(url, data=f, headers={"Content-Type": "application/octet-stream"}, timeout=(15, 900))
    if r.status_code >= 300:
        raise RuntimeError(f"presigned PUT returned HTTP {r.status_code}: {r.text[:200]}")
    return r.status_code


def deliver(cfg, run_name: str, output_name: str, lora_path: str, samples: list[str], prompts: list[str],
            train: dict[str, Any], token: str | None, persistent: bool) -> tuple[dict[str, Any], list[str]]:
    out: dict[str, Any] = {
        "lora_file": os.path.basename(lora_path),
        "lora_path": lora_path,
        "lora_size_mb": round(os.path.getsize(lora_path) / 2**20, 2),
        "persistent_storage": persistent,
    }
    warnings: list[str] = []
    delivered = persistent
    card = model_card(cfg, output_name, prompts, train)
    with open(os.path.join(os.path.dirname(lora_path), "README.md"), "w", encoding="utf-8") as f:
        f.write(card)

    if cfg.hf_repo_id and token:
        try:
            out["hf_repo_url"] = push_to_hf(cfg, token, lora_path, samples, card)
            delivered = True
        except Exception as exc:
            out["hf_error"] = redact(str(exc)).splitlines()[0][:400]
            log.error("HF upload failed: %s", out["hf_error"])
    if cfg.upload_url:
        try:
            out["upload_url_status"] = put_presigned(cfg.upload_url, lora_path)
            delivered = True
        except Exception as exc:
            out["upload_url_error"] = redact(str(exc))[:400].replace(cfg.upload_url, "<presigned url>")
    if cfg.upload_to_bucket and bucket_configured():
        try:
            prefix = f"image-lora/{run_name}"
            out["bucket_lora_url"] = upload_bucket(lora_path, prefix)
            out["bucket_sample_urls"] = [upload_bucket(s, prefix + "/samples") for s in samples]
            delivered = True
        except Exception as exc:
            out["bucket_error"] = redact(str(exc))[:400]
    if not delivered:
        warnings.append("the LoRA was not delivered anywhere persistent: attach a network volume, or set "
                        "hf_repo_id (+HF_TOKEN), upload_url, or BUCKET_* env vars. It is lost when the worker stops.")
    return out, warnings
