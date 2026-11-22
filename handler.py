"""Runpod Serverless entry point: SDXL / FLUX.1 LoRA training with kohya-ss/sd-scripts.

Local dry run (no GPU, no model weights):
    python handler.py --test_input '{"input": {"dry_run": true, "dataset_url": "https://.../data.zip"}}'
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
import time
import traceback
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import __version__  # noqa: E402
from src.config import ConfigError, resolve_config  # noqa: E402
from src.dataset import DatasetError, prepare_dataset  # noqa: E402
from src.paths import configure_caches, has_volume, model_cache_dir, output_root  # noqa: E402
from src.secret_utils import RedactFilter, get_hf_token, redact  # noqa: E402

configure_caches()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
for h in logging.getLogger().handlers:
    h.addFilter(RedactFilter())
log = logging.getLogger("image-lora-worker")


def _progress_fn(job: dict):
    def send(payload: dict) -> None:
        log.info("progress: %s", payload)
        if job.get("id") and not str(job["id"]).startswith("local") and os.environ.get("RUNPOD_POD_ID"):
            try:
                import runpod
                runpod.serverless.progress_update(job, payload)
            except Exception as exc:  # best effort
                log.debug("progress_update failed: %s", exc)
    return send


def handler(job: dict) -> dict[str, Any]:
    result = _handle(job)
    # Always start the next job on a fresh worker (frees GPU memory, drops partial downloads).
    result["refresh_worker"] = True
    return result


def _handle(job: dict) -> dict[str, Any]:
    started = time.time()
    try:
        cfg = resolve_config(job.get("input") or {})
    except ConfigError as exc:
        return {"error": f"invalid input: {exc}"}

    run_name = cfg.run_name or str(job.get("id") or f"run-{int(started)}")
    output_name = cfg.output_name or "_".join(
        x for x in ((cfg.trigger_word or "lora").replace(" ", "_"), cfg.base_model.replace("-", "_"), "lora") if x)
    token = get_hf_token()
    progress = _progress_fn(job)
    persistent = has_volume()
    run_dir = os.path.join(output_root(), run_name)
    work = os.path.join("/tmp/image-lora/work", run_name)
    images_dir = os.path.join(work, "images")
    shutil.rmtree(work, ignore_errors=True)

    result: dict[str, Any] = {
        "worker_version": __version__,
        "run_name": run_name,
        "base_model": cfg.base_model,
        "config": cfg.public_dict(),
        "warnings": list(cfg.warnings),
    }
    if not persistent:
        result["warnings"].append("no network volume: base models are re-downloaded on every cold start "
                                  "(attach one to cache them under /runpod-volume/image-lora/models)")

    # --- dataset (before any GPU work) -------------------------------------
    captioner = None
    if cfg.caption_mode == "florence2" and not cfg.dry_run:
        from src.captions import florence_captioner
        captioner = florence_captioner(cache_dir=os.path.join(model_cache_dir(), "hf"))
    try:
        progress({"stage": "preparing_dataset"})
        report = prepare_dataset(cfg, token, work, images_dir, captioner=captioner)
    except DatasetError as exc:
        return {"error": f"dataset: {exc}", **{k: result[k] for k in ("run_name", "config")}}
    except Exception as exc:
        log.error("dataset failed: %s", redact(traceback.format_exc()))
        return {"error": f"dataset failed: {redact(str(exc)).splitlines()[0][:400]}"}
    result["dataset"] = report.as_dict()
    result["warnings"] += report.warnings

    from src.train import build_prompts
    prompts = build_prompts(cfg)
    result["sample_prompts"] = prompts

    if cfg.dry_run:
        from src.models import check_access
        access = check_access(cfg, token)
        result.update({
            "dry_run": True,
            "status": "ok" if all(v.get("ok") for v in access.values()) else "error",
            "model_access": access,
            "model_download_gb": round(sum(v.get("size_gb", 0) for v in access.values()), 2),
            "elapsed_seconds": round(time.time() - started, 2),
        })
        if result["status"] == "error":
            result["error"] = "base model files not reachable: " + "; ".join(
                f"{k}: {v.get('error')}" for k, v in access.items() if not v.get("ok"))
        shutil.rmtree(work, ignore_errors=True)
        return result

    # --- models --------------------------------------------------------------
    from src.models import ModelError, ensure_models
    from src.train import (TrainingError, build_command, find_outputs, gpu_info, run_training,
                           write_dataset_toml, write_prompt_file)

    gpu = gpu_info()
    result["gpu"] = gpu
    if not gpu["gpu"]:
        return {"error": "no CUDA GPU visible to the worker", **{k: result[k] for k in ("run_name", "config")}}
    if cfg.is_flux and gpu["vram_gb"] < 22:
        result["warnings"].append(f"FLUX on {gpu['vram_gb']} GB VRAM is slow (block swap); use a 24-48 GB GPU")
    try:
        models, minfo = ensure_models(cfg, token, progress)
    except ModelError as exc:
        return {"error": str(exc), **{k: result[k] for k in ("run_name", "config")}}
    result["models"] = minfo

    # --- training --------------------------------------------------------------
    os.makedirs(run_dir, exist_ok=True)
    toml_path = os.path.join(work, "dataset.toml")
    write_dataset_toml(cfg, images_dir, toml_path, report.images_used)
    prompt_file = None
    if prompts:
        prompt_file = os.path.join(work, "sample_prompts.txt")
        write_prompt_file(cfg, prompts, prompt_file)
    cmd = build_command(cfg, models, toml_path, run_dir, output_name, prompt_file, gpu["vram_gb"])
    log_path = os.path.join(run_dir, "train.log")
    timeout_s = int(os.environ.get("TRAIN_TIMEOUT_SECONDS", "0")) or None
    try:
        progress({"stage": "training", "step": 0, "total_steps": cfg.steps})
        train = run_training(cmd, log_path, cfg.steps, progress, timeout_s=timeout_s)
    except TrainingError as exc:
        outs = find_outputs(run_dir, output_name)
        return {"error": redact(str(exc))[:4000], "run_name": run_name, "gpu": gpu,
                "checkpoints": outs["checkpoints"], "log_path": log_path}
    except Exception as exc:
        log.error("training crashed: %s", redact(traceback.format_exc()))
        return {"error": f"training crashed: {redact(str(exc)).splitlines()[0][:500]}", "run_name": run_name}
    result["training"] = train

    outs = find_outputs(run_dir, output_name)
    if not outs["lora_path"]:
        return {"error": "training finished but no LoRA file was written", "run_name": run_name,
                "training": train, "log_path": log_path}

    # --- deliver ---------------------------------------------------------------
    from src.outputs import deliver, encode_sample
    progress({"stage": "uploading"})
    final_samples = [s for s in outs["samples"] if f"_{cfg.steps:06d}_" in os.path.basename(s)]
    before_samples = [s for s in outs["samples"] if "_000000_" in os.path.basename(s)]
    artifacts, warns = deliver(cfg, run_name, output_name, outs["lora_path"], final_samples + before_samples,
                               prompts, train, token, persistent)
    result["warnings"] += warns
    artifacts["checkpoints"] = [os.path.basename(c) for c in outs["checkpoints"]]
    result["artifacts"] = artifacts
    samples_out = []
    for i, s in enumerate(final_samples):
        item: dict[str, Any] = {"prompt": prompts[i] if i < len(prompts) else None, "file": os.path.basename(s)}
        if cfg.return_samples_base64:
            item["image_base64"] = encode_sample(s)
            item["mime"] = "image/jpeg"
        samples_out.append(item)
    result["samples"] = samples_out
    if before_samples:
        result["samples_before_training"] = [
            {"prompt": prompts[i] if i < len(prompts) else None, "file": os.path.basename(s),
             **({"image_base64": encode_sample(s, max_side=512, quality=80), "mime": "image/jpeg"}
                if cfg.return_samples_base64 else {})}
            for i, s in enumerate(before_samples)]
    result.update({"status": "completed", "elapsed_seconds": round(time.time() - started, 1)})
    shutil.rmtree(work, ignore_errors=True)
    return result


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
