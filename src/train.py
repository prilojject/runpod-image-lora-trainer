"""Build the kohya sd-scripts command for SDXL / FLUX.1 LoRA training and run it."""

from __future__ import annotations

import glob
import logging
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from typing import Any

from .config import render_prompt
from .secret_utils import redact

log = logging.getLogger(__name__)

SD_SCRIPTS = os.environ.get("SD_SCRIPTS_DIR", "/opt/sd-scripts")

DEFAULT_SAMPLE_STEPS = {"sdxl": 28, "flux-dev": 20, "flux-schnell": 4}
STEP_RE = re.compile(r"steps:\s+\d+%.*?\|\s*(\d+)/(\d+)\s*\[.*?(?:avr_loss=([0-9.]+))?\]")
LOSS_RE = re.compile(r"avr_loss=([0-9.eE+-]+)")


class TrainingError(RuntimeError):
    pass


def gpu_info() -> dict[str, Any]:
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            return {"gpu": p.name, "vram_gb": round(p.total_memory / 2**30, 1)}
    except Exception:  # pragma: no cover
        pass
    return {"gpu": None, "vram_gb": 0.0}


def write_dataset_toml(cfg, images_dir: str, path: str, n_images: int) -> None:
    # Repeat small datasets so an "epoch" is not just a handful of steps (fewer dataloader restarts).
    repeats = max(1, min(100, 100 // max(1, n_images)))
    lines = [
        "[general]",
        'caption_extension = ".txt"',
        "shuffle_caption = false",
        "keep_tokens = 1",
        "",
        "[[datasets]]",
        f"resolution = {cfg.resolution}",
        f"batch_size = {cfg.batch_size}",
        "enable_bucket = true",
        f"min_bucket_reso = {max(256, cfg.resolution // 2)}",
        f"max_bucket_reso = {min(2048, cfg.resolution * 2)}",
        "bucket_no_upscale = false",
        "",
        "  [[datasets.subsets]]",
        f"  image_dir = {images_dir!r}".replace("'", '"'),
        f"  num_repeats = {repeats}",
        f"  flip_aug = {'true' if cfg.flip_aug else 'false'}",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_prompts(cfg) -> list[str]:
    return [render_prompt(p, cfg.trigger_word, cfg.class_word or "") for p in cfg.sample_prompts]


def write_prompt_file(cfg, prompts: list[str], path: str) -> None:
    steps = cfg.sample_steps or DEFAULT_SAMPLE_STEPS[cfg.base_model]
    res = cfg.resolution
    out = []
    for i, p in enumerate(prompts):
        clean = p.replace("\n", " ").replace(" --", " -")
        if cfg.base_model == "sdxl":
            out.append(f"{clean} --n lowres, blurry, deformed, watermark, text --w {res} --h {res} "
                       f"--d {cfg.seed + i} --l 6.0 --s {steps}")
        elif cfg.base_model == "flux-dev":
            out.append(f"{clean} --w {res} --h {res} --d {cfg.seed + i} --g 3.5 --s {steps}")
        else:  # schnell: guidance-distilled, few steps
            out.append(f"{clean} --w {res} --h {res} --d {cfg.seed + i} --g 0.0 --s {steps}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out) + "\n")


def build_command(cfg, models: dict[str, str], dataset_toml: str, out_dir: str, output_name: str,
                  prompt_file: str | None, vram_gb: float) -> list[str]:
    common = [
        f"--dataset_config={dataset_toml}",
        f"--output_dir={out_dir}",
        f"--output_name={output_name}",
        "--save_model_as=safetensors",
        f"--network_dim={cfg.rank}",
        f"--network_alpha={cfg.alpha:g}",
        f"--learning_rate={cfg.learning_rate:g}",
        f"--max_train_steps={cfg.steps}",
        f"--lr_scheduler={cfg.lr_scheduler}",
        f"--lr_warmup_steps={cfg.warmup_steps}",
        f"--seed={cfg.seed}",
        "--mixed_precision=bf16",
        "--gradient_checkpointing",
        "--sdpa",
        "--cache_latents",
        "--cache_text_encoder_outputs",
        "--max_data_loader_n_workers=2",
        "--persistent_data_loader_workers",
        "--console_log_simple",
    ]
    opt = {
        "adamw8bit": ["--optimizer_type=AdamW8bit"],
        "adamw": ["--optimizer_type=AdamW"],
        "adafactor": ["--optimizer_type=Adafactor", "--optimizer_args", "relative_step=False",
                      "scale_parameter=False", "warmup_init=False", "--max_grad_norm=0.0"],
        "prodigy": ["--optimizer_type=prodigy", "--optimizer_args", "decouple=True", "weight_decay=0.01",
                    "d_coef=1", "use_bias_correction=True", "safeguard_warmup=True"],
    }[cfg.optimizer]
    if cfg.save_every_n_steps:
        common.append(f"--save_every_n_steps={cfg.save_every_n_steps}")
    if prompt_file:
        common += [f"--sample_prompts={prompt_file}", f"--sample_every_n_steps={cfg.steps}"]
        if cfg.sample_before:
            common.append("--sample_at_first")

    if cfg.base_model == "sdxl":
        script = os.path.join(SD_SCRIPTS, "sdxl_train_network.py")
        args = [
            f"--pretrained_model_name_or_path={models['checkpoint']}",
            "--network_module=networks.lora",
            "--save_precision=fp16",
            "--no_half_vae",
        ]
        if prompt_file:
            args.append("--sample_sampler=euler_a")
        if cfg.train_text_encoder:
            # TE training is incompatible with cached TE outputs.
            common.remove("--cache_text_encoder_outputs")
            args.append(f"--text_encoder_lr={cfg.learning_rate / 2:g}")
        else:
            args.append("--network_train_unet_only")
    else:
        script = os.path.join(SD_SCRIPTS, "flux_train_network.py")
        args = [
            f"--pretrained_model_name_or_path={models['checkpoint']}",
            f"--clip_l={models['clip_l']}",
            f"--t5xxl={models['t5xxl']}",
            f"--ae={models['ae']}",
            "--network_module=networks.lora_flux",
            "--network_train_unet_only",
            "--save_precision=bf16",
            "--fp8_base",
            "--guidance_scale=1.0",
            "--timestep_sampling=flux_shift",
            "--model_prediction_type=raw",
            "--discrete_flow_shift=3.1582",
            "--loss_type=l2",
        ]
        if vram_gb and vram_gb < 30:
            # 24 GB cards: swap some transformer blocks to CPU to keep headroom for sampling.
            args.append("--blocks_to_swap=8")
        if vram_gb and vram_gb < 20:
            args[-1] = "--blocks_to_swap=18"
    return [sys.executable, "-m", "accelerate.commands.launch", "--num_cpu_threads_per_process=2",
            "--mixed_precision=bf16", "--num_processes=1", "--num_machines=1", "--dynamo_backend=no",
            script, *common, *opt, *args]


def run_training(cmd: list[str], log_path: str, total_steps: int, progress, timeout_s: int | None = None
                 ) -> dict[str, Any]:
    """Run kohya, stream its log, report progress, return parsed stats."""
    log.info("launching: %s", redact(" ".join(shlex.quote(c) for c in cmd)))
    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    t0 = time.time()
    state = {"step": 0, "loss": None, "last_report": 0.0, "first_step_t": None, "tail": []}

    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(cmd, cwd=SD_SCRIPTS, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                env=env, text=True, bufsize=1, errors="replace")

        def watchdog():
            if timeout_s:
                try:
                    proc.wait(timeout=timeout_s)
                except subprocess.TimeoutExpired:
                    log.error("training exceeded %ss, terminating", timeout_s)
                    proc.terminate()

        threading.Thread(target=watchdog, daemon=True).start()
        buf = ""
        while True:
            ch = proc.stdout.read(1)
            if ch == "" and proc.poll() is not None:
                break
            if ch not in ("\n", "\r"):
                buf += ch
                continue
            line, buf = redact(buf), ""
            if not line.strip():
                continue
            logf.write(line + "\n")
            state["tail"] = (state["tail"] + [line])[-60:]
            m = re.search(r"\|\s*(\d+)/(\d+)\s*\[", line)
            if m and "steps" in line:
                step = int(m.group(1))
                if step > 0 and state["first_step_t"] is None:
                    state["first_step_t"] = time.time()
                state["step"] = step
                lm = LOSS_RE.search(line)
                if lm:
                    try:
                        state["loss"] = float(lm.group(1))
                    except ValueError:
                        pass
                now = time.time()
                if now - state["last_report"] > 15:
                    state["last_report"] = now
                    progress({"stage": "training", "step": step, "total_steps": total_steps,
                              "loss": state["loss"]})
            elif not m:
                log.info("kohya: %s", line[:500])
        rc = proc.wait()
    elapsed = time.time() - t0
    if rc != 0:
        tail = "\n".join(state["tail"][-25:])
        raise TrainingError(f"sd-scripts exited with code {rc}. Last log lines:\n{tail[-3000:]}")
    train_s = (time.time() - state["first_step_t"]) if state["first_step_t"] else None
    return {"global_step": state["step"], "final_avr_loss": state["loss"],
            "total_seconds": round(elapsed, 1),
            "train_loop_seconds": round(train_s, 1) if train_s else None}


def find_outputs(out_dir: str, output_name: str) -> dict[str, Any]:
    final = os.path.join(out_dir, f"{output_name}.safetensors")
    ckpts = sorted(glob.glob(os.path.join(out_dir, f"{output_name}-*.safetensors")))
    samples = sorted(glob.glob(os.path.join(out_dir, "sample", "*.png")))
    return {"lora_path": final if os.path.isfile(final) else None, "checkpoints": ckpts, "samples": samples}
