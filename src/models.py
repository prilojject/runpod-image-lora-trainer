"""Base model files: download at job start, cache on the network volume if attached."""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import time
from typing import Any

import requests

from .paths import model_cache_dir
from .secret_utils import redact

log = logging.getLogger(__name__)

# (repo_id, filename) for each component. Single-file layouts as expected by kohya sd-scripts.
MODEL_FILES: dict[str, dict[str, tuple[str, str]]] = {
    "sdxl": {
        "checkpoint": ("stabilityai/stable-diffusion-xl-base-1.0", "sd_xl_base_1.0.safetensors"),
    },
    "flux-dev": {
        "checkpoint": ("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors"),
        "ae": ("black-forest-labs/FLUX.1-dev", "ae.safetensors"),
        "clip_l": ("comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
        "t5xxl": ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
    },
    "flux-schnell": {
        "checkpoint": ("black-forest-labs/FLUX.1-schnell", "flux1-schnell.safetensors"),
        "ae": ("black-forest-labs/FLUX.1-schnell", "ae.safetensors"),
        "clip_l": ("comfyanonymous/flux_text_encoders", "clip_l.safetensors"),
        "t5xxl": ("comfyanonymous/flux_text_encoders", "t5xxl_fp16.safetensors"),
    },
}

GATED = {"black-forest-labs/FLUX.1-dev"}


class ModelError(RuntimeError):
    pass


def _parse_hf_file(spec: str) -> tuple[str, str, str | None]:
    owner, rest = spec.split("/", 1)
    repo, path = rest.split("/", 1)
    revision = None
    if "@" in repo:
        repo, revision = repo.split("@", 1)
    return f"{owner}/{repo}", path, revision


def components(cfg) -> dict[str, tuple[str, str, str | None]]:
    """name -> (kind, ref, filename_or_revision). kind = 'hf' | 'url'."""
    files = {k: ("hf", repo, fn) for k, (repo, fn) in MODEL_FILES[cfg.base_model].items()}
    if cfg.model_file:
        if cfg.model_file.startswith("http"):
            files["checkpoint"] = ("url", cfg.model_file, "")
        else:
            files["checkpoint"] = ("hf", cfg.model_file, "")
    if getattr(cfg, "ae_file", "") and "ae" in files:
        files["ae"] = ("url", cfg.ae_file, "") if cfg.ae_file.startswith("http") else ("hf", cfg.ae_file, "")
    return files


def check_access(cfg, token: str | None) -> dict[str, Any]:
    """Metadata-only check (no weights) that every component is reachable with the given token."""
    from huggingface_hub import get_hf_file_metadata, hf_hub_url  # lazy

    out: dict[str, Any] = {}
    for name, (kind, ref, fn) in components(cfg).items():
        try:
            if kind == "url":
                r = requests.head(ref, allow_redirects=True, timeout=20)
                ok = r.status_code < 400
                out[name] = {"ok": ok, "status": r.status_code,
                             "size_gb": round(int(r.headers.get("content-length", 0)) / 1e9, 2)}
                continue
            if fn == "":
                repo, fn, rev = _parse_hf_file(ref)
            else:
                repo, rev = ref, None
            meta = get_hf_file_metadata(hf_hub_url(repo, fn, revision=rev), token=token)
            out[name] = {"ok": True, "repo": repo, "file": fn, "size_gb": round((meta.size or 0) / 1e9, 2)}
        except Exception as exc:
            msg = redact(str(exc)).splitlines()[0][:240]
            hint = ""
            if "401" in msg or "403" in msg or "gated" in msg.lower():
                hint = " (gated: accept the model license on huggingface.co and set HF_TOKEN with read access)"
            out[name] = {"ok": False, "error": msg + hint}
    return out


def _download_url(url: str, dest_dir: str) -> str:
    name = hashlib.sha256(url.split("?")[0].encode()).hexdigest()[:16] + ".safetensors"
    dest = os.path.join(dest_dir, "url", name)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    with requests.get(url, stream=True, timeout=(15, 600), allow_redirects=True) as r:
        if r.status_code != 200:
            raise ModelError(f"model_file URL returned HTTP {r.status_code}")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=8 << 20):
                f.write(chunk)
    os.replace(tmp, dest)
    return dest


def ensure_models(cfg, token: str | None, progress=None) -> tuple[dict[str, str], dict[str, Any]]:
    """Download every component (cached). Returns name -> local path and timing info."""
    from huggingface_hub import hf_hub_download  # lazy

    cache = model_cache_dir()
    os.makedirs(cache, exist_ok=True)
    t0 = time.time()
    paths: dict[str, str] = {}
    for name, (kind, ref, fn) in components(cfg).items():
        repo = ref
        if progress:
            progress({"stage": "downloading_model", "component": name})
        try:
            if kind == "url":
                paths[name] = _download_url(ref, cache)
            else:
                repo, rev = ref, None
                if fn == "":
                    repo, fn, rev = _parse_hf_file(ref)
                paths[name] = hf_hub_download(repo_id=repo, filename=fn, revision=rev, token=token,
                                              cache_dir=os.path.join(cache, "hf"))
        except ModelError:
            raise
        except Exception as exc:
            msg = redact(str(exc)).splitlines()[0][:300]
            if repo in GATED or "401" in msg or "403" in msg:
                msg += " (gated model: accept the license on huggingface.co and set HF_TOKEN)"
            raise ModelError(f"download of {name} failed: {msg}") from exc
    sizes = {k: round(os.path.getsize(v) / 1e9, 2) for k, v in paths.items()}
    free_gb = round(shutil.disk_usage(cache).free / 1e9, 1)
    return paths, {"download_seconds": round(time.time() - t0, 1), "sizes_gb": sizes,
                   "cache_dir": cache, "disk_free_gb": free_gb}
