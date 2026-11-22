"""Fetch, unpack and validate the training images; write the kohya image folder."""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
import tarfile
import zipfile
from dataclasses import dataclass, field
from typing import Any

import requests

from .secret_utils import redact

log = logging.getLogger(__name__)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
MAX_DOWNLOAD_BYTES = 4 * 2**30          # 4 GiB archive limit
MAX_UNPACKED_BYTES = 8 * 2**30
MIN_SIDE = 256


class DatasetError(ValueError):
    pass


@dataclass
class DatasetReport:
    source: str
    images_found: int = 0
    images_used: int = 0
    images_skipped: list[str] = field(default_factory=list)
    captions_from_files: int = 0
    captions_generated: int = 0
    captions_trigger_only: int = 0
    min_side_px: int = 0
    max_side_px: int = 0
    small_images: int = 0
    example_captions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["images_skipped"] = self.images_skipped[:20]
        return d


def _download(url: str, dest: str) -> str:
    log.info("downloading dataset archive")
    with requests.get(url, stream=True, timeout=(15, 300), allow_redirects=True) as r:
        if r.status_code != 200:
            raise DatasetError(f"dataset_url returned HTTP {r.status_code}")
        size = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise DatasetError("dataset archive is larger than 4 GiB")
                f.write(chunk)
    return dest


def _safe_member(base: str, name: str) -> str | None:
    target = os.path.realpath(os.path.join(base, name))
    if not target.startswith(os.path.realpath(base) + os.sep):
        return None
    return target


def _extract(archive: str, dest: str) -> None:
    os.makedirs(dest, exist_ok=True)
    total = 0
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            for m in z.infolist():
                if m.is_dir():
                    continue
                total += m.file_size
                if total > MAX_UNPACKED_BYTES:
                    raise DatasetError("dataset archive unpacks to more than 8 GiB")
                target = _safe_member(dest, m.filename)
                if target is None:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with z.open(m) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
        return
    try:
        with tarfile.open(archive) as t:
            for m in t.getmembers():
                if not m.isfile():
                    continue
                total += m.size
                if total > MAX_UNPACKED_BYTES:
                    raise DatasetError("dataset archive unpacks to more than 8 GiB")
                target = _safe_member(dest, m.name)
                if target is None:
                    continue
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with t.extractfile(m) as src, open(target, "wb") as out:
                    shutil.copyfileobj(src, out)
    except tarfile.ReadError as exc:
        raise DatasetError("dataset_url must point to a .zip or .tar(.gz) archive") from exc


def _collect_pairs(root: str) -> list[tuple[str, str | None]]:
    """(image_path, caption_text|None) for every image under root (skips macOS junk)."""
    pairs = []
    for dirpath, dirnames, files in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith((".", "__MACOSX"))]
        for f in sorted(files):
            if f.startswith(".") or not f.lower().endswith(IMAGE_EXTS):
                continue
            img = os.path.join(dirpath, f)
            cap = None
            txt = os.path.splitext(img)[0] + ".txt"
            if os.path.isfile(txt):
                with open(txt, encoding="utf-8", errors="replace") as fh:
                    cap = re.sub(r"\s+", " ", fh.read()).strip() or None
            pairs.append((img, cap))
    pairs.sort(key=lambda x: x[0])
    return pairs


def _fetch_hf_dataset(cfg, token: str | None, work: str) -> list[tuple[str, str | None]]:
    from huggingface_hub import snapshot_download  # lazy

    local = snapshot_download(repo_id=cfg.hf_dataset, repo_type="dataset", token=token,
                              local_dir=os.path.join(work, "hf"),
                              allow_patterns=["*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG",
                                              "*.txt", "*.parquet", "*.jsonl", "*.json", "*.csv"])
    pairs = _collect_pairs(local)
    if pairs:
        return pairs
    # Table dataset (parquet etc.) with an image column.
    import datasets  # lazy

    ds = datasets.load_dataset(local, split=cfg.hf_dataset_split)
    cols = ds.column_names
    if cfg.image_column not in cols:
        raise DatasetError(f"hf_dataset has no image column '{cfg.image_column}' (columns: {cols})")
    cap_col = cfg.caption_column or next((c for c in ("text", "caption", "prompt", "description") if c in cols), "")
    out_dir = os.path.join(work, "hf_rows")
    os.makedirs(out_dir, exist_ok=True)
    pairs = []
    for i, row in enumerate(ds):
        if i >= cfg.max_images:
            break
        img = row[cfg.image_column]
        path = os.path.join(out_dir, f"{i:05d}.png")
        img.convert("RGB").save(path)
        cap = str(row[cap_col]).strip() if cap_col and row.get(cap_col) else None
        pairs.append((path, cap))
    return pairs


def _base_caption(cfg) -> str:
    return " ".join(x for x in (cfg.trigger_word, cfg.class_word) if x).strip()


def prepare_dataset(cfg, token: str | None, work: str, out_images: str,
                    captioner=None) -> DatasetReport:
    """Download + validate; write images and .txt captions into out_images."""
    from PIL import Image, ImageOps  # lazy

    os.makedirs(work, exist_ok=True)
    if cfg.dataset_url:
        report = DatasetReport(source="url")
        archive = os.path.join(work, "dataset.archive")
        try:
            _download(cfg.dataset_url, archive)
        except DatasetError:
            raise
        except Exception as exc:
            raise DatasetError(f"could not download dataset_url: {redact(str(exc))[:300]}") from exc
        _extract(archive, os.path.join(work, "unpacked"))
        os.remove(archive)
        pairs = _collect_pairs(os.path.join(work, "unpacked"))
    else:
        report = DatasetReport(source=f"hf:{cfg.hf_dataset}")
        try:
            pairs = _fetch_hf_dataset(cfg, token, work)
        except DatasetError:
            raise
        except Exception as exc:
            raise DatasetError(f"could not load hf_dataset: {redact(str(exc)).splitlines()[0][:300]}") from exc

    report.images_found = len(pairs)
    if not pairs:
        raise DatasetError("no images found (expected .jpg/.jpeg/.png/.webp files, captions as same-name .txt)")
    if len(pairs) > cfg.max_images:
        report.warnings.append(f"{len(pairs)} images found, using the first {cfg.max_images} (max_images)")
        pairs = pairs[: cfg.max_images]

    os.makedirs(out_images, exist_ok=True)
    base = _base_caption(cfg)
    need_caption: list[tuple[str, str]] = []
    sides = []
    used = 0
    for i, (src, cap) in enumerate(pairs):
        try:
            with Image.open(src) as im:
                im = ImageOps.exif_transpose(im)
                if im.mode != "RGB":
                    im = im.convert("RGB")
                w, h = im.size
                if min(w, h) < MIN_SIDE:
                    report.images_skipped.append(f"{os.path.basename(src)}: too small ({w}x{h})")
                    continue
                if min(w, h) < cfg.resolution * 0.75:
                    report.small_images += 1
                sides += [min(w, h), max(w, h)]
                dst = os.path.join(out_images, f"{i:05d}.png" if src.lower().endswith(".png") else f"{i:05d}.jpg")
                if dst.endswith(".png"):
                    im.save(dst)
                else:
                    im.save(dst, quality=95)
        except Exception as exc:
            report.images_skipped.append(f"{os.path.basename(src)}: unreadable ({str(exc)[:80]})")
            continue
        used += 1
        mode = cfg.caption_mode
        if mode == "trigger" or (cap is None and mode in ("auto", "existing")):
            text = base
            report.captions_trigger_only += 1
            if cap is None and mode == "existing":
                report.warnings.append("caption_mode=existing but some images have no .txt; used trigger word")
        elif cap is None and mode == "florence2":
            need_caption.append((dst, ""))
            text = None
        else:
            text = cap
            report.captions_from_files += 1
        if text is not None:
            _write_caption(dst, text, cfg)
    report.images_used = used
    if used == 0:
        raise DatasetError("no usable images (all were unreadable or smaller than 256 px)")

    if need_caption:
        caps = None
        if captioner is not None:
            try:
                caps = captioner([p for p, _ in need_caption])
            except Exception as exc:
                report.warnings.append(f"auto-caption failed ({redact(str(exc))[:200]}); used trigger word instead")
        for j, (p, _) in enumerate(need_caption):
            if caps and caps[j]:
                _write_caption(p, caps[j], cfg)
                report.captions_generated += 1
            else:
                _write_caption(p, base, cfg)
                report.captions_trigger_only += 1

    if sides:
        report.min_side_px, report.max_side_px = min(sides), max(sides)
    if used < 5:
        report.warnings.append(f"only {used} images; 10-30 varied images usually give a better LoRA")
    if report.small_images:
        report.warnings.append(f"{report.small_images} image(s) are smaller than 75% of resolution "
                               f"{cfg.resolution}px and will be upscaled")
    if report.captions_trigger_only and not cfg.trigger_word:
        report.warnings.append("some images have no caption and no trigger_word was set")
    for f in sorted(os.listdir(out_images)):
        if f.endswith(".txt") and len(report.example_captions) < 3:
            with open(os.path.join(out_images, f), encoding="utf-8") as fh:
                report.example_captions.append(fh.read().strip()[:300])
    return report


def _write_caption(image_path: str, text: str, cfg) -> None:
    text = (text or "").strip()
    tw = cfg.trigger_word
    if tw and cfg.prepend_trigger and not re.search(rf"(^|\W){re.escape(tw)}(\W|$)", text, re.I):
        text = f"{tw}, {text}" if text else tw
    with open(os.path.splitext(image_path)[0] + ".txt", "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
