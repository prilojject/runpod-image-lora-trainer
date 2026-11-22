"""Input schema for the image LoRA trainer.

Every parameter comes from (highest priority first):

1. the job ``input`` object,
2. an environment variable (what the Runpod Hub deploy form sets),
3. the preset (face / style / object) defaults,
4. the built-in default.

Dependency-free so it can be unit-tested and used for a dry run without a GPU.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

HF_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
URL_RE = re.compile(r"^https?://\S+$")
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
TRIGGER_RE = re.compile(r"^[^\n\r,]{1,64}$")
# "owner/repo/path/to/file.safetensors" (optionally "@revision" after the repo).
HF_FILE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*(@[A-Za-z0-9._/-]+)?/.+\.safetensors$")

TRUE_STRINGS = {"1", "true", "yes", "y", "on", "enabled"}
FALSE_STRINGS = {"0", "false", "no", "n", "off", "disabled", ""}

BASE_MODELS = ("sdxl", "flux-dev", "flux-schnell")

# Preset defaults. Values are applied only when the field was not set by input or env.
PRESETS: dict[str, dict[str, Any]] = {
    "face": {
        "steps": 1500, "rank": 16, "learning_rate": 1e-4, "flip_aug": False,
        "class_word": "person",
        "sample_prompts": [
            "portrait photo of {trigger}, natural light, 85mm lens, detailed skin",
            "{trigger} as an astronaut on the moon, cinematic lighting",
            "close-up photo of {trigger} smiling in a cafe, shallow depth of field",
            "oil painting of {trigger}, renaissance style",
        ],
    },
    "style": {
        "steps": 2000, "rank": 32, "learning_rate": 1e-4, "flip_aug": True,
        "class_word": "",
        "sample_prompts": [
            "a quiet city street at dusk, {trigger} style",
            "a cat sitting on a windowsill, {trigger} style",
            "a mountain lake with a small wooden boat, {trigger} style",
            "portrait of an old fisherman, {trigger} style",
        ],
    },
    "object": {
        "steps": 1200, "rank": 16, "learning_rate": 1e-4, "flip_aug": False,
        "class_word": "",
        "sample_prompts": [
            "a photo of {trigger} on a wooden table, soft daylight",
            "a photo of {trigger} on a snowy mountain top",
            "{trigger} in a product photo, white studio background",
            "a watercolor painting of {trigger}",
        ],
    },
}


@dataclass(frozen=True)
class Param:
    name: str
    type: type
    default: Any
    env: str | None = None
    min: float | None = None
    max: float | None = None
    choices: tuple | None = None
    help: str = ""


PARAMS: list[Param] = [
    # --- model -------------------------------------------------------------
    Param("base_model", str, "sdxl", "BASE_MODEL", choices=BASE_MODELS,
          help="sdxl = SDXL 1.0 base; flux-dev = FLUX.1 [dev] (gated, needs HF_TOKEN, non-commercial license); "
               "flux-schnell = FLUX.1 [schnell] (Apache-2.0, experimental for LoRA training)."),
    Param("model_file", str, "", "MODEL_FILE",
          help="Optional custom checkpoint: HF 'owner/repo/file.safetensors' or an https URL to a .safetensors "
               "(single-file SDXL checkpoint, or FLUX.1 transformer in the BFL single-file layout)."),
    Param("ae_file", str, "", "AE_FILE", help="FLUX only: optional custom AE file (HF 'owner/repo/file' or URL)."),
    # --- data --------------------------------------------------------------
    Param("dataset_url", str, "", "DATASET_URL",
          help="https URL to a .zip / .tar(.gz) of images with optional same-name .txt captions."),
    Param("hf_dataset", str, "", "HF_DATASET",
          help="Hugging Face dataset repo id: image files (+ .txt captions) or a table with an image column."),
    Param("hf_dataset_split", str, "train", "HF_DATASET_SPLIT"),
    Param("image_column", str, "image", "IMAGE_COLUMN"),
    Param("caption_column", str, "", "CAPTION_COLUMN", help="Table datasets: caption column ('' = auto-detect)."),
    Param("max_images", int, 200, "MAX_IMAGES", min=1, max=5000),
    Param("trigger_word", str, "", "TRIGGER_WORD", help="Unique token for the concept, e.g. 'zxowl'."),
    Param("class_word", str, None, "CLASS_WORD", help="Generic class, e.g. 'man', 'dog', 'figurine'."),
    Param("caption_mode", str, "auto", "CAPTION_MODE", choices=("auto", "existing", "trigger", "florence2"),
          help="auto = use .txt captions when present, else trigger word; trigger = ignore captions; "
               "florence2 = auto-caption images without a caption using Florence-2."),
    Param("prepend_trigger", bool, True, "PREPEND_TRIGGER",
          help="Make sure every caption starts with the trigger word."),
    # --- training ----------------------------------------------------------
    Param("preset", str, "object", "PRESET", choices=tuple(PRESETS)),
    Param("steps", int, None, "STEPS", min=10, max=20000),
    Param("rank", int, None, "LORA_RANK", min=1, max=256),
    Param("alpha", float, 0.0, "LORA_ALPHA", min=0.0, max=1024, help="0 = same as rank."),
    Param("learning_rate", float, None, "LEARNING_RATE", min=1e-7, max=1e-2),
    Param("resolution", int, 1024, "RESOLUTION", min=256, max=2048),
    Param("batch_size", int, 1, "BATCH_SIZE", min=1, max=16),
    Param("optimizer", str, "adamw8bit", "OPTIMIZER", choices=("adamw8bit", "adamw", "adafactor", "prodigy")),
    Param("lr_scheduler", str, "constant_with_warmup", "LR_SCHEDULER",
          choices=("constant", "constant_with_warmup", "cosine", "linear")),
    Param("warmup_steps", int, -1, "WARMUP_STEPS", min=-1, max=5000, help="-1 = 5% of steps."),
    Param("flip_aug", bool, None, "FLIP_AUG"),
    Param("train_text_encoder", bool, False, "TRAIN_TEXT_ENCODER",
          help="SDXL only: also train the text encoders (slower, more VRAM). Default: UNet only."),
    Param("seed", int, 42, "SEED", min=0),
    Param("save_every_n_steps", int, 0, "SAVE_EVERY_N_STEPS", min=0,
          help="Intermediate checkpoints (kept on the network volume). 0 = only the final LoRA."),
    # --- samples -----------------------------------------------------------
    Param("sample_prompts", list, None, "SAMPLE_PROMPTS",
          help="List (or '|'-separated string) of prompts; '{trigger}' is replaced. Default: preset prompts."),
    Param("num_samples", int, 2, "NUM_SAMPLES", min=0, max=4),
    Param("sample_steps", int, 0, "SAMPLE_STEPS", min=0, max=100, help="0 = model default (SDXL 28, dev 20, schnell 4)."),
    Param("sample_before", bool, False, "SAMPLE_BEFORE", help="Also render the prompts before training (baseline)."),
    Param("return_samples_base64", bool, True, "RETURN_SAMPLES_BASE64"),
    # --- output -------------------------------------------------------------
    Param("output_name", str, "", "OUTPUT_NAME", help="LoRA file name (default: '<trigger>_<base_model>_lora')."),
    Param("run_name", str, "", "RUN_NAME", help="Folder name for outputs. Default: the Runpod job id."),
    Param("hf_repo_id", str, "", "HF_OUTPUT_REPO", help="Upload the LoRA + samples here (needs HF_TOKEN with write)."),
    Param("hf_private", bool, True, "HF_PRIVATE"),
    Param("upload_url", str, "", None, help="Presigned HTTP PUT URL for the .safetensors file."),
    Param("upload_to_bucket", bool, True, "UPLOAD_TO_BUCKET",
          help="If BUCKET_* env vars are set, upload the LoRA + samples and return presigned URLs."),
    # --- control -----------------------------------------------------------
    Param("dry_run", bool, False, "DRY_RUN",
          help="Download + validate the dataset, check model access, return the plan. No GPU work."),
]

PARAM_BY_NAME = {p.name: p for p in PARAMS}

ALIASES = {
    "model": "base_model",
    "dataset": "dataset_url",
    "zip_url": "dataset_url",
    "trigger": "trigger_word",
    "lr": "learning_rate",
    "max_train_steps": "steps",
    "network_dim": "rank",
    "lora_rank": "rank",
    "network_alpha": "alpha",
    "lora_alpha": "alpha",
    "output_repo": "hf_repo_id",
    "presigned_url": "upload_url",
}


class ConfigError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


@dataclass
class ResolvedConfig:
    values: dict[str, Any]
    sources: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __getattr__(self, item: str) -> Any:
        try:
            return self.values[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc

    @property
    def is_flux(self) -> bool:
        return self.values["base_model"].startswith("flux")

    def public_dict(self) -> dict[str, Any]:
        out = dict(self.values)
        if out.get("upload_url"):
            out["upload_url"] = "<presigned url>"
        return out


def _coerce(param: Param, raw: Any) -> Any:
    t = param.type
    if t is bool:
        if isinstance(raw, bool):
            return raw
        if isinstance(raw, (int, float)) and raw in (0, 1):
            return bool(raw)
        s = str(raw).strip().lower()
        if s in TRUE_STRINGS:
            return True
        if s in FALSE_STRINGS:
            return False
        raise ValueError(f"{param.name}: expected boolean, got {raw!r}")
    if t is int:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected integer, got boolean")
        if isinstance(raw, float):
            if not raw.is_integer():
                raise ValueError(f"{param.name}: expected integer, got {raw!r}")
            return int(raw)
        s = str(raw).strip()
        try:
            return int(float(s)) if re.fullmatch(r"-?\d+(\.0+)?", s) else int(s)
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected integer, got {raw!r}") from exc
    if t is float:
        if isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected number, got boolean")
        try:
            return float(str(raw).strip())
        except ValueError as exc:
            raise ValueError(f"{param.name}: expected number, got {raw!r}") from exc
    if t is list:
        if isinstance(raw, list):
            return [str(x) for x in raw]
        if isinstance(raw, str):
            return [x.strip() for x in raw.split("|") if x.strip()]
        raise ValueError(f"{param.name}: expected a list of strings")
    if t is str:
        if not isinstance(raw, (str, int, float)) or isinstance(raw, bool):
            raise ValueError(f"{param.name}: expected string")
        return str(raw).strip()
    raise TypeError(t)  # pragma: no cover


def resolve_config(job_input: Mapping[str, Any] | None,
                   env: Mapping[str, str] | None = None) -> ResolvedConfig:
    env = os.environ if env is None else env
    job_input = dict(job_input or {})
    errors: list[str] = []
    warnings: list[str] = []

    for alias, canonical in ALIASES.items():
        if alias in job_input:
            if canonical in job_input:
                errors.append(f"both '{alias}' and '{canonical}' given")
            else:
                job_input[canonical] = job_input.pop(alias)

    for secret_like in ("hf_token", "HF_TOKEN", "token", "aws_secret_access_key", "secret_key"):
        if secret_like in job_input:
            errors.append("do not send tokens/keys in the job input; set HF_TOKEN / BUCKET_* as Runpod secrets")
            job_input.pop(secret_like, None)

    unknown = sorted(k for k in job_input if k not in PARAM_BY_NAME)
    if unknown:
        errors.append(f"unknown input field(s): {', '.join(unknown)}")

    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    for p in PARAMS:
        raw, src = None, "default"
        if p.name in job_input and job_input[p.name] is not None:
            raw, src = job_input[p.name], "input"
        elif p.env and env.get(p.env) not in (None, ""):
            raw, src = env.get(p.env), "env"
        val = p.default
        if src != "default":
            try:
                val = _coerce(p, raw)
            except ValueError as exc:
                errors.append(str(exc))
        values[p.name], sources[p.name] = val, src

    # Preset defaults fill anything still unset.
    preset = PRESETS.get(values["preset"])
    if preset is None:
        errors.append(f"preset: must be one of {list(PRESETS)}, got {values['preset']!r}")
        preset = PRESETS["object"]
    for k, v in preset.items():
        if values.get(k) is None:
            values[k], sources[k] = (list(v) if isinstance(v, list) else v), "preset"

    for p in PARAMS:
        val = values[p.name]
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            if p.min is not None and val < p.min:
                errors.append(f"{p.name}: must be >= {p.min}, got {val}")
            if p.max is not None and val > p.max:
                errors.append(f"{p.name}: must be <= {p.max}, got {val}")
        if p.choices and val not in p.choices:
            errors.append(f"{p.name}: must be one of {list(p.choices)}, got {val!r}")

    # --- cross-field checks ---------------------------------------------
    if values["base_model"] == "flux-schnell" and values.get("learning_rate"):
        warnings.append("flux-schnell LoRA training is experimental; flux-dev gives better results")
    if values["base_model"] == "flux-dev":
        warnings.append("FLUX.1 [dev] is under the FLUX.1 [dev] Non-Commercial License: outputs/LoRAs follow its terms")

    # A per-job dataset choice overrides a default dataset set on the endpoint.
    if sources["hf_dataset"] == "input" and sources["dataset_url"] == "env":
        values["dataset_url"] = ""
    if sources["dataset_url"] == "input" and sources["hf_dataset"] == "env":
        values["hf_dataset"] = ""
    url, hfd = values["dataset_url"], values["hf_dataset"]
    if url and hfd:
        errors.append("give either 'dataset_url' or 'hf_dataset', not both")
    elif not url and not hfd:
        errors.append("dataset: set 'dataset_url' (zip of images + .txt captions) or 'hf_dataset'")
    elif url and not URL_RE.match(url):
        errors.append("dataset_url: expected an http(s) URL")
    elif hfd and not HF_REPO_RE.match(hfd):
        errors.append("hf_dataset: expected a Hugging Face dataset id like 'owner/name'")

    mf = values["model_file"]
    if mf and not (URL_RE.match(mf) or HF_FILE_RE.match(mf)):
        errors.append("model_file: expected 'owner/repo/file.safetensors' or an https URL")

    if values["ae_file"] and not (URL_RE.match(values["ae_file"]) or HF_FILE_RE.match(values["ae_file"])):
        errors.append("ae_file: expected 'owner/repo/file.safetensors' or an https URL")

    tw = values["trigger_word"]
    if tw and not TRIGGER_RE.match(tw):
        errors.append("trigger_word: one short phrase, no commas or newlines (max 64 chars)")
    if not tw:
        if values["caption_mode"] == "trigger":
            errors.append("caption_mode 'trigger' needs trigger_word")
        else:
            warnings.append("no trigger_word: the LoRA will be driven by your captions only")

    if values["alpha"] in (0, 0.0, None):
        values["alpha"] = float(values["rank"])
    if values["warmup_steps"] < 0:
        values["warmup_steps"] = max(0, int(values["steps"] * 0.05))
    if values["warmup_steps"] >= values["steps"]:
        errors.append("warmup_steps must be smaller than steps")

    if values["resolution"] % 64:
        errors.append("resolution must be a multiple of 64")
    if values["base_model"] == "sdxl" and values["resolution"] < 768:
        warnings.append("SDXL is trained at 1024px; resolution < 768 usually hurts quality")

    prompts = [p for p in (values["sample_prompts"] or []) if p.strip()]
    if any(len(p) > 1000 for p in prompts):
        errors.append("sample_prompts: each prompt max 1000 characters")
    values["sample_prompts"] = prompts[: values["num_samples"]] if values["num_samples"] else []

    if values["train_text_encoder"] and values["base_model"] != "sdxl":
        warnings.append("train_text_encoder is only supported for sdxl here; ignored")
        values["train_text_encoder"] = False

    if values["hf_repo_id"] and not HF_REPO_RE.match(values["hf_repo_id"]):
        errors.append("hf_repo_id: expected 'owner/name'")
    if values["upload_url"] and not values["upload_url"].startswith("https://"):
        errors.append("upload_url: expected an https presigned PUT URL")
    if values["run_name"] and not RUN_NAME_RE.match(values["run_name"]):
        errors.append("run_name: letters, digits, '.', '_' or '-' (max 100 chars)")
    if values["output_name"] and not RUN_NAME_RE.match(values["output_name"]):
        errors.append("output_name: letters, digits, '.', '_' or '-' (max 100 chars)")

    has_token = bool((env.get("HF_TOKEN") or env.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
                     and not (env.get("HF_TOKEN") or "").startswith("{{"))
    if values["base_model"] in ("flux-dev", "flux-schnell") and not has_token and not (mf and values["ae_file"]):
        repo = "FLUX.1-dev" if values["base_model"] == "flux-dev" else "FLUX.1-schnell"
        errors.append(f"{values['base_model']} is a gated model: accept the license on "
                      f"huggingface.co/black-forest-labs/{repo} and set HF_TOKEN (read access) as a Runpod secret")
    if values["hf_repo_id"] and not has_token:
        msg = "hf_repo_id is set but HF_TOKEN is not; add HF_TOKEN (write access) as a Runpod secret"
        (warnings if values["dry_run"] else errors).append(msg)

    if errors:
        raise ConfigError(errors)
    return ResolvedConfig(values=values, sources=sources, warnings=warnings)


def render_prompt(template: str, trigger: str, class_word: str) -> str:
    subject = " ".join(x for x in (trigger, class_word) if x).strip()
    out = template.replace("{trigger}", subject or "the subject")
    return re.sub(r"\s+", " ", out).strip()
