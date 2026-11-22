# Image LoRA Trainer (SDXL & FLUX.1) — Runpod Serverless worker

Train your own **SDXL** or **FLUX.1** LoRA — a face, an art style, a product or any object — on a
**Runpod Serverless** GPU with [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts), the trainer
behind most community LoRAs. You send a URL to a zip of your images (with optional `.txt` captions) and a
trigger word; you get back a **`.safetensors` LoRA** (on your Hugging Face repo, S3 bucket, presigned URL or
network volume) plus **2–4 sample images** rendered with the new LoRA. No Pod, no UI, pay per second of
training.

Useful search terms: *train SDXL LoRA on Runpod serverless*, *FLUX LoRA training API*, *kohya serverless worker*,
*Dreambooth LoRA Runpod template*, *train a face LoRA via API*, *FLUX.1 dev LoRA trainer Runpod Hub*.

<!-- The badge and Hub URL go live after Runpod reviews the listing. -->
[![Runpod](https://api.runpod.io/badge/prilojject/runpod-image-lora-trainer)](https://console.runpod.io/hub/prilojject/runpod-image-lora-trainer)

**Deploy on Runpod Hub:** [console.runpod.io/hub/prilojject/runpod-image-lora-trainer](https://console.runpod.io/hub/prilojject/runpod-image-lora-trainer)

> **Listing status:** TODO — paste the live Hub URL here once moderation approves the repo.

Source: [github.com/prilojject/runpod-image-lora-trainer](https://github.com/prilojject/runpod-image-lora-trainer) (MIT).
For agents: [llms.txt](./llms.txt) · [AGENTS.md](./AGENTS.md) · [openapi.yaml](./openapi.yaml) · [CITATION.cff](./CITATION.cff)

---

## What you send / what you get

**You send** (job `input`; defaults can come from the Hub deploy form):

| Field | Example | Notes |
|---|---|---|
| `base_model` | `sdxl` / `flux-dev` / `flux-schnell` | SDXL 1.0 base, FLUX.1 [dev] (gated, non-commercial), FLUX.1 [schnell] (experimental). |
| `dataset_url` | `https://.../photos.zip` | `.zip` or `.tar(.gz)` of `.jpg/.png/.webp` images, optional same-name `.txt` captions. |
| `hf_dataset` | `diffusers/dog-example` | Alternative to `dataset_url`: a HF dataset with image files (+ `.txt`) or an image column. |
| `trigger_word` | `zxowl` | Rare token that activates the LoRA. Prepended to every caption. |
| `class_word` | `figurine`, `man`, `dog` | Optional generic class used in trigger-only captions and sample prompts. |
| `preset` | `face` / `style` / `object` | Sensible steps / rank / lr / sample prompts per subject type. |
| `steps`, `rank`, `learning_rate`, `resolution` | `1500`, `16`, `1e-4`, `1024` | Override the preset. |
| `caption_mode` | `auto` / `florence2` / `trigger` / `existing` | `florence2` auto-captions images that have no `.txt`. |
| `hf_repo_id` | `your-name/my-lora` | Upload LoRA + samples + model card (needs `HF_TOKEN` with write access). |
| `upload_url` | presigned `PUT` URL | The `.safetensors` is `PUT` there (S3/R2/GCS presigned URL). |

**You get** (job `output`, trimmed):

```json
{
  "status": "completed",
  "gpu": {"gpu": "NVIDIA GeForce RTX 4090", "vram_gb": 23.5},
  "training": {"global_step": 1200, "final_avr_loss": 0.091, "train_loop_seconds": 840.2},
  "artifacts": {
    "lora_file": "zxowl_sdxl_lora.safetensors",
    "lora_size_mb": 43.3,
    "hf_repo_url": "https://huggingface.co/your-name/zxowl-sdxl-lora",
    "bucket_lora_url": "https://... (only if BUCKET_* is configured)"
  },
  "samples": [{"prompt": "a photo of zxowl figurine on a wooden table, soft daylight",
               "image_base64": "/9j/4AAQ...", "mime": "image/jpeg"}],
  "dataset": {"images_used": 6, "captions_from_files": 6, "warnings": []}
}
```

The LoRA works in ComfyUI, AUTOMATIC1111 / Forge, Diffusers (`pipe.load_lora_weights(...)`) and anything else
that reads kohya-format LoRAs.

> **Deliver the LoRA somewhere.** Serverless workers are ephemeral. Set at least one of: a network volume
> (files land in `/runpod-volume/image-lora/runs/<run_name>/`), `hf_repo_id` + `HF_TOKEN`, `upload_url`, or the
> `BUCKET_*` env vars. Without them the response carries only the samples and a warning.

## Dataset format

```
photos.zip
├── img_001.jpg
├── img_001.txt      # "a photo of zxowl figurine on a wooden table"  (optional)
├── img_002.png
└── ...
```

* 5–30 varied images are typical (faces: different angles/lighting; styles: varied subjects; objects: varied backgrounds).
* Any aspect ratio — images are bucketed; images smaller than 256 px are skipped, small ones are upscaled (warned).
* Captions: same file name with `.txt`. Missing captions fall back to `trigger_word [class_word]`
  (or Florence-2 auto-captions with `caption_mode: "florence2"`). The trigger word is prepended if a caption lacks it.
* Everything is validated **before** any model download or GPU work; problems come back in `dataset.warnings` / `error`.
* Example dataset: [`examples/datasets/owl_zxowl.zip`](examples/datasets/owl_zxowl.zip) (6 AI-generated images of a ceramic owl, with captions).

## Quick start

Training takes minutes, so always use the **async** `/run` route:

* Runpod's default **execution timeout is 600 s** — raise it on the endpoint (Edit endpoint → Advanced → Execution timeout)
  or per request with `policy.executionTimeout` (ms). A 1500-step SDXL LoRA needs ~20–40 min; FLUX ~1–2 h.
* Job `policy.ttl` (default 24 h) covers queue + run time.
* Results of async jobs are kept for 30 min — the LoRA itself is on your HF repo / bucket / volume.

**1. Validate (no GPU work, no weights):**

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{"input": {"dry_run": true, "base_model": "sdxl", "trigger_word": "zxowl", "class_word": "figurine",
       "dataset_url": "https://raw.githubusercontent.com/prilojject/runpod-image-lora-trainer/main/examples/datasets/owl_zxowl.zip"}}'
```

Returns the dataset report (images, captions, sizes), the rendered sample prompts and a metadata-only check that
the base model files are reachable with your token (FLUX gating shows up here).

**2. Train:**

```bash
curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  -d '{
    "input": {
      "base_model": "sdxl",
      "dataset_url": "https://example.com/my-face-photos.zip",
      "trigger_word": "ohwx", "class_word": "man", "preset": "face",
      "steps": 1500, "num_samples": 4,
      "hf_repo_id": "your-name/ohwx-sdxl-lora"
    },
    "policy": {"executionTimeout": 10800000, "ttl": 86400000}
  }'
# poll: GET https://api.runpod.ai/v2/$ENDPOINT_ID/status/<job id>  (progress: stage, step, loss)
```

More bodies in [`examples/`](examples): smoke test, SDXL object/face, FLUX.1 [dev] style, HF dataset, dry run.

**Decode the samples:**

```python
import base64, json, requests
out = requests.get(f"https://api.runpod.ai/v2/{EP}/status/{JOB}", headers={"Authorization": f"Bearer {KEY}"}).json()["output"]
for i, s in enumerate(out["samples"]):
    open(f"sample_{i}.jpg", "wb").write(base64.b64decode(s["image_base64"]))
```

## Parameters

Job input > Hub form (env var) > preset > default.

| Input | Env | Default | Meaning |
|---|---|---|---|
| `base_model` | `BASE_MODEL` | `sdxl` | `sdxl`, `flux-dev`, `flux-schnell` |
| `model_file` | `MODEL_FILE` | — | Custom checkpoint: HF `owner/repo/file.safetensors` or https URL (single-file SDXL, or BFL-layout FLUX transformer) |
| `ae_file` | `AE_FILE` | — | FLUX: custom AE file |
| `dataset_url` / `hf_dataset` | `DATASET_URL` / `HF_DATASET` | — | One of them is required |
| `image_column`, `caption_column`, `hf_dataset_split` | same, upper-case | `image`, auto, `train` | Table-style HF datasets |
| `max_images` | `MAX_IMAGES` | `200` | |
| `trigger_word`, `class_word` | `TRIGGER_WORD`, `CLASS_WORD` | —, preset | |
| `caption_mode` | `CAPTION_MODE` | `auto` | `auto`, `existing`, `trigger`, `florence2` |
| `prepend_trigger` | `PREPEND_TRIGGER` | `true` | |
| `preset` | `PRESET` | `object` | `face` (1500 steps, r16), `style` (2000, r32, flip aug), `object` (1200, r16) |
| `steps` | `STEPS` | preset | Optimizer steps (10–20000) |
| `rank`, `alpha` | `LORA_RANK`, `LORA_ALPHA` | preset, = rank | |
| `learning_rate` | `LEARNING_RATE` | `1e-4` | |
| `resolution` | `RESOLUTION` | `1024` | multiple of 64; aspect-ratio buckets |
| `batch_size` | `BATCH_SIZE` | `1` | |
| `optimizer` | `OPTIMIZER` | `adamw8bit` | `adamw8bit`, `adamw`, `adafactor`, `prodigy` (use lr 1.0) |
| `lr_scheduler`, `warmup_steps` | `LR_SCHEDULER`, `WARMUP_STEPS` | `constant_with_warmup`, 5 % | |
| `flip_aug` | `FLIP_AUG` | preset | Horizontal flips (off for faces/objects) |
| `train_text_encoder` | `TRAIN_TEXT_ENCODER` | `false` | SDXL only |
| `seed` | `SEED` | `42` | |
| `save_every_n_steps` | `SAVE_EVERY_N_STEPS` | `0` | Intermediate LoRAs (network volume) |
| `sample_prompts` | `SAMPLE_PROMPTS` | preset | List or `|`-separated; `{trigger}` → trigger + class word |
| `num_samples` | `NUM_SAMPLES` | `2` | 0–4 |
| `sample_steps` | `SAMPLE_STEPS` | SDXL 28, dev 20, schnell 4 | |
| `sample_before` | `SAMPLE_BEFORE` | `false` | Also render the prompts before training (baseline) |
| `return_samples_base64` | `RETURN_SAMPLES_BASE64` | `true` | JPEG, max 1024 px |
| `output_name`, `run_name` | `OUTPUT_NAME`, `RUN_NAME` | `<trigger>_<model>_lora`, job id | |
| `hf_repo_id`, `hf_private` | `HF_OUTPUT_REPO`, `HF_PRIVATE` | —, `true` | |
| `upload_url` | — | — | Presigned `PUT` URL for the LoRA (per job only) |
| `upload_to_bucket` | `UPLOAD_TO_BUCKET` | `true` | Uses `BUCKET_ENDPOINT_URL`, `BUCKET_ACCESS_KEY_ID`, `BUCKET_SECRET_ACCESS_KEY` (+ `BUCKET_NAME`) |
| `dry_run` | `DRY_RUN` | `false` | Validate only |
| — | `TRAIN_TIMEOUT_SECONDS` | `0` | Watchdog: stop sd-scripts after N s |
| — | `HF_TOKEN` | — | Runpod secret. Read: FLUX (gated). Write: `hf_repo_id` |

Tokens and keys are **never** accepted in the job input (the job is rejected) — use Runpod secrets.

## Models, GPUs and cost

| Base model | Download | GPU | Notes |
|---|---|---|---|
| `sdxl` | 6.9 GB (`stabilityai/stable-diffusion-xl-base-1.0`) | 24 GB (RTX 4090, L4, A5000, 3090) | UNet LoRA, cached TE outputs, bf16, gradient checkpointing |
| `flux-dev` | 34 GB (FLUX.1-dev + AE, CLIP-L, T5-XXL fp16) | 48 GB recommended (L40S, A6000, A40); 24 GB works with block swap | fp8 base, `flux_shift` timesteps, guidance 1.0 |
| `flux-schnell` | 34 GB | same | Experimental — LoRAs on a distilled model are weaker |

* **No weights are baked into the image** (image ≈ 8.5 GB). Models download at job start with `hf_transfer`.
  Attach a **network volume** to cache them in `/runpod-volume/image-lora/models` — later cold starts skip the download.
* Measured smoke test (RTX 4090, Runpod Serverless, SDXL, 6 images, 30 steps at 1024 px, 1 worker, no volume): see
  [GPU test results](#measured-gpu-test).
* Hub default endpoint execution timeout is **600 s** — raise it, or the job is killed mid-training.

## FLUX.1 license and access

* **FLUX.1 [dev]** is released under the
  [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).
  LoRAs trained on it are derivatives and follow those terms. The worker adds the license to the generated model card.
* Both FLUX.1 repos are **gated** on Hugging Face: accept the terms on the model page with your account, create a
  token with read access, and add it as a Runpod secret (`HF_TOKEN = {{ RUNPOD_SECRET_hf_token }}`).
  `dry_run` tells you if access works before you pay for a GPU.
* SDXL 1.0 is under CreativeML Open RAIL++-M. This worker's code is MIT.

## Measured GPU test

TODO_GPU_RESULTS

## How it works

1. Validate input → download/unpack the dataset (zip-slip safe, 4 GiB limit) → check images and captions.
2. Optional Florence-2 captions for images without `.txt`.
3. Download base model files (cached on the volume) → write a kohya dataset TOML and sample-prompt file.
4. Run `sdxl_train_network.py` / `flux_train_network.py` (kohya-ss/sd-scripts **v0.13.0**, pinned) with
   `accelerate`, streaming step/loss as job progress.
5. Render samples with the trained LoRA at the final step → upload LoRA, samples and a model card → return.

The worker sets `refresh_worker: true`, so every job gets a clean worker (GPU memory fully freed).

## FAQ

**Why kohya-ss/sd-scripts and not ai-toolkit?** sd-scripts v0.13 trains SDXL and FLUX.1 from one maintained,
version-tagged code base with a headless CLI, fp8 FLUX training, block swap for 24 GB cards and built-in sampling.
It is also the format every UI loads.

**Can I resume?** Use `save_every_n_steps` with a network volume to keep intermediate LoRAs; a failed job returns the
list of checkpoints. Full optimizer-state resume is not exposed.

**Does it train the text encoder?** SDXL: optional (`train_text_encoder: true`). FLUX: UNet/transformer only (standard).

**Multi-GPU / full fine-tune / SD 1.5 / SD3?** Out of scope for this worker.

## Local development

```bash
docker build -t runpod-image-lora .
docker run --rm runpod-image-lora python -u handler.py --test_input "$(cat examples/dry_run.json)"
pip install -r requirements-dev.txt && pytest -q   # in the image (needs Pillow/requests)
```

## Credits

[kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts) (Apache-2.0) ·
[Florence-2](https://huggingface.co/florence-community/Florence-2-base) (MIT) ·
[Runpod Serverless](https://docs.runpod.io/serverless/overview).
