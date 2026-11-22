# AGENTS.md

Instructions for coding agents working in `prilojject/runpod-image-lora-trainer`.

## What this repo is

A Runpod Serverless worker (Hub listing, `type: serverless`) that trains SDXL / FLUX.1 LoRAs with
kohya-ss/sd-scripts (pinned tag in the `Dockerfile`) and returns a `.safetensors` LoRA plus sample images.
It is not a UI and not a Pod template.

## Layout

- `handler.py` — Runpod entry point; orchestrates dataset → models → training → delivery
- `src/config.py` — job input > env (Hub form) > preset > defaults; validation; presets
- `src/dataset.py` — download/unpack (zip-slip safe) and validate images + captions; HF datasets
- `src/captions.py` — optional Florence-2 auto-captions
- `src/models.py` — base model files (download at start, cache on `/runpod-volume`), access check
- `src/train.py` — kohya dataset TOML, sample prompts, command line, log streaming
- `src/outputs.py` — HF upload + model card, S3 (BUCKET_*), presigned PUT, base64 samples
- `src/secret_utils.py` — tokens from env only; log redaction
- `.runpod/hub.json`, `.runpod/tests.json`, `.runpod/icon.png` — Hub metadata, test, icon
- `examples/` — request bodies and a 6-image example dataset
- `tests/` — unit tests (no GPU, no weight downloads)

## Local checks (no GPU, no paid cloud)

```bash
docker build -t runpod-image-lora .
docker run --rm -v $PWD:/repo -w /repo runpod-image-lora sh -c "pip install pytest && pytest -q"
docker run --rm runpod-image-lora python -u handler.py --test_input "$(cat examples/dry_run.json)"
```

Do not add tests that download model weights or call paid APIs.

## Secrets

- `HF_TOKEN` and `BUCKET_*` come from env / Runpod secrets only. The worker rejects tokens in job input — keep it that way.
- Never print tokens, presigned URLs or `.env` files in logs, README examples or issues.

## Conventions

- Never bake model weights into the image. Downloads go to the network volume cache when present.
- Validate the dataset before any GPU work.
- Keep `tests.json` short enough for the Hub test timeout (≤ 60 steps on SDXL).
- After changing `hub.json` / `tests.json` run `pytest` (`tests/test_hub_files.py`).
- FLUX.1 [dev] is non-commercial: keep the license notes in README and the generated model card.
- Public docs are English.

## Paid GPU jobs

- Only start Runpod GPU work when a human asked for a GPU test. One worker max, short config, timeouts set.
- Delete endpoints/pods/registry auths right after the test and verify nothing is left running.
- Training uses async `POST /run` with `policy.executionTimeout`; the 600 s default kills real jobs.

## Releases

A GitHub release is required for the Hub to (re)index the repo. Bump `src/__init__.py` `__version__` and tag `vX.Y.Z`.
