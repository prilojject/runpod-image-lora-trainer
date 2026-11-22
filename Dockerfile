# Image LoRA trainer (SDXL / FLUX.1) for Runpod Serverless, built on kohya-ss/sd-scripts.
# No model weights are baked in: base models are downloaded at job start and cached on the
# network volume (/runpod-volume) when one is attached.
ARG BASE_IMAGE=pytorch/pytorch:2.14.1-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG SD_SCRIPTS_REF=v0.13.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    SD_SCRIPTS_DIR=/opt/sd-scripts

# git: clone sd-scripts; libgl1/libglib2.0-0: opencv; build-essential: triton/bitsandbytes JIT bits.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates libgl1 libglib2.0-0 build-essential \
    && rm -rf /var/lib/apt/lists/*

# kohya-ss/sd-scripts pinned to a release tag. Keep the base image's torch stack frozen.
RUN git clone --depth 1 --branch ${SD_SCRIPTS_REF} https://github.com/kohya-ss/sd-scripts.git ${SD_SCRIPTS_DIR} \
    && cd ${SD_SCRIPTS_DIR} && git rev-parse HEAD > /opt/sd-scripts.commit \
    && pip freeze | grep -iE '^(torch|torchvision|torchaudio|triton|nvidia-[a-z0-9-]+)==' > /tmp/constraints.txt \
    && cat /tmp/constraints.txt \
    && pip install -r requirements.txt -c /tmp/constraints.txt \
    && rm -rf .git

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt -c /tmp/constraints.txt \
    && pip freeze > /app/requirements.lock \
    && python -c "import torch, transformers, diffusers, accelerate, bitsandbytes, runpod; print('torch', torch.__version__, '| transformers', transformers.__version__, '| diffusers', diffusers.__version__, '| accelerate', accelerate.__version__, '| bnb', bitsandbytes.__version__, '| runpod', runpod.__version__)"

# accelerate default config: single GPU, no prompts at runtime.
RUN mkdir -p /root/.cache/huggingface/accelerate && printf '%s\n' \
    'compute_environment: LOCAL_MACHINE' 'distributed_type: "NO"' 'downcast_bf16: "no"' \
    'machine_rank: 0' 'main_training_function: main' 'mixed_precision: bf16' 'num_machines: 1' \
    'num_processes: 1' 'rdzv_backend: static' 'same_network: true' 'use_cpu: false' \
    > /root/.cache/huggingface/accelerate/default_config.yaml

COPY handler.py /app/handler.py
COPY src /app/src

# No secrets are baked in: HF_TOKEN / BUCKET_* come from Runpod secrets at runtime.
CMD ["python", "-u", "/app/handler.py"]
