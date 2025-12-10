# Dockerfile for infra_training (KubeTrainer) - GPU-optimized, multi-arg
# Usage:
#  docker build -t kubetrainer-trainer:latest .
#  or with args:
#  docker build --build-arg PY_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime --build-arg USERNAME=kubetrainer -t kubetrainer-trainer:latest .

################################################################
# BASE IMAGE
# We default to an official PyTorch image with CUDA runtime to simplify GPU support.
# If you need a different CUDA version or a CPU-only image, override PY_IMAGE at build time.
################################################################
ARG PY_IMAGE=pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime
FROM ${PY_IMAGE} as base

# metadata
LABEL org.opencontainers.image.title="KubeTrainer - Infra Training"
LABEL org.opencontainers.image.description="Container image for infra_training pipeline (training, RAG, inference)"
LABEL maintainer="Brendan (bdbrink) <you@your.email>"

################################################################
# Build-time args (can override to match your environment)
################################################################
ARG USERNAME=kubetrainer
ARG USER_UID=1000
ARG USER_GID=1000
ARG PYTHONUNBUFFERED=1
ARG POETRY=false

ENV PYTHONUNBUFFERED=${PYTHONUNBUFFERED}
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8

################################################################
# System packages (add anything your scripts need)
################################################################
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    build-essential \
    python3-dev \
    python3-venv \
    libsndfile1 \
    ffmpeg \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

################################################################
# Create a non-root user for security
################################################################
RUN groupadd --gid ${USER_GID} ${USERNAME} \
 && useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} --shell /bin/bash

# Workdir and copy source
WORKDIR /app

# Copy only dependency files first for caching
# If you have a requirements.txt at repo root (recommended), COPY that.
# Fallback: we'll provide a requirements.txt template below if you don't have one.
COPY infra_training/requirements.txt infra_training/requirements.txt

################################################################
# Install python deps
# We prefer pip install. If you use Poetry or Pipenv, adapt this section.
################################################################
# Upgrade pip and install build-time helpers
RUN python -m pip install --upgrade pip setuptools wheel

# If you want to accelerate installs in CI, consider pip cache mounting.
RUN if [ -f infra_training/requirements.txt ]; then \
      python -m pip install -r infra_training/requirements.txt; \
    fi

################################################################
# Copy app code
################################################################
COPY infra_training/ /app/infra_training

# Expose model assets+data directories as volumes (created on container start)
ENV MODELS_DIR=/data/models
ENV DATA_DIR=/data/datasets
ENV INDEX_DIR=/data/index
ENV LOG_DIR=/data/logs
RUN mkdir -p ${MODELS_DIR} ${DATA_DIR} ${INDEX_DIR} ${LOG_DIR} && chown -R ${USERNAME}:${USERNAME} /data

# Set user
USER ${USERNAME}
WORKDIR /app

################################################################
# Entrypoint script (we'll add it below as an actual file)
################################################################
COPY /entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose port for serving/inference (if infra_training serves models)
EXPOSE 8000

# Healthcheck (simple ping to integrated script or server)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD [ -f /tmp/kubetrainer_ready ] || exit 1

# Final entrypoint (accepts options to run train/serve/eval)
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["--help"]
