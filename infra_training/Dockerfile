# ================================================
# Base image with PyTorch + CUDA
# ================================================
FROM pytorch/pytorch:2.2.0-cuda11.8-cudnn8-runtime AS base

# Do everything in /app
WORKDIR /app

# ================================================
# System dependencies
# ================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    python3-dev \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# ================================================
# Copy only requirements first for caching
# ================================================
COPY infra_training/requirements.txt /app/requirements.txt

# Install Python deps
RUN pip install --no-cache-dir -r /app/requirements.txt


# ================================================
# Copy application code
# ================================================
COPY infra_training /app/infra_training

# Make an entrypoint (with auto-chmod)
COPY --chmod=755 infra_training/entrypoint.sh /app/entrypoint.sh

# ================================================
# Default runtime configuration
# ================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Expose whatever port your trainer uses (optional)
# EXPOSE 8000

# ================================================
# Entrypoint + default command
# ================================================
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-m", "infra_training"]
