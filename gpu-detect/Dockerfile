# ================================================
# gpu-detect Dockerfile
# Purpose: Detect GPUs in a cluster node and exit
# ================================================

############################
# Builder stage
############################
FROM rust:1.72-bookworm AS builder

WORKDIR /build

# Install minimal build deps (ssl needed for many crates)
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Copy manifests first for dependency caching
COPY Cargo.toml Cargo.lock ./

# Dummy src to cache deps
RUN mkdir src && echo "fn main() {}" > src/main.rs
RUN cargo build --release || true

# Copy real source
COPY src ./src

# Build actual binary
RUN cargo build --release


############################
# Runtime stage
############################
FROM debian:bookworm-slim AS runtime

# Runtime deps only (no compiler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Create non-root user (best practice)
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} detector \
 && useradd -m -u ${UID} -g ${GID} detector

WORKDIR /app

# Copy binary
COPY --from=builder /build/target/release/gpu-detect /app/gpu-detect

RUN chown detector:detector /app/gpu-detect

USER detector

# gpu-detect prints results and exits
ENTRYPOINT ["/app/gpu-detect"]
