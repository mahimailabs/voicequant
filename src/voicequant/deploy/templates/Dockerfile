# VoiceQuant GPU Container — Multi-stage build
#
# Runs VoiceQuant inference server with TurboQuant KV cache compression.
# Uses vLLM as the inference backend with compressed KV cache for ~5x
# more concurrent voice sessions on the same GPU.
#
# Build:
#   docker build -t voicequant:latest .
#
# Run:
#   docker run --gpus all -p 8000:8000 \
#       -e VOICEQUANT_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ \
#       -e VOICEQUANT_TQ_BITS=4 \
#       -v ~/.cache/huggingface:/root/.cache/huggingface \
#       voicequant:latest
#
# Environment variables:
#   VOICEQUANT_MODEL           — HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct-AWQ)
#   VOICEQUANT_TQ_BITS         — TurboQuant bits: 3 or 4 (default: 4)
#   VOICEQUANT_RESIDUAL_WINDOW — FP16 residual window tokens (default: 256)
#   VOICEQUANT_MAX_CONCURRENT  — Max concurrent sequences (default: 64)
#   VOICEQUANT_GPU_MEMORY      — GPU memory utilization 0.0-1.0 (default: 0.90)
#   HF_TOKEN                   — HuggingFace token for gated models

# ============================================================
# Stage 1: Builder — install vLLM, TurboQuant, VoiceQuant
# ============================================================
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        git build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Use python3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir torch>=2.3 --index-url https://download.pytorch.org/whl/cu124

# Install vLLM
RUN pip install --no-cache-dir "vllm>=0.8"

# Install VoiceQuant with CUDA kernels
COPY . /build/voicequant
RUN pip install --no-cache-dir "/build/voicequant[cuda]"

# ============================================================
# Stage 2: Runtime — lean image with only necessary deps
# ============================================================
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv curl && \
    rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Model cache directory
RUN mkdir -p /root/.cache/huggingface
VOLUME ["/root/.cache/huggingface"]

# Default configuration for voice AI workloads
ENV VOICEQUANT_MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
ENV VOICEQUANT_TQ_BITS="4"
ENV VOICEQUANT_RESIDUAL_WINDOW="256"
ENV VOICEQUANT_MAX_CONCURRENT="64"
ENV VOICEQUANT_GPU_MEMORY="0.90"

# Expose the API port
EXPOSE 8000

# Health check: probe the /v1/health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Start the VoiceQuant server using the CLI
ENTRYPOINT ["python", "-m", "voicequant.cli", "serve"]
CMD [ \
    "--model", "${VOICEQUANT_MODEL}", \
    "--tq-bits", "${VOICEQUANT_TQ_BITS}", \
    "--tq-residual-window", "${VOICEQUANT_RESIDUAL_WINDOW}", \
    "--max-concurrent", "${VOICEQUANT_MAX_CONCURRENT}", \
    "--gpu-memory", "${VOICEQUANT_GPU_MEMORY}" \
]
