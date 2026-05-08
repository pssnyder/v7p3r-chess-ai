# Multi-stage Dockerfile for V7P3R AI Training
# Supports GPU (CUDA) and CPU training with auto-recovery

# Stage 1: Base image with PyTorch and CUDA
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    stockfish \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Stage 2: Dependencies
FROM base AS dependencies

# Copy requirements first for layer caching
COPY requirements.txt /workspace/

# Install dependencies with better error handling
# Note: PyTorch is already in base image, skip torch installation
RUN pip install --no-cache-dir \
    numpy>=1.24.0 \
    python-chess>=1.999 \
    tensorboard>=2.13.0 \
    pandas>=2.0.0 \
    tqdm>=4.65.0 \
    h5py>=3.9.0 \
    pyyaml>=6.0 \
    jsonschema>=4.17.0 \
    colorama>=0.4.6 \
    matplotlib>=3.7.0 \
    seaborn>=0.12.0 \
    && pip install --no-cache-dir \
    wandb \
    psutil \
    py-cpuinfo \
    || echo "Some optional dependencies failed, continuing..."

# Install chess-specific packages (may fail, non-critical)
RUN pip install --no-cache-dir stockfish || echo "Stockfish wrapper not needed (using system binary)"

# Stage 3: Training environment
FROM dependencies AS training

# Copy all source code
COPY . /workspace/

# Create necessary directories
RUN mkdir -p \
    /workspace/data \
    /workspace/models \
    /workspace/logs \
    /workspace/checkpoints \
    /workspace/tensorboard_logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STOCKFISH_PATH=/usr/games/stockfish
ENV CUDA_VISIBLE_DEVICES=0
ENV OMP_NUM_THREADS=8
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Healthcheck - verify training is progressing
HEALTHCHECK --interval=5m --timeout=30s --start-period=5m --retries=3 \
    CMD python scripts/training_health_check.py || exit 1

# Set up volume mount points
VOLUME ["/workspace/data", "/workspace/models", "/workspace/logs"]

# Default command: Run 48-hour simple training
CMD ["python", "scripts/simple_48h_training.py", "--duration-hours", "48", "--auto-recover"]
