#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./docker_run.sh                  # interactive shell
#   ./docker_run.sh bash data01.sh   # one-off command (e.g. python data01.py)
#   ./docker_run.sh bash train.sh    # training

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TTY_FLAGS="-it"
if [ ! -t 0 ]; then TTY_FLAGS=""; fi

docker run --rm $TTY_FLAGS \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    --shm-size=16g \
    --cpus="${CPUS:-32}" \
    --memory="${MEM:-96g}" \
    --memory-swap="${MEM:-96g}" \
    -v "${REPO}":/workspace \
    -w /workspace \
    -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
    matmoe:latest \
    "$@"
