#!/usr/bin/env bash
# Serve Qwen3-ASR-1.7B for --asr-load gpu.
#
# Runs from .venv-qwen, which is deliberately separate: vLLM needs torch
# 2.13+cu129 and a downgraded cu12 torchcodec, and putting that in the main venv
# would fight the onnxruntime-gpu stack the TTS backend depends on.
#
# The memory flags are not defaults -- they are the numbers this GPU needs.
# gpu_memory_utilization alone reserves nothing for KV cache here (measured:
# -0.26 GiB at 0.88), so kv-cache-memory is set explicitly.
set -euo pipefail
# On this machine (WSL2, RTX 2000 Ada 8GiB), nvidia-smi reports 8GiB total
# but CUDA only sees ~6.93GiB free on startup -- WSL2's GPU memory accounting
# quirk, not another process. 0.97 (7.76 GiB) doesn't fit; 0.80 (6.4 GiB) does
# and still leaves room for Kokoro TTS on the GPU alongside it.
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
exec .venv-qwen/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-ASR-1.7B \
  --port "${QWEN_PORT:-8773}" \
  --gpu-memory-utilization 0.80 \
  --kv-cache-memory 470396416 \
  --max-model-len 448 \
  --enforce-eager \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"audio":1}' \
  --served-model-name qwen3-asr
