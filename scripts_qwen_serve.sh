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
exec .venv-qwen/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-ASR-1.7B \
  --port "${QWEN_PORT:-8773}" \
  --gpu-memory-utilization 0.97 \
  --kv-cache-memory 470396416 \
  --max-model-len 448 \
  --enforce-eager \
  --limit-mm-per-prompt '{"audio":1}' \
  --served-model-name qwen3-asr
