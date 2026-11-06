#!/usr/bin/env bash
# Optional: Qwen3-ASR-1.7B, 8x faster than Moonshine on this hardware but with
# no partial transcripts and a ~7GB VRAM appetite. Needs its OWN venv -- vLLM
# requires torch 2.13+cu129 and a downgraded cu12 torchcodec, which would fight
# the onnxruntime-gpu stack the TTS backend depends on if installed together.
# See docs/ARCHITECTURE.md §... (search "Qwen3-ASR") for the full tradeoff.
#
#   bash scripts/setup_qwen.sh
#   ./scripts_qwen_serve.sh                              # terminal 1
#   python -m speech_server --prewarm --asr-load gpu --tts-load gpu   # terminal 2
#
# ----------------------------------------------------------------------------
# THE VLLM WHEEL URL BELOW WILL ROT. It names a specific commit on vLLM's
# nightly cu129 channel (https://wheels.vllm.ai/nightly/cu129/vllm/), which is
# a frozen snapshot, not a maintained release line -- the default `pip install
# vllm` resolves to a CUDA-13 build that will not initialize on a driver below
# R580 (see ARCHITECTURE.md; this cost real debugging time to pin down).
#
# If this 404s on a new machine:
#   1. curl -s https://wheels.vllm.ai/nightly/cu129/vllm/ | grep -o 'vllm-[^"]*\.whl'
#      and pick the newest x86_64 entry.
#   2. That wheel's own metadata names its required torch build -- download it
#      with `pip download --no-deps <url> -d /tmp` and
#      `unzip -p /tmp/*.whl '*/METADATA' | grep -i 'Requires-Dist: torch'`.
#      Update TORCH_PIN below to match; it will not necessarily still be
#      2.13.0+cu129.
#   3. Re-run. If vLLM then fails to import, re-check the torchcodec step: the
#      newest cu126-tagged torchcodec build may have moved past 0.9.1.
# ----------------------------------------------------------------------------
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

VLLM_WHEEL="https://wheels.vllm.ai/77154ca6116ae34cbfe314beef1016bde9b63989/vllm-0.32.0rc1.dev100%2Bg77154ca61.cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
TORCH_PIN="torch==2.13.0+cu129"
TORCHCODEC_PIN="torchcodec==0.9.1+cu126"  # newest CUDA-12 build; vLLM's own floor (>=0.14) has none

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }

echo "== .venv-qwen (separate from the main venv; needs ~11 GB) =="
uv venv --python 3.12 .venv-qwen

echo "== vLLM + a CUDA-12 torch, pinned (see header before changing) =="
UV_HTTP_TIMEOUT=600 uv pip install --python .venv-qwen \
  "$VLLM_WHEEL" "$TORCH_PIN" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match

echo "== downgrade torchcodec: PyPI's default is a CUDA-13 build =="
echo "   (symptom if skipped: 'from vllm import LLM' raises ImportError on"
echo "   libtorchcodec_image.so -- missing libnvrtc.so.13/libcudart.so.13)"
uv pip install --python .venv-qwen --no-deps \
  "$TORCHCODEC_PIN" \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  --index-strategy unsafe-best-match

echo "== audio decode for the /v1/audio/transcriptions endpoint =="
echo "   (vllm[audio]'s own extra pulls the CUDA-13 torchcodec right back;"
echo "   soundfile/librosa is what the server actually falls back to)"
uv pip install --python .venv-qwen soundfile librosa

echo
echo "== verifying the pin held (should print 2.13.0+cu129 / True) =="
.venv-qwen/bin/python -c "
import torch
print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available())
import vllm
print('vllm', vllm.__version__)
"

echo
echo "== done. Model weights (4.4 GB) download on first serve. =="
echo "   ./scripts_qwen_serve.sh"
