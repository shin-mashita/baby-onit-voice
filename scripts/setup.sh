#!/usr/bin/env bash
# Base install for the voice agent: harness + ASR/TTS client + dashboard, all
# on CPU. Run this first on a new machine; everything else is optional.
#
#   bash scripts/setup.sh              # text harness + voice stack, CPU only
#   bash scripts/setup.sh --gpu-tts    # + Kokoro TTS on CUDA (needs a GPU)
#
# What this does NOT do:
#   - Qwen3-ASR: separate venv, separate script. See scripts/setup_qwen.sh.
#   - Download speech models: they fetch on first run (python -m speech_server
#     --prewarm), not here, so this script works offline-capable until then.
#
# Read docs/ARCHITECTURE.md §1.1 before touching the GPU TTS pin in
# pyproject.toml -- it records two silent-fallback traps this install avoids.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

command -v uv >/dev/null || { echo "uv not found: https://docs.astral.sh/uv/" >&2; exit 1; }

echo "== base venv =="
uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install -e '.[voice]'

if [[ "${1:-}" == "--gpu-tts" ]]; then
  echo "== GPU TTS (Kokoro on CUDA) =="
  if ! python3 -c "import ctypes,glob,os,site; cands=[p for sp in site.getsitepackages() for p in glob.glob(os.path.join(sp,'nvidia','cuda_runtime','lib','libcudart.so*'))]; import sys; sys.exit(0 if cands or __import__('shutil').which('nvidia-smi') else 1)" 2>/dev/null; then
    echo "  no nvidia-smi found; skipping (CPU TTS will still work)" >&2
  fi
  # kokoro-onnx --no-deps: its metadata requires a plain `onnxruntime`
  # unconditionally, which would fight the pinned onnxruntime-gpu below for
  # the same import path. See the comment on voice-gpu-tts in pyproject.toml.
  uv pip install --no-deps kokoro-onnx
  uv pip install -e '.[voice-gpu-tts]'
  echo "  verifying CUDA actually initializes (a stale/mismatched driver"
  echo "  silently falls back to CPU otherwise -- see ARCHITECTURE.md §1.1):"
  python3 -c "
import onnxruntime as ort
print('  onnxruntime', ort.__version__, '| providers:', ort.get_available_providers())
print('  CUDAExecutionProvider available:', 'CUDAExecutionProvider' in ort.get_available_providers())
"
fi

echo
echo "== done =="
echo "  source .venv/bin/activate"
echo "  python -m speech_server --prewarm   # first run downloads ASR+TTS models"
echo "  python -m voice_ui                  # agent + dashboard, http://127.0.0.1:7860"
