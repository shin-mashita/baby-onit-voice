#!/usr/bin/env bash
# Bring up the GPU-host backends -- Qwen3-ASR (vLLM) + speech_server (ASR
# client + Kokoro TTS, both on GPU) -- in the background. Idempotent: safe to
# re-run any time (e.g. after a reboot, or when the network changes); each
# stage is skipped if it's already healthy rather than restarted blind.
#
#   bash scripts/serve_backends.sh
#   tail -f logs/qwen_vllm.log logs/speech_server.log   # watch startup
#
# Does NOT depend on this machine's own IP. The reverse tunnel to the Pi
# (pi-reverse-tunnel.service) makes an OUTBOUND connection to the Pi's
# address, so a WSL2/network IP change on *this* side never breaks it --
# only a change to the Pi's own IP does. See README.md "Bringing it back
# up" for the one-line fix if that ever happens.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs

qwen_up() { curl -s -m 2 http://127.0.0.1:8773/health >/dev/null 2>&1; }
# curl without -f exits 0 on ANY HTTP response (even 4xx/5xx) and nonzero
# only on a connection-level failure -- exactly "is something listening",
# no status-code string parsing needed (and curl can print a literal "000"
# via -w on some connect failures, which broke an earlier version of this
# check that compared against "000" as a string).
speech_up() { curl -s -m 2 -o /dev/null http://127.0.0.1:8772/mcp 2>/dev/null; }

echo "== Qwen3-ASR (vLLM, GPU) =="
if qwen_up; then
  echo "  already up"
else
  nohup ./scripts_qwen_serve.sh > logs/qwen_vllm.log 2>&1 &
  echo $! > logs/qwen_vllm.pid
  echo "  starting (pid $(cat logs/qwen_vllm.pid)) -- this loads a 1.7B model onto the GPU, can take ~1 min"
  ok=0
  for _ in $(seq 1 60); do
    if qwen_up; then ok=1; break; fi
    sleep 5
  done
  if [[ "$ok" != "1" ]]; then
    echo "  FAILED to come up in 5 min -- check logs/qwen_vllm.log" >&2
    exit 1
  fi
  echo "  healthy"
fi

echo "== speech_server (ASR client + Kokoro TTS, GPU) =="
if speech_up; then
  echo "  already up"
else
  # shellcheck disable=SC1091
  source .venv/bin/activate
  nohup python -m speech_server --prewarm --asr-load gpu --tts-load gpu \
    --qwen-url http://127.0.0.1:8773/v1 > logs/speech_server.log 2>&1 &
  echo $! > logs/speech_server.pid
  echo "  starting (pid $(cat logs/speech_server.pid))"
  ok=0
  for _ in $(seq 1 30); do
    if speech_up; then ok=1; break; fi
    sleep 2
  done
  if [[ "$ok" != "1" ]]; then
    echo "  FAILED to come up in 1 min -- check logs/speech_server.log" >&2
    exit 1
  fi
  echo "  healthy"
fi

echo "== reverse tunnel to the Pi =="
if systemctl --user is-active --quiet pi-reverse-tunnel.service 2>/dev/null; then
  echo "  active"
else
  echo "  NOT active. One-time setup: see README.md \"Bringing it back up\"."
  echo "  (If it's installed but stopped: systemctl --user start pi-reverse-tunnel.service)"
fi

echo
echo "== done =="
echo "  logs: logs/qwen_vllm.log, logs/speech_server.log"
echo "  stop: bash scripts/stop_backends.sh"
