#!/usr/bin/env bash
# Stop the GPU-host backends started by serve_backends.sh.
#
#   bash scripts/stop_backends.sh
#
# Leaves the reverse tunnel to the Pi (pi-reverse-tunnel.service) running --
# it's a separate systemd unit, costs nothing idle, and the Pi's voice_ui
# will just fail its own health checks until the backends are back, same as
# any other outage.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

for f in logs/qwen_vllm.pid logs/speech_server.pid; do
  [[ -f "$f" ]] || continue
  pid="$(cat "$f")"
  if kill "$pid" 2>/dev/null; then
    echo "stopped $(basename "$f" .pid) (pid $pid)"
  fi
  rm -f "$f"
done

# Belt and suspenders: catches the case a .pid file was stale (process
# restarted under a new pid some other way) or missing entirely.
pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
pkill -f "python -m speech_server" 2>/dev/null || true

echo "done"
