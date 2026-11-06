#!/usr/bin/env bash
# Pi-side setup for the voice agent: baby_onit harness + voice_client (mic/
# speaker, VAD, turn detection) + voice_ui (dashboard), all running on the
# Pi. ASR and TTS run remotely on a GPU machine's `speech_server` -- this
# script installs NEITHER speech_server's code's heavy deps (moonshine-voice,
# onnxruntime-gpu) NOR a local ASR/TTS engine; only the client-side pieces.
#
# Usage: bash setup.sh
#
# Idempotent: safe to re-run.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== system audio deps (PortAudio/ALSA) =="
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends libportaudio2 portaudio19-dev libasound2-dev

echo "== uv =="
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

echo "== venv =="
uv venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

# keyring's default backend talks to the desktop's DBus Secret Service, which
# waits forever for a GUI unlock prompt that can never appear over SSH --
# `baby-onit setup` hangs silently with no output. Force the file-based
# fallback (~/.baby-onit/secrets.yaml, 0600) by picking it every time this
# venv is activated, not just for this script's own run.
if ! grep -q PYTHON_KEYRING_BACKEND .venv/bin/activate; then
  echo 'export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring' >> .venv/bin/activate
fi
export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring

echo "== baby-onit core (harness: openai client, config, tools) =="
uv pip install -e .

echo "== voice client + dashboard deps (NOT the 'voice' extra -- that pulls"
echo "   moonshine-voice, which is speech_server's ASR/TTS engine and dead"
echo "   weight here since ASR/TTS run remotely) =="
uv pip install mcp websockets starlette uvicorn sounddevice numpy fastapi onnxruntime

echo "== ALSA plug device for the USB mic/speaker =="
echo "   (raw hw:N,0 only accepts the card's native rate -- usually 48kHz --"
echo "   and InputStream fails with 'Invalid sample rate' when the agent"
echo "   asks for 16kHz directly; 'plug' resamples transparently)"
USB_CARD="$(arecord -l 2>/dev/null | grep -oP '(?<=^card )\d+(?=.*USB)' | head -1 || true)"
if [[ -n "$USB_CARD" ]]; then
  if [[ ! -f "$HOME/.asoundrc" ]]; then
    cat > "$HOME/.asoundrc" <<EOF
pcm.voiceagent {
    type plug
    slave.pcm "hw:${USB_CARD},0"
}
ctl.voiceagent {
    type hw
    card ${USB_CARD}
}
EOF
    echo "  wrote ~/.asoundrc wrapping card ${USB_CARD} as 'voiceagent'"
  else
    echo "  ~/.asoundrc already exists, left untouched"
  fi
else
  echo "  no USB audio card found in 'arecord -l' -- skipping; if your mic"
  echo "  needs rate conversion, write ~/.asoundrc by hand (see README.md)"
fi

echo "== pre-fetch Silero VAD model (small, ~2MB; avoids a first-run download) =="
python3 -c "from voice_client.vad import model_path; print('VAD model at', model_path())"

echo "== config =="
mkdir -p "$HOME/.baby-onit"
if [[ ! -f "$HOME/.baby-onit/config.yaml" ]]; then
  cp config/config.yaml.template "$HOME/.baby-onit/config.yaml"
  echo "  wrote ~/.baby-onit/config.yaml (edit serving.host if needed)"
else
  echo "  ~/.baby-onit/config.yaml already exists, left untouched"
fi
if [[ ! -f "$HOME/.baby-onit/voice.yaml" ]]; then
  cp config/voice.yaml.template "$HOME/.baby-onit/voice.yaml"
  echo "  wrote ~/.baby-onit/voice.yaml -- EDIT SPEECH_SERVER_HOST before running"
else
  echo "  ~/.baby-onit/voice.yaml already exists, left untouched"
fi

echo
echo "== done =="
echo "  1. Edit ~/.baby-onit/voice.yaml: replace SPEECH_SERVER_HOST with the"
echo "     GPU machine's reachable IP (see README.md's networking note)."
echo "  2. source .venv/bin/activate && baby-onit setup"
echo "     -- paste the LLM endpoint's API key (stored in"
echo "     ~/.baby-onit/secrets.yaml, 0600, never in config.yaml; this venv"
echo "     forces the file-based fallback -- see setup.sh's keyring note)."
echo "  3. Check mic + speaker are seen: arecord -l / aplay -l"
echo "     (USB audio device should be listed)."
echo "  4. Run it: bash run.sh"
