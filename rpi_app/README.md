# rpi_app

Copy-pasteable Pi deployment of the voice agent. The Pi runs the agent
harness (`baby_onit.py`), the mic/speaker client (`voice_client`), and the
dashboard (`voice_ui`) -- all three in one process. ASR and TTS run
remotely: a separate GPU machine runs `speech_server` (Qwen3-ASR + Kokoro
TTS), and this app just opens websockets to it. No ASR/TTS engine, no GPU
deps, and no `moonshine-voice` are installed on the Pi -- see `setup.sh`'s
header for why.

## 1. Copy this folder to the Pi

```bash
scp -r rpi_app ubemama@<pi-ip>:~/Desktop/
```

## 2. Set up

```bash
ssh ubemama@<pi-ip>
cd ~/Desktop/rpi_app
bash setup.sh
```

Installs PortAudio/ALSA system libs, `uv`, a venv, baby-onit's core deps,
and the voice-client-only subset of the `voice` extra (mcp, websockets,
starlette, uvicorn, sounddevice, numpy, fastapi, onnxruntime -- no
moonshine-voice). Pre-fetches the Silero VAD model. Writes
`~/.baby-onit/config.yaml` and `~/.baby-onit/voice.yaml` from the templates
in `config/`, if they don't already exist.

## 3. Networking: reaching the GPU machine's speech_server

`~/.baby-onit/voice.yaml` defaults to `127.0.0.1` for `asr.url`/`tts.mcp_url`/
`tts.audio_url` -- not because speech_server runs on the Pi, but because the
GPU machine (WSL2, whose own IP is NAT'd and unreachable from the Pi
directly) runs a **reverse SSH tunnel** back to the Pi, forwarding its own
8771/8772 onto the Pi's loopback. From inside WSL2:

```bash
ssh -N -R 8771:127.0.0.1:8771 -R 8772:127.0.0.1:8772 ubemama@<pi-ip>
```

This works because WSL2 can always reach the Pi outbound even though the
Pi can't reach WSL2 inbound. On the laptop this repo was built on, it runs
as a systemd --user service (`~/.config/systemd/user/pi-reverse-tunnel.service`,
`Restart=always`) with `loginctl enable-linger` so it survives logout/reboot.
Passwordless key auth (`ssh-copy-id`) is required for an unattended service.

If your GPU machine *isn't* behind WSL2 NAT, or you've enabled WSL2
"mirrored" networking mode (Windows 11 23H2+: `.wslconfig` ->
`[wsl2]\nnetworkingMode=mirrored`), skip the tunnel and put that machine's
real LAN IP in `voice.yaml` instead of `127.0.0.1`.

Verify reachability before step 4 (from the Pi):

```bash
nc -zv 127.0.0.1 8771
nc -zv 127.0.0.1 8772
```

## 4. Set the LLM provider's API key

`~/.baby-onit/config.yaml` ships with a placeholder `serving.host` -- edit it
first if you're pointing at a different LLM provider than the one this repo
was built against. config.yaml never holds a secret either way.

Run the interactive wizard once to store the key:

```bash
cd ~/Desktop/rpi_app && source .venv/bin/activate && baby-onit setup
```

**If this hangs with no output at all:** that's `keyring` trying to reach
the desktop's DBus Secret Service, which waits forever for a GUI unlock
prompt that can never appear over SSH. `setup.sh` already appends
`export PYTHON_KEYRING_BACKEND=keyring.backends.fail.Keyring` to this venv's
`activate` script, which forces the plain-file fallback
(`~/.baby-onit/secrets.yaml`, mode 0600) instead -- re-source the venv
(`source .venv/bin/activate`) if you hit this on an older checkout.

```bash
source .venv/bin/activate
baby-onit setup
```

Paste the key when prompted for the `http://...` host already shown; blank
answers keep anything already stored.

## 5. Check the mic/speaker

```bash
arecord -l   # should list the USB audio device
aplay -l
```

`setup.sh` writes `~/.asoundrc` defining a `voiceagent` ALSA "plug" device
wrapping the USB card, and `voice.yaml.template` points `audio.input_device`/
`output_device` at it. This is needed because a raw ALSA `hw:N,0` device only
accepts its native sample rate (usually 48kHz) -- opening it directly at the
16kHz the agent needs fails with `PortAudioError: Invalid sample rate
[PaErrorCode -9997]`. `plug` resamples transparently. Verify it after
setup.sh runs:

```bash
source .venv/bin/activate
python3 -c "import sounddevice as sd; print(sd.query_devices())"
# should list 'voiceagent' with input and output channels
```

If `setup.sh` didn't detect a USB card (no `USB` match in `arecord -l`),
write `~/.asoundrc` by hand -- substitute your card number from `arecord -l`:

```
pcm.voiceagent {
    type plug
    slave.pcm "hw:<card>,0"
}
ctl.voiceagent {
    type hw
    card <card>
}
```

## 6. Run

```bash
bash run.sh
```

Opens the dashboard at `http://<pi-ip>:7860`. The mic starts listening
immediately; speak, and the agent should respond through the speaker once
`speech_server` on the GPU machine answers the ASR/TTS calls.

If it logs `no speak tool: is speech_server running at ...?` at startup, the
GPU machine's `speech_server` either isn't running or isn't reachable --
recheck step 3.

## Files

- `baby_onit.py`, `voice_client/`, `voice_ui/`, `speech_server/` (source
  only -- `speech_server`'s code ships here just to satisfy the package
  build layout; its deps are never installed on the Pi), `pyproject.toml`,
  `README.md`, `LICENSE` -- copied from the main repo.
- `config/*.template` -- copied to `~/.baby-onit/` by `setup.sh` on first
  run only (never overwrites an existing file).
- `setup.sh`, `run.sh` -- the two scripts you actually run.
