# Reproducing this on another machine

Everything below assumes a Linux x86_64 box with Python 3.12 and
[uv](https://docs.astral.sh/uv/). The voice stack is CPU-capable throughout;
a GPU only buys speed, and every GPU path has a CPU fallback.

## 1. Clone and base install (always do this)

```bash
git clone <this repo> voice-agent && cd voice-agent
bash scripts/setup.sh
```

Gives you `baby-onit` (the text harness), `speech_server` (ASR + TTS), and
`voice_ui` (the dashboard) — all CPU-only. Speech models are **not** bundled;
they download on first run.

```bash
source .venv/bin/activate
python -m speech_server --prewarm     # downloads Moonshine ASR + TTS models
python -m voice_ui                    # http://127.0.0.1:7860
```

That's the whole reproduction for the default, two-process configuration.
Everything from here is optional, and each option is independent of the others.

## 2. Optional: TTS on the GPU

```bash
bash scripts/setup.sh --gpu-tts
```

~5x faster synthesis (RTF 0.047 vs 0.25, measured) in ~620 MiB of VRAM. Falls
back to CPU automatically if CUDA is unusable — `speech_server`'s log says
which one actually won; **don't assume the GPU path is active without reading
that line**. Two silent-fallback traps are documented in
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) §1.1:

* `onnxruntime-gpu` ≥1.23 needs CUDA 13 / driver R580+ and will *not* error
  on an older driver — it just runs on CPU. The setup script pins `1.22.0`,
  the last CUDA-12 build.
* The pip `nvidia/*/lib` wheels are not on the dynamic loader's search path by
  default. The code works around this (`KokoroOnnxTts._preload_cuda_libs`),
  but if you ever see `libcublasLt.so.12: cannot open shared object file`,
  that is this trap, not a missing package.

Run `python -m speech_server --prewarm --tts-load gpu` once to confirm — that
flag *requires* CUDA and fails loudly instead of degrading, which is the
fastest way to find out whether the GPU path actually works on a new machine.

## 3. Optional: Qwen3-ASR (a third process)

```bash
bash scripts/setup_qwen.sh
```

8x faster than Moonshine (RTF 0.11 vs 0.87) and tied on accuracy (0.0% WER on
a real recording, both), but needs a **separate venv** (`.venv-qwen`, vLLM
pins its own torch build), **~6.5 GB of VRAM measured with TTS also running**,
and gives up live partial transcripts — Qwen is an encoder feeding an LLM
decoder, not a transducer, so text only arrives after you stop speaking, not
while you talk. Full writeup, with the measured numbers: ARCHITECTURE.md §7.

```bash
./scripts_qwen_serve.sh                                            # terminal 1
python -m speech_server --prewarm --asr-load gpu --tts-load gpu    # terminal 2
python -m voice_ui                                                 # terminal 3
```

**The vLLM wheel URL in `scripts/setup_qwen.sh` will eventually rot** — it
names a specific commit on a frozen nightly channel. The script's header
explains exactly how to find a replacement when it 404s.

## 4. Network access

`voice_ui` binds to `127.0.0.1` by default. For other machines on the LAN:

```bash
python -m voice_ui --host 0.0.0.0
```

This has **no authentication** — anyone who can reach the port can read the
conversation and control the agent. Fine on a trusted LAN; not otherwise.

## 5. Audio routing (mic/speaker, not just software)

Two independent devices are strongly recommended — input and output sharing
one device risks the agent hearing itself and barge-in firing on its own
voice. Verified safe on this build's dev machine (0.99× idle floor with the
agent speaking); re-measure on any new machine before trusting it:

```bash
pactl info | grep -i 'default source\|default sink'
```

## 6. Bringup matrix

| Configuration | Processes | ASR | TTS |
|---|---|---|---|
| Default | 2 | Moonshine, CPU, live partials | Kokoro, CPU or GPU |
| + GPU TTS | 2 | same | Kokoro, GPU (`--tts-load gpu`) |
| + Qwen3-ASR | 3 | Qwen, GPU, no partials | Kokoro, must be CPU or GPU (both fit together, measured 6.4 GB total) |

## Where to go deeper

* [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — every measured number, every
  trap hit while building this, and why each design decision was made.
* [docs/VOICE_PROTOCOL.md](docs/VOICE_PROTOCOL.md) — the frozen client/server
  wire contract.
* [docs/PLAN_ASYNC_CONVERSATION.md](docs/PLAN_ASYNC_CONVERSATION.md) —
  background tasks and conversational turn-taking, if extending either.
* [speech_server/README.md](speech_server/README.md),
  [voice_client/README.md](voice_client/README.md),
  [voice_ui/README.md](voice_ui/README.md) — per-component detail.
