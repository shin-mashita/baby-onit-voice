# Full-Duplex Voice Agent — Architecture

A full-duplex voice agent built on [baby-onit](https://github.com/sibyl-oracles/baby-onit):
the user speaks, the agent answers out loud immediately, and **keeps talking
while it works** — narrating long tool-using tasks instead of going silent.
The user can barge in and cut it off mid-sentence.

Target deployment: a **Raspberry Pi 5** runs orchestration and VAD; a **laptop**
hosts the ASR and TTS endpoints; the LLM is reached through a **Portkey** gateway.

> Status: architecture + component validation complete. All latency figures in
> this document are **measured on the target laptop** (Dell Precision 5570,
> i7-12700H, 20 cores, Ubuntu 22.04), not estimated. See
> [Measured performance](#measured-performance).

---

## 1. Decisions and why

| Concern | Decision | Rationale |
|---|---|---|
| Orchestration + VAD | Raspberry Pi 5 | User requirement; the Pi stays light |
| ASR + TTS | Laptop, over the network | Keeps heavy inference off the Pi |
| ASR engine | Moonshine **Streaming v2** | Genuinely frame-streaming; partials arrive mid-utterance |
| TTS engine | Moonshine TTS (CPU). `kokoro-onnx` implemented but **unusable here** | This machine's GPU cannot initialize CUDA — see below |
| LLM brain | Portkey (OpenAI-compatible) | Provider-agnostic; no local VRAM cost |
| Agent harness | baby-onit on branch `feature/voice-agent` | Small, single-file, easy to patch |

### 1.1 Two findings that reshaped the original plan

**vLLM cannot serve either speech model.** This is categorical, not a tuning
problem. vLLM's supported-models matrix lists 14 ASR architectures
(`WhisperForConditionalGeneration`, `VoxtralFor…`, `CanaryFor…`, `Qwen3ASRFor…`,
…) and **zero TTS architectures of any kind**. Moonshine is not among the ASR
entries. Kokoro is StyleTTS2 + ISTFTNet — not an autoregressive transformer LM —
so it falls outside vLLM's execution model entirely. The original plan to serve
both via vLLM was dropped.
Source: <https://docs.vllm.ai/en/latest/models/supported_models.html>

**Moonshine Voice already is the whole speech stack.** `pip install
moonshine-voice` (v0.1.5, MIT) ships streaming ASR, TTS, VAD, mic capture,
diarization and voice cloning in one ~19 MB wheel with **no torch dependency**
— it pulls only `numpy`, `sounddevice`, `tqdm`, `platformdirs`, `filelock`,
`google-crc32c`. Models are ONNX Runtime `.ort` files. This collapsed three
planned services into one.

It also **bundles Kokoro as a vocoder**: voice ids are prefixed `kokoro_`,
`piper_`, or `zipvoice_`. All 20 Kokoro voices are available (62 voices total
for `en-us`), so the "Moonshine vs Kokoro" tradeoff disappeared — we run
Moonshine's pipeline *with* a Kokoro voice (`kokoro_af_heart`).

> **Moonshine cannot use the GPU, and fails silently if you ask it to.**
> `libmoonshine.so` bundles its own ONNX Runtime registering only CPU, CoreML
> and NNAPI; no CUDA provider library ships with the package and the bundled
> runtime links no libcuda/libcudnn. There *is* a `use_cuda` option string, and
> passing `options({"use_cuda": True})` is **accepted without error and does
> nothing** — VRAM stays at idle through a successful synthesis. Verified.
>
> Since that CPU-only synthesis is what causes the ASR contention in §5.11, TTS
> now defaults to **kokoro-onnx on CUDA** instead. It is the same Kokoro model
> and the same voices (`af_heart` et al.) that the Moonshine path was already
> using, so this changes where synthesis runs, not how the agent sounds.
> `--tts-backend auto` falls back to Moonshine when CUDA or kokoro-onnx is
> unavailable.
>
> **The GPU path is implemented but does not work on this machine.**
> `KokoroOnnxTts` is complete and correct, and selectable with
> `--tts-backend kokoro`. It currently runs on CPU, where it is *slower* than
> Moonshine (746 ms vs ~500 ms TTFA), so **the default stays `moonshine`**.
>
> The blocker is the GPU itself, not the code. Diagnosis, in order:
>
> 1. `onnxruntime-gpu` 1.31 requires CUDA 13 + cuDNN 9; driver 535 is CUDA 12.2.
>    It logs an error, silently drops to CPU, and synthesizes anyway. Pinned
>    **1.22.0**, the last CUDA-12 build.
> 2. ORT could not find the pip CUDA libraries — the dynamic loader reads
>    `LD_LIBRARY_PATH` only at process start. Preloading the 22 `nvidia/*/lib`
>    shared objects with `ctypes.CDLL(..., RTLD_GLOBAL)` fixes that.
> 3. With the libraries loaded, the CUDA EP reached `cudaSetDevice` and failed:
>    `CUDA failure 3: initialization error ; GPU=-1`.
> 4. A bare runtime probe, independent of onnxruntime, fails the same way:
>    `cudaGetDeviceCount -> rc=3 (initialization error), devices=0`.
> 5. Not a version mismatch: pinning the runtime to **12.2.140**, exactly
>    matching the 12.2 driver, fails identically.
> 6. Not permissions or modules: kernel module and userspace both 535.309.01,
>    `nvidia`/`nvidia_uvm`/`nvidia_modeset`/`nvidia_drm` all loaded, every
>    `/dev/nvidia*` node present and world read-writable.
>
> What is left is the GPU's own state. `nvidia-smi` warns **"infoROM is corrupted
> at gpu 0000:01:00.0"**, every inforom object reports `N/A`, and `nvidia-smi -q`
> **hangs** (killed at 120 s) while basic queries return instantly. CUDA
> initialization failing on a card whose inforom will not read is consistent.
>
> **This needs a reboot or a driver reinstall to confirm, which is a privileged
> system change and the operator's call, not something to work around in this
> repo.** Until then the CPU path plus mic-parking (§5.11) is the supported
> configuration, and it is a complete, working system.

---

## 2. Topology

```
┌─ Raspberry Pi 5 ────────────────┐      ┌─ Laptop (20-core CPU) ──────────┐
│                                 │      │                                 │
│   mic                           │      │   speech_server/                │
│    │                            │      │                                 │
│    ▼                            │      │   ┌─────────────────────────┐   │
│   AEC (webrtc)                  │      │   │ ASR   ws://host:8771/asr│   │
│    │                            │      │   │  Moonshine Streaming v2 │   │
│    ▼                            │      │   └─────────────────────────┘   │
│   VAD ──── PCM frames ──────────┼─────▶│          ▲ 16 kHz f32 frames    │
│    │                            │      │          │                      │
│    │       partial / final  ◀───┼──────┼──────────┘ JSON events          │
│    │                            │      │                                 │
│    ▼                            │      │   ┌─────────────────────────┐   │
│   turn detector                 │      │   │ TTS  MCP  :8772/mcp     │   │
│    │  (trailing silence)        │      │   │  speak / stop / status  │◀──┼── control
│    ▼                            │      │   └─────────────────────────┘   │
│   baby-onit (voice mode) ───────┼─────▶│   ┌─────────────────────────┐   │
│    │                            │      │   │ TTS  ws://:8772/audio   │   │
│    │       PCM chunks      ◀────┼──────┼───│  Kokoro vocoder, 24 kHz │   │
│    ▼                            │      │   └─────────────────────────┘   │
│   player ──▶ speaker            │      └─────────────────────────────────┘
│    │                            │
│    └── LLM ────────────────────────────────▶ Portkey ──▶ model
└─────────────────────────────────┘
```

### 2.1 The load-bearing idea: split TTS into a control plane and a data plane

MCP tools return JSON text, so **audio must not flow through MCP**.

- **Control plane (MCP).** `speak(text)` enqueues an utterance server-side and
  returns immediately with `{utterance_id, status: "speaking"}`. Also
  `stop_speaking()` and `is_speaking()`.
- **Data plane (websocket).** PCM streams down a separate persistent socket to
  the player, framed by the `utterance_id` / `is_final` fields that
  `TtsChunk` already carries natively.

This is what makes "talk while working" possible: the agent loop proceeds to the
next tool call while audio is still playing. **A blocking `speak()` would
serialise speech and work and defeat the entire goal.**

### 2.2 Phase 1 vs Phase 2

Identical process topology; only hostnames in `voice.yaml` change.

- **Phase 1 (now)** — all four processes on the laptop, against its built-in
  mic/speakers. The network split is exercised over loopback from day one.
- **Phase 2** — `pip install moonshine-voice` on the Pi (ARM64 wheels are
  published), repoint `voice.yaml`, done.

---

## 3. Measured performance

All on the laptop, CPU-only, English.

### ASR, measured on real speech

Earlier figures in this document came from Kokoro-synthesized audio, on which all
three streaming models score 0.0% WER -- it cannot discriminate and it overstates
real-world accuracy. On a real recording of a human (6.0 s, -18.5 dBFS):

| model | RTF | WER | error |
|---|---|---|---|
| `TINY_STREAMING` | 0.27 | 5.3% | flappy -> floppy |
| `SMALL_STREAMING` | 0.61 | 5.3% | flappy -> floppy |
| **`MEDIUM_STREAMING`** | **0.88** | **0.0%** | — |

Clipping was ruled out as a cause: the recording had 43 samples at the rail, and
attenuating it to a 0.70 and 0.50 peak (0 clipped) changed the WER not at all.

`python -m voice_client.miccheck "<what you will say>"` reproduces this table on
demand, and `--wav` re-runs it against a saved recording.

### ASR (Moonshine Streaming v2), synthesized audio

| Metric | `TINY_STREAMING` | `SMALL_STREAMING` |
|---|---|---|
| Model load (warm) | 0.10 s | 0.20 s |
| Real-time factor | **≈ 0.23** | ≈ 0.64 |
| First partial emitted | 1.10 s | 1.32 s |
| Final vs. end of audio | **−0.33 s** | −0.20 s |

The negative figure is the important one: **transcription completes *before* the
audio does.** The sliding-window encoder really does work ahead of the speaker.
Partials arrive from ~1.1 s into a 6.6 s utterance, confirming true streaming
rather than VAD-chunked re-decode.

`TINY_STREAMING` is the default. `SMALL_STREAMING` at RTF 0.64 leaves too little
headroom once TTS shares the CPU.

### TTS (Kokoro `af_heart`, 24 kHz)

| Metric | Value |
|---|---|
| Warm load | 0.59 s |
| One-shot `synthesize()` RTF | 0.227 |
| **Time-to-first-audio, one-shot** | **0.315 s** |
| **Time-to-first-audio, at token cadence** | **0.677 s** |
| Streaming RTF (at token cadence) | 0.425 |
| `cancel_stream()` | **0.0 ms** |

Barge-in truncated a ~16 s reply to 1.86 s of audio, with `is_streaming`
flipping to `False` immediately.

An earlier revision of this document claimed `cancel_stream()` blocks 82-156 ms
with synthesis in flight. **That was wrong.** The call is effectively free; it
only measured slow against a consumer that was busy-spinning on `next_chunk()`
with no sleep, which starved the GIL. With that fixed the call costs nothing.

The deferred-cancel design (a canceller thread, rather than calling it inline on
the shared event loop) is still right -- a native call has no business on the
loop that serves the audio sockets -- but it is defensive, not a workaround for a
slow call. End-to-end `stop_speaking` is **2.9 ms**.

### Combined budget — RTF is the wrong model here

Adding the RTFs (0.23 + 0.23 ≈ 0.46, "comfortable") is **wrong**, and this
document said it. RTF implicitly assumes each model occupies about one core.
Measured, a single TTS synthesis runs **49 threads at ~1393 % CPU — roughly 14
of this box's 20 cores**. ASR fans out the same way. Two such workloads do not
add up to half a machine; they oversubscribe it and fight over memory bandwidth.

See §5.11 for what that costs and how the client avoids paying it.

### Through the server, not the bare engine

The figures above are the engine called directly. Through `speech_server` —
MCP round trip, queue pickup, synthesis, websocket — measured over 10 warm
trials with an audio client attached:

| Metric | Median | Range |
|---|---|---|
| `speak` control-plane round trip | **1.8 ms** (p90 2.1) | 2.6–3.5 ms under load |
| First `speak` on a new MCP connection | 62 ms | connect+initialize ≈53 ms |
| **Time to first PCM frame** | **551 ms** | 383–626 ms |
| `stop_speaking` round trip | 2.8–3.1 ms | — |

The ~236 ms over the bare engine's 315 ms is the server layer and is the price
of the control/data split. An earlier build measured 473–1408 ms with wild
variance; that was the deferred-cancel race in §5.8, not a throughput limit.

### End-to-end: what the user actually experiences

```
user stops speaking
   +700 ms   turn detector confirms end of turn (silence_ms)
   +551 ms   first filler audio leaves the speaker  (TTFA)
   = ~1.25 s before the agent makes a sound
```

Both terms are tunable and neither is dominated by the LLM — the whole point of
the deterministic filler (§5.3) is that time-to-first-token is hidden behind it.
`silence_ms` is the bigger lever: dropping it to 500 ms brings this to ~1.05 s at
the cost of cutting off slower speakers mid-thought. **This is the number to
optimise if the agent feels sluggish, and it should be measured with real mic
audio before being tuned** — every figure here is from synthesized speech, which
has a softer onset than a live microphone.

---

## 4. Components

### 4.1 Harness patches — `baby_onit.py`

baby-onit is one 1546-line file. Patches keep that ethos. Verified gaps in
stock code at HEAD `f356bed`:

| # | Change | Location |
|---|---|---|
| 1a | Async tool support: `iscoroutinefunction` branch beside the `to_thread` path | `dispatch()` `:840` |
| 1b | **MCP client** — the first one in this repo (§4.2) | new section S5.5 |
| 1c | **True streaming** in `Provider.chat` (§4.3) | `:269-299`, `:301-400` |
| 1d | Voice system-prompt variant keyed on `cfg["voice"]` | `SYSTEM_PROMPT` `:408` |
| 1e | **Bypass the P5 tool-hiding guard in voice mode** | `:945-949` |
| 1f | `serving.extra_headers` → `AsyncOpenAI(default_headers=…)` for gateways | `_make_client()` `:217` |
| 1g | Make provider quirks model-aware, not host-aware | `:355`, `:402` |
| 1h | `cmd_voice` owning one persistent event loop + cancellable agent task | S9 |

Reused as-is: `tool()` (`:725`) for reflective schema generation,
`on_event(kind, text)` (`:1152`) as the voice-UI seam, `_tool_error()` (`:817`)
for MCP error shapes, `session_history.jsonl` for conversational continuity.

#### Why 1e is not optional

`baby_onit.py:945` hides the **entire** tool surface on iteration 1 unless the
task text matches `_TASK_TOOL_HINTS` (`:928`). The phrase *"Set an alarm at
7 am"* matches no hint word, so the model sees no tools, cannot call `speak`,
and **the agent answers silently.** One of the project's own target utterances
breaks on stock code.

#### Why 1c matters

`_stream_with_retry()` (`:269`) does
`return [c async for c in await ...create(**body)]` — it drains the whole stream
into a list, then `chat()` replays it into `stream_cb` *after the fact* (`:379`)
purely to timestamp time-to-first-token. Nothing can speak incrementally until
this becomes a real async generator.

### 4.2 MCP client

baby-onit has **no MCP client** — zero `mcp` dependency, and `docs/PLAN.md:78`
records that MCP was deliberately dropped in favour of in-process functions. We
add one: streamable-HTTP transport, servers declared under an `mcp.servers`
config block, `tools/list` fetched at startup and merged into the existing
`TOOLS` registry so MCP tools are indistinguishable to the model.

### 4.3 LLM → TTS token pipeline

Moonshine's `push_text()` **buffers until a sentence completes** (synthesising
half a clause gets the prosody wrong), so the pipeline is:

```python
with tts.say_stream() as speech:        # non-blocking handle
    for delta in llm_deltas():          # from patched Provider.chat
        tts.push_text(delta)            # held until sentence boundary
    tts.end_input()
```

`stream()` distinguishes "no complete sentence yet" from "finished" from
"cancelled"; a raw `next_chunk()` loop cannot. `cancel_stream()` from another
thread ends the iteration mid-reply — that is the barge-in path.

### 4.4 `speech_server/` (laptop)

- `asr_ws.py` — websocket; 16 kHz mono float frames in, `{type, text, ts}` out.
  **`Stream.add_audio()` is synchronous and does the encoding work on the
  calling thread**, so it must run in a thread executor or it will stall the
  asyncio loop.
- `tts_mcp.py` — MCP server: `speak`, `stop_speaking`, `is_speaking`.
- `tts_audio_ws.py` — PCM data plane, framed by `utterance_id` / `is_final`.
- `backends.py` — `TtsBackend` protocol (`synthesize` / `stop` / `is_talking`),
  `MoonshineTts` default, `KokoroTts` stub for the GPU path.

### 4.5 `voice_client/` (Pi 5, or laptop in Phase 1)

- `capture.py` — `sounddevice` mic → AEC → VAD → ASR websocket.
- `turn.py` — end-of-turn detection (§5.1).
- `player.py` — PCM queue → speaker, with `flush()` for barge-in.
- `barge_in.py` — VAD-during-playback → flush player, `stop_speaking()`,
  cancel the agent task.

---

## 5. Risks and mitigations

### 5.1 `LineCompleted` is a line boundary, not a turn boundary — **measured**

Feeding one 6.6 s utterance produced **two** `LineCompleted` events:

```
 5.62  FINAL  "I am thinking of creating a simple web game. Can you create me?"
 7.81  FINAL  "A very good flappy bird app."
```

A driver that dispatched an agent turn per `on_line_completed` would fire **two
turns for one sentence** — the first truncated mid-question, the second a
fragment. Both would be nonsense. The library's own `MicTranscriber.on_line()`
fires on `LineCompleted` too (`mic_transcriber.py:219`), so it offers no
end-of-turn detection either.

**Mitigation.** Turn-taking is driven by the **VAD's trailing silence**, not by
ASR line events. `turn.py` accumulates completed lines and dispatches only after
~700 ms of silence. ASR lines update the live transcript; the VAD decides when
the human is done.

### 5.2 Acoustic echo — **measured as a non-issue on this hardware**

Measured on the development machine with a duplex `sd.playrec()` while capturing
the sink monitor to prove output was real:

| | rms | peak | VAD frames >= 0.5 |
|---|---|---|---|
| Idle room | 0.00249 | 0.0102 | 0 / 93 |
| Mic while the agent speaks | 0.00260 | 0.0105 | **0 / 275** |
| Sink monitor, same moment | 0.194 | 0.3195 | — (proves playback was real) |

The mic sits at **1.0x the idle floor** while the agent talks, and the barge-in
detector fires zero times with `agent_speaking=True`. The output goes to a UGREEN
CM564 USB adapter whose headphone output and mic input share no acoustic path,
so there is nothing to cancel. **AEC is not needed in this configuration.**

Two caveats, because this result is about the hardware and not the software:

* The built-in analog sink is currently **muted**. Switching playback to laptop
  speakers puts the mic and speaker back in the same air and the analysis below
  applies again in full.
* The Pi 5 will have its own audio hardware; this has to be re-measured there.

Also worth knowing when writing these tests: `sd.play()` and `sd.rec()` share one
module-level stream, so issuing `rec` after `play` **silently cancels the
playback**. That makes a mic look perfectly isolated when in fact nothing was
ever played. Two of my own measurements were wrong this way before I started
capturing the sink monitor as a control. Use `sd.playrec()` for duplex.

### 5.2.1 If echo does appear (speaker configurations)

Mic and speaker on one device means the agent's own voice re-enters the mic and
the barge-in VAD interrupts *itself*, every utterance. Three layers, all needed:

1. PulseAudio `module-echo-cancel aec_method=webrtc` — already installed on the
   laptop (`libwebrtc-util.so`, `libwebrtc_audio_processing.so.1`,
   `libspeexdsp.so.1` all present; server is PulseAudio, not PipeWire).
2. Gate the VAD against `is_speaking()`.
3. Require ~300 ms of *sustained* speech before honouring a barge-in.

Headphones make Phase 1 tractable while this is tuned.

### 5.3 "Always call the TTS MCP first" is fragile — don't rely on the prompt

A prompt instruction is soft (models drop it under load) and costs a full LLM
round-trip before the first sound. **The harness speaks deterministically
instead**: the moment the turn detector fires, the voice driver speaks a short
filler locally, before the first LLM token exists. Zero prompt-compliance risk,
and it hides the entire time-to-first-token. The system prompt then only has to
handle *narration during work*, a much softer ask. The prompt instruction stays
as reinforcement, not as the mechanism.

### 5.4 Barge-in must cancel the agent, not just the audio

Flushing the speaker while the agent keeps working leaves it monologuing into a
dead channel. Cancellation reaches the `asyncio.Task` wrapping `agent_loop` and
the in-flight LLM request. Completed tool results stay in history, so an
interrupted turn is resumable.

### 5.5 Cancelling the agent is not the same as cancelling the turn

Observed in live use as replies arriving out of order or overlapping. Barge-in
cancelled the in-flight `agent_loop` task and flushed the speaker, but four
other things survived the cancel and resurfaced afterwards:

1. **The `_pending_turn` queue was never drained.** An utterance queued while
   the agent was working got answered after the interruption — a reply to
   something the user had already moved past.
2. **A `_resolve_turn` waiting on ASR kept waiting.** Its text arrived after the
   barge-in and was queued behind it.
3. **Every `_on_turn_end` spawned another resolver.** Two racing on the same
   `transcript`/`partial` could dispatch overlapping text for one utterance,
   which sounds like the agent answering twice.
4. **`speak` enqueues server-side.** A new turn's filler landed *behind* a stale
   answer still synthesising, so the two played in the wrong order.

The fix is a **generation counter**, bumped by every barge-in. `_resolve_turn`
and `handle_turn` capture it on entry and discard their work if it has moved on;
the queue is drained on interrupt; one resolver runs at a time; and a new turn
calls `stop_speaking` before adding to the TTS queue. Regression-tested in
`voice_client/selftest.py` [5].

The general shape is the same as §5.8: **cancellation has to reach every stage
of the pipeline, not just the one holding the task handle.**

### 5.5.1 Narration must not hijack the loop

"Describe what you're building" invites a `speak`-per-thought token burn.
Narration is bounded to **tool boundaries**, driven from the existing
`on_event("tool", …)` callback rather than left to model discretion.

### 5.6 Keyterm biasing is inert without `keyterm_boost`

An earlier revision of this document claimed `set_keyterms()` fixed "floppy" ->
"flappy". **That was wrong, and the test that produced it was confounded** --
the model was changed from TINY to SMALL in the same run, so the improvement was
the model, not the biasing.

Measured properly, on a real recording of a human saying "...a very good flappy
bird app", with `set_keyterms(["flappy", "flappy bird"])` applied:

| `keyterm_boost` | TINY | SMALL |
|---|---|---|
| unset / 1.0 / 2.0 | floppy | floppy |
| **5.0** | **flappy** | **flappy** |
| 10.0 | flappy | flappy |

Below ~5 the biasing is **indistinguishable from not setting keyterms at all**;
the terms are accepted and silently do nothing. The strength is a load-time
`options={"keyterm_boost": ...}` on the `Transcriber`, not a parameter of
`set_keyterms()`, which is easy to miss -- the docstring mentions it in one line.

**There is a hard ceiling.** At 20.0 and above the decoder stops transcribing
and emits a keyterm on repeat; with four irrelevant terms loaded the entire
6-second utterance came back as `"kubernetes kubernetes kubernetes..."`. Even at
5.0 there is mild bleed: irrelevant terms turned "simple web game" into "simple
webgig". So the terms must be few and genuinely expected.

`KEYTERM_BOOST = 5.0` is the default, `--keyterm-boost` overrides it, and values
above `KEYTERM_BOOST_MAX = 10.0` are clamped with a warning.

### 5.6.1 Model size still matters more

On the same recording, `MEDIUM_STREAMING` transcribed "flappy" correctly **with
no keyterms at all**, where TINY and SMALL both needed boost 5.0. Keyterm
biasing only helps words you thought to list; a larger model helps everywhere.
Use both.

### 5.7 Never `break` out of a cancelled TTS stream

Abandoning a cancelled `stream()` generator early -- `break` plus `close()` --
leaves the native engine wedged: the *next* `stream()` yields zero chunks and
`is_streaming` sticks at `True` permanently, for the life of the process. This
is not documented upstream; it was found by a barge-in selftest failing on its
second utterance.

On barge-in, keep iterating the generator and discard what it yields until it
terminates on its own. That costs nothing -- `cancel_stream()` has already made
it return immediately -- and it is the only safe way to end a reply early.

### 5.8 A deferred cancel can kill the wrong utterance

`stop_speaking` hands `cancel_stream()` to a canceller thread and returns in
~3 ms, so there is a window between the two. A `speak` issued inside it can have
its `stream()` started *before* the cancel lands, so the cancel kills **the new
utterance instead of the one it was aimed at**.

That window is exactly the barge-in-then-filler sequence -- interrupt the agent,
immediately acknowledge the new utterance -- which is the single most common
thing this system does. The symptom is "the agent goes silent after an
interrupt", intermittently: it reproduced 1 run in 3.

Fix: a `_cancel_done` event, cleared by `stop_speaking` and set by the canceller
once `cancel_stream()` has returned. The synthesis worker waits on it before
touching the engine, bounded so a wedged canceller cannot mute the server
permanently, and re-checks the generation afterwards so a barge-in arriving
during the wait still drops the right utterance.

This shares a root cause with §5.7: the engine has process-wide state, and any
two operations that touch it concurrently need ordering, not just locking.

### 5.9 `utterance_id` is per-`stream()`, not per-`speak()`

`TtsChunk.utterance_id` restarts at 1 on every `stream()` call, so raw ids
repeat across `speak` calls and cannot identify an utterance on the wire. The
TTS service therefore adds a per-`speak` base offset to keep ids globally
monotonic, reserving the range with the engine's `split_say_utterances()` so the
id `speak` returns is the one the first chunk actually carries.

### 5.10 Portkey: the model slug does the routing, not headers

Verified working against the live gateway. Portkey's `@<integration>/<model>`
model id carries the routing, and the Portkey key travels in `Authorization:
Bearer` like any OpenAI-compatible key, so **no extra headers are needed**:

```yaml
serving:
  host: https://api.portkey.ai/v1
  model: '@gcp-vertexai-us-global-integration/anthropic.claude-sonnet-5'
```

`serving.upstream` can be left at `auto` here -- the model id contains "claude",
so the Anthropic quirk (no `temperature`+`top_p`) is inferred correctly. Set it
explicitly only when the model id does not name the family.

`serving.extra_headers` stays for gateways that *do* route on headers
(virtual keys, LiteLLM). It takes a **mapping**, and writing a bare header name
there is the obvious mistake:

```yaml
    extra_headers:                      # right
      x-portkey-virtual-key: vk-abc123
    extra_headers: x-portkey-api-key    # wrong: a name with no value
```

`Provider._extra_headers()` rejects the second form with a message naming the
key, because unvalidated it surfaces as `TypeError: 'str' object is not a
mapping` from inside the OpenAI SDK, several frames from anything recognisable.

### 5.11 ASR and TTS contend for memory bandwidth — park the mic

The single biggest latency effect in the system, and the reason the RTF
arithmetic in §3 was misleading.

| Condition | TTS time-to-first-audio |
|---|---|
| ASR idle | ~500 ms |
| **ASR ingesting concurrently** | **~2350 ms** |
| ASR parked while the agent speaks | **494 ms** |

Concurrent ASR ingest costs TTS **4.3x** its time-to-first-audio. The obvious
fixes do not work: running ASR in a **separate process** (1597 ms) or pinning the
two to **disjoint CPU sets** (1546 ms) change nothing, and capping
`OMP_NUM_THREADS=4` cut thread count 49 -> 33 while leaving the penalty at 4.1x.
It is not the GIL and not core scheduling -- it is two ONNX graphs competing for
memory bandwidth and last-level cache.

**The fix is to not do both at once.** Nothing needs transcribing while the agent
is talking: barge-in runs off the local Silero VAD, not off ASR. So
`driver._on_mic_frame` stops forwarding mic audio while playback is active, and
keeps the frames in a short ring (sized to `barge_in_ms` plus margin) so an
interruption does not lose its opening syllables. That recovers TTFA to 494 ms.

This is also the strongest argument for keeping the speech models **off** the
Raspberry Pi. A workload that wants ~14 cores for one synthesis has no business
on 4 Cortex-A76s; the split topology is not just convenient, it is required.

### 5.12 Diagnosability: keep a transcript

A voice session leaves no scrollback worth reading — the terminal shows log
lines, not the dialogue — so "it went wrong" cannot be traced to the turn that
caused it. `voice_client` appends a JSONL record of every turn, answer,
barge-in and discarded utterance to `~/baby-sandbox/voice_turns.jsonl`
(`transcript_log: null` disables it). The generation number is in every record,
which is what makes an ordering complaint diagnosable at all:

```jsonc
{"t":"...","gen":1,"event":"barge_in","dropped_samples":0,"discarded_turns":2}
{"t":"...","gen":2,"event":"discarded_utterance","gen_at_capture":1,"text":"..."}
{"t":"...","gen":2,"event":"turn","text":"Set an alarm at 7 am"}
{"t":"...","gen":2,"event":"answer","text":"Done, your alarm is set for seven a.m."}
```

### 5.13 Operational notes

- Disk is at 95 % (48 GB free). Moonshine wheels are ~19 MB and models a few MB,
  so this is fine — but a Kokoro-FastAPI Docker image would be tight.
- First TTS `load()` took 103 s because it downloads voice assets; warm load is
  0.59 s. Pre-warm assets during provisioning, not on first utterance.

---

## 6. Verification

1. **Harness unaffected** — `python3 _selftest.py` → `offline: PASS`;
   `python3 _difftest.py <pristine>` → `DIFFTEST: PASS`. Note difftest never
   exercises `chat()`, so the streaming refactor needs its own test.
2. **Portkey reachable** — `baby-onit setup --show`, then
   `baby-onit run "what is 2+2"`.
3. **ASR offline** — feed a recorded WAV through the websocket at 1× realtime;
   assert partials arrive before the utterance ends.
4. **TTS control/data split** — `speak` over MCP must return in <50 ms while
   audio is still streaming on the data plane.
5. **The silent-agent regression** — `--voice "Set an alarm at 7 am"` must
   produce a `speak` call (fails on stock code, §4.1 item 1e).
6. **Turn detection** — the two-line utterance in §5.1 must dispatch exactly
   **one** agent turn.
7. **Full duplex** — filler audible before the first LLM token; narration on
   tool boundaries; `flappy_bird` files appear while audio still plays.
8. **Barge-in** — audio stops <300 ms, agent task cancels, new utterance starts
   a turn.
9. **Echo rejection** — with speakers (not headphones) and AEC loaded, a full
   answer plays with zero self-interruptions.
10. **Phase 2** — repoint `voice.yaml` at the Pi, re-run 7 and 8 over the LAN,
    record added latency.

---

## References

- [Moonshine Voice (PyPI)](https://pypi.org/project/moonshine-voice/) · MIT, v0.1.5
- [Announcing Moonshine Voice](https://petewarden.com/2026/02/13/announcing-moonshine-voice/)
- [Moonshine v2: Ergodic Streaming Encoder ASR](https://arxiv.org/pdf/2602.12241)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)
- [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) — GPU fallback for `TtsBackend`
