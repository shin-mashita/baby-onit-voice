# Voice Agent Wire Protocol

Frozen contract between `voice_client/` (Pi 5) and `speech_server/` (laptop).
Both sides code against this; neither side may change it unilaterally.

Default ports: ASR `8771`, TTS `8772`.

---

## 1. ASR — `ws://<host>:8771/asr`

### Client → server

**Audio frames: binary.** Little-endian `float32` PCM, mono, 16 kHz,
nominally 100 ms per frame (1600 samples / 6400 bytes). Any frame length is
accepted; the server does not require a fixed size.

**Control: text (JSON).**

```jsonc
{"type": "start",     "keyterms": ["flappy bird"], "context": null}
{"type": "reset"}      // drop the in-flight utterance, keep the socket
{"type": "finalize"}   // the utterance ended; emit a transcript for what you have
{"type": "stop"}       // close the stream cleanly
```

`keyterms` is applied via `Transcriber.set_keyterms()`. Required in practice:
without it "flappy" transcribes as "floppy" / "flashlight" (measured) — and note
it needs `keyterm_boost >= 5.0` at load time or it is silently inert. Qwen3-ASR
exposes no keyterm biasing and logs that it is ignoring the terms.

`finalize` exists because the engine may not know when the human stopped. A
frame-streaming engine (Moonshine) segments internally and ignores it. A
non-streaming engine (Qwen3-ASR, an encoder feeding an autoregressive decoder)
has no per-frame emission, so `finalize` is the **only** thing that makes it
produce a transcript, and such an engine emits **no `partial` events at all**.
The client's VAD owns the turn boundary either way, so it always sends it.

### Server → client — text (JSON), one object per line

```jsonc
{"type": "ready",   "model": "tiny-streaming-en", "sample_rate": 16000}
{"type": "partial", "line_id": 3, "text": "I am thinking of creating", "t": 2.41}
{"type": "line",    "line_id": 3, "text": "... Can you create me?",    "t": 5.62}
{"type": "error",   "message": "..."}
```

`t` is seconds since `start`.

`no_speech` says the server examined a finalized buffer and found nothing worth
transcribing. It exists so a client does not wait out its transcript timeout on
a turn that was never going to produce text — a silent drop is indistinguishable
from a slow backend. Only non-streaming engines emit it.

> **`line` is NOT a turn boundary.** One 6.6 s utterance emitted two `line`
> events (measured — see `ARCHITECTURE.md` §5.1). The client accumulates lines
> and decides turn boundaries from **VAD trailing silence**. The server never
> claims to know when the human is finished.

---

## 2. TTS control plane — MCP at `http://<host>:8772/mcp`

Streamable-HTTP transport. Tools exposed to the model:

| Tool | Arguments | Returns | Blocking? |
|---|---|---|---|
| `speak` | `text: str` | `{"utterance_id": int, "status": Status, "reason"?: str}` | **No** — returns immediately |
| `stop_speaking` | — | `{"stopped": bool}` | No |
| `is_speaking` | — | `{"speaking": bool, "queued": int}` | No |

`speak` MUST return before audio finishes — that is the whole point. It enqueues
server-side and the PCM leaves via the data plane. Measured: **2.6–9.8 ms**,
including while synthesis of a previous utterance is in flight.

`Status` is one of:

| Status | `utterance_id` | Meaning |
|---|---|---|
| `"speaking"` | the real id | Queued for synthesis. |
| `"idle"` | `0` | Nothing to say (empty or whitespace text). |
| `"rejected"` | `0` | Queue full; `reason` says which (`"queue_full"`). |

Back-pressure is reported, never blocked on: waiting for queue space would blow
the 50 ms budget and stall the agent loop mid-answer.

### 2.1 `utterance_id` is assigned by the service, not by the engine

`TtsChunk.utterance_id` restarts at 1 on every `stream()` call, so engine-native
ids repeat across `speak` calls and identify nothing on the wire. The service
adds a per-`speak` base offset so ids are globally monotonic for the life of the
process, reserving each range with `split_say_utterances()` so that the id
`speak` returns is the one the first chunk actually carries.

---

## 3. TTS data plane — `ws://<host>:8772/audio`

One persistent socket per client, opened at startup.

### Server → client

Each audio chunk is **one binary frame**: a 16-byte little-endian header
followed by `float32` PCM samples.

```
offset  size  type      field
     0     4  uint32    utterance_id
     4     4  uint32    sample_rate      (24000 for Kokoro)
     8     4  uint32    n_samples
    12     1  uint8     is_final         (1 = last chunk of this utterance)
    13     3  —         reserved (zero)
    16   4*n  float32   samples
```

`utterance_id` and `is_final` come straight from Moonshine's `TtsChunk` — no
framing is invented here. Note `utterance_id` increments **per sentence**, not
per `speak()` call (measured), so one `speak` of three sentences yields ids
1, 2, 3, each closed by its own `is_final`.

### Client → server — text (JSON)

```jsonc
{"type": "flush"}   // barge-in: drop everything queued, stop synthesis now
```

The client also drops its own playback buffer locally on barge-in; it does not
wait for the server to acknowledge.

---

## 4. Barge-in sequence

Ordered so audio stops before anything slower is attempted:

1. Client VAD detects ≥ 300 ms sustained speech while the player is active.
2. Client **flushes its playback buffer** (local, instant).
3. Client sends `{"type":"flush"}` on the audio socket.
4. Server calls `cancel_stream()` **on a dedicated thread** — it blocks 82–156 ms
   when synthesis is in flight, and the servers share one event loop, so calling
   it inline stalls ASR intake and pushes this sequence to ~249 ms.
5. Client cancels the in-flight `agent_loop` task.
6. The new utterance proceeds as a normal turn.

Steps 2–4 must complete within 300 ms of detection. Measured end to end: **2.9 ms**,
because the speaker is silenced by dropping the queue (step 2–3) rather than by
waiting on the cancel.

A cancelled `stream()` generator must be **drained to termination, never
`break`-ed out of** — abandoning it early wedges the engine permanently (see
`ARCHITECTURE.md` §5.7).

---

## 5. Config — `voice.yaml`

```yaml
voice:
  asr:
    url: ws://127.0.0.1:8771/asr        # Phase 2: ws://<laptop>:8771/asr
    model_arch: TINY_STREAMING          # SMALL_STREAMING too slow alongside TTS
    language: en
    keyterms: ["flappy bird", "web game"]
  tts:
    mcp_url:   http://127.0.0.1:8772/mcp
    audio_url: ws://127.0.0.1:8772/audio
    language: en-us
    voice: kokoro_af_heart              # kokoro_ / piper_ / zipvoice_ prefixes
  turn:
    silence_ms: 700                     # trailing silence that ends a turn
    barge_in_ms: 300                    # sustained speech that interrupts
  audio:
    input_device: null                  # null = system default
    output_device: null
    aec: true                           # expect module-echo-cancel loaded
  filler:
    enabled: true                       # deterministic pre-LLM acknowledgement
    phrases: ["Sure, let me look at that.", "On it.", "Let me check."]
```
