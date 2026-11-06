# `speech_server`

The laptop half of the voice agent: streaming ASR and TTS, served over the wire
contract in [`docs/VOICE_PROTOCOL.md`](../docs/VOICE_PROTOCOL.md). CPU-only,
one process, one event loop.

```
                       ┌──────────────────────── speech_server ───────────────────────┐
  voice_client         │                                                              │
      │ f32 16 kHz     │  :8771  ws /asr      Transcriber (TINY_STREAMING)            │
      ├───────────────▶│                      one per connection, own worker thread   │
      │◀───────────────│                      JSON: ready / partial / line / error    │
      │ JSON events    │                                                              │
      │                │  :8772  http /mcp    speak / stop_speaking / is_speaking     │
      ├───────────────▶│            │                                                 │
      │                │            ▼                                                 │
      │                │        TtsService ── utterance queue ── synthesis thread     │
      │                │            │                                                 │
      │◀───────────────│  :8772  ws /audio ◀── AudioHub ◀────────┘  framed f32 24 kHz │
      │ 16-byte hdr    │                                                              │
      └ PCM            └──────────────────────────────────────────────────────────────┘
```

## Run

```bash
source .venv/bin/activate
python -m speech_server --prewarm
```

| Endpoint | Default | What |
|---|---|---|
| `ws://<host>:8771/asr` | 8771 | Audio in, JSON transcription events out (§1) |
| `http://<host>:8772/mcp` | 8772 | MCP control plane, streamable HTTP (§2) |
| `ws://<host>:8772/audio` | 8772 | Framed PCM out, `{"type":"flush"}` in (§3) |

The MCP server and the audio websocket deliberately share one port: they are
two planes of one service, and `voice.yaml` configures them as a pair.

### Options

```
--host 0.0.0.0            bind address for both ports
--asr-port 8771           --tts-port 8772
--language en             ASR language
--model-arch TINY_STREAMING
                          SMALL_STREAMING and up are too slow alongside TTS
                          (RTF 0.64 vs 0.23 measured)
--tts-language en-us      --voice kokoro_af_heart
--tts-backend moonshine   moonshine | kokoro (kokoro is a stub, see backends.py)
--update-interval 0.1     ASR partial cadence
--prewarm                 see below
--log-level info
```

`mv.list_tts_voices("en-us")` lists the 62 available voices, including all 20
`kokoro_*`.

### `--prewarm`

The ASR model and the TTS voice are always loaded during startup, never on a
live utterance. `--prewarm` goes further and runs one throwaway synthesis and
one throwaway transcription so that every lazy native initialization is paid
before the first client connects. Use it in production; the cost is about 0.7 s.

The first load of an *unseen* voice downloads its assets, which took 103 s when
measured. That download happens during startup either way, but prewarm during
provisioning rather than discovering it on a user's first turn.

## Shutdown

SIGINT or SIGTERM stops both uvicorn servers, closes every socket, drains the
synthesis queue and calls `tts.close()` and `Transcriber.close()`. Signal
handling is this module's rather than uvicorn's, because uvicorn's own handler
would only stop one of the two servers.

## Verify

With the server already running:

```bash
python -m speech_server.selftest
```

33 checks across six groups: ASR streaming, the `speak` latency budget,
barge-in from both planes, `reset` semantics, and mid-utterance disconnects.
It feeds `sample.wav` at 1x realtime and synthesizes one through the server if
that file is missing.

It is written to run against a **long-lived, shared** server, which is how this
one is actually used. Two consequences worth keeping if you edit it: frames are
matched to the utterance under test by `utterance_id` (ids are monotonic for the
server's whole life, so a reconnecting client can otherwise be handed tail frames
from an utterance queued before the test existed), and latency is *measured and
printed* rather than asserted -- the only timing assertions are the protocol's
own budgets (`speak` < 50 ms, barge-in < 300 ms).

## Layout

| File | Role |
|---|---|
| `asr_ws.py` | §1 endpoint. One `Transcriber` and one worker thread per connection |
| `tts_mcp.py` | §2 MCP tools plus `TtsService`, the shared queue and synthesis worker |
| `tts_audio_ws.py` | §3 data plane: `AudioHub`, frame packing, barge-in fan-out |
| `backends.py` | `TtsBackend` protocol, `MoonshineTts`, `KokoroTts` stub |
| `__main__.py` | Argument parsing, logging, both servers, signal handling |
| `selftest.py` | End-to-end checks against a running server |

## Threading

Two rules explain most of the design.

**`Stream.add_audio()` and `TextToSpeech.stream()` synthesize on the calling
thread.** Neither may touch the event loop. ASR gets one worker thread per
connection which owns its `Transcriber` outright; TTS gets one shared synthesis
worker, because the engine synthesizes one thing at a time regardless.

**The library's `stream()` busy-spins.** It polls `next_chunk()` with no sleep,
so for the ~380 ms before the first sentence is ready it spins at full speed on
the synthesis thread, starving every other Python thread. `MoonshineTts.stream()`
reimplements that loop over the same public API with a 2 ms sleep; the library's
own `_ChunkPump` sleeps 5 ms in exactly this spot. Verified chunk-for-chunk
identical to the library's output across six texts.

**`cancel_stream()` is cheap; only the loop is precious.** The call itself costs
0.0 ms. It *appears* to block 82-156 ms only when measured against the
busy-spinning consumer above -- that was GIL starvation, not the call. It is
still issued from a dedicated canceller thread, so that a native call can never
stall ASR intake, and the worker waits for a pending cancel before starting the
next utterance (a cancel landing after the next `stream()` begins would cancel
the wrong utterance). What actually silences the speaker is dropping the queued
frames, which happens on the loop and is immediate: `stop_speaking` returns in
~3 ms.

## Latency, measured

| Path | Measured |
|---|---|
| `speak` MCP round trip (20 warm calls) | 1.8 / 2.8 / 3.6 ms (min/med/p95) |
| `stop_speaking` round trip | ~3 ms |
| Time-to-first-audio, ASR idle | 305-418 ms |
| Time-to-first-audio, **ASR ingesting concurrently** | **1.3-1.8 s** |
| Filler TTFA after a turn + 700 ms silence | 418 / 517 / 551 ms (min/med/max) |
| Transport overhead (server first-chunk vs client) | 10-30 ms |

**ASR and TTS contend, and it is not fixable in this layer.** One synthesis uses
**15.7 of 20 cores** (the process runs 49 threads), and so does ASR, so running
them together oversubscribes the machine and inflates TTFA ~4x. Measured three
ways: same process 1.47 s, separate processes 1.60 s, separate processes pinned
to disjoint core sets 1.55 s -- so it is neither the GIL nor core scheduling but
memory-bandwidth and cache contention between two ONNX workloads. §3's
"ASR 0.23 + TTS 0.23 = 0.46 RTF, comfortable" assumed roughly one core each.

The number that governs the filler budget is the third row, not the second: in
the real sequence the user has stopped speaking before the agent starts, so the
overlap is brief. Budget **~550 ms**, not 300 ms.

## Deviations from the frozen protocol

Two, both documented in the task report and neither a wire change:

1. **`utterance_id` is offset, not raw.** §3 says the id comes "straight from
   Moonshine's `TtsChunk`". Measured, the engine restarts it at 1 on *every*
   `stream()` call, so raw ids repeat across `speak` calls and §2's returned
   `utterance_id` cannot identify anything. `TtsService` adds a per-`speak`
   base so ids stay globally monotonic; the per-sentence increment and the
   `is_final` grouping are unchanged. The base is reserved using the engine's
   own `split_say_utterances()`, so the id `speak` returns is the one the first
   chunk actually carries.

2. **`speak` can return `status: "rejected"`.** §2 only pins the happy path.
   With 32 utterances already queued, `speak` returns
   `{"utterance_id": 0, "status": "rejected", "reason": "queue_full"}` rather
   than blocking, which would break the < 50 ms budget. Empty text returns
   `status: "idle"`.
