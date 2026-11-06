# voice_client

Microphone, VAD, turn-taking, playback and barge-in. Everything here is light —
in the deployed topology this runs on the Raspberry Pi while the speech models
run in `speech_server` on the laptop. See `docs/ARCHITECTURE.md` for the
reasoning and `docs/VOICE_PROTOCOL.md` for the wire format.

## Run

```bash
python -m speech_server --prewarm        # on the laptop, first
python -m voice_client                   # listens on the default mic
python -m voice_client --no-mic --say "what's the weather today"   # one typed turn
```

`voice.yaml` (see `docs/VOICE_PROTOCOL.md` §5) sets the endpoints. Phase 2 is
the same command with the hostnames pointed at the laptop.

## Modules

| File | Role |
|---|---|
| `vad.py` | Silero VAD over onnxruntime. 0.087 ms/frame. |
| `turn.py` | Turn + barge-in state machine, from VAD frames only. |
| `capture.py` | Mic → VAD → ASR websocket. 16 kHz, 512-sample blocks. |
| `player.py` | PCM queue → speaker, with an instant `flush()`. |
| `driver.py` | The conversational loop; wires the above to a `VoiceSession`. |
| `config.py` | `voice.yaml` loading over documented defaults. |

## Three things worth knowing before changing this

**1. Turn boundaries come from the VAD, never from ASR events.** Moonshine's
`line` event is a *line* boundary: one 6.6 s utterance emitted two of them, the
first cut mid-question ("...Can you create me?"). Dispatching an agent turn per
line fires several nonsense turns per sentence. `turn.py` fuses lines and waits
for `silence_ms` of quiet instead. There is a regression test for exactly this.

**2. The Silero ONNX contract is not self-describing.** Every call takes **576
samples**: the 512 new samples prefixed by the **last 64 of the previous frame**.
Feed a bare 512 and it returns ~0.001 for everything — which reads as "no
speech", not as an error, so it fails silently. `vad.py` handles this; don't
"simplify" it.

**3. Barge-in order is deliberate: flush locally, then tell the server, then
cancel the agent.** The local flush silences the speaker in microseconds, while
the server's `cancel_stream()` blocks 82–156 ms when synthesis is in flight.
Doing it the other way round leaves the agent audibly talking over the user for
a sixth of a second.

## Echo cancellation

Mic and speaker on one device means the agent hears itself and the barge-in VAD
interrupts it, every utterance. Moonshine ships barge-in disabled by default for
this reason. Three layers, all needed:

```bash
pactl load-module module-echo-cancel aec_method=webrtc   # already available here
```

plus the `is_playing` gate in `driver._on_mic_frame`, plus `barge_in_ms` of
*sustained* speech. Use headphones while tuning.
