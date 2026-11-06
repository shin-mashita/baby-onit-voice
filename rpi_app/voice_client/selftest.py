"""Offline regression tests for the voice client. `python -m voice_client.selftest`

No microphone, no speech server, no LLM. Every case here encodes a bug that was
real at some point, so a failure means a regression rather than a flaky
environment. The live paths are covered by `speech_server/selftest.py` and by
driving `python -m voice_client` by hand.
"""

from __future__ import annotations

import asyncio
import sys

import numpy as np

from .turn import FRAME_MS, TurnDetector
from .vad import FRAME_SAMPLES, SileroVad

PASS, FAIL = 0, 0


def check(ok: bool, label: str) -> None:
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  [PASS] {label}")
    else:
        FAIL += 1
        print(f"  [FAIL] {label}")


def frames(ms: float) -> int:
    return max(1, round(ms / FRAME_MS))


def test_turn_detector() -> None:
    print("\n[1] turn detection: ASR lines are not turn boundaries")

    # The motivating bug: one 6.6s utterance produced two ASR `line` events,
    # the first cut mid-question. Turn boundaries come from silence instead.
    ends = []
    d = TurnDetector(silence_ms=700, on_turn_end=lambda: ends.append(1))
    for _ in range(frames(2500)):
        d.feed(True)
    for _ in range(frames(400)):
        d.feed(False)  # internal pause -> ASR splits a line here
    for _ in range(frames(2500)):
        d.feed(True)
    for _ in range(frames(900)):
        d.feed(False)
    check(len(ends) == 1, f"a 400ms internal pause does not split a turn (got {len(ends)})")

    ends.clear()
    d = TurnDetector(silence_ms=700, on_turn_end=lambda: ends.append(1))
    for _ in range(2):
        for _ in range(frames(1500)):
            d.feed(True)
        for _ in range(frames(1200)):
            d.feed(False)
    check(len(ends) == 2, f"a 1200ms gap does split two turns (got {len(ends)})")

    ends.clear()
    d = TurnDetector(silence_ms=700, on_turn_end=lambda: ends.append(1))
    for _ in range(frames(5000)):
        d.feed(False)
    check(not ends, "silence alone never starts a turn")


def test_barge_in() -> None:
    print("\n[2] barge-in needs sustained speech, not a blip")

    hits = []
    d = TurnDetector(barge_in_ms=300, on_barge_in=lambda: hits.append(1))
    d.agent_speaking = True
    for _ in range(frames(100)):
        d.feed(True)  # speaker bleed through imperfect echo cancellation
    d.feed(False)
    check(not hits, "a 100ms blip does not interrupt the agent")

    for _ in range(frames(400)):
        d.feed(True)
    check(len(hits) == 1, f"400ms of sustained speech does interrupt (got {len(hits)})")

    hits.clear()
    d = TurnDetector(barge_in_ms=300, on_barge_in=lambda: hits.append(1))
    d.agent_speaking = True
    for _ in range(frames(3000)):
        d.feed(True)
    check(len(hits) == 1, f"barge-in fires once, not per frame (got {len(hits)})")


def test_vad() -> None:
    print("\n[3] VAD separates speech from silence")

    vad = SileroVad()
    silence = np.zeros(FRAME_SAMPLES * 60, dtype=np.float32)
    p = np.array([vad.probability(silence[i : i + FRAME_SAMPLES])
                  for i in range(0, len(silence) - FRAME_SAMPLES + 1, FRAME_SAMPLES)])
    check(p.max() < 0.2, f"digital silence scores near zero (max {p.max():.3f})")

    # A frame of the wrong length must raise rather than quietly mis-score: the
    # model needs 512 new samples plus 64 of carried context, and feeding a bare
    # 512 with no context returns ~0.001 for everything, which reads as "no
    # speech" rather than as an error.
    vad.reset()
    try:
        vad.probability(np.zeros(256, dtype=np.float32))
        check(False, "a wrong-sized frame raises")
    except ValueError:
        check(True, "a wrong-sized frame raises")


def test_transcript_grace() -> None:
    print("\n[4] a turn survives ASR arriving after the VAD boundary")

    from .driver import VoiceDriver

    class Stub:
        def ask(self, *a, **k): ...
        def barge_in(self): ...

    cfg = {
        "asr": {"url": "", "keyterms": []},
        "tts": {"mcp_url": "", "audio_url": ""},
        "turn": {"silence_ms": 700, "barge_in_ms": 300, "vad_threshold": 0.5,
                 "transcript_grace_ms": 4000, "line_settle_ms": 200},
        "audio": {"input_device": None, "output_device": None, "aec": True},
        "filler": {"enabled": False, "phrases": []},
    }

    async def run() -> None:
        d = VoiceDriver(Stub(), cfg, mic=False)

        # The VAD finds the boundary before ASR has the words; reading the
        # transcript synchronously dropped the utterance entirely.
        d.transcript, d.partial = [], ""
        d._on_turn_end()

        async def late() -> None:
            await asyncio.sleep(1.0)
            d.transcript.append("set an alarm at seven a m")

        asyncio.create_task(late())
        text = await asyncio.wait_for(d._pending_turn.get(), 6)
        check(text == "set an alarm at seven a m", f"late transcript still dispatches ({text!r})")

        d.transcript, d.partial = [], "can you create me"
        d._on_turn_end()

        async def trailing() -> None:
            await asyncio.sleep(0.1)
            d.transcript.append("can you create me a flappy bird app")
            d.partial = ""

        asyncio.create_task(trailing())
        text = await asyncio.wait_for(d._pending_turn.get(), 6)
        check("flappy bird app" in text, f"a trailing line replaces a mid-word partial ({text!r})")

        d.transcript, d.partial = [], ""
        d.transcript_grace_s = 0.4
        d._on_turn_end()
        try:
            got = await asyncio.wait_for(d._pending_turn.get(), 1.5)
            check(False, f"silence invented a turn ({got!r})")
        except asyncio.TimeoutError:
            check(True, "genuine silence is dropped, not invented into a turn")

    asyncio.run(run())


def test_ordering() -> None:
    """Superseded work must be discarded, not replayed. This is the "jumbling" bug."""
    print("\n[5] barge-in discards superseded work instead of replaying it")

    from .driver import VoiceDriver

    answered: list[str] = []

    class Stub:
        def __init__(self): self._t = None
        async def ask(self, text, stream_cb=None):
            answered.append(text)
            self._t = asyncio.create_task(asyncio.sleep(5))
            try:
                await self._t
                return f"answer to {text}"
            except asyncio.CancelledError:
                return None
        def barge_in(self):
            if self._t and not self._t.done():
                self._t.cancel()

    cfg = {
        "asr": {"url": "", "keyterms": []},
        "tts": {"mcp_url": "", "audio_url": ""},
        "turn": {"silence_ms": 700, "barge_in_ms": 300, "vad_threshold": 0.5,
                 "transcript_grace_ms": 2000, "line_settle_ms": 100},
        "audio": {"input_device": None, "output_device": None, "aec": True},
        "filler": {"enabled": False, "phrases": []},
    }

    async def run() -> None:
        d = VoiceDriver(Stub(), cfg, mic=False)

        # A turn already waiting in the queue when the user interrupts is stale:
        # answering it means replying to something they have moved past.
        d._pending_turn.put_nowait("stale utterance one")
        d._pending_turn.put_nowait("stale utterance two")
        before = d._generation
        d._on_barge_in()
        check(d._generation == before + 1, "barge-in bumps the generation")
        check(d._pending_turn.empty(), "barge-in drains queued turns")

        # A resolver mid-wait when the interruption lands must drop its text.
        d.transcript, d.partial = [], ""
        d._on_turn_end()
        await asyncio.sleep(0.05)
        d._on_barge_in()              # supersede it while it waits for ASR
        d.transcript.append("text from the superseded utterance")
        await asyncio.sleep(0.4)
        check(d._pending_turn.empty(), "a superseded resolver does not queue its text")

        # And an answer that completes after an interruption must stay silent.
        d.transcript, d.partial = [], ""
        spoken: list[str] = []
        d.speak = lambda t: _noop(spoken, t)
        d._clear_stale_speech = _anoop
        turn = asyncio.create_task(d.handle_turn("interrupted question"))
        await asyncio.sleep(0.1)
        d._on_barge_in()
        result = await asyncio.wait_for(turn, 3)
        check(result is None, f"a superseded turn returns None (got {result!r})")
        check(not spoken, f"a superseded turn speaks nothing (spoke {spoken})")

    async def _anoop() -> None: ...
    async def _noop(acc, t) -> None: acc.append(t)

    asyncio.run(run())


def test_audio_mute_after_barge_in() -> None:
    """Audio arriving after a barge-in must be dropped, not queued.

    The player flush empties the queue, but frames already in flight refill it
    and the agent audibly resumes. On GPU the server delivers a whole reply at
    ~17x realtime, so this race is the common case rather than a corner.
    """
    print("\n[6] barge-in mutes inbound audio until the next deliberate speak")

    import struct

    import numpy as np

    from .driver import _HEADER, VoiceDriver

    class Stub:
        async def ask(self, *a, **k): return "done"
        def barge_in(self): ...

    cfg = {
        "asr": {"url": "", "keyterms": []},
        "tts": {"mcp_url": "", "audio_url": ""},
        "turn": {"silence_ms": 700, "barge_in_ms": 300, "vad_threshold": 0.5,
                 "transcript_grace_ms": 2000, "line_settle_ms": 100},
        "audio": {"input_device": None, "output_device": None, "aec": True},
        "filler": {"enabled": False, "phrases": []},
        "transcript_log": None,
    }

    async def run() -> None:
        d = VoiceDriver(Stub(), cfg, mic=False)
        frame = _HEADER.pack(1, 24000, 480, 0) + np.zeros(480, dtype="<f4").tobytes()

        def feed(msg: bytes) -> None:
            """What _pump_audio does with one frame, minus the socket."""
            if d._audio_muted:
                return
            uid, rate, n, final = _HEADER.unpack_from(msg, 0)
            d.player.enqueue(np.frombuffer(msg, dtype="<f4", count=n, offset=_HEADER.size), uid)

        feed(frame)
        check(d.player.pending_seconds > 0, "audio queues normally before a barge-in")

        d._on_barge_in()
        check(d.player.pending_seconds == 0, "barge-in empties the player")
        check(d._audio_muted, "barge-in sets the audio mute")

        feed(frame)  # a frame that was already in flight
        check(d.player.pending_seconds == 0,
              "a frame arriving after barge-in is dropped, not queued")

        d._audio_muted = False  # what speak() does
        feed(frame)
        check(d.player.pending_seconds > 0, "a deliberate new utterance un-mutes")

    asyncio.run(run())


def test_hesitation_endpointing() -> None:
    """A thinking pause must not end the turn."""
    print("\n[7] a thinking pause extends the silence window")

    from .turn import ends_in_hesitation

    for text, expect in (
        ("Umam, I think.", True),          # what Qwen returns for "uhmm I think"
        ("so", True),
        ("I need", True),
        ("I want a good Flappy Bird app.", False),
        ("What is the weather today?", False),
    ):
        got = ends_in_hesitation(text)
        check(got == expect, f"{text!r} -> {'pause' if got else 'complete'}")

    # The window actually widens, and only while the tail looks unfinished.
    live = {"text": "umm I think"}
    ends = []
    d = TurnDetector(silence_ms=700, hesitation_extra_ms=500,
                     on_turn_end=lambda: ends.append(1),
                     peek_text=lambda: live["text"])
    for _ in range(frames(500)):
        d.feed(True)
    for _ in range(frames(800)):          # longer than 700, shorter than 1200
        d.feed(False)
    check(not ends, "800ms of silence does not end a turn that trails off")
    live["text"] = "umm I think I want a flappy bird app"
    for _ in range(frames(500)):          # now past 1200ms total
        d.feed(False)
    check(len(ends) == 1, f"it ends once the tail is complete (got {len(ends)})")


def test_continuation_merge() -> None:
    """A withdrawn turn carries its words into the next one."""
    print("\n[8] speech resuming just after dispatch withdraws and merges")

    from .driver import VoiceDriver

    class Stub:
        def __init__(self): self.barges = 0
        async def ask(self, *a, **k): return "done"
        def barge_in(self): self.barges += 1

    cfg = {
        "asr": {"url": "", "keyterms": []},
        "tts": {"mcp_url": "", "audio_url": ""},
        "turn": {"silence_ms": 900, "barge_in_ms": 300, "vad_threshold": 0.5,
                 "transcript_grace_ms": 2000, "line_settle_ms": 100,
                 "continuation_ms": 700, "filler_delay_ms": 250,
                 "hesitation_extra_ms": 500},
        "audio": {"input_device": None, "output_device": None, "aec": True},
        "filler": {"enabled": False, "phrases": []},
        "transcript_log": None,
    }

    async def run() -> None:
        stub = Stub()
        d = VoiceDriver(stub, cfg, mic=False)

        d.transcript, d.partial = ["umm I think"], ""
        d._on_turn_end()
        first = await asyncio.wait_for(d._pending_turn.get(), 4)
        check(first == "umm I think", f"the premature turn is dispatched ({first!r})")
        check(d._carry_window_open(), "the withdrawal window is open right after dispatch")

        d._maybe_withdraw()
        check(d._carry == "umm I think", f"its text is carried ({d._carry!r})")
        check(stub.barges == 1, "the in-flight turn is cancelled")

        d.transcript, d.partial = ["I want a flappy bird app"], ""
        d._on_turn_end()
        merged = await asyncio.wait_for(d._pending_turn.get(), 4)
        check(merged == "umm I think I want a flappy bird app",
              f"the next turn is the merged sentence ({merged!r})")
        check(d._carry == "", "the carry is consumed, not repeated")

        # And a genuine new utterance long after must NOT merge.
        d._dispatched_at = 0.0
        check(not d._carry_window_open(), "the window closes once it lapses")

    asyncio.run(run())


def test_speech_over_playback_reaches_asr() -> None:
    """Speech over the agent must be forwarded, not parked into a 0.58s ring."""
    print("\n[9] talking over the agent forwards audio instead of discarding it")

    import numpy as np

    from .driver import VoiceDriver

    class Stub:
        async def ask(self, *a, **k): return "done"
        def barge_in(self): ...

    cfg = {
        "asr": {"url": "", "keyterms": []},
        "tts": {"mcp_url": "", "audio_url": ""},
        "turn": {"silence_ms": 900, "barge_in_ms": 300, "vad_threshold": 0.5,
                 "transcript_grace_ms": 2000, "line_settle_ms": 100,
                 "continuation_ms": 700, "filler_delay_ms": 250,
                 "hesitation_extra_ms": 500, "park_ring_s": 3.0},
        "audio": {"input_device": None, "output_device": None, "aec": True},
        "filler": {"enabled": False, "phrases": []},
        "transcript_log": None,
    }

    async def run() -> None:
        d = VoiceDriver(Stub(), cfg, mic=False)
        check(d._park_max * FRAME_MS / 1000 >= 2.5,
              f"the park ring holds >=2.5s (got {d._park_max * FRAME_MS / 1000:.2f}s)")

        sent: list[int] = []
        d._send_audio = lambda payload: _count(sent, payload)
        d._reset_then_send = lambda payload: _count(sent, payload)
        d._asr = object()
        frame = np.zeros(512, dtype=np.float32)

        # The agent is talking and the human is silent: park, send nothing.
        d.player.enqueue(np.zeros(24000, dtype=np.float32))
        for _ in range(20):
            d._on_mic_frame(frame, False)
        check(not sent, f"agent-only audio is parked, not sent (sent {len(sent)})")
        check(len(d._parked) == 20, f"and is retained ({len(d._parked)} frames)")

        # The human starts talking over it: every frame from here must go.
        for _ in range(30):
            d._on_mic_frame(frame, True)
        await asyncio.sleep(0.05)
        check(len(sent) >= 30,
              f"speech over playback is forwarded ({len(sent)} sends for 30 frames)")

    def _count(acc, payload):
        acc.append(len(payload))
        async def _noop(): return None
        return _noop()

    asyncio.run(run())


def test_frame_header() -> None:
    print("\n[10] audio frame header matches the protocol")

    from .driver import _HEADER

    check(_HEADER.size == 16, f"header is 16 bytes (got {_HEADER.size})")
    packed = _HEADER.pack(7, 24000, 512, 1)
    uid, rate, n, final = _HEADER.unpack_from(packed, 0)
    check((uid, rate, n, final) == (7, 24000, 512, 1), "header round-trips")


def main() -> int:
    for t in (test_turn_detector, test_barge_in, test_vad,
              test_transcript_grace, test_ordering, test_audio_mute_after_barge_in,
              test_hesitation_endpointing, test_continuation_merge,
              test_speech_over_playback_reaches_asr, test_frame_header):
        t()
    print(f"\n{PASS}/{PASS + FAIL} checks passed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
