"""The conversational loop: ASR in, agent turn, TTS out, barge-in across all three.

Sequencing here is the whole feature, so the order of operations matters:

* A turn is dispatched on **VAD trailing silence**, never on an ASR `line`
  event -- one utterance emits several lines (ARCHITECTURE §5.1).
* The filler is spoken by this driver the instant a turn ends, **before** the
  first LLM token exists. Asking the model to "always speak first" costs a whole
  round trip and is only as reliable as prompt adherence (ARCHITECTURE §5.3).
* Barge-in flushes the local speaker first, then tells the server to cancel,
  then cancels the agent task -- cheapest and most audible action first.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import struct
import time
from pathlib import Path

import numpy as np

from .config import load as load_config
from .player import AudioPlayer
from .turn import TurnDetector

log = logging.getLogger(__name__)

_HEADER = struct.Struct("<IIIB3x")  # utterance_id, sample_rate, n_samples, is_final


class VoiceDriver:
    """Wires the audio I/O to a baby_onit.VoiceSession."""

    def __init__(self, session, cfg: dict | None = None, mic: bool = True):
        self.cfg = cfg or load_config()
        self.session = session  # baby_onit.VoiceSession
        self.use_mic = mic
        self.loop: asyncio.AbstractEventLoop | None = None
        self.player = AudioPlayer(device=self.cfg["audio"]["output_device"])
        self.turns = TurnDetector(
            silence_ms=self.cfg["turn"]["silence_ms"],
            barge_in_ms=self.cfg["turn"]["barge_in_ms"],
            on_turn_end=self._on_turn_end,
            on_barge_in=self._on_barge_in,
        )
        self.capture = None
        self._asr = None
        self._audio = None
        self.transcript: list[str] = []  # completed ASR lines for the current turn
        self.partial = ""
        # Mic frames held while the agent speaks; see _on_mic_frame. Sized to
        # barge_in_ms plus a margin, so the interrupting words survive the gate.
        self._parked: list[np.ndarray] = []
        self._park_max = max(16, 2 * self.turns.barge_in_frames)
        # How long to wait for ASR text after the VAD says the turn ended, and
        # how long to let a trailing line settle. Sized off MEDIUM_STREAMING's
        # post-reset first-partial latency (~3.7s measured); a smaller model
        # needs far less but costs nothing by waiting.
        t = self.cfg["turn"]
        self.transcript_grace_s = t.get("transcript_grace_ms", 4000) / 1000
        self.line_settle_s = t.get("line_settle_ms", 400) / 1000
        self._pending_turn: asyncio.Queue[str] = asyncio.Queue()
        self._tasks: list[asyncio.Task] = []
        # Bumped by every barge-in. Work captured under an older generation is
        # discarded rather than replayed -- see _on_barge_in.
        self._generation = 0
        self._resolver: asyncio.Task | None = None
        # Append-only record of what the conversation actually did. A voice
        # session leaves no scrollback worth reading -- the terminal shows log
        # lines, not the dialogue -- so without this an "it went wrong" report
        # cannot be traced to the turn that caused it.
        tl = self.cfg.get("transcript_log", "~/baby-sandbox/voice_turns.jsonl")
        self.transcript_log = Path(tl).expanduser() if tl else None

    # ---------------------------------------------------------------- lifecycle

    async def __aenter__(self) -> VoiceDriver:
        import websockets

        self.loop = asyncio.get_running_loop()
        self._asr = await websockets.connect(self.cfg["asr"]["url"], max_size=None)
        await self._asr.send(
            json.dumps({"type": "start", "keyterms": self.cfg["asr"]["keyterms"], "context": None})
        )
        # The server builds a Transcriber per connection and only sends `ready`
        # once the model is loaded -- a cold resolve can download, so this is the
        # go-signal, not the socket opening. Generous timeout for that first run.
        await self._await_ready(timeout=self.cfg["asr"].get("ready_timeout_s", 60))
        self._audio = await websockets.connect(self.cfg["tts"]["audio_url"], max_size=None)
        self.player.start()
        self._tasks = [
            asyncio.create_task(self._pump_asr(), name="asr-rx"),
            asyncio.create_task(self._pump_audio(), name="audio-rx"),
            asyncio.create_task(self._pump_turns(), name="turns"),
        ]
        if self.use_mic:
            from .capture import MicCapture

            self.capture = MicCapture(
                self.loop,
                self._on_mic_frame,
                device=self.cfg["audio"]["input_device"],
                vad_threshold=self.cfg["turn"]["vad_threshold"],
            )
            self.capture.start()
        return self

    async def _await_ready(self, timeout: float) -> None:
        """Block until the ASR server reports its model is loaded."""
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            left = deadline - asyncio.get_running_loop().time()
            if left <= 0:
                raise TimeoutError(f"ASR never sent `ready` within {timeout}s")
            msg = await asyncio.wait_for(self._asr.recv(), left)
            if isinstance(msg, bytes):
                continue
            ev = json.loads(msg)
            if ev.get("type") == "ready":
                log.info("asr ready: model=%s rate=%s", ev.get("model"), ev.get("sample_rate"))
                return
            if ev.get("type") == "error":
                raise RuntimeError(f"ASR error before ready: {ev.get('message')}")

    async def __aexit__(self, *exc) -> None:
        for t in self._tasks:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.capture:
            self.capture.close()
        self.player.close()
        for ws in (self._asr, self._audio):
            if ws is not None:
                await ws.close()

    # ------------------------------------------------------------------- audio

    def _on_mic_frame(self, frame: np.ndarray, is_speech: bool) -> None:
        """One 512-sample mic frame, already VAD-scored. Runs on the event loop."""
        speaking = self.player.is_playing
        self.turns.agent_speaking = speaking
        self.turns.feed(is_speech)
        if self._asr is None:
            return
        if speaking:
            # ASR and TTS contend hard for memory bandwidth: concurrent ASR
            # ingest pushes TTS time-to-first-audio from ~500 ms to ~2300 ms
            # (measured, 4.3x). Nothing needs transcription while the agent
            # talks -- barge-in runs off the local VAD, not off ASR -- so the
            # mic is parked here and the cores go to synthesis. Frames are kept
            # in a short ring so a barge-in does not lose its opening syllables.
            self._parked.append(frame)
            if len(self._parked) > self._park_max:
                self._parked.pop(0)
            return
        if self._parked:
            parked, self._parked = self._parked, []
            # Unparking leaves a wall-clock hole in the ASR stream: frames from
            # before playback, then nothing, then this burst. `add_audio` just
            # appends -- the streaming encoder has no notion of elapsed time --
            # so it would splice across the hole and could emit one line merging
            # pre-playback speech with the interruption. `reset` drops the
            # in-flight utterance and restarts the clock, keeping keyterms, so
            # the transcript resumes cleanly from the barge-in onward.
            asyncio.create_task(self._reset_then_send(b"".join(f.tobytes() for f in parked)))
        # float32 LE PCM, exactly as VOICE_PROTOCOL §1 specifies.
        asyncio.create_task(self._send_audio(frame.tobytes()))

    async def _reset_then_send(self, payload: bytes) -> None:
        """Clear the ASR stream across an unpark gap, then replay the parked burst."""
        self.transcript.clear()
        self.partial = ""
        try:
            await self._asr.send(json.dumps({"type": "reset"}))
        except Exception as e:  # noqa: BLE001 — a failed reset only risks a garbled line
            log.warning("asr reset failed: %s", e)
        await self._send_audio(payload)

    async def _send_audio(self, payload: bytes) -> None:
        try:
            await self._asr.send(payload)
        except Exception as e:  # noqa: BLE001 — a dropped socket is not fatal mid-turn
            log.warning("asr send failed: %s", e)

    async def _pump_asr(self) -> None:
        """Keep the live transcript current. Never decides turn boundaries."""
        async for msg in self._asr:
            if isinstance(msg, bytes):
                continue
            ev = json.loads(msg)
            kind = ev.get("type")
            if kind == "partial":
                self.partial = ev.get("text", "")
            elif kind == "line":
                self.partial = ""
                if ev.get("text"):
                    self.transcript.append(ev["text"])
            elif kind == "error":
                log.error("asr: %s", ev.get("message"))

    async def _pump_audio(self) -> None:
        """Unframe PCM from the TTS data plane into the speaker queue."""
        async for msg in self._audio:
            if not isinstance(msg, (bytes, bytearray)) or len(msg) < _HEADER.size:
                continue
            uid, rate, n, is_final = _HEADER.unpack_from(msg, 0)
            samples = np.frombuffer(msg, dtype="<f4", count=n, offset=_HEADER.size)
            if rate != self.player.sample_rate:
                self.player.sample_rate = rate  # engine owns the rate; trust it
            self.player.enqueue(samples, utterance_id=uid)
            log.debug("audio uid=%d n=%d final=%d", uid, n, is_final)

    # -------------------------------------------------------------------- turns

    def _on_turn_end(self) -> None:
        """VAD says the human stopped. Promote the transcript into a turn.

        At most one resolver runs at a time: two of them race on the same
        `transcript`/`partial` and can dispatch overlapping text for a single
        utterance, which reads as the agent answering twice.

        The VAD decides the boundary, but ASR supplies the words, and ASR is
        slower than the VAD -- measurably so on the larger models (first partial
        ~1.3 s, and ~3.7 s after a reset on MEDIUM_STREAMING). Reading the
        transcript synchronously here drops any utterance whose text has not
        landed yet: the user speaks and is silently ignored. So wait for it.
        """
        if self._resolver is not None and not self._resolver.done():
            self._resolver.cancel()
        self._resolver = asyncio.create_task(self._resolve_turn(self._generation))

    async def _resolve_turn(self, gen: int) -> None:
        """Wait a bounded time for ASR to catch up, then dispatch the turn.

        `gen` is the generation this utterance belongs to. A barge-in arriving
        while we wait bumps it, and the text we were assembling then describes
        speech the user has already superseded, so it is dropped rather than
        queued behind the interruption.
        """
        deadline = asyncio.get_running_loop().time() + self.transcript_grace_s
        while not (self.transcript or self.partial.strip()):
            if asyncio.get_running_loop().time() > deadline:
                log.warning(
                    "turn dropped: no transcript within %.1fs of end-of-speech "
                    "(ASR may be lagging; consider a smaller --model-arch)",
                    self.transcript_grace_s,
                )
                return
            await asyncio.sleep(0.05)
        # Give a trailing line a moment to replace a mid-word partial, so the
        # turn is not dispatched on "can you create me" when "...a flappy bird
        # app" is 200 ms behind it. With only a partial in hand, wait the whole
        # window: a partial is mid-utterance by definition, and a single quiet
        # 50 ms tick means nothing when the model's update interval is 100 ms.
        # With a completed line already in hand, a brief stability check is
        # enough to catch a second line still in flight.
        loop = asyncio.get_running_loop()
        settle = loop.time() + self.line_settle_s
        while loop.time() < settle:
            before = (len(self.transcript), self.partial)
            await asyncio.sleep(0.05)
            changed = (len(self.transcript), self.partial) != before
            if self.transcript and not changed:
                break
        text = " ".join(self.transcript).strip() or self.partial.strip()
        self.transcript.clear()
        self.partial = ""
        if gen != self._generation:
            log.info("discarding superseded utterance (gen %d != %d): %r",
                     gen, self._generation, text[:60])
            self._record("discarded_utterance", gen_at_capture=gen, text=text)
            return
        if text:
            self._pending_turn.put_nowait(text)

    def _on_barge_in(self) -> None:
        """Sustained speech while the agent talks. Silence it, cheapest first.

        Cancelling the in-flight agent task is not enough on its own. Anything
        already queued behind it -- a turn waiting in _pending_turn, a
        _resolve_turn still waiting on ASR, an utterance the TTS service is
        mid-synthesis on -- outlives the cancel and surfaces later as a reply to
        something the user has already moved past. Interrupting therefore
        invalidates the whole generation, and everything below checks it.
        """
        self._generation += 1
        dropped = self.player.flush()
        stale = self._drain_pending()
        log.info(
            "barge-in (gen %d): dropped %d samples, discarded %d queued turn(s)",
            self._generation, dropped, stale,
        )
        self._record("barge_in", dropped_samples=dropped, discarded_turns=stale)
        if self._audio is not None:
            asyncio.create_task(self._send_flush())
        self.session.barge_in()

    def _record(self, event: str, **fields) -> None:
        """Append one conversation event. Never raises -- logging must not break a turn."""
        if self.transcript_log is None:
            return
        try:
            self.transcript_log.parent.mkdir(parents=True, exist_ok=True)
            rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "gen": self._generation,
                   "event": event, **fields}
            with self.transcript_log.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            log.debug("transcript log write failed: %s", e)

    def _drain_pending(self) -> int:
        """Discard queued turns. They predate the interruption, so they are stale."""
        n = 0
        while True:
            try:
                self._pending_turn.get_nowait()
                n += 1
            except asyncio.QueueEmpty:
                return n

    async def _send_flush(self) -> None:
        try:
            await self._audio.send(json.dumps({"type": "flush"}))
        except Exception as e:  # noqa: BLE001
            log.warning("flush send failed: %s", e)

    async def _pump_turns(self) -> None:
        """Serialize turns: one agent answer at a time."""
        while True:
            text = await self._pending_turn.get()
            await self.handle_turn(text)

    async def handle_turn(self, text: str) -> str | None:
        """Speak a filler, run the agent, speak the answer. None if superseded."""
        gen = self._generation
        log.info("turn (gen %d): %r", gen, text)
        self._record("turn", text=text)
        # Clear anything the TTS service is still synthesizing from a previous
        # turn before adding to it. `speak` enqueues server-side, so without
        # this a new filler plays *behind* a stale answer and the two arrive in
        # the wrong order.
        await self._clear_stale_speech()
        await self.speak_filler()
        answer = await self.session.ask(text)
        if answer is None or gen != self._generation:
            log.info("turn (gen %d) superseded; answer withheld", gen)
            self._record("answer_withheld", gen_at_start=gen,
                         reason="cancelled" if answer is None else "superseded")
            return None
        # The agent's closing prose is the actual answer, and the agent_loop
        # returns it as text rather than speaking it -- only its explicit
        # `speak` calls make sound. Without this the agent narrates what it is
        # about to do and then goes quiet exactly when it has something to say.
        self._record("answer", text=answer)
        if answer.strip():
            await self.speak(answer)
        return answer

    async def _clear_stale_speech(self) -> None:
        """Drop audio still queued from an earlier turn, locally and server-side."""
        if not self.player.is_playing and "stop_speaking" not in _tools(self.session):
            return
        self.player.flush()
        try:
            await _dispatch(self.session, "stop_speaking", {})
        except Exception as e:  # noqa: BLE001 — best effort; a stale tail is not fatal
            log.debug("stop_speaking before new turn failed: %s", e)

    async def speak(self, text: str) -> None:
        """Say something through the TTS control plane, best-effort."""
        if "speak" not in _tools(self.session):
            log.warning("no `speak` tool registered; answer will not be spoken")
            return
        try:
            await _dispatch(self.session, "speak", {"text": text})
        except Exception as e:  # noqa: BLE001 — a mute answer beats a dead turn
            log.warning("speak failed: %s", e)

    async def speak_filler(self) -> None:
        """Deterministic acknowledgement, before the first LLM token exists."""
        f = self.cfg["filler"]
        if not f.get("enabled") or "speak" not in _tools(self.session):
            return
        await self.speak(random.choice(f["phrases"]))


def _tools(session) -> dict:
    import baby_onit

    return baby_onit.TOOLS


async def _dispatch(session, name: str, args: dict) -> str:
    import baby_onit

    return await baby_onit.dispatch(name, args)
