"""ASR endpoint -- `ws://<host>:8771/asr`, VOICE_PROTOCOL.md §1.

Binary frames in (little-endian float32 mono 16 kHz), JSON events out.

`Stream.add_audio()` is synchronous and does the encoding work on the calling
thread, so it runs on a per-connection worker thread. That worker owns the
Transcriber and its Stream outright -- start, reset and stop are queued to it
as commands rather than applied from the loop -- so there is no cross-thread
mutation of native state anywhere.

Moonshine's transcription callbacks fire on its own internal thread; they are
hopped back to the loop with `call_soon_threadsafe` and serialized through one
asyncio queue so JSON events leave in the order they were produced.
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
#: Matches the measured first-partial latency; smaller just burns CPU.
DEFAULT_UPDATE_INTERVAL = 0.1

#: Strength of keyterm biasing, applied at Transcriber load time.
#: 0-2 is indistinguishable from no biasing at all; 5.0 is the lowest value
#: measured to actually work; 20.0 and above makes the decoder emit a keyterm
#: repeatedly instead of transcribing. Raise with care and re-check a known
#: utterance afterwards.
KEYTERM_BOOST = 5.0
KEYTERM_BOOST_MAX = 10.0
#: ~10 s of 100 ms frames. Past this the worker is hopelessly behind and
#: dropping is better than growing without bound.
AUDIO_QUEUE_DEPTH = 100


def model_label(arch: Any, language: str) -> str:
    """e.g. ModelArch.TINY_STREAMING + "en" -> "tiny-streaming-en"."""
    return f"{arch.name.lower().replace('_', '-')}-{language}"


@dataclass
class AsrModel:
    """Resolved model, shared by every connection. Resolving downloads."""

    path: str
    arch: Any
    language: str

    @property
    def label(self) -> str:
        return model_label(self.arch, self.language)

    @classmethod
    def resolve(cls, language: str, arch_name: str) -> "AsrModel":
        import moonshine_voice as mv

        arch = getattr(mv.ModelArch, arch_name)
        path, arch = mv.get_model_for_language(language, arch)
        return cls(path=str(path), arch=arch, language=language)


class _Command:
    """Marker types queued to the worker thread alongside audio."""

    __slots__ = ()


@dataclass(slots=True)
class _Start(_Command):
    keyterms: list[str] | None
    context: str | None


@dataclass(slots=True)
class _Reset(_Command):
    pass


@dataclass(slots=True)
class _Finalize(_Command):
    """The client's VAD says the utterance ended. Only non-streaming engines care.

    Moonshine segments internally and ignores this; Qwen has no frame-level
    emission at all, so it is the only signal that a transcript should be
    produced. See VOICE_PROTOCOL.md 1.
    """


class _Stop(_Command):
    pass


class AsrSession:
    """One `/asr` connection: a websocket, a Transcriber and a worker thread."""

    def __init__(self, ws: WebSocket, model: AsrModel, *, update_interval: float,
                 keyterm_boost: float = KEYTERM_BOOST) -> None:
        self._ws = ws
        self._model = model
        self._update_interval = update_interval
        self._loop = asyncio.get_running_loop()
        self._events: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self._work: queue.Queue[Any] = queue.Queue(maxsize=AUDIO_QUEUE_DEPTH)
        self._worker: threading.Thread | None = None
        self._closing = threading.Event()
        self._t_start: float | None = None
        self._dropped = 0
        # Remembered so `reset` can rebuild the stream with the same biasing.
        self._keyterms: list[str] | None = None
        self._keyterm_boost = keyterm_boost
        self._context: str | None = None
        self.peer = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"

    @property
    def engine_label(self) -> str:
        """What is actually transcribing. Reported in `ready` and in logs."""
        return self._model.label

    # -- lifecycle ---------------------------------------------------------

    async def run(self) -> None:
        await self._ws.accept()
        # Transcriber construction loads the model (0.10 s warm) and must not
        # run on the loop.
        await asyncio.to_thread(self._build)
        self._worker = threading.Thread(target=self._run_worker, name="asr-worker", daemon=True)
        self._worker.start()
        await self._ws.send_text(
            json.dumps({"type": "ready", "model": self.engine_label, "sample_rate": SAMPLE_RATE})
        )
        log.info("asr client %s ready (model=%s)", self.peer, self.engine_label)

        sender = asyncio.create_task(self._send_loop(), name="asr-send")
        try:
            await self._recv_loop()
        finally:
            self._closing.set()
            # Unblocks a worker parked in get(); if the queue is full the
            # worker is not parked and will see _closing on its next pass.
            self._submit(_Stop())
            if self._worker is not None:
                await asyncio.to_thread(self._worker.join, 5.0)
            await self._events.put(None)
            try:
                await sender
            except asyncio.CancelledError:
                pass
            await asyncio.to_thread(self._teardown)
            if self._dropped:
                log.warning("asr client %s dropped %d audio frames", self.peer, self._dropped)
            log.info("asr client %s closed", self.peer)

    def _build(self) -> None:
        from moonshine_voice.transcriber import Transcriber

        # One Transcriber per connection: set_keyterms() lives on the
        # Transcriber, not the Stream, so a shared instance would let one
        # client's `start.keyterms` clobber another's.
        # keyterm_boost is load-time, not settable per-stream, and without it
        # set_keyterms() is inert: measured, a boost of None/1.0/2.0 leaves
        # "flappy" transcribed as "floppy" no matter what terms are supplied,
        # while 5.0 fixes it even on TINY_STREAMING. Capped deliberately -- at
        # 20.0 the decoder collapses into emitting a keyterm over and over
        # ("kubernetes kubernetes kubernetes...") and the transcript is lost.
        self._transcriber = Transcriber(
            self._model.path,
            self._model.arch,
            update_interval=self._update_interval,
            options={"keyterm_boost": str(self._keyterm_boost)},
        )
        self._stream = None

    def _teardown(self) -> None:
        if getattr(self, "_stream", None) is not None:
            try:
                self._stream.close()
            except Exception:
                log.exception("asr stream close failed")
            self._stream = None
        try:
            self._transcriber.close()
        except Exception:
            log.exception("asr transcriber close failed")

    # -- websocket -> worker ----------------------------------------------

    async def _recv_loop(self) -> None:
        while True:
            try:
                message = await self._ws.receive()
            except WebSocketDisconnect:
                return
            kind = message.get("type")
            if kind == "websocket.disconnect":
                return
            if (payload := message.get("bytes")) is not None:
                self._submit(np.frombuffer(payload, dtype="<f4"))
            elif (text := message.get("text")) is not None:
                if not await self._handle_control(text):
                    return

    async def _handle_control(self, text: str) -> bool:
        """Returns False when the client asked to close the stream."""
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            await self._emit({"type": "error", "message": "control message is not JSON"})
            return True
        kind = payload.get("type")
        if kind == "start":
            keyterms = payload.get("keyterms") or None
            self._submit(_Start(keyterms=keyterms, context=payload.get("context")))
        elif kind == "reset":
            self._submit(_Reset())
        elif kind == "finalize":
            self._submit(_Finalize())
        elif kind == "stop":
            self._submit(_Stop())
            return False
        else:
            await self._emit({"type": "error", "message": f"unknown control type {kind!r}"})
        return True

    def _submit(self, work: Any) -> None:
        try:
            self._work.put_nowait(work)
        except queue.Full:
            # Only audio is ever dropped; commands are rare enough that the
            # queue being full of audio means the worker is already in trouble.
            self._dropped += 1

    # -- worker -> websocket ----------------------------------------------

    async def _send_loop(self) -> None:
        while True:
            event = await self._events.get()
            if event is None:
                return
            try:
                await self._ws.send_text(json.dumps(event))
            except (WebSocketDisconnect, RuntimeError, ConnectionError):
                return

    async def _emit(self, event: dict[str, Any]) -> None:
        await self._events.put(event)

    def _emit_threadsafe(self, event: dict[str, Any]) -> None:
        try:
            self._loop.call_soon_threadsafe(self._events.put_nowait, event)
        except RuntimeError:
            pass

    # -- worker thread -----------------------------------------------------

    def _run_worker(self) -> None:
        while not self._closing.is_set():
            work = self._work.get()
            try:
                if isinstance(work, _Stop):
                    self._stop_stream(flush=True)
                    return
                if isinstance(work, _Finalize):
                    self._finalize()
                elif isinstance(work, _Start):
                    self._start_stream(work)
                elif isinstance(work, _Reset):
                    # `reset` drops the in-flight utterance, so the trailing
                    # `line` that stop() would otherwise flush must not escape.
                    self._stop_stream(flush=False)
                    self._start_stream(_Start(keyterms=self._keyterms, context=self._context))
                else:
                    if self._stream is None:
                        # Audio before an explicit `start`: begin anyway rather
                        # than discard the leading syllables of a turn.
                        log.info("asr client %s sent audio before start", self.peer)
                        self._start_stream(_Start(keyterms=None, context=None))
                    self._add_audio(work)
            except Exception as exc:
                log.exception("asr worker error")
                self._emit_threadsafe({"type": "error", "message": str(exc)})

    def _add_audio(self, frames: np.ndarray) -> None:
        self._stream.add_audio(frames.tolist(), sample_rate=SAMPLE_RATE)

    def _finalize(self) -> None:
        """Moonshine segments on its own, so an explicit finalize is a no-op."""
        log.debug("asr client %s finalize ignored (streaming engine)", self.peer)

    def _start_stream(self, start: _Start) -> None:
        if self._stream is not None:
            self._stop_stream(flush=False)
        self._keyterms = start.keyterms
        self._context = start.context
        if start.keyterms:
            # Measured: without this, "flappy" transcribes as "floppy".
            self._transcriber.set_keyterms(start.keyterms)
        if start.context:
            self._transcriber.set_context(start.context)
        self._stream = self._transcriber.create_stream()
        self._stream.add_listener(self._on_event)
        self._stream.start()
        self._t_start = time.monotonic()

    def _stop_stream(self, *, flush: bool) -> None:
        """Close the stream. `flush=False` discards its final `LineCompleted`.

        `stop()` runs the tail of the audio through the encoder and emits one
        last line; detaching the listener first is what makes that line
        disappear, since a stream with no listeners cannot report anything.
        """
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        if not flush:
            try:
                stream.remove_listener(self._on_event)
            except Exception:
                log.exception("asr listener detach failed")
        try:
            stream.stop()
        finally:
            stream.close()

    def _on_event(self, event: Any) -> None:
        from moonshine_voice.transcriber import Error, LineCompleted, LineTextChanged

        elapsed = round(time.monotonic() - (self._t_start or time.monotonic()), 3)
        if isinstance(event, LineTextChanged):
            self._emit_threadsafe(
                {
                    "type": "partial",
                    "line_id": event.line.line_id,
                    "text": event.line.text,
                    "t": elapsed,
                }
            )
        elif isinstance(event, LineCompleted):
            self._emit_threadsafe(
                {
                    "type": "line",
                    "line_id": event.line.line_id,
                    "text": event.line.text,
                    "t": elapsed,
                }
            )
        elif isinstance(event, Error):
            self._emit_threadsafe({"type": "error", "message": str(event.error)})


class QwenAsrSession(AsrSession):
    """`/asr` backed by Qwen3-ASR over a vLLM OpenAI-compatible endpoint.

    Qwen3-ASR is an audio encoder feeding an autoregressive LLM decoder, not a
    transducer, so there is no per-frame emission to stream: it cannot produce a
    partial before a decode pass runs, and a decode pass needs a bounded chunk.
    Rather than pretend otherwise with a 2-5 s chunk, this buffers the whole
    utterance and transcribes once when the client's VAD says the turn ended.

    What that costs, stated plainly:
      * **No `partial` events at all.** The live transcript disappears.
      * Text arrives after the utterance, not during it. Measured RTF 0.109, so
        ~0.7 s for a 6 s utterance on top of the client's silence window.
    What it buys: 0.0% WER on a real recording where TINY and SMALL both
    substituted "floppy" for "flappy", and correct casing and punctuation.

    It also runs out-of-process, which is not incidental -- vLLM needs
    torch 2.13+cu129 and a downgraded cu12 torchcodec, and co-installing that
    with the onnxruntime-gpu stack the TTS backend uses would break one of them.
    """

    #: Refuse to buffer more than this. A client that never finalizes would
    #: otherwise grow this without bound.
    MAX_UTTERANCE_S = 120.0

    #: Minimum RMS for a buffer to be worth transcribing. Qwen hallucinates
    #: confidently on near-silence -- measured, 2 s of noise at the idle floor
    #: returned "Que?" with no language set and "Okay." with language=en, so
    #: forcing a language changes only WHICH language it invents. The measured
    #: idle floor on this hardware is ~0.0025-0.0035 rms and real speech ~0.12,
    #: so this threshold sits an order of magnitude clear of both.
    MIN_SPEECH_RMS = 0.015

    def __init__(self, ws: WebSocket, model: AsrModel, *, update_interval: float,
                 keyterm_boost: float = KEYTERM_BOOST,
                 qwen_url: str = "http://127.0.0.1:8773/v1",
                 qwen_model: str = "qwen3-asr") -> None:
        super().__init__(ws, model, update_interval=update_interval,
                         keyterm_boost=keyterm_boost)
        self._qwen_url = qwen_url.rstrip("/")
        self._qwen_model = qwen_model
        self._language = model.language
        self._buf: list[np.ndarray] = []
        self._buffered = 0
        self._line_id = 0

    @property
    def engine_label(self) -> str:
        # Not self._model.label: a Moonshine arch is still resolved at startup
        # (the server shares one model object), but it is not what transcribes.
        return f"{self._qwen_model}@vllm"

    # No Moonshine model to build or tear down.
    def _build(self) -> None:
        self._transcriber = None
        self._stream = None

    def _teardown(self) -> None:
        self._buf.clear()

    def _start_stream(self, start: _Start) -> None:
        self._keyterms = start.keyterms
        self._context = start.context
        self._buf.clear()
        self._buffered = 0
        self._stream = _QWEN_ACTIVE  # sentinel: the base worker checks for None
        self._t_start = time.monotonic()
        if start.keyterms:
            log.info("asr client %s: keyterms via hotwords %s", self.peer, start.keyterms)

    def _stop_stream(self, *, flush: bool) -> None:
        if flush and self._buffered:
            self._finalize()
        self._buf.clear()
        self._buffered = 0
        self._stream = None

    def _add_audio(self, frames: np.ndarray) -> None:
        if self._buffered >= self.MAX_UTTERANCE_S * SAMPLE_RATE:
            return
        self._buf.append(frames)
        self._buffered += len(frames)

    def _finalize(self) -> None:
        """Transcribe everything buffered and emit it as one `line`."""
        if not self._buf:
            # Say so rather than returning silently: the client cannot tell an
            # empty buffer from a slow backend, so it waited out its full 4 s
            # transcript grace and reported a dropped turn with no explanation.
            self._emit_threadsafe({"type": "no_speech", "rms": 0.0, "seconds": 0.0})
            return
        audio = np.concatenate(self._buf)
        self._buf.clear()
        self._buffered = 0
        if len(audio) < SAMPLE_RATE // 10:  # <100 ms is not speech
            return
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < self.MIN_SPEECH_RMS:
            # Silence in, invented words out. Drop it rather than hand the agent
            # a hallucinated turn.
            log.info("asr client %s: dropping %.2fs at rms %.4f (below %.3f)",
                     self.peer, len(audio) / SAMPLE_RATE, rms, self.MIN_SPEECH_RMS)
            # Tell the client, rather than leaving it to time out: a silent drop
            # looks identical to a slow ASR, and the client waits its full
            # transcript grace (4 s) before giving up on a turn that was never
            # going to produce text.
            self._emit_threadsafe(
                {"type": "no_speech", "rms": round(rms, 5),
                 "seconds": round(len(audio) / SAMPLE_RATE, 2)}
            )
            return
        elapsed = round(time.monotonic() - (self._t_start or time.monotonic()), 3)
        try:
            text = self._transcribe(audio)
        except Exception as exc:  # noqa: BLE001 — surfaced to the client, not fatal
            log.exception("qwen transcription failed")
            self._emit_threadsafe({"type": "error", "message": f"qwen: {exc}"})
            return
        if not text:
            return
        self._line_id += 1
        self._emit_threadsafe(
            {"type": "line", "line_id": self._line_id, "text": text, "t": elapsed}
        )

    def _transcribe(self, audio: np.ndarray) -> str:
        """POST one utterance to /v1/audio/transcriptions. Runs on the worker thread."""
        import io
        import urllib.request
        import wave

        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes())

        # The endpoint accepts `language` and `hotwords` even though the model
        # card documents neither. `language` pins the output language, which
        # matters because auto-detect on marginal audio has been observed
        # answering in Spanish and Chinese. `hotwords` is Qwen's equivalent of
        # Moonshine's keyterms, so the client's terms are honoured after all.
        boundary = "----speechserver"
        fields = [("model", self._qwen_model), ("language", self._language)]
        if self._keyterms:
            fields.append(("hotwords", ", ".join(self._keyterms)))
        head = "".join(
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n"
            for k, v in fields
        )
        body = (
            head
            + f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f'filename="utterance.wav"\r\nContent-Type: audio/wav\r\n\r\n'
        ).encode() + buf.getvalue() + f"\r\n--{boundary}--\r\n".encode()

        req = urllib.request.Request(
            f"{self._qwen_url}/audio/transcriptions",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            return (json.load(r).get("text") or "").strip()


#: Sentinel standing in for Moonshine's Stream, so the base worker's
#: `if self._stream is None` start-on-first-audio logic still applies.
_QWEN_ACTIVE = object()


def build_asr_app(
    model: AsrModel,
    *,
    update_interval: float = DEFAULT_UPDATE_INTERVAL,
    keyterm_boost: float = KEYTERM_BOOST,
    qwen_url: str | None = None,
):
    """`/asr` app. Passing qwen_url selects Qwen3-ASR instead of Moonshine."""
    from starlette.applications import Starlette
    from starlette.routing import WebSocketRoute

    async def endpoint(ws: WebSocket) -> None:
        if qwen_url:
            session: AsrSession = QwenAsrSession(
                ws, model, update_interval=update_interval,
                keyterm_boost=keyterm_boost, qwen_url=qwen_url,
            )
        else:
            session = AsrSession(ws, model, update_interval=update_interval,
                                 keyterm_boost=keyterm_boost)
        try:
            await session.run()
        except WebSocketDisconnect:
            pass

    return Starlette(routes=[WebSocketRoute("/asr", endpoint, name="asr")])
