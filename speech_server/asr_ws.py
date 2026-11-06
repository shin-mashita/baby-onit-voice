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

    # -- lifecycle ---------------------------------------------------------

    async def run(self) -> None:
        await self._ws.accept()
        # Transcriber construction loads the model (0.10 s warm) and must not
        # run on the loop.
        await asyncio.to_thread(self._build)
        self._worker = threading.Thread(target=self._run_worker, name="asr-worker", daemon=True)
        self._worker.start()
        await self._ws.send_text(
            json.dumps({"type": "ready", "model": self._model.label, "sample_rate": SAMPLE_RATE})
        )
        log.info("asr client %s ready (model=%s)", self.peer, self._model.label)

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
                if isinstance(work, _Start):
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
                    self._stream.add_audio(work.tolist(), sample_rate=SAMPLE_RATE)
            except Exception as exc:
                log.exception("asr worker error")
                self._emit_threadsafe({"type": "error", "message": str(exc)})

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


def build_asr_app(
    model: AsrModel,
    *,
    update_interval: float = DEFAULT_UPDATE_INTERVAL,
    keyterm_boost: float = KEYTERM_BOOST,
):
    from starlette.applications import Starlette
    from starlette.routing import WebSocketRoute

    async def endpoint(ws: WebSocket) -> None:
        session = AsrSession(ws, model, update_interval=update_interval,
                             keyterm_boost=keyterm_boost)
        try:
            await session.run()
        except WebSocketDisconnect:
            pass

    return Starlette(routes=[WebSocketRoute("/asr", endpoint, name="asr")])
