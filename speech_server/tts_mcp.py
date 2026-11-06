"""TTS control plane -- MCP at `http://<host>:8772/mcp`, VOICE_PROTOCOL.md §2.

`TtsService` is the shared state between this control plane and the §3 data
plane: one backend, one utterance queue, one synthesis worker thread. The whole
point of the split is that `speak` returns in milliseconds while audio keeps
leaving over the websocket, so **nothing here may await audio**.

Concurrency contract:

* `speak`, `stop_speaking`, `is_speaking` run on the asyncio loop and only ever
  touch the queue and a few counters under `_lock`. None of them block.
* The synthesis worker owns the backend's `stream()` and holds no lock while
  iterating it, so a `stop_speaking` concurrent with synthesis cannot deadlock.
* Cancellation is a generation counter, not a flag. Each queued utterance is
  tagged with the generation current at enqueue time; `stop_speaking` bumps it.
  The worker drops any utterance whose tag is stale, both at dequeue and on
  every chunk, which closes the window where a barge-in lands between
  `queue.get()` and the first `cancel_stream()`-observable moment.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

from starlette.applications import Starlette
from starlette.routing import WebSocketRoute

from .backends import TtsBackend
from .tts_audio_ws import AudioHub, frame_chunk

log = logging.getLogger(__name__)

#: Deep enough to absorb a burst of narration, shallow enough that a runaway
#: agent gets told "rejected" instead of queueing minutes of speech.
DEFAULT_QUEUE_DEPTH = 32

#: Upper bound on how long the synthesis worker waits for a pending cancel.
#: The cancel normally completes in well under a millisecond, so this is pure
#: insurance against a wedged canceller thread muting the server for good.
CANCEL_WAIT_TIMEOUT = 2.0


@dataclass
class _Utterance:
    generation: int
    base_id: int
    n_utterances: int
    text: str
    #: monotonic() at enqueue, so the worker can report time-to-first-audio.
    queued_at: float = 0.0


_SHUTDOWN = _Utterance(-1, 0, 0, "")


class TtsService:
    """Utterance queue plus the single synthesis worker that drains it."""

    def __init__(
        self,
        backend: TtsBackend,
        hub: AudioHub,
        *,
        queue_depth: int = DEFAULT_QUEUE_DEPTH,
    ) -> None:
        self._backend = backend
        self._hub = hub
        self._queue: queue.Queue[_Utterance] = queue.Queue(maxsize=queue_depth)
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        # cancel_stream() itself costs 0.0 ms, but it is still a native call
        # and both uvicorn servers share one event loop, so a dedicated thread
        # makes it structurally impossible for it to stall ASR intake. What
        # actually silences the speaker is dropping the queued frames, and that
        # is immediate and happens on the loop.
        self._cancel_requested = threading.Event()
        # Clear while a cancel is pending or in flight. The synthesis worker
        # waits on this before starting an utterance, because a cancel_stream()
        # that lands after the next stream() has begun cancels the WRONG
        # utterance -- and "barge in, then speak a filler" is the single most
        # common sequence this server sees.
        self._cancel_done = threading.Event()
        self._cancel_done.set()
        self._canceller: threading.Thread | None = None
        self._closing = False
        # True from stop_speaking() until the next speak(). Makes is_speaking()
        # answer from intent rather than from the engine's lagging state.
        self._cancelling = False
        self._generation = 0
        self._speaking = False
        #: Next utterance_id to hand out. See `speak` for why we offset the
        #: engine's own ids instead of forwarding them raw.
        self._next_id = 1
        self._max_emitted = 0

    def start(self) -> None:
        if self._worker is not None:
            return
        self._canceller = threading.Thread(target=self._run_canceller, name="tts-cancel", daemon=True)
        self._canceller.start()
        self._worker = threading.Thread(target=self._run, name="tts-synth", daemon=True)
        self._worker.start()

    # -- control plane -----------------------------------------------------

    def speak(self, text: str) -> dict[str, Any]:
        """Enqueue an utterance and return. Never synthesizes on this thread."""
        text = (text or "").strip()
        if not text:
            return {"utterance_id": 0, "status": "idle"}

        # Splitting with the engine's own splitter tells us how many
        # utterance_ids this text will consume, so the id we return is the one
        # the first chunk will actually carry.
        #
        # Caveat on how far that is guaranteed: split_say_utterances() has a
        # docstring, and the *splitting rules* it describes are a contract
        # (terminators, "Dr. Smith", "J. R. R. Tolkien", per-language
        # abbreviations). What is NOT documented anywhere is that stream()
        # assigns utterance_id in lockstep with that same split -- the docstring
        # only ties the splitter to say(). That equivalence was verified
        # empirically, 7/7 cases including the abbreviation and
        # no-terminator edge cases, and nothing in the library promises it will
        # hold. The _max_emitted shift in _run() exists precisely because this
        # prediction is inferred rather than guaranteed: if a future version
        # splits differently the ids stay collision-free, and only the id
        # returned by a queued speak() goes stale.
        try:
            n_utterances = max(1, len(self._backend.split_utterances(text)))
        except Exception:
            log.exception("utterance split failed; assuming one")
            n_utterances = 1

        with self._lock:
            base_id = self._next_id
            item = _Utterance(self._generation, base_id, n_utterances, text, time.monotonic())
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                log.warning("speak rejected: %d utterances already queued", self._queue.qsize())
                return {"utterance_id": 0, "status": "rejected", "reason": "queue_full"}
            self._next_id += n_utterances
            self._cancelling = False

        log.info("speak queued id=%d n=%d chars=%d", base_id, n_utterances, len(text))
        return {"utterance_id": base_id, "status": "speaking"}

    def stop_speaking(self) -> dict[str, Any]:
        """Barge-in. Drops the queue, cancels synthesis, clears the send queues."""
        with self._lock:
            self._generation += 1
            generation = self._generation
            dropped = self._drain_locked()
            was_speaking = self._speaking or self._backend.is_talking
            self._cancelling = True
        # Hand the blocking native cancel to the canceller thread, then silence
        # the speaker here and now by discarding everything already queued.
        self._cancel_done.clear()
        self._cancel_requested.set()
        self._hub.drop_pending(generation)
        stopped = was_speaking or dropped > 0
        log.info("stop_speaking: speaking=%s dropped=%d", was_speaking, dropped)
        return {"stopped": stopped}

    def is_speaking(self) -> dict[str, Any]:
        """Report *intent*, not the engine's lagging view.

        Post-condition callers depend on: once `stop_speaking` has returned,
        this reports False. The engine keeps reporting `is_streaming` until the
        worker has finished tearing the cancelled generation down, so reading it
        directly hands a barge-in path a stale True and makes it branch as
        though the agent were still talking.
        """
        with self._lock:
            if self._cancelling:
                # Stopped and nothing queued since: we are silent by decision,
                # whatever the engine still thinks.
                return {"speaking": False, "queued": self._queue.qsize()}
            speaking = self._speaking
            queued = self._queue.qsize()
        return {"speaking": speaking or self._backend.is_talking, "queued": queued}

    def _drain_locked(self) -> int:
        dropped = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return dropped
            dropped += 1

    # -- synthesis worker --------------------------------------------------

    def _run_canceller(self) -> None:
        while True:
            self._cancel_requested.wait()
            if self._closing:
                return
            # Clear before cancelling, so a barge-in arriving during the call
            # is not swallowed.
            self._cancel_requested.clear()
            try:
                self._backend.stop()
            except Exception:
                log.exception("backend cancel failed")
            # A barge-in that arrived during the call gets its own pass, so
            # releasing the worker now would let it start under that cancel.
            if not self._cancel_requested.is_set():
                self._cancel_done.set()

    def _run(self) -> None:
        log.info("synthesis worker started (backend=%s)", self._backend.name)
        while True:
            item = self._queue.get()
            if item is _SHUTDOWN:
                break
            # Let any in-flight cancel_stream() finish before touching the
            # engine, or it would swallow this utterance instead of the one it
            # was aimed at. Bounded so a wedged canceller cannot mute us.
            if not self._cancel_done.wait(timeout=CANCEL_WAIT_TIMEOUT):
                log.warning("proceeding with utterance %d: cancel still pending", item.base_id)
            with self._lock:
                if item.generation != self._generation:
                    log.debug("dropping utterance %d: cancelled while queued", item.base_id)
                    continue
                if item.base_id <= self._max_emitted:
                    # The previous utterance produced more sentences than its
                    # split predicted, so its ids ran into this one's block.
                    # Shift rather than collide; the id handed to the caller is
                    # then one short, which is the lesser evil.
                    log.warning(
                        "utterance id %d already used; shifting to %d",
                        item.base_id,
                        self._max_emitted + 1,
                    )
                    item.base_id = self._max_emitted + 1
                self._speaking = True
            try:
                self._synthesize(item)
            except Exception:
                log.exception("synthesis failed for utterance %d", item.base_id)
            finally:
                with self._lock:
                    self._speaking = False
        log.info("synthesis worker stopped")

    def _synthesize(self, item: _Utterance) -> None:
        chunks = self._backend.stream(item.text)
        n_frames = 0
        truncated = False
        try:
            for chunk in chunks:
                if truncated:
                    # Keep draining. Abandoning a cancelled generator leaves
                    # the engine's single in-flight generation half-torn-down:
                    # measured, the NEXT stream() then yields zero chunks and
                    # is_streaming sticks at True forever. Draining to the
                    # generator's own terminal state costs 0.0 ms because
                    # cancel_stream() has already made it return immediately.
                    continue
                with self._lock:
                    if item.generation != self._generation:
                        truncated = True
                    else:
                        # The engine restarts utterance_id at 1 for every
                        # stream(), so it indexes sentences within this reply,
                        # not globally.
                        utterance_id = item.base_id - 1 + chunk.utterance_id
                        if utterance_id > self._max_emitted:
                            self._max_emitted = utterance_id
                if truncated:
                    # Idempotent, and needed because a barge-in can bump the
                    # generation a moment before its own cancel_stream() lands.
                    self._backend.stop()
                    log.info("utterance %d truncated after %d frames", item.base_id, n_frames)
                    continue
                if n_frames == 0:
                    log.info(
                        "utterance %d first audio %.0f ms after enqueue",
                        item.base_id,
                        (time.monotonic() - item.queued_at) * 1000,
                    )
                self._hub.publish(frame_chunk(utterance_id, chunk), item.generation)
                n_frames += 1
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()

    # -- shutdown ----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            self._generation += 1
            self._drain_locked()
        self._backend.stop()
        self._cancel_done.set()
        if self._worker is not None:
            self._queue.put(_SHUTDOWN)
            self._worker.join(timeout=5.0)
            self._worker = None
        if self._canceller is not None:
            self._closing = True
            self._cancel_requested.set()
            self._canceller.join(timeout=5.0)
            self._canceller = None
        self._backend.close()


def build_tts_app(service: TtsService, hub: AudioHub) -> Starlette:
    """MCP control plane and the §3 audio websocket on one port."""
    from mcp.server.mcpserver import MCPServer

    mcp = MCPServer(
        name="speech-server-tts",
        instructions=(
            "Speak out loud to the user. `speak` returns immediately while the "
            "audio is still streaming, so keep working after calling it."
        ),
    )

    @mcp.tool(
        description=(
            "Speak text aloud to the user. Returns immediately -- the audio "
            "streams out of band -- so continue working after calling this."
        )
    )
    async def speak(text: str) -> dict[str, Any]:
        return service.speak(text)

    @mcp.tool(description="Stop speaking immediately and discard anything queued.")
    async def stop_speaking() -> dict[str, Any]:
        return service.stop_speaking()

    @mcp.tool(description="Whether audio is currently playing, and how much is queued.")
    async def is_speaking() -> dict[str, Any]:
        return service.is_speaking()

    app = mcp.streamable_http_app(streamable_http_path="/mcp", host="0.0.0.0")
    app.routes.append(WebSocketRoute("/audio", hub.endpoint, name="audio"))
    return app
