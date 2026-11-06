"""TTS data plane -- `ws://<host>:8772/audio`, VOICE_PROTOCOL.md §3.

Binary frames only in the server->client direction: a 16-byte little-endian
header followed by float32 PCM. The only client->server message is
`{"type": "flush"}`, the barge-in.

`AudioHub` is the fan-out. The synthesis worker thread calls `publish()`; the
hub hops that onto the event loop and appends to each client's send queue, so
no socket write ever happens on the synthesis thread and no synthesis ever
happens on the loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
from typing import Callable

import numpy as np
from starlette.websockets import WebSocket, WebSocketDisconnect, WebSocketState

from .backends import AudioChunk

log = logging.getLogger(__name__)

#: offset 0 uint32 utterance_id | 4 uint32 sample_rate | 8 uint32 n_samples
#: offset 12 uint8 is_final | 13..15 reserved, zero. 16 bytes total.
HEADER = struct.Struct("<IIIB3x")
assert HEADER.size == 16

#: Roughly 0.5 s of audio per chunk, so this is ~1 minute of slack before a
#: wedged client starts losing audio. Dropping beats unbounded memory growth.
DEFAULT_CLIENT_QUEUE = 128


def frame_chunk(utterance_id: int, chunk: AudioChunk) -> bytes:
    """Pack one chunk into the §3 wire frame."""
    pcm = np.asarray(chunk.samples, dtype="<f4")
    header = HEADER.pack(utterance_id, chunk.sample_rate, pcm.size, 1 if chunk.is_final else 0)
    return header + pcm.tobytes()


def unpack_frame(frame: bytes) -> tuple[int, int, int, bool, np.ndarray]:
    """Inverse of `frame_chunk`. Used by the client and by selftest."""
    utterance_id, sample_rate, n_samples, is_final = HEADER.unpack_from(frame, 0)
    samples = np.frombuffer(frame, dtype="<f4", count=n_samples, offset=HEADER.size)
    return utterance_id, sample_rate, n_samples, bool(is_final), samples


class _Client:
    __slots__ = ("ws", "queue", "peer")

    def __init__(self, ws: WebSocket, maxsize: int) -> None:
        self.ws = ws
        self.queue: asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=maxsize)
        self.peer = f"{ws.client.host}:{ws.client.port}" if ws.client else "?"


class AudioHub:
    """Fan-out of framed PCM to every connected `/audio` client."""

    def __init__(self, *, client_queue: int = DEFAULT_CLIENT_QUEUE) -> None:
        self._clients: set[_Client] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._client_queue = client_queue
        self._on_flush: Callable[[], object] | None = None
        # Frames tagged with a generation older than this were synthesized
        # before the last barge-in and must not reach a speaker. Without it a
        # chunk already in flight on the loop can land after drop_pending().
        self._min_generation = 0

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def set_flush_handler(self, handler: Callable[[], object]) -> None:
        """Wire `{"type":"flush"}` to `TtsService.stop_speaking`.

        Late-bound because the hub has to exist before the service that owns it.
        """
        self._on_flush = handler

    @property
    def client_count(self) -> int:
        return len(self._clients)

    # -- producer side (synthesis worker thread) ---------------------------

    def publish(self, frame: bytes, generation: int) -> None:
        """Queue one framed chunk for every client. Safe from any thread."""
        loop = self._loop
        if loop is None:
            log.warning("audio frame dropped: hub has no event loop yet")
            return
        try:
            loop.call_soon_threadsafe(self._publish_now, frame, generation)
        except RuntimeError:
            # Loop closed underneath us during shutdown.
            pass

    def _publish_now(self, frame: bytes, generation: int) -> None:
        if generation < self._min_generation:
            return
        for client in list(self._clients):
            try:
                client.queue.put_nowait(frame)
            except asyncio.QueueFull:
                log.warning("audio client %s is not draining; dropping frame", client.peer)

    # -- barge-in ----------------------------------------------------------

    def drop_pending(self, generation: int) -> int:
        """Discard every queued frame and refuse anything older than `generation`.

        Safe from any thread. Returns the number of frames dropped when called
        on the loop thread, 0 otherwise (the work is deferred).
        """
        loop = self._loop
        if loop is None:
            return 0
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            return self._drop_now(generation)
        loop.call_soon_threadsafe(self._drop_now, generation)
        return 0

    def _drop_now(self, generation: int) -> int:
        self._min_generation = max(self._min_generation, generation)
        dropped = 0
        for client in self._clients:
            while True:
                try:
                    client.queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                dropped += 1
        if dropped:
            log.info("barge-in dropped %d queued audio frames", dropped)
        return dropped

    # -- websocket endpoint ------------------------------------------------

    async def endpoint(self, ws: WebSocket) -> None:
        await ws.accept()
        if self._loop is None:
            self.bind_loop(asyncio.get_running_loop())
        client = _Client(ws, self._client_queue)
        self._clients.add(client)
        log.info("audio client %s connected (%d total)", client.peer, len(self._clients))
        sender = asyncio.create_task(self._send_loop(client), name="audio-send")
        try:
            await self._recv_loop(client)
        finally:
            self._clients.discard(client)
            sender.cancel()
            try:
                await sender
            except asyncio.CancelledError:
                pass
            log.info("audio client %s disconnected (%d left)", client.peer, len(self._clients))

    async def _send_loop(self, client: _Client) -> None:
        while True:
            frame = await client.queue.get()
            if frame is None:
                return
            try:
                await client.ws.send_bytes(frame)
            except (WebSocketDisconnect, RuntimeError, ConnectionError):
                # Client vanished mid-utterance; the recv loop will clean up.
                return

    async def _recv_loop(self, client: _Client) -> None:
        while True:
            try:
                message = await client.ws.receive_text()
            except WebSocketDisconnect:
                return
            except RuntimeError:
                # Starlette raises this when the socket is already closed.
                return
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                log.warning("audio client %s sent non-JSON control message", client.peer)
                continue
            if payload.get("type") == "flush":
                log.info("barge-in from audio client %s", client.peer)
                if self._on_flush is not None:
                    self._on_flush()
            else:
                log.warning("audio client %s sent unknown control %r", client.peer, payload)

    async def close(self) -> None:
        for client in list(self._clients):
            if client.ws.client_state is WebSocketState.CONNECTED:
                try:
                    await client.ws.close()
                except Exception:
                    pass
        self._clients.clear()
