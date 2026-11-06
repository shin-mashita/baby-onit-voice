"""Speaker playback with an instant flush, for barge-in.

Audio arrives as framed PCM on the TTS data plane (VOICE_PROTOCOL §3) at the
engine's 24 kHz. Playback is the first thing cut on barge-in: dropping the local
queue silences the agent in microseconds, which is why the barge-in sequence
does that *before* asking the server to cancel synthesis -- `cancel_stream()`
blocks 82-156 ms when synthesis is in flight (ARCHITECTURE §3).
"""

from __future__ import annotations

import logging
import threading
from collections import deque

import numpy as np

log = logging.getLogger(__name__)


class AudioPlayer:
    """Queue-fed speaker output. Thread-safe; flush() is immediate."""

    def __init__(self, sample_rate: int = 24000, device: int | str | None = None):
        self.sample_rate = sample_rate
        self.device = device
        self._q: deque[np.ndarray] = deque()
        self._lock = threading.Lock()
        self._stream = None
        self._cursor = np.zeros(0, dtype=np.float32)
        self.utterance_id: int | None = None

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=1024,
            callback=self._callback,
        )
        self._stream.start()

    def _callback(self, outdata, frames, _time, status):
        if status:
            log.debug("playback status: %s", status)
        need, out = frames, []
        while need > 0:
            if len(self._cursor) == 0:
                with self._lock:
                    if not self._q:
                        break
                    self._cursor = self._q.popleft()
            take = min(need, len(self._cursor))
            out.append(self._cursor[:take])
            self._cursor = self._cursor[take:]
            need -= take
        buf = np.concatenate(out) if out else np.zeros(0, dtype=np.float32)
        if len(buf) < frames:  # underrun: pad with silence rather than glitch
            buf = np.concatenate([buf, np.zeros(frames - len(buf), dtype=np.float32)])
        outdata[:, 0] = buf

    def enqueue(self, samples: np.ndarray, utterance_id: int | None = None) -> None:
        with self._lock:
            self._q.append(np.asarray(samples, dtype=np.float32))
        if utterance_id is not None:
            self.utterance_id = utterance_id

    def flush(self) -> int:
        """Drop everything queued and stop mid-sample. Returns frames dropped."""
        with self._lock:
            dropped = sum(len(c) for c in self._q)
            self._q.clear()
        dropped += len(self._cursor)
        self._cursor = np.zeros(0, dtype=np.float32)
        return dropped

    @property
    def pending_seconds(self) -> float:
        with self._lock:
            queued = sum(len(c) for c in self._q)
        return (queued + len(self._cursor)) / self.sample_rate

    @property
    def is_playing(self) -> bool:
        return self.pending_seconds > 0

    def close(self) -> None:
        self.flush()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
