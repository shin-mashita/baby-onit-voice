"""Microphone capture: mic -> VAD -> turn detection -> ASR websocket.

The mic runs at 16 kHz mono in 512-sample blocks, which is exactly one Silero
frame, so no rebuffering is needed between capture and VAD.

sounddevice delivers blocks on a PortAudio thread, not the event loop, so frames
cross over via call_soon_threadsafe. Nothing blocking may run in the callback.
"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .vad import FRAME_SAMPLES, SAMPLE_RATE, SileroVad

log = logging.getLogger(__name__)


class MicCapture:
    """Owns the input stream and fans each frame out to the VAD and the socket."""

    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        on_frame,  # called on the loop with (np.ndarray, is_speech)
        device: int | str | None = None,
        vad_threshold: float = 0.5,
    ):
        self.loop = loop
        self.on_frame = on_frame
        self.device = device
        self.vad = SileroVad(threshold=vad_threshold)
        self._stream = None
        self.muted = False
        self.frames_seen = 0

    def start(self) -> None:
        import sounddevice as sd

        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.device,
            blocksize=FRAME_SAMPLES,
            callback=self._callback,
        )
        self._stream.start()
        log.info("mic open: %d Hz, %d-sample blocks", SAMPLE_RATE, FRAME_SAMPLES)

    def _callback(self, indata, _frames, _time, status):
        if status:
            log.debug("capture status: %s", status)
        if self.muted:
            return
        frame = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
        if len(frame) != FRAME_SAMPLES:
            return  # a short final block is not worth rebuffering for
        # The VAD runs here, on the audio thread: 0.08 ms/frame against a 32 ms
        # budget, and it keeps barge-in latency off the event loop's queue.
        speech = self.vad.is_speech(frame)
        self.frames_seen += 1
        self.loop.call_soon_threadsafe(self.on_frame, frame, speech)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
