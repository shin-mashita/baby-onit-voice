"""Silero VAD over onnxruntime — frame-level speech probability.

Moonshine segments server-side but exposes no frame-level speech signal, and
barge-in needs one locally: the decision to cut the agent off has to be made on
the device holding the microphone, within a few audio frames, with no network
round trip. Silero is the right size for that (2 MB, ~0.08 ms/frame here, ARM64
wheels for the Pi) and is MIT licensed.

Deliberately not using the `silero-vad` pip package: it pulls torch, which is
~2 GB and pointless on a Pi when onnxruntime already runs the same graph.

The ONNX contract is not self-describing and cost an afternoon to pin down, so:
every call takes **576 samples** -- the 512 new samples of this frame prefixed
by the **last 64 samples of the previous frame**. Feeding a bare 512 returns
~0.001 for everything, which looks exactly like "no speech" rather than like an
error. The (2, 1, 128) LSTM state is returned by each call and fed into the next.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512  # what the caller supplies per frame (32 ms at 16 kHz)
CONTEXT_SAMPLES = 64  # prepended from the previous frame; see module docstring

_MODEL_URL = "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
_CACHE = Path.home() / ".cache" / "voice-agent" / "silero_vad.onnx"


def model_path(cache: Path | None = None) -> Path:
    """Path to the ONNX file, downloading it once if absent."""
    p = cache or _CACHE
    if not p.is_file():
        p.parent.mkdir(parents=True, exist_ok=True)
        log.info("downloading Silero VAD to %s", p)
        tmp = p.with_suffix(".part")
        urllib.request.urlretrieve(_MODEL_URL, tmp)
        tmp.replace(p)  # atomic: a half-download never looks like a cached model
    return p


class SileroVad:
    """Stateful speech detector. Feed exactly FRAME_SAMPLES at a time.

    Both the LSTM state and the 64-sample context carry across frames, so one
    instance tracks one audio stream; call reset() between unrelated streams.
    """

    def __init__(self, path: Path | None = None, threshold: float = 0.5):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # One 576-sample frame on one thread beats thread-pool overhead, and
        # leaves the cores for whatever else shares the box.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3  # the graph logs benign If/LSTM warnings
        self._s = ort.InferenceSession(
            str(path or model_path()), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.threshold = threshold
        self._sr = np.array(SAMPLE_RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Clear the LSTM state and the carried context between streams."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._ctx = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)

    def probability(self, frame: np.ndarray) -> float:
        """Speech probability for one FRAME_SAMPLES frame of float32 PCM."""
        frame = np.asarray(frame, dtype=np.float32)
        if len(frame) != FRAME_SAMPLES:
            raise ValueError(f"need exactly {FRAME_SAMPLES} samples, got {len(frame)}")
        inp = np.concatenate([self._ctx, frame]).reshape(1, -1)
        out, self._state = self._s.run(None, {"input": inp, "state": self._state, "sr": self._sr})
        self._ctx = frame[-CONTEXT_SAMPLES:]
        return float(out[0][0])

    def is_speech(self, frame: np.ndarray) -> bool:
        """Whether this frame is speech at the configured threshold."""
        return self.probability(frame) >= self.threshold
