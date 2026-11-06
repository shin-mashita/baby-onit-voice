"""TTS backends behind one Protocol.

Every backend here synthesizes **on the calling thread**, so callers must drive
them from a worker thread. Calling `stream()` or `synthesize()` from the asyncio
loop stalls the sockets for the duration of the synthesis.

`stop()` is the one method that is safe to call from any thread, because it is
the barge-in path and has to land in well under the 300 ms budget of
VOICE_PROTOCOL.md §4.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Protocol, Sequence, runtime_checkable

log = logging.getLogger(__name__)

#: How long to sleep when the engine has no chunk ready yet. Chunks arrive
#: roughly every 115 ms of wall time, so this costs at most a couple of ms of
#: latency while removing essentially all of the GIL pressure.
CHUNK_POLL_INTERVAL = 0.002

DEFAULT_LANGUAGE = "en-us"
DEFAULT_VOICE = "kokoro_af_heart"

#: Abbreviations whose trailing period does not end a sentence. Without these a
#: naive split turns "Dr. Smith" into two utterances and Kokoro reads the pause.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "etc", "e.g",
    "i.e", "approx", "dept", "est", "fig", "no", "inc", "ltd", "co", "corp",
    "am", "pm", "a.m", "p.m",
}
_SENT_END = re.compile(r"(?<=[.!?\u3002\uff01\uff1f\u061f\u0964])[\s]+|(?<=[\u3002\uff01\uff1f\u061f\u0964])(?=\S)")
_CLAUSE = re.compile(r"(?<=[,;:\u3001\uff0c])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split on terminators, rejoining where the period was an abbreviation."""
    parts = [p for p in _SENT_END.split(text.strip()) if p and p.strip()]
    out: list[str] = []
    for part in parts:
        if out and (
            # "Dr." / "e.g." -- the previous fragment ended on an abbreviation
            _prev_is_abbrev(out[-1])
            # "J. R. R. Tolkien" -- a single-letter initial is never a sentence
            or re.search(r"\b[A-Za-z]\.$", out[-1])
        ):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return [p.strip() for p in out if p.strip()]


def _prev_is_abbrev(prev: str) -> bool:
    tail = prev.rstrip()
    if not tail.endswith("."):
        return False
    word = tail[:-1].split()[-1].lower() if tail[:-1].split() else ""
    return word in _ABBREV


def _split_long(sentence: str, limit: int) -> list[str]:
    """Break an over-long sentence on clause boundaries, then on whitespace.

    A `create()` call cannot be interrupted, so an unbroken sentence holds off
    barge-in for its entire synthesis; this bounds that wait.
    """
    if len(sentence) <= limit:
        return [sentence]
    out, buf = [], ""
    for clause in _CLAUSE.split(sentence):
        cand = f"{buf} {clause}".strip() if buf else clause
        if len(cand) <= limit:
            buf = cand
            continue
        if buf:
            out.append(buf)
        buf = clause
    if buf:
        out.append(buf)
    # Still too long: a clause with no internal punctuation. Fall back to words.
    final: list[str] = []
    for piece in out:
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit) or limit
            final.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            final.append(piece)
    return final or [sentence]


@dataclass
class _Chunk:
    """AudioChunk for backends whose engine does not supply one."""

    samples: Sequence[float]
    sample_rate: int
    text: str
    utterance_id: int
    is_final: bool


class AudioChunk(Protocol):
    """One piece of synthesized audio.

    Structural, not a base class: Moonshine's own `TtsChunk` already satisfies
    it, so the Moonshine backend can yield native chunks without a copy, while
    another backend is free to yield anything with these fields.
    """

    samples: Sequence[float]
    sample_rate: int
    text: str
    utterance_id: int
    is_final: bool


@runtime_checkable
class TtsBackend(Protocol):
    """What `TtsService` needs from a speech synthesizer."""

    #: Short label for logs and the MCP `ready` payload.
    name: str

    def load(self) -> "TtsBackend":
        """Fetch assets and initialize. Blocks; call during startup, not a turn."""
        ...

    def synthesize(self, text: str) -> tuple[list[float], int]:
        """One-shot synthesis. Returns `(samples, sample_rate)`."""
        ...

    def stream(self, text: str | Iterable[str] | None = None) -> Iterator[AudioChunk]:
        """Iterate chunks as they are synthesized, on the calling thread."""
        ...

    def split_utterances(self, text: str) -> list[str]:
        """Split text the way the engine will when it assigns `utterance_id`.

        Needed because `utterance_id` counts sentences, so the number of ids a
        `speak()` will consume is only knowable by splitting the text the same
        way the engine does. See `TtsService` for why that matters.
        """
        ...

    def stop(self) -> None:
        """Barge-in: drop queued text and abandon the reply in progress."""
        ...

    @property
    def is_talking(self) -> bool:
        """Whether a reply is part-synthesized."""
        ...

    def close(self) -> None:
        ...


class MoonshineTts:
    """Moonshine Voice TTS with a Kokoro voice. CPU-only, ~0.23 RTF."""

    name = "moonshine"

    def __init__(self, language: str = DEFAULT_LANGUAGE, voice: str = DEFAULT_VOICE) -> None:
        self.language = language
        self.voice = voice
        self._tts = None
        # Serializes load/close against stop(), which the loop thread calls.
        self._lifecycle = threading.Lock()

    def load(self) -> "MoonshineTts":
        from moonshine_voice.tts import TextToSpeech

        with self._lifecycle:
            if self._tts is None:
                # First load of an unseen voice downloads assets (~100 s measured).
                # --prewarm exists so that never happens on a live utterance.
                self._tts = TextToSpeech().language(self.language).voice(self.voice).load()
        return self

    @property
    def _engine(self):
        tts = self._tts
        if tts is None:
            raise RuntimeError("MoonshineTts.load() has not been called")
        return tts

    def synthesize(self, text: str) -> tuple[list[float], int]:
        return self._engine.synthesize(text)

    def stream(self, text: str | Iterable[str] | None = None) -> Iterator[AudioChunk]:
        """Like `TextToSpeech.stream()`, but it yields the GIL while waiting.

        The library's `stream()` polls `next_chunk()` in a loop with no sleep,
        so for the ~390 ms before the first sentence is ready it busy-spins at
        full speed on this thread. That starves every other Python thread in the
        process: measured, it pushed time-to-first-PCM from ~390 ms to 1.4 s with
        high variance, because the event loop could not get the GIL to send the
        frame, and it made a `cancel_stream()` from another thread appear to
        block for 82-156 ms when the call itself costs 0.0 ms. The library's own
        `_ChunkPump` sleeps 5 ms in exactly this spot; `stream()` just omits it.

        The loop below is the library's, minus the `Iterable[str]` support we do
        not use, plus that sleep. Returning on `not is_streaming` after a `None`
        chunk is the library's own termination condition, which collapses
        "finished" and "cancelled" -- fine here, because `TtsService` tracks
        cancellation with its generation counter rather than inferring it.
        """
        tts = self._engine
        if text is None:
            raise ValueError("MoonshineTts.stream() requires text")
        if not isinstance(text, str):
            text = "".join(text)
        tts.push_text(text)
        tts.end_input()
        while True:
            chunk = tts.next_chunk()
            if chunk is not None:
                yield chunk
                continue
            if not tts.is_streaming:
                return
            time.sleep(CHUNK_POLL_INTERVAL)

    def split_utterances(self, text: str) -> list[str]:
        from moonshine_voice.tts import split_say_utterances

        return split_say_utterances(text, self.language)

    def stop(self) -> None:
        # Safe from any thread: it ends an in-flight stream() iteration running
        # on the synthesis worker, and the call itself costs 0.0 ms measured.
        # It only *appeared* to block for 82-156 ms while the consumer was the
        # library's busy-spinning stream(); see this class's stream() for why.
        # `TtsService` still calls it off the event loop, since a native call
        # blocking the loop would stall ASR intake too.
        tts = self._tts
        if tts is None:
            return
        try:
            tts.cancel_stream()
        except Exception:
            log.exception("cancel_stream failed")

    @property
    def is_talking(self) -> bool:
        # `is_streaming`, not the engine's `is_talking()`: the latter reports on
        # Moonshine's own sounddevice playback, which we never use -- our audio
        # leaves over the websocket data plane instead.
        tts = self._tts
        return bool(tts is not None and tts.is_streaming)

    def close(self) -> None:
        with self._lifecycle:
            if self._tts is not None:
                self._tts.close()
                self._tts = None


class KokoroOnnxTts:
    """Kokoro-82M via kokoro-onnx, on the GPU when one is usable.

    The point of this backend is core contention, not voice quality: Moonshine's
    TTS is CPU-only and one synthesis takes ~14 of 20 cores, which costs
    concurrent ASR 4.3x its time-to-first-audio (ARCHITECTURE.md §5.11). This
    runs the same Kokoro model the Moonshine path already uses -- the voices are
    literally the same `af_heart` et al -- so moving it to the GPU buys the cores
    back without changing how the agent sounds.

    Falls back to CPU automatically when CUDA is unusable, because the failure
    mode otherwise is silent: onnxruntime logs an error, drops to
    CPUExecutionProvider and synthesizes anyway, so without an explicit check
    you cannot tell a working GPU path from a working CPU one.
    """

    name = "kokoro-onnx"

    MODEL_URL = (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
    )
    VOICES_URL = (
        "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"
    )
    SAMPLE_RATE = 24000

    #: Clause-split anything longer than this many characters. A `create()` call
    #: cannot be interrupted, so one very long sentence would hold off barge-in
    #: for its whole synthesis; splitting bounds that wait.
    MAX_SENTENCE_CHARS = 220

    def __init__(
        self,
        voice: str = "af_heart",
        lang: str = "en-us",
        cache: Path | None = None,
        prefer_gpu: bool = True,
    ) -> None:
        # Accepts Moonshine-style "kokoro_af_heart" so one config value works
        # for either backend.
        self.voice = voice.split("kokoro_", 1)[-1] if voice.startswith("kokoro_") else voice
        self.lang = lang if "-" in lang else "en-us"
        self.cache = cache or (Path.home() / ".cache" / "voice-agent")
        self.prefer_gpu = prefer_gpu
        self._k = None
        self._provider = "none"
        self._cancel = threading.Event()
        self._talking = False
        self._lifecycle = threading.Lock()

    # ------------------------------------------------------------------ assets

    def _asset(self, name: str, url: str) -> Path:
        p = self.cache / name
        if not p.is_file():
            self.cache.mkdir(parents=True, exist_ok=True)
            log.info("downloading %s", name)
            tmp = p.with_suffix(p.suffix + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(p)  # atomic: a partial file never looks cached
        return p

    @staticmethod
    def _preload_cuda_libs() -> int:
        """Pull the pip-installed CUDA libraries into the process by hand.

        onnxruntime-gpu does not add the `nvidia/*/lib` wheel directories to the
        search path, and the dynamic loader reads LD_LIBRARY_PATH only at process
        start, so an in-process fix cannot use it. Without this the CUDA provider
        fails with "libcublasLt.so.12: cannot open shared object file", falls back
        to CPU, and synthesizes anyway -- a working GPU path and a working CPU
        path look identical apart from one log line.
        """
        import ctypes
        import glob
        import site

        roots = list(site.getsitepackages())
        if (user := site.getusersitepackages()) not in roots:
            roots.append(user)
        loaded = 0
        for root in roots:
            # cublasLt before cublas: the latter needs the former already resident.
            for so in sorted(glob.glob(os.path.join(root, "nvidia", "*", "lib", "*.so*"))):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                    loaded += 1
                except OSError:
                    continue  # a lib this build does not need
        return loaded

    def load(self) -> "KokoroOnnxTts":
        if self.prefer_gpu:
            n = self._preload_cuda_libs()
            log.debug("preloaded %d CUDA libraries for onnxruntime", n)

        import onnxruntime as ort
        from kokoro_onnx import Kokoro

        with self._lifecycle:
            if self._k is not None:
                return self
            model = self._asset("kokoro-v1.0.onnx", self.MODEL_URL)
            voices = self._asset("voices-v1.0.bin", self.VOICES_URL)
            opts = ort.SessionOptions()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.prefer_gpu and "CUDAExecutionProvider" in ort.get_available_providers()
                else ["CPUExecutionProvider"]
            )
            sess = ort.InferenceSession(str(model), sess_options=opts, providers=providers)
            # What we asked for and what we got differ whenever the CUDA EP
            # fails to initialize, so report the latter.
            self._provider = sess.get_providers()[0]
            if self.prefer_gpu and self._provider != "CUDAExecutionProvider":
                log.warning(
                    "Kokoro wanted CUDA but is running on %s -- the GPU is NOT being used, "
                    "and ASR/TTS core contention still applies",
                    self._provider,
                )
            else:
                log.info("Kokoro on %s, voice=%s", self._provider, self.voice)
            self._k = Kokoro.from_session(sess, str(voices))
        return self

    @property
    def provider(self) -> str:
        """Execution provider actually in use, e.g. CUDAExecutionProvider."""
        return self._provider

    @property
    def on_gpu(self) -> bool:
        return self._provider == "CUDAExecutionProvider"

    @property
    def _engine(self):
        k = self._k
        if k is None:
            raise RuntimeError("KokoroOnnxTts.load() has not been called")
        return k

    # ------------------------------------------------------------- synthesis

    def synthesize(self, text: str) -> tuple[list[float], int]:
        samples, sr = self._engine.create(text, voice=self.voice, lang=self.lang)
        return list(samples), int(sr)

    def stream(self, text: str | Iterable[str] | None = None) -> Iterator[AudioChunk]:
        """Yield one chunk per sentence, so the first audio does not wait for the rest.

        Unlike the Moonshine backend, `utterance_id` here is assigned by the
        same `split_utterances()` the service calls, so the id a `speak()`
        returns is *guaranteed* to be the id its first chunk carries rather than
        inferred from engine behaviour (contrast ARCHITECTURE.md §5.9).
        """
        if text is None:
            raise ValueError("KokoroOnnxTts.stream() requires text")
        if not isinstance(text, str):
            text = "".join(text)
        parts = self.split_utterances(text)
        # Clearing here would drop a barge-in issued in the instant before this
        # call -- the same race as the deferred cancel in ARCHITECTURE.md §5.8,
        # wearing a different hat. It is safe *only* because TtsService's
        # `_cancel_done` gate makes the synthesis worker wait for a pending
        # cancel to land before entering stream(). That gate is therefore not
        # Moonshine-specific defensive code: removing it reintroduces the bug
        # here too.
        self._cancel.clear()
        self._talking = True
        try:
            for i, part in enumerate(parts, start=1):
                if self._cancel.is_set():
                    return
                samples, sr = self._engine.create(part, voice=self.voice, lang=self.lang)
                if self._cancel.is_set():
                    return  # drop audio synthesized across a barge-in
                yield _Chunk(
                    samples=samples, sample_rate=int(sr), text=part,
                    utterance_id=i, is_final=True,
                )
        finally:
            self._talking = False

    def split_utterances(self, text: str) -> list[str]:
        """Sentence split, then clause-split anything over MAX_SENTENCE_CHARS."""
        out: list[str] = []
        for s in _split_sentences(text):
            out.extend(_split_long(s, self.MAX_SENTENCE_CHARS))
        return out or ([text.strip()] if text.strip() else [])

    def stop(self) -> None:
        """Barge-in. Safe from any thread; lands between sentences."""
        self._cancel.set()

    @property
    def is_talking(self) -> bool:
        return self._talking

    def close(self) -> None:
        with self._lifecycle:
            self._k = None


#: Where a model is loaded. "auto" prefers the GPU and silently accepts CPU;
#: "gpu" and "cpu" are assertions -- "gpu" fails loudly rather than degrading,
#: because a silent CPU fallback is indistinguishable from success apart from
#: one log line (see KokoroOnnxTts.load).
DEVICES = ("auto", "gpu", "cpu")


def make_backend(
    kind: str, *, language: str, voice: str, device: str = "auto"
) -> TtsBackend:
    """Build a backend by name, falling back to Moonshine when asked to.

    "kokoro" is the default because it keeps synthesis off the CPU cores ASR
    needs; "kokoro-or-moonshine" additionally survives a missing kokoro-onnx or
    an unusable GPU by handing back the CPU engine, which is slower in aggregate
    but always available.
    """
    if device not in DEVICES:
        raise ValueError(f"tts device must be one of {DEVICES}, not {device!r}")
    if kind == "moonshine":
        if device == "gpu":
            raise ValueError(
                "moonshine TTS is CPU-only: its bundled onnxruntime registers no CUDA "
                "provider and `use_cuda` is accepted but inert. Use --tts-backend kokoro "
                "for the GPU path."
            )
        return MoonshineTts(language=language, voice=voice)
    prefer_gpu = device != "cpu"
    if kind in ("kokoro", "kokoro-onnx"):
        backend = KokoroOnnxTts(voice=voice, lang=language, prefer_gpu=prefer_gpu)
        if device == "gpu":
            backend.load()  # surface a CUDA failure now, not on the first utterance
            if not backend.on_gpu:
                raise RuntimeError(
                    f"--tts-load gpu requested but Kokoro resolved to {backend.provider}. "
                    "Refusing to run on CPU under an explicit gpu request."
                )
        return backend
    if kind in ("auto", "kokoro-or-moonshine"):
        try:
            return KokoroOnnxTts(voice=voice, lang=language, prefer_gpu=prefer_gpu).load()
        except Exception as e:  # noqa: BLE001 -- any failure means fall back
            log.warning("kokoro-onnx unavailable (%s: %s); falling back to Moonshine TTS",
                        type(e).__name__, e)
            return MoonshineTts(language=language, voice=voice)
    raise ValueError(f"unknown TTS backend {kind!r}")
