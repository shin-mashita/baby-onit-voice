"""Entry point: `python -m speech_server [--prewarm]`.

Runs both endpoints in one process and one event loop. Two uvicorn servers,
because the protocol puts ASR and TTS on separate ports; signal handling is
ours rather than uvicorn's, so one SIGINT brings both down together.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import sys
import time

import uvicorn

from . import ASR_PORT, TTS_PORT
from .asr_ws import (
    DEFAULT_UPDATE_INTERVAL,
    KEYTERM_BOOST,
    KEYTERM_BOOST_MAX,
    AsrModel,
    build_asr_app,
)
from .backends import DEFAULT_LANGUAGE, DEFAULT_VOICE, DEVICES, make_backend
from .tts_audio_ws import AudioHub
from .tts_mcp import TtsService, build_tts_app

log = logging.getLogger("speech_server")


def configure_logging(level: str) -> None:
    """Timestamped structured logging to stderr.

    Explicit handlers because mcp and uvicorn both install their own on import;
    stdout has to stay clean for anything piping this process.
    """
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s.%(msecs)03dZ %(levelname)-7s %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    handler.formatter.converter = time.gmtime
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(getattr(logging, level.upper()))
    for noisy in ("uvicorn", "uvicorn.error", "uvicorn.access", "mcp", "sse_starlette"):
        chatty = logging.getLogger(noisy)
        chatty.handlers[:] = []
        chatty.propagate = True
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m speech_server", description=__doc__)
    p.add_argument("--host", default="0.0.0.0", help="bind address for both ports")
    p.add_argument("--asr-port", type=int, default=ASR_PORT)
    p.add_argument("--tts-port", type=int, default=TTS_PORT)
    p.add_argument("--language", default="en", help="ASR language")
    p.add_argument("--tts-language", default=DEFAULT_LANGUAGE)
    p.add_argument("--voice", default=DEFAULT_VOICE)
    p.add_argument(
        "--model-arch",
        default="MEDIUM_STREAMING",
        choices=[
            "TINY",
            "BASE",
            "TINY_STREAMING",
            "BASE_STREAMING",
            "SMALL_STREAMING",
            "MEDIUM_STREAMING",
        ],
        help=(
            "MEDIUM_STREAMING (default): most accurate, RTF 0.81 -- viable only "
            "because voice_client parks the mic during playback so ASR never "
            "shares cores with TTS. SMALL_STREAMING (RTF 0.60) is the safer "
            "choice if anything else loads this machine. TINY_STREAMING (0.22) "
            "is the fastest but degrades from 10 dB SNR (measured). "
            "BASE_STREAMING has no English model"
        ),
    )
    p.add_argument(
        "--tts-backend",
        default="auto",
        choices=["auto", "kokoro", "moonshine"],
        help=(
            "auto (default): kokoro-onnx on CUDA, falling back to moonshine if the "
            "GPU is unusable. On GPU kokoro runs at RTF 0.047 in ~620 MiB of VRAM, "
            "which also takes synthesis off the CPU cores ASR needs. moonshine: "
            "CPU-only, RTF 0.25, and contends with ASR for cores"
        ),
    )
    p.add_argument(
        "--tts-load",
        default="auto",
        choices=DEVICES,
        help=(
            "where to load TTS. auto: prefer GPU, accept CPU. gpu: require CUDA and "
            "fail rather than silently degrade. cpu: force CPU (Kokoro RTF 0.25)"
        ),
    )
    p.add_argument(
        "--asr-load",
        default="cpu",
        choices=DEVICES,
        help=(
            "where to load ASR. cpu (default): Moonshine, CPU-only, genuinely "
            "frame-streaming with partials. gpu: Qwen3-ASR via a vLLM server "
            "(--qwen-url); needs ~7.2 GiB so TTS must be on CPU, and emits NO partials"
        ),
    )
    p.add_argument(
        "--qwen-url",
        default="http://127.0.0.1:8773/v1",
        help="OpenAI-compatible vLLM endpoint serving Qwen3-ASR (used by --asr-load gpu)",
    )
    p.add_argument(
        "--keyterm-boost",
        type=float,
        default=KEYTERM_BOOST,
        help=(
            f"strength of keyterm biasing (default {KEYTERM_BOOST}). Below ~5 it has "
            f"no measurable effect; above {KEYTERM_BOOST_MAX} the decoder starts "
            "emitting a keyterm repeatedly instead of transcribing"
        ),
    )
    p.add_argument("--update-interval", type=float, default=DEFAULT_UPDATE_INTERVAL)
    p.add_argument(
        "--prewarm",
        action="store_true",
        help="run a throwaway synthesis and transcription so no lazy native "
        "initialization lands on a live utterance",
    )
    p.add_argument("--log-level", default="info")
    args = p.parse_args(argv)
    if args.asr_load == "gpu" and args.tts_load == "gpu":
        # This combination DOES fit, but only when the Qwen server is started
        # with explicit KV sizing. Measured together: 6459 MiB of 8192, with
        # 1733 MiB spare (Qwen 5825 + Kokoro 634). Loading Qwen through the
        # in-process LLM() API instead takes 7389 MiB, because its memory
        # profiling overshoots, and then the pair genuinely does not fit --
        # which is why scripts_qwen_serve.sh pins --kv-cache-memory.
        log.info(
            "asr+tts both on GPU: fits with scripts_qwen_serve.sh (measured 6459/8192 MiB). "
            "If the Qwen server was started without --kv-cache-memory it will take 7389 MiB "
            "and this pair will OOM."
        )
    if args.keyterm_boost > KEYTERM_BOOST_MAX:
        log.warning(
            "--keyterm-boost %.1f exceeds the safe ceiling %.1f; clamping. Past this "
            "the decoder repeats a keyterm instead of transcribing (measured).",
            args.keyterm_boost, KEYTERM_BOOST_MAX,
        )
        args.keyterm_boost = KEYTERM_BOOST_MAX
    return args


def _prewarm(backend, model: AsrModel, update_interval: float) -> None:
    import numpy as np
    from moonshine_voice.transcriber import Transcriber

    t0 = time.monotonic()
    samples, rate = backend.synthesize("Ready.")
    log.info("prewarm: tts synthesized %.2fs of audio in %.2fs", len(samples) / rate, time.monotonic() - t0)

    t0 = time.monotonic()
    transcriber = Transcriber(model.path, model.arch, update_interval=update_interval)
    try:
        stream = transcriber.create_stream()
        stream.start()
        stream.add_audio(np.zeros(16000, dtype=np.float32).tolist(), sample_rate=16000)
        stream.stop()
        stream.close()
    finally:
        transcriber.close()
    log.info("prewarm: asr round trip in %.2fs", time.monotonic() - t0)


async def _serve(args: argparse.Namespace) -> int:
    loop = asyncio.get_running_loop()

    t0 = time.monotonic()
    model = await asyncio.to_thread(AsrModel.resolve, args.language, args.model_arch)
    log.info("asr model %s resolved in %.2fs", model.label, time.monotonic() - t0)
    if args.asr_load == "gpu":
        log.warning(
            "ASR is Qwen3-ASR at %s: NO partial transcripts (it has no frame-level "
            "emission); text arrives once per utterance when the client finalizes",
            args.qwen_url,
        )

    # t0 before make_backend, not after: the "auto" backend has to load in order
    # to find out whether CUDA works before it can decide to fall back, so a t0
    # placed after the call reports ~0.00s and leaves a first run's ~320 MB
    # asset download as unattributed dead time. backend.load() is idempotent, so
    # the await below is a no-op when the factory already did the work.
    t0 = time.monotonic()
    backend = make_backend(
        args.tts_backend, language=args.tts_language, voice=args.voice, device=args.tts_load
    )
    await asyncio.to_thread(backend.load)
    # Say which provider actually won: a CUDA EP that fails to initialize falls
    # back to CPU and still synthesizes, so "kokoro" alone does not mean "GPU".
    where = getattr(backend, "provider", "cpu")
    log.info(
        "tts backend %s voice=%s on %s loaded in %.2fs",
        backend.name, args.voice, where, time.monotonic() - t0,
    )
    if backend.name == "moonshine":
        log.warning(
            "Moonshine TTS is CPU-only: one synthesis takes ~14 cores and costs "
            "concurrent ASR about 4.3x its time-to-first-audio"
        )

    if args.prewarm:
        await asyncio.to_thread(_prewarm, backend, model, args.update_interval)

    hub = AudioHub()
    hub.bind_loop(loop)
    service = TtsService(backend, hub)
    hub.set_flush_handler(service.stop_speaking)
    service.start()

    servers = [
        _make_server(
            build_asr_app(
                model,
                update_interval=args.update_interval,
                keyterm_boost=args.keyterm_boost,
                qwen_url=args.qwen_url if args.asr_load == "gpu" else None,
            ),
            args.host, args.asr_port,
        ),
        _make_server(build_tts_app(service, hub), args.host, args.tts_port),
    ]

    stopping = asyncio.Event()

    def request_stop(signame: str) -> None:
        if not stopping.is_set():
            log.info("%s received; shutting down", signame)
            stopping.set()
            for server in servers:
                server.should_exit = True

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, request_stop, sig.name)

    tasks = [asyncio.create_task(s.serve(), name=f"uvicorn-{i}") for i, s in enumerate(servers)]

    async def announce() -> None:
        """Report the endpoints only once both sockets are actually bound.

        Logging this before awaiting startup claims we are listening even when
        the bind is about to fail, which is exactly the case an operator is
        reading the log to diagnose.
        """
        while not all(s.started for s in servers):
            if any(t.done() for t in tasks):
                return
            await asyncio.sleep(0.02)
        log.info(
            "listening: ws://%s:%d/asr  http://%s:%d/mcp  ws://%s:%d/audio",
            args.host,
            args.asr_port,
            args.host,
            args.tts_port,
            args.host,
            args.tts_port,
        )

    announcer = asyncio.create_task(announce(), name="announce")
    exit_code = 0
    try:
        await asyncio.gather(*tasks)
    except SystemExit as exc:
        # uvicorn sys.exit()s out of its own startup when a port is taken, which
        # otherwise surfaces as an unretrieved-task traceback several frames deep.
        # Double-starting is an easy mistake; it deserves one clear line.
        exit_code = exc.code if isinstance(exc.code, int) else 1
        log.error(
            "could not bind %s on ports %d/%d -- is a speech_server already running?",
            args.host,
            args.asr_port,
            args.tts_port,
        )
    finally:
        announcer.cancel()
        for task in tasks:
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.gather(announcer, *tasks, return_exceptions=True)
        await hub.close()
        await asyncio.to_thread(service.close)
        log.info("shutdown complete")
    return exit_code


def _make_server(app, host: str, port: int) -> uvicorn.Server:
    # log_config=None matters: uvicorn.Config otherwise reinstalls its own
    # handlers on instantiation, which happens *after* configure_logging() and
    # leaves uvicorn's lines bypassing our timestamped stderr format entirely.
    config = uvicorn.Config(
        app, host=host, port=port, log_level="warning", ws="auto", log_config=None
    )
    server = uvicorn.Server(config)
    # We own SIGINT: uvicorn's own handler would only stop one of the two.
    server.capture_signals = contextlib.nullcontext
    return server


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    try:
        return asyncio.run(_serve(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
