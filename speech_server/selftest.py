"""End-to-end check against a running `python -m speech_server`.

    python -m speech_server --prewarm &
    python -m speech_server.selftest

Covers ARCHITECTURE.md §6 items 3 and 4:

1. ASR -- feed a WAV at 1x realtime, assert partials arrive before the audio
   ends (true streaming, not VAD-chunked re-decode).
2. TTS control/data split -- `speak` must return in < 50 ms while PCM is still
   arriving on the data plane.
3. Barge-in -- `stop_speaking` must truncate playback, from either plane.
4. Robustness -- `reset` must drop the in-flight utterance, and a client that
   vanishes mid-utterance must not take the server with it.

Exits non-zero on failure and prints the measured numbers either way.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import wave
from dataclasses import dataclass, field

import numpy as np
import websockets
from mcp import Client

from . import ASR_PORT, TTS_PORT
from .tts_audio_ws import unpack_frame

SAMPLE_RATE = 16000
FRAME_MS = 100
SPEAK_BUDGET_MS = 50.0
#: Generous on purpose. Time-to-first-audio is ~380 ms with ASR idle but ~1.5 s
#: while ASR is ingesting concurrently, so a tight window here would fail a
#: perfectly correct server. The latency itself is measured, not asserted.
FIRST_AUDIO_TIMEOUT = 6.0
#: VOICE_PROTOCOL.md §4: steps 2-4 of a barge-in must complete within 300 ms.
BARGE_IN_BUDGET_S = 0.3

SAMPLE_WAV = (
    "/tmp/claude-1000/-home-cencarna-ced-voice-agent/"
    "0620f885-cf81-47b9-b494-fd04194360b6/scratchpad/sample.wav"
)
SAMPLE_TEXT = (
    "I am thinking of creating a simple web game. Can you create me a very "
    "good flappy bird app?"
)
LONG_TEXT = (
    "This is a deliberately long reply that keeps going for a while. "
    "It has several sentences so there is plenty left to interrupt. "
    "We want to know how fast the cancel actually lands. "
    "Here is yet more text that should never be spoken at all. "
    "And one final sentence to be sure the queue is not empty."
)


@dataclass
class Results:
    checks: list[tuple[bool, str]] = field(default_factory=list)

    def check(self, ok: bool, label: str) -> bool:
        self.checks.append((bool(ok), label))
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        return bool(ok)

    @property
    def failed(self) -> int:
        return sum(1 for ok, _ in self.checks if not ok)


# -- audio helpers --------------------------------------------------------


def resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate:
        return samples.astype(np.float32)
    n_out = int(round(len(samples) * dst_rate / src_rate))
    xp = np.arange(len(samples), dtype=np.float64)
    x = np.linspace(0, len(samples) - 1, n_out)
    return np.interp(x, xp, samples).astype(np.float32)


def read_wav(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as w:
        if w.getsampwidth() != 2 or w.getnchannels() != 1:
            raise ValueError(f"{path}: expected 16-bit mono")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
        return pcm.astype(np.float32) / 32768.0, w.getframerate()


def write_wav(path: str, samples: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


async def ensure_sample(tts_url: str, audio_url: str, path: str) -> tuple[np.ndarray, int]:
    """Reuse the recorded WAV, else have the server speak one and capture it."""
    try:
        samples, rate = read_wav(path)
        print(f"  using {path} ({len(samples) / rate:.2f}s @ {rate} Hz)")
        return resample(samples, rate, SAMPLE_RATE), SAMPLE_RATE
    except (OSError, ValueError):
        pass
    print(f"  {path} unusable; synthesizing a sample through the server")
    collected: list[np.ndarray] = []
    rate = 24000
    async with websockets.connect(audio_url, max_size=None) as audio:
        async with Client(tts_url) as mcp:
            await mcp.call_tool("speak", {"text": SAMPLE_TEXT})
        while True:
            try:
                frame = await asyncio.wait_for(audio.recv(), timeout=5.0)
            except asyncio.TimeoutError:
                break
            _, rate, _, _, pcm = unpack_frame(frame)
            collected.append(pcm)
    samples = np.concatenate(collected) if collected else np.zeros(1, dtype=np.float32)
    write_wav(path, samples, rate)
    return resample(samples, rate, SAMPLE_RATE), SAMPLE_RATE


# -- 1. ASR ---------------------------------------------------------------


async def test_asr(url: str, samples: np.ndarray, results: Results) -> None:
    print("\n[1] ASR: partials must arrive before the audio ends")
    duration = len(samples) / SAMPLE_RATE
    block = int(SAMPLE_RATE * FRAME_MS / 1000)
    events: list[tuple[float, dict]] = []

    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=60.0))
        results.check(ready.get("type") == "ready", f"ready received: {ready}")
        await ws.send(json.dumps({"type": "start", "keyterms": ["flappy bird", "web game"]}))

        t0 = time.monotonic()

        async def reader() -> None:
            async for message in ws:
                events.append((time.monotonic() - t0, json.loads(message)))

        reader_task = asyncio.create_task(reader())

        # 1x realtime, exactly as a live mic would deliver it.
        for i in range(0, len(samples), block):
            await ws.send(samples[i : i + block].astype("<f4").tobytes())
            await asyncio.sleep(FRAME_MS / 1000)
        t_fed = time.monotonic() - t0

        # Transcription completes slightly before the audio does (measured
        # -0.33 s), but give the tail room on a loaded machine.
        await asyncio.sleep(2.5)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(reader_task, timeout=5.0)
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            reader_task.cancel()

    partials = [(t, e) for t, e in events if e["type"] == "partial"]
    lines = [(t, e) for t, e in events if e["type"] == "line"]
    errors = [e for _, e in events if e["type"] == "error"]

    print(f"  audio {duration:.2f}s fed in {t_fed:.2f}s; {len(partials)} partials, {len(lines)} lines")
    for t, e in lines:
        print(f"    line t={t:.2f}s line_id={e['line_id']} {e['text']!r}")

    results.check(not errors, f"no error events ({errors})")
    results.check(bool(partials), "at least one partial arrived")
    if partials:
        first = partials[0][0]
        print(f"  first partial at {first:.2f}s of {duration:.2f}s of audio")
        results.check(first < t_fed, f"first partial {first:.2f}s precedes end of audio {t_fed:.2f}s")
        results.check(
            all("line_id" in e and "t" in e for _, e in partials),
            "partials carry line_id and t",
        )
    if lines:
        text = " ".join(e["text"] for _, e in lines).lower()
        print(f"  transcript: {text!r}")
        results.check("flappy" in text, "keyterm biasing kept 'flappy'")


# -- 2. TTS control/data split -------------------------------------------


async def _settle(mcp, frames: list[tuple[float, tuple]], quiet_for: float = 0.4) -> None:
    """Stop playback and wait until the socket has been quiet for `quiet_for`.

    A fixed sleep is not enough: frames already in flight when stop_speaking
    lands keep arriving for a moment, and if one of them is still in the client
    buffer when the next test clears `frames`, that test sees a stale frame,
    believes audio is already flowing, and polls the control plane before the
    utterance it cares about has started.
    """
    await mcp.call_tool("stop_speaking", {})
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        status = json.loads((await mcp.call_tool("is_speaking", {})).content[0].text)
        n = len(frames)
        await asyncio.sleep(quiet_for)
        if not status.get("speaking") and status.get("queued") == 0 and len(frames) == n:
            break
    frames.clear()


async def _drain(audio, stop: asyncio.Event, frames: list[tuple[float, tuple]]) -> None:
    while not stop.is_set():
        try:
            frame = await asyncio.wait_for(audio.recv(), timeout=0.25)
        except asyncio.TimeoutError:
            continue
        except websockets.exceptions.ConnectionClosed:
            return
        frames.append((time.monotonic(), unpack_frame(frame)))


def _own(frames: list[tuple[float, tuple]], first_id: int) -> list[tuple[float, tuple]]:
    """Frames belonging to the utterance under test.

    The server is long-lived and shared, so a reconnecting client can still be
    handed tail frames from an utterance queued before this test existed -- ids
    are monotonic across the server's whole life, so they are the discriminator.
    """
    return [f for f in frames if f[1][0] >= first_id]


async def test_speak_returns_fast(tts_url: str, audio_url: str, results: Results) -> None:
    print("\n[2] TTS: speak must return in <50 ms while PCM is still arriving")
    frames: list[tuple[float, tuple]] = []
    stop = asyncio.Event()
    async with websockets.connect(audio_url, max_size=None) as audio:
        drain = asyncio.create_task(_drain(audio, stop, frames))
        async with Client(tts_url) as mcp:
            await _settle(mcp, frames)

            t0 = time.monotonic()
            result = await mcp.call_tool("speak", {"text": LONG_TEXT})
            speak_ms = (time.monotonic() - t0) * 1000
            payload = json.loads(result.content[0].text)
            print(f"  speak returned in {speak_ms:.1f} ms -> {payload}")
            results.check(speak_ms < SPEAK_BUDGET_MS, f"speak returned in {speak_ms:.1f} ms (<{SPEAK_BUDGET_MS:.0f})")
            results.check(payload.get("status") == "speaking", "status is 'speaking'")
            results.check(isinstance(payload.get("utterance_id"), int), "utterance_id is an int")

            # Wait for audio to actually start rather than assuming a latency.
            # A fixed window here turns a slow-but-correct server into a bogus
            # failure, and time-to-first-audio varies by ~4x depending on
            # whether ASR is ingesting concurrently (see README).
            first_id = payload["utterance_id"]
            deadline = time.monotonic() + FIRST_AUDIO_TIMEOUT
            while not _own(frames, first_id) and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            mine = _own(frames, first_id)
            status = json.loads((await mcp.call_tool("is_speaking", {})).content[0].text)
            print(f"  is_speaking -> {status}, {len(mine)} frames received")
            results.check(status.get("speaking") is True, "is_speaking true while audio streams")
            results.check(bool(mine), "PCM arrived on the data plane")
            if mine:
                first_at = mine[0][0] - t0
                print(
                    f"  time-to-first-audio {first_at * 1000:.0f} ms"
                    f" (~380 ms expected with ASR idle, ~1.5 s if ASR is ingesting)"
                )
                results.check(
                    speak_ms / 1000 < first_at,
                    f"speak returned ({speak_ms:.1f} ms) before the first PCM frame ({first_at * 1000:.0f} ms)",
                )
                uid, rate, _, _, _ = mine[0][1]
                results.check(rate == 24000, f"sample_rate is {rate}")
                results.check(
                    uid == payload["utterance_id"],
                    f"first frame utterance_id {uid} matches speak's {payload['utterance_id']}",
                )

            n_before = len(_own(frames, first_id))
            deadline = time.monotonic() + FIRST_AUDIO_TIMEOUT
            while len(_own(frames, first_id)) <= n_before and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            results.check(
                len(_own(frames, first_id)) > n_before, "frames kept arriving after speak returned"
            )

            await mcp.call_tool("stop_speaking", {})
        stop.set()
        await drain

    mine = _own(frames, first_id)
    ids = sorted({f[1][0] for f in mine})
    finals = sorted(f[1][0] for f in mine if f[1][3])
    print(f"  utterance ids seen: {ids}; is_final closed: {finals}")
    results.check(
        all(a < b for a, b in zip(ids, ids[1:])),
        "utterance ids are monotonically increasing",
    )


# -- 3. Barge-in ----------------------------------------------------------


async def test_stop_truncates(tts_url: str, audio_url: str, results: Results) -> None:
    print("\n[3] TTS: stop_speaking must truncate playback")
    frames: list[tuple[float, tuple]] = []
    stop = asyncio.Event()
    async with websockets.connect(audio_url, max_size=None) as audio:
        drain = asyncio.create_task(_drain(audio, stop, frames))
        async with Client(tts_url) as mcp:
            await _settle(mcp, frames)

            await mcp.call_tool("speak", {"text": LONG_TEXT})
            await asyncio.sleep(1.0)

            t0 = time.monotonic()
            stopped = json.loads((await mcp.call_tool("stop_speaking", {})).content[0].text)
            stop_ms = (time.monotonic() - t0) * 1000
            n_at_stop = len(frames)
            print(f"  stop_speaking returned in {stop_ms:.1f} ms -> {stopped}")
            results.check(stopped.get("stopped") is True, "stop_speaking reported stopped")
            results.check(stop_ms < 300, f"stop_speaking returned in {stop_ms:.1f} ms (<300)")

            await asyncio.sleep(1.5)
            after = len(frames) - n_at_stop
            audio_s = sum(f[1][2] / f[1][1] for f in frames)
            print(f"  {n_at_stop} frames before stop, {after} after; {audio_s:.2f}s of audio total")
            results.check(after <= 1, f"at most one in-flight frame after stop (got {after})")

            status = json.loads((await mcp.call_tool("is_speaking", {})).content[0].text)
            print(f"  is_speaking after stop -> {status}")
            results.check(status.get("speaking") is False, "is_speaking false after stop")
            results.check(status.get("queued") == 0, "queue empty after stop")
            # LONG_TEXT is ~20 s spoken; anything near that means no truncation.
            results.check(audio_s < 8.0, f"playback truncated to {audio_s:.2f}s of a ~20s reply")

            # The engine must still work afterwards.
            again = json.loads((await mcp.call_tool("speak", {"text": "Still here."})).content[0].text)
            await asyncio.sleep(1.5)
            print(f"  post-barge-in speak -> {again}, {len(frames) - n_at_stop - after} new frames")
            results.check(
                len(frames) > n_at_stop + after,
                "synthesis works again after a barge-in",
            )
            await mcp.call_tool("stop_speaking", {})
        stop.set()
        await drain


# -- 4. Barge-in over the data plane --------------------------------------


async def test_flush_control(tts_url: str, audio_url: str, results: Results) -> None:
    print("\n[4] TTS: {'type':'flush'} on the audio socket is a barge-in")
    frames: list[tuple[float, tuple]] = []
    stop = asyncio.Event()
    async with websockets.connect(audio_url, max_size=None) as audio:
        drain = asyncio.create_task(_drain(audio, stop, frames))
        async with Client(tts_url) as mcp:
            await _settle(mcp, frames)
            spoken = json.loads(
                (await mcp.call_tool("speak", {"text": LONG_TEXT})).content[0].text
            )
            first_id = spoken["utterance_id"]
            # Flush only once audio is genuinely flowing. Flushing on a fixed
            # delay tests nothing when time-to-first-audio runs long, which it
            # does by ~4x whenever ASR is ingesting concurrently.
            deadline = time.monotonic() + FIRST_AUDIO_TIMEOUT
            while not _own(frames, first_id) and time.monotonic() < deadline:
                await asyncio.sleep(0.01)
            results.check(bool(_own(frames, first_id)), "audio was flowing before the flush")

            n_at_flush = len(_own(frames, first_id))
            t_flush = time.monotonic()
            await audio.send(json.dumps({"type": "flush"}))
            await asyncio.sleep(1.5)
            mine = _own(frames, first_id)
            after = len(mine) - n_at_flush
            # Bound this in time, not in frames. `flush` is fire-and-forget, so
            # unlike the synchronous `stop_speaking` of [3] the client cannot
            # know when the server handled it, and any number of frames may
            # already be in the server's send queue at that moment. What §4
            # actually requires is that audio stops within 300 ms.
            late = [f for f in mine if f[0] > t_flush + BARGE_IN_BUDGET_S]
            stopped_after = (max(f[0] for f in mine) - t_flush) * 1000 if mine else 0.0
            status = json.loads((await mcp.call_tool("is_speaking", {})).content[0].text)
            print(
                f"  {n_at_flush} frames at flush, {after} after, last arrived "
                f"{stopped_after:+.0f} ms from flush; is_speaking -> {status}"
            )
            results.check(
                not late,
                f"audio stopped within {BARGE_IN_BUDGET_S * 1000:.0f} ms of flush "
                f"({len(late)} late frames)",
            )
            results.check(status.get("speaking") is False, "is_speaking false after flush")
        stop.set()
        await drain


# -- 5. ASR reset ---------------------------------------------------------


async def test_asr_reset(url: str, samples: np.ndarray, results: Results) -> None:
    print("\n[5] ASR: reset drops the in-flight utterance and keeps the socket")
    block = int(SAMPLE_RATE * FRAME_MS / 1000)
    events: list[dict] = []
    async with websockets.connect(url, max_size=None) as ws:
        await asyncio.wait_for(ws.recv(), timeout=60.0)
        await ws.send(json.dumps({"type": "start", "keyterms": ["flappy bird"]}))

        async def reader() -> None:
            async for message in ws:
                events.append(json.loads(message))

        reader_task = asyncio.create_task(reader())

        # Half the utterance, then abandon it.
        for i in range(0, len(samples) // 2, block):
            await ws.send(samples[i : i + block].astype("<f4").tobytes())
            await asyncio.sleep(FRAME_MS / 1000)
        await ws.send(json.dumps({"type": "reset"}))
        n_before = len(events)

        for i in range(0, len(samples), block):
            await ws.send(samples[i : i + block].astype("<f4").tobytes())
            await asyncio.sleep(FRAME_MS / 1000)
        await asyncio.sleep(2.0)
        await ws.send(json.dumps({"type": "stop"}))
        try:
            await asyncio.wait_for(reader_task, timeout=5.0)
        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
            reader_task.cancel()

    after = events[n_before:]
    errors = [e for e in after if e["type"] == "error"]
    lines = [e for e in after if e["type"] == "line"]
    print(f"  {n_before} events before reset, {len(after)} after ({len(lines)} lines)")
    results.check(bool(events[:n_before]), "events arrived before the reset")
    results.check(not errors, f"no errors after reset ({errors})")
    results.check(bool(after), "the socket still works after reset")
    if after:
        print(f"  first post-reset event: {after[0]['type']} at t={after[0]['t']:.2f}s")
        # A flushed line from the abandoned utterance would land here instead,
        # carrying the old clock.
        results.check(after[0]["type"] == "partial", "reset did not flush a line for the dropped utterance")
        results.check(after[0]["t"] < 2.5, f"the t clock restarted at reset (t={after[0]['t']:.2f}s)")


# -- 6. Mid-utterance disconnect -----------------------------------------


async def test_disconnect_mid_utterance(tts_url: str, audio_url: str, results: Results) -> None:
    print("\n[6] TTS: an audio client vanishing mid-utterance is survivable")
    audio = await websockets.connect(audio_url, max_size=None)
    async with Client(tts_url) as mcp:
        await mcp.call_tool("speak", {"text": LONG_TEXT})
        await asyncio.sleep(0.8)
        received = 0
        while True:
            try:
                await asyncio.wait_for(audio.recv(), timeout=0.05)
            except asyncio.TimeoutError:
                break
            received += 1
        # Drop the socket with no flush, mid-utterance.
        audio.transport.abort()
        print(f"  dropped the socket after {received} frames, no flush sent")
        await asyncio.sleep(1.5)

        status = json.loads((await mcp.call_tool("is_speaking", {})).content[0].text)
        print(f"  is_speaking with no client attached -> {status}")
        results.check(isinstance(status.get("speaking"), bool), "control plane still answers")

        # Deliberately no settling delay: speaking straight after a barge-in is
        # the real pattern (stop, then a filler), and it races the deferred
        # cancel_stream(). Sleeping here would hide that.
        await mcp.call_tool("stop_speaking", {})
        async with websockets.connect(audio_url, max_size=None) as reconnected:
            await mcp.call_tool("speak", {"text": "Back again now."})
            frames = 0
            for _ in range(10):
                try:
                    await asyncio.wait_for(reconnected.recv(), timeout=3.0)
                except asyncio.TimeoutError:
                    break
                frames += 1
            print(f"  reconnected client received {frames} frames")
            results.check(frames > 0, "a reconnected client still receives audio")
            await mcp.call_tool("stop_speaking", {})


# -- driver ---------------------------------------------------------------


async def main_async(args: argparse.Namespace) -> int:
    asr_url = f"ws://{args.host}:{args.asr_port}/asr"
    tts_url = f"http://{args.host}:{args.tts_port}/mcp"
    audio_url = f"ws://{args.host}:{args.tts_port}/audio"
    print(f"speech_server selftest against {args.host} (asr {args.asr_port}, tts {args.tts_port})")

    results = Results()
    print("\n[0] sample audio")
    samples, _ = await ensure_sample(tts_url, audio_url, args.wav)

    await test_asr(asr_url, samples, results)
    await test_speak_returns_fast(tts_url, audio_url, results)
    await test_stop_truncates(tts_url, audio_url, results)
    await test_flush_control(tts_url, audio_url, results)
    await test_asr_reset(asr_url, samples, results)
    await test_disconnect_mid_utterance(tts_url, audio_url, results)

    total = len(results.checks)
    print(f"\n{total - results.failed}/{total} checks passed")
    return 1 if results.failed else 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m speech_server.selftest")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--asr-port", type=int, default=ASR_PORT)
    p.add_argument("--tts-port", type=int, default=TTS_PORT)
    p.add_argument("--wav", default=SAMPLE_WAV)
    args = p.parse_args(argv)
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
