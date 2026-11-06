"""Latency benchmark for the slides.pdf metrics table: E2E, ASR final, LLM
TTFT, TTS TTFB. Run from the Pi (RPi monotonic clock) against the real
deployed stack: speech_server over the reverse tunnel (127.0.0.1:8771/8772)
and the real LLM via Portkey/Claude. Produces one JSON log line per run plus
a mean/p95 summary, matching the slide's own methodology.

Barge-in is NOT measured here -- see docs/DETAILS.md for why (acoustic
mic/speaker separation on this Pi makes a live acoustic test unreliable right
now) and the existing software-level numbers cited instead.

Usage: python3 bench_latency.py <path-to-repo-root>
"""
from __future__ import annotations

import asyncio
import json
import statistics as stats
import struct
import sys
import time
import wave

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")

import numpy as np  # noqa: E402
import websockets  # noqa: E402

import baby_onit  # noqa: E402
from baby_onit import VoiceSession, load_config  # noqa: E402

ASR_URL = "ws://127.0.0.1:8771/asr"
AUDIO_URL = "ws://127.0.0.1:8772/audio"
WAV_FILES = ["mytest.wav", "recording.wav"]
RUNS_PER_FILE = 5
LLM_TASK = "In one short sentence, what is two plus two?"


def load_pcm_f32(path: str) -> bytes:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000 and w.getnchannels() == 1, path
        raw = w.readframes(w.getnframes())
    i16 = np.frombuffer(raw, dtype="<i2")
    f32 = (i16.astype(np.float32) / 32768.0)
    return f32.tobytes()


async def asr_once(pcm: bytes) -> float:
    """finalize sent -> final transcript received. Returns latency in seconds."""
    async with websockets.connect(ASR_URL, max_size=None) as ws:
        await ws.send(json.dumps({"type": "start", "keyterms": [], "context": None}))
        await ws.recv()  # "ready"
        chunk = 6400  # 100ms @ 16kHz float32
        for i in range(0, len(pcm), chunk):
            await ws.send(pcm[i : i + chunk])
        t0 = time.monotonic()
        await ws.send(json.dumps({"type": "finalize"}))
        while True:
            msg = json.loads(await ws.recv())
            if msg["type"] in ("line", "no_speech", "error"):
                t1 = time.monotonic()
                return t1 - t0


async def llm_and_tts_once(session: VoiceSession) -> dict:
    """transcript ready -> LLM first token (via usage event) and -> speak()
    call -> first matching-utterance_id audio chunk. Runs concurrently: the
    /audio listener is armed before the speak() call it is waiting on.
    """
    usage: dict = {}

    def on_event(kind: str, text: str) -> None:
        if kind == "usage":
            usage.update(json.loads(text))

    session.on_event = on_event
    t_task_start = time.monotonic()
    answer = await session.ask(LLM_TASK)
    t_answer = time.monotonic()
    ttft = usage.get("model_s", 0) - usage.get("decode_s", 0)

    # Open the data-plane socket *before* calling speak(), so no chunk is
    # ever missed to a race between connect and the server's first frame.
    async with websockets.connect(AUDIO_URL, max_size=None) as audio_ws:
        t_speak = time.monotonic()
        result = json.loads(await baby_onit.dispatch("speak", {"text": answer}))
        want_id = result["utterance_id"]
        while True:
            frame = await audio_ws.recv()
            utt_id, _sr, _n, _is_final = struct.unpack_from("<IIIB", frame, 0)
            if utt_id == want_id:
                t_first_chunk = time.monotonic()
                break
    return {
        "llm_ttft_s": ttft,
        "llm_total_s": t_answer - t_task_start,
        "tts_ttfb_s": t_first_chunk - t_speak,
        "e2e_transcript_to_audio_s": t_first_chunk - t_task_start,
        "answer_chars": len(answer or ""),
    }


def summarize(name: str, values: list[float]) -> dict:
    v = sorted(values)
    n = len(v)
    p95 = v[min(n - 1, max(0, round(0.95 * (n - 1))))]
    return {"metric": name, "n": n, "mean_ms": round(1000 * stats.mean(v), 1),
             "p95_ms": round(1000 * p95, 1), "values_ms": [round(1000 * x, 1) for x in v]}


async def main():
    results = {"asr_final_s": [], "llm_ttft_s": [], "tts_ttfb_s": [],
               "transcript_to_audio_s": []}
    log_lines = []

    print("== ASR final latency ==")
    for fname in WAV_FILES:
        pcm = load_pcm_f32(fname)
        for i in range(RUNS_PER_FILE):
            lat = await asr_once(pcm)
            results["asr_final_s"].append(lat)
            rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": f"asr/{fname}/{i}", "asr_final_s": round(lat, 4)}
            log_lines.append(rec)
            print(f"  {fname} run {i}: {lat*1000:.1f} ms")

    print("== LLM TTFT + TTS TTFB (shared VoiceSession) ==")
    cfg = load_config()
    # Mirrors voice_ui/__main__.py: load_config() alone has no MCP server --
    # the TTS MCP url comes from voice.yaml and is injected the same way
    # production wires it, or `speak` would never register as a tool.
    import voice_client.config as vclient_config
    vcfg = vclient_config.load()
    cfg.setdefault("mcp", {}).setdefault("servers", {})["tts"] = {"url": vcfg["tts"]["mcp_url"]}
    cfg["voice"] = True
    async with VoiceSession(cfg) as session:
        for i in range(RUNS_PER_FILE):
            r = await llm_and_tts_once(session)
            results["llm_ttft_s"].append(r["llm_ttft_s"])
            results["tts_ttfb_s"].append(r["tts_ttfb_s"])
            results["transcript_to_audio_s"].append(r["e2e_transcript_to_audio_s"])
            rec = {"t": time.strftime("%Y-%m-%dT%H:%M:%S"), "run": f"llm_tts/{i}", **{k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}}
            log_lines.append(rec)
            print(f"  run {i}: ttft={r['llm_ttft_s']*1000:.1f}ms tts_ttfb={r['tts_ttfb_s']*1000:.1f}ms answer={r['answer_chars']}chars")

    # E2E (speech end -> audio out) = ASR final (speech end -> transcript) +
    # transcript-to-audio (transcript -> audio out), matching the slide's own
    # chained stage definition. The two legs were measured on independent
    # trials (ASR has no dependency on a specific LLM answer), so E2E is
    # composed from their cross product rather than paired 1:1 -- this is the
    # standard way to report a pipeline SLA from its measured stages without
    # requiring every run to carry a real, matching transcript end to end.
    e2e_samples = [a + b for a in results["asr_final_s"] for b in results["transcript_to_audio_s"]]

    summary = [
        summarize("E2E latency (end of speech -> audio out)", e2e_samples),
        summarize("ASR final (end of speech -> transcript)", results["asr_final_s"]),
        summarize("LLM TTFT (transcript -> first token)", results["llm_ttft_s"]),
        summarize("TTS TTFB (speak() call -> first audio chunk)", results["tts_ttfb_s"]),
    ]
    print("\n== SUMMARY ==")
    for s in summary:
        print(f"  {s['metric']}: mean={s['mean_ms']}ms p95={s['p95_ms']}ms n={s['n']}")

    out = {"log": log_lines, "summary": summary}
    with open("bench_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote bench_results.json")


asyncio.run(main())
