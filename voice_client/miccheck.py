"""Record yourself and see exactly what each ASR model hears.

    python -m voice_client.miccheck "the sentence you are about to say"

Every accuracy figure in ARCHITECTURE.md came from Kokoro-synthesized speech,
which is unnaturally clean and well-articulated -- all three streaming models
score 0.0% WER on it, so it cannot discriminate and it overstates real-world
quality. This records your actual voice through your actual microphone and
transcribes it with each model, which is the only measurement that reflects what
you hear in a live session.
"""

from __future__ import annotations

import argparse
import difflib
import sys
import time

import numpy as np

SR = 16000
ARCHS = ("TINY_STREAMING", "SMALL_STREAMING", "MEDIUM_STREAMING")


def norm(s: str) -> str:
    return " ".join("".join(c for c in s.lower() if c.isalnum() or c.isspace()).split())


def wer(ref: str, hyp: str) -> float:
    r, h = ref.split(), hyp.split()
    if not r:
        return float("nan")
    sm = difflib.SequenceMatcher(None, r, h)
    return sum(max(i2 - i1, j2 - j1) for t, i1, i2, j1, j2 in sm.get_opcodes() if t != "equal") / len(r)


def record(seconds: float, device=None) -> np.ndarray:
    import sounddevice as sd

    print(f"recording {seconds:.0f}s -- speak now", flush=True)
    a = sd.rec(int(SR * seconds), samplerate=SR, channels=1, dtype="float32", device=device)
    for i in range(int(seconds), 0, -1):
        print(f"  {i}...", end=" ", flush=True)
        time.sleep(1)
    sd.wait()
    print("done\n")
    return a[:, 0]


def report_level(a: np.ndarray) -> None:
    rms, peak = float(np.sqrt((a**2).mean())), float(np.abs(a).max())
    dbfs = 20 * np.log10(max(rms, 1e-9))
    clipped = int((np.abs(a) > 0.99).sum())
    near = int((np.abs(a) > 0.95).sum())
    print(f"level: rms {rms:.4f} ({dbfs:.1f} dBFS)  peak {peak:.3f}  "
          f"clipped {clipped}  near-clip {near}")
    # Speech should sit around -30 to -20 dBFS. Too quiet starves the model of
    # signal; clipping destroys the waveform the encoder is reading.
    if peak < 0.05:
        print("  WARNING: very quiet. Raise the input gain:")
        print("           pactl set-source-volume @DEFAULT_SOURCE@ 150%")
    elif clipped or peak > 0.98:
        print(f"  WARNING: clipping ({clipped} samples at the rail, peak {peak:.3f}).")
        print("           The encoder reads a flattened waveform where this happens.")
        print("           Lower the input gain and re-test:")
        print("           pactl set-source-volume @DEFAULT_SOURCE@ 70%")
    elif dbfs < -45:
        print("  WARNING: low level; consider raising the input gain a little.")
    else:
        print("  level looks usable")
    print()


def transcribe(audio: np.ndarray, arch_name: str, keyterms: list[str]) -> str:
    import moonshine_voice as mv
    from moonshine_voice.transcriber import LineCompleted, Transcriber

    path, arch = mv.get_model_for_language("en", getattr(mv.ModelArch, arch_name))
    tr = Transcriber(path, arch, update_interval=0.1)
    if keyterms:
        tr.set_keyterms(keyterms)
    lines: list[str] = []
    st = tr.create_stream()
    st.add_listener(lambda e: lines.append(e.line.text) if isinstance(e, LineCompleted) else None)
    st.start()
    blk = int(SR * 0.1)
    t0 = time.perf_counter()
    for i in range(0, len(audio), blk):
        st.add_audio(audio[i : i + blk].tolist(), sample_rate=SR)
    st.stop()
    tr.close()
    return norm(" ".join(lines)), time.perf_counter() - t0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m voice_client.miccheck", description=__doc__)
    ap.add_argument("reference", nargs="*", help="what you intend to say (for WER)")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--keyterms", nargs="*", default=["flappy bird", "web game"])
    ap.add_argument("--device", default=None, help="input device (default: system default)")
    ap.add_argument("--save", help="write the recording to this wav for re-testing")
    ap.add_argument("--wav", help="re-test a saved wav instead of recording")
    ap.add_argument("--gain", type=float, default=1.0, help="scale the audio before transcribing")
    args = ap.parse_args(argv)

    ref = norm(" ".join(args.reference))
    if ref:
        print(f"reference: {ref!r}\n")

    if args.wav:
        import moonshine_voice as mv

        raw, rate = mv.load_wav_file(args.wav)
        audio = np.asarray(raw, dtype=np.float32)
        if rate != SR:
            audio = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * SR / rate)),
                              np.arange(len(audio)), audio).astype(np.float32)
        print(f"loaded {args.wav}: {len(audio)/SR:.2f}s\n")
    else:
        audio = record(args.seconds, args.device)
    if args.gain != 1.0:
        audio = (audio * args.gain).astype(np.float32)
        print(f"applied gain x{args.gain}\n")
    report_level(audio)

    if args.save:
        import wave

        with wave.open(args.save, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes())
        print(f"saved {args.save}\n")

    dur = len(audio) / SR
    print(f"{'model':18} {'RTF':>6} {'WER':>8}")
    print("-" * 40)
    hyps = {}
    for name in ARCHS:
        try:
            hyp, compute = transcribe(audio, name, args.keyterms)
            hyps[name] = hyp
            w = f"{wer(ref, hyp)*100:7.1f}%" if ref else "      -"
            print(f"{name:18} {compute/dur:6.3f} {w}")
        except Exception as e:  # noqa: BLE001 — one bad model must not hide the rest
            print(f"{name:18} ERROR {type(e).__name__}: {str(e)[:40]}")
    print()
    for name, hyp in hyps.items():
        print(f"{name}:\n  {hyp}")
        if ref and hyp != ref:
            # Show the actual substitutions rather than a bare percentage.
            sm = difflib.SequenceMatcher(None, ref.split(), hyp.split())
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag != "equal":
                    print(f"  {tag}: {' '.join(ref.split()[i1:i2])!r} -> "
                          f"{' '.join(hyp.split()[j1:j2])!r}")
    print("\nIf every model is wrong in the same way, the problem is the audio, not")
    print("the model -- check the level warnings above and your mic placement.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
