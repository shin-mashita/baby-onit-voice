"""voice.yaml loading, with the defaults documented in docs/VOICE_PROTOCOL.md §5."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "asr": {
        "url": "ws://127.0.0.1:8771/asr",
        "language": "en",
        # Without keyterms "flappy" transcribes as "floppy"/"flashlight" (measured).
        "keyterms": ["flappy bird", "web game"],
    },
    "tts": {
        "mcp_url": "http://127.0.0.1:8772/mcp",
        "audio_url": "ws://127.0.0.1:8772/audio",
    },
    "turn": {
        # Trailing silence that ends a user turn. ASR line events are NOT turn
        # boundaries -- one utterance routinely emits several (ARCHITECTURE §5.1).
        # 900, not 700: at 700 a thinking pause ("uhmm I think <pause> I want...")
        # ended the turn and the agent talked over the rest of the sentence.
        "silence_ms": 900,
        # After a turn is dispatched, speech resuming inside this window -- and
        # before the agent has made a sound -- withdraws it and merges the text
        # into the next one. This is what makes a premature endpoint survivable
        # rather than fatal, which is most of what "conversational" means here.
        "continuation_ms": 700,
        # Hold the filler briefly so a withdrawn turn stays silent. Costs nothing
        # perceptually: the filler exists to cover LLM latency, not VAD latency.
        "filler_delay_ms": 250,
        # Extend the silence window when the transcript trails off mid-thought.
        # Qwen punctuates everything as complete ("Umam, I think."), so
        # punctuation cannot signal this and a lexical check is what is left.
        "hesitation_extra_ms": 500,
        # Sustained speech required to interrupt the agent. Guards against the
        # mic hearing the speakers; see the AEC discussion in ARCHITECTURE §5.2.
        "barge_in_ms": 300,
        "vad_threshold": 0.5,
        # The VAD finds the boundary before ASR has the words. Wait this long
        # for text rather than dropping the utterance (see _resolve_turn).
        "transcript_grace_ms": 4000,
        # Let a trailing ASR line land before dispatching, so a turn is not
        # built from a mid-word partial.
        "line_settle_ms": 400,
    },
    "audio": {"input_device": None, "output_device": None, "aec": True},
    # Append-only JSONL of turns, answers and barge-ins. Set to null to disable.
    "transcript_log": "~/baby-sandbox/voice_turns.jsonl",
    "filler": {
        # Spoken by the harness the instant a turn ends, before the first LLM
        # token exists. Deterministic, so it cannot be skipped under load.
        "enabled": True,
        "phrases": ["Sure, let me look at that.", "On it.", "Let me check."],
    },
}


def _merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for k, v in (extra or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load(path: str | Path | None = None) -> dict:
    """DEFAULTS <- voice.yaml, deep-merged. Reads the `voice:` key if present."""
    cfg = dict(DEFAULTS)
    for cand in (path, "voice.yaml", Path.home() / ".baby-onit" / "voice.yaml"):
        if cand and Path(cand).expanduser().is_file():
            loaded = yaml.safe_load(Path(cand).expanduser().read_text()) or {}
            return _merge(cfg, loaded.get("voice", loaded))
    return cfg
