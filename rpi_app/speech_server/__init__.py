"""Speech server for the full-duplex voice agent.

Implements `docs/VOICE_PROTOCOL.md`: streaming ASR on one port, and TTS split
into an MCP control plane and a websocket data plane on another.

    python -m speech_server --prewarm
"""

from __future__ import annotations

__all__ = [
    "ASR_PORT",
    "TTS_PORT",
    "AsrModel",
    "AudioHub",
    "KokoroTts",
    "MoonshineTts",
    "TtsService",
    "build_asr_app",
    "build_tts_app",
    "frame_chunk",
    "unpack_frame",
]

ASR_PORT = 8771
TTS_PORT = 8772


def __getattr__(name: str):
    # Lazy so that `import speech_server` stays cheap for a client that only
    # wants the port numbers or the frame codec.
    if name in ("AsrModel", "build_asr_app"):
        from . import asr_ws

        return getattr(asr_ws, name)
    if name in ("KokoroTts", "MoonshineTts"):
        from . import backends

        return getattr(backends, name)
    if name in ("AudioHub", "frame_chunk", "unpack_frame"):
        from . import tts_audio_ws

        return getattr(tts_audio_ws, name)
    if name in ("TtsService", "build_tts_app"):
        from . import tts_mcp

        return getattr(tts_mcp, name)
    raise AttributeError(name)
