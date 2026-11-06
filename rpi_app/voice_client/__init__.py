"""Voice agent client: microphone, VAD, turn-taking, playback, barge-in.

Runs on the Raspberry Pi in the deployed topology; everything here is light.
The speech models live in `speech_server` on another host -- see
docs/ARCHITECTURE.md for why, and docs/VOICE_PROTOCOL.md for the wire format.
"""

from __future__ import annotations

__all__ = ["AudioPlayer", "MicCapture", "SileroVad", "TurnDetector", "load_config"]


def __getattr__(name: str):  # lazy: importing config must not pull onnxruntime
    if name == "AudioPlayer":
        from .player import AudioPlayer

        return AudioPlayer
    if name == "MicCapture":
        from .capture import MicCapture

        return MicCapture
    if name == "SileroVad":
        from .vad import SileroVad

        return SileroVad
    if name == "TurnDetector":
        from .turn import TurnDetector

        return TurnDetector
    if name == "load_config":
        from .config import load

        return load
    raise AttributeError(name)
