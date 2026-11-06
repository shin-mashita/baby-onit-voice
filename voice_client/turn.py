"""Turn-taking and barge-in decisions, driven by VAD frames alone.

Deliberately independent of ASR events. Moonshine's `LineCompleted` is a *line*
boundary, not a turn boundary: one 6.6 s utterance emitted two of them
(ARCHITECTURE §5.1), so dispatching per line fires several garbage agent turns
per sentence. Lines update the live transcript; silence decides when the human
has actually stopped talking.
"""

from __future__ import annotations

from collections.abc import Callable

from .vad import FRAME_SAMPLES, SAMPLE_RATE

FRAME_MS = 1000.0 * FRAME_SAMPLES / SAMPLE_RATE  # 32 ms


class TurnDetector:
    """Folds a stream of speech/not-speech frames into turn and barge-in events.

    on_turn_end fires once per utterance, after `silence_ms` of quiet following
    speech. on_barge_in fires when `barge_in_ms` of *sustained* speech arrives
    while the agent is talking -- sustained, because a single frame of speaker
    bleed through imperfect echo cancellation must not cut the agent off.
    """

    def __init__(
        self,
        silence_ms: int = 700,
        barge_in_ms: int = 300,
        on_turn_end: Callable[[], None] | None = None,
        on_barge_in: Callable[[], None] | None = None,
    ):
        self.silence_frames = max(1, round(silence_ms / FRAME_MS))
        self.barge_in_frames = max(1, round(barge_in_ms / FRAME_MS))
        self.on_turn_end = on_turn_end
        self.on_barge_in = on_barge_in
        self.agent_speaking = False
        self._speech_run = 0
        self._silence_run = 0
        self._in_utterance = False

    def reset(self) -> None:
        self._speech_run = self._silence_run = 0
        self._in_utterance = False

    def feed(self, is_speech: bool) -> None:
        """Advance the state machine by one frame."""
        if is_speech:
            self._speech_run += 1
            self._silence_run = 0
            if self.agent_speaking:
                # While the agent talks, speech is a candidate interruption, not
                # the start of a turn -- the turn only begins once we cut it off.
                if self._speech_run == self.barge_in_frames and self.on_barge_in:
                    self.on_barge_in()
                return
            self._in_utterance = True
            return

        self._speech_run = 0
        if not self._in_utterance:
            return
        self._silence_run += 1
        if self._silence_run >= self.silence_frames:
            self._in_utterance = False
            self._silence_run = 0
            if self.on_turn_end:
                self.on_turn_end()
