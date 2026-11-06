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

#: Words that mean "still thinking", so the silence after them is a pause rather
#: than an ending. Qwen punctuates "uhmm I think" as a finished sentence
#: ("Umam, I think."), so punctuation cannot distinguish this and a lexical
#: check is what remains. Matched against the tail of the live transcript.
HESITATION_TAIL = (
    "uhm", "uhmm", "um", "umm", "uh", "er", "erm", "hmm", "ah",
    "so", "and", "but", "or", "then", "like", "well",
    "i think", "i thiink", "i want", "i need", "maybe", "perhaps",
    "it's", "its", "the", "a", "an", "to", "for", "of", "my",
)


def ends_in_hesitation(text: str) -> bool:
    """True when the transcript trails off mid-thought rather than finishing."""
    t = "".join(c for c in (text or "").lower() if c.isalnum() or c.isspace()).strip()
    if not t:
        return False
    words = t.split()
    return words[-1] in HESITATION_TAIL or " ".join(words[-2:]) in HESITATION_TAIL


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
        hesitation_extra_ms: int = 0,
        peek_text: Callable[[], str] | None = None,
    ):
        self.silence_frames = max(1, round(silence_ms / FRAME_MS))
        self.hesitation_frames = max(0, round(hesitation_extra_ms / FRAME_MS))
        # Supplied by the driver so the detector can consult the live transcript
        # without owning it; returns "" when there is nothing yet.
        self.peek_text = peek_text
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
        needed = self.silence_frames
        if self.hesitation_frames and self.peek_text is not None:
            # Re-checked every frame rather than latched: the transcript is
            # still arriving during the pause, so a tail that looked unfinished
            # at 300 ms may be complete by 800 ms.
            if ends_in_hesitation(self.peek_text()):
                needed += self.hesitation_frames
        if self._silence_run >= needed:
            self._in_utterance = False
            self._silence_run = 0
            if self.on_turn_end:
                self.on_turn_end()
