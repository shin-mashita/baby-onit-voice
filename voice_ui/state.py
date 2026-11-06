"""Shared home state: what the voice commands actually change.

Mirrors the state model of the `tiny_vcm` app in ../tmp/app so the same commands
mean the same things -- timers, alarms, thermostat, lights, media, reminders --
and a dashboard built for one is recognisable on the other.

The difference is upstream of here. tiny_vcm classifies an utterance into one of
eleven intents with MiniLM and fills slots deterministically; this agent hands
the utterance to an LLM which calls these as tools. The state and the surface it
presents are deliberately the same; only the routing changed.

One lock guards the whole thing. Mutations arrive from the agent's tool threads
and reads from the web server's event loop, and the state is small enough that
one lock costs nothing and removes a class of bug.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

LOCK = threading.RLock()

#: Rooms the lights understand, from the original app's config.json.
ROOMS = ("kitchen", "bedroom", "living room")


def _now() -> float:
    return time.time()


def fmt_duration(secs: float) -> str:
    """90 -> '1m 30s'. Spoken back by the agent, so no bare seconds past a minute."""
    secs = int(round(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    parts = [f"{h}h" for _ in (1,) if h] + [f"{m}m" for _ in (1,) if m] + [f"{s}s" for _ in (1,) if s]
    return " ".join(parts) or "0s"


def fmt_remaining(secs: float) -> str:
    return "done" if secs <= 0 else fmt_duration(secs)


@dataclass
class HomeState:
    """Everything the voice commands can change, plus a log of what changed it."""

    lights: dict[str, dict] = field(default_factory=dict)
    timers: list[dict] = field(default_factory=list)
    alarms: list[dict] = field(default_factory=list)
    reminders: list[dict] = field(default_factory=list)
    thermostat: dict | None = None
    media: dict = field(default_factory=lambda: {"state": "stopped", "track": None})
    weather: dict | None = None
    log: list[dict] = field(default_factory=list)
    _seq: int = 0

    def next_id(self) -> int:
        self._seq += 1
        return self._seq

    def note(self, intent: str, msg: str) -> None:
        """Record an action. Newest first, because that is what a dashboard shows."""
        self.log.insert(0, {"ts": time.strftime("%H:%M:%S"), "intent": intent, "msg": msg})
        del self.log[200:]

    # -- views ------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """JSON-ready view for the dashboard. Shape follows the original app."""
        with LOCK:
            now = _now()
            return {
                "lights": [
                    {"room": r, "on": st["on"], "level": st.get("level"), "color": st.get("color")}
                    for r, st in sorted(self.lights.items())
                ],
                "timers": [
                    {
                        "id": t["id"],
                        "label": t["label"],
                        "done": t["ends_at"] <= now,
                        "remaining": fmt_remaining(t["ends_at"] - now),
                    }
                    for t in self.timers
                ],
                "alarms": [
                    {"id": a["id"], "time": a["time"], "label": a.get("label"),
                     "enabled": a["enabled"]}
                    for a in self.alarms
                ],
                "reminders": list(self.reminders),
                "thermostat": self.thermostat,
                "media": dict(self.media),
                "weather": self.weather,
                "log": self.log[:25],
            }

    def due_timers(self) -> list[dict]:
        """Timers that have just elapsed. Marks them so each fires once."""
        with LOCK:
            now, due = _now(), []
            for t in self.timers:
                if not t.get("fired") and t["ends_at"] <= now:
                    t["fired"] = True
                    due.append(t)
            return due

    def due_alarms(self) -> list[dict]:
        """Alarms whose time has arrived today. Fires once per day."""
        with LOCK:
            today, hhmm, due = time.strftime("%Y-%m-%d"), time.strftime("%H:%M"), []
            for a in self.alarms:
                if a["enabled"] and a["time"] == hhmm and a.get("fired_on") != today:
                    a["fired_on"] = today
                    due.append(a)
            return due


#: One process, one home.
STATE = HomeState()
