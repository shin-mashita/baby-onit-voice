"""The tiny_vcm command set, as agent tools.

The original app routed an utterance through MiniLM intent classification and
deterministic slot filling, then called one handler per intent. Here the LLM
picks the tool and supplies the slots, so the classifier and the slot grammar
are gone -- but the ten actions, and the state they mutate, are the same.

Each returns compact JSON. The agent speaks its own summary, so these report
facts rather than prose, and errors use baby-onit's machine-readable codes so a
wrong argument is recoverable in one retry rather than re-tried verbatim.
"""

from __future__ import annotations

import json
import re
import time

from .state import LOCK, ROOMS, STATE, fmt_duration

_UNIT_SECONDS = {"s": 1, "sec": 1, "second": 1, "seconds": 1,
                 "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
                 "h": 3600, "hr": 3600, "hour": 3600, "hours": 3600}
_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "fifteen": 15,
    "twenty": 20, "thirty": 30, "forty": 40, "forty five": 45, "fifty": 50, "sixty": 60,
    "half": 0.5, "quarter": 0.25,
}


def _err(code: str, tool: str, **fields) -> str:
    return json.dumps({"error": code, "tool": tool, **fields}, ensure_ascii=False)


def parse_duration(text: str) -> float | None:
    """'5 minutes', 'an hour and a half', '90 seconds' -> seconds.

    The LLM usually passes something already tidy, but it also passes raw
    phrases, so this accepts both rather than rejecting the harder half.
    """
    t = (text or "").strip().lower()
    if not t:
        return None
    if re.fullmatch(r"\d+(\.\d+)?", t):  # a bare number is minutes, as spoken
        return float(t) * 60
    total, found = 0.0, False
    for num, unit in re.findall(r"(\d+(?:\.\d+)?|[a-z]+(?: [a-z]+)?)\s*(s|sec|secs|seconds?|m|min|mins|minutes?|h|hr|hours?)\b", t):
        mult = _UNIT_SECONDS.get(unit.rstrip("s")) or _UNIT_SECONDS.get(unit)
        if mult is None:
            continue
        try:
            n = float(num)
        except ValueError:
            n = _WORD_NUMBERS.get(num.strip())
            if n is None:
                continue
        total += n * mult
        found = True
    return total if found else None


def parse_clock(text: str) -> str | None:
    """'7am', '6:30', 'quarter past 5 pm', 'noon' -> 'HH:MM' (24h)."""
    t = (text or "").strip().lower()
    if not t:
        return None
    if "noon" in t:
        return "12:00"
    if "midnight" in t:
        return "00:00"
    m = re.search(r"\bquarter past (\d{1,2})", t)
    if m:
        h, mins = int(m.group(1)), 15
    else:
        m = re.search(r"\bhalf past (\d{1,2})", t)
        if m:
            h, mins = int(m.group(1)), 30
        else:
            m = re.search(r"(\d{1,2})(?::(\d{2}))?", t)
            if not m:
                return None
            h, mins = int(m.group(1)), int(m.group(2) or 0)
    if "pm" in t and h != 12:
        h += 12
    elif "am" in t and h == 12:
        h = 0
    if not (0 <= h <= 23 and 0 <= mins <= 59):
        return None
    return f"{h:02d}:{mins:02d}"


def _room(room: str | None) -> str:
    r = (room or "").strip().lower()
    if not r or r in ("all", "everywhere", "the house"):
        return "all"
    for known in ROOMS:
        if known in r or r in known:
            return known
    return r


# --------------------------------------------------------------------- tools

def tool_set_timer(duration: str) -> str:
    """Start a countdown timer. duration like '5 minutes' or '90 seconds'."""
    secs = parse_duration(duration)
    if not secs:
        return _err("E_ARGS", "set_timer", bad=f"cannot parse duration {duration!r}",
                    accepts=["5 minutes", "90 seconds", "1 hour 30 minutes"])
    with LOCK:
        label = fmt_duration(secs)
        STATE.timers.append({"id": STATE.next_id(), "label": label, "total": secs,
                             "ends_at": time.time() + secs, "fired": False})
        STATE.note("set_timer", f"timer set for {label}")
    return json.dumps({"set": label, "seconds": int(secs), "total": len(STATE.timers)})


def tool_set_alarm(when: str, label: str = "alarm") -> str:
    """Set an alarm. when like '7am', '06:30', 'quarter past 5 pm'."""
    hhmm = parse_clock(when)
    if not hhmm:
        return _err("E_ARGS", "set_alarm", bad=f"cannot parse time {when!r}",
                    accepts=["7am", "06:30", "quarter past 5 pm", "noon"])
    with LOCK:
        STATE.alarms.append({"id": STATE.next_id(), "time": hhmm, "label": label,
                             "enabled": True, "fired_on": None})
        STATE.note("set_alarm", f"alarm set for {hhmm}")
    return json.dumps({"set": hhmm, "label": label, "total": len(STATE.alarms)})


def tool_set_thermostat(temperature: float, unit: str = "C", room: str = "") -> str:
    """Set the target temperature."""
    u = (unit or "C").strip().upper()[:1]
    if u not in ("C", "F"):
        return _err("E_ARGS", "set_thermostat", bad=f"unit {unit!r}", accepts=["C", "F"])
    with LOCK:
        STATE.thermostat = {"value": temperature, "unit": u, "room": _room(room) if room else None}
        STATE.note("set_thermostat", f"thermostat -> {temperature}{u}")
    return json.dumps(STATE.thermostat)


def tool_control_lights(state: str, room: str = "") -> str:
    """Turn lights on or off. state is 'on' or 'off'."""
    st = (state or "").strip().lower()
    if st not in ("on", "off"):
        return _err("E_ARGS", "control_lights", bad=f"state {state!r}", accepts=["on", "off"])
    r = _room(room)
    with LOCK:
        if st == "on":
            # A plain "turn on" is a fresh on: full brightness, no colour -- not
            # whatever level adjust_lights last left behind. Same rule as the
            # original app, because anything else surprises people.
            STATE.lights[r] = {"on": True, "level": 100, "color": None}
        else:
            cur = STATE.lights.get(r, {"on": False, "level": 100, "color": None})
            cur["on"] = False
            STATE.lights[r] = cur
        STATE.note("control_lights", f"lights {st} ({r})")
    return json.dumps({"room": r, **STATE.lights[r]})


def tool_adjust_lights(room: str = "", level: int = -1, color: str = "") -> str:
    """Dim or recolour lights. level 0-100, color like 'blue' or 'warm white'."""
    if level < 0 and not color:
        return _err("E_ARGS", "adjust_lights", bad="nothing to change",
                    accepts=["level 0-100", "color name"])
    if level >= 0 and not (0 <= level <= 100):
        return _err("E_ARGS", "adjust_lights", bad=f"level {level}", accepts=["0-100"])
    r = _room(room)
    with LOCK:
        cur = STATE.lights.get(r, {"on": True, "level": 100, "color": None})
        if level >= 0:
            cur["level"], cur["on"] = level, True
        if color:
            cur["color"], cur["on"] = color.strip().lower(), True
        STATE.lights[r] = cur
        STATE.note("adjust_lights", f"lights adjusted ({r})")
    return json.dumps({"room": r, **cur})


def tool_media_control(action: str) -> str:
    """Control playback: play, pause, stop, next, previous, volume_up, volume_down."""
    a = (action or "").strip().lower().replace(" ", "_")
    allowed = ("play", "pause", "stop", "resume", "next", "previous",
               "volume_up", "volume_down")
    if a not in allowed:
        return _err("E_ARGS", "media_control", bad=f"action {action!r}", accepts=list(allowed))
    with LOCK:
        mapped = {"play": "playing", "resume": "playing", "pause": "paused",
                  "stop": "stopped"}.get(a)
        if mapped:
            STATE.media["state"] = mapped
        STATE.media["last_action"] = a
        STATE.note("media_control", f"media: {a}")
    return json.dumps(dict(STATE.media))


def tool_play_music(query: str = "music") -> str:
    """Start playing something."""
    with LOCK:
        STATE.media = {"state": "playing", "track": (query or "music").strip(),
                       "last_action": "play"}
        STATE.note("play_music", f"playing {STATE.media['track']}")
    return json.dumps(dict(STATE.media))


def tool_add_reminder(content: str, when: str = "") -> str:
    """Add a reminder."""
    if not (content or "").strip():
        return _err("E_ARGS", "add_reminder", bad="content is empty")
    with LOCK:
        STATE.reminders.append({"text": content.strip(), "time": parse_clock(when) if when else None})
        STATE.note("reminders", "reminder added")
    return json.dumps({"added": content.strip(), "total": len(STATE.reminders)})


def tool_list_reminders() -> str:
    """List reminders."""
    with LOCK:
        return json.dumps({"reminders": list(STATE.reminders), "total": len(STATE.reminders)})


def tool_call_contact(contact: str, message: str = "") -> str:
    """Place a call or send a message. A stub, as in the original app.

    There is no telephony behind this -- the tiny_vcm handler returned
    "calling <contact>" and nothing more. Kept so the command set matches and so
    the agent reports it honestly rather than implying a call was placed.
    """
    who = (contact or "").strip()
    if not who:
        return _err("E_ARGS", "call_contact", bad="no contact named")
    kind = "message" if message else "call"
    with LOCK:
        STATE.note("call", f"{kind} {who} (not connected: no telephony)")
    return json.dumps({
        "action": kind, "contact": who, **({"message": message} if message else {}),
        "placed": False,
        "note": "no telephony is wired up; say that you cannot actually place it",
    })


def tool_home_status() -> str:
    """Everything at once: lights, timers, alarms, thermostat, media, reminders.

    One call rather than six, because the common question is "what's going on"
    and six round trips would be six seconds of silence.
    """
    return json.dumps(STATE.snapshot(), ensure_ascii=False)


def tool_cancel_timer(timer_id: int = 0) -> str:
    """Cancel a timer by id, or all of them with 0."""
    with LOCK:
        before = len(STATE.timers)
        if timer_id:
            STATE.timers = [t for t in STATE.timers if t["id"] != timer_id]
        else:
            STATE.timers = []
        removed = before - len(STATE.timers)
        STATE.note("set_timer", f"cancelled {removed} timer(s)")
    return json.dumps({"cancelled": removed, "remaining": len(STATE.timers)})


#: (name, description, fn) in the shape baby_onit.tool() expects.
COMMAND_TOOLS = [
    ("set_timer", "Start a countdown timer. duration like '5 minutes' or '90 seconds'.", tool_set_timer),
    ("cancel_timer", "Cancel a timer by id; 0 cancels all.", tool_cancel_timer),
    ("set_alarm", "Set an alarm. when like '7am', '06:30' or 'quarter past 5 pm'.", tool_set_alarm),
    ("set_thermostat", "Set the target temperature. unit 'C' or 'F'.", tool_set_thermostat),
    ("control_lights", "Turn lights on or off. state 'on'/'off', optional room.", tool_control_lights),
    ("adjust_lights", "Dim or recolour lights. level 0-100 and/or a colour.", tool_adjust_lights),
    ("media_control", "play, pause, stop, next, previous, volume_up, volume_down.", tool_media_control),
    ("play_music", "Start playing music or a named track.", tool_play_music),
    ("add_reminder", "Add a reminder.", tool_add_reminder),
    ("list_reminders", "List the reminders.", tool_list_reminders),
    ("call_contact", "Call or message a contact. Not actually connected -- reports back that it cannot place it.", tool_call_contact),
    ("home_status", "Everything at once: lights, timers, alarms, thermostat, media, reminders.", tool_home_status),
]

#: Safe to run concurrently with other reads.
READ_ONLY = {"list_reminders", "home_status"}
#: Asking twice is normal, so exempt from the repeated-call guard.
POLLABLE = {"home_status", "list_reminders"}
