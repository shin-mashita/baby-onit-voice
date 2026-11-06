"""python -m voice_ui — the voice agent with a dashboard.

Runs the voice client and a FastAPI dashboard in one event loop, so the browser
sees state the moment a tool changes it rather than on a poll interval.

    python -m speech_server --prewarm      # ASR + TTS, first
    python -m voice_ui                     # then this; open http://127.0.0.1:7860
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import sys
from pathlib import Path

import baby_onit

from voice_client.config import load as load_voice_config
from voice_client.driver import VoiceDriver

from .server import HUB, build_app
from .state import STATE
from .tools import COMMAND_TOOLS, POLLABLE, READ_ONLY

log = logging.getLogger("voice_ui")


def register_command_tools() -> None:
    """Graft the tiny_vcm command set onto the agent's tool registry.

    `set_alarm` is replaced deliberately: baby_onit ships a session-local one
    that the dashboard cannot see, and two alarm stores would disagree on screen.
    """
    for name, desc, fn in COMMAND_TOOLS:
        baby_onit.tool(name, desc, fn)
    baby_onit.READ_ONLY.update(READ_ONLY)
    baby_onit.POLLABLE.update(POLLABLE)
    # The in-process alarm list is now dead: the dashboard reads STATE.
    baby_onit.TOOLS.pop("list_alarms", None)


def _wire_events() -> None:
    """Push to the browser whenever a tool changes something."""
    original = baby_onit.dispatch

    async def dispatch(name: str, args: dict) -> str:
        out = await original(name, args)
        HUB.push()  # cheap, and a tool may have changed anything
        return out

    baby_onit.dispatch = dispatch


async def _timer_watcher(driver: VoiceDriver, period: float = 1.0) -> None:
    """Speak when a timer or alarm comes due, and keep the countdown ticking."""
    while True:
        await asyncio.sleep(period)
        fired = False
        for t in STATE.due_timers():
            STATE.note("set_timer", f"timer finished ({t['label']})")
            HUB.say("system", f"Timer finished: {t['label']}")
            with contextlib.suppress(Exception):
                await driver.speak(f"Your {t['label']} timer is up.")
            fired = True
        for a in STATE.due_alarms():
            STATE.note("set_alarm", f"alarm {a['time']}")
            HUB.say("system", f"Alarm: {a['time']}")
            with contextlib.suppress(Exception):
                await driver.speak(f"It's {a['time']}. Your alarm is going off.")
            fired = True
        # Timers show a live countdown, so refresh even when nothing fired.
        if fired or STATE.timers:
            HUB.push()


def _on_event(kind: str, text: str) -> None:
    """Mirror agent-loop events into the dashboard."""
    if kind == "tool":
        HUB.say("system", text.split("(")[0], tool=True)
    elif kind == "status":
        HUB.set_status("thinking" if "think" in text else HUB.status, HUB.partial)


async def run(args: argparse.Namespace) -> int:
    import uvicorn

    vcfg = load_voice_config(args.voice_config)
    cfg = baby_onit.load_config(args.config)
    cfg["voice"] = True
    cfg.setdefault("mcp", {}).setdefault("servers", {})["tts"] = {"url": vcfg["tts"]["mcp_url"]}
    Path(cfg["data_path"]).expanduser().mkdir(parents=True, exist_ok=True)

    register_command_tools()
    _wire_events()
    HUB.bind_loop(asyncio.get_running_loop())

    async with baby_onit.VoiceSession(cfg, on_event=_on_event) as session:
        if "speak" not in baby_onit.TOOLS:
            log.error("no `speak` tool: is speech_server running at %s?", vcfg["tts"]["mcp_url"])
            return 2

        async with VoiceDriver(session, vcfg, mic=not args.no_mic) as driver:
            HUB.driver = driver
            _patch_driver_for_ui(driver)
            HUB.set_status("muted" if args.no_mic else "listening")

            server = uvicorn.Server(uvicorn.Config(
                build_app(), host=args.host, port=args.port,
                log_level="warning", access_log=False,
            ))
            watcher = asyncio.create_task(_timer_watcher(driver), name="timers")
            log.info("dashboard: http://%s:%d", args.host.replace("0.0.0.0", "127.0.0.1"), args.port)
            try:
                await server.serve()
            finally:
                watcher.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await watcher
    return 0


def _patch_driver_for_ui(driver: VoiceDriver) -> None:
    """Mirror the driver's own events into the dashboard, without forking it."""
    hand, speak = driver.handle_turn, driver.speak

    async def handle_turn(text: str):
        HUB.say("user", text)
        HUB.set_status("thinking")
        out = await hand(text)
        HUB.set_status("listening")
        if out:
            HUB.say("agent", out)
        return out

    async def speak_(text: str):
        HUB.set_status("speaking", "")
        return await speak(text)

    driver.handle_turn = handle_turn
    driver.speak = speak_

    # Live partials: show what the ASR is hearing as it hears it.
    pump = driver._pump_asr

    async def pump_asr():
        import json as _json

        async for msg in driver._asr:
            if isinstance(msg, bytes):
                continue
            ev = _json.loads(msg)
            if ev.get("type") == "partial":
                driver.partial = ev.get("text", "")
                HUB.set_status(HUB.status, driver.partial)
            elif ev.get("type") == "line":
                driver.partial = ""
                if ev.get("text"):
                    driver.transcript.append(ev["text"])
                HUB.set_status(HUB.status, "")
            elif ev.get("type") == "no_speech":
                HUB.set_status(HUB.status, "")
            elif ev.get("type") == "error":
                HUB.say("system", f"asr error: {ev.get('message')}")
        del pump

    driver._pump_asr = pump_asr


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m voice_ui", description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--voice-config", help="voice.yaml")
    ap.add_argument("--config", help="baby-onit config.yaml")
    ap.add_argument("--no-mic", action="store_true", help="dashboard only; type commands")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )
    try:
        raise SystemExit(asyncio.run(run(args)))
    except KeyboardInterrupt:
        print("\nbye", file=sys.stderr)


if __name__ == "__main__":
    main()
