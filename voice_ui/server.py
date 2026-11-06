"""FastAPI dashboard wrapping the voice client.

Serves the same surface as the tiny_vcm dashboard -- live state, a transcript,
push-to-talk, media buttons -- over the full-duplex agent instead of an intent
classifier.

State reaches the browser over a websocket rather than by polling: the agent can
change things at any moment (a timer fires, a background build finishes), and a
poll interval is either too slow to feel live or too fast to be free.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from .state import LOCK, STATE

log = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).with_name("static")


class Hub:
    """Fans state and conversation events out to every connected browser."""

    def __init__(self) -> None:
        self._clients: set[Any] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self.transcript: list[dict] = []
        self.status = "idle"
        self.partial = ""
        self.driver: Any = None  # set by __main__ for push-to-talk and mute

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    async def attach(self, ws: Any) -> None:
        self._clients.add(ws)
        await ws.send_text(json.dumps(self.payload()))

    def detach(self, ws: Any) -> None:
        self._clients.discard(ws)

    # -- outbound ---------------------------------------------------------

    def payload(self) -> dict:
        return {
            "type": "state",
            "status": self.status,
            "partial": self.partial,
            "transcript": self.transcript[-40:],
            **STATE.snapshot(),
        }

    def push(self) -> None:
        """Broadcast. Safe from any thread -- the agent's tools run in threads."""
        if self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._schedule)
        except RuntimeError:
            pass

    def _schedule(self) -> None:
        asyncio.create_task(self._broadcast())

    async def _broadcast(self) -> None:
        if not self._clients:
            return
        msg = json.dumps(self.payload())
        for ws in list(self._clients):
            try:
                await ws.send_text(msg)
            except Exception:  # noqa: BLE001 — a dead browser must not stop the others
                self.detach(ws)

    # -- inbound ----------------------------------------------------------

    def say(self, role: str, text: str, **extra) -> None:
        """Record a line of conversation and show it."""
        if not (text or "").strip():
            return
        self.transcript.append({"role": role, "text": text.strip(), **extra})
        del self.transcript[:-200]
        self.push()

    def set_status(self, status: str, partial: str = "") -> None:
        self.status, self.partial = status, partial
        self.push()


HUB = Hub()


def build_app(hub: Hub = HUB):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="voice-agent")

    @app.get("/")
    def index():
        return FileResponse(str(STATIC_DIR / "dashboard.html"))

    @app.get("/api/state")
    def api_state():
        return JSONResponse(hub.payload())

    @app.post("/api/say")
    async def api_say(body: dict):
        """Type a command instead of speaking it -- the old app's text tab."""
        text = (body or {}).get("text", "").strip()
        if not text:
            return JSONResponse({"ok": False, "error": "empty"}, status_code=400)
        if hub.driver is None:
            return JSONResponse({"ok": False, "error": "no driver"}, status_code=503)
        # handle_turn records the user line itself (see _patch_driver_for_ui),
        # so logging it here too showed every typed command twice.
        asyncio.create_task(hub.driver.handle_turn(text))
        return JSONResponse({"ok": True})

    @app.post("/api/interrupt")
    async def api_interrupt():
        """Stop the agent talking. The dashboard's equivalent of talking over it."""
        if hub.driver is None:
            return JSONResponse({"ok": False, "error": "no driver"}, status_code=503)
        hub.driver._on_barge_in()
        hub.say("system", "interrupted")
        return JSONResponse({"ok": True})

    @app.post("/api/mute")
    async def api_mute(body: dict | None = None):
        """Mute the microphone, so the dashboard can be used without listening."""
        muted = bool((body or {}).get("muted", True))
        if hub.driver is not None and hub.driver.capture is not None:
            hub.driver.capture.muted = muted
        hub.set_status("muted" if muted else "idle")
        return JSONResponse({"ok": True, "muted": muted})

    @app.post("/api/media/{action}")
    async def api_media(action: str):
        """Media buttons, as in the original dashboard."""
        from .tools import tool_media_control

        out = json.loads(tool_media_control(action))
        hub.push()
        if "error" in out:
            return JSONResponse({"ok": False, **out}, status_code=400)
        return JSONResponse({"ok": True, **out})

    @app.post("/api/lights/{room}/{state}")
    async def api_lights(room: str, state: str):
        from .tools import tool_control_lights

        out = json.loads(tool_control_lights(state, "" if room == "all" else room))
        hub.push()
        return JSONResponse({"ok": "error" not in out, **out})

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        await hub.attach(ws)
        try:
            while True:
                await ws.receive_text()  # browser sends nothing; this detects close
        except WebSocketDisconnect:
            pass
        finally:
            hub.detach(ws)

    return app
