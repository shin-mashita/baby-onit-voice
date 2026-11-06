"""python -m voice_client — run the voice agent.

Phase 1 runs this on the same box as speech_server; Phase 2 runs it on the Pi
with voice.yaml pointing at the laptop. Nothing but the hostnames changes.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

import baby_onit

from .config import load as load_config
from .driver import VoiceDriver

log = logging.getLogger("voice_client")


def _say(kind: str, text: str) -> None:
    """Render agent-loop events for a human watching the terminal."""
    if kind in ("tool", "status", "turn"):
        print(f"  [{kind}] {text}", file=sys.stderr)


async def run(args) -> int:
    vcfg = load_config(args.voice_config)
    cfg = baby_onit.load_config(args.config)
    cfg["voice"] = True  # spoken prompt + always-on tool surface
    if args.data_path:
        cfg["data_path"] = args.data_path
    # The TTS MCP server is where `speak` comes from; without it the agent is mute.
    cfg.setdefault("mcp", {}).setdefault("servers", {})["tts"] = {"url": vcfg["tts"]["mcp_url"]}
    # agent_loop sets the module-level DATA_PATH jail itself, but the directory
    # has to exist before the first write_file lands in it.
    Path(cfg["data_path"]).expanduser().mkdir(parents=True, exist_ok=True)

    async with baby_onit.VoiceSession(cfg, on_event=_say) as session:
        if "speak" not in baby_onit.TOOLS:
            log.error("no `speak` tool: is speech_server running at %s?", vcfg["tts"]["mcp_url"])
            return 2
        async with VoiceDriver(session, vcfg, mic=not args.no_mic) as driver:
            if args.say:
                answer = await driver.handle_turn(" ".join(args.say))
                print(answer or "(cancelled)")
                # One-shot mode would otherwise exit straight into __aexit__,
                # which cancels background tasks -- so a delegated build would
                # never finish and --say could not test one at all.
                await _await_background(session, args.wait_tasks)
                return 0
            print("listening — speak, or Ctrl-C to quit", file=sys.stderr)
            await asyncio.Event().wait()  # driven entirely by mic events
    return 0


async def _await_background(session, timeout: float) -> None:
    """Let delegated tasks finish before a one-shot run tears the session down."""
    if timeout <= 0 or not getattr(session, "tasks", None):
        return
    pending = [t for t in session.tasks.values() if t.state == "running"]
    if not pending:
        return
    log.info("waiting up to %.0fs for %d background task(s)", timeout, len(pending))
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        running = [t for t in session.tasks.values() if t.state == "running"]
        if not running:
            break
        await asyncio.sleep(1.0)
    for t in session.tasks.values():
        log.info("task %d: %s%s", t.task_id, t.state,
                 f" -- {t.error}" if t.error else "")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="voice_client", description=__doc__)
    ap.add_argument("--voice-config", help="voice.yaml (default: ./voice.yaml)")
    ap.add_argument("--config", help="baby-onit config.yaml")
    ap.add_argument("--data-path", help="agent working directory")
    ap.add_argument("--no-mic", action="store_true", help="skip mic capture (for --say)")
    ap.add_argument("--say", nargs="+", help="run one typed turn instead of listening")
    ap.add_argument(
        "--wait-tasks",
        type=float,
        default=300.0,
        help="with --say, seconds to wait for delegated background tasks (0 to skip)",
    )
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
