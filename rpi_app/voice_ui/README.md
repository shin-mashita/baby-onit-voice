# voice_ui

A web dashboard over the full-duplex voice agent, keeping the command set of the
`tiny_vcm` app in `../tmp/app`.

```bash
python -m speech_server --prewarm       # ASR + TTS
python -m voice_ui                      # dashboard + microphone
python -m voice_ui --no-mic             # dashboard only, type commands
# http://127.0.0.1:7860
```

## What carried over, and what changed

`tiny_vcm` classified an utterance into one of eleven intents with MiniLM and
filled slots with a deterministic grammar, then called one handler per intent.
Here an LLM picks the tool and supplies the slots. **The ten actions and the
state they mutate are unchanged** — `voice_ui/state.py` mirrors the original
`STATE`, so the dashboard shows the same things.

| Original intent | Here |
|---|---|
| `set_timer` | `set_timer`, `cancel_timer` |
| `set_alarm` | `set_alarm` |
| `set_thermostat` | `set_thermostat` |
| `control_lights` | `control_lights` |
| `adjust_lights` | `adjust_lights` |
| `media_control` | `media_control` |
| `play_music` | `play_music` |
| `reminders` | `add_reminder`, `list_reminders` |
| `ask_question` | answered directly, with `web_search`/`fetch_content` |
| `call` | `call_contact` — a stub, as it was there |
| `out_of_scope` | no longer a class; the agent just answers |

Two things the original could not do, which come free with an LLM: it holds a
conversation rather than matching one command at a time, and long jobs go to
`start_task` and run in the background while you keep talking.

`call_contact` reports `"placed": false` and tells the agent to say it cannot
actually place a call. The original returned `"calling <contact>"` with no
telephony behind it; this is the same non-capability, stated rather than implied.

## Surface

| Route | |
|---|---|
| `GET /` | the dashboard |
| `GET /api/state` | current state + transcript |
| `WS /ws` | state pushed on every change |
| `POST /api/say` | type a command (the old app's text tab) |
| `POST /api/interrupt` | stop the agent talking |
| `POST /api/mute` | mute the microphone |
| `POST /api/media/{action}` | media buttons |
| `POST /api/lights/{room}/{state}` | light buttons |

State is **pushed over a websocket, not polled**: the agent changes things at
moments the browser cannot predict — a timer fires, a background build finishes —
and a poll interval is either too slow to feel live or too fast to be free.

## Notes

* `set_alarm` **replaces** baby-onit's built-in one. That version keeps alarms in
  a module-level list the dashboard cannot see, and two alarm stores would
  disagree on screen.
* `home_status` returns everything in one call. The common question is "what's
  going on", and six separate tool calls would be six seconds of silence.
* Timers and alarms are spoken when they come due, by a watcher in `__main__`.
* State is session-scoped and not persisted, matching the original.
