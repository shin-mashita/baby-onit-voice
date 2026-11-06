# System details

A reference for how the running system fits together: topology, the exact
prompt the model sees, the full mic→speaker path, how context is managed
turn to turn, and which process serves which component. Each section cites
the source file it was pulled from, since this is meant to stay accurate as
the code changes rather than be a point-in-time description.

## 1. System diagram

```mermaid
flowchart TB
    subgraph client["voice_client (Pi 5 target; this laptop in Phase 1)"]
        MIC["mic"] --> VAD["Silero VAD v5\n(onnxruntime, local)"]
        VAD --> TURN["TurnDetector\nsilence_ms / hesitation / continuation_ms"]
        TURN -->|"turn text"| DRIVER["VoiceDriver"]
        PLAYER["AudioPlayer\n(speaker)"] 
        VAD -->|"sustained speech\nwhile agent talks"| BARGE["barge_in()"]
        BARGE -->|"flush, mute, cancel"| DRIVER
        BARGE -->|"flush()"| PLAYER
    end

    subgraph speech["speech_server (laptop, 2 ports)"]
        ASRWS["/asr websocket\n:8771"]
        TTSMCP["/mcp MCP\n:8772 (control)"]
        TTSWS["/audio websocket\n:8772 (data)"]
        ASRENGINE["Moonshine MEDIUM_STREAMING\n(default, CPU)\nor Qwen3-ASR-1.7B (vLLM, GPU)"]
        TTSENGINE["Kokoro voice via\nMoonshine Voice (CPU)\nor kokoro-onnx (GPU)"]
        ASRWS --> ASRENGINE
        TTSMCP --> TTSENGINE
        TTSENGINE --> TTSWS
    end

    subgraph harness["baby_onit.py VoiceSession"]
        LOOP["agent_loop\n(messages, compaction, tool dispatch)"]
        TOOLS["TOOLS registry\n23 tools: files, web, home commands,\nbackground tasks, speak/stop_speaking"]
        LOOP <--> TOOLS
    end

    LLM["Portkey gateway\n-> Claude (Anthropic, via Vertex AI)\nremote, HTTPS"]

    DRIVER -- "16kHz PCM, streamed" --> ASRWS
    ASRWS -- "partial / line / no_speech" --> DRIVER
    DRIVER -- "finalized turn text" --> LOOP
    LOOP -- "chat completions" --> LLM
    LLM -- "tokens, tool_calls" --> LOOP
    TOOLS -- "speak(text) tool call" --> TTSMCP
    TTSWS -- "framed PCM" --> DRIVER
    DRIVER --> PLAYER
```

## 2. System prompt, verbatim

Two constants in `baby_onit.py`. `SYSTEM_PROMPT` is always sent; `VOICE_PROMPT`
is appended only when `cfg["voice"]` is set (`build_system_prompt()`,
`baby_onit.py:567`).

### `SYSTEM_PROMPT` (`baby_onit.py:499`)

```
You are baby-onit, a helpful AI agent that completes tasks by calling tools.

Working directory: {data_path} (today: {today}). Every file path you write starts here.

Tool routing:
- Files in the working directory: read_file, grep, local_search.
- Public facts, news, current events: web search; fetch pages for details.
- Anything that can change (prices, versions, dates) needs a tool result, never memory. Prefer the primary source over anyone summarizing it.
- Finish with a concise answer; cite the files or URLs you used.

Credentials (pre-wired by the harness; never print, echo, or embed a token):
- GitHub: git clone/pull/push with plain https://github.com/... URLs just works — the harness authenticates git via GIT_ASKPASS. Never hunt for a token or put one in a URL.
- Hugging Face: HF_TOKEN rides in the bash environment for model-hub access. Use `hf download ...` or `huggingface-cli download ...`; a 401 on a gated repo means accept the terms on the model page, then retry.
```

### `VOICE_PROMPT`, appended in voice mode (`baby_onit.py:522`)

```
You are speaking out loud. Everything you write is sent to a speech synthesizer and heard by a person, so:
- Write what a person would say, not what they would read. No markdown, no bullet lists, no tables, no code blocks, no emoji, no URLs read out character by character. Numbers and units in words when that is how you would say them.
- Short sentences. The synthesizer speaks one sentence at a time, so a long clause is a long silence.
- Answer the question first, in one or two sentences. Offer detail only if it was asked for.

Long jobs go to the background. Anything that will take more than a few seconds -- writing an app, a large refactor -- goes to `start_task`, which returns an id at once. Never do a long job in the foreground: the person cannot talk to you while you do, which is the single worst thing you can do to a conversation.
- "Long" includes starting or serving anything: a dev server, a Flask app, a watcher. Those never finish, so running one in the foreground hangs the conversation until the tool times out.
- Call `start_task` once per request. It returns an id; that means the work is under way. Do not call it again for the same job, and do not call it again because you are unsure -- call `check_task` instead.
- After starting one, say so out loud and stop. Do not keep working in the foreground to "help" -- the task is already doing it.
- When asked how it is going, `check_task` and report the step it is on.
- Stop one only when asked, with `cancel_task`.

Speaking is a tool call: `speak`. It returns immediately -- it does not wait for the audio to finish -- so you can keep working while your words play.
- Call `speak` as your FIRST action on any task that will take more than a moment, before any other tool. Silence reads as a broken agent.
- While a long task runs, call `speak` again at each step to say what you are doing now. Keep each one to a sentence.
- Your final answer is spoken too, so end in plain prose.

A voice activity detector is listening the whole time you talk. If the person starts speaking, your speech is cut off mid-sentence and the rest of your reply is discarded. That is normal and expected -- it is not an error, and you should not apologise for it or try to resume where you left off. Treat whatever they said as the next thing to answer, and assume they only heard the part that played before they interrupted.
```

Everything in this second block is *harness-enforced engineering, restated as
instruction* — not the only line of defense. `speak`-first is backed by the
deterministic filler (§3 below); barge-in is backed by the generation counter
in `voice_client/driver.py`, not by the model remembering to behave. The
prompt is reinforcement, not the mechanism.

## 3. The mic-to-speaker path

**Mic → VAD.** `voice_client/capture.py` opens the input device at 16kHz in
512-sample blocks — exactly one Silero v5 frame, so no rebuffering happens
between capture and scoring. Every frame is run through `SileroVad.probability()`
on the capture thread itself (0.08ms measured), not deferred to the event loop,
because barge-in latency is bounded by how fast a frame can be scored.

**VAD → turn boundary.** `TurnDetector` (`voice_client/turn.py`) never
consults ASR text for *when* a turn ends — only the VAD's silence run. ASR
`line` events are not turn boundaries: one utterance can emit several
(measured, see ARCHITECTURE.md §5.1), so dispatching per-line would fire
nonsense turns mid-sentence. The boundary is `silence_ms` (900ms default),
extended by `hesitation_extra_ms` while the live transcript ends in a filler
word (`uhm`, `so`, `I think`, …), and recoverable for `continuation_ms` after
dispatch — if the human resumes before the agent has made a sound, the
dispatched turn is withdrawn and merged into the next one rather than treated
as final.

**Turn text → ASR.** PCM streams continuously to `speech_server`'s
`/asr` websocket. A streaming engine (Moonshine) emits `partial`/`line` events
as it goes; a non-streaming engine (Qwen3-ASR) emits nothing until the client
sends `{"type":"finalize"}` at the VAD boundary, at which point it transcribes
the whole buffered utterance in one call. Either way, the *decision* that the
turn is over is always the client's VAD, never the ASR engine's.

**Turn text → harness.** `VoiceDriver._resolve_turn` waits up to
`transcript_grace_ms` (4000ms) for the text to actually arrive — the VAD
boundary and the ASR's own latency are different events, and reading the
transcript synchronously at the VAD boundary silently dropped utterances when
the ASR hadn't caught up yet (fixed; see ARCHITECTURE.md §5). Once text exists,
`handle_turn` hands it to `VoiceSession.ask()`, which runs `agent_loop`.

**Harness → speaker.** The agent calls `speak` (an MCP tool) the instant it
starts working — enforced by a deterministic local filler spoken *before* the
first LLM token exists, not left to the model to remember. `speak` enqueues
server-side and returns in ~2ms; it never waits for audio. PCM streams back on
a *separate* websocket (`/audio`, not the MCP control channel), framed by
`utterance_id`/`is_final`, straight to `AudioPlayer`.

**Barge-in.** The VAD alone decides this — not ASR, not the LLM. Sustained
speech (`barge_in_ms`, 300ms) while `player.is_playing` is true triggers, in
order: (1) `player.flush()` — local, ~0.05ms; (2) `_audio_muted = True`, so
frames already in flight from the server are dropped rather than refilling the
just-emptied queue; (3) `{"type":"flush"}` sent to the TTS server; (4)
`session.barge_in()`, which cancels the in-flight `agent_loop` task — but
**not** any background task (§4). A generation counter is bumped on every
barge-in; any in-flight work (a queued turn, a resolver waiting on ASR, an
answer about to be spoken) checks it before acting and discards itself if
superseded, rather than surfacing late and out of order.

## 4. Context engineering

**One system message, assembled once per turn.** `build_system_prompt(cfg)`
renders `SYSTEM_PROMPT` (+ `VOICE_PROMPT` in voice mode) with `{data_path}` and
`{today}` substituted — no RAG, no per-tool prompt fragments, no templating
engine. It is `messages[0]` and never mutated mid-loop.

**Fresh by default, replay on request.** `cfg["session_memory"]` defaults to
`False`: `chat` clears `session_history.jsonl` on startup rather than ignoring
it, so a new session never silently inherits an old conversation, and every
turn's `history_budget_tokens` isn't spent on stale context. `chat --resume`
opts back in for one session. `VoiceSession.history` is in-memory only and
never touches disk at all.

**Background tasks get an isolated history.** `start_background()`
(`baby_onit.py`) gives each delegated `agent_loop` its own `history: list`,
never the foreground's. Two conversations never interleave their message
lists, and a worker cannot call `start_task`/`check_task`/`cancel_task` itself
— `cfg["background"]=True` strips those three tools from its own `call_tools`,
closing the recursion a worker delegating to itself would otherwise open.

**Compaction, pre-emptive not reactive** (`agent_loop`, `baby_onit.py:1302`).
Checked at the top of every iteration, *before* the next request is sent —
checking after would always be one iteration stale, since the estimate can't
see tool results appended since the last check. Trigger: estimated tokens
`> 0.85 * max_context_tokens`. On trigger, a second `provider.chat()` call
summarizes everything except the system message into "task, findings, files
touched, what remains"; the transcript is then replaced with that summary plus
a reminder of which tools exist, so the model doesn't need to re-derive what
it already knows is available. `_fit_messages()` is the last-resort hard trim
if compaction alone doesn't fit.

**Replayed history decays, not the live turn.** `build_session_messages()`
keeps the most recent `HISTORY_KEEP_FULL` answers verbatim and truncates older
ones to their opening — a long session degrades gracefully rather than
overflowing, and only applies to *replayed* history (`--resume`), never to the
turn in progress.

**The P5 tool-surface guard is bypassed in voice/background mode.**
Stock baby-onit hides the entire tool surface on iteration 1 unless the task
text matches a keyword list, to stop plain Q&A from over-triggering tool use.
Voice and background turns always see the full tool surface instead — the
guard's premise (a human typing a plain question) doesn't hold when `speak`
itself is a required tool call, and the stock behavior silently answered
`"Set an alarm at 7am"` with zero tools (measured regression, now fixed).

## 5. Components, how served

| Component | Transport | Tools | Model used | Served via | Where |
|---|---|---|---|---|---|
| **ASR** (default) | WebSocket, `ws://host:8771/asr`, binary f32 PCM in / JSON events out | `start`, `reset`, `finalize`, `stop` control messages (VOICE_PROTOCOL.md §1) | Moonshine `MEDIUM_STREAMING` (0.0% WER measured on real speech) | In-process inside `speech_server`, via the `moonshine-voice` Python package | Laptop (CPU) |
| **ASR** (optional) | HTTP, `/v1/audio/transcriptions`, multipart WAV | same control messages; server buffers and POSTs on `finalize` | Qwen3-ASR-1.7B | `vllm.entrypoints.openai.api_server`, separate venv (`.venv-qwen`) | Laptop (GPU, ~5.8GB VRAM) |
| **TTS** (default) | MCP (control) `http://host:8772/mcp` + WebSocket (data) `ws://host:8772/audio` | `speak`, `stop_speaking`, `is_speaking` (MCP tools, exposed to the LLM) | Kokoro voice (`kokoro_af_heart`) via Moonshine Voice | In-process inside `speech_server` | Laptop (CPU, RTF 0.25) or GPU (RTF 0.047) |
| **TTS** (GPU path) | same MCP + WebSocket | same three tools | Kokoro-82M | `kokoro-onnx` on CUDA (`onnxruntime-gpu==1.22.0` pinned) | Laptop (GPU, ~620MB VRAM) |
| **LLM** | HTTPS, OpenAI-compatible `/chat/completions` | the 23-tool `TOOLS` registry (files, web, home commands, `speak`/`stop_speaking`, `start_task`/`check_task`/`cancel_task`) | `claude-sonnet-5` (model id configurable; routed by Portkey's `@integration/model` slug) | Portkey gateway → Claude on Vertex AI | **Remote** — the only non-local component |
| **VAD** | none — a function call, not a service | none; not tool-callable | Silero VAD v5 | `onnxruntime` `InferenceSession`, loaded directly in `voice_client` | Same process as the mic (Pi 5 target; laptop in Phase 1) |
| **Harness** | in-process | owns all of the above as tools/sessions | — (orchestration only) | `baby_onit.VoiceSession` + `voice_client.VoiceDriver`, one Python process | Pi 5 target; laptop in Phase 1 |

Full measured numbers, the structural reason Qwen can't stream, and every
silent-fallback trap hit while building this are in
[ARCHITECTURE.md](ARCHITECTURE.md) — this document is the map, that one is the
evidence.
