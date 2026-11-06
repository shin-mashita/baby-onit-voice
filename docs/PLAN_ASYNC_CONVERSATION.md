# Plan: concurrent conversation and work, and conversational endpointing

Two problems from live use, both architectural.

---

## Problem 1 — the agent cannot talk while it works

> "I asked it to develop a flappy bird app. While it is building, I cannot
> interrupt it with 'What is the status of the build right now?'"

### Why it happens

`handle_turn` awaits `session.ask(text)`, which runs `agent_loop` to completion.
For the flappy-bird task that is **61 seconds** of `speak → bash → write_file →
bash` inside a single turn. During it:

* the turn queue is serialized by `_pump_turns`, so a new utterance waits;
* barge-in **cancels** the agent task, so interrupting *destroys the build*.

So the only two outcomes today are "wait 61 s" or "lose the work". Neither is
what the user wants, which is to **ask a question and have the build continue**.

Note this is not the `speak`-while-working feature, which already works — the
agent does narrate via `speak` during the task. The missing capability is
**accepting input** while working.

### Why the obvious fixes are wrong

* **Don't cancel on barge-in during a long task.** Then the user cannot stop a
  runaway task at all, which is worse.
* **Run two `agent_loop`s over one history.** They would interleave tool calls
  and corrupt each other's message list; `agent_loop` owns `messages` and
  assumes it is the only writer.
* **Make the build a `bash &` background job.** The work is an LLM writing code,
  not a shell command. There is nothing to background.

### Proposed design: a worker sub-agent, and a conversational foreground

Give the agent the ability to **delegate a long task to a background
`agent_loop`** and keep the foreground loop short and responsive.

```
user utterance
      |
      v
foreground agent_loop  (short turns, always interruptible)
      |
      |-- start_task("build a flappy bird game")  --> background agent_loop
      |                                                  |  own messages
      |-- speak("I've started on it")                     |  own data_path
      |                                                   |  writes files
      v                                                   v
   returns in ~2s                                    runs for ~60s
      |                                                   |
user: "what's the status?"                                |
      |                                                   |
      |-- check_task(1) <--------------------------------'
      |     -> {"state":"running","step":"write_file index.html","elapsed":34}
      |-- speak("Still going -- just wrote the HTML canvas")
```

**Three new in-process tools**, registered like the existing eight:

| Tool | Returns | Notes |
|---|---|---|
| `start_task(task)` | `{"task_id": 1, "state": "running"}` | Spawns a detached `agent_loop`; returns immediately |
| `check_task(task_id)` | `{"state", "step", "elapsed_s", "last_output"}` | State comes from the worker's own `on_event` stream |
| `cancel_task(task_id)` | `{"cancelled": bool}` | Explicit, so barge-in no longer has to mean "destroy the work" |

**Implementation points, in dependency order:**

1. **`VoiceSession` gains a task registry** — `dict[int, _BackgroundTask]`, each
   holding an `asyncio.Task`, its own `history` list, a ring buffer of recent
   `on_event` tuples, and a terminal result. Tasks are owned by the session, so
   `__aexit__` cancels them rather than leaking them past shutdown.
2. **Background loops get their own `messages`.** Each `agent_loop` call already
   builds its own list from `session_history`; passing a *separate* history list
   per task is enough for isolation. No locking needed because they never share
   one.
3. **`barge_in()` stops cancelling background work.** It cancels only the
   foreground turn. Killing a build becomes a deliberate `cancel_task`, which is
   also something the user can now *ask* for.
4. **Tool concurrency.** `dispatch()` already offloads blocking tools to a
   thread and runs read-only tools concurrently, so a foreground `check_task`
   does not block a background `write_file`. The one real hazard is two loops
   writing the same file; the worker gets its own subdirectory under
   `data_path`, which also makes "what did it build" answerable.
5. **System prompt.** Add a rule to `VOICE_PROMPT`: anything expected to take
   more than a few seconds goes to `start_task`, and the foreground turn then
   says so out loud. This is the same reasoning as §5.3 — the *harness* should
   not rely on the model volunteering good latency behaviour, so `start_task`
   should also be suggested by a keyword/length heuristic, not prompt alone.

**Verification:** start the flappy-bird build, then ask "what's the status"
within 10 s and assert (a) the question is answered in under 3 s, (b) the build
still completes, (c) the files land. A regression test can use a stub worker
that sleeps, so it needs no LLM.

**Cost and risk.** Two concurrent LLM conversations means roughly double the
token spend during a build, and a background failure is invisible unless
surfaced — `check_task` must report errors, not just progress. The `_ALARMS`
precedent applies: keep task state session-scoped and non-persistent, since a
task that outlives the process would report into nothing.

---

## Problem 2 — a thinking pause ends the turn

> "'UHmmmm I thiink <pause> I want a good flappy bird app' — the TTS interrupts
> me immediately right after the pause."

### Why it happens

`turn.silence_ms` is **700 ms**. Any pause longer than that ends the turn, the
filler fires immediately, and the agent talks over the rest of the sentence.

### What does not work, measured

**Punctuation-based endpointing is out.** The obvious idea is to wait longer when
the transcript looks unfinished. But Qwen terminates *everything*:

```
"Uhmmm I think"                  -> 'Umam, I think.'
"I want a good flappy bird app"  -> 'I want a good Flappy Bird app.'
```

Both get a full stop, so punctuation carries no completeness signal. Discarded.

### Proposed design, cheapest first

1. **Raise `silence_ms` to 900 ms.** One line, helps immediately, costs 200 ms on
   every turn. A stopgap, not the fix.

2. **Continuation merge — the real fix.** Make a premature endpoint *recoverable*
   instead of fatal. After a turn is dispatched, keep a short window
   (`continuation_ms`, ~600 ms) during which, if speech resumes **before the
   agent has produced audio**, the pending turn is withdrawn and its text is
   prepended to the next one. This turns the hard 700 ms decision into a soft
   one, which is what makes it feel conversational: being cut off is survivable.
   It composes with the existing generation counter — a withdrawn turn is just a
   superseded generation.

3. **Hesitation-aware extension.** If the transcript so far ends in a filler or
   dangling function word (`uhm`, `uh`, `so`, `and`, `I think`, `maybe`), extend
   the silence window by ~500 ms. A small lexical list, not a model. Cheap, and
   it targets exactly the reported phrasing.

4. **Delay the filler by ~250 ms.** The filler currently fires the instant the
   turn ends, which is precisely what the user hears talking over them. A short
   delay makes a withdrawn turn silent rather than audible, and costs nothing
   perceptually because the filler exists to cover *LLM* latency, not VAD
   latency.

**Verification:** extend `voice_client/selftest.py` with VAD frame sequences
encoding "speech, 800 ms pause, speech" and assert one turn with merged text;
and "speech, 1500 ms pause, speech" and assert two turns. Both are offline.

---

## Suggested order

1. `no_speech` event — **done**, shipped with this plan.
2. `silence_ms` 900 + filler delay — minutes, immediate relief.
3. Continuation merge — the real conversational fix.
4. Background tasks — the largest change; needs the tool trio, the registry,
   and the barge-in semantics change.

Items 2 and 3 are independent of 4 and worth landing first, because they improve
every turn rather than only long ones.
