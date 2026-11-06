# baby-onit

A tiny, **~1470-line single-file agent harness** distilled from [onit](https://github.com/sibyl-oracles/onit) for teaching how an agent harness actually works. Suitable for tiny devices too!

```
model <--> agent loop <--> tools
   |           |             |
config.yaml  system      in-process
+ keychain   prompt      functions
```

One Python file (`baby_onit.py`), Python 3.10+, no MCP servers, no load
balancer, no verification pass — only the core loop, with every section
labeled by (a) the concept it teaches and (b) the onit file it was distilled
from, so you can go from the toy to the real thing.

## Size

| Metric | Lines |
|---|---|
| Total (`baby_onit.py`) | 1466 |
| Code (non-blank, non-comment) | 1249 |
| Comments | 33 |
| Blank | 184 |

*Recompute with:* `wc -l baby_onit.py` and
`grep -vE '^\s*(#|$)' baby_onit.py | wc -l`. Input uses stdlib `readline`
(up/down history, in-place editing, `⏎` prompt) in ~15 lines instead of the
earlier 412-line raw-mode editor.

## Quickstart

### 1. Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv && source .venv/bin/activate
git clone https://github.com/sibyl-oracles/baby-onit.git
cd baby-onit
uv pip install -e . -U
```

### 2. Configure

```bash
baby-onit setup
```

Pick an endpoint, paste an API key, and choose a data directory. Setup also
prompts for the optional GitHub / Hugging Face / Tavily keys — press enter to
keep what's already stored, or type `d` to delete. Rerunning setup is safe:
blank answers keep every previous value, and the config is merged, not
overwritten.

```bash
baby-onit setup --show   # print config, all endpoints (key + remembered model),
                         # and secrets — no prompts
```

### 3. Use

**Interactive chat:**

```bash
baby-onit chat
```

**Run a single task:**

```bash
baby-onit run "show me all big files in my Downloads folder"
```

```
Here are the largest files in your Downloads folder, sorted by size:

## 🔴 Over 100 MB
| Size | File |
|---|---|
| 764 MB | `oMLX-0.6.3rc2-macos26-27.dmg` |
| 300 MB | `Codex.dmg` |
```

### Common flags

Pass before or after the subcommand. An omitted flag leaves the config value
untouched (defaults below are the built-in values when the config is silent):

| Flag | Purpose | Default |
|---|---|---|
| `--host` | endpoint host | `http://localhost:11434` |
| `--model` | model name | auto-detect from the endpoint |
| `--data-path` | working directory | `~/baby-sandbox` |
| `--no-think` | disable thinking | on |
| `--max-iterations` | cap the agent loop | `-1` (no cap) |
| `--max-context-tokens` | compaction trigger (tokens) | `262144` |
| `--history-budget-tokens` | cap on replayed session history (tokens) | `16000` |
| `--max-tokens` | max output tokens per response | `32768` |
| `--config` | alternate config file | `~/.baby-onit/config.yaml` |

Long sessions are replayed into every request. Some providers (notably
OpenRouter stealth models) fail *gracelessly* — an empty body instead of a
400 — once the payload gets large, so the replayed history is trimmed to
`history_budget_tokens` before the first call, and the request is retried
up to 3× with backoff if the provider still returns an empty response.

A single tool result can also blow the context window on its own: a `grep`
match inside a minified-JSON or base64 file can be megabytes on one line.
Every tool result is therefore clamped to 40 000 characters before it enters
the transcript, and `grep` stops at 50 matches / 20 000 bytes / 300 chars per
line. Context compaction is checked at the top of each loop iteration against
a fresh estimate, so an oversized result is summarised *before* the request
that would carry it is sent.

## Voice agent (full-duplex, Raspberry Pi)

This repo also carries a full-duplex voice agent built on the harness above:
`voice_client` + `voice_ui` run on a **Raspberry Pi** (mic, speaker, VAD,
dashboard); `speech_server` (Qwen3-ASR + Kokoro TTS, both on GPU) runs on a
**separate GPU machine**; the LLM is Claude Sonnet 5 via Portkey, reached from
the Pi over the open internet. Full architecture, measured latency, and the
verbatim system prompt are in [docs/DETAILS.md](docs/DETAILS.md); an
evaluation infographic is at
[docs/eval_infographic.html](docs/eval_infographic.html).

### One-time setup

**GPU machine** (serves ASR + TTS):

```bash
bash scripts/setup.sh --gpu-tts      # base venv + Kokoro TTS on CUDA
bash scripts/setup_qwen.sh           # Qwen3-ASR, separate venv (.venv-qwen)
```

If the GPU machine is reachable from the Pi's LAN directly, skip the tunnel
below and put that machine's real IP in the Pi's `voice.yaml`. If it's
behind NAT (e.g. **WSL2**, whose own IP is not reachable from other LAN
devices), set up a reverse SSH tunnel instead — it works because the GPU
machine can always reach the Pi outbound, even though the Pi can't reach it
inbound:

```bash
ssh-copy-id <pi-user>@<pi-host>                      # passwordless, needed for an unattended tunnel
sed -e 's/PI_USER/<pi-user>/' -e 's/PI_HOST/<pi-host>/' \
  scripts/pi-reverse-tunnel.service.template \
  > ~/.config/systemd/user/pi-reverse-tunnel.service
systemctl --user daemon-reload
systemctl --user enable --now pi-reverse-tunnel.service
loginctl enable-linger "$USER"                       # survives logout/reboot, needs sudo once
```

**Raspberry Pi** (runs the harness + voice client + dashboard):

```bash
scp -r rpi_app <pi-user>@<pi-host>:~/Desktop/
ssh <pi-user>@<pi-host> 'cd ~/Desktop/rpi_app && bash setup.sh'
```

Then on the Pi: edit `~/.baby-onit/voice.yaml` (point `SPEECH_SERVER_HOST`
at the GPU machine, or `127.0.0.1` if using the tunnel above), and run
`baby-onit setup` once to store the LLM API key. Full detail, including the
ALSA fix for USB mic/speaker sample-rate mismatches, is in
[rpi_app/README.md](rpi_app/README.md).

### Bringing it up

**GPU machine** — one script, idempotent (safe to re-run; skips anything
already healthy):

```bash
bash scripts/serve_backends.sh       # starts Qwen3-ASR (vLLM) + speech_server in the background
tail -f logs/*.log                   # watch startup
bash scripts/stop_backends.sh        # stop both; leaves the tunnel running
```

**Raspberry Pi**:

```bash
ssh <pi-user>@<pi-host> 'cd ~/Desktop/rpi_app && bash run.sh'
# dashboard: http://<pi-ip>:7860
```

### Bringing it back up when the network changes

The reverse tunnel is an **outbound** connection from the GPU machine to the
Pi, so a network change on the GPU machine's side (a new WSL2 IP after a
reboot, a different Wi-Fi network, a new router) never breaks it — only a
change to the **Pi's** own address does, and `systemd`'s `Restart=always`
already recovers from everything else (a dropped SSH session, the GPU
machine rebooting). If the Pi's IP changes:

```bash
# on the GPU machine, after editing PI_HOST in the unit file:
systemctl --user edit --full pi-reverse-tunnel.service   # or re-run the sed command above
systemctl --user restart pi-reverse-tunnel.service
```

Everything else — `serve_backends.sh` on the GPU machine, `run.sh` on the
Pi — has no network-identity baked in beyond `127.0.0.1`, so re-running them
after any network change (including this laptop's own IP changing) just
works without edits.

## How it works

The full architecture — the section-by-section map of `baby_onit.py`, the
system diagram, the agent loop, the two client dialects, the security model,
and the secrets handling — is documented in
[docs/baby_onit_features.md](docs/baby_onit_features.md)
(PDF: [docs/baby_onit_features.pdf](docs/baby_onit_features.pdf)).

## License

Apache-2.0 (inherited from onit).
