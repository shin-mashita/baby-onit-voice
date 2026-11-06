"""baby-onit — a tiny, single-file agent harness distilled from onit
(https://github.com/sibyl-oracles/onit, ~60k-LOC). Keeps only the core loop.
each section header (S1..S9) names the concept it teaches and the onit file it
came from. Usage: baby-onit setup | chat | run "task"
"""

# S0. Imports — onit splits these across 104 files; a teaching harness needs one file.
from __future__ import annotations

import argparse
import asyncio
import contextlib
import getpass
import inspect
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

# S1. Config loader — every knob lives in one deep-merged dict (onit: pydantic over defaults).
DEFAULTS = {
    "serving": {
        "provider": "auto",  # auto|ollama|vllm|sglang|openrouter|vercel|openai|claude
        "host": "http://localhost:11434",
        "model": "",  # blank = auto-detect via client.models.list()
        "think": True,  # route reasoning to a separate field, not into the answer
        "max_tokens": 32768,
        "max_chat_iterations": -1,  # -1 = no turn cap (onit's default)
        "max_context_tokens": 262144,  # compaction trigger threshold
        "history_budget_tokens": 8000,  # hard cap on replayed history (pre-call trim)
        "temperature": 0.6,
        "top_p": 0.95,
        # Gateways (Portkey, LiteLLM) route on headers, not on the URL path.
        "extra_headers": {},
        # A gateway host hides the upstream model, so the provider quirks keyed
        # on host substrings below cannot see what they are talking to. Name the
        # upstream family here when fronting one: anthropic|openai|generic.
        "upstream": "auto",
    },
    "data_path": "~/baby-sandbox",  # the agent's jail root (cwd for all tools)
    "enforce_jail": True,  # confine file tools to data_path (onit: _validate_*_path)
    "block_dangerous": True,  # refuse NEVER_ASK commands in bash (onit: command_policy)
    "voice": False,  # voice mode: spoken-style prompt + always-on tool surface
    # Replay prior turns from session_history.jsonl on `chat` startup. Off by
    # default: a new session starting inside an old conversation is surprising,
    # and stale turns silently consume history_budget_tokens of every request.
    # `chat --resume` opts in for one session; `\reset` clears mid-session.
    "session_memory": False,
    "mcp": {"servers": {}},  # name -> {url, headers}; streamable-HTTP MCP servers
}


def load_config(path: str | None = None) -> dict:
    """defaults <- ~/.baby-onit/config.yaml <- explicit path (deep-merged)."""

    def merge(base: dict, extra: dict) -> dict:
        out = dict(base)
        for k, v in (extra or {}).items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out

    cfg = dict(DEFAULTS)
    for cand in (path, str(Path.home() / ".baby-onit" / "config.yaml")):
        if cand and Path(cand).expanduser().is_file():
            cfg = merge(cfg, yaml.safe_load(Path(cand).expanduser().read_text()) or {})
            break
    cfg["data_path"] = str(Path(cfg["data_path"]).expanduser())
    return cfg


def _load_yaml(p: Path) -> dict:
    """yaml.safe_load of a file, or {} when it is missing or corrupt."""
    try:
        return yaml.safe_load(Path(p).read_text()) or {}
    except Exception:  # noqa: BLE001 — missing/corrupt config falls back to defaults
        return {}


# S2. Keychain / secrets — secrets never live in config.yaml: env var > keychain > 0600 file.
SERVICE = "baby-onit"
SECRETS_FILE = Path.home() / ".baby-onit" / "secrets.yaml"
MODELS_FILE = Path.home() / ".baby-onit" / "models.yaml"
SECRET_ENV = {
    "github_token": "GITHUB_TOKEN",
    "huggingface_token": "HF_TOKEN",
    "ollama_api_key": "OLLAMA_API_KEY",
    "tavily_api_key": "TAVILY_API_KEY",
}
SECRET_NAMES = tuple(SECRET_ENV)


def _keyring(name: str, value: str | None = None) -> str | bool | None:
    """Read (value=None) or write a secret in the OS keychain; None/False on error."""
    try:
        import keyring

        return (
            keyring.get_password(SERVICE, name)
            if value is None
            else (keyring.set_password(SERVICE, name, value) or True)
        )
    except Exception:  # noqa: BLE001 — keyring backend can be missing or locked
        return None if value is None else False


def _file_secrets() -> dict:
    """The 0600 fallback file's contents ({} when missing/corrupt)."""
    return _load_yaml(SECRETS_FILE)


def _file_set(name: str, value: str | None) -> None:
    """Write (or delete, when value is None) a secret in the 0600 fallback file."""
    data = _file_secrets()
    data.pop(name, None) if value is None else data.update({name: value})
    SECRETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SECRETS_FILE.write_text(yaml.safe_dump(data))
    os.chmod(SECRETS_FILE, 0o600)


def get_secret(name: str, cfg: dict | None = None) -> str | None:
    """env > keychain > file. Endpoint keys are stored as endpoint_key:<host>."""
    env = os.environ.get(SECRET_ENV.get(name, "")) or os.environ.get(
        "BABY_ONIT_API_KEY" if name.startswith("endpoint_key:") else "BABY_ONIT_" + name.upper()
    )
    return env or _keyring(name) or _file_secrets().get(name)


def set_secret(name: str, value: str | None) -> str:
    """Store a secret. Returns where it landed (for the setup wizard to say)."""
    return "keychain" if value and _keyring(name, value) else (str(SECRETS_FILE), _file_set(name, value))[0]


def _load_models(base: Path | None = None) -> dict:
    """Remembered model per endpoint (models.yaml); {} when missing/corrupt."""
    return _load_yaml((base or MODELS_FILE.parent) / "models.yaml")


def _save_models(models: dict, base: Path | None = None) -> None:
    """Write models.yaml, creating the parent directory if needed."""
    p = (base or MODELS_FILE.parent) / "models.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(models, sort_keys=False))


def forget_endpoint(host: str) -> list[str]:
    """Delete one endpoint's API key (keychain + file) and remembered model."""
    norm, removed = normalize_host(host, is_ollama_host(host)), []
    if _keyring(f"endpoint_key:{norm}") or f"endpoint_key:{norm}" in _file_secrets():
        _keyring(f"endpoint_key:{norm}", "")
        _file_set(f"endpoint_key:{norm}", None)
        removed.append("API key")
    models = _load_models()
    if models.pop(norm, None):
        _save_models(models)
        removed.append("remembered model")
    return removed


def known_endpoints() -> list[dict]:
    """Endpoints ever configured: config host + endpoint_key:<host> entries."""
    out: dict[str, dict] = {}
    cfg, models = load_config(), _load_models()
    for host in (
        [cfg["serving"]["host"]]
        + [n.split(":", 1)[1] for n in _file_secrets() if n.startswith("endpoint_key:")]
        + list(PRESET_HOSTS.values())
    ):
        if not (host := (host or "").strip()):
            continue
        norm = normalize_host(host, is_ollama_host(host))
        ep = out.setdefault(
            norm, {"host": norm, "key": get_secret(f"endpoint_key:{norm}"), "model": "", "active": False}
        )
        if host == cfg["serving"]["host"]:
            ep["active"] = True
        ep["model"] = models.get(norm) or (cfg["serving"].get("model") or "" if ep["active"] else "")
    return sorted(out.values(), key=lambda e: (not e["active"], e["host"]))


# S3. Provider adapters — one chat call, many servers (any OpenAI-compatible endpoint).
def is_ollama_host(host: str) -> bool:
    """True if the host URL points at an Ollama server (native API, not /v1 shim)."""
    return "ollama" in host.lower()


def normalize_host(host: str, ollama: bool) -> str:
    """Strip /v1 or /chat/completions so either spelling works in config."""
    h = host.strip().rstrip("/")
    if ollama:
        return re.sub(r"/v1$", "", h)
    h = h.removesuffix("/chat/completions")
    return h if h.endswith("/v1") else h + "/v1"


class Provider:
    """Thin async wrapper exposing one `chat()` for both client families."""

    def __init__(self, cfg: dict):
        """Build the client for the configured host; validate the URL up front."""
        self.cfg, self.host = cfg, cfg["serving"]["host"]
        if not re.match(r"^https?://", self.host):
            raise SystemExit(f"serving.host {self.host!r} is not a URL — run: baby-onit setup")
        self.ollama = is_ollama_host(self.host)
        self.host = normalize_host(self.host, self.ollama)
        self.model, self._model_cache = cfg["serving"]["model"] or None, None
        self.client = self._make_client()

    def _ensure_client(self):
        """Usable client, rebuilt if a previous turn closed it (separate loops)."""
        for obj in (self.client, getattr(self.client, "_client", None)):
            try:
                flag = getattr(obj, "is_closed", None)
                flag = flag() if callable(flag) else flag
            except Exception as e:  # noqa: BLE001 — client internals vary across SDK versions
                print(f"(client close probe skipped: {e})")
                continue
            if flag:
                self.client = self._make_client()
                break
        return self.client

    def _make_client(self):
        """Instantiate the right client family (ollama.AsyncClient or AsyncOpenAI)."""
        key = get_secret(f"endpoint_key:{self.host}", self.cfg)
        if self.ollama:  # ollama's client takes a float timeout, not httpx's dict
            import ollama

            return ollama.AsyncClient(
                host=self.host, timeout=300.0, headers={"Authorization": f"Bearer {key}"} if key else None
            )
        import httpx
        from openai import AsyncOpenAI

        return AsyncOpenAI(
            base_url=self.host,
            api_key=key or "EMPTY",
            max_retries=0,
            timeout=httpx.Timeout(connect=30, read=300, write=30, pool=30),
            # Portkey/LiteLLM select the upstream provider with x-portkey-* (or
            # equivalent) headers; without these a gateway host 404s or 401s.
            default_headers=self._extra_headers(),
        )

    def _extra_headers(self) -> dict[str, str] | None:
        """serving.extra_headers as a header mapping, or None.

        Validated here because the natural mistake is to write the header *name*
        as a bare string. Left unchecked that reaches httpx as a str and dies
        several frames deep in the SDK with "'str' object is not a mapping",
        which says nothing about the config key that caused it.
        """
        raw = self.cfg["serving"].get("extra_headers")
        if not raw:
            return None
        if not isinstance(raw, dict):
            raise SystemExit(
                f"serving.extra_headers must be a mapping of header name to value, "
                f"not {type(raw).__name__} ({raw!r}).\nIn config.yaml:\n"
                f"  serving:\n    extra_headers:\n"
                f"      x-portkey-virtual-key: your-virtual-key"
            )
        return {str(k): str(v) for k, v in raw.items()}

    async def close_client(self) -> None:
        """Drop pooled connections on the loop that opened them."""
        try:
            await (getattr(self.client, "aclose", None) or getattr(self.client, "close", None))()
        except Exception as e:  # noqa: BLE001 — half-closed pool is harmless; the next turn reconnects
            print(f"(close skipped: {e})")

    async def list_models(self) -> list[str]:
        """List model ids visible at the endpoint (cached after first call)."""
        if self._model_cache is None:
            try:
                r = await (self.client.list() if self.ollama else self.client.models.list())
                self._model_cache = [m.model if self.ollama else m.id for m in (r.models if self.ollama else r.data)]
            except Exception:  # noqa: BLE001 — endpoint unreachable: empty cache, retried next turn
                self._model_cache = []
        return self._model_cache

    async def autodetect_model(self) -> str:
        """Pick the first visible model; set self.model and return it."""
        models = await self.list_models()
        await self.close_client()  # pool opened on this call's loop; close here
        if not models:
            raise SystemExit(f"No models visible at {self.host} — set serving.model")
        self.model = models[0]
        return self.model

    def remember_model(self, model: str | None = None) -> None:
        """Persist the model for this endpoint in models.yaml."""
        if (model := model or self.model) and self.host:
            models = _load_models()
            models[self.host] = model
            _save_models(models)

    async def _stream_with_retry(self, body: dict):
        """Yield completion chunks as they arrive, retrying an empty body.

        A retry is only honourable until the first chunk reaches the caller:
        past that point the text has already been handed to the voice pipeline
        and replaying the call would speak the opening twice.
        """
        for attempt in range(3):
            started = False
            try:
                async for c in await self.client.chat.completions.create(**body):
                    started = True
                    yield c
                return
            except Exception as e:
                if started or "empty response" not in str(e).lower():
                    raise
                if attempt == 2:
                    raise RuntimeError(
                        f"provider returned an empty response 3x in a row (model={self.model}, "
                        f"host={self.host}) — usually a provider-side flake on a large payload; "
                        f"lower serving.history_budget_tokens or set serving.model to another model"
                    ) from e
                await asyncio.sleep(1.5 * (attempt + 1))

    async def _stream_ollama(self, kwargs: dict):
        """Ollama stream with the same empty-body retry as _stream_with_retry."""
        for attempt in range(3):
            started = False
            try:
                async for c in await self.client.chat(**kwargs):
                    started = True
                    yield c
                return
            except Exception as e:
                if started or "empty response" not in str(e).lower():
                    raise
                if attempt == 2:
                    raise RuntimeError(
                        f"provider returned an empty response 3x in a row (model={self.model}, "
                        f"host={self.host}) — usually a provider-side flake on a large payload; "
                        f"lower serving.history_budget_tokens or set serving.model to another model"
                    ) from e
                await asyncio.sleep(1.5 * (attempt + 1))

    async def chat(
        self, messages: list[dict], tools: list[dict] | None = None, stream_cb: Callable[[str], None] | None = None
    ) -> tuple[dict, dict]:
        """One chat completion. Returns (assistant_message_dict, usage_dict)."""
        s = self.cfg["serving"]
        self._ensure_client()
        if self.ollama:
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "stream": True,
                "think": s["think"],
                "options": {"temperature": s["temperature"], "top_p": s["top_p"], "num_predict": s["max_tokens"]},
            }
            if tools:
                kwargs["tools"] = tools
            parts: list[str] = []
            think_parts: list[str] = []
            calls: list[dict] = []
            u = None
            async for c in self._stream_ollama(kwargs):
                piece = c.message.content or ""
                parts.append(piece)
                think_parts.append(getattr(c.message, "thinking", "") or "")
                if piece and stream_cb:
                    stream_cb(piece)  # live, as it arrives; never stream thinking
                calls += [
                    {"name": tc.function.name, "arguments": dict(tc.function.arguments or {})}
                    for tc in c.message.tool_calls or []
                ]
                if c.done:
                    u = getattr(c, "usage", None) or c  # nested or flat
            content = "".join(parts)
            thinking = "".join(think_parts)
            raw_calls = [{"function": {"name": c["name"], "arguments": c["arguments"]}} for c in calls]
            usage = (
                {
                    "prompt_tokens": getattr(u, "prompt_eval_count", 0) or 0,
                    "completion_tokens": getattr(u, "eval_count", 0) or 0,
                }
                if u
                else {}
            )
            return (
                {
                    "role": "assistant",
                    "content": content,
                    "thinking": thinking,
                    "tool_calls": calls,
                    "raw_tool_calls": raw_calls,
                },
                usage,
            )
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = tools
        upstream = self.upstream_family()
        if upstream != "anthropic":  # newer Claude models reject these params
            body["temperature"], body["top_p"] = s["temperature"], s["top_p"]
        body["max_completion_tokens" if upstream == "openai" else "max_tokens"] = s["max_tokens"]
        if s["think"] and self.host_has_thinking():
            body["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
        usage: dict = {}
        parts: list[str] = []
        pending: dict[str, dict] = {}  # index -> accumulating tool call
        # One pass: emit each delta to stream_cb the moment it lands, so the
        # voice pipeline can start speaking before the completion is finished.
        async for c in self._stream_with_retry(body):
            if c.usage:
                usage = {
                    "prompt_tokens": c.usage.prompt_tokens,
                    "completion_tokens": c.usage.completion_tokens,
                }
            if not c.choices:
                continue
            d = c.choices[0].delta
            piece = d.content or ""
            if piece:
                parts.append(piece)
                if stream_cb:
                    stream_cb(piece)
            for tc in d.tool_calls or []:
                slot = pending.setdefault(tc.index, {"name": "", "arguments": "", "id": tc.id})
                if tc.function:
                    slot["name"] = tc.function.name or slot["name"]
                    slot["arguments"] += tc.function.arguments or ""
        content = "".join(parts)
        calls, raw_calls = [], []
        for slot in pending.values():
            if not slot["name"]:
                continue  # truncated mid-call: no function name arrived
            try:
                args = json_repair(slot["arguments"]) if slot["arguments"] else {}
            except Exception:  # noqa: BLE001 — truncated mid-args: dispatch reports the parse error
                args = {}
            calls.append({"name": slot["name"], "arguments": args})
            raw_calls.append(
                {"id": slot["id"], "type": "function", "function": {"name": slot["name"], "arguments": slot["arguments"]}}
            )
        return (
            {
                "role": "assistant",
                "content": content,
                "thinking": "",
                "tool_calls": calls,
                "raw_tool_calls": raw_calls,
            },
            usage,
        )

    def upstream_family(self) -> str:
        """Which dialect the far end speaks: anthropic|openai|generic.

        The host URL is only a reliable signal when you talk to the provider
        directly. Behind a gateway (Portkey, LiteLLM) every model shares one
        hostname, so serving.upstream names the family explicitly; "auto" keeps
        the original host sniff for direct endpoints.
        """
        declared = (self.cfg["serving"].get("upstream") or "auto").lower()
        if declared != "auto":
            return declared
        if "api.anthropic.com" in self.host:
            return "anthropic"
        if "api.openai.com" in self.host:
            return "openai"
        # A gateway with no declared upstream: infer from the model id, which is
        # the only thing left that names the real model.
        m = (self.model or "").lower()
        if "claude" in m:
            return "anthropic"
        if m.startswith(("gpt-", "o1", "o3", "o4")):
            return "openai"
        return "generic"

    def host_has_thinking(self) -> bool:
        """vLLM/SGLang expose thinking via chat_template_kwargs; no-op elsewhere."""
        if (self.cfg["serving"].get("upstream") or "auto").lower() != "auto":
            return False  # a named upstream is a hosted API, not a local server
        return any(k in self.host for k in ("vllm", "sglang", "8000", "8001", "30000"))


# S4. System prompt — the harness's constitution: identity, working dir, tool routing, credentials.
SYSTEM_PROMPT = """\
You are baby-onit, a helpful AI agent that completes tasks by calling tools.

Working directory: {data_path} (today: {today}). Every file path you write \
starts here.

Tool routing:
- Files in the working directory: read_file, grep, local_search.
- Public facts, news, current events: web search; fetch pages for details.
- Anything that can change (prices, versions, dates) needs a tool result, \
never memory. Prefer the primary source over anyone summarizing it.
- Finish with a concise answer; cite the files or URLs you used.

Credentials (pre-wired by the harness; never print, echo, or embed a token):
- GitHub: git clone/pull/push with plain https://github.com/... URLs just \
works — the harness authenticates git via GIT_ASKPASS. Never hunt for a token or put one in a URL.
- Hugging Face: HF_TOKEN rides in the bash environment for model-hub access. Use `hf download ...` or `huggingface-cli download ...`; a 401 on a gated repo means accept the terms on the model page, then retry.
"""


# Voice mode: everything the model says is spoken aloud by a TTS engine, so the
# shape of a good answer changes completely. Appended to SYSTEM_PROMPT, never a
# replacement -- tool routing and credential rules still apply.
VOICE_PROMPT = """\

You are speaking out loud. Everything you write is sent to a speech synthesizer \
and heard by a person, so:
- Write what a person would say, not what they would read. No markdown, no \
bullet lists, no tables, no code blocks, no emoji, no URLs read out character \
by character. Numbers and units in words when that is how you would say them.
- Short sentences. The synthesizer speaks one sentence at a time, so a long \
clause is a long silence.
- Answer the question first, in one or two sentences. Offer detail only if it \
was asked for.

Long jobs go to the background. Anything that will take more than a few \
seconds -- writing an app, a large refactor -- goes to `start_task`, which \
returns an id at once. Never do a long job in the foreground: the person cannot \
talk to you while you do, which is the single worst thing you can do to a \
conversation.
- "Long" includes **starting or serving anything**: a dev server, a Flask app, \
a watcher. Those never finish, so running one in the foreground hangs the \
conversation until the tool times out.
- Call `start_task` **once** per request. It returns an id; that means the work \
is under way. Do not call it again for the same job, and do not call it again \
because you are unsure -- call `check_task` instead.
- After starting one, say so out loud and stop. Do not keep working in the \
foreground to "help" -- the task is already doing it.
- When asked how it is going, `check_task` and report the step it is on.
- Stop one only when asked, with `cancel_task`.

Speaking is a tool call: `speak`. It returns immediately -- it does not wait \
for the audio to finish -- so you can keep working while your words play.
- Call `speak` as your FIRST action on any task that will take more than a \
moment, before any other tool. Silence reads as a broken agent.
- While a long task runs, call `speak` again at each step to say what you are \
doing now. Keep each one to a sentence.
- Your final answer is spoken too, so end in plain prose.

A voice activity detector is listening the whole time you talk. If the person \
starts speaking, your speech is cut off mid-sentence and the rest of your reply \
is discarded. That is normal and expected -- it is not an error, and you should \
not apologise for it or try to resume where you left off. Treat whatever they \
said as the next thing to answer, and assume they only heard the part that \
played before they interrupted.
"""


def build_system_prompt(cfg: dict) -> str:
    """Render the system prompt with the working directory and today's date."""
    prompt = SYSTEM_PROMPT
    if cfg.get("voice"):
        prompt += VOICE_PROMPT
    return prompt.format(data_path=cfg["data_path"], today=time.strftime("%Y-%m-%d"))


# S5. Tool implementations — tools are functions with a JSON-schema business card.
DATA_PATH: str = ""  # jail root, set by run(); all paths resolve inside it

# Hard ceiling on any single tool result (chars): one unbounded result (a 3 MB
# base64 grep match, a runaway log) can blow the context window in one iteration.
TOOL_RESULT_MAX_CHARS = 40_000
GREP_LINE_MAX_CHARS = 300  # a single matched line longer than this is elided
GREP_MAX_MATCHES = 50  # stop after this many matching lines
GREP_MAX_CHARS = 20_000  # ...or this many bytes of matches, whichever first


def _cap_result(text: str, limit: int = TOOL_RESULT_MAX_CHARS) -> str:
    """Clamp one tool result to `limit` chars, keeping head and tail."""
    if len(text) <= limit:
        return text
    head, tail = limit * 3 // 4, limit // 4 - 80
    return (
        f"{text[:head]}\n\n...[tool result truncated: {len(text):,} chars total, "
        f"showing first {head:,} and last {tail:,} — narrow the query "
        f"(pattern, path, file_pattern) or read a specific file]...\n\n{text[-tail:]}"
    )


# Executables refused outright in bash (onit: command_policy.py NEVER_ASK_COMMANDS):
# privilege escalation, namespace escape, host-OS writes. Matches the binary name,
# so read-only forms (`systemctl status`) are refused too; ssh/scp/rsync are NOT
# listed — the jail is about the local filesystem. block_dangerous: false disables.
# one string reads as a list; SIM905 would explode it to 43 noisy lines
NEVER_ASK_COMMANDS = frozenset(
    "sudo su doas pkexec setcap setpriv capsh chroot nsenter unshare docker dockerd podman "  # noqa: SIM905
    "nerdctl ctr containerd runc kubectl helm minikube lxc lxc-attach machinectl useradd "
    "usermod userdel groupadd gpasswd passwd chpasswd visudo chown chgrp newgrp mount "
    "umount systemctl service at crontab shutdown reboot halt poweroff".split()
)


def _resolve(path: str) -> Path:
    """Resolve a path inside DATA_PATH (the jail); ValueError on escape."""
    p, base = Path(path).expanduser(), Path(DATA_PATH).resolve()
    resolved = (base / p) if not p.is_absolute() else p
    if not load_config().get("enforce_jail", True):
        return resolved
    if (resolved := resolved.resolve()) != base and not str(resolved).startswith(str(base) + os.sep):
        raise ValueError(f"Path outside jail root {base}: {path}")
    return resolved


def _gate_bash(command: str) -> str | None:
    """Refuse dangerous bash (NEVER_ASK set + curl|sh); None = allow."""
    if not load_config().get("block_dangerous", True):
        return None
    for exe in NEVER_ASK_COMMANDS:  # match the executable as a word, not a substring
        if re.search(rf"(?:^|[;&|`$(\s]){re.escape(exe)}(?:\s|$)", command):
            return (
                f"Refused: '{exe}' is on the never-ask list "
                f"(privilege escalation, remote access, or host control)."
            )
    if re.search(r"\b(?:curl|wget)\b[^\n|]*\|\s*(?:ba|z|da)?sh\b", command):
        return "Refused: piping a remote download straight into a shell."
    return None


def tool_web_search(query: str, max_results: int = 5, type: str = "web") -> str:
    """Tavily (if key) > Ollama web search API."""
    max_results = max(1, min(int(max_results), 10))
    if key := get_secret("tavily_api_key"):
        try:
            import requests

            r = requests.post(
                "https://api.tavily.com/search",
                timeout=15,
                json={
                    "api_key": key,
                    "query": query,
                    "max_results": max_results,
                    "topic": "news" if type == "news" else "general",
                },
            )
            r.raise_for_status()
            return json.dumps(
                [
                    {"title": x.get("title"), "url": x.get("url"), "snippet": x.get("content", "")}
                    for x in r.json().get("results", [])
                ]
            )
        except Exception as e:  # noqa: BLE001 — tavily failed; ollama web search fallback follows
            return f"(tavily failed: {e})"
    key = get_secret("ollama_api_key") or get_secret(f"endpoint_key:{load_config()['serving']['host']}")
    try:
        import ollama

        client = ollama.Client(
            host="https://api.ollama.com", headers={"Authorization": f"Bearer {key}"} if key else None
        )
        return json.dumps(
            [
                {"title": r.title, "url": r.url, "snippet": r.content}
                for r in client.web_search(query=query, max_results=max_results).results
            ]
        )
    except Exception as e:  # noqa: BLE001 — last tier; report the failure
        return f"(web search failed: tavily {e!r})"


def tool_fetch_content(url: str) -> str:
    """GET a page, extract text (BeautifulSoup) or PDF text (pypdf)."""
    import requests

    resp = requests.get(url, timeout=30, headers={"User-Agent": "baby-onit/0.1"})
    if "pdf" in resp.headers.get("content-type", "") or url.lower().endswith(".pdf"):
        from pypdf import PdfReader

        return "\n".join(p.extract_text() or "" for p in PdfReader(resp.content).pages)
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    return soup.get_text("\n", strip=True)[:50_000]


def tool_bash(command: str, timeout: int = 300) -> str:
    """Run a shell command with cwd=DATA_PATH."""
    if gated := _gate_bash(command):
        return json.dumps({"error": gated, "command": command, "status": "refused"})
    env = dict(os.environ)
    if token := get_secret("github_token"):  # GIT_ASKPASS: git reads the token at call time
        env.update(GITHUB_TOKEN=token, GIT_ASKPASS=str(Path(__file__).with_name("_askpass.sh")))
    if token := get_secret("huggingface_token"):  # HF_TOKEN: model-hub auth
        env.setdefault("HF_TOKEN", token)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=DATA_PATH or os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,  # exit code is surfaced to the model, not raised
        )
        return _cap_result(
            (proc.stdout or "") + (("\n[stderr] " + proc.stderr) if proc.stderr else "") or "(no output)"
        )
    except subprocess.TimeoutExpired:
        return f"(timed out after {timeout}s)"


def tool_write_file(path: str, content: str, mode: str = "write") -> str:
    """Write text to a file (mode='append' to add). Creates parent dirs."""
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    (p.write_text(content, encoding="utf-8") if mode == "write" else p.open("a", encoding="utf-8").write(content))
    return f"wrote {len(content)} chars to {p}"


def tool_read_file(path: str, max_chars: int = 20_000) -> str:
    """Read a file's text (PDFs supported). Truncates to max_chars."""
    p = _resolve(path)
    if p.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        return "\n".join(pg.extract_text() or "" for pg in PdfReader(str(p)).pages)[:max_chars]
    text = p.read_text(errors="replace")
    return text[:max_chars] + ("\n...[truncated]" if len(text) > max_chars else "")


def tool_edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace an exact old_string with new_string. Errors if ambiguous."""
    p = _resolve(path)
    text = p.read_text()
    if not (n := text.count(old_string)):
        return "ERROR: old_string not found"
    if n > 1 and not replace_all:
        return f"ERROR: old_string appears {n} times; pass replace_all=true"
    p.write_text(text.replace(old_string, new_string, -1 if replace_all else 1))
    return f"edited {p} ({n if replace_all else 1} replacement(s))"


# --- local_search: the only nontrivial tool left in -------------------------
# CONCEPT: retrieval = chunk -> index -> rank -> fuse. onit's local_search
# stop-words for BM25; one string reads as a list (SIM905 would explode it)
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or "  # noqa: SIM905
    "that the to was were will with".split()
)


def _tokens(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stop-words removed."""
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]


def _chunks(text: str, size: int = 1200, overlap: int = 150) -> list[str]:
    """Paragraph-aware chunking with overlap (same defaults as onit)."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    out, cur = [], ""
    for p in paras:
        while len(p) > size:  # oversized paragraph: hard-split
            out.append(p[:size])
            p = p[size - overlap :]
        if len(cur) + len(p) + 2 > size and cur:
            out.append(cur)
            cur = cur[-overlap:] + "\n\n" + p if overlap else p
        else:
            cur = (cur + "\n\n" + p).strip()
    return out + ([cur] if cur else []) or [text]


def _bm25(corpus: list[str], query: str, k1: float = 1.5, b: float = 0.75):
    """Okapi BM25. Returns [(score, doc_idx)] sorted desc."""
    docs, qt = [_tokens(d) for d in corpus], set(_tokens(query))
    avgdl = sum(map(len, docs)) / max(len(docs), 1)
    df = Counter(t for d in docs for t in set(d))  # one pass, not one scan per term

    def score(d):
        return sum(
            math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
            * (c := d.count(t))
            * (k1 + 1)
            / (c + k1 * (1 - b + b * len(d) / avgdl))
            for t in set(d) & qt
        )

    return sorted(((score(d), i) for i, d in enumerate(docs)), reverse=True)


_INDEX: dict[str, dict] = {}  # path -> {"chunks": [...], "mtime": float}


def _index_dir(root: Path) -> dict[str, dict]:
    """Build/refresh the in-memory BM25 index for files under root."""
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in (".md", ".txt", ".csv", ".json", ".yaml", ".py"):
            continue
        try:
            if (mtime := p.stat().st_mtime) != _INDEX.get(str(p), {}).get("mtime"):
                _INDEX[str(p)] = {"mtime": mtime, "chunks": _chunks(p.read_text(errors="replace"))}
        except Exception as e:  # noqa: BLE001 — unreadable file keeps its stale chunks
            print(f"(index skipped {p.name}: {e})")  # keep the stale chunk set
    return _INDEX


def tool_local_search(query: str, top_k: int = 5) -> str:
    """BM25 over DATA_PATH files. onit adds dense embeddings + RRF fusion."""
    corpus, owners = [], []
    for path, rec in _index_dir(Path(DATA_PATH)).items():
        corpus += rec["chunks"]
        owners += [(path, i) for i in range(len(rec["chunks"]))]
    if not corpus:
        return "(no indexable files under data_path)"
    hits = [(s, i) for s, i in _bm25(corpus, query)[:top_k] if s > 0]
    return json.dumps(
        [
            {
                "rank": r + 1,
                "score": round(s, 3),
                "file": owners[i][0],
                "chunk": owners[i][1],
                "text": corpus[i][:600],
            }
            for r, (s, i) in enumerate(hits)
        ]
    )


def tool_set_alarm(when: str, label: str = "alarm") -> str:
    """Schedule a one-shot alarm. `when` is HH:MM (24h), "7am", or "in 10 minutes"."""
    import datetime as _dt

    now = _dt.datetime.now()
    t = (when or "").strip().lower()
    if m := re.fullmatch(r"in\s+(\d+)\s*(s|sec|secs|second|seconds|m|min|mins|minute|minutes|h|hour|hours)", t):
        n, unit = int(m[1]), m[2][0]
        at = now + _dt.timedelta(**{{"s": "seconds", "m": "minutes", "h": "hours"}[unit]: n})
    elif m := re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", t):
        hh, mm, ap = int(m[1]), int(m[2] or 0), m[3]
        if ap == "pm" and hh != 12:
            hh += 12
        elif ap == "am" and hh == 12:
            hh = 0
        if not (0 <= hh <= 23 and 0 <= mm <= 59):
            return _tool_error("E_ARGS", tool="set_alarm", bad=f"{when!r} is not a valid time")
        at = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if at <= now:
            at += _dt.timedelta(days=1)  # a time already past means tomorrow
    else:
        return _tool_error(
            "E_ARGS", tool="set_alarm", bad=f"cannot parse {when!r}",
            accepts=["HH:MM", "7am", "in 10 minutes"],
        )
    _ALARMS.append({"at": at.isoformat(timespec="seconds"), "label": label})
    delta = at - now
    mins = round(delta.total_seconds() / 60)
    return json.dumps(
        {"set": at.strftime("%H:%M"), "label": label, "in_minutes": mins, "total": len(_ALARMS)}
    )


def tool_list_alarms() -> str:
    """List the alarms set this session, soonest first."""
    return json.dumps(sorted(_ALARMS, key=lambda a: a["at"])) if _ALARMS else "no alarms set"


def tool_grep(pattern: str, path: str = ".", file_pattern: str = "*") -> str:
    """Recursive regex search across DATA_PATH, bounded on matches/bytes/line-len."""
    rx, out, total = re.compile(pattern), [], 0
    for p in sorted(_resolve(path).rglob(file_pattern)):
        try:
            for i, line in enumerate(p.read_text(errors="replace").splitlines()):
                if not rx.search(line):
                    continue
                line = line.strip()
                if len(line) > GREP_LINE_MAX_CHARS:  # minified JSON, base64 blobs
                    line = line[:GREP_LINE_MAX_CHARS] + f"...[{len(line):,} chars on this line]"
                hit = f"{p.relative_to(DATA_PATH)}:{i+1}: {line}"
                out.append(hit)
                total += len(hit) + 1
                if len(out) >= GREP_MAX_MATCHES or total >= GREP_MAX_CHARS:
                    return "\n".join(out) + "\n...[grep stopped: match/byte cap reached — narrow the pattern or path]"
        except Exception as e:  # noqa: BLE001 — unreadable path is skipped by grep
            print(f"(grep skipped {p.name}: {e})")
    return "\n".join(out) or "(no matches)"


# S6. Tool registry + dispatch — the model sees a *schema*; the harness runs a *function*.
TOOLS: dict[str, tuple[dict, Callable[..., str]]] = {}


def tool(name: str, description: str, fn: Callable[..., str]) -> None:
    """Register a tool + build its JSON schema from the function's type hints."""
    props, required = {}, []
    for pname, param in inspect.signature(fn).parameters.items():
        ann = param.annotation
        if isinstance(ann, str):
            ann = {"str": str, "int": int, "float": float, "bool": bool}.get(ann, str)
        jtype = {str: "string", int: "integer", float: "number", bool: "boolean"}.get(ann, "string")
        props[pname] = {"type": jtype, "description": f"{pname} ({jtype})"}
        if param.default is inspect.Parameter.empty:
            required.append(pname)
    TOOLS[name] = (
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {"type": "object", "properties": props, "required": required},
            },
        },
        fn,
    )


#: The VoiceSession that owns delegated work, set while a voice session is
#: open. Module-level for the same reason _ALARMS is: TOOLS holds plain
#: functions, so a tool reaches its session through here rather than through a
#: parameter the model would have to supply.
_TASK_HOST: Any = None


async def tool_start_task(task: str) -> str:
    """Delegate a long task to a background agent. Returns at once with its id."""
    if _TASK_HOST is None:
        return _tool_error(
            "E_TOOL", tool="start_task",
            err="no voice session; background tasks exist only in voice mode",
        )
    if not (task or "").strip():
        return _tool_error("E_ARGS", tool="start_task", bad="task is empty")
    return json.dumps(_TASK_HOST.start_background(task.strip()))


async def tool_check_task(task_id: int = 0) -> str:
    """Progress of a background task, or of all of them when task_id is 0."""
    if _TASK_HOST is None:
        return _tool_error("E_TOOL", tool="check_task", err="no voice session")
    return json.dumps(_TASK_HOST.background_status(task_id or None))


async def tool_cancel_task(task_id: int) -> str:
    """Stop a background task."""
    if _TASK_HOST is None:
        return _tool_error("E_TOOL", tool="cancel_task", err="no voice session")
    return json.dumps(await _TASK_HOST.cancel_background(task_id))


for _name, _desc, _fn in [
    ("web_search", "Search the web. Use type='news' for dated/recent items.", tool_web_search),
    ("fetch_content", "Fetch a URL and return its text (handles PDFs).", tool_fetch_content),
    (
        "bash",
        (
            "Run a shell command inside the working directory. GitHub git operations (clone/pull/push) "
            "and Hugging Face downloads are pre-authenticated — just use plain https:// URLs; never handle tokens."
        ),
        tool_bash,
    ),
    ("write_file", "Write text to a file (mode='append' to add).", tool_write_file),
    ("read_file", "Read a file's text (PDFs supported).", tool_read_file),
    ("edit_file", "Replace an exact old_string with new_string.", tool_edit_file),
    ("local_search", "BM25 search over files in the working directory.", tool_local_search),
    ("grep", "Recursive regex search across files.", tool_grep),
    ("set_alarm", "Set a one-shot alarm. when='07:00', '7am', or 'in 10 minutes'.", tool_set_alarm),
    ("list_alarms", "List alarms set this session.", tool_list_alarms),
    (
        "start_task",
        (
            "Delegate a long job (writing an app, a big refactor) to a background "
            "agent and get its id back immediately, so you stay free to talk. Use "
            "this for anything that will take more than a few seconds."
        ),
        tool_start_task,
    ),
    (
        "check_task",
        "Progress of a background task; task_id=0 lists all of them.",
        tool_check_task,
    ),
    ("cancel_task", "Stop a background task.", tool_cancel_task),
]:
    tool(_name, _desc, _fn)
READ_ONLY = {"web_search", "fetch_content", "read_file", "local_search", "grep",
             "list_alarms", "check_task"}
#: Tools a delegated worker must not see. A worker that can delegate will:
#: observed, one request produced task 1 -> task 2 -> task 3, each worker
#: re-reading "long jobs go to the background" and obeying it. A worker IS the
#: background; its job is to do the work.
DELEGATION_TOOLS = {"start_task", "cancel_task", "check_task"}

#: Exempt from the repeated-call guard in dispatch(). Asking the same status
#: question twice is not a loop, it is the point: "how is it going" five times
#: during a long build is normal conversation, and the guard would answer E_LOOP
#: instead of the status.
POLLABLE = {"check_task", "list_alarms", "is_speaking"}
# Session-scoped alarms. Deliberately not persisted: the voice agent is a
# long-lived process, and an alarm that outlives it would fire into silence.
_ALARMS: list[dict] = []
_call_history: list[tuple[str, str]] = []


# S6.5. MCP client — the one kind of tool that cannot be an in-process function.
# baby-onit deliberately inlines its tools (docs/PLAN.md), and that is still the
# right default. The voice agent needs an exception: the speech engine runs on a
# different machine from the harness, owns a GPU-sized pile of model state, and
# is shared with the audio data plane. MCP is how the model reaches across that
# process boundary without the harness caring where the far end lives.
class McpHub:
    """Connects streamable-HTTP MCP servers and grafts their tools onto TOOLS.

    Sessions are async context managers held open on a single event loop, so the
    hub must be opened and closed on the loop that will run the tool calls. Voice
    mode owns one long-lived loop (see run_voice) and opens the hub once; the
    one-shot paths open and close it per turn.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.servers: dict[str, dict] = (cfg.get("mcp") or {}).get("servers") or {}
        self.sessions: dict[str, Any] = {}  # server name -> ClientSession
        self.registered: list[str] = []  # TOOLS keys this hub added, for clean removal
        self._holders: list[asyncio.Task] = []
        self._shutdown: asyncio.Event | None = None

    async def open(self, on_event: Callable[[str, str], None] | None = None) -> int:
        """Connect every configured server and register its tools. Returns the count."""
        if not self.servers:
            return 0
        self._shutdown = asyncio.Event()
        n = 0
        for name, spec in self.servers.items():
            url = spec if isinstance(spec, str) else spec.get("url")
            if not url:
                continue
            ready: asyncio.Future = asyncio.get_running_loop().create_future()
            self._holders.append(asyncio.create_task(self._hold(name, url, ready), name=f"mcp:{name}"))
            try:
                session, listed = await ready
            except Exception as e:  # noqa: BLE001 — a dead MCP server must not kill the agent
                if on_event:
                    on_event("status", f"mcp {name}: unreachable at {url} ({type(e).__name__}: {e})")
                continue
            self.sessions[name] = session
            for t in listed.tools:
                n += 1 if self._register(name, session, t, on_event) else 0
        return n

    async def _hold(self, name: str, url: str, ready: asyncio.Future) -> None:
        """Own one session for its whole life, inside a single task.

        The transport nests anyio task groups, and an anyio cancel scope has to
        be exited by the task that entered it. An AsyncExitStack unwound from
        close() is a different task, which fails with "attempted to exit cancel
        scope in a different task". So the context is entered here, parked on an
        event, and exited here -- enter and exit are always the same task.
        """
        from mcp import Client

        try:
            # Client takes the URL directly and does connect+initialize itself,
            # so there is no transport plumbing to hold here.
            async with Client(url) as session:
                listed = await session.list_tools()
                if not ready.done():
                    ready.set_result((session, listed))
                assert self._shutdown is not None
                await self._shutdown.wait()
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 — surfaced to open() via the future
            if not ready.done():
                ready.set_exception(e)

    def _register(self, server: str, session: Any, t: Any, on_event=None) -> bool:
        """Graft one MCP tool into TOOLS under a non-colliding name."""
        key = t.name
        if key in TOOLS:  # in-process tools win; the MCP one gets qualified
            key = f"{server}_{t.name}"
            if key in TOOLS:
                if on_event:
                    on_event("status", f"mcp {server}: skipping {t.name} (name taken)")
                return False

        async def call(_s=session, _n=t.name, **args) -> str:
            res = await _s.call_tool(_n, args)
            return _mcp_result_text(res)

        # The model only ever sees the schema, so hand it the server's own
        # input_schema verbatim rather than re-deriving one from `call`'s **args.
        schema = getattr(t, "input_schema", None) or {"type": "object", "properties": {}, "required": []}
        TOOLS[key] = (
            {
                "type": "function",
                "function": {
                    "name": key,
                    "description": (t.description or f"{t.name} (via MCP server {server})").strip(),
                    "parameters": schema,
                },
            },
            call,
        )
        self.registered.append(key)
        return True

    async def close(self) -> None:
        """Drop every session and un-register the tools this hub added."""
        for key in self.registered:
            TOOLS.pop(key, None)
        self.registered.clear()
        self.sessions.clear()
        if self._shutdown is not None:
            self._shutdown.set()  # releases each holder to unwind its own contexts
        if self._holders:
            await asyncio.gather(*self._holders, return_exceptions=True)
            self._holders.clear()
        self._shutdown = None


def _mcp_result_text(res: Any) -> str:
    """Flatten an MCP CallToolResult into the plain string dispatch() expects.

    A failing MCP tool reports prose in a normal content block, but baby-onit's
    tool layer speaks machine-readable error JSON (see _tool_error) because the
    model recovers from that in one retry and re-calls the same broken form when
    it is handed prose. So a server-side failure is re-wrapped as E_TOOL.
    """
    if getattr(res, "is_error", False):
        return _tool_error("E_TOOL", tool="mcp", err=_mcp_join(res)[:200])
    out = _mcp_join(res)
    sc = getattr(res, "structured_content", None)
    return out or (json.dumps(sc, ensure_ascii=False) if sc else "")


def _mcp_join(res: Any) -> str:
    """Concatenate the text blocks of an MCP result, ignoring image/audio parts."""
    return "\n".join(
        c.text for c in (getattr(res, "content", None) or []) if getattr(c, "type", None) == "text"
    ).strip()


def json_repair(s: str) -> Any:
    """Models emit near-JSON: single quotes, trailing commas. Fix and parse.

    Also strips the nanochat streaming parser's <|python_start|>/<|python_end|>
    markers, which can ride along on tool-call arguments (measured 2026-11-02
    on the v6m gate server: arguments arrive as '{"p":"v"}<|python_end|>').
    """
    s = re.sub(r"^```[a-z]*\n|\n```$", "", s.strip())  # strip a code fence
    s = s.replace("<|python_start|>", "").replace("<|python_end|>", "").strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    if s.startswith("{"):
        last = s.rfind("}")
        if last != -1:
            trimmed = s[:last + 1]
            try:
                return json.loads(trimmed)
            except json.JSONDecodeError:
                s = trimmed
    return json.loads(re.sub(r"(?<!\\)'", '"', re.sub(r",\s*([}\]])", r"\1", s)))


def _coerce_args(name: str, args: dict) -> dict:
    """Cast args to the declared types (models emit `max_results: "10"`; SDKs die on it)."""
    hints = inspect.signature(TOOLS[name][1]).parameters
    out = {}
    for k, v in args.items():
        target = hints[k].annotation if k in hints else None
        if isinstance(target, str):
            target = {"int": int, "float": float, "bool": bool, "str": str}.get(target)
        try:
            if target is int and not isinstance(v, bool):
                out[k] = int(float(str(v).strip()))
            elif target is float and not isinstance(v, (int, float)):
                out[k] = float(str(v).strip())
            elif target is bool and not isinstance(v, bool):
                out[k] = str(v).strip().lower() in ("true", "1", "yes")
            else:
                out[k] = v
        except (ValueError, TypeError):
            out[k] = v  # leave it; the tool's own error path reports it
    return out


def _tool_error(code: str, **fields) -> str:
    """Machine-readable tool error: one JSON line the model can parse and act on.
    Replaces the old prose 'ERROR: ...' strings -- utak trained on structured
    feedback recovers in one retry; prose errors read as tool output and it
    re-calls the same broken form."""
    import json as _json
    return _json.dumps({"error": code, **fields}, ensure_ascii=False)


async def dispatch(name: str, args: dict) -> str:
    """Run one tool call with timeout + repeated-call guard. Returns a str."""
    if name not in TOOLS:
        return _tool_error("E_UNKNOWN_TOOL", tool=name, accepts=sorted(TOOLS))
    try:
        args = args if isinstance(args, dict) else json_repair(str(args))
    except Exception:  # noqa: BLE001 — model may emit truncated JSON; the error is the reply
        return _tool_error("E_PARSE", tool=name, raw=str(args)[:200])
    key = (name, json.dumps(args := _coerce_args(name, args), sort_keys=True))
    if name in POLLABLE:
        return await _invoke(name, args)
    _call_history.append(key)
    if _call_history.count(key) >= 5:
        return _tool_error("E_LOOP", tool=name, repeats=5,
                           hint="same call 5x — change approach or answer without tools")
    return await _invoke(name, args)


async def _invoke(name: str, args: dict) -> str:
    """Run one tool, mapping its failures onto the machine-readable error codes."""
    fn = TOOLS[name][1]
    try:
        # MCP tools are natively async; in-process tools are blocking functions
        # that have to leave the event loop free for the audio sockets. Build
        # the awaitable inside the try: an async fn validates its arguments at
        # call time, and that TypeError is the E_ARGS path below.
        call = fn(**args) if inspect.iscoroutinefunction(fn) else asyncio.to_thread(fn, **args)
        return str(await asyncio.wait_for(call, timeout=330))
    except TypeError as e:
        import inspect as _inspect
        sig = list(_inspect.signature(TOOLS[name][1]).parameters)
        return _tool_error("E_ARGS", tool=name, bad=str(e)[:160], accepts=sig)
    except Exception as e:  # noqa: BLE001 — tool errors are surfaced to the model as text
        return _tool_error("E_TOOL", tool=name, err=f"{type(e).__name__}: {str(e)[:160]}")


def _make_first_token_cb(sink: list) -> Callable:
    """Return a stream callback that records the first chunk's arrival time into sink."""

    def cb(_delta: str) -> None:
        if not sink:
            sink.append(time.monotonic())

    return cb


# S7. Agent loop — a while-loop with four moves: send messages, run tools, append results, stop.
# P5 over-call guard: a plain Q&A task ("what is 2+2?") must not see a tool
# surface on the first model call -- utak over-triggers on bare questions when
# tools are present (measured: CF1/CF2 0/3 with tools vs 3/3 without).
_TASK_TOOL_HINTS = (
    "file", "files", "directory", "read", "write", "edit", "append", "grep",
    "search", "fetch", "url", "http", "web", "news", "latest", "current",
    "today", "price", "version", "release", "repo", "git", "commit", "push",
    "list", "find", "run", "command", "script", "log", "json", "csv", "yaml",
)


def _task_needs_tools(task: str) -> bool:
    """True when the task text plausibly requires a tool (keyword gate)."""
    t = (task or "").lower()
    return any(h in t for h in _TASK_TOOL_HINTS)


async def agent_loop(
    task: str,
    cfg: dict,
    provider: Provider,
    on_event: Callable[[str, str], None] | None = None,
    max_iter: int | None = None,
    session_history: list[dict] | None = None,
    verbose: bool = False,
) -> str:
    """Run the task to completion; return the final assistant text."""
    global DATA_PATH
    DATA_PATH = cfg["data_path"]
    max_iter = max_iter if max_iter is not None else cfg["serving"]["max_chat_iterations"]
    limit = cfg["serving"]["max_context_tokens"]
    messages: list[dict] = [{"role": "system", "content": build_system_prompt(cfg)}]
    messages += build_session_messages(session_history or [], cfg["serving"]["history_budget_tokens"])
    messages.append({"role": "user", "content": task})
    _call_history.clear()
    iteration, answer, consecutive_errors = 1, "", 0
    say = on_event or (lambda kind, text: None)
    while True:
        if max_iter > 0 and iteration > max_iter:
            say("turn", f"iteration cap reached ({max_iter})")
            break
        # Compact *before* sending, not after. prompt_tokens from the previous
        # call cannot see the tool results appended since, so a check placed
        # after the append is always one iteration stale — it lets a single
        # oversized result through to a 400. The estimate below is current.
        if _messages_tokens(messages) > 0.85 * limit and (max_iter <= 0 or iteration < max_iter):
            say("status", "compacting context\u2026")
            ct0 = time.monotonic()
            summary, _ = await provider.chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "Summarize the conversation so far: task, findings, files touched, what remains."
                        ),
                    },
                    {
                        "role": "user",
                        "content": "\n".join(
                            f"{m['role']}: {m.get('content') or ''}"
                            + (f" {json.dumps(m['tool_calls'])}" if m.get("tool_calls") else "")
                            for m in messages[1:]
                        )[:20_000],
                    },
                ]
            )
            say("status", f"compacted in {time.monotonic() - ct0:.1f}s")
            tool_names = ", ".join(sorted(TOOLS)) or "none"
            messages = [
                messages[0],
                {
                    "role": "user",
                    "content": (
                        f"[compacted]\n{summary['content']}\n\nContinue the task. "
                        f"Tools available: {tool_names}. Call one if you still need it; "
                        f"answer in prose when you are done."
                    ),
                },
            ]
        messages = _fit_messages(messages, limit)
        t0 = time.monotonic()
        first_t: list[float] = []

        say("status", "thinking\u2026")  # live cue: the model call can take a while

        # P5: on the first model call, hide the tool surface from tasks that
        # cannot need one; later iterations always see tools (mid-episode).
        # Voice mode opts out: speaking IS a tool call here, so hiding the tool
        # surface makes the agent answer silently. The guard keys off keywords
        # in the task text, and spoken requests routinely miss them -- "set an
        # alarm at 7 am" matches no hint, so without this the agent would think
        # with no way to say anything.
        call_tools = [
            schema
            for name, (schema, _) in TOOLS.items()
            if not (cfg.get("background") and name in DELEGATION_TOOLS)
        ]
        # Voice mode and delegated workers both opt out: a task that was handed
        # to start_task is by definition work, and "Create a polished Flappy
        # Bird game" matches no hint word, so the guard would hand the worker no
        # tools and it would answer in prose having built nothing.
        if (
            iteration == 1
            and not cfg.get("voice")
            and not cfg.get("background")
            and not _task_needs_tools(task)
        ):
            call_tools = None
        msg, usage = await provider.chat(
            messages,
            call_tools,
            stream_cb=_make_first_token_cb(first_t),
        )
        model_s = time.monotonic() - t0
        prompt_tokens = usage.get("prompt_tokens", 0)
        if on_event:
            on_event(
                "usage",
                json.dumps(
                    {
                        "prompt_tokens": prompt_tokens,
                        "model_s": round(model_s, 3),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "decode_s": round(model_s - (first_t[0] - t0) if first_t else model_s, 3),
                    }
                ),
            )
        calls = msg.get("tool_calls") or []
        if not calls:
            answer = msg["content"]
            if verbose:
                say("status", f"stop: no tool calls, {len(answer)} chars of content")
            break
        raw_calls = msg.pop("raw_tool_calls", None) or [
            {"function": {"name": c["name"], "arguments": json.dumps(c["arguments"])}} for c in calls
        ]
        for c, raw in zip(calls, raw_calls):
            c["id"] = raw.get("id") or c.get("id") or c["name"]

        async def run_one(c):
            say("tool", f"{c['name']}({json.dumps(c['arguments'])[:120]})")
            result = _cap_result(await dispatch(c["name"], c["arguments"]))
            say("result", result[:200].replace("\n", " "))
            return c, result

        results = list(await asyncio.gather(*(run_one(c) for c in calls if c["name"] in READ_ONLY)))
        results += [await run_one(c) for c in calls if c["name"] not in READ_ONLY]  # writes: serial, in order
        # T1.2: an iteration whose tool results were ALL machine errors is a
        # free retry (bounded at 3 consecutive) so one bad call does not burn
        # the loop budget; after 3, counting resumes and max_iter terminates.
        def _is_err(r):
            try:
                d = json.loads(r[1])
                return isinstance(d, dict) and "error" in d
            except Exception:
                return False
        if results and all(_is_err(r) for r in results):
            consecutive_errors += 1
            if consecutive_errors < 3:
                say("status", f"tool errors ({consecutive_errors}/3 free retries)")
                continue
        else:
            consecutive_errors = 0
        iteration += 1
        # Appended here, not before the tool calls run: a "free retry" above
        # (`continue`) must never leave this message in `messages` without its
        # tool-result siblings immediately after -- Anthropic/Vertex rejects a
        # `tool_use` block with no `tool_result` in the very next message
        # (measured: orphaned assistant tool_calls message from the dangling
        # append, surfaced as "tool_use ids were found without tool_result
        # blocks"). Appending both only on this surviving path keeps that
        # invariant automatically, including for any future early-exit added
        # to this loop.
        messages.append({"role": "assistant", "content": msg["content"], "tool_calls": raw_calls})
        for c, result in results:
            # tool result for either family (ollama: tool_name; OpenAI: tool_call_id)
            messages.append(
                {
                    "role": "tool",
                    "tool_name": c["name"],
                    "name": c["name"],
                    "tool_call_id": c.get("id") or c["name"],
                    "content": result,
                }
            )
    if answer:
        return answer
    # Two exit paths: a real cap (max_iter > 0) vs. the model ending the turn
    # with no tool calls AND no content (empty final message).
    return (
        f"(iteration cap reached after {max_iter} turns without a final answer)"
        if max_iter > 0
        else "(model ended the turn with no tool calls and no final answer \u2014 rephrase or retry)"
    )


# S7.5. Session memory — each REPL line is a fresh agent_loop task; prior turns are replayed.
HISTORY_FILE = "session_history.jsonl"
HISTORY_KEEP_FULL = 3  # most recent answers kept verbatim
HISTORY_DECAY_CHARS = 800  # older answers cut to this many chars


def _est_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token). Good enough to size a history budget."""
    return max(1, (len(text) + 3) // 4)


def _msg_text(m: dict) -> str:
    """One message as flat text: its content plus any tool calls."""
    return (m.get("content") or "") + (json.dumps(m["tool_calls"]) if m.get("tool_calls") else "")


def _messages_tokens(messages: list[dict]) -> int:
    """Estimated prompt size of a message list (look-ahead: sees pending tool results)."""
    return sum(_est_tokens(_msg_text(m)) for m in messages)


def _fit_messages(messages: list[dict], limit: int) -> list[dict]:
    """Last-resort shrink: drop the largest whole exchanges (assistant + its tool
    results, never the task/final message) until the estimate fits `limit`."""
    out = list(messages)
    while _messages_tokens(out) > limit and len(out) > 2:
        units, i = [], 1
        while i < len(out) - 1:  # never the final message
            j = i + 1
            if out[i].get("tool_calls"):
                while j < len(out) and out[j].get("role") == "tool":
                    j += 1
            units.append((_messages_tokens(out[i:j]), i, j))
            i = j
        candidates = [u for u in units if not (u[1] == 1 and out[1].get("role") == "user")]
        if not candidates:
            break  # only the task is left; nothing safe to drop
        _, lo, hi = max(candidates, key=lambda u: u[0])
        del out[lo:hi]
    return out


def _history_path(cfg: dict) -> Path:
    """Path to the session's JSONL history file."""
    return Path(cfg["data_path"]) / HISTORY_FILE


def load_session_history(cfg: dict) -> list[dict]:
    """The session's prior exchanges: [{"task", "response"}, ...]."""
    p = _history_path(cfg)
    out: list[dict] = []
    for line in p.read_text(errors="replace").splitlines() if p.is_file() else []:
        try:
            if (rec := json.loads(line)) and rec.get("task"):
                out.append(rec)
        except json.JSONDecodeError:
            continue
    return out


def append_session_history(cfg: dict, task: str, response: str) -> None:
    """Append one exchange to the history file (one JSON per line)."""
    p = _history_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.open("a", encoding="utf-8").write(json.dumps({"task": task, "response": response}) + "\n")


def build_session_messages(history: list, budget_tokens: int | None = None) -> list:
    """Replayed history as user/assistant pairs, oldest dropped until budget_tokens fits.

    Older answers are cut to their opening (questions stay whole) before the
    budget pass, so a long session degrades gracefully instead of overflowing.
    """
    n, msgs = len(history), []
    for i, rec in enumerate(history):
        r = rec.get("response", "")
        if i < n - HISTORY_KEEP_FULL and len(r) > HISTORY_DECAY_CHARS:
            r = r[:HISTORY_DECAY_CHARS] + "\n...[earlier answer trimmed]"
        msgs.append({"role": "user", "content": rec.get("task", "")})
        if r:
            msgs.append({"role": "assistant", "content": r})
    if budget_tokens and budget_tokens > 0:
        total = sum(_est_tokens(m.get("content", "") or "") for m in msgs)
        while total > budget_tokens and len(msgs) > 2:
            total -= _est_tokens(msgs[0].get("content", "") or "")
            msgs.pop(0)
    return msgs


# S8. Text UI — renders loop events for the human; one-way display, never talks to the model.
def ui_banner(cfg: dict) -> None:
    """Print the startup banner (host, model, dir, tool count)."""
    from rich import box
    from rich.console import Console
    from rich.panel import Panel

    s = cfg["serving"]
    Console().print(
        Panel(
            f"[bold]baby-onit[/] — tiny agent harness distilled from onit\n"
            f"host   [cyan]{s['host']}[/]\nmodel  [cyan]{s['model'] or '(auto)'}[/]\n"
            f"dir    [cyan]{cfg['data_path']}[/]\n"
            f"tools  [cyan]{len(TOOLS)}[/] · \\quit to exit · \\help for commands",
            box=box.ROUNDED,
            border_style="blue",
        )
    )


class TurnUI:
    """Per-turn status: tool lines print live, erased once the answer lands."""

    def __init__(self, console, provider, cfg: dict) -> None:
        """Set up per-turn state; console=None means non-TTY (no live output)."""
        self.model = getattr(provider, "model", None) or "?"
        self.provider = _provider_label(getattr(provider, "host", ""))
        self.data_path = cfg["data_path"]
        self.is_tty = bool(console) and sys.stdout.isatty()
        self.width = max(20, getattr(console, "width", 80))
        self._printed: list[str] = []
        self.prompt_tokens = self.completion_tokens = 0
        self.model_s = self.decode_s = 0.0

    def event(self, kind: str, text: str) -> None:
        """Handle one agent_loop event: accumulate usage or print a live line."""
        if kind == "usage":
            try:
                u = json.loads(text)
            except (ValueError, TypeError):
                return
            for k in ("prompt_tokens", "completion_tokens", "model_s", "decode_s"):
                setattr(self, k, getattr(self, k) + u.get(k, 0))
            return
        if not self.is_tty:
            return
        head = " ".join((f"call {text}" if kind == "tool" else text).split())[: self.width - 1]
        print(head, flush=True)
        self._printed.append(head)

    def finish(self) -> None:
        """Erase the intermediate tool lines once the final answer is printed."""
        if self.is_tty and (n := len(self._printed)):  # up n, clear each, back down
            print(f"\033[{n}A" + "\033[2K\n" * (n - 1) + "\033[2K", end="", flush=True)
        self._printed = []

    def footer(self) -> str:
        """One-line stats footer: model, provider, dir, tokens, tok/s, time."""
        bits = [
            f"model [cyan]{self.model}[/]",
            f"provider [cyan]{self.provider}[/]",
            f"dir [cyan]{self.data_path}[/]",
            (
                f"{self.prompt_tokens + self.completion_tokens:,} tok "
                f"({self.prompt_tokens:,} in / {self.completion_tokens:,} out)"
            ),
        ]
        if self.decode_s > 0.05:
            bits.append(f"{self.completion_tokens / self.decode_s:.1f} tok/s")
        return " · ".join(bits + [f"{self.model_s:.1f}s"])


def _provider_label(host: str) -> str:
    """Short human label for the endpoint, like onit's footer."""
    h = (host or "").lower()
    if "api.ollama.com" in h:
        return "ollama-cloud"
    if "api.anthropic.com" in h:
        return "claude"
    if "localhost" in h or "127.0.0.1" in h:
        return "openai-compat" if h.endswith("/v1") else "ollama"
    return "openai-compat" if "/v1" in h or "openai" in h else (h.split("/")[0] or "?")


# S8.5. Voice session — one long-lived loop instead of one loop per turn.
# chat/run call asyncio.run() per turn (see _run_turn), which is fine for a REPL
# but cannot host a microphone: the ASR socket, the audio socket and the MCP
# sessions all have to outlive a single answer, and a barge-in has to cancel a
# turn that is still running. This owns that lifecycle; voice_client drives it.
def _task_overlap(a: str, b: str) -> float:
    """Jaccard overlap of the content words of two task descriptions.

    Deliberately crude: it only needs to catch the model re-delegating the same
    job in slightly different words, not to understand the request.
    """
    stop = {"a", "an", "the", "and", "or", "to", "for", "of", "in", "with", "as",
            "create", "make", "build", "me", "please", "very", "good", "can", "you"}
    def words(t: str) -> set[str]:
        return {w for w in "".join(
            c if c.isalnum() or c.isspace() else " " for c in (t or "").lower()
        ).split() if w not in stop and len(w) > 2}
    wa, wb = words(a), words(b)
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


@dataclass
class _BackgroundTask:
    """One delegated agent_loop and everything needed to report on it."""

    task_id: int
    task: str
    started: float
    handle: Any = None  # asyncio.Task
    state: str = "running"  # running|done|failed|cancelled
    steps: Any = None  # deque of recent (kind, text) events
    result: str | None = None
    error: str | None = None
    history: Any = None  # the worker's own message history, isolated from the foreground

    @property
    def elapsed_s(self) -> float:
        return round(time.monotonic() - self.started, 1)

    def summary(self) -> dict:
        last = list(self.steps or [])[-1] if self.steps else None
        return {
            "task_id": self.task_id,
            "state": self.state,
            "elapsed_s": self.elapsed_s,
            "step": f"{last[0]}: {last[1][:90]}" if last else None,
            **({"result": self.result} if self.result else {}),
            **({"error": self.error} if self.error else {}),
        }


class VoiceSession:
    """A persistent agent session: MCP tools open, one turn in flight, cancellable.

    Usage from the audio client::

        async with VoiceSession(cfg) as vs:
            answer = await vs.ask("what's the weather today")   # cancellable
            ...
            vs.barge_in()        # from the VAD callback; cancels the live turn
    """

    def __init__(self, cfg: dict, on_event: Callable[[str, str], None] | None = None):
        self.cfg = cfg
        self.on_event = on_event
        self.provider = Provider(cfg)
        self.hub = McpHub(cfg)
        self.history: list[dict] = []  # replayed as user/assistant pairs across turns
        self._turn: asyncio.Task | None = None
        self.interrupted = False
        # Delegated work. Keyed by id so the model can refer to a task by number
        # in speech ("task one is still going"). Session-scoped and never
        # persisted, for the same reason as _ALARMS: a task that outlived the
        # process would report into nothing.
        self.tasks: dict[int, _BackgroundTask] = {}
        self._next_task_id = 1

    async def __aenter__(self) -> VoiceSession:
        global _TASK_HOST
        _TASK_HOST = self
        if not self.provider.model:
            self.provider.model = await self.provider.autodetect_model()
            self.provider.remember_model()
        n = await self.hub.open(on_event=self.on_event)
        if self.on_event:
            self.on_event("status", f"mcp: {n} tool(s) from {len(self.hub.sessions)} server(s)")
        return self

    async def __aexit__(self, *exc) -> None:
        global _TASK_HOST
        await self.cancel()
        for task_id in list(self.tasks):
            await self.cancel_background(task_id)
        if _TASK_HOST is self:
            _TASK_HOST = None
        await self.hub.close()
        await self.provider.close_client()

    async def ask(self, task: str, stream_cb: Callable[[str], None] | None = None) -> str | None:
        """Run one turn. Returns None if a barge-in cancelled it."""
        await self.cancel()  # a new utterance always supersedes the previous turn
        self.interrupted = False
        self._turn = asyncio.create_task(
            agent_loop(
                task,
                self.cfg,
                self.provider,
                on_event=self.on_event,
                session_history=self.history,
                verbose=self.cfg.get("verbose"),
            )
        )
        try:
            answer = await self._turn
        except asyncio.CancelledError:
            # Expected on barge-in. Tool results already in self.history stay
            # there, so the next turn still knows what was done.
            return None
        finally:
            self._turn = None
        self.history.append({"task": task, "response": answer})
        return answer

    def barge_in(self) -> None:
        """Cancel the turn in flight. Safe to call from a VAD callback.

        Deliberately does NOT touch background tasks. Interrupting to ask a
        question used to destroy a 60-second build, because the only way to be
        heard was to cancel the loop doing the work. Stopping that work is now
        an explicit `cancel_task`, which is also something the user can ask for
        out loud.
        """
        self.interrupted = True
        if self._turn is not None and not self._turn.done():
            self._turn.cancel()

    # -- delegated work ----------------------------------------------------

    #: Most concurrent background tasks. Each is a full LLM conversation, so
    #: this is a spend ceiling as much as a resource one.
    MAX_BACKGROUND = 2
    #: Only near-verbatim re-delegation is deduplicated. A looser threshold was
    #: tried and rejected: at 0.35, which was needed to catch the observed
    #: rewording ("Create a polished Flappy Bird game as a single-page web app"
    #: vs "Build a Flappy Bird clone web app", 0.44), bag-of-words overlap also
    #: merged "build a snake game in javascript" with "build a tetris game in
    #: javascript" (0.50) and "refactor the auth module" with "add tests to the
    #: auth module" (0.40). Silently refusing a genuinely different request is
    #: worse than one wasted task, so MAX_BACKGROUND is the real guard and this
    #: only catches literal repeats.
    DEDUPE_OVERLAP = 0.9

    def start_background(self, task: str) -> dict:
        """Spawn a detached agent_loop for a long task. Returns immediately.

        Deduplicates against running work. Observed: one "build me a flappy bird
        app" produced **three** tasks, because the foreground loop kept calling
        start_task without knowing it had already delegated. Three concurrent
        builds means triple the tokens and two abandoned at shutdown, so a
        near-identical request returns the task already doing the job.
        """
        from collections import deque

        running = [t for t in self.tasks.values() if t.state == "running"]
        for t in running:
            if _task_overlap(t.task, task) >= self.DEDUPE_OVERLAP:
                return {"task_id": t.task_id, "state": "running", "deduplicated": True,
                        "note": "a task for this is already running; check_task it"}
        if len(running) >= self.MAX_BACKGROUND:
            return {
                "error": "E_TOO_MANY_TASKS",
                "running": [t.task_id for t in running],
                "hint": f"at most {self.MAX_BACKGROUND} background tasks; "
                        "cancel_task one or wait for it",
            }

        task_id = self._next_task_id
        self._next_task_id += 1
        bg = _BackgroundTask(
            task_id=task_id, task=task, started=time.monotonic(),
            steps=deque(maxlen=24), history=[],
        )

        def on_event(kind: str, text: str) -> None:
            if kind in ("tool", "status", "result"):
                bg.steps.append((kind, text))
            if self.on_event:  # keep the foreground UI informed too
                self.on_event(kind, f"[task {task_id}] {text}")

        # The worker gets voice off (it is not the one talking, and the voice
        # prompt is what tells an agent to delegate) and background on, which
        # strips the delegation tools. Without both, workers recurse.
        worker_cfg = {**self.cfg, "voice": False, "background": True}

        async def runner() -> None:
            try:
                bg.result = await agent_loop(
                    task, worker_cfg, self.provider, on_event=on_event,
                    session_history=bg.history, verbose=self.cfg.get("verbose"),
                )
                bg.state = "done"
            except asyncio.CancelledError:
                bg.state = "cancelled"
                raise
            except Exception as e:  # noqa: BLE001 — a failed task must be reportable
                bg.state = "failed"
                bg.error = f"{type(e).__name__}: {e}"
                print(f"(background task {task_id} failed: {bg.error})")

        bg.handle = asyncio.create_task(runner(), name=f"bg-task-{task_id}")
        self.tasks[task_id] = bg
        # Report what else is running: the observed triple-delegation happened
        # because the loop had no way to know it had already started one.
        others = [t.task_id for t in self.tasks.values()
                  if t.state == "running" and t.task_id != task_id]
        out: dict = {"task_id": task_id, "state": "running"}
        if others:
            out["also_running"] = others
            out["note"] = "you already have work running; do not start it again"
        return out

    def background_status(self, task_id: int | None = None) -> dict:
        """Status of one task, or all of them when task_id is omitted."""
        if task_id is None:
            return {"tasks": [t.summary() for t in self.tasks.values()]}
        bg = self.tasks.get(task_id)
        if bg is None:
            return {"error": "E_NO_TASK", "task_id": task_id,
                    "known": sorted(self.tasks)}
        return bg.summary()

    async def cancel_background(self, task_id: int) -> dict:
        bg = self.tasks.get(task_id)
        if bg is None:
            return {"error": "E_NO_TASK", "task_id": task_id, "known": sorted(self.tasks)}
        if bg.handle is not None and not bg.handle.done():
            bg.handle.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await bg.handle
            return {"cancelled": True, "task_id": task_id}
        return {"cancelled": False, "task_id": task_id, "state": bg.state}

    async def cancel(self) -> None:
        """Cancel the live turn and wait for it to actually unwind."""
        t = self._turn
        if t is not None and not t.done():
            t.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await t
        self._turn = None


def _run_turn(provider, coro):
    """One turn on a fresh loop, then close the client's pool on that loop."""

    async def _run():
        out = await coro
        await provider.close_client()
        return out

    return asyncio.run(_run())


_EXIT_CMDS = {"\\quit", "\\q", "\\bye", "\\b", "exit"}


def _read_multiline(console) -> str:
    """Read a task. Enter submits; a trailing backslash continues to the next line.

    A lone '.' also submits (legacy multi-line habit). Exit commands
    (\\quit, \\bye, …) are recognised immediately, before any buffering.
    """
    buf: list = []
    while True:
        try:
            s = input("❯ " if not buf else "")
        except (EOFError, KeyboardInterrupt):
            break
        if not buf and s.strip() in _EXIT_CMDS:
            return s.strip()
        if s.strip() == ".":
            break
        if s.endswith("\\"):  # explicit continuation: drop the marker, keep reading
            buf.append(s[:-1])
            continue
        buf.append(s)
        break  # Enter submits
    return "\n".join(buf).strip()


def ui_chat(cfg: dict) -> None:
    """Interactive REPL: each task is a fresh agent_loop task."""
    from rich.console import Console
    from rich.markdown import Markdown

    console = Console()
    ui_banner(cfg)
    provider = Provider(cfg)
    if not provider.model:
        provider.model = asyncio.run(provider.autodetect_model())
        provider.remember_model()  # persist per-endpoint so setup remembers it
        console.print(f"[dim]auto-detected model: {provider.model}[/]")
    # S7.5: prior turns are replayed only when asked for. Otherwise the file is
    # cleared here rather than merely ignored, so it holds this session alone and
    # a later --resume picks up the last conversation instead of every one ever.
    if cfg.get("session_memory"):
        history = load_session_history(cfg)
        if history:
            console.print(f"[dim]resumed session: {len(history)} prior turn(s)[/]")
    else:
        had = len(load_session_history(cfg))
        _history_path(cfg).unlink(missing_ok=True)
        history = []
        if had:
            console.print(f"[dim]fresh session: cleared {had} prior turn(s) · --resume to keep them[/]")
    seed = [rec["task"] for rec in history if rec.get("task", "").strip()]
    while True:
        try:
            import readline

            if seed and not readline.get_current_history_length():
                [readline.add_history(s) for s in seed]  # seed once: prior tasks
            seed = None  # readline owns history from here on
            line = _read_multiline(console)
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in _EXIT_CMDS:
            break
        if line.startswith("\\"):
            if line == "\\help":
                console.print(
                    "\\model show model · \\host show host · \\key set key · "
                    "\\reset clear session memory · \\quit exit"
                )
                console.print(
                    "[dim]sessions start fresh; `chat --resume` replays the previous one[/]"
                )
            elif line == "\\reset":
                _history_path(cfg).unlink(missing_ok=True)
                history = []
                console.print("[dim]session memory cleared[/]")
            elif line == "\\model":
                console.print(f"model {provider.model or '(none)'} @ {provider.host}")
            elif line == "\\host":
                console.print(provider.host)
            elif line == "\\key":
                if key := getpass.getpass("API key (enter to keep): "):
                    console.print(f"stored in {set_secret(f'endpoint_key:{provider.host}', key)}")
            else:
                continue
        turn = TurnUI(console, provider, cfg)
        try:
            answer = _run_turn(
                provider,
                agent_loop(line, cfg, provider, on_event=turn.event, session_history=history, verbose=cfg.get("verbose")),
            )
        except (KeyboardInterrupt, EOFError):
            turn.finish()
            console.print("[dim]interrupted[/]")
            continue
        turn.finish()
        console.print(Markdown(answer))
        console.print(f"[dim]─ {turn.footer()}[/]")
        history.append({"task": line, "response": answer})
        append_session_history(cfg, line, answer)
    console.print("[dim]bye[/]")


# S9. Setup wizard + CLI — setup writes config.yaml and stores secrets so chat needs zero flags.
PRESET_HOSTS = {
    "ollama": "http://localhost:11434",
    "ollama-cloud": "https://api.ollama.com",
    "vllm": "http://localhost:8000/v1",
    "sglang": "http://localhost:30000/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "vercel": "https://ai-gateway.vercel.sh/v1",
    "openai": "https://api.openai.com/v1",
    "claude": "https://api.anthropic.com/v1",
}


def _print_endpoints(console, title: str = "endpoints") -> None:
    """Table of every known endpoint: key status + remembered model."""
    from rich.table import Table

    tbl = Table(title=title)
    for col in ("endpoint", "key", "model"):
        tbl.add_column(col)
    for ep in known_endpoints():
        tbl.add_row(
            ("[bold]" if ep["active"] else "") + ep["host"],
            f"••••{ep['key'][-4:]}" if ep["key"] else "[dim]none[/]",
            ep["model"] or "[dim](auto)[/]",
        )
    console.print(tbl)


def _print_secrets(console, title: str) -> None:
    """Table of the optional secrets"""
    from rich.table import Table

    tbl = Table(title=title)
    for col in ("name", "status", "env"):
        tbl.add_column(col)
    for name in SECRET_NAMES:
        val = get_secret(name)
        tbl.add_row(name, f"[green]set (••••{val[-4:]})[/]" if val else "[dim]not set[/]", SECRET_ENV.get(name, ""))
    console.print(tbl)


def cmd_setup(args) -> None:
    """Interactive setup wizard: pick endpoint, set key, model, data path, tokens."""
    from rich.console import Console

    console = Console()
    dest = Path(args.config or Path.home() / ".baby-onit" / "config.yaml").expanduser()
    if getattr(args, "reset", False):
        prev = {}
        console.print("[yellow]--reset: starting fresh, all previous values ignored[/]")
    else:
        prev = _load_yaml(dest)
    prev_serving = prev.get("serving") or {}
    if getattr(args, "show", False):  # print config + secrets, no prompts
        cfg = load_config(args.config)
        console.print(f"[bold]config[/] : {args.config or '~/.baby-onit/config.yaml'}")
        for k in (
            "provider",
            "host",
            "model",
            "think",
            "max_tokens",
            "max_chat_iterations",
            "max_context_tokens",
            "temperature",
            "top_p",
        ):
            console.print(f"  {k:<22} {cfg['serving'].get(k, DEFAULTS['serving'].get(k))}")
        console.print(f"  {'data_path':<22} {cfg['data_path']}")
        _print_endpoints(console)
        _print_secrets(console, "secrets")
        return
    d_host = prev_serving.get("host") or DEFAULTS["serving"]["host"]
    d_model, d_path = prev_serving.get("model") or "", prev.get("data_path") or DEFAULTS["data_path"]
    norm_host = normalize_host(d_host, is_ollama_host(d_host))
    d_key = get_secret(f"endpoint_key:{norm_host}")
    console.print(
        "[bold]baby-onit setup[/] — enter a number, paste a URL, enter to keep current, "
        "or 'd' to delete the current endpoint's key + remembered model"
    )
    console.print(
        f"  current: host={d_host}  model={d_model or '(auto)'}  "
        f"key={'••••' + d_key[-4:] if d_key else '(none)'}  dir={d_path}"
    )
    for i, (name, url) in enumerate(PRESET_HOSTS.items(), 1):
        console.print(f"  {i}. {name:<13} {url}")
    _print_endpoints(console)
    choice, presets = input("endpoint: ").strip(), list(PRESET_HOSTS.values())
    if choice == "d":  # delete/clear the current endpoint's configuration
        removed = forget_endpoint(d_host)
        console.print(
            f"  {d_host}: " + (", ".join(f"deleted {r}" for r in removed) if removed else "nothing was stored")
        )
        choice = input("endpoint (enter to keep current, or pick another): ").strip()
    host = (
        d_host
        if not choice
        else (
            presets[int(choice) - 1]
            if choice.isdigit() and 1 <= int(choice) <= len(presets)
            else PRESET_HOSTS.get(choice, choice) or d_host
        )
    )
    if not re.match(r"^https?://", host):
        raise SystemExit(f"{host!r} is not a valid endpoint URL (need http:// or https://)")
    norm = normalize_host(host, is_ollama_host(host))
    if norm != norm_host:  # switched endpoints: prefill that endpoint's remembered model
        d_model = next((e["model"] for e in known_endpoints() if e["host"] == norm and e["model"]), "")
    prev_key = get_secret(f"endpoint_key:{norm}")
    key = getpass.getpass(f"API key (enter to keep {'••••' + prev_key[-4:] if prev_key else 'none'}): ")
    if key:
        console.print(f"  stored in {set_secret(f'endpoint_key:{norm}', key)}")
    elif not prev_key:
        console.print("[yellow]  no API key set — only keyless endpoints will work[/]")
    model = input(f"model [{d_model or 'auto-detect'}]: ").strip() or d_model
    data_path = input(f"data path [{d_path}]: ").strip() or d_path
    cfg = {**prev, "data_path": data_path, "serving": {**prev_serving, "host": host, "model": model}}
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(yaml.safe_dump(cfg, sort_keys=False))
    if model:  # remember this endpoint's model for the next setup
        models = _load_models(dest.parent)
        models[norm] = model
        _save_models(models, dest.parent)
    console.print(f"[green]wrote {dest}[/] — run: baby-onit chat")
    _print_secrets(console, "optional keys")
    for name in SECRET_NAMES:
        prev_tok = get_secret(name)
        tok = getpass.getpass(
            f"{name} (enter to keep {'••••' + prev_tok[-4:] if prev_tok else 'none'}, 'd' to delete): "
        ).strip()
        if tok == "d":
            if prev_tok:
                _keyring(name, "")
                _file_set(name, None)  # clear keychain + fallback file
            console.print(f"  {name}: {'deleted' if prev_tok else 'not set'}")
        elif tok:
            console.print(f"  {name}: stored in {set_secret(name, tok)}")
        else:
            console.print(f"  {name}: {'kept' if prev_tok else 'not set (optional)'}")
    _print_endpoints(console, title="endpoints (key + remembered model)")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse args, load config, dispatch to setup/chat/run."""
    ap = argparse.ArgumentParser(prog="baby-onit", description="Tiny single-file agent harness distilled from onit.")
    # Defaults are stated in the help text (onit cli.py convention), not via
    # default=, so an absent flag leaves the config value untouched.
    for flag, kw in [
        ("--config", {"help": "path to config.yaml (default: ~/.baby-onit/config.yaml)"}),
        ("--host", {"help": "endpoint URL (default: http://localhost:11434)"}),
        ("--model", {"help": "model id (default: auto-detect from the endpoint)"}),
        ("--data-path", {"help": "working directory for tools (default: ~/baby-sandbox)"}),
        ("--no-think", {"action": "store_true", "help": "disable thinking mode (default: on)"}),
        ("--max-iterations", {"type": int, "help": "cap on agent-loop turns (default: -1 = no cap)"}),
        (
            "--max-context-tokens",
            {
                "type": int,
                "help": "compaction trigger in tokens (default: 262144; e.g. 1000000 for a 1M-context model)",
            },
        ),
        (
            "--history-budget-tokens",
            {
                "type": int,
                "help": (
                    "cap on replayed session history "
                    "in tokens (default: 16000; lower it if a provider flakes on large payloads)"
                ),
            },
        ),
        ("--max-tokens", {"type": int, "help": "max output tokens per response (default: 32768)"}),
    ]:
        ap.add_argument(flag, **kw)
    sub = ap.add_subparsers(dest="cmd")
    p_setup = sub.add_parser("setup", help="configure endpoint + secrets")
    p_setup.add_argument("--show", action="store_true", help="print config + secrets, no prompts")
    p_setup.add_argument("--reset", action="store_true", help="ignore existing config, start fresh")
    p_run = sub.add_parser("run", help="run one task, print the answer, exit")
    p_run.add_argument("task", nargs="+")
    # The same flag after the subcommand too. default=SUPPRESS: when the flag
    # is absent the subparser leaves the namespace alone instead of
    # overwriting the top-level value with its own default.
    p_chat = sub.add_parser("chat", help="interactive chat")
    p_chat.add_argument(
        "--resume",
        action="store_true",
        default=argparse.SUPPRESS,
        help="replay prior turns from session_history.jsonl (default: start fresh)",
    )
    for p in (
        p_chat,
        p_run,
    ):
        p.add_argument(
            "--voice",
            action="store_true",
            default=argparse.SUPPRESS,
            help="spoken-style prompt + always-on tool surface (needs an mcp.servers TTS entry)",
        )
        p.add_argument(
            "--max-context-tokens",
            type=int,
            default=argparse.SUPPRESS,
            help="compaction trigger in tokens (default: 262144)",
        )
        p.add_argument(
            "--max-tokens",
            type=int,
            default=argparse.SUPPRESS,
            help="max output tokens per response (default: 32768)",
        )
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    for k, v in (
        ("host", args.host),
        ("model", args.model),
        ("think", False if args.no_think else None),
        ("max_chat_iterations", args.max_iterations),
        ("max_context_tokens", getattr(args, "max_context_tokens", None)),
        ("history_budget_tokens", getattr(args, "history_budget_tokens", None)),
        ("max_tokens", getattr(args, "max_tokens", None)),
    ):
        if v is not None:
            cfg["serving"][k] = v
    if getattr(args, "voice", False):  # voice is top-level, not a serving knob
        cfg["voice"] = True
    if getattr(args, "resume", False):
        cfg["session_memory"] = True
    if args.data_path:
        cfg["data_path"] = str(Path(args.data_path).expanduser())
    if args.cmd == "setup":
        cmd_setup(args)
    elif args.cmd == "chat":
        ui_chat(cfg)
    elif args.cmd == "run":
        provider = Provider(cfg)
        if not provider.model:
            provider.model = asyncio.run(provider.autodetect_model())
            provider.remember_model()
        turn = TurnUI(None, provider, cfg)
        print(
            _run_turn(
                provider,
                agent_loop(" ".join(args.task), cfg, provider, on_event=turn.event, verbose=cfg.get("verbose")),
            )
        )
        turn.finish()
        if sys.stderr.isatty():
            print(f"─ {turn.footer()}", file=sys.stderr)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
