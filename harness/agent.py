"""Agent: the nim-chat agentic loop, decoupled from FastAPI.

Everything model/tool related is imported from server.py (untouched): key pool + failover,
streaming with reasoning, the tool registry, prompts, history trimming and the SQLite
conversation store.  This module adds:

* `Agent.run()`  – async event stream identical in shape to the web UI's SSE events
* `Agent.ask()`  – convenience: final answer as a string (`ask_sync` for plain scripts)
* approval hook  – `approve(name, args)` decides whether a guarded tool may run
* custom tools   – `Agent.register_tool()` / `ToolDef` for callers to add their own
* delegate_task  – a sub-agent tool so the model can fan work out to a fresh agent
* attachments    – files run through PaliGemma (images) / Whisper (audio) like the web UI
* stop()         – cooperative cancellation, partial turns are still persisted
"""
from __future__ import annotations

import asyncio
import inspect
import json
import mimetypes
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server as srv  # noqa: E402  (importing builds the FastAPI app but never starts uvicorn)

Approver = Callable[[str, dict], bool | str | Awaitable[bool | str]]

GUARDED_TOOLS = {"run_shell", "python_exec", "write_file", "edit_file", "delegate_task"}
AUDIO_EXT = (".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mp4", ".flac", ".opus")


class AgentError(RuntimeError):
    pass


@dataclass
class ToolDef:
    """A tool the caller adds on top of server.TOOLS. `fn` may be sync or async."""

    name: str
    fn: Callable[..., Any]
    desc: str
    params: dict = field(default_factory=dict)
    required: list[str] = field(default_factory=list)

    def spec(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.desc,
                "parameters": {"type": "object", "properties": self.params, "required": self.required},
            },
        }


def list_tools() -> dict[str, str]:
    """name -> description for every built-in tool (server tools + harness extras)."""
    out = {name: t["desc"] for name, t in srv.TOOLS.items()}
    out["delegate_task"] = DELEGATE_DESC
    return out


DELEGATE_DESC = (
    "Hand a self-contained sub-task to a fresh sub-agent (same model and tools, empty context) and get back "
    "its final answer. Use it for parallelisable or noisy work (long research, big file sweeps) so the main "
    "context stays small. Give it every detail it needs; it cannot see this conversation."
)


class Agent:
    """One conversation with the Arety agent loop.

    Parameters mirror the web UI's per-chat settings. `conversation_id` resumes a stored chat
    (shared with the web UI); `save=False` keeps the conversation in memory only.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        system_prompt: str | None = None,
        tools: list[str] | None = None,
        agent: bool = True,
        max_steps: int | None = None,
        temperature: float = 0.6,
        max_tokens: int = 8192,
        thinking: bool = True,
        approve: Approver | None = None,
        guarded: set[str] | None = None,
        save: bool = True,
        conversation_id: str | None = None,
        title_prefix: str = "",
        extra_tools: list[ToolDef] | None = None,
        _depth: int = 0,
    ):
        self.model = model or srv.OMNI_MODEL_ID
        self.system_prompt = system_prompt
        self.enabled_tools = tools  # None → all
        self.agent_on = agent
        self.max_steps = max_steps or srv.MAX_STEPS
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking = thinking
        self.approve = approve
        self.guarded = set(GUARDED_TOOLS if guarded is None else guarded)
        self.save = save
        self.title_prefix = title_prefix
        self._depth = _depth
        self._stop = False
        self._running = False
        self._extra: dict[str, ToolDef] = {}
        for t in extra_tools or []:
            self.register_tool(t)
        if _depth == 0:
            self.register_tool(ToolDef("delegate_task", self._delegate, DELEGATE_DESC,
                                       {"task": {"type": "string", "description": "complete, self-contained instructions"}},
                                       ["task"]))
        self.conv: dict = {}
        if conversation_id:
            existing = srv.conv_get(conversation_id) if save else None
            if existing:
                self.conv = existing
                self.model = existing.get("model") or self.model
                if system_prompt is None and existing.get("system_prompt"):
                    self.system_prompt = existing["system_prompt"]
        if not self.conv:
            self.conv = {"id": conversation_id or uuid.uuid4().hex[:12], "title": "", "model": self.model,
                         "system_prompt": self.system_prompt or "", "settings": {"source": "cli"}, "messages": [],
                         "created": time.time()}

    # ------------------------------------------------------------------ public API

    @property
    def id(self) -> str:
        return self.conv["id"]

    @property
    def messages(self) -> list[dict]:
        return self.conv["messages"]

    def register_tool(self, tool: ToolDef) -> None:
        self._extra[tool.name] = tool

    def unregister_tool(self, name: str) -> None:
        self._extra.pop(name, None)

    def stop(self) -> None:
        self._stop = True

    def reset(self) -> None:
        """Start a new conversation (same settings)."""
        self.conv = {"id": uuid.uuid4().hex[:12], "title": "", "model": self.model, "system_prompt": self.system_prompt or "",
                     "settings": {"source": "cli"}, "messages": [], "created": time.time()}

    async def ask(self, message: str, attachments: list[str] | None = None) -> str:
        """Run a turn and return the final assistant text (tool calls happen silently)."""
        out: list[str] = []
        errors: list[str] = []
        async for ev in self.run(message, attachments):
            if ev["type"] == "delta":
                out.append(ev["text"])
            elif ev["type"] == "error":
                errors.append(ev["message"])
        text = "".join(out).strip()
        if not text and errors:
            raise AgentError(errors[-1])
        return text

    def ask_sync(self, message: str, attachments: list[str] | None = None) -> str:
        return asyncio.run(self.ask(message, attachments))

    async def run(self, message: str, attachments: list[str] | None = None) -> AsyncIterator[dict]:
        """Run one user turn; yields event dicts (same `type`s as the web UI's SSE stream)."""
        if self._running:
            raise AgentError("agent is already running a turn")
        if not srv.KEYS:
            yield {"type": "error", "message": f"no NVIDIA_API_KEYS configured (looked in {srv.ENV_FILE})", "fatal": True}
            return
        self._running, self._stop = True, False
        try:
            async for ev in self._turn(message, attachments or []):
                yield ev
        finally:
            self._running = False

    # ------------------------------------------------------------------ internals

    def _save(self) -> None:
        if self.save:
            srv.conv_save(self.conv)

    def _tool_specs(self) -> list[dict]:
        specs = srv.tool_specs(self.enabled_tools)
        for name, t in self._extra.items():
            if self.enabled_tools is None or name in self.enabled_tools:
                specs.append(t.spec())
        return specs

    async def _run_tool(self, name: str, args: dict) -> str:
        if name in self.guarded and self.approve is not None:
            try:
                verdict = self.approve(name, args)
                if inspect.isawaitable(verdict):
                    verdict = await verdict
            except Exception as e:  # noqa: BLE001
                verdict = f"approval hook failed: {e}"
            if verdict is not True:
                reason = verdict if isinstance(verdict, str) and verdict else "denied by user"
                return f"[{name} not run: {reason}]"
        t = self._extra.get(name)
        if t is None:
            return await srv.run_tool(name, args)
        allowed = set(t.params) | set(t.required)
        args = {k: v for k, v in (args or {}).items() if k in allowed}
        try:
            if inspect.iscoroutinefunction(t.fn):
                return str(await t.fn(**args))
            return str(await asyncio.to_thread(t.fn, **args))
        except TypeError as e:
            return f"[bad arguments: {e}]"
        except Exception as e:  # noqa: BLE001
            return f"[tool error: {type(e).__name__}: {e}]"

    async def _delegate(self, task: str) -> str:
        sub = Agent(self.model, system_prompt=self.system_prompt, tools=self.enabled_tools, max_steps=self.max_steps,
                    temperature=self.temperature, max_tokens=self.max_tokens, thinking=self.thinking,
                    approve=self.approve, guarded=self.guarded, save=False, _depth=self._depth + 1)
        for name, t in self._extra.items():
            if name != "delegate_task":
                sub.register_tool(t)
        used: list[str] = []
        out: list[str] = []
        err = ""
        last_result = ""
        async for ev in sub.run(task):
            if ev["type"] == "delta":
                out.append(ev["text"])
            elif ev["type"] == "tool_call":
                used.append(ev["name"])
            elif ev["type"] == "tool_result":
                last_result = f"{ev['name']} → {ev['output']}"
            elif ev["type"] == "error":
                err = ev["message"]
        text = "".join(out).strip() or (f"[sub-agent produced no answer: {err}]" if err else "[sub-agent produced no answer]")
        trace = f"\n\n[sub-agent used {len(used)} tool calls: {', '.join(dict.fromkeys(used))}]" if used else ""
        if last_result:
            trace += "\n[last tool result: " + srv._clip(last_result, 2000) + "]"
        return srv._clip(text + trace, 20000)

    async def _attach(self, paths: list[str], user_text: str) -> tuple[str, list[dict]]:
        """Describe / transcribe / inline attachments the way the web UI does. Returns (extra text, sense notes)."""
        parts: list[str] = []
        notes: list[dict] = []
        for p in paths:
            fp = Path(p).expanduser()
            if not fp.is_absolute():
                fp = Path.cwd() / fp
            if not fp.exists():
                parts.append(f"\n\n[Attachment {p} not found]")
                continue
            mime = mimetypes.guess_type(fp.name)[0] or ""
            try:
                if mime.startswith("image/"):
                    res = await srv.describe_image(fp.read_bytes(), user_text)
                    notes.append({"kind": "vision", "name": fp.name, **res})
                    parts.append(f"\n\n[Image {fp.name} at {fp}. PaliGemma sees: {res.get('description', '')}"
                                 + (f" | Answer to the user's question: {res['answer']}" if res.get("answer") else "") + "]")
                elif mime.startswith(("audio/", "video/")) or fp.suffix.lower() in AUDIO_EXT:
                    r = await srv.transcribe_bytes(fp.read_bytes(), fp.suffix)
                    notes.append({"kind": "audio", "name": fp.name, "transcript": r["text"], "engine": r["engine"]})
                    parts.append(f"\n\n[Audio {fp.name} at {fp}; Whisper transcript: {r['text'] or '(no speech)'}]")
                else:
                    try:
                        text = fp.read_text(errors="strict")
                    except (UnicodeDecodeError, OSError):
                        parts.append(f"\n\n[Attached file {fp.name} at {fp} ({fp.stat().st_size} bytes, binary)]")
                    else:
                        parts.append(f"\n\n[Attached file {fp.name} at {fp}]\n```\n{text[:40000]}\n```")
            except Exception as e:  # noqa: BLE001
                detail = getattr(e, "detail", None) or f"{type(e).__name__}: {e}"
                notes.append({"kind": "error", "name": fp.name, "error": str(detail)})
                parts.append(f"\n\n[Attachment {fp.name} at {fp}: could not process ({detail})]")
        return "".join(parts), notes

    async def _turn(self, message: str, attachments: list[str]) -> AsyncIterator[dict]:
        d = self.conv
        user_text = (message or "").strip()
        if not user_text and not attachments:
            yield {"type": "error", "message": "empty message", "fatal": True}
            return
        extra, senses = await self._attach(attachments, user_text)
        d["model"] = self.model
        d["messages"].append({"role": "user", "content": user_text + extra, "ts": time.time(), "senses": senses,
                              "attachments": [{"name": Path(a).name, "path": str(Path(a).expanduser().resolve())} for a in attachments]})
        if not d["title"]:
            d["title"] = self.title_prefix + (user_text or Path(attachments[0]).name)[:60]
        self._save()

        is_omni = self.model == srv.OMNI_MODEL_ID
        use_native_vision = srv.vision_capable(self.model) and not is_omni
        brain = srv.OMNI_CHAT_MODEL if is_omni else self.model
        fallback = srv.OMNI_FALLBACK_MODEL if is_omni else None
        first_token_timeout = srv.OMNI_FIRST_TOKEN_TIMEOUT if is_omni else None
        yield {"type": "start", "conversation_id": d["id"], "title": d["title"], "model": self.model}
        for n in senses:
            yield {"type": "sense", **n}
        if fallback and srv._SLOW_UNTIL.get(brain, 0) > time.time():
            mins = int((srv._SLOW_UNTIL[brain] - time.time()) / 60) + 1
            yield {"type": "notice", "text": f"{brain.split('/')[-1]} was unresponsive recently; using {fallback.split('/')[-1]} (retrying it in {mins} min)."}
            brain, first_token_timeout = fallback, None
        sys_prompt = (self.system_prompt or "").strip() or srv.default_system_prompt()
        tools = self._tool_specs() if self.agent_on else []
        use_tools = bool(tools)
        no_think = not self.thinking
        use_no_think = no_think and brain.startswith("nvidia/")
        total_usage: dict[str, int] = {}
        t0 = time.time()
        steps_done = 0
        empty_retries = 0
        assistant: dict | None = None
        try:
            for step in range(self.max_steps):
                if self._stop:
                    yield {"type": "stopped"}
                    break
                payload = {
                    "model": brain,
                    "messages": [{"role": "system", "content": sys_prompt}] + srv.build_history(d["messages"], vision=use_native_vision),
                    "stream": True,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                }
                if use_tools:
                    payload["tools"] = tools
                    payload["tool_choice"] = "auto"
                if use_no_think:
                    payload["chat_template_kwargs"] = {"enable_thinking": False}
                steps_done = step + 1
                yield {"type": "step", "n": steps_done, "brain": brain}
                assistant = {"role": "assistant", "content": "", "reasoning": "", "tool_calls": [], "ts": time.time(), "model": brain}
                error = None
                slow = False
                async for ev in srv.stream_completion(payload, first_token_timeout=first_token_timeout):
                    if self._stop:
                        break
                    t = ev["type"]
                    if t == "slow":
                        slow = True
                        break
                    if t == "content":
                        assistant["content"] += ev["text"]
                        yield {"type": "delta", "text": ev["text"]}
                    elif t == "reasoning":
                        assistant["reasoning"] += ev["text"]
                        yield {"type": "reasoning", "text": ev["text"]}
                    elif t == "tool_calls":
                        assistant["tool_calls"] = ev["calls"]
                    elif t == "usage":
                        for k, v in ev["usage"].items():
                            if isinstance(v, int):
                                total_usage[k] = total_usage.get(k, 0) + v
                    elif t == "error":
                        error = ev
                if slow and fallback and brain != fallback:
                    srv._SLOW_UNTIL[brain] = time.time() + srv.SLOW_BREAKER_SECONDS
                    yield {"type": "notice", "text": f"{brain.split('/')[-1]} gave no answer within {int(first_token_timeout)} s; switching to {fallback.split('/')[-1]}."}
                    brain = fallback
                    use_no_think = no_think and brain.startswith("nvidia/")
                    first_token_timeout = None
                    continue
                if slow:
                    yield {"type": "error", "message": f"{brain} did not respond in time."}
                    break
                if error and error.get("fatal") and error.get("status") in (400, 422) and not assistant["content"]:
                    if use_no_think:
                        use_no_think = False
                        continue
                    if use_tools:
                        use_tools = False
                        yield {"type": "notice", "text": "This model rejected tool definitions; continuing without tools."}
                        continue
                if assistant["content"] or assistant["reasoning"] or assistant["tool_calls"]:
                    d["messages"].append(assistant)
                    self._save()
                elif not error and not self._stop:
                    assistant = None
                    if empty_retries < 1:
                        empty_retries += 1
                        yield {"type": "notice", "text": f"{brain.split('/')[-1]} returned an empty response; retrying."}
                        continue
                    yield {"type": "error", "message": f"{brain} returned an empty response twice."}
                    break
                assistant = None
                if error:
                    yield {"type": "error", "message": error["message"]}
                    break
                calls = d["messages"][-1].get("tool_calls") if d["messages"][-1]["role"] == "assistant" else None
                if not calls:
                    break
                for tc in calls:
                    try:
                        args = json.loads(tc["arguments"] or "{}")
                        if not isinstance(args, dict):
                            args = {}
                        out = None
                    except json.JSONDecodeError:
                        args = {}
                        out = f"[could not parse arguments: {tc['arguments'][:200]}]"
                    yield {"type": "tool_call", "id": tc["id"], "name": tc["name"], "arguments": args}
                    if out is None:
                        out = "[stopped by user]" if self._stop else await self._run_tool(tc["name"], args)
                    d["messages"].append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": out, "ts": time.time()})
                    yield {"type": "tool_result", "id": tc["id"], "name": tc["name"], "output": out}
                self._save()
                if self._stop:
                    yield {"type": "stopped"}
                    break
            else:
                yield {"type": "notice", "text": f"Stopped after {self.max_steps} agent steps."}
        except (asyncio.CancelledError, GeneratorExit):
            # keep whatever streamed so far so the conversation stays consistent
            if assistant and (assistant["content"] or assistant["reasoning"]):
                assistant["tool_calls"] = []
                d["messages"].append(assistant)
            self._save()
            raise
        self._save()
        yield {"type": "done", "conversation_id": d["id"], "title": d["title"], "usage": total_usage,
               "steps": steps_done, "seconds": round(time.time() - t0, 2)}
