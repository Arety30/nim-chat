"""Arety CLI — presentation layer. THIS IS THE FILE TO DESIGN.

cli.py owns argument parsing, commands and the agent loop; everything the user *sees* is here:

* Theme      – colours, symbols and labels (override any of them without code in
               ~/.config/nim-chat/cli-theme.json, e.g. {"colors": {"user": "35"}, "symbols": {"tool": "⚙"}})
* banner()   – what prints when an interactive session starts
* prompt()   – the input prompt
* Renderer   – turns harness events (start/step/reasoning/delta/tool_call/tool_result/sense/notice/error/stopped/done)
               into terminal output. Change how streaming text, tool cards and results look here.
* ask_approval() – the y/n/always question for guarded tools
* HELP_TEXT, print_models/status/conversations/conversation – the read-only views

Keep the function names and signatures; cli.py calls them. Colours are ANSI SGR codes ("1;34" = bold blue).
"""
from __future__ import annotations

import json
import os
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path

APP_NAME = "Arety CLI"
THEME_FILE = Path(os.environ.get("ARETY_CLI_THEME", Path.home() / ".config/nim-chat/cli-theme.json"))


# ----------------------------------------------------------------------------- theme


@dataclass
class Theme:
    colors: dict = field(default_factory=lambda: {
        "brand": "1;35",      # banner title
        "dim": "2",           # meta lines, tool output
        "user": "32",         # "you ›" prompt
        "assistant": "34",    # assistant label in transcripts
        "reasoning": "90",    # streamed thinking
        "tool": "33",         # tool call cards + approval questions
        "tool_output": "2",   # tool result lines
        "sense": "35",        # PaliGemma / Whisper notes
        "notice": "36",
        "error": "31",
        "ok": "32",
        "warn": "33",
        "id": "36",
    })
    symbols: dict = field(default_factory=lambda: {
        "prompt": "you › ",
        "continue": "... ",
        "assistant": "arety › ",
        "reasoning": "💭 ",
        "tool": "▶ ",
        "tool_line": "  │ ",
        "eye": "👁 ",
        "ear": "👂 ",
        "notice": "ⓘ ",
        "error": "✖ ",
        "warn": "⚠ ",
        "meta": "· ",
        "question": "? ",
    })
    labels: dict = field(default_factory=lambda: {
        "stopped": "[stopped]",
        "interrupted": "[interrupted]",
        "more_lines": "… {n} more lines (use --verbose)",
        "continue_hint": "continue with: arety -c {id}",
        "approval": "allow {name}",
        "approval_choices": "  [y]es / [n]o / [a]lways: ",
    })
    tool_result_lines: int = 12      # lines of tool output shown when not --verbose
    tool_result_width: int = 200     # chars per shown line
    arg_width: int = 160             # width of the one-line argument summary on tool cards

    def __post_init__(self):
        if THEME_FILE.exists():
            try:
                data = json.loads(THEME_FILE.read_text())
            except (OSError, json.JSONDecodeError) as e:
                sys.stderr.write(f"[theme ignored: {e}]\n")
                data = {}
            for key in ("colors", "symbols", "labels"):
                getattr(self, key).update(data.get(key) or {})
            for key in ("tool_result_lines", "tool_result_width", "arg_width"):
                if key in data:
                    setattr(self, key, int(data[key]))


THEME = Theme()
COLOR_ON = sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def c(role: str, s: str) -> str:
    """Colour `s` with the theme colour for `role` (no-op when not a tty / NO_COLOR)."""
    code = THEME.colors.get(role)
    return f"\033[{code}m{s}\033[0m" if COLOR_ON and code else s


def sym(name: str) -> str:
    return THEME.symbols.get(name, "")


def label(name: str, **kw) -> str:
    return THEME.labels.get(name, name).format(**kw)


def out(s: str = "", end: str = "\n") -> None:
    sys.stdout.write(s + end)
    sys.stdout.flush()


def err(s: str) -> None:
    sys.stderr.write(s + "\n")
    sys.stderr.flush()


# ----------------------------------------------------------------------------- small formatters


def fmt_args(args: dict, width: int | None = None) -> str:
    width = width or THEME.arg_width
    parts = []
    for k, v in args.items():
        s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        s = s.replace("\n", "⏎ ")
        if len(s) > 90:
            s = s[:87] + "…"
        parts.append(f"{k}={s}")
    line = "  ".join(parts)
    return line if len(line) <= width else line[: width - 1] + "…"


def fmt_tokens(u: dict) -> str:
    t = u.get("total_tokens") or (u.get("prompt_tokens", 0) + u.get("completion_tokens", 0))
    return f"{t / 1000:.1f}k tok" if t >= 1000 else f"{t} tok"


def fmt_when(ts: float | None) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts or 0))


# ----------------------------------------------------------------------------- session chrome


def banner(model: str, cid: str, n_prev: int, tools_on: bool, approve_mode: str, recent: list[dict]) -> None:
    """Printed once when an interactive session starts."""
    meta = f" {sym('meta')}{model} {sym('meta')}chat {cid}"
    if n_prev:
        meta += f" {sym('meta')}{n_prev} earlier messages"
    meta += f" {sym('meta')}tools {'on' if tools_on else 'off'} ({approve_mode}) {sym('meta')}/help"
    out(c("brand", APP_NAME) + c("dim", meta))
    for m in recent:
        who = c("user", "you") if m["role"] == "user" else c("assistant", "arety")
        out(c("dim", f"  {who}: {m['content'][:120].replace(chr(10), ' ')}"))


def prompt() -> str:
    """Input prompt; \\001/\\002 tell readline the escape codes take no width."""
    p = sym("prompt")
    code = THEME.colors.get("user")
    return f"\001\033[{code}m\002{p}\001\033[0m\002" if COLOR_ON and code else p


def continue_prompt() -> str:
    return c("dim", sym("continue"))


def session_notice(text: str) -> None:
    """Feedback for slash commands (/model, /tools, …)."""
    out(c("dim", text))


def session_error(text: str) -> None:
    out(c("error", text))


def ask_approval(name: str, args: dict) -> str:
    """Show a guarded tool call and return the raw answer the user typed (lower-cased)."""
    out()
    out(c("tool", f"{sym('question')}{label('approval', name=name)}") + "  " + c("dim", fmt_args(args, 400)))
    code = str(args.get("command", args.get("code", "")))
    if name in ("run_shell", "python_exec") and "\n" in code:
        out(c("dim", textwrap.indent(code, "    ")))
    try:
        return input(c("tool", label("approval_choices"))).strip().lower()
    except EOFError:
        return "n"


HELP_TEXT = """\
/help              this help
/new               start a new conversation
/list              recent conversations        /load ID   resume one
/model [NAME]      show or switch the model    /models    list featured models
/tools [a,b|all|none]  show or set enabled tools
/system [TEXT|@file]   show or set the system prompt (for the next turns)
/attach PATH …     attach files to the next message (images → PaliGemma, audio → Whisper)
/yes /ask /deny    tool approval mode          /think     toggle model thinking
/verbose           toggle full tool output     /reasoning toggle showing reasoning
/memory            show long-term memory       /status    key pool status
/id                print this conversation id  /quit      exit (also Ctrl-D)
Multi-line input: end a line with \\ to continue, or wrap in triple quotes (\"\"\")."""


# ----------------------------------------------------------------------------- event rendering


class Renderer:
    """Turns harness events into terminal output. One instance per session; reset() per turn."""

    def __init__(self, verbose: bool = False, quiet: bool = False, show_reasoning: bool = True):
        self.verbose, self.quiet, self.show_reasoning = verbose, quiet, show_reasoning
        self.reset()

    def reset(self) -> None:
        self.in_reasoning = False
        self.printed = False
        self.final: list[str] = []

    def final_text(self) -> str:
        return "".join(self.final).strip()

    def _end_reasoning(self) -> None:
        if self.in_reasoning:
            out()
            self.in_reasoning = False

    def handle(self, ev: dict) -> None:
        fn = getattr(self, "on_" + ev["type"], None)
        if fn:
            fn(ev)

    def on_start(self, ev: dict) -> None:
        if self.verbose:
            out(c("dim", f"{sym('meta')}chat {ev['conversation_id']} {sym('meta')}{ev['model']}"))

    def on_step(self, ev: dict) -> None:
        if self.verbose and ev["n"] > 1:
            out(c("dim", f"{sym('meta')}step {ev['n']}"))

    def on_delta(self, ev: dict) -> None:
        self.final.append(ev["text"])
        if self.quiet:
            return
        self._end_reasoning()
        out(ev["text"], end="")
        self.printed = True

    def on_reasoning(self, ev: dict) -> None:
        if self.quiet or not self.show_reasoning:
            return
        if not self.in_reasoning:
            out(c("reasoning", sym("reasoning")), end="")
            self.in_reasoning = True
        out(c("reasoning", ev["text"]), end="")

    def on_tool_call(self, ev: dict) -> None:
        if self.quiet:
            return
        self._end_reasoning()
        if self.printed:
            out()
            self.printed = False
        out(c("tool", f"{sym('tool')}{ev['name']}") + "  " + c("dim", fmt_args(ev["arguments"])))

    def on_tool_result(self, ev: dict) -> None:
        if self.quiet:
            return
        lines = ev["output"].rstrip().splitlines() or ["(empty)"]
        limit = None if self.verbose else THEME.tool_result_lines
        shown = lines if limit is None else lines[:limit]
        w = THEME.tool_result_width
        for ln in shown:
            out(c("tool_output", sym("tool_line") + (ln if self.verbose or len(ln) <= w else ln[: w - 3] + "…")))
        if limit is not None and len(lines) > limit:
            out(c("tool_output", sym("tool_line") + label("more_lines", n=len(lines) - limit)))

    def on_sense(self, ev: dict) -> None:
        if self.quiet:
            return
        if ev["kind"] == "vision":
            out(c("sense", f"{sym('eye')}{ev.get('name')}: ") + c("dim", ev.get("description", "")[:300]))
        elif ev["kind"] == "audio":
            out(c("sense", f"{sym('ear')}{ev.get('name')}: ") + c("dim", (ev.get("transcript") or "(no speech)")[:300]))
        else:
            out(c("error", f"{sym('warn')}{ev.get('name')}: {ev.get('error')}"))

    def on_notice(self, ev: dict) -> None:
        if not self.quiet:
            self._end_reasoning()
            out(c("notice", f"{sym('notice')}{ev['text']}"))

    def on_error(self, ev: dict) -> None:
        self._end_reasoning()
        err(c("error", f"{sym('error')}{ev['message']}"))

    def on_stopped(self, ev: dict) -> None:
        self._end_reasoning()
        out(c("dim", "\n" + label("stopped")))

    def on_done(self, ev: dict) -> None:
        self._end_reasoning()
        if self.quiet:
            out(self.final_text())
            return
        if self.printed:
            out()
        u = ev.get("usage") or {}
        n = ev.get("steps", 0)
        bits = [f"{n} step{'s' if n != 1 else ''}", f"{ev.get('seconds', 0)}s"]
        if u:
            bits.insert(0, fmt_tokens(u))
        out(c("dim", sym("meta") + f" {sym('meta')}".join(bits)))


# ----------------------------------------------------------------------------- read-only views


def print_models(info: dict, omni_id: str) -> None:
    out(c("brand", "featured"))
    for m in info["featured"]:
        tag = "  " + c("dim", "(default; DeepSeek brain + PaliGemma eyes + Whisper ears)") if m == omni_id else ""
        out(f"  {m}{tag}")
    if info["all"]:
        out(c("brand", f"all chat models ({len(info['all'])})"))
        for m in info["all"]:
            out(f"  {m}")
    else:
        out(c("dim", "(could not fetch the model list from NIM; featured list shown)"))


def print_status(keys: list[dict], slow: dict, meta: dict) -> None:
    for k in keys:
        cool = c("warn", f"cooling {k['cooling_for']}s") if k["cooling_for"] else c("ok", "ready")
        out(f"  {k['key']}  {cool}  ok={k['ok']} err={k['err']}" + (f"  last={k['last_status']} {k['last_err'][:80]}" if k["last_err"] else ""))
    if slow:
        out(c("warn", "  slow brains: " + ", ".join(f"{m} ({s}s)" for m, s in slow.items())))
    out(c("dim", f"  keys {meta['keys']} {sym('meta')}base {meta['base_url']} {sym('meta')}env {meta['env']}"))
    out(c("dim", f"  workspace {meta['workspace']} {sym('meta')}db {meta['db']}"))


def print_tools(tools: dict[str, str], guarded: set[str]) -> None:
    for name, desc in tools.items():
        mark = c("warn", sym("warn")) if name in guarded else "  "
        out(f"{mark}{c('brand', name):<28} {desc}")
    out(c("dim", f"{sym('warn')}= guarded: asks for approval on a tty (override with -y / --deny)"))


def print_conversations(rows: list[dict]) -> None:
    if not rows:
        out("(no conversations)")
        return
    for r in rows:
        out(f"  {c('id', r['id'])}  {c('dim', fmt_when(r.get('updated') or r.get('created')))}  {(r.get('title') or '')[:60]}  {c('dim', r.get('model') or '')}")


def print_conversation(d: dict) -> None:
    out(c("brand", d.get("title") or "New chat") + c("dim", f"  {sym('meta')}{d['id']} {sym('meta')}{d.get('model')}"))
    for m in d["messages"]:
        if m["role"] == "user":
            content = m["content"] if isinstance(m["content"], str) else " ".join(p.get("text", "[image]") for p in m["content"])
            out(c("user", sym("prompt")) + content)
        elif m["role"] == "assistant":
            if m.get("content"):
                out(c("assistant", sym("assistant")) + m["content"])
            for tc in m.get("tool_calls") or []:
                out(c("tool", f"{sym('tool')}{tc['name']}") + "  " + c("dim", tc["arguments"][:200].replace("\n", " ")))
        elif m["role"] == "tool":
            out(c("tool_output", sym("tool_line") + m["content"][:300].replace("\n", "\n" + sym("tool_line"))))
