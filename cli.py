#!/usr/bin/env python3
"""Arety CLI: command-line interface to the nim-chat agent harness.

    arety                       interactive chat (streaming, tools, slash commands)
    arety "fix the failing test" one-shot task (same as `arety run ...`)
    arety run -q "2+2?"          print only the final answer
    arety run --json "..."       one JSON event per line (pipe into other programs)
    cat notes.md | arety run     prompt from stdin
    arety models | status | tools | list | show ID | rm ID

Logic lives here; everything visual lives in cli_ui.py (design it there).
Shares keys, tools, memory and the conversation store with the web UI (server.py).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cli_ui as ui  # noqa: E402
import server as srv  # noqa: E402
from harness import Agent, GUARDED_TOOLS, list_tools  # noqa: E402

SUBCOMMANDS = ("chat", "run", "models", "status", "tools", "list", "show", "rm")
HISTORY_FILE = srv.DATA_DIR / "cli_history"
TITLE_PREFIX = "⌨ "  # CLI conversations in the shared db


# ----------------------------------------------------------------------------- approval


def make_approver(mode: str):
    """mode: 'yes' (auto-approve), 'ask' (prompt on a tty), 'deny'."""
    if mode == "yes":
        return None
    if mode == "deny":
        return lambda name, args: "tool execution disabled (--deny)"
    state = {"always": False}

    def approver(name: str, args: dict):
        if state["always"]:
            return True
        ans = ui.ask_approval(name, args)
        if ans in ("a", "always"):
            state["always"] = True
            return True
        return ans in ("y", "yes", "")

    return approver


# ----------------------------------------------------------------------------- agent construction


def read_system(spec: str | None) -> str | None:
    if not spec:
        return None
    if spec.startswith("@"):
        return Path(spec[1:]).expanduser().read_text()
    return spec


def build_agent(a: argparse.Namespace, conversation_id: str | None = None) -> Agent:
    tools = None
    if a.no_tools:
        tools = []
    elif a.tools:
        tools = [t.strip() for t in a.tools.split(",") if t.strip()]
        unknown = [t for t in tools if t not in list_tools()]
        if unknown:
            ui.err(ui.c("error", f"unknown tools: {', '.join(unknown)}  (see `arety tools`)"))
            sys.exit(2)
    approve_mode = a.approve or ("ask" if sys.stdin.isatty() else "yes")
    return Agent(
        a.model,
        system_prompt=read_system(a.system),
        tools=tools,
        agent=not a.no_tools,
        max_steps=a.max_steps,
        temperature=a.temperature,
        max_tokens=a.max_tokens,
        thinking=not a.no_think,
        approve=make_approver(approve_mode),
        save=not a.no_save,
        conversation_id=conversation_id,
        title_prefix=TITLE_PREFIX,
    )


async def run_turn(agent: Agent, text: str, attachments: list[str], render: ui.Renderer, as_json: bool = False) -> None:
    render.reset()
    async for ev in agent.run(text, attachments):
        if as_json:
            ui.out(json.dumps(ev, ensure_ascii=False))
        else:
            render.handle(ev)


def last_cli_conversation() -> str | None:
    for c in srv.conv_list():
        if (c.get("title") or "").startswith(TITLE_PREFIX):
            return c["id"]
    return None


def status_data() -> tuple[list[dict], dict, dict]:
    slow = {m: int(t - time.time()) for m, t in srv._SLOW_UNTIL.items() if t > time.time()}
    meta = {"keys": len(srv.KEYS), "base_url": srv.BASE_URL, "env": str(srv.ENV_FILE),
            "workspace": str(srv.WORKSPACE), "db": str(srv.DATA_DIR / "chat.db")}
    return srv.POOL.status(), slow, meta


# ----------------------------------------------------------------------------- commands


def cmd_run(a: argparse.Namespace) -> int:
    text = " ".join(a.prompt).strip()
    if text == "-" or (not text and not sys.stdin.isatty()):
        text = sys.stdin.read().strip()
    if not text and not a.attach:
        ui.err("nothing to do: pass a prompt, pipe one on stdin, or use --attach")
        return 2
    cid = a.continue_id or (last_cli_conversation() if a.continue_last else None)
    agent = build_agent(a, cid)
    render = ui.Renderer(verbose=a.verbose, quiet=a.quiet, show_reasoning=not a.no_reasoning)
    try:
        asyncio.run(run_turn(agent, text, a.attach or [], render, as_json=a.json))
    except KeyboardInterrupt:
        ui.out(ui.c("dim", "\n" + ui.label("interrupted")))
        return 130
    if not a.quiet and not a.json and agent.save:
        ui.out(ui.c("dim", ui.sym("meta") + ui.label("continue_hint", id=agent.id)))
    return 0


def _setup_readline() -> None:
    try:
        import atexit
        import readline
    except ImportError:
        return
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        readline.read_history_file(HISTORY_FILE)
    except OSError:
        pass
    readline.set_history_length(2000)
    atexit.register(lambda: readline.write_history_file(HISTORY_FILE))


def _read_input() -> str | None:
    """One user entry; supports trailing-backslash and triple-quote multi-line. None = EOF."""
    try:
        text = input(ui.prompt())
    except EOFError:
        ui.out()
        return None
    except KeyboardInterrupt:
        ui.out()
        return ""
    if text.strip().startswith('"""'):
        buf = [text.strip()[3:]]
        while True:
            try:
                more = input(ui.continue_prompt())
            except (EOFError, KeyboardInterrupt):
                break
            if more.rstrip().endswith('"""'):
                buf.append(more.rstrip()[:-3])
                break
            buf.append(more)
        text = "\n".join(buf)
    while text.endswith("\\"):
        try:
            text = text[:-1] + "\n" + input(ui.continue_prompt())
        except (EOFError, KeyboardInterrupt):
            text = text[:-1]
            break
    return text.strip()


def cmd_chat(a: argparse.Namespace) -> int:
    _setup_readline()
    cid = a.continue_id or (last_cli_conversation() if a.continue_last else None)
    agent = build_agent(a, cid)
    render = ui.Renderer(verbose=a.verbose, quiet=False, show_reasoning=not a.no_reasoning)
    approve_mode = a.approve or "ask"
    pending: list[str] = []
    recent = [m for m in agent.messages[-4:] if m["role"] in ("user", "assistant") and isinstance(m.get("content"), str) and m["content"]]
    ui.banner(agent.model, agent.id, len(agent.messages), agent.agent_on, approve_mode, recent)
    while True:
        text = _read_input()
        if text is None:
            break
        if not text:
            continue
        if text.startswith("/"):
            try:
                parts = shlex.split(text)
            except ValueError:
                parts = text.split()
            cmd, rest = parts[0].lower(), parts[1:]
            arg = text[len(parts[0]):].strip()
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/help":
                ui.out(ui.HELP_TEXT)
            elif cmd == "/new":
                agent.reset()
                ui.session_notice(f"new conversation {agent.id}")
            elif cmd == "/id":
                ui.out(agent.id)
            elif cmd == "/list":
                ui.print_conversations(srv.conv_list()[:15])
            elif cmd == "/load":
                if not rest:
                    ui.out("usage: /load ID")
                elif not srv.conv_get(rest[0]):
                    ui.session_error("no such conversation")
                else:
                    a.model = None
                    agent = build_agent(a, rest[0])
                    ui.session_notice(f"loaded {agent.id} · {agent.model} · {len(agent.messages)} messages")
            elif cmd == "/model":
                if arg:
                    agent.model = arg
                    agent.conv["model"] = arg
                ui.session_notice(f"model: {agent.model}")
            elif cmd == "/models":
                ui.print_models(asyncio.run(srv.models(refresh=False)), srv.OMNI_MODEL_ID)
            elif cmd == "/tools":
                if arg == "all":
                    agent.enabled_tools, agent.agent_on = None, True
                elif arg == "none":
                    agent.enabled_tools, agent.agent_on = [], False
                elif arg:
                    agent.enabled_tools, agent.agent_on = [t.strip() for t in arg.split(",") if t.strip()], True
                names = list(list_tools()) if agent.enabled_tools is None else agent.enabled_tools
                ui.session_notice("tools: " + (", ".join(names) if agent.agent_on and names else "off"))
            elif cmd == "/system":
                if arg:
                    agent.system_prompt = read_system(arg)
                    agent.conv["system_prompt"] = agent.system_prompt
                ui.session_notice((agent.system_prompt or "(default Arety prompt)")[:800])
            elif cmd == "/attach":
                pending.extend(rest)
                ui.session_notice(f"attached for next message: {', '.join(pending)}")
            elif cmd in ("/yes", "/ask", "/deny"):
                approve_mode = cmd[1:]
                agent.approve = make_approver(approve_mode)
                ui.session_notice(f"tool approval: {approve_mode}")
            elif cmd == "/think":
                agent.thinking = not agent.thinking
                ui.session_notice(f"thinking: {'on' if agent.thinking else 'off'}")
            elif cmd == "/verbose":
                render.verbose = not render.verbose
                ui.session_notice(f"verbose: {'on' if render.verbose else 'off'}")
            elif cmd == "/reasoning":
                render.show_reasoning = not render.show_reasoning
                ui.session_notice(f"show reasoning: {'on' if render.show_reasoning else 'off'}")
            elif cmd == "/memory":
                ui.out(srv.MEMORY_FILE.read_text() if srv.MEMORY_FILE.exists() else "(no memory yet)")
            elif cmd == "/status":
                ui.print_status(*status_data())
            else:
                ui.session_error(f"unknown command {cmd} (try /help)")
            continue
        attachments, pending = pending, []
        try:
            asyncio.run(run_turn(agent, text, attachments, render))
        except KeyboardInterrupt:
            agent.stop()
            ui.out(ui.c("dim", "\n" + ui.label("stopped")))
        ui.out()
    return 0


def cmd_show(a: argparse.Namespace) -> int:
    d = srv.conv_get(a.id)
    if not d:
        ui.err("no such conversation")
        return 1
    if a.json:
        ui.out(json.dumps(d, ensure_ascii=False, indent=2))
    else:
        ui.print_conversation(d)
    return 0


# ----------------------------------------------------------------------------- argparse


def add_agent_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-m", "--model", help=f"model id (default {srv.OMNI_MODEL_ID}; see `arety models`)")
    p.add_argument("-s", "--system", help="system prompt text, or @file")
    p.add_argument("-t", "--tools", help="comma-separated tool names to enable (default: all)")
    p.add_argument("--no-tools", action="store_true", help="plain chat, no agent loop")
    p.add_argument("-c", "--continue", dest="continue_id", metavar="ID", help="resume a stored conversation")
    p.add_argument("-C", "--continue-last", action="store_true", help="resume the most recent CLI conversation")
    p.add_argument("--no-save", action="store_true", help="do not store the conversation in the shared chat db")
    g = p.add_mutually_exclusive_group()
    g.add_argument("-y", "--yes", dest="approve", action="store_const", const="yes", help="run guarded tools without asking")
    g.add_argument("--ask", dest="approve", action="store_const", const="ask", help="ask before guarded tools (default on a tty)")
    g.add_argument("--deny", dest="approve", action="store_const", const="deny", help=f"never run guarded tools ({', '.join(sorted(GUARDED_TOOLS))})")
    p.add_argument("--max-steps", type=int, default=None, help=f"agent loop limit (default {srv.MAX_STEPS})")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--no-think", action="store_true", help="disable model thinking (nvidia/* models)")
    p.add_argument("--no-reasoning", action="store_true", help="hide the reasoning stream")
    p.add_argument("-a", "--attach", action="append", metavar="PATH", help="attach a file (repeatable)")
    p.add_argument("-v", "--verbose", action="store_true", help="full tool output, step markers")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arety", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd")
    pc = sub.add_parser("chat", help="interactive chat (default)")
    add_agent_args(pc)
    pr = sub.add_parser("run", help="one-shot task; prompt from args or stdin")
    add_agent_args(pr)
    pr.add_argument("-q", "--quiet", action="store_true", help="print only the final answer")
    pr.add_argument("--json", action="store_true", help="emit one JSON event per line instead of rendering")
    pr.add_argument("prompt", nargs="*")
    pm = sub.add_parser("models", help="list models on NIM")
    pm.add_argument("--refresh", action="store_true")
    sub.add_parser("status", help="API key pool + paths")
    sub.add_parser("tools", help="list tools")
    pl = sub.add_parser("list", help="recent conversations")
    pl.add_argument("-n", type=int, default=20)
    ps = sub.add_parser("show", help="print a conversation")
    ps.add_argument("id")
    ps.add_argument("--json", action="store_true")
    pd = sub.add_parser("rm", help="delete a conversation")
    pd.add_argument("id")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = build_parser()
    if not argv:
        argv = ["chat"]
    elif argv[0] not in SUBCOMMANDS and argv[0] not in ("-h", "--help"):
        if not argv[0].startswith("-"):
            argv = ["run", *argv]  # arety "do something"
        else:
            _, rest = p.parse_known_args(["chat", *argv])
            argv = ["run", *argv] if rest or not sys.stdin.isatty() else ["chat", *argv]  # flags + prompt, or piped stdin → run
    a = p.parse_args(argv)
    if a.cmd in ("chat", "run") and not srv.KEYS:
        ui.err(ui.c("error", f"no NVIDIA_API_KEYS configured (looked in {srv.ENV_FILE})"))
        return 2
    if a.cmd == "chat":
        return cmd_chat(a)
    if a.cmd == "run":
        return cmd_run(a)
    if a.cmd == "models":
        ui.print_models(asyncio.run(srv.models(refresh=a.refresh)), srv.OMNI_MODEL_ID)
    elif a.cmd == "status":
        ui.print_status(*status_data())
    elif a.cmd == "tools":
        ui.print_tools(list_tools(), GUARDED_TOOLS)
    elif a.cmd == "list":
        ui.print_conversations(srv.conv_list()[:a.n])
    elif a.cmd == "show":
        return cmd_show(a)
    elif a.cmd == "rm":
        srv.conv_delete(a.id)
        ui.out(f"deleted {a.id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
