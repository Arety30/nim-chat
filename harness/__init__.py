"""Arety agent harness: the nim-chat agent loop as a reusable Python library.

    from harness import Agent

    agent = Agent(model="arety/omni")
    print(agent.ask_sync("What is in the workspace?"))

    async for ev in agent.run("Write a haiku to haiku.txt"):
        ...   # start / step / delta / reasoning / tool_call / tool_result / notice / error / done

The harness reuses everything from server.py (key pool with failover, tools, streaming,
prompts, SQLite conversations) without modifying it, so CLI and web chats share one store.
"""
from .agent import Agent, AgentError, GUARDED_TOOLS, ToolDef, list_tools

__all__ = ["Agent", "AgentError", "GUARDED_TOOLS", "ToolDef", "list_tools"]
