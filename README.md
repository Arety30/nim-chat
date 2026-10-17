# nim-chat (Arety Chat)

nim-chat is a self-hosted chat application and agent harness that runs on NVIDIA NIM's hosted chat-completions inference endpoints. A single FastAPI file serves a browser UI with streaming replies, visible reasoning, tool cards and file attachments; the same file is imported as a library by a terminal client (`arety`) and by a small Python SDK (`harness/`). Its distinguishing idea is "Arety Omni": several free-tier API keys and three separate hosted models (a large text model as the brain, a vision-language model as the eyes, and a speech model as the ears) are composed behind one logical model id, with round-robin key pooling, cooldowns, and automatic fallback to a faster model when the primary one stalls. The problem it solves is getting a reliable, multimodal, tool-using assistant out of rate-limited hosted endpoints without a GPU of one's own.

## Where it is used

Abrar runs nim-chat as a personal assistant that lives on a small always-on virtual machine behind a reverse proxy and is also installed on his laptop. The web UI is used from the phone (camera button, hands-free Converse voice mode), the `arety` CLI is used from the terminal for one-shot tasks and scripting, and the Python harness is reused by other projects that need an agent loop with tools. Web, CLI and library share one conversation database, so a chat started in the terminal can be continued in the browser.

## How it works

Everything lives in `server.py`; `harness/agent.py` and `cli.py` import it as a module (importing builds the FastAPI app but never starts uvicorn).

Request flow for one turn (`POST /api/chat`, Server-Sent Events):
1. The conversation is loaded from SQLite (`~/.local/share/nim-chat/chat.db`, one JSON blob per conversation) or created. Per-chat settings (model, system prompt, temperature, max tokens, enabled tools, `mode`) are merged with the request.
2. Senses run first. Each attachment is classified by MIME type: images go to the vision model (`describe_image`, downscaled to 768 px) unless the chosen chat model has native vision; audio and video go to speech recognition (`transcribe_bytes`, NIM gRPC first, local faster-whisper as fallback). The results are appended to the user message as bracketed notes, so any text model can "see" and "hear".
3. The history is converted to API messages and trimmed from the oldest end to `HISTORY_CHAR_BUDGET` without splitting an assistant/tool group.
4. The agent loop runs up to `MAX_STEPS` (default 25). Each step streams a completion; `ThinkSplitter` routes `<think>...</think>` spans and `reasoning_content` deltas to a separate reasoning channel. Tool calls are executed (`run_tool`), their outputs appended as `tool` messages, and the loop continues until the model answers without tools, the user hits stop (`/api/stop/{id}`), or the step limit is reached. If the model rejects tool definitions, the turn is retried without tools.
5. Every event (`start`, `sense`, `step`, `delta`, `reasoning`, `tool_call`, `tool_result`, `notice`, `error`, `stopped`, `done`) is written as SSE with a 15-second heartbeat; partial turns are persisted.

Key pool and Omni fallback (`KeyPool`, `stream_completion`):
- `NVIDIA_API_KEYS` is a comma-separated list. Keys are tried round-robin; a key that returns 401/403 is cooled for 15 minutes, 429 for 45 seconds, 5xx for 15-20 seconds. `NVIDIA_KEY_ASR/VLM/CHAT` optionally pin a "home" key index per service so free-tier quotas are spent where they were granted.
- When the model is `arety/omni`, the brain is `OMNI_CHAT_MODEL`. If no first token arrives within `OMNI_FIRST_TOKEN_TIMEOUT` (25 s; 8 s in Converse) the stream is abandoned, a `notice` is sent, the turn restarts on `OMNI_FALLBACK_MODEL`, and the slow brain is marked unresponsive for `SLOW_BREAKER_SECONDS` (default 600) so following turns skip straight to the fallback.

Tools (`TOOLS` in `server.py`): `run_shell`, `python_exec`, `read_file`, `write_file`, `edit_file` (exact string replacement), `list_dir`, `web_search` (parses DuckDuckGo HTML results), `web_fetch` (HTML to text), `remember` (appends dated bullets to `MEMORY.md`, which is injected into the system prompt), `get_datetime`, `look_at_image`, `transcribe_audio`. Relative paths resolve inside the workspace (`~/nim-chat-workspace`), and files written there are served at `/api/files/<path>` so the model can hand back download links. Tool output is clipped to 14 000 characters (head and tail kept).

Converse mode: the browser captures speech (Web Speech API, or an energy-based VAD recorder posting to `/api/stt`), the server answers with a spoken-style system prompt, thinking disabled, and a 400-token cap, and the reply is read aloud sentence by sentence via `/api/tts` (ElevenLabs when a key is set, otherwise the browser's own voices).

| Path | Role |
|---|---|
| `server.py` | Config loading, key pool, SQLite store, senses, tools, streaming, prompts, auth (password + signed cookie), all HTTP routes, static UI |
| `static/index.html` | Single-page UI: conversation list, composer with attachments and camera, tool cards, reasoning panel, per-chat settings, Converse orb |
| `harness/agent.py` | `Agent` class: async `run()` event stream, `ask()`/`ask_sync()`, approval hook for guarded tools (`run_shell`, `python_exec`, `write_file`, `edit_file`, `delegate_task`), `register_tool(ToolDef)`, `delegate_task` sub-agent, attachments, `stop()` |
| `cli.py` | `arety` commands and argument parsing: `chat` (default), `run`, `models`, `status`, `tools`, `list`, `show`, `rm`; slash commands in interactive mode |
| `cli_ui.py` | All terminal presentation: theme (overridable JSON), banner, prompt, event renderer, approval question, list views |
| `arety` | Bash launcher that picks `.venv`, `~/venv` or `python3` |
| `deploy.sh` | rsync to a remote host, create venv, register a supervisord program, adjust the reverse proxy timeout, health-check |

## Setup

Requirements: Python 3.11+, an NVIDIA NIM account with one or more API keys, and optionally `ffmpeg` for local speech recognition.

```bash
git clone <repo-url> ~/nim-chat
cd ~/nim-chat
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-voice.txt    # optional: faster-whisper fallback for /api/stt
mkdir -p ~/.config/nim-chat && chmod 700 ~/.config/nim-chat
```

Create `~/.config/nim-chat/env` (path overridable with `NIM_CHAT_ENV`); values already in the environment take precedence:

```ini
NVIDIA_API_KEYS=nvapi-KEY_ONE,nvapi-KEY_TWO
NVIDIA_KEY_CHAT=0            # optional home key index per service
NVIDIA_KEY_VLM=1
NVIDIA_KEY_ASR=2
CHAT_PASSWORD=choose-a-password
SESSION_SECRET=long-random-string
HOST=127.0.0.1
PORT=8000
DEFAULT_MODEL=arety/omni
OMNI_CHAT_MODEL=deepseek-ai/deepseek-v4-pro-0813
OMNI_FALLBACK_MODEL=nvidia/nemotron-3-super-120b-a12b
CONVERSE_MODEL=nvidia/nemotron-3.5-lightning-30b-a3b
VLM_MODEL=google/paligemma
ASR_MODEL=openai/whisper-large-v3
ELEVENLABS_API_KEY=            # optional, enables server-side TTS
WHISPER_MODEL=base             # local fallback STT size
MAX_STEPS=25
TOOL_TIMEOUT=90
HISTORY_CHAR_BUDGET=90000
NIM_CHAT_DATA=~/.local/share/nim-chat
NIM_CHAT_WORKSPACE=~/nim-chat-workspace
```

If `CHAT_PASSWORD` is empty the UI has no login. The CLI theme can be customised without code in `~/.config/nim-chat/cli-theme.json` (or `$ARETY_CLI_THEME`).

## Usage

Web server:

```bash
.venv/bin/python server.py            # http://127.0.0.1:8000
curl http://127.0.0.1:8000/healthz
```

Open the page, enter the password, pick a model (the list comes from `/api/models` with embedding/guard/reward models filtered out), attach files or a photo, and chat. Use the stop button to interrupt a tool-heavy turn and "Converse" for the voice call.

Terminal client (`ln -s ~/nim-chat/arety ~/bin/arety` is convenient):

```bash
arety                                     # interactive chat; /help lists slash commands
arety "summarise ~/notes.md"              # one-shot; guarded tools ask y/n/always
arety run -q "What is 17*23?"             # final answer only
cat report.txt | arety run -q             # prompt from stdin
arety run --json "list files in the workspace"   # one JSON event per line
arety -m nvidia/nemotron-3.5-lightning-30b-a3b --no-think "quick question"
arety -t web_search,web_fetch "latest Python release?"
arety -a photo.jpg "what is in this picture?"
arety -C "and then?"                      # continue the last CLI conversation
arety models | status | tools | list | show ID | rm ID
```

Interactive slash commands include `/new`, `/load ID`, `/model`, `/tools`, `/system`, `/attach PATH`, `/yes` `/ask` `/deny`, `/think`, `/reasoning`, `/verbose`, `/memory`, `/status`, `/quit`.

From Python:

```python
from harness import Agent, ToolDef

agent = Agent("arety/omni", approve=lambda name, args: True)
agent.register_tool(ToolDef("add", lambda a, b: a + b, "Add two numbers",
                            {"a": {"type": "number"}, "b": {"type": "number"}}, ["a", "b"]))
print(agent.ask_sync("What is 2+3? Use the add tool."))

async for ev in agent.run("describe this", attachments=["shot.png"]):
    print(ev["type"], ev.get("text", ""))
```

Deployment: `VM=HOST ./deploy.sh` rsyncs the project to a remote user account, installs the venv, writes a supervisord program entry and restarts it. It assumes the remote layout used on Abrar's machine (a `~/cloud/supervisord.conf` and an Apache site file) and will need editing elsewhere.

## Project structure

```
nim-chat/
  server.py              # the whole backend (~1350 lines)
  static/index.html      # single-page web UI
  harness/
    __init__.py
    agent.py             # Agent, ToolDef, GUARDED_TOOLS, list_tools
  cli.py                 # arety commands
  cli_ui.py              # arety presentation layer
  arety                  # launcher script
  deploy.sh              # remote deploy helper
  requirements.txt       # fastapi, uvicorn, httpx, python-multipart, Pillow, nvidia-riva-client
  requirements-voice.txt # faster-whisper (optional)
```

Runtime data lives outside the repo: `~/.config/nim-chat/env` (secrets), `~/.local/share/nim-chat/chat.db` and `cli_history`, `~/nim-chat-workspace/` (uploads, tool working directory, `MEMORY.md`).

## Limitations and notes

- `run_shell` and `python_exec` execute on the server with the account's full permissions; the only guard is the UI password (web) or the approval prompt (CLI). Do not expose the server publicly without the password and a reverse proxy.
- Rate limits and model availability depend entirely on the hosted provider; the key pool softens 429s but cannot remove them. Model ids in the defaults will age.
- `web_search` scrapes DuckDuckGo's HTML page with a regex and breaks when that markup changes.
- Speech recognition over gRPC requires `nvidia-riva-client`; the local faster-whisper fallback needs `ffmpeg` and downloads a model on first use.
- Conversations are stored as a JSON blob per row, fine for one user but not designed for concurrent multi-user editing.
- Single-user authentication: one shared password and a signed cookie, no accounts.
- `deploy.sh` is tailored to one hosting layout and should be treated as an example.
