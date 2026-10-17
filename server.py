#!/usr/bin/env python3
"""nim-chat: chat UI + agentic tool harness on NVIDIA NIM (build.nvidia.com).

Two (or more) nvapi- keys are pooled behind one logical model: requests are
spread round-robin and fail over automatically on 401/403/429/5xx.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import inspect
import html as htmllib
import json
import mimetypes
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# ----------------------------------------------------------------------------- config

ENV_FILE = Path(os.environ.get("NIM_CHAT_ENV", Path.home() / ".config/nim-chat/env"))


def _load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_env(ENV_FILE)

KEYS = [k.strip() for k in os.environ.get("NVIDIA_API_KEYS", "").split(",") if k.strip()]
BASE_URL = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "nvidia/nemotron-3-super-120b-a12b")
CHAT_PASSWORD = os.environ.get("CHAT_PASSWORD", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)
HOST = os.environ.get("HOST", "127.0.0.1")
PORT = int(os.environ.get("PORT", "8000"))
DATA_DIR = Path(os.environ.get("NIM_CHAT_DATA", Path.home() / ".local/share/nim-chat"))
WORKSPACE = Path(os.environ.get("NIM_CHAT_WORKSPACE", Path.home() / "nim-chat-workspace"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "25"))
TOOL_TIMEOUT = int(os.environ.get("TOOL_TIMEOUT", "90"))
HISTORY_CHAR_BUDGET = int(os.environ.get("HISTORY_CHAR_BUDGET", "90000"))
CONVERSE_MODEL = os.environ.get("CONVERSE_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.environ.get("ELEVENLABS_VOICE_ID", "l5LPd4RiIgGFgzEi8Smo")
ELEVENLABS_TTS_MODEL = os.environ.get("ELEVENLABS_TTS_MODEL", "eleven_flash_v2_5")
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base")

# "Arety Omni": one logical model = DeepSeek V4 Pro (brain) + PaliGemma (eyes) + Whisper-large-v3 (ears)
OMNI_MODEL_ID = "arety/omni"
OMNI_CHAT_MODEL = os.environ.get("OMNI_CHAT_MODEL", "deepseek-ai/deepseek-v4-pro-0813")
OMNI_FALLBACK_MODEL = os.environ.get("OMNI_FALLBACK_MODEL", "nvidia/nemotron-3-super-120b-a12b")
OMNI_CONVERSE_FALLBACK_MODEL = os.environ.get("OMNI_CONVERSE_FALLBACK_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
OMNI_FIRST_TOKEN_TIMEOUT = float(os.environ.get("OMNI_FIRST_TOKEN_TIMEOUT", "25"))
CONVERSE_FIRST_TOKEN_TIMEOUT = float(os.environ.get("CONVERSE_FIRST_TOKEN_TIMEOUT", "8"))
VLM_MODEL = os.environ.get("VLM_MODEL", "google/paligemma")
VLM_URL = os.environ.get("VLM_URL", f"https://ai.api.nvidia.com/v1/vlm/{VLM_MODEL}")
ASR_MODEL = os.environ.get("ASR_MODEL", "openai/whisper-large-v3")
ASR_GRPC = os.environ.get("ASR_GRPC", "grpc.nvcf.nvidia.com:443")
ASR_FUNCTION_ID = os.environ.get("ASR_FUNCTION_ID", "00000000-0000-0000-0000-000000000000")
ASR_LANGUAGE = os.environ.get("ASR_LANGUAGE", "en-US")
KEY_ROLES = {r: int(os.environ.get(f"NVIDIA_KEY_{r}", "-1")) for r in ("ASR", "VLM", "CHAT")}
VISION_MODELS = ("vision", "gemma-3", "gemma-4", "phi-3-vision", "cosmos", "omni-30b", "neva", "vila", "kosmos", "fuyu")


def vision_capable(model: str) -> bool:
    return any(v in model for v in VISION_MODELS)

DATA_DIR.mkdir(parents=True, exist_ok=True)
WORKSPACE.mkdir(parents=True, exist_ok=True)
(WORKSPACE / "uploads").mkdir(exist_ok=True)
MEMORY_FILE = WORKSPACE / "MEMORY.md"
STATIC = Path(__file__).parent / "static"
STARTED = time.time()

if not KEYS:
    print(f"!! no NVIDIA_API_KEYS configured (looked in {ENV_FILE})", file=sys.stderr)

# ----------------------------------------------------------------------------- key pool


class KeyPool:
    """Round-robin over several API keys with per-key cooldown → behaves like one model."""

    COOLDOWN = {401: 900, 403: 900, 429: 45, 500: 15, 502: 15, 503: 20, 504: 15}

    def __init__(self, keys: list[str]):
        self.keys = [
            {"key": k, "cool_until": 0.0, "ok": 0, "err": 0, "last_status": None, "last_err": "", "last_used": 0.0}
            for k in keys
        ]
        self._i = 0

    def order(self, role: str | None = None) -> list[dict]:
        """Keys to try, best first: the role's home key (if any), then round-robin; cooling keys go last."""
        if not self.keys:
            return []
        n = len(self.keys)
        home = KEY_ROLES.get(role or "", -1)
        if 0 <= home < n:
            rotated = [self.keys[home]] + [self.keys[j] for j in range(n) if j != home]
        else:
            rotated = [self.keys[(self._i + j) % n] for j in range(n)]
            self._i = (self._i + 1) % n
        now = time.time()
        return sorted(rotated, key=lambda k: 0 if k["cool_until"] <= now else 1)

    def report(self, entry: dict, status: int | None, err: str = "") -> None:
        entry["last_used"] = time.time()
        entry["last_status"] = status
        if status == 200:
            entry["ok"] += 1
            entry["last_err"] = ""
            return
        entry["err"] += 1
        entry["last_err"] = (err or "")[:300]
        cd = self.COOLDOWN.get(status or 0, 10)
        entry["cool_until"] = time.time() + cd

    def status(self) -> list[dict]:
        now = time.time()
        return [
            {
                "key": e["key"][:10] + "…" + e["key"][-4:],
                "cooling_for": max(0, int(e["cool_until"] - now)),
                "ok": e["ok"],
                "err": e["err"],
                "last_status": e["last_status"],
                "last_err": e["last_err"],
            }
            for e in self.keys
        ]


POOL = KeyPool(KEYS)

# ----------------------------------------------------------------------------- storage


def _db() -> sqlite3.Connection:
    con = sqlite3.connect(DATA_DIR / "chat.db")
    con.row_factory = sqlite3.Row
    return con


with _db() as _c:
    _c.execute(
        """CREATE TABLE IF NOT EXISTS conversations(
        id TEXT PRIMARY KEY, title TEXT, model TEXT, system_prompt TEXT,
        settings TEXT, messages TEXT, created REAL, updated REAL)"""
    )


def conv_list() -> list[dict]:
    with _db() as c:
        rows = c.execute("SELECT id,title,model,created,updated FROM conversations ORDER BY updated DESC").fetchall()
    return [dict(r) for r in rows]


def conv_get(cid: str) -> dict | None:
    with _db() as c:
        r = c.execute("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["messages"] = json.loads(d["messages"] or "[]")
    d["settings"] = json.loads(d["settings"] or "{}")
    return d


def conv_save(d: dict) -> None:
    d["updated"] = time.time()
    with _db() as c:
        c.execute(
            """INSERT INTO conversations(id,title,model,system_prompt,settings,messages,created,updated)
               VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET title=excluded.title, model=excluded.model,
               system_prompt=excluded.system_prompt, settings=excluded.settings,
               messages=excluded.messages, updated=excluded.updated""",
            (
                d["id"], d.get("title") or "New chat", d.get("model") or DEFAULT_MODEL,
                d.get("system_prompt") or "", json.dumps(d.get("settings") or {}),
                json.dumps(d.get("messages") or []), d.get("created") or time.time(), d["updated"],
            ),
        )


def conv_delete(cid: str) -> None:
    with _db() as c:
        c.execute("DELETE FROM conversations WHERE id=?", (cid,))


# ----------------------------------------------------------------------------- eyes (PaliGemma) & ears (Whisper NIM)


def _shrink_image(data: bytes, max_px: int = 768, quality: int = 82) -> tuple[bytes, str]:
    """Return (jpeg bytes, mime) resized so the VLM payload stays well under the 180 KB inline limit."""
    try:
        from PIL import Image
        import io
        im = Image.open(io.BytesIO(data))
        im = im.convert("RGB")
        im.thumbnail((max_px, max_px))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        out = buf.getvalue()
        if len(out) > 170_000:
            buf = io.BytesIO(); im.thumbnail((512, 512)); im.save(buf, "JPEG", quality=70); out = buf.getvalue()
        return out, "image/jpeg"
    except Exception:  # noqa: BLE001
        return data, "image/jpeg"


async def vlm_ask(image_bytes: bytes, question: str, max_tokens: int = 256) -> str:
    """Ask PaliGemma about an image (NIM VLM endpoint, inline <img> payload)."""
    jpg, mime = _shrink_image(image_bytes)
    b64 = base64.b64encode(jpg).decode()
    body = {"messages": [{"role": "user", "content": f'{question} <img src="data:{mime};base64,{b64}" />'}],
            "max_tokens": max_tokens, "temperature": 0.2, "stream": False}
    last = "no keys"
    for entry in POOL.order("VLM"):
        if entry["cool_until"] > time.time():
            continue
        try:
            async with httpx.AsyncClient(timeout=60) as cl:
                r = await cl.post(VLM_URL, json=body, headers={"Authorization": f"Bearer {entry['key']}", "Accept": "application/json"})
        except httpx.HTTPError as e:
            POOL.report(entry, None, str(e)); last = str(e); continue
        if r.status_code != 200:
            POOL.report(entry, r.status_code, r.text); last = f"HTTP {r.status_code}: {r.text[:200]}"; continue
        POOL.report(entry, 200)
        return (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip() or "[no answer]"
    return f"[vision error: {last}]"


async def describe_image(image_bytes: bytes, user_question: str = "") -> dict:
    desc = await vlm_ask(image_bytes, "Describe this image in detail, including any visible text, objects, people, colors and setting.")
    out = {"description": desc}
    if user_question.strip():
        out["answer"] = await vlm_ask(image_bytes, user_question.strip()[:500])
    return out


def _asr_available() -> bool:
    try:
        import riva.client  # noqa: F401
        return bool(ASR_FUNCTION_ID)
    except Exception:  # noqa: BLE001
        return False


def _asr_recognize_sync(wav: bytes, language: str, key: str) -> str:
    import riva.client
    auth = riva.client.Auth(None, True, ASR_GRPC, [("function-id", ASR_FUNCTION_ID), ("authorization", "Bearer " + key)])
    asr = riva.client.ASRService(auth)
    cfg = riva.client.RecognitionConfig(language_code=language or ASR_LANGUAGE, max_alternatives=1,
                                        enable_automatic_punctuation=True, audio_channel_count=1)
    resp = asr.offline_recognize(wav, cfg)
    return " ".join(alt.transcript for res in resp.results for alt in res.alternatives[:1]).strip()


async def _to_wav16k(data: bytes, suffix: str) -> bytes:
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / ("in" + (suffix or ".webm")); wav = Path(td) / "out.wav"
        src.write_bytes(data)
        p = await asyncio.to_thread(subprocess.run, ["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-ac", "1", "-ar", "16000", str(wav)],
                                    capture_output=True, text=True)
        if p.returncode or not wav.exists():
            raise HTTPException(400, f"ffmpeg failed: {p.stderr[:200]}")
        return wav.read_bytes()


async def transcribe_bytes(data: bytes, suffix: str = ".webm", language: str = "", engine: str = "auto") -> dict:
    """Whisper-large-v3 on NIM (gRPC) first, local faster-whisper as fallback."""
    wav = await _to_wav16k(data, suffix)
    errors = []
    if engine in ("auto", "nim") and _asr_available():
        lang = language or ASR_LANGUAGE
        if len(lang) == 2:
            lang = {"en": "en-US", "ar": "ar-AR", "ur": "ur-PK", "hi": "hi-IN", "fr": "fr-FR", "de": "de-DE", "es": "es-ES"}.get(lang, lang)
        for entry in POOL.order("ASR"):
            if entry["cool_until"] > time.time():
                continue
            try:
                text = await asyncio.wait_for(asyncio.to_thread(_asr_recognize_sync, wav, lang, entry["key"]), 60)
                POOL.report(entry, 200)
                return {"text": text, "engine": ASR_MODEL, "language": lang}
            except Exception as e:  # noqa: BLE001
                POOL.report(entry, None, str(e)); errors.append(f"nim: {type(e).__name__}: {str(e)[:120]}")
    if engine in ("auto", "local") and _whisper_available():
        try:
            model = await _get_whisper()

            def run():
                import io, tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(wav); path = f.name
                try:
                    segs, info = model.transcribe(path, language=(language[:2] if language else None), beam_size=1, vad_filter=True)
                    return " ".join(sg.text.strip() for sg in segs).strip(), info.language
                finally:
                    os.unlink(path)
            text, lang = await asyncio.to_thread(run)
            return {"text": text, "engine": f"faster-whisper-{WHISPER_MODEL}", "language": lang}
        except Exception as e:  # noqa: BLE001
            errors.append(f"local: {e}")
    raise HTTPException(502, "transcription failed: " + ("; ".join(errors) or "no STT engine available"))


# ----------------------------------------------------------------------------- tools


def _resolve(p: str) -> Path:
    path = Path(os.path.expanduser(p or "."))
    return path if path.is_absolute() else WORKSPACE / path


def _clip(s: str, n: int = 14000) -> str:
    if len(s) <= n:
        return s
    return s[: n // 2] + f"\n… [{len(s) - n} chars truncated] …\n" + s[-n // 2 :]


def tool_run_shell(command: str, timeout: int | None = None) -> str:
    timeout = min(int(timeout or TOOL_TIMEOUT), 600)
    try:
        p = subprocess.run(
            command, shell=True, cwd=WORKSPACE, capture_output=True, text=True, timeout=timeout,
            executable="/bin/bash",
        )
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]"
    out = p.stdout
    if p.stderr:
        out += ("\n[stderr]\n" if out else "[stderr]\n") + p.stderr
    out = out.strip() or "[no output]"
    return _clip(out) + (f"\n[exit code {p.returncode}]" if p.returncode else "")


def tool_python_exec(code: str, timeout: int | None = None) -> str:
    timeout = min(int(timeout or TOOL_TIMEOUT), 600)
    try:
        p = subprocess.run(
            [sys.executable, "-c", code], cwd=WORKSPACE, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return f"[timed out after {timeout}s]"
    out = (p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr else "")).strip() or "[no output]"
    return _clip(out) + (f"\n[exit code {p.returncode}]" if p.returncode else "")


def tool_read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    fp = _resolve(path)
    if not fp.exists():
        return f"[not found: {fp}]"
    if fp.is_dir():
        return tool_list_dir(path)
    try:
        text = fp.read_text(errors="replace")
    except Exception as e:  # noqa: BLE001
        return f"[error: {e}]"
    lines = text.splitlines()
    s = max(1, int(start_line or 1))
    e = min(len(lines), int(end_line or len(lines)))
    body = "\n".join(f"{i:5d}| {lines[i - 1]}" for i in range(s, e + 1))
    return _clip(f"{fp} ({len(lines)} lines)\n{body}")


def file_url(fp: Path) -> str | None:
    """Download URL for a file inside the workspace (served by /api/files), else None."""
    try:
        rel = fp.resolve().relative_to(WORKSPACE.resolve())
    except ValueError:
        return None
    return "/api/files/" + "/".join(rel.parts)


def tool_write_file(path: str, content: str) -> str:
    fp = _resolve(path)
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(content)
    url = file_url(fp)
    return f"wrote {len(content)} chars to {fp}" + (f" (user download link: {url})" if url else "")


def tool_edit_file(path: str, old_text: str, new_text: str) -> str:
    fp = _resolve(path)
    if not fp.exists():
        return f"[not found: {fp}]"
    text = fp.read_text()
    n = text.count(old_text)
    if n == 0:
        return "[old_text not found in file]"
    if n > 1:
        return f"[old_text occurs {n} times; make it unique]"
    fp.write_text(text.replace(old_text, new_text, 1))
    return f"edited {fp}"


def tool_list_dir(path: str = ".") -> str:
    fp = _resolve(path)
    if not fp.exists():
        return f"[not found: {fp}]"
    rows = []
    for child in sorted(fp.iterdir(), key=lambda c: (not c.is_dir(), c.name.lower()))[:400]:
        try:
            size = child.stat().st_size
        except OSError:
            size = 0
        rows.append(f"{'d' if child.is_dir() else 'f'} {size:>10} {child.name}")
    return f"{fp}\n" + ("\n".join(rows) or "[empty]")


_TAG_RE = re.compile(r"<(script|style|noscript|svg)[^>]*>.*?</\1>", re.S | re.I)
_HTML_RE = re.compile(r"<[^>]+>")


def _html_to_text(raw: str) -> str:
    raw = _TAG_RE.sub(" ", raw)
    raw = re.sub(r"<br\s*/?>|</p>|</div>|</li>|</h\d>|</tr>", "\n", raw, flags=re.I)
    text = htmllib.unescape(_HTML_RE.sub(" ", raw))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


async def tool_web_fetch(url: str, max_chars: int = 12000) -> str:
    if not re.match(r"^https?://", url):
        url = "https://" + url
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30, headers={"User-Agent": "Mozilla/5.0 nim-chat"}) as cl:
            r = await cl.get(url)
    except Exception as e:  # noqa: BLE001
        return f"[fetch error: {e}]"
    ctype = r.headers.get("content-type", "")
    body = r.text if "json" in ctype or "text" in ctype or "xml" in ctype else f"[binary {ctype}, {len(r.content)} bytes]"
    if "html" in ctype:
        body = _html_to_text(body)
    return f"HTTP {r.status_code} {url}\n" + _clip(body, int(max_chars))


async def tool_web_search(query: str, max_results: int = 8) -> str:
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30, headers={"User-Agent": "Mozilla/5.0 nim-chat"}) as cl:
            r = await cl.post("https://html.duckduckgo.com/html/", data={"q": query})
    except Exception as e:  # noqa: BLE001
        return f"[search error: {e}]"
    results = []
    for m in re.finditer(
        r'<a rel="nofollow" class="result__a" href="([^"]+)"[^>]*>(.*?)</a>.*?class="result__snippet"[^>]*>(.*?)</a>',
        r.text, re.S,
    ):
        href, title, snip = m.groups()
        um = re.search(r"uddg=([^&]+)", href)
        if um:
            from urllib.parse import unquote
            href = unquote(um.group(1))
        results.append(f"- {_html_to_text(title)}\n  {href}\n  {_html_to_text(snip)}")
        if len(results) >= int(max_results):
            break
    return "\n".join(results) or "[no results]"


def tool_remember(note: str) -> str:
    with MEMORY_FILE.open("a") as f:
        f.write(f"- ({datetime.now().strftime('%Y-%m-%d')}) {note.strip()}\n")
    return "saved to memory"


def tool_get_datetime() -> str:
    now = datetime.now()
    return f"{now.strftime('%A %Y-%m-%d %H:%M:%S')} local, {datetime.now(timezone.utc).strftime('%H:%M')} UTC"


TOOLS: dict[str, dict] = {
    "run_shell": {
        "fn": tool_run_shell,
        "desc": "Run a bash command on the server (cwd = workspace). Use for anything: git, pip, curl, system inspection, running scripts.",
        "params": {"command": {"type": "string"}, "timeout": {"type": "integer", "description": "seconds, default 90"}},
        "required": ["command"],
    },
    "python_exec": {
        "fn": tool_python_exec,
        "desc": "Execute a Python snippet and return stdout/stderr. Good for calculations, data processing, quick scripts.",
        "params": {"code": {"type": "string"}, "timeout": {"type": "integer"}},
        "required": ["code"],
    },
    "read_file": {
        "fn": tool_read_file,
        "desc": "Read a text file with line numbers (optionally a line range).",
        "params": {"path": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}},
        "required": ["path"],
    },
    "write_file": {
        "fn": tool_write_file,
        "desc": "Create or overwrite a file with the given content (parent dirs are created).",
        "params": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    },
    "edit_file": {
        "fn": tool_edit_file,
        "desc": "Replace one unique occurrence of old_text with new_text inside a file.",
        "params": {"path": {"type": "string"}, "old_text": {"type": "string"}, "new_text": {"type": "string"}},
        "required": ["path", "old_text", "new_text"],
    },
    "list_dir": {
        "fn": tool_list_dir,
        "desc": "List a directory.",
        "params": {"path": {"type": "string"}},
        "required": [],
    },
    "web_search": {
        "fn": tool_web_search,
        "desc": "Search the web (DuckDuckGo). Returns titles, URLs and snippets.",
        "params": {"query": {"type": "string"}, "max_results": {"type": "integer"}},
        "required": ["query"],
    },
    "web_fetch": {
        "fn": tool_web_fetch,
        "desc": "Fetch a URL and return its readable text (HTML is stripped).",
        "params": {"url": {"type": "string"}, "max_chars": {"type": "integer"}},
        "required": ["url"],
    },
    "remember": {
        "fn": tool_remember,
        "desc": "Save a durable note about the user or project to long-term memory (shown in every future chat).",
        "params": {"note": {"type": "string"}},
        "required": ["note"],
    },
    "look_at_image": {
        "fn": None,  # async, bound below
        "desc": "Look at an image file (uploaded photo, camera shot, screenshot) with the PaliGemma vision model and answer a question about it.",
        "params": {"path": {"type": "string", "description": "path of the image (uploads live in the workspace uploads/ folder)"}, "question": {"type": "string"}},
        "required": ["path"],
    },
    "transcribe_audio": {
        "fn": None,
        "desc": "Transcribe an audio or video file to text with Whisper-large-v3.",
        "params": {"path": {"type": "string"}, "language": {"type": "string", "description": "e.g. en-US; default auto/en"}},
        "required": ["path"],
    },
    "get_datetime": {
        "fn": tool_get_datetime,
        "desc": "Current date and time.",
        "params": {},
        "required": [],
    },
}


async def tool_look_at_image(path: str, question: str = "") -> str:
    fp = _resolve(path)
    if not fp.exists():
        return f"[not found: {fp}]"
    q = question.strip() or "Describe this image in detail, including any visible text."
    return await vlm_ask(fp.read_bytes(), q)


async def tool_transcribe_audio(path: str, language: str = "") -> str:
    fp = _resolve(path)
    if not fp.exists():
        return f"[not found: {fp}]"
    try:
        r = await transcribe_bytes(fp.read_bytes(), fp.suffix, language)
    except HTTPException as e:
        return f"[{e.detail}]"
    return r["text"] or "[no speech detected]"


TOOLS["look_at_image"]["fn"] = tool_look_at_image
TOOLS["transcribe_audio"]["fn"] = tool_transcribe_audio


def tool_specs(enabled: list[str] | None = None) -> list[dict]:
    out = []
    for name, t in TOOLS.items():
        if enabled is not None and name not in enabled:
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": t["desc"],
                    "parameters": {"type": "object", "properties": t["params"], "required": t["required"]},
                },
            }
        )
    return out


async def run_tool(name: str, args: dict) -> str:
    t = TOOLS.get(name)
    if not t:
        return f"[unknown tool {name}]"
    allowed = set(t["params"]) | set(t["required"])
    args = {k: v for k, v in (args or {}).items() if k in allowed}
    try:
        fn = t["fn"]
        if inspect.iscoroutinefunction(fn):
            return await fn(**args)
        return await asyncio.to_thread(fn, **args)
    except TypeError as e:
        return f"[bad arguments: {e}]"
    except Exception as e:  # noqa: BLE001
        return f"[tool error: {type(e).__name__}: {e}]"


# ----------------------------------------------------------------------------- model streaming


class ThinkSplitter:
    """Route <think>…</think> spans (deepseek/glm style) to the reasoning channel while streaming."""

    OPEN, CLOSE = "<think>", "</think>"

    def __init__(self):
        self.buf = ""
        self.inside = False

    def feed(self, text: str) -> list[tuple[str, str]]:
        self.buf += text
        out: list[tuple[str, str]] = []
        while self.buf:
            tag = self.CLOSE if self.inside else self.OPEN
            idx = self.buf.find(tag)
            if idx == -1:
                # keep a possible partial tag at the end
                keep = 0
                for n in range(min(len(tag) - 1, len(self.buf)), 0, -1):
                    if tag.startswith(self.buf[-n:]):
                        keep = n
                        break
                emit, self.buf = self.buf[: len(self.buf) - keep], self.buf[len(self.buf) - keep :]
                if emit:
                    out.append(("reasoning" if self.inside else "content", emit))
                break
            emit = self.buf[:idx]
            if emit:
                out.append(("reasoning" if self.inside else "content", emit))
            self.buf = self.buf[idx + len(tag) :]
            self.inside = not self.inside
        return out

    def flush(self) -> list[tuple[str, str]]:
        out = [("reasoning" if self.inside else "content", self.buf)] if self.buf else []
        self.buf = ""
        return out


async def stream_completion(payload: dict, first_token_timeout: float | None = None, role: str = "CHAT") -> AsyncIterator[dict]:
    """Yield events: content/reasoning/tool_calls/usage/error/slow, failing over across keys.

    `slow` is yielded (and the stream abandoned) when the model sends nothing within first_token_timeout.
    """
    attempts = max(1, len(POOL.keys)) * 2
    last_err = "no API keys configured"
    tried = 0
    for _round in range(2):
        for entry in POOL.order(role):
            tried += 1
            if tried > attempts:
                break
            wait = entry["cool_until"] - time.time()
            if wait > 0:
                if wait > 8:
                    continue
                await asyncio.sleep(wait)
            headers = {"Authorization": f"Bearer {entry['key']}", "Accept": "text/event-stream"}
            started_output = False
            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, read=180.0)) as client:
                    req = client.build_request("POST", f"{BASE_URL}/chat/completions", json=payload, headers=headers)
                    try:
                        r = await asyncio.wait_for(client.send(req, stream=True), first_token_timeout)
                    except asyncio.TimeoutError:
                        yield {"type": "slow", "model": payload.get("model"), "after": first_token_timeout}
                        return
                    try:
                        if r.status_code != 200:
                            body = (await r.aread()).decode(errors="replace")
                            POOL.report(entry, r.status_code, body)
                            last_err = f"HTTP {r.status_code}: {body[:400]}"
                            if r.status_code in (400, 404, 422):
                                yield {"type": "error", "message": last_err, "fatal": True, "status": r.status_code}
                                return
                            continue
                        POOL.report(entry, 200)
                        tool_acc: dict[int, dict] = {}
                        splitter = ThinkSplitter()
                        lines = r.aiter_lines().__aiter__()
                        while True:
                            try:
                                if first_token_timeout and not started_output:
                                    line = await asyncio.wait_for(lines.__anext__(), first_token_timeout)
                                else:
                                    line = await lines.__anext__()
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                yield {"type": "slow", "model": payload.get("model"), "after": first_token_timeout}
                                return
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                continue
                            if chunk.get("usage"):
                                yield {"type": "usage", "usage": chunk["usage"]}
                            for ch in chunk.get("choices") or []:
                                delta = ch.get("delta") or {}
                                rc = delta.get("reasoning_content") or delta.get("reasoning")
                                if rc:
                                    started_output = True
                                    yield {"type": "reasoning", "text": rc}
                                if delta.get("content"):
                                    started_output = True
                                    for kind, txt in splitter.feed(delta["content"]):
                                        yield {"type": kind, "text": txt}
                                for tc in delta.get("tool_calls") or []:
                                    started_output = True
                                    idx = tc.get("index", len(tool_acc))
                                    acc = tool_acc.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                                    if tc.get("id"):
                                        acc["id"] = tc["id"]
                                    fn = tc.get("function") or {}
                                    if fn.get("name") and not acc["name"]:
                                        acc["name"] = fn["name"]
                                    if fn.get("arguments"):
                                        acc["arguments"] += fn["arguments"]
                        for kind, txt in splitter.flush():
                            yield {"type": kind, "text": txt}
                        if tool_acc:
                            calls = []
                            for i in sorted(tool_acc):
                                a = tool_acc[i]
                                calls.append({"id": a["id"] or f"call_{uuid.uuid4().hex[:8]}", "name": a["name"], "arguments": a["arguments"]})
                            yield {"type": "tool_calls", "calls": calls}
                        return
                    finally:
                        await r.aclose()
            except (httpx.HTTPError, asyncio.TimeoutError) as e:
                POOL.report(entry, None, str(e))
                last_err = f"{type(e).__name__}: {e}"
                if started_output:
                    yield {"type": "error", "message": f"stream interrupted: {last_err}"}
                    return
                continue
        await asyncio.sleep(2)
    yield {"type": "error", "message": f"all keys failed: {last_err}", "fatal": True}


# ----------------------------------------------------------------------------- prompts / history


def default_system_prompt() -> str:
    import platform

    mem = ""
    if MEMORY_FILE.exists():
        mem = MEMORY_FILE.read_text()[-4000:]
    return (
        "You are Arety, a capable, direct AI assistant with an agentic tool harness. "
        "You run on NVIDIA NIM and can act on the server that hosts this chat. You have senses: images the user "
        "attaches are described to you by the PaliGemma vision model (and you can ask it follow-up questions with "
        "look_at_image), audio is transcribed by Whisper (transcribe_audio).\n"
        f"Environment: {platform.node()} ({platform.system()} {platform.release()}), workspace {WORKSPACE}, "
        f"today {datetime.now().strftime('%Y-%m-%d')}.\n"
        "Guidelines: prefer tools over guessing when facts, files, commands or current data are involved; "
        "chain several tool calls when a task needs it; keep going until the task is actually done; "
        "report real results (including failures) plainly. Use markdown; put code in fenced blocks. "
        "Use the remember tool for durable facts about the user or their projects.\n"
        "Deliverables: the user cannot see the server's filesystem. When you produce something for them (a worksheet, "
        "document, script, table), put its full content in your reply, and if you also saved it in the workspace give "
        "the markdown download link that write_file returns (paths under the workspace are served at /api/files/<relative path>). "
        "Never just say a file was saved.\n"
        + (f"\nLong-term memory notes:\n{mem}" if mem.strip() else "")
    )


def converse_system_prompt() -> str:
    mem = MEMORY_FILE.read_text()[-3000:] if MEMORY_FILE.exists() else ""
    return (
        "You are Arety in Converse mode: a live spoken conversation. The user's words come from speech recognition "
        "(so tolerate small transcription errors) and your reply is read aloud by text-to-speech.\n"
        "Speak naturally and briefly: one to three sentences unless the user asks for detail. Plain prose only: "
        "no markdown, no bullet lists, no headings, no code blocks, no emojis, no URLs. Say numbers and units the way "
        "a person would. Be warm, direct and a little playful. If you need a tool, use it quickly and summarise the "
        "result in speech. Ask a short follow-up question only when it moves things forward.\n"
        f"Today is {datetime.now().strftime('%A %Y-%m-%d')}."
        + (f"\nThings you remember about the user:\n{mem}" if mem.strip() else "")
    )


def build_history(messages: list[dict], vision: bool = False) -> list[dict]:
    """Convert stored messages to API messages, dropping UI-only fields and trimming to a char budget."""
    api: list[dict] = []
    for m in messages:
        role = m["role"]
        if role == "user":
            content = m["content"]
            if isinstance(content, list) and not vision:
                content = "\n".join(p.get("text", "") if p.get("type") == "text" else "[image attached]" for p in content)
            api.append({"role": "user", "content": content})
        elif role == "assistant":
            msg: dict[str, Any] = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function", "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                    for tc in m["tool_calls"]
                ]
                if not msg["content"]:
                    msg["content"] = None
            api.append(msg)
        elif role == "tool":
            api.append({"role": "tool", "tool_call_id": m["tool_call_id"], "name": m.get("name", ""), "content": m["content"]})
    # trim oldest turns to fit budget, never splitting an assistant/tool group
    def size(x: dict) -> int:
        c = x.get("content")
        return len(json.dumps(c)) if not isinstance(c, str) else len(c)

    total = sum(size(x) for x in api)
    while len(api) > 2 and total > HISTORY_CHAR_BUDGET:
        total -= size(api.pop(0))
        while len(api) > 1 and api[0]["role"] != "user":
            total -= size(api.pop(0))
    return api


# ----------------------------------------------------------------------------- auth

COOKIE = "nc_session"


def _token() -> str:
    return hmac.new(SESSION_SECRET.encode(), b"nim-chat-session", hashlib.sha256).hexdigest()


def _authed(req: Request) -> bool:
    if not CHAT_PASSWORD:
        return True
    return hmac.compare_digest(req.cookies.get(COOKIE, ""), _token())


app = FastAPI(title="nim-chat")


@app.middleware("http")
async def auth_mw(request: Request, call_next):
    p = request.url.path
    if p.startswith("/api/") and p not in ("/api/login", "/api/me") and not _authed(request):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/api/me")
async def me(request: Request):
    return {"authed": _authed(request), "needs_password": bool(CHAT_PASSWORD), "default_model": DEFAULT_MODEL}


@app.post("/api/login")
async def login(request: Request):
    body = await request.json()
    if CHAT_PASSWORD and not hmac.compare_digest(body.get("password", ""), CHAT_PASSWORD):
        await asyncio.sleep(1)
        raise HTTPException(401, "wrong password")
    resp = JSONResponse({"ok": True})
    resp.set_cookie(COOKIE, _token(), max_age=60 * 60 * 24 * 90, httponly=True, samesite="lax", secure=request.url.scheme == "https" or "example.com" in request.headers.get("host", ""))
    return resp


@app.post("/api/logout")
async def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(COOKIE)
    return resp


# ----------------------------------------------------------------------------- api

_models_cache: dict = {"at": 0, "data": []}
_SKIP = ("embed", "reward", "guard", "safety", "parse", "detector", "nvclip", "translate", "deplot", "fuyu", "kosmos",
         "neva", "vila", "diffusion", "muse", "calibration", "cosmos", "topic-control", "starcoder", "codegemma", "chatqa")


@app.get("/api/models")
async def models(refresh: bool = False):
    if refresh or time.time() - _models_cache["at"] > 600 or not _models_cache["data"]:
        ids: list[str] = []
        for entry in POOL.order():
            try:
                async with httpx.AsyncClient(timeout=20) as cl:
                    r = await cl.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {entry['key']}"})
                if r.status_code == 200:
                    ids = [m["id"] for m in r.json().get("data", [])]
                    break
            except httpx.HTTPError:
                continue
        chat = sorted(i for i in ids if not any(s in i for s in _SKIP))
        if chat:
            _models_cache.update(at=time.time(), data=chat)
    data = _models_cache["data"]
    featured = [OMNI_MODEL_ID] + [m for m in (
        DEFAULT_MODEL, "nvidia/nemotron-3-ultra-550b-a55b", "nvidia/nemotron-3.5-lightning-30b-a3b", "z-ai/glm-5.3-flash",
        "openai/gpt-oss-20b", "deepseek-ai/deepseek-v4-flash-0731", "deepseek-ai/deepseek-v4-pro-0813", "moonshotai/kimi-k3",
        "moonshotai/kimi-k2.6", "mistralai/mistral-large-2-instruct", "google/gemma-4-31b-it", "meta/llama-3.2-90b-vision-instruct",
    ) if m in data or not data]
    return {"default": DEFAULT_MODEL, "featured": list(dict.fromkeys(featured)), "all": data,
            "omni": {"id": OMNI_MODEL_ID, "brain": OMNI_CHAT_MODEL, "fallback": OMNI_FALLBACK_MODEL, "vision": VLM_MODEL, "asr": ASR_MODEL}}


@app.get("/api/status")
async def status():
    return {"keys": POOL.status(), "uptime": int(time.time() - STARTED), "workspace": str(WORKSPACE),
            "slow_brains": {m: int(t - time.time()) for m, t in _SLOW_UNTIL.items() if t > time.time()},
            "tools": list(TOOLS), "default_model": DEFAULT_MODEL, "memory": MEMORY_FILE.read_text() if MEMORY_FILE.exists() else ""}


@app.get("/api/conversations")
async def conversations():
    return conv_list()


@app.get("/api/conversations/{cid}")
async def conversation(cid: str):
    d = conv_get(cid)
    if not d:
        raise HTTPException(404)
    return d


@app.patch("/api/conversations/{cid}")
async def conversation_patch(cid: str, request: Request):
    d = conv_get(cid)
    if not d:
        raise HTTPException(404)
    body = await request.json()
    for k in ("title", "model", "system_prompt", "settings"):
        if k in body:
            d[k] = body[k]
    if "messages" in body:  # used by "delete message / regenerate from here"
        d["messages"] = body["messages"]
    conv_save(d)
    return {"ok": True}


@app.delete("/api/conversations/{cid}")
async def conversation_delete(cid: str):
    conv_delete(cid)
    return {"ok": True}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    name = re.sub(r"[^A-Za-z0-9._-]", "_", file.filename or "upload")
    dest = WORKSPACE / "uploads" / f"{int(time.time())}_{name}"
    data = await file.read()
    dest.write_bytes(data)
    mime = file.content_type or mimetypes.guess_type(name)[0] or "application/octet-stream"
    info: dict[str, Any] = {"path": str(dest), "name": name, "size": len(data), "mime": mime}
    if mime.startswith("image/"):
        small, smime = _shrink_image(data)
        info["data_url"] = f"data:{smime};base64,{base64.b64encode(small).decode()}"
        thumb, tmime = _shrink_image(data, max_px=256, quality=70)
        info["thumb"] = f"data:{tmime};base64,{base64.b64encode(thumb).decode()}"
    else:
        try:
            text = data.decode("utf-8")
            info["text"] = text[:60000]
        except UnicodeDecodeError:
            pass
    return info


@app.get("/api/files/{path:path}")
async def files(path: str, dl: bool = False):
    """Serve a file from the workspace (what the agent wrote or the user uploaded)."""
    base = WORKSPACE.resolve()
    fp = (base / path).resolve()
    if fp == base or base not in fp.parents:
        raise HTTPException(403, "outside workspace")
    if not fp.is_file():
        raise HTTPException(404, "not found")
    mime = mimetypes.guess_type(fp.name)[0] or "application/octet-stream"
    if mime.startswith("text/") or fp.suffix in (".md", ".json", ".py", ".sh", ".csv", ".txt"):
        mime = "text/plain; charset=utf-8" if not dl else mime
    headers = {"Content-Disposition": f'{"attachment" if dl else "inline"}; filename="{fp.name}"'}
    return FileResponse(fp, media_type=mime, headers=headers)


@app.post("/api/memory")
async def memory_set(request: Request):
    body = await request.json()
    MEMORY_FILE.write_text(body.get("text", ""))
    return {"ok": True}


_STOP: set[str] = set()
_SLOW_UNTIL: dict[str, float] = {}  # brain model → time until which we skip it (circuit breaker after a first-token timeout)
SLOW_BREAKER_SECONDS = float(os.environ.get("SLOW_BREAKER_SECONDS", "600"))


@app.post("/api/stop/{cid}")
async def stop(cid: str):
    _STOP.add(cid)
    return {"ok": True}


def _sse(ev: dict) -> str:
    return f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"


async def heartbeat(agen: AsyncIterator[str], every: float = 15.0) -> AsyncIterator[str]:
    """Interleave SSE comment pings so proxies (Cloudflare/Apache) never see an idle stream."""
    q: asyncio.Queue = asyncio.Queue()
    _END = object()

    async def pump():
        try:
            async for item in agen:
                await q.put(item)
        except Exception as e:  # noqa: BLE001
            await q.put(_sse({"type": "error", "message": f"server error: {type(e).__name__}: {e}"}))
        finally:
            await q.put(_END)

    task = asyncio.create_task(pump())
    try:
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=every)
            except asyncio.TimeoutError:
                yield ": ping\n\n"
                continue
            if item is _END:
                break
            yield item
    finally:
        task.cancel()


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    cid = body.get("conversation_id") or uuid.uuid4().hex[:12]
    d = conv_get(cid) or {"id": cid, "title": "", "model": DEFAULT_MODEL, "system_prompt": "", "settings": {}, "messages": [], "created": time.time()}
    if body.get("model"):
        d["model"] = body["model"]
    if "system_prompt" in body:
        d["system_prompt"] = body["system_prompt"] or ""
    settings = {**d.get("settings", {}), **(body.get("settings") or {})}
    d["settings"] = settings
    mode = body.get("mode") or settings.get("mode") or "chat"
    settings["mode"] = mode
    converse = mode == "converse"
    agent_on = settings.get("agent", not converse)
    temperature = float(settings.get("temperature", 0.7 if converse else 0.6))
    max_tokens = int(settings.get("max_tokens", 400 if converse else 8192))
    enabled_tools = settings.get("tools")  # None → all
    if converse and enabled_tools is None:
        enabled_tools = ["web_search", "web_fetch", "get_datetime", "remember", "run_shell", "python_exec"]
    no_think = converse and not settings.get("thinking")

    user_text = (body.get("message") or "").strip()
    attachments = body.get("attachments") or []
    regenerate = bool(body.get("regenerate"))
    is_omni = d["model"] == OMNI_MODEL_ID
    use_native_vision = vision_capable(d["model"]) and not is_omni
    # senses: run PaliGemma / Whisper over attachments before the brain sees the turn
    sense_notes: list[dict] = []
    for a in attachments:
        try:
            if a.get("mime", "").startswith("image/") and not use_native_vision and a.get("path"):
                res = await describe_image(Path(a["path"]).read_bytes(), user_text)
                sense_notes.append({"kind": "vision", "name": a.get("name"), **res})
            elif (a.get("mime", "").startswith(("audio/", "video/")) or Path(a.get("name", "")).suffix.lower() in (".mp3", ".wav", ".m4a", ".ogg", ".webm", ".mp4", ".flac")) and a.get("path"):
                r = await transcribe_bytes(Path(a["path"]).read_bytes(), Path(a["path"]).suffix)
                sense_notes.append({"kind": "audio", "name": a.get("name"), "transcript": r["text"], "engine": r["engine"]})
        except HTTPException as e:
            sense_notes.append({"kind": "error", "name": a.get("name"), "error": str(e.detail)})
        except Exception as e:  # noqa: BLE001
            sense_notes.append({"kind": "error", "name": a.get("name"), "error": f"{type(e).__name__}: {e}"})
    if regenerate:
        # drop trailing assistant/tool messages so the last user turn is re-answered
        while d["messages"] and d["messages"][-1]["role"] != "user":
            d["messages"].pop()
    elif user_text or attachments:
        content: Any = user_text
        parts_text = []
        images = []
        for a in attachments:
            if a.get("data_url") and use_native_vision:
                images.append(a["data_url"])
            elif a.get("data_url"):
                note = next((n for n in sense_notes if n.get("name") == a.get("name") and n["kind"] == "vision"), None)
                if note:
                    parts_text.append(f"\n\n[Image {a.get('name')} saved at {a.get('path')}. PaliGemma sees: {note['description']}"
                                      + (f" | Answer to the user's question: {note['answer']}" if note.get("answer") else "") + "]")
                else:
                    parts_text.append(f"\n\n[Image {a.get('name')} saved at {a.get('path')} (vision unavailable)]")
            elif any(n.get("name") == a.get("name") and n["kind"] == "audio" for n in sense_notes):
                note = next(n for n in sense_notes if n.get("name") == a.get("name") and n["kind"] == "audio")
                parts_text.append(f"\n\n[Audio {a.get('name')} saved at {a.get('path')}; Whisper transcript: {note['transcript'] or '(no speech)'}]")
            elif a.get("text") is not None:
                parts_text.append(f"\n\n[Attached file {a.get('name')} saved at {a.get('path')}]\n```\n{a['text'][:40000]}\n```")
            else:
                parts_text.append(f"\n\n[Attached file {a.get('name')} saved at {a.get('path')} ({a.get('size')} bytes)]")
        text_full = user_text + "".join(parts_text)
        if images:
            content = [{"type": "text", "text": text_full or "(image)"}] + [{"type": "image_url", "image_url": {"url": u}} for u in images]
        else:
            content = text_full
        d["messages"].append({"role": "user", "content": content, "ts": time.time(), "senses": sense_notes,
                              "attachments": [{k: a.get(k) for k in ("name", "path", "size", "mime", "thumb")} for a in attachments]})
        if not d["title"]:
            d["title"] = ("🎙 " if converse else "") + (user_text or (attachments[0].get("name") if attachments else "New chat"))[:60]
    else:
        raise HTTPException(400, "empty message")
    conv_save(d)
    _STOP.discard(cid)

    async def gen():
        yield _sse({"type": "start", "conversation_id": cid, "title": d["title"], "model": d["model"]})
        for n in (d["messages"][-1].get("senses") or []) if d["messages"] else []:
            yield _sse({"type": "sense", **n})
        brain = OMNI_CHAT_MODEL if is_omni else d["model"]
        fallback = (OMNI_CONVERSE_FALLBACK_MODEL if converse else OMNI_FALLBACK_MODEL) if is_omni else None
        first_token_timeout = (CONVERSE_FIRST_TOKEN_TIMEOUT if converse else OMNI_FIRST_TOKEN_TIMEOUT) if is_omni else None
        if fallback and _SLOW_UNTIL.get(brain, 0) > time.time():
            yield _sse({"type": "notice", "text": f"{brain.split('/')[-1]} was unresponsive recently; using {fallback.split('/')[-1]} (retrying it in {int((_SLOW_UNTIL[brain] - time.time()) / 60) + 1} min)."})
            brain, first_token_timeout = fallback, None
        sys_prompt = d["system_prompt"].strip() or (converse_system_prompt() if converse else default_system_prompt())
        tools = tool_specs(enabled_tools) if agent_on else []
        use_tools = bool(tools)
        use_no_think = no_think and brain.startswith("nvidia/")
        total_usage: dict[str, int] = {}
        for step in range(MAX_STEPS):
            if cid in _STOP or await request.is_disconnected():
                yield _sse({"type": "stopped"})
                break
            payload = {
                "model": brain,
                "messages": [{"role": "system", "content": sys_prompt}] + build_history(d["messages"], vision=use_native_vision),
                "stream": True,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if use_tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            if use_no_think:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            yield _sse({"type": "step", "n": step + 1, "brain": brain})
            assistant = {"role": "assistant", "content": "", "reasoning": "", "tool_calls": [], "ts": time.time(), "model": brain}
            error = None
            slow = False
            async for ev in stream_completion(payload, first_token_timeout=first_token_timeout):
                if cid in _STOP or await request.is_disconnected():
                    break
                t = ev["type"]
                if t == "slow":
                    slow = True
                    break
                if t == "content":
                    assistant["content"] += ev["text"]
                    yield _sse({"type": "delta", "text": ev["text"]})
                elif t == "reasoning":
                    assistant["reasoning"] += ev["text"]
                    yield _sse({"type": "reasoning", "text": ev["text"]})
                elif t == "tool_calls":
                    assistant["tool_calls"] = ev["calls"]
                elif t == "usage":
                    for k, v in ev["usage"].items():
                        if isinstance(v, int):
                            total_usage[k] = total_usage.get(k, 0) + v
                elif t == "error":
                    error = ev
            if slow and fallback and brain != fallback:
                _SLOW_UNTIL[brain] = time.time() + SLOW_BREAKER_SECONDS
                yield _sse({"type": "notice", "text": f"{brain.split('/')[-1]} gave no answer within {int(first_token_timeout)} s; switching to {fallback.split('/')[-1]}."})
                brain = fallback
                use_no_think = no_think and brain.startswith("nvidia/")
                first_token_timeout = None
                continue
            if slow:
                yield _sse({"type": "error", "message": f"{brain} did not respond in time."})
                break
            if error and error.get("fatal") and error.get("status") in (400, 422) and not assistant["content"]:
                if use_no_think:  # model rejects chat_template_kwargs → retry without
                    use_no_think = False
                    continue
                if use_tools:  # model rejects tools → retry this step without them
                    use_tools = False
                    yield _sse({"type": "notice", "text": "This model rejected tool definitions; continuing without tools."})
                    continue
            if assistant["content"] or assistant["reasoning"] or assistant["tool_calls"]:
                d["messages"].append(assistant)
                conv_save(d)
            if error:
                yield _sse({"type": "error", "message": error["message"]})
                break
            if not assistant["tool_calls"]:
                break
            # execute tools
            for tc in assistant["tool_calls"]:
                try:
                    args = json.loads(tc["arguments"] or "{}")
                    if not isinstance(args, dict):
                        args = {}
                except json.JSONDecodeError:
                    args = {}
                    out = f"[could not parse arguments: {tc['arguments'][:200]}]"
                else:
                    out = None
                yield _sse({"type": "tool_call", "id": tc["id"], "name": tc["name"], "arguments": args})
                if out is None:
                    if cid in _STOP:
                        out = "[stopped by user]"
                    else:
                        out = await run_tool(tc["name"], args)
                d["messages"].append({"role": "tool", "tool_call_id": tc["id"], "name": tc["name"], "content": out, "ts": time.time()})
                yield _sse({"type": "tool_result", "id": tc["id"], "name": tc["name"], "output": out})
            conv_save(d)
            if cid in _STOP:
                yield _sse({"type": "stopped"})
                break
        else:
            yield _sse({"type": "notice", "text": f"Stopped after {MAX_STEPS} agent steps."})
        _STOP.discard(cid)
        conv_save(d)
        yield _sse({"type": "done", "conversation_id": cid, "title": d["title"], "usage": total_usage})

    return StreamingResponse(heartbeat(gen()), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ----------------------------------------------------------------------------- converse: STT + TTS

_whisper = {"model": None, "lock": asyncio.Lock(), "error": ""}


def _whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


async def _get_whisper():
    if _whisper["model"] is None:
        async with _whisper["lock"]:
            if _whisper["model"] is None:
                from faster_whisper import WhisperModel
                _whisper["model"] = await asyncio.to_thread(WhisperModel, WHISPER_MODEL, device="cpu", compute_type="int8")
    return _whisper["model"]


@app.get("/api/converse/config")
async def converse_config():
    return {
        "model": CONVERSE_MODEL,
        "elevenlabs": bool(ELEVENLABS_API_KEY),
        "voice_id": ELEVENLABS_VOICE_ID,
        "server_stt": _whisper_available() or _asr_available(),
        "asr_nim": _asr_available(),
        "asr_model": ASR_MODEL,
        "omni": OMNI_MODEL_ID,
    }


@app.post("/api/stt")
async def stt(file: UploadFile = File(...), language: str = "", engine: str = "auto"):
    data = await file.read()
    if len(data) < 500:
        return {"text": "", "engine": "none"}
    return await transcribe_bytes(data, Path(file.filename or "a.webm").suffix or ".webm", language, engine)


@app.post("/api/tts")
async def tts(request: Request):
    if not ELEVENLABS_API_KEY:
        raise HTTPException(501, "ElevenLabs not configured")
    body = await request.json()
    text = (body.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "empty text")
    voice = body.get("voice_id") or ELEVENLABS_VOICE_ID
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream?output_format=mp3_22050_32"
    client = httpx.AsyncClient(timeout=60)
    req = client.build_request("POST", url, json={"text": text, "model_id": body.get("model_id") or ELEVENLABS_TTS_MODEL},
                               headers={"xi-api-key": ELEVENLABS_API_KEY})
    r = await client.send(req, stream=True)
    if r.status_code != 200:
        detail = (await r.aread()).decode(errors="replace")[:300]
        await r.aclose(); await client.aclose()
        raise HTTPException(502, f"ElevenLabs {r.status_code}: {detail}")

    async def body_iter():
        try:
            async for chunk in r.aiter_bytes():
                yield chunk
        finally:
            await r.aclose(); await client.aclose()

    return StreamingResponse(body_iter(), media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@app.get("/")
async def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/healthz")
async def healthz():
    return {"ok": True, "keys": len(KEYS)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
