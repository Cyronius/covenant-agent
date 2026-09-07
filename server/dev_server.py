"""Local backend for the kanban-ui demo (client/kanban-ui) and the eval
harness's browser-parity checks. Pure Python stdlib.

Jobs:
  1. Serves the GGUF model files straight out of baselines/qwen/models/
     (Range requests, never copied), the grammar at
     baselines/qwen/agent_core.gbnf, and — when it exists — the built
     kanban-ui app from client/kanban-ui/dist (SPA fallback to index.html).
     In development Vite serves the app itself and proxies to this server
     (client/kanban-ui/vite.config.ts).
  2. POST /kanban_prompt: builds a real TOOLS/FIELDS/CONSTANTS context for a
     free-typed request against the client's board (handle_kanban_prompt).
  3. POST /plan (+ GET /plan/status): server-side inference — the same GGUF
     and grammar the browser path uses, run through llama-cpp-python on
     this machine's CPU (plan s2-consolidated-program §A7). --model picks
     the checkpoint; it loads lazily on first use.
  4. POST /validate: takes generated Agent Core program text and round-trips
     it through the *existing* Python pipeline (core.pipeline.build) and
     sandbox executor (harness.run.run_sandbox) — the same path
     harness/run.py's run_task() uses. Not a reimplementation.

Pure Python stdlib, plus llama-cpp-python for /plan only.

Run (from the repo root):
    python server/dev_server.py --port 8080 [--model baselines/qwen/models/<gguf>]

See server/README.md for the request/response contracts.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# server/dev_server.py -> repo root is one level up.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.ir import TaskContext, parse_type  # noqa: E402
from core.pipeline import build  # noqa: E402
from harness.context import build_context, sandbox_from_context, serialize_context  # noqa: E402
from harness.run import run_sandbox  # noqa: E402
from runtime.worlds import get_world, rpg  # noqa: E402

APP_DIST = ROOT / "client" / "kanban-ui" / "dist"
RPG_DIST = ROOT / "client" / "rpg-ui" / "dist"
MODELS_DIR = ROOT / "baselines" / "qwen" / "models"
GRAMMAR_FILE = ROOT / "baselines" / "qwen" / "agent_core.gbnf"
TASKS_FILE = ROOT / "data" / "curriculum_tasks.jsonl"
DEFAULT_PLAN_MODEL = MODELS_DIR / "qwen3.5-0.8b-s2-pruned-q8.gguf"
# Our own tuned checkpoints were SFT'd against the hand-rolled ChatML markup
# in baselines/qwen/run_a.py; any other GGUF gets its own chat template
# applied by llama-cpp-python instead (see ServerPlanner.generate_chat).
TUNED_PREFIXES = ("qwen3.5-0.8b-s", "qwen3.5-2b-cond", "qwen3.5-0.8b-cond")
# The writer uses the UNTUNED base weights: the merged S1 checkpoint has lost
# its general writing (it echoes the data list back; measured 2026-09-02,
# plan s2-consolidated-program §A8), while the base 0.8B writes a proper
# short message. Falls back to the planner weights if this file is absent.
DEFAULT_WRITER_MODEL = MODELS_DIR / "Qwen3.5-0.8B-Q8_0.gguf"


class ServerPlanner:
    """Lazily-loaded llama-cpp-python model + grammar behind POST /plan.
    llama.cpp contexts are not thread-safe and ThreadingHTTPServer is
    threaded, so generation is serialized with a lock."""

    def __init__(self, model_path: Path | None, n_ctx: int = 4096,
                 writer_path: Path | None = None):
        self.model_path = model_path
        self.writer_path = writer_path
        self.n_ctx = n_ctx
        self._llm = None
        self._writer = None
        self._grammar = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return bool(self.model_path and self.model_path.exists())

    def status(self) -> dict:
        return {"available": self.available,
                "model": self.model_path.name if self.model_path else None,
                "loaded": self._llm is not None,
                "writer_model": (self.writer_path.name if self.writer_path and self.writer_path.exists()
                                 else (self.model_path.name if self.model_path else None)),
                **({} if self.available else
                   {"reason": f"model file not found: {self.model_path}"})}

    def _ensure(self):
        if self._llm is None:
            from llama_cpp import Llama, LlamaGrammar
            self._grammar = LlamaGrammar.from_string(
                GRAMMAR_FILE.read_text(encoding="utf-8"), verbose=False)
            self._llm = Llama(model_path=str(self.model_path), n_ctx=self.n_ctx,
                              verbose=False)

    def switch(self, name: str) -> dict:
        """Load a different checkpoint from MODELS_DIR. One planner at a time:
        a 2B Q8 is ~2.5 GB, so the previous model is dropped rather than
        cached. The writer instance is untouched."""
        target = MODELS_DIR / name
        if "/" in name or "\\" in name or not target.exists():
            raise ValueError(f"no such model: {name}")
        with self._lock:
            if self.model_path and target.samefile(self.model_path) and self._llm:
                return self.status()
            self._llm = None
            self._grammar = None
            self.model_path = target
        return self.status()

    def generate(self, prompt: str, max_tokens: int, stop: list) -> dict:
        with self._lock:
            self._ensure()
            t0 = time.perf_counter()
            res = self._llm.create_completion(
                prompt, grammar=self._grammar, temperature=0.0,
                max_tokens=max_tokens, stop=stop or ["<|im_end|>"])
            gen_ms = (time.perf_counter() - t0) * 1000
        choice = res["choices"][0]
        return {"text": choice["text"], "finish_reason": choice["finish_reason"],
                "tokens_out": res.get("usage", {}).get("completion_tokens", 0),
                "tokens_in": res.get("usage", {}).get("prompt_tokens", 0),
                "gen_ms": gen_ms, "model": self.model_path.name}

    def generate_chat(self, system: str, user: str, max_tokens: int) -> dict:
        """For a model that is not one of ours: llama-cpp-python applies the
        GGUF's own tokenizer.chat_template, so the client does not have to
        know the markup. Same grammar, same greedy decoding."""
        with self._lock:
            self._ensure()
            t0 = time.perf_counter()
            res = self._llm.create_chat_completion(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                grammar=self._grammar, temperature=0.0, max_tokens=max_tokens)
            gen_ms = (time.perf_counter() - t0) * 1000
        choice = res["choices"][0]
        return {"text": choice["message"].get("content") or "",
                "finish_reason": choice.get("finish_reason"),
                "tokens_out": res.get("usage", {}).get("completion_tokens", 0),
                "tokens_in": res.get("usage", {}).get("prompt_tokens", 0),
                "gen_ms": gen_ms, "model": self.model_path.name}

    def _ensure_writer(self):
        """Base weights if present (separate llama.cpp instance, ~0.8 GB more
        RAM); otherwise the planner weights."""
        if self.writer_path and self.writer_path.exists():
            if self._writer is None:
                from llama_cpp import Llama
                self._writer = Llama(model_path=str(self.writer_path), n_ctx=2048,
                                     verbose=False)
            return self._writer
        self._ensure()
        return self._llm

    def warm(self) -> dict:
        with self._lock:
            self._ensure()
        return self.status()

    WRITER_SYSTEM = ("You write short, plain workplace messages. Output only the "
                     "message text — no greeting line, no sign-off, no markdown.")

    def write(self, brief: str, data: list, max_tokens: int = 120) -> dict:
        """POST /write: the writer tool (plan §A8, option a — same weights,
        second prompt, NO grammar). The planner never sees this text; it only
        routes it."""
        lines = []
        for rec in data or []:
            if not isinstance(rec, dict):
                lines.append(f"- {rec}")
                continue
            bits = [str(rec.get("title") or rec.get("name") or rec.get("id"))]
            if "status" in rec:
                bits.append(f"status {rec['status']}")
            if isinstance(rec.get("due"), (int, float)):
                import datetime as _dt
                bits.append("due " + _dt.datetime.fromtimestamp(
                    rec["due"], _dt.timezone.utc).strftime("%Y-%m-%d"))
            lines.append("- " + ", ".join(bits))
        user = (f"Brief: {brief}\n" + ("Items:\n" + "\n".join(lines) if lines else "Items: none")
                + "\n\nWrite the message.")
        prompt = (f"<|im_start|>system\n{self.WRITER_SYSTEM}<|im_end|>\n"
                  f"<|im_start|>user\n{user}<|im_end|>\n"
                  f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
        with self._lock:
            llm = self._ensure_writer()
            t0 = time.perf_counter()
            res = llm.create_completion(
                prompt, temperature=0.0, max_tokens=max_tokens, stop=["<|im_end|>"])
            gen_ms = (time.perf_counter() - t0) * 1000
        text = res["choices"][0]["text"].strip()
        return {"text": text, "gen_ms": gen_ms,
                "tokens_out": res.get("usage", {}).get("completion_tokens", 0)}


PLANNER = ServerPlanner(DEFAULT_PLAN_MODEL, writer_path=DEFAULT_WRITER_MODEL)
SELF_URL: str | None = None  # set in main(); lets the sandbox call back for EXTERNAL tools

EXTRA_MIME_TYPES = {
    ".wasm": "application/wasm",
    ".js": "text/javascript",
    ".mjs": "text/javascript",
    ".json": "application/json",
    ".html": "text/html",
    ".gbnf": "text/plain",
    ".gguf": "application/octet-stream",
}

RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)")


def guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in EXTRA_MIME_TYPES:
        return EXTRA_MIME_TYPES[ext]
    ctype, _ = mimetypes.guess_type(str(path))
    return ctype or "text/plain"


def load_tasks(path: Path) -> dict:
    """Index data/curriculum_tasks.jsonl by task id, once at startup."""
    tasks = {}
    if not path.exists():
        return tasks
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tasks[row["id"]] = row
    return tasks


TASKS: dict = {}


class DevHandler(BaseHTTPRequestHandler):
    server_version = "CovenantAgentDevPOC/0.1"

    # ---- helpers ---------------------------------------------------
    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, text: str, content_type: str = "text/plain") -> None:
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _resolve_static_path(self, url_path: str) -> Path | None:
        """Map a URL path to a file under a built app, guarding traversal.
        `/rpg/...` serves client/rpg-ui/dist, everything else the kanban app.
        Unknown paths fall back to that app's index.html (SPA)."""
        rel = url_path.lstrip("/")
        root = APP_DIST
        if rel == "rpg" or rel.startswith("rpg/"):
            root = RPG_DIST
            rel = rel[4:]
        if rel == "" or rel == "/":
            rel = "index.html"
        candidate = (root / rel).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError:
            return None
        if not candidate.is_file():
            candidate = root / "index.html"
        return candidate

    def _serve_file(self, path: Path, support_range: bool = False) -> None:
        if not path.exists() or not path.is_file():
            self._send_text(404, f"Not found: {path.name}\n")
            return
        content_type = guess_content_type(path)
        size = path.stat().st_size

        if support_range:
            range_header = self.headers.get("Range")
            if range_header:
                m = RANGE_RE.match(range_header.strip())
                if not m:
                    self._send_text(416, "Invalid Range header\n")
                    return
                start_s, end_s = m.groups()
                if start_s == "" and end_s == "":
                    self._send_text(416, "Invalid Range header\n")
                    return
                if start_s == "":
                    # suffix range: last N bytes
                    length = int(end_s)
                    start = max(0, size - length)
                    end = size - 1
                else:
                    start = int(start_s)
                    end = int(end_s) if end_s != "" else size - 1
                end = min(end, size - 1)
                if start > end or start >= size:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                chunk_len = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(chunk_len))
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = chunk_len
                    while remaining > 0:
                        block = f.read(min(1 << 20, remaining))
                        if not block:
                            break
                        self.wfile.write(block)
                        remaining -= len(block)
                return
            # Non-range GET on a range-capable resource: whole file, but
            # advertise Range support.
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            with open(path, "rb") as f:
                while True:
                    block = f.read(1 << 20)
                    if not block:
                        break
                    self.wfile.write(block)
            return

        # Plain static file, no range support needed.
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        with open(path, "rb") as f:
            self.wfile.write(f.read())

    # ---- routing -----------------------------------------------------
    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        try:
            if path.startswith("/models/"):
                filename = path[len("/models/"):]
                if "/" in filename or "\\" in filename or filename in ("", "."):
                    self._send_text(400, "Bad model filename\n")
                    return
                self._serve_file(MODELS_DIR / filename, support_range=True)
                return

            if path == "/agent_core.gbnf":
                self._serve_file(GRAMMAR_FILE)
                return

            if path == "/plan/status":
                self._send_json(200, PLANNER.status())
                return

            if path == "/models":
                # matched before the static fallback below, which would
                # otherwise answer a bare /models with index.html
                self._send_json(200, list_models())
                return

            static_path = self._resolve_static_path(path)
            if static_path is None:
                self._send_text(403, "Forbidden\n")
                return
            self._serve_file(static_path)
        except Exception:
            self._send_text(500, f"Internal error:\n{traceback.format_exc()}\n")

    def do_HEAD(self) -> None:
        """Headers only. wllama sends a HEAD for the GGUF before it starts
        ranged GETs; without this, BaseHTTPRequestHandler answers 501 and the
        browser console shows an error on every model load (it still works —
        wllama falls back — but the error is noise, and Content-Length is
        what lets it show real download progress).

        A HEAD response carries no body, error paths included, so this sends
        a bare status rather than going through _send_text."""
        def status_only(code: int) -> None:
            self.send_response(code)
            self.send_header("Content-Length", "0")
            self.end_headers()

        path = self.path.split("?", 1)[0]
        try:
            if path.startswith("/models/"):
                filename = path[len("/models/"):]
                if "/" in filename or "\\" in filename or filename in ("", "."):
                    status_only(400)
                    return
                target = MODELS_DIR / filename
            elif path == "/agent_core.gbnf":
                target = GRAMMAR_FILE
            elif path in ("/plan/status", "/models"):
                status_only(200)
                return
            else:
                target = self._resolve_static_path(path)
            if target is None:
                status_only(403)
                return
            if not target.exists() or not target.is_file():
                status_only(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", guess_content_type(target))
            self.send_header("Content-Length", str(target.stat().st_size))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
        except Exception:
            status_only(500)

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ("/validate", "/kanban_prompt", "/plan", "/write",
                        "/rpg_new", "/rpg_prompt"):
            self._send_text(404, "Not found\n")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw.decode("utf-8"))
            handlers = {"/plan": handle_plan, "/write": handle_write,
                        "/kanban_prompt": handle_kanban_prompt,
                        "/rpg_new": handle_rpg_new,
                        "/rpg_prompt": handle_rpg_prompt,
                        "/validate": handle_validate}
            resp = handlers[path](req)
            self._send_json(200, resp)
        except Exception as exc:
            self._send_json(500, {
                "status": "server_error",
                "error": {"code": "SERVER_ERROR", "message": str(exc)},
                "traceback": traceback.format_exc(),
            })

    def log_message(self, fmt, *args):  # quieter default logging
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def end_headers(self) -> None:
        # Required for wllama's multi-threaded WASM path (SharedArrayBuffer):
        # without these, isSupportMultiThread() is false and generation
        # silently falls back to a single thread. See client/kanban-ui/src/lib/llm.md
        # "Multi-threading" note. Applied to every response; harmless
        # elsewhere since this is a same-origin, single-page dev server.
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


# A handful of reusable message bodies for `send_message`-style calls.
# The grammar can only ever emit a pre-declared C-symbol as a literal (spec
# §"every literal value must be a C symbol") — the model cannot compose new
# text on the fly. For the three fixed curriculum tasks that's fine (the
# task itself supplies the exact message); for free-typed kanban_prompt
# requests there's no such pre-written text, so we offer this small fixed
# bank instead. Real limitation, not hidden: client/kanban-ui/README.md and
# the UI say so.
GENERIC_MESSAGES = [
    "This needs your attention.",
    "Heads up — this is overdue.",
    "Following up on this — any update?",
    "Please take a look when you get a chance.",
    "Reminder: this is due soon.",
]


_WORD_RE = re.compile(r"[a-z]+")
_CARD_NUM_RE = re.compile(r"#?\bcard\s*#?(\d+)\b|(?<!\w)#(\d+)\b")
# Structural/action vocabulary (tool names, effects, board jargon) excluded
# from title-word matching — these show up in both requests and titles
# ("Delete card 4." vs. a card titled "...DELETE effects") without actually
# identifying which card is meant.
_STOPWORDS = {
    "card", "cards", "board", "user", "users", "delete", "archive", "message",
    "status", "update", "assign", "assigned", "create", "done", "todo",
    "doing", "overdue", "urgent", "set", "send", "fetch", "list", "find",
}


def relevant_cards(request: str, cards: list) -> list:
    """Only surface a card as an ID:card constant when the request plausibly
    names it — by number ("card 3", "#3") or a shared title word — rather
    than dumping the whole board as constants on every request.

    This isn't a shortcut: it's matching how data/curriculum_tasks.jsonl's
    own tasks are actually built. A bulk/rule request like "archive the
    overdue cards assigned to Bob" carries ZERO card constants in the real
    curriculum data — the reference solution calls list_cards() and FILTERs,
    which is what the model was tuned on. Dumping all N cards as constants
    for every free-typed request pushed a real, confirmed-by-testing
    failure: "Set card 3 to done." picked the wrong card, consistently,
    against both the tuned 0.8B and the larger 2B checkpoint — a much
    harder disambiguation problem than either was trained to solve, not a
    prompt-wiring bug. Named cards still resolve correctly with this
    filter (and bulk requests still work via list_cards()+FILTER either
    way — this only trims which cards get a shortcut constant)."""
    numbers = {n for pair in _CARD_NUM_RE.findall(request.lower()) for n in pair if n}
    req_words = set(_WORD_RE.findall(request.lower()))
    matched = []
    for card in cards:
        num = card["id"].rsplit("_", 1)[-1]
        if num in numbers:
            matched.append(card)
            continue
        title_words = {w for w in _WORD_RE.findall(card["title"].lower()) if len(w) > 3} - _STOPWORDS
        if title_words & req_words:
            matched.append(card)
    return matched


# --- free-text literal extraction (plan s2-consolidated-program §A2) --------
# The IR carries every literal through a C symbol, so a create-style request
# is only expressible if the serializer lifts its title and due date into
# constants. This is deliberately a small, deterministic heuristic (it has to
# run wherever the demo runs), not an NLP pass. Extracted literals are
# appended AFTER the fixed constants so the indices of cards/users/statuses/
# messages stay stable for authored references (harness/demo_suite.py).

_DQUOTE_RE = re.compile(r'["“”]([^"“”]{2,80})["“”]')
_CREATE_NOUN_RE = re.compile(
    r"\b(?:create|add|open|make|start)\b[^.]*?\b(?:new\s+)?"
    r"(?:issue|card|ticket|task|item)\b\s*(?:for|called|titled|named|about|:|to)?\s+(.+)",
    re.I)
# Where a title stops: a column clause, a due clause, an assignment clause,
# or a conjunction introducing another action.
_TITLE_STOP_RE = re.compile(
    r"\s+(?:(?:in|into|to|on)\s+(?:the\s+)?(?:todo|doing|done)\b"
    r"|due\b|assigned?\b|owned\b|owner\b|and\s+(?:assign|make|set|move|put)\b)",
    re.I)
_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday",
             "saturday", "sunday")
_DATE_PATTERNS = [
    (re.compile(r"\btoday\b", re.I), lambda m: 0),
    (re.compile(r"\btomorrow\b", re.I), lambda m: 1),
    (re.compile(r"\bnext week\b", re.I), lambda m: 7),
    (re.compile(r"\bin\s+(\d+)\s+days?\b", re.I), lambda m: int(m.group(1))),
    (re.compile(r"\bin\s+(\d+)\s+weeks?\b", re.I), lambda m: 7 * int(m.group(1))),
    (re.compile(r"\b(\d+)\s+days?\s+from\s+now\b", re.I), lambda m: int(m.group(1))),
    (re.compile(r"\bin\s+a\s+week\b", re.I), lambda m: 7),
]
_NEXT_WEEKDAY_RE = re.compile(r"\bnext\s+(" + "|".join(_WEEKDAYS) + r")\b", re.I)
_DEFAULT_DUE_DAYS = 7
# FORMAT templates (spec §4) the model can fill from fields. Offered only
# when the request's verb calls for them, so most prompts stay short.
_COPY_RE = re.compile(r"\b(duplicate|copy|clone)\b", re.I)
_REMIND_RE = re.compile(r"\b(remind|reminder|nudge|ping)\b", re.I)


def literals_from_request(request: str, now: int) -> list:
    """STR/TIME constants a free-typed request implies: quoted spans, a
    title when the request creates something, and a due date phrase (or a
    default due when creating with no date given)."""
    out = []
    for m in _DQUOTE_RE.finditer(request):
        out.append({"type": "STR", "value": m.group(1).strip(),
                    "desc": f'the text "{m.group(1).strip()}"'})
    creating = False
    m = _CREATE_NOUN_RE.search(request)
    if m:
        creating = True
        title = m.group(1)
        stop = _TITLE_STOP_RE.search(title)
        if stop:
            title = title[:stop.start()]
        title = title.strip(" .,;:!\"“”")
        if title and not any(c["value"] == title for c in out):
            out.append({"type": "STR", "value": title,
                        "desc": f"title: {title}"})
    days = None
    for pat, fn in _DATE_PATTERNS:
        dm = pat.search(request)
        if dm:
            days = fn(dm)
            phrase = dm.group(0)
            break
    if days is None:
        wm = _NEXT_WEEKDAY_RE.search(request)
        if wm:
            # world `now` is a fixed epoch; weekday arithmetic uses UTC
            import datetime as _dt
            today = _dt.datetime.fromtimestamp(now, _dt.timezone.utc).weekday()
            target = _WEEKDAYS.index(wm.group(1).lower())
            days = (target - today) % 7 or 7
            phrase = wm.group(0)
    if _COPY_RE.search(request):
        out.append({"type": "STR", "value": "Copy of {0}",
                    "desc": "title template: Copy of {0} (fill {0} with the original title)"})
    if _REMIND_RE.search(request):
        out.append({"type": "STR", "value": "Reminder: {0} is due {1}.",
                    "desc": "message template: Reminder: {0} is due {1}. (fill {0} with the card title, {1} with its due date)"})
    if days is not None:
        out.append({"type": "TIME", "value": now + days * 86400,
                    "desc": f"{phrase} (a date)"})
    elif creating:
        out.append({"type": "TIME", "value": now + _DEFAULT_DUE_DAYS * 86400,
                    "desc": "one week from now (default due date)"})
    return out


def constants_from_board(request: str, state: dict, now: int | None = None) -> list:
    """Builds the CONSTANTS list (harness.context.build_context's `constants`
    param shape) from a client-supplied fake board — every user on it (a
    small, cheap set) plus only the cards relevant_cards() thinks the
    request is actually naming, plus the literals a free-typed kanban
    request commonly needs (booleans, workflow statuses, and
    GENERIC_MESSAGES for SEND calls)."""
    constants = []
    for card in relevant_cards(request, state.get("entities", {}).get("card", [])):
        # desc must let the model match how a person actually phrases it
        # ("card 3", the same convention data/curriculum_tasks.jsonl's own
        # constants use — desc: "card 4", not a title) — a title-only desc
        # gave the model no way to resolve "card 3" and it picked the wrong
        # card entirely (caught by testing, not by inspection). Title comes
        # after too, for requests that describe a card by content instead
        # of number.
        num = card["id"].rsplit("_", 1)[-1]
        label = f"card {num}" if num.isdigit() else card["id"]
        constants.append({"type": "ID:card", "value": card["id"],
                           "desc": f'{label} — {card["title"]}'})
    for user in state.get("entities", {}).get("user", []):
        constants.append({"type": "ID:user", "value": user["id"],
                           "desc": user["name"]})
    constants.append({"type": "BOOL", "value": True, "desc": "true"})
    constants.append({"type": "BOOL", "value": False, "desc": "false"})
    for status in ("todo", "doing", "done"):
        constants.append({"type": "STR", "value": status,
                           "desc": f"the {status} status"})
    for msg in GENERIC_MESSAGES:
        constants.append({"type": "STR", "value": msg,
                           "desc": f"message: {msg}"})
    if now is None:
        now = get_world("kanban")["now"]
    constants.extend(literals_from_request(request, now))
    # The writer tool's brief: the request itself, verbatim (plan §A8). Last,
    # so every other index stays stable for authored references.
    constants.append({"type": "STR", "value": request,
                      "desc": "the request itself, verbatim (brief for write_text)"})
    return constants


def list_models() -> dict:
    """GET /models — the checkpoints on this machine, and how each one wants
    to be prompted. `template: "qwen"` means the client builds the prompt
    itself (byte-identical to the browser path); `"chat"` means it sends
    system/user and the server applies the GGUF's own template."""
    models = []
    for path in sorted(MODELS_DIR.glob("*.gguf")):
        # LoRA adapters live here too (run_a.py --lora) and are not loadable
        # as a planner — they'd be offered in the picker and fail on select.
        if "lora" in path.name.lower():
            continue
        tuned = path.name.lower().startswith(TUNED_PREFIXES)
        models.append({"name": path.name,
                       "template": "qwen" if tuned else "chat",
                       "tuned": tuned,
                       "size_mb": round(path.stat().st_size / 1e6)})
    return {"default": PLANNER.model_path.name if PLANNER.model_path else None,
            "loaded": PLANNER.model_path.name if (
                PLANNER.model_path and PLANNER._llm is not None) else None,
            "models": models}


def handle_plan(req: dict) -> dict:
    """POST /plan -> {text, tokens_out, gen_ms, ...}. Either
    {prompt} (client-templated, our tuned checkpoints) or {system, user}
    (server-templated, any other GGUF). {warm: true} just loads the model;
    {model: name} switches checkpoint first."""
    name = req.get("model")
    if name:
        try:
            PLANNER.switch(str(name))
        except ValueError as exc:
            return {"error": {"code": "NO_MODEL", "message": str(exc)}}
    if not PLANNER.available:
        return {"error": {"code": "NO_MODEL", "message": PLANNER.status().get("reason")}}
    if req.get("warm"):
        return PLANNER.warm()
    max_tokens = int(req.get("max_tokens") or 250)
    user = req.get("user")
    if isinstance(user, str) and user:
        return PLANNER.generate_chat(req.get("system") or "", user, max_tokens)
    prompt = req.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        return {"error": {"code": "BAD_REQUEST",
                          "message": "plan needs 'prompt' or 'user'"}}
    stop = req.get("stop") or ["<|im_end|>"]
    return PLANNER.generate(prompt, max_tokens, list(stop))


def handle_write(req: dict) -> dict:
    """POST /write — the sandbox's EXTERNAL callback: {kind, params} ->
    {text}. Only kind 'write_text' exists today."""
    if not PLANNER.available:
        return {"error": {"code": "NO_MODEL", "message": PLANNER.status().get("reason")}}
    if req.get("kind") != "write_text":
        return {"error": {"code": "BAD_REQUEST", "message": f"unknown external kind {req.get('kind')!r}"}}
    params = req.get("params") or []
    brief = str(params[0]) if params else ""
    data = params[1] if len(params) > 1 and isinstance(params[1], list) else []
    return PLANNER.write(brief, data)


def handle_kanban_prompt(req: dict) -> dict:
    """POST /kanban_prompt — builds a fresh TaskContext for the `kanban`
    world from a client-supplied request string and board, using the same
    build_context()/serialize_context() the offline curriculum generator
    uses (harness/context.py), instead of looking up a fixed task_id. This
    is what lets a person type an arbitrary kanban request and still get a
    real, grammar-matched TOOLS/FIELDS/CONSTANTS prompt for it. See
    client/kanban-ui/README.md."""
    request = req.get("request", "")
    state = req.get("state")
    if not request or not isinstance(state, dict):
        return {"error": {"code": "BAD_REQUEST",
                           "message": "kanban_prompt needs 'request' and 'state'"}}
    world = get_world("kanban")
    ctx, _sandbox = build_context(
        world, constants_from_board(request, state, world["now"]))
    return {
        "input_text": serialize_context(request, ctx),
        "context": ctx.to_json(),
        "world": "kanban",
        "now": world["now"],
    }


def handle_rpg_new(req: dict) -> dict:
    """POST /rpg_new {scenario?} -> {state}. The server owns the starting
    dungeon so the client never carries a second copy of the map."""
    scenario = req.get("scenario") or "keep"
    if scenario not in rpg.SCENARIOS:
        return {"error": {"code": "BAD_REQUEST",
                          "message": f"unknown scenario {scenario!r}"}}
    return {"state": rpg.new_state(scenario)}


def handle_rpg_prompt(req: dict) -> dict:
    """POST /rpg_prompt {state} -> the turn's prompt.

    The RPG's analogue of /kanban_prompt: instead of a typed request, the
    request text *is* the rendered observation (rpg.observe), and the
    constants are the things currently in view. `observation` carries the
    structured window so the UI draws fog from the server's perception rule
    rather than a second implementation of it. `state` comes back with
    `memory` updated — the client must thread that copy forward."""
    state = req.get("state")
    if not isinstance(state, dict):
        return {"error": {"code": "BAD_REQUEST", "message": "rpg_prompt needs 'state'"}}
    world = get_world("rpg")
    obs = rpg.observe(state)
    state = dict(state, memory=obs.memory)
    ctx, _sandbox = build_context(world, obs.constants)
    return {
        "input_text": serialize_context(obs.request, ctx),
        "context": ctx.to_json(),
        "world": "rpg",
        "now": world["now"],
        "state": state,
        "observation": {"request": obs.request, "window": obs.window,
                        "nearby": obs.nearby,
                        "outcome": rpg.outcome(state)},
    }


def handle_validate(req: dict) -> dict:
    """Implements POST /validate. See server/README.md for the
    full request/response contract, including the pause_types/pause_envs
    continuation contract."""
    task_id = req.get("task_id")
    text = req.get("text", "")
    incoming_state = req.get("state")
    incoming_registers = req.get("registers")
    pause_types = req.get("pause_types")  # echoed back from a prior response's pause_envs
    incoming_approval = req.get("approval")  # optional per-request override of the default approval token
    # Freeform mode (client/kanban-ui's typed chat): a client that already
    # called POST /kanban_prompt sends that response's "context"/"world"/"now"
    # back here instead of a task_id — same compile/execute path, just a
    # freshly-built context instead of a data/curriculum_tasks.jsonl lookup.
    inline_context = req.get("context")
    inline_world = req.get("world")
    inline_now = req.get("now")

    if task_id is not None:
        task = TASKS.get(task_id)
        if task is None:
            return {"status": "server_error",
                    "error": {"code": "UNKNOWN_TASK", "message": f"no such task_id: {task_id!r}"}}
        ctx = TaskContext.from_json(task["context"])
        world_name = task["world"]
        now = task["now"]
        default_state = task["state"]
        default_approval = task.get("approval", False)
        error_injection = task.get("error_injection", [])
    elif inline_context is not None and inline_world:
        ctx = TaskContext.from_json(inline_context)
        world_name = inline_world
        now = inline_now if inline_now is not None else get_world(inline_world)["now"]
        default_state = None  # freeform requests always carry their own state
        default_approval = False  # no stored default; must come from "approval" below
        error_injection = []
    else:
        return {"status": "server_error",
                "error": {"code": "BAD_REQUEST",
                          "message": "/validate needs either task_id or (context + world)"}}

    registers = incoming_registers or {}
    if incoming_registers and pause_types:
        # Continuation after a PAUSE: re-derive ctx.initial_registers the
        # same way harness/run.py's run_task() does after a pause, using
        # the pause_envs the client echoed back from our previous response.
        ctx.initial_registers = {
            r: parse_type(t) for r, t in pause_types.items() if r in registers}

    world = get_world(world_name)
    sandbox_ctx = sandbox_from_context(ctx, world)

    state = default_state if incoming_state is None else incoming_state
    if state is None:
        return {"status": "server_error",
                "error": {"code": "BAD_REQUEST", "message": "no state: freeform requests must send one"}}

    result = build(text, ctx)
    if not result.compile_ok:
        return {
            "status": "static_error",
            "diagnostics": result.rendered_diagnostics(),
            "final_state": None,
            "registers": None,
            "calls": [],
            "return_value": None,
            "pause_envs": None,
            "error": None,
        }

    payload = {
        "js": result.js,
        "state": state,
        "tools": sandbox_ctx["tools"],
        "fields": sandbox_ctx["fields"],
        "constants": sandbox_ctx["constants"],
        "now": now,
        # A client may deliberately request an unapproved run (to exercise the
        # real EFFECT_BLOCKED gate) by sending "approval": false; omitting the
        # field keeps the previous behavior of trusting the default above.
        "approval": default_approval if incoming_approval is None else bool(incoming_approval),
        "error_injection": error_injection,
        "initial_registers": registers,
    }
    if SELF_URL and PLANNER.available:
        payload["external_url"] = SELF_URL + "/write"  # EXTERNAL tools call back here
    if world.get("post_hook"):
        # worlds with a per-turn phase (the rpg enemy turn) — the sandbox runs
        # it once after the program ends, so the demo and the offline suite
        # advance the game identically
        payload["post_hook"] = world["post_hook"]
    sres = run_sandbox(payload)

    return {
        "status": sres.get("status"),
        "diagnostics": result.rendered_diagnostics(),
        "final_state": sres.get("state"),
        "registers": sres.get("registers"),
        "calls": sres.get("calls", []),
        "return_value": sres.get("return_value"),
        "pause_envs": result.pause_envs if sres.get("status") == "paused" else None,
        "error": sres.get("error"),
        "reason": sres.get("reason"),  # ABORT reason when status == 'aborted'
        "refs": sres.get("refs") or [],  # ABORT referents (spec §4 0.3.0)
    }


def main() -> None:
    global TASKS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=str(DEFAULT_PLAN_MODEL),
                    help="GGUF for POST /plan server-side inference (loaded lazily)")
    ap.add_argument("--ctx", type=int, default=4096, help="n_ctx for /plan")
    ap.add_argument("--writer-model", default=str(DEFAULT_WRITER_MODEL),
                    help="GGUF for POST /write (base weights; falls back to --model if missing)")
    args = ap.parse_args()
    PLANNER.model_path = Path(args.model) if args.model else None
    PLANNER.n_ctx = args.ctx
    PLANNER.writer_path = Path(args.writer_model) if args.writer_model else None
    global SELF_URL
    SELF_URL = f"http://{args.host}:{args.port}"

    TASKS = load_tasks(TASKS_FILE)

    print(f"covenant-agent dev server")
    print(f"  repo root:      {ROOT}")
    print(f"  app (if built): {APP_DIST}")
    print(f"  /rpg (if built):{RPG_DIST}")
    print(f"  /plan model:    {PLANNER.model_path} (available={PLANNER.available})")
    print(f"  models dir:     {MODELS_DIR}")
    print(f"  grammar file:   {GRAMMAR_FILE}")
    print(f"  tasks indexed:  {len(TASKS)} (from {TASKS_FILE})")
    print(f"  listening on:   http://{args.host}:{args.port}")

    httpd = ThreadingHTTPServer((args.host, args.port), DevHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
