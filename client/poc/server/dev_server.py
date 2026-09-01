"""Dev-only local server for the browser-inference stand-in (client/poc).

Two jobs:
  1. Static file server for client/poc/** (index.html, src/*.js, vendor/*
     wasm, fixtures/*), plus Range-request support for the large GGUF model
     file served straight out of baselines/qwen/models/ (never copied), and
     the grammar file at baselines/qwen/agent_core.gbnf.
  2. POST /validate: takes generated Agent Core program text from the
     in-browser model and round-trips it through the *existing* Python
     pipeline (core.pipeline.build) and sandbox executor
     (harness.run.run_sandbox) — the same path harness/run.py's run_task()
     uses for the reference/model planners. This is a dev shortcut per
     .claude/plans/browser-inference-standin.md scope item 4(a); it is not
     a reimplementation of parse/typecheck/compile/execute.

Pure Python stdlib only — no Flask/FastAPI, none are installed and none are
needed for this POC.

Run:
    python client/poc/server/dev_server.py --port 8080

See client/poc/server/README.md for the full /validate request/response
contract and a worked curl example.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import re
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# client/poc/server/dev_server.py -> repo root is three levels up.
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.ir import TaskContext, parse_type  # noqa: E402
from core.pipeline import build  # noqa: E402
from harness.context import build_context, sandbox_from_context, serialize_context  # noqa: E402
from harness.run import run_sandbox  # noqa: E402
from runtime.worlds import get_world  # noqa: E402

CLIENT_POC = ROOT / "client" / "poc"
MODELS_DIR = ROOT / "baselines" / "qwen" / "models"
GRAMMAR_FILE = ROOT / "baselines" / "qwen" / "agent_core.gbnf"
TASKS_FILE = ROOT / "data" / "curriculum_tasks.jsonl"

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
        """Map a URL path to a file under client/poc, guarding traversal."""
        rel = url_path.lstrip("/")
        if rel == "" or rel == "/":
            rel = "index.html"
        candidate = (CLIENT_POC / rel).resolve()
        try:
            candidate.relative_to(CLIENT_POC.resolve())
        except ValueError:
            return None
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

            static_path = self._resolve_static_path(path)
            if static_path is None:
                self._send_text(403, "Forbidden\n")
                return
            self._serve_file(static_path)
        except Exception:
            self._send_text(500, f"Internal error:\n{traceback.format_exc()}\n")

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path not in ("/validate", "/kanban_prompt"):
            self._send_text(404, "Not found\n")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            req = json.loads(raw.decode("utf-8"))
            resp = handle_kanban_prompt(req) if path == "/kanban_prompt" else handle_validate(req)
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
        # silently falls back to a single thread. See client/poc/src/llm.md
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


def constants_from_board(request: str, state: dict) -> list:
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
    return constants


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
    ctx, _sandbox = build_context(world, constants_from_board(request, state))
    return {
        "input_text": serialize_context(request, ctx),
        "context": ctx.to_json(),
        "world": "kanban",
        "now": world["now"],
    }


def handle_validate(req: dict) -> dict:
    """Implements POST /validate. See client/poc/server/README.md for the
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
    }


def main() -> None:
    global TASKS
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    TASKS = load_tasks(TASKS_FILE)

    print(f"covenant-agent dev server")
    print(f"  repo root:      {ROOT}")
    print(f"  static root:    {CLIENT_POC}")
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
