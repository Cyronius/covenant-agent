"""Load covenant-agent tasks and turn them into (input text, target program) pairs.

The input is exactly what the real planner sees, built by covenant-agent's own
serializer so we never drift from it. The target is the reference program with a
derived EFFECTS header prepended -- see derive_effects for why.
"""
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

def _find_covenant() -> Path:
    """Where the parser, typechecker and serializer live.

    Walk up from this file until a directory holding `harness/context.py` and
    `core/` appears. That finds the covenant-agent checkout when this runs from
    `models/tiny/`, and it finds the pod bundle, where pack.py puts `core/` and
    `harness/` beside `tiny/`. `CANVAS_COVENANT` overrides both.

    Getting this wrong costs pod time immediately, because the tokenizer imports
    this module and so training fails before the first step.
    """
    env = os.environ.get("CANVAS_COVENANT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for cand in here.parents:
        if (cand / "harness" / "context.py").exists() and (cand / "core").is_dir():
            return cand
    raise SystemExit(f"covenant-agent not found above {here}; "
                     "set CANVAS_COVENANT to the checkout or the bundle root")


COVENANT = _find_covenant()
if str(COVENANT) not in sys.path:
    sys.path.insert(0, str(COVENANT))

from harness.context import TaskContext, serialize_context  # noqa: E402

MAX_PROGRAM_TOKENS = 60


@dataclass
class Example:
    task_id: str
    level: int
    world: str
    source: str        # serialized context: the model-facing input
    target: str        # program text including the derived EFFECTS header
    row: dict          # the original task, needed later to score in the sandbox


def derive_effects(program: str, ctx: TaskContext) -> str:
    """Union of the declared effects of every tool the program CALLs.

    The reference programs omit the EFFECTS header, and the real spec makes it
    optional. We add it because it is a dependency that points backwards: the
    first line is determined by lines below it. An autoregressive model has to
    predict it before writing the calls; a diffusion model can fill it last.
    That asymmetry is one of the things this experiment measures, so the header
    has to actually be in the target.
    """
    order = ["READ", "WRITE", "DELETE", "SEND", "PAY", "EXTERNAL"]
    found = set()
    for line in program.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and parts[0] == "CALL":
            tool = ctx.tools.get(parts[1])
            if tool:
                found.update(tool.effects)
    return "EFFECTS " + " ".join(e for e in order if e in found) if found else ""


def program_tokens(program: str) -> list[str]:
    """Whitespace tokens with structure made explicit.

    Two spaces of indent become one IND token and a line break becomes NL, so
    the canvas holds a flat token sequence that still round-trips to text.
    """
    out: list[str] = []
    for line in program.splitlines():
        stripped = line.lstrip(" ")
        depth = (len(line) - len(stripped)) // 2
        if not stripped:
            continue
        out.extend(["IND"] * depth)
        out.extend(stripped.split())
        out.append("NL")
    return out


def detokenize(tokens: list[str]) -> str:
    """Inverse of program_tokens. Unknown or padding tokens are dropped."""
    lines: list[str] = []
    depth = 0
    cur: list[str] = []
    for tok in tokens:
        if tok in ("PAD", "MASK"):
            continue
        if tok == "IND":
            if not cur:
                depth += 1
            continue
        if tok == "NL":
            if cur:
                lines.append("  " * depth + " ".join(cur))
            cur, depth = [], 0
            continue
        cur.append(tok)
    if cur:
        lines.append("  " * depth + " ".join(cur))
    return "\n".join(lines) + "\n" if lines else ""


def load(path: Path, limit: int | None = None,
         max_tokens: int = MAX_PROGRAM_TOKENS) -> list[Example]:
    """Single-segment tasks only: no PAUSE, so the whole program fits one canvas."""
    out: list[Example] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if limit is not None and len(out) >= limit:
                break
            row = json.loads(line)
            segments = row.get("reference", {}).get("segments") or []
            if len(segments) != 1:
                continue
            ctx = TaskContext.from_json(row["context"])
            program = segments[0]
            header = derive_effects(program, ctx)
            target = (header + "\n" + program) if header else program
            if len(program_tokens(target)) > max_tokens:
                continue
            out.append(Example(
                task_id=row["id"],
                level=row.get("level", -1),
                world=row.get("world", "?"),
                source=serialize_context(row["request"], ctx),
                target=target,
                row=row,
            ))
    return out


def split(examples: list[Example], seed: int = 0,
          holdout_world: str | None = None
          ) -> tuple[list[Example], list[Example], list[Example], list[Example]]:
    """train, val, test, holdout-world.

    Held out by world, not by row, so the held-out split measures whether the
    encoder can read an unfamiliar tool's description rather than whether it
    memorized a symbol.
    """
    held = [e for e in examples if holdout_world and e.world == holdout_world]
    rest = [e for e in examples if not (holdout_world and e.world == holdout_world)]
    rng = random.Random(seed)
    rng.shuffle(rest)
    n = len(rest)
    v, t = int(n * 0.90), int(n * 0.95)
    return rest[:v], rest[v:t], rest[t:], held


if __name__ == "__main__":
    src = COVENANT / "data" / "famc_tasks.jsonl"
    ex = load(src, limit=2000)
    print(f"{len(ex)} examples from {src.name}")
    e = ex[0]
    print(f"\n--- {e.task_id} (level {e.level}, {e.world}) ---")
    print(e.source[:400])
    print("--- target ---")
    print(e.target)
    print("--- tokens ---")
    toks = program_tokens(e.target)
    print(toks)
    assert detokenize(toks).strip() == e.target.strip(), "round-trip failed"
    lens = sorted(len(program_tokens(x.target)) for x in ex)
    print(f"\nprogram tokens: p50 {lens[len(lens)//2]} "
          f"p95 {lens[int(len(lens)*0.95)]} max {lens[-1]}")
    srcs = sorted(len(x.source) for x in ex)
    print(f"source chars:   p50 {srcs[len(srcs)//2]} p95 {srcs[int(len(srcs)*0.95)]}")
