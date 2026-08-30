"""R1 scorer: strong-model-authored programs through the F3 harness.

Authored rows (JSONL) pair a task id with the author's Agent Core segments
and an idiomatic-JS equivalent:

  {"id": "...", "segments": ["CALL T0 ...", ...], "js": "async ..."}
  {"id": "...", "ac": "CALL T0 ..."}            # single-segment shorthand

Scoring runs each authored program with harness.run.run_task (the source of
truth for goal_success). A task that PAUSEs past its authored segments is
not failed: it is emitted to --pending as a continuation-authoring request
(register snapshot + prior segments) and excluded from final rates until a
merged authored file supplies the next segment.

Token counts use GPT-2 BPE (tiktoken) so the AC-vs-JS ratio is measured
with a standard subword tokenizer that is, if anything, unfavorable to
Agent Core symbols; whitespace counts are logged alongside.

CLI:
  python -m harness.r1 --tasks data/r1_tasks.jsonl \
      --authored data/r1_authored.jsonl \
      --out results/r1_metrics.jsonl --pending data/r1_pending.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness.run import load_tasks, run_task  # noqa: E402

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("gpt2")

    def bpe_tokens(text: str) -> int:
        return len(_ENC.encode(text, disallowed_special=()))
except ImportError:  # pragma: no cover
    _ENC = None

    def bpe_tokens(text: str) -> int:
        return -1


def load_authored(paths: List[Path]) -> Dict[str, dict]:
    """Later files override/extend earlier ones; continuation passes append
    segments to the same id."""
    authored: Dict[str, dict] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                segs = row.get("segments") or [row["ac"]]
                entry = authored.setdefault(
                    row["id"], {"segments": [], "js": None})
                if row.get("continuation"):
                    entry["segments"] += segs
                else:
                    entry["segments"] = list(segs)
                if row.get("js"):
                    entry["js"] = row["js"]
    return authored


def authored_planner(segments: List[str], pending_sink: list):
    def plan(request, ctx, seg_idx, registers):
        if seg_idx < len(segments):
            return segments[seg_idx]
        # only reached when the previous segment PAUSEd and the author has
        # not yet supplied a continuation
        pending_sink.append({"seg_idx": seg_idx, "registers": registers})
        return None

    return plan


def score(tasks: list, authored: Dict[str, dict]):
    rows, pending, unauthored = [], [], []
    for task in tasks:
        entry = authored.get(task["id"])
        if entry is None or not entry["segments"]:
            unauthored.append(task["id"])
            continue
        sink: list = []
        row = run_task(task, authored_planner(entry["segments"], sink))
        ac_text = "\n".join(entry["segments"])
        js_text = entry.get("js") or ""
        row["ac_tokens"] = bpe_tokens(ac_text)
        row["js_tokens"] = bpe_tokens(js_text) if js_text else None
        row["ac_tokens_ws"] = len(ac_text.split())
        row["js_tokens_ws"] = len(js_text.split()) if js_text else None
        row["token_ratio"] = (row["ac_tokens"] / row["js_tokens"]
                              if js_text and row["ac_tokens"] >= 0
                              and row["js_tokens"] else None)
        row["n_segments_authored"] = len(entry["segments"])
        row["needs_continuation"] = bool(sink)
        rows.append(row)
        if sink:
            pending.append({
                "id": task["id"],
                "request": task["request"],
                "input_text": task["input_text"],
                "prior_segments": entry["segments"],
                "seg_idx": sink[0]["seg_idx"],
                "registers": sink[0]["registers"],
            })
    return rows, pending, unauthored


def pct(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


def summarize(rows: list, pending: list, unauthored: list) -> dict:
    done = [r for r in rows if not r["needs_continuation"]]
    ratios = [r["token_ratio"] for r in done
              if r["token_ratio"] and r["compile_ok"]]
    by_level: Dict[int, dict] = {}
    for r in done:
        b = by_level.setdefault(r["level"], {"n": 0, "compile": 0, "goal": 0})
        b["n"] += 1
        b["compile"] += r["compile_ok"]
        b["goal"] += r["goal_success"]
    summary = {
        "n_tasks_scored": len(rows),
        "n_complete": len(done),
        "n_pending_continuation": len(pending),
        "n_unauthored": len(unauthored),
        "compile_ok_rate": (sum(r["compile_ok"] for r in done) / len(done)
                            if done else None),
        "goal_success_rate": (sum(r["goal_success"] for r in done) / len(done)
                              if done else None),
        "token_ratio_median": statistics.median(ratios) if ratios else None,
        "token_ratio_mean": statistics.fmean(ratios) if ratios else None,
        "token_ratio_n": len(ratios),
        "ac_tokens_p50": pct([r["ac_tokens"] for r in done
                              if r["ac_tokens"] >= 0], 0.5),
        "ac_tokens_p95": pct([r["ac_tokens"] for r in done
                              if r["ac_tokens"] >= 0], 0.95),
        "by_level": {k: by_level[k] for k in sorted(by_level)},
        "tokenizer": "gpt2-bpe" if _ENC else "MISSING-tiktoken",
    }
    return summary


def main():
    ap = argparse.ArgumentParser(prog="harness.r1")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--authored", required=True, nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pending", default=None)
    args = ap.parse_args()

    tasks = load_tasks(Path(args.tasks))
    authored = load_authored([Path(p) for p in args.authored])
    rows, pending, unauthored = score(tasks, authored)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    if args.pending:
        with open(args.pending, "w") as f:
            for p in pending:
                f.write(json.dumps(p) + "\n")

    summary = summarize(rows, pending, unauthored)
    print(json.dumps(summary, indent=2))
    if unauthored:
        print(f"unauthored ({len(unauthored)}): {unauthored[:10]} ...",
              file=sys.stderr)


if __name__ == "__main__":
    main()
