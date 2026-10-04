"""Drive the harness loop with the tiny model, so a paused task is scored by
what the continuation does (plan step 4).

`evaluate.py` scores one cached row at a time: an input goes in, a program
comes out, and the sandbox says whether that program reached the goal. A task
that pauses cannot be scored that way, because the registers its second
segment starts from depend on what the first segment actually did when it
ran. So this runs the real loop — `harness/run.py`'s `run_task` — with a
planner that is the model:

  segment 0   encode the context, sample a program, hand it to the harness
  PAUSE       the harness runs it, binds the registers, reseeds the context
  segment 1   re-serialize *with those registers*, encode again, sample again

The re-serialization is the whole point. `harness/context.py` renders each
bound register as its symbol, its type and a bounded view of what it holds,
and `prep.encode_one` turns that into the register region the encoder reads.

  python play.py --ckpt runs/ar_s0/best.pt --cache data_cache_struct \\
      --tasks ../../data/curriculum_tasks.jsonl --level 10
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

COVENANT = Path(__file__).resolve().parents[2]
if str(COVENANT) not in sys.path:
    sys.path.insert(0, str(COVENANT))

from canvas import Layout, TaskCodec, context_symbols, load_keywords
from evaluate import load_model
from prep import encode_one
from sample import ar_backoff, ar_sample, diffusion_sample, to_text

from harness.context import serialize_context  # noqa: E402
from harness.run import run_task  # noqa: E402


class ModelPlanner:
    """A planner in `harness/run.py`'s sense: (request, ctx, seg_idx,
    registers) -> program text. Keeps what it wrote, for the report."""

    def __init__(self, model, tk, keywords, layout, dims, row, device,
                 steps=8, max_segments=4, reader=None, backoff=0):
        self.model, self.tk, self.device = model, tk, device
        # a reader model (train.py --reader) reads a context no cache has
        # seen through live_reader.py; `backoff` is evaluate.py --backoff
        self.reader, self.backoff = reader, backoff
        self.tries: list[int] = []
        self.keywords, self.layout, self.dims = keywords, layout, dims
        self.syms = context_symbols(row["context"])
        self.request = row["request"]
        self.steps, self.max_segments = steps, max_segments
        self.programs: list[str] = []
        self.inputs: list[str] = []

    def __call__(self, request, ctx, seg_idx, registers):
        if seg_idx >= self.max_segments:
            return None
        source = serialize_context(self.request, ctx, registers,
                                   names=bool(self.dims.get("names")))
        self.inputs.append(source)
        inputs = encode_one(source, self.syms, self.tk, self.layout,
                            self.dims, self.device)
        if getattr(self.model.c, "reader", False):
            from live_reader import reader_inputs
            inputs.update(reader_inputs(source, self.dims, self.layout.max_tool,
                                        self.reader, self.device, self.model.c))
        codec = TaskCodec(self.keywords, self.layout, **self.syms)
        with torch.no_grad():
            if self.model.c.causal and self.backoff:
                from core.pipeline import build
                canvas, tr = ar_backoff(self.model, inputs, codec,
                                        lambda text: bool(build(text, ctx).compile_ok),
                                        tries=self.backoff)
                self.tries.append(tr.tries)
            elif self.model.c.causal:
                canvas, _ = ar_sample(self.model, inputs, codec)
            else:
                canvas, _ = diffusion_sample(self.model, inputs, codec,
                                             steps=self.steps)
        text = to_text(canvas, codec)
        self.programs.append(text)
        return text


def main() -> int:
    ap = argparse.ArgumentParser(prog="play")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True,
                    help="the cache this checkpoint was trained on: its "
                         "tokenizer, keyword table and layout")
    ap.add_argument("--tasks", required=True, help="a task JSONL")
    ap.add_argument("--level", type=int, default=None)
    ap.add_argument("--id-contains", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--steps", type=int, default=8,
                    help="diffusion sampler steps; ignored for the ar arm")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="write per-task rows here")
    ap.add_argument("--show", type=int, default=3,
                    help="print this many programs in full")
    ap.add_argument("--reader-live", default=None, metavar="SPEC",
                    help="a --reader checkpoint: run its reader live for contexts "
                         "no cache has: `electra` (electra_reader.py; the reader "
                         "of an electra_cache.py cache), or an embedding head's head.pt")
    ap.add_argument("--backoff", type=int, default=0, metavar="K",
                    help="evaluate.py --backoff: on a compile failure, up to K "
                         "runner-up tools at CALL slots")
    ap.add_argument("--max-const", type=int, default=None, metavar="N",
                    help="widen the constant slots at inference. Constants reach "
                         "the pointer as a set of line vectors, so nothing learned "
                         "depends on the count; the demo's kanban board has 16, "
                         "the S6 caches' layout 10")
    ap.add_argument("--max-field", type=int, default=None, metavar="N",
                    help="widen the field slots at inference, as --max-const "
                         "does: field pointers are this task's own line "
                         "vectors too. The service worlds carry up to 51 "
                         "fields against the clt layout's 28")
    ap.add_argument("--retype", action="store_true",
                    help="re-render rows not written typed under the S6 surface "
                         "(symbols typed, enums, kinds), as run_a.py and the demo "
                         "server do")
    args = ap.parse_args()

    # the generated theme worlds are registered at generation time, not baked
    # into the registry; without this every themed task dies on a missing
    # world and it reads as a model failure (models/tiny/sandbox.py)
    from sandbox import register_themes
    register_themes()

    device = torch.device(args.device)
    cache = Path(args.cache)
    cfg = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    if cfg.get("binding") != "structural":
        raise SystemExit("play.py drives the structural binding only")
    from tokenizers import Tokenizer
    tk = Tokenizer.from_file(str(cache / "in_tok.json"))
    tk.no_truncation()
    tk.no_padding()
    keywords = load_keywords(cache / "keywords.json")
    layout = Layout.from_dict(cfg["layout"])
    dims = {"max_line": cfg["max_line"], "max_req": cfg["max_req"],
            "max_reg": cfg.get("max_reg", 8),
            # a split / named cache re-serializes and re-encodes the same way
            **{k: cfg[k] for k in ("names", "split", "desc_chars", "max_sig",
                                   "max_desc", "max_name", "name_words", "chunk_words", "electra_reader")
               if k in cfg}}
    model = load_model(Path(args.ckpt), device)
    if args.max_const and args.max_const > layout.max_const:
        import dataclasses
        layout = dataclasses.replace(layout, max_const=args.max_const)
        model.c.max_const = args.max_const
    if args.max_field and args.max_field > layout.max_field:
        import dataclasses
        layout = dataclasses.replace(layout, max_field=args.max_field)
        model.c.max_field = args.max_field
    reader = None
    if getattr(model.c, "reader", False):
        if not args.reader_live:
            raise SystemExit("a --reader checkpoint needs --reader-live DIR|FILE")
        from live_reader import check_reader, open_reader
        reader = open_reader(args.reader_live)
        check_reader(reader, args.cache, model.c)

    rows = []
    with open(args.tasks, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            if args.level is not None and row.get("level") != args.level:
                continue
            if args.id_contains and args.id_contains not in row["id"]:
                continue
            if args.retype and row.get("symbols") != "typed":
                from harness.retype import retype_task
                from runtime.worlds import get_world
                row = retype_task(row, get_world(row["world"]), symbols="typed",
                                  enums=True, kinds=True)
            rows.append(row)
            if args.limit and len(rows) >= args.limit:
                break
    if not rows:
        raise SystemExit("no task matched")

    out, stat = [], Counter()
    for row in rows:
        planner = ModelPlanner(model, tk, keywords, layout, dims, row, device,
                               steps=args.steps, reader=reader, backoff=args.backoff)
        try:
            result = run_task(row, planner)
        except Exception as exc:                       # noqa: BLE001
            stat["crashed"] += 1
            out.append({"id": row["id"], "error": repr(exc)})
            continue
        segments_expected = len(row["reference"]["segments"])
        stat["tasks"] += 1
        stat["goal"] += bool(result["goal_success"])
        stat["compiled"] += bool(result["compile_ok"])
        stat["parsed"] += bool(result["parse_ok"])
        if segments_expected > 1:
            stat["paused_tasks"] += 1
            stat["paused_goal"] += bool(result["goal_success"])
            stat["reached_second_segment"] += result["segments"] > 1
        out.append({
            "id": row["id"], "level": row.get("level"),
            "world": row.get("world"),
            "goal_success": result["goal_success"],
            "status": result["status"], "segments": result["segments"],
            "segments_expected": segments_expected,
            "programs": planner.programs, "tries": planner.tries,
            "reference": row["reference"]["segments"],
        })

    n = max(stat["tasks"], 1)
    print(f"\n{stat['tasks']} tasks: goal {stat['goal']}/{n} "
          f"({stat['goal'] / n:.1%}), compile {stat['compiled']}/{n}, "
          f"parse {stat['parsed']}/{n}"
          + (f", crashed {stat['crashed']}" if stat["crashed"] else ""))
    if stat["paused_tasks"]:
        p = stat["paused_tasks"]
        print(f"of the {p} that pause: {stat['reached_second_segment']}/{p} "
              f"got a second segment at all, {stat['paused_goal']}/{p} reached "
              f"the goal")
    for row in out[:args.show]:
        print(f"\n--- {row['id']} ({row.get('status')}, "
              f"goal={row.get('goal_success')})")
        for i, p in enumerate(row.get("programs") or []):
            print(f"  segment {i}:")
            for line in p.strip().splitlines():
                print("    " + line)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for row in out:
                fh.write(json.dumps(row) + "\n")
        print(f"\nwrote {len(out)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
