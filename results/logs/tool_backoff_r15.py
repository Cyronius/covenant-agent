"""R15: tool backoff. Decode greedily; if the program does not compile, go back
to a CALL slot, put the model's runner-up tool there, and decode greedily from
the slot after it. Candidates are every CALL slot's `--alts` runner-up tools,
tried in order of the model's probability for them, up to `--budget` extra
decodes -- the same budget as `evaluate.py --best-of 4`, which resamples the
whole program at temperature 0.7 instead. The first candidate that compiles
is kept; if none does, the greedy program is.

It exists because R15 found the residual "wrong entity's tool" failure is a
swap between two tools of the same shape (one entity's getter or lister for
another's), and 97% of those programs fail the typechecker. Half of the tools
that fixed it had under 20% probability, which temperature sampling rarely
draws.

Writes evaluate.py-format JSONL, so `evaluate.py --score` and fail_kinds.py
read it unchanged. Needs the sandbox (the compile check), so laptop only.

Kept as the record of how R15/R16's numbers were made. For new runs use
`evaluate.py --generate --backoff 3` (sample.ar_backoff), which writes the
same programs.

  cd models/tiny && python ../../results/logs/tool_backoff_r15.py       --ckpt runs/pod_s6split/runs/s6off_A0/best.pt --cache data_cache_s6off       --gen-out runs/pod_s6split/out/s6off_A0_plain_tb3.jsonl
"""
import argparse, json, pickle, re, sys, time
from pathlib import Path

sys.path.insert(0, ".")
import torch
import torch.nn.functional as F

from evaluate import Split, load_model
from sample import Trace, ar_sample, local_mask, to_text

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--cache", required=True)
ap.add_argument("--split", default="holdout")
ap.add_argument("--id-not-contains", default="+")
ap.add_argument("--id-contains", default=None)
ap.add_argument("--limit", type=int, default=2000)
ap.add_argument("--budget", type=int, default=3)
ap.add_argument("--alts", type=int, default=2, help="runner-up tools considered per CALL slot")
ap.add_argument("--gen-out", required=True)
args = ap.parse_args()

device = torch.device("cpu")
cache = Path(args.cache)
split = Split(cache, args.split, device)
model = load_model(Path(args.ckpt), device)

from core.pipeline import build
from harness.context import TaskContext
from corpus import _replay
by_id = {r["id"]: r for r in pickle.load(open(cache / "rows.pkl", "rb"))[args.split]}


def check(task_id, text):
    base, _, seg = task_id.partition("#s")
    row = by_id.get(base)
    if row is None:
        return True
    ctx = TaskContext.from_json(row["context"])
    if seg and int(seg):
        starts = _replay(row, row["reference"]["segments"])
        if starts is None:
            return True
        ctx = starts[int(seg)][0]
    return bool(build(text, ctx).compile_ok)


@torch.no_grad()
def forced(inputs, ov, prefix, mem):
    """Greedy left to right with slots [0, len(prefix)) forced; mirrors ar_sample."""
    n = model.c.canvas
    canvas = torch.full((1, n), ov.mask, dtype=torch.long)
    out = torch.full((1, n), ov.pad, dtype=torch.long)
    for i in range(n):
        if i < len(prefix):
            tok = prefix[i]
        else:
            row = model.decode(inputs, canvas, mem=mem)[0, i]
            if i > 0:
                row = row.masked_fill(local_mask(out[:, :i + 1], ov)[0, i], float("-inf"))
            tok = int(row.argmax())
        out[0, i] = tok
        if tok == ov.pad:
            break
        if i + 1 < n:
            canvas[0, i + 1] = tok
    return out


@torch.no_grad()
def slot_probs(inputs, ov, ids, i, mem):
    """The model's distribution at slot i given ids[:i], masked as ar_sample masks it."""
    n = model.c.canvas
    canvas = torch.full((1, n), ov.mask, dtype=torch.long)
    out = torch.full((1, n), ov.pad, dtype=torch.long)
    for j in range(i):
        out[0, j] = ids[j]
        canvas[0, j + 1] = ids[j]
    row = model.decode(inputs, canvas, mem=mem)[0, i]
    row = row.masked_fill(local_mask(out[:, :i + 1], ov)[0, i], float("-inf"))
    return F.softmax(row.float(), dim=-1)


which = [i for i in range(len(split))
         if (args.id_contains is None or args.id_contains in split.meta[i]["task_id"])
         and (not args.id_not_contains or args.id_not_contains not in split.meta[i]["task_id"])]
which = which[:args.limit]
out_rows, t0 = [], time.time()
stats = {"greedy_ok": 0, "rescued": 0, "failed": 0}
for k, i in enumerate(which):
    inputs = split.inputs(i, model)
    ov = split.codec(i)
    tid = split.meta[i]["task_id"]
    canvas, tr = ar_sample(model, inputs, ov, trace=Trace())
    tries, rescued_at = 1, None
    if check(tid, to_text(canvas, ov)):
        stats["greedy_ok"] += 1
    else:
        ids = canvas[0].tolist()
        toks = ov.decode(ids)
        mem = model.encode_inputs(inputs)
        cands = []
        for s in range(1, len(toks)):
            if toks[s] == "PAD":
                break
            if toks[s - 1] == "CALL" and re.fullmatch(r"T\d+", toks[s]):
                p = slot_probs(inputs, ov, ids, s, mem)
                p[ids[s]] = 0
                top = p.topk(args.alts)
                for pr, alt in zip(top.values.tolist(), top.indices.tolist()):
                    if pr > 0 and re.fullmatch(r"T\d+", ov.decode([alt])[0]):
                        cands.append((pr, s, alt))
        cands.sort(reverse=True)
        for pr, s, alt in cands[:args.budget]:
            cand = forced(inputs, ov, ids[:s] + [alt], mem)
            tries += 1
            if check(tid, to_text(cand, ov)):
                canvas, rescued_at = cand, (s, round(pr, 3))
                break
        stats["rescued" if rescued_at else "failed"] += 1
    tgt = split.target(i)
    meta = {kk: v for kk, v in split.meta[i].items() if kk != "syms"}
    out_rows.append({**meta, "program": to_text(canvas, ov), "reference": to_text(tgt, ov),
                     "passes": tr.passes, "steps": tr.steps, "repairs": 0, "unmask_step": tr.unmask_step,
                     "canvas": canvas[0].tolist(), "canvas_tokens": ov.decode(canvas[0].tolist()),
                     "reference_tokens": ov.decode(tgt[0].tolist()), "compiled_inline": None,
                     "tries": tries, "rescued_at": rescued_at,
                     "loops": model.c.dec_loops, "dec_layers": model.c.dec_layers})
    if (k + 1) % 100 == 0:
        print(f"  {k+1}/{len(which)} {(time.time()-t0)/(k+1):.2f}s/ex {stats}", flush=True)
dest = Path(args.gen_out)
with open(dest, "w", encoding="utf-8") as fh:
    for r in out_rows:
        fh.write(json.dumps(r) + "\n")
print(f"wrote {len(out_rows)} -> {dest}  {stats}")
