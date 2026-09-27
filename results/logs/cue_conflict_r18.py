"""R18: when a tool's types and its text disagree, which does the planner follow?

R17's hypothesis: a planner trained only on signature-unique rows picks a tool
by its types, and S6's decoy rows (where types never decide) teach it not to.
The swapped tools in R15/R17 are same-shape pairs -- one entity's getter or
lister against another's -- that differ in their types and their text.

For every tool decision in the S6 plain exam whose reference tool R has a
same-shape sibling S (same parameter and return kinds, another entity), the
reference program is teacher-forced up to the decision, and the planner's
choice is read twice:

  base   the task as it is
  swap   R and S exchange names and descriptions; symbols, types, fields and
         constants stay where they are

In `swap`, R still has the types the program needs (its ID parameter matches
the constant the program passes, its return type owns the fields the program
reads) but carries S's text, and S carries R's. A planner that decides by type
still picks R; one that decides by text picks S. Further conditions (--conds):

  swap_name   names only swapped
  swap_desc   descriptions only swapped
  blank       both tools' entity words (spot, advertiser) replaced by `item`
              in name and description, so the text no longer says which
              entity; a planner that used those words falls toward the sibling

  cd models/tiny && python ../../results/logs/cue_conflict_r18.py [--n 800]
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path.cwd()
sys.path.insert(0, str(HERE))
from canvas import Layout, TaskCodec, context_symbols, load_keywords  # noqa: E402
from corpus import COVENANT  # noqa: E402
from evaluate import load_model  # noqa: E402
from prep import encode_one  # noqa: E402

sys.path.insert(0, str(COVENANT))
from harness.context import TaskContext, serialize_context  # noqa: E402

MASK_ID = 1
MODELS = [  # label, checkpoint, its cache; "S6 rows" is the share of training rows from S6
    ("A0   S5 rows (no S6)", "runs/pod_s6split/runs/s5g_A0/best.pt", "data_cache_s5g"),
    ("A0   0% S6, seed 0", "runs/pod_s6split/runs/s6off_A0/best.pt", "data_cache_s6off"),
    ("A0   0% S6, seed 1", "runs/pod_s6off2/runs/s6off_A0s1/best.pt", "data_cache_s6off"),
    ("A0   25% S6", "runs/pod_s6frac/runs/s6d25_A0/best.pt", "data_cache_s6d25"),
    ("A0   100% S6 (R13)", "runs/pod_planner/runs/s6_A0/best.pt", "data_cache_s6g"),
    ("SPt  0% S6", "runs/pod_s6off2/runs/s6off_SPt/best.pt", "data_cache_s6off"),
    ("SPt  25% S6", "runs/pod_s6frac/runs/s6d25_SPt/best.pt", "data_cache_s6d25"),
    ("SPt  50% S6", "runs/pod_s6frac/runs/s6d50_SPt/best.pt", "data_cache_s6d50"),
    ("SPt  100% S6 (R13)", "runs/pod_planner/runs/s6_SPt/best.pt", "data_cache_s6g"),
]


def entity(t: dict) -> str | None:
    m = re.search(r"OBJ:(\w+)", t.get("returns") or "")
    if m:
        return m.group(1)
    for p in t.get("params") or []:
        m = re.match(r"ID:(\w+)", p.get("type") or "")
        if m:
            return m.group(1)
    return None


def shape(t: dict) -> str:
    """Parameter and return kinds with every entity name blanked out."""
    blank = lambda s: re.sub(r"(ID|OBJ):\w+", r"\1:_", s or "")  # noqa: E731
    return json.dumps([[blank(p.get("type")) for p in t.get("params") or []],
                       blank(t.get("returns")), sorted(t.get("effects") or [])])


def decisions(row: dict) -> list[tuple[str, str]]:
    """(reference tool, same-shape sibling) for each tool the reference calls
    that has exactly one sibling of the same shape and another entity."""
    tools = {t["sym"]: t for t in row["context"]["tools"]}
    out = []
    for r in dict.fromkeys(re.findall(r"CALL (T\d+)", row["reference"]["segments"][0])):
        sib = [s for s, t in tools.items() if s != r and shape(t) == shape(tools[r])
               and entity(t) and entity(t) != entity(tools[r])]
        if len(sib) == 1:
            out.append((r, sib[0]))
    return out


def blank_words(text: str, words: set[str]) -> str:
    """Every entity word in `words` -> item, in snake, camel and prose alike:
    get_spot_rec, getSpotRecord and "Fetch one ad spot record" all lose
    `spot`. A lowercase match must not follow a letter, so `lot` survives
    in `pilot`; a capitalised one may (camel case)."""
    for w in sorted(words, key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z]){w}(?:s|es)?(?![a-z])", "item", text)
        text = re.sub(rf"{w.capitalize()}(?:s|es)?(?![a-z])", "Item", text)
    return text


def blanked(ctx: dict, a: str, b: str) -> tuple[dict, bool]:
    """Both tools' entity words replaced by `item` in name and description,
    so their text no longer says which entity; types are untouched. Returns
    the context and whether both texts changed."""
    c = copy.deepcopy(ctx)
    ts = [next(t for t in c["tools"] if t["sym"] == s) for s in (a, b)]
    words = {w for t in ts for w in (entity(t) or "").split("_") if len(w) > 2}
    changed = True
    for t in ts:
        before = (t["name"], t["desc"])
        t["name"], t["desc"] = blank_words(t["name"], words), blank_words(t["desc"], words)
        changed &= (t["name"], t["desc"]) != before
    return c, changed


def swapped(ctx: dict, a: str, b: str, keys=("name", "desc")) -> dict:
    c = copy.deepcopy(ctx)
    ta = next(t for t in c["tools"] if t["sym"] == a)
    tb = next(t for t in c["tools"] if t["sym"] == b)
    for k in keys:
        ta[k], tb[k] = tb[k], ta[k]
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="plain-exam tasks to probe")
    ap.add_argument("--json", default="../../results/logs/cue_conflict_r18.json")
    ap.add_argument("--conds", default="base,swap,blank",
                    help="any of base, swap (name and description), swap_name, "
                         "swap_desc, blank; base is always run")
    args = ap.parse_args()
    conds = ["base"] + [c for c in args.conds.split(",") if c != "base"]
    from sandbox import register_themes
    register_themes()
    torch.set_grad_enabled(False)

    rows = pickle.load(open("data_cache_s6off/rows.pkl", "rb"))["holdout"]
    rows = [r for r in rows if "+" not in r["id"] and len(r["reference"]["segments"]) == 1]
    rows = [r for r in rows if decisions(r)][:args.n]
    print(f"{len(rows)} single-segment plain-exam tasks with a same-shape decision")

    by_cache = defaultdict(list)
    for m in MODELS:
        by_cache[m[2]].append(m)
    results = {}
    for cache_name, models in by_cache.items():
        cache = Path(cache_name)
        cfg = json.loads((cache / "config.json").read_text(encoding="utf-8"))
        layout = Layout.from_dict(cfg["layout"])
        keywords = load_keywords(cache / "keywords.json")
        dims = {k: cfg.get(k) for k in ("max_line", "max_req", "desc_chars", "split",
                                         "max_sig", "max_desc", "max_name", "name_words")}
        dims["max_reg"] = cfg.get("max_reg", 8)
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(str(cache / "in_tok.json"))
        tk.no_truncation(); tk.no_padding()
        # the guard: the rebuilt base inputs must be the cache's own tensors
        meta = json.loads((cache / "holdout_meta.json").read_text(encoding="utf-8"))
        idx = {m["task_id"]: i for i, m in enumerate(meta)}
        held = torch.load(cache / "holdout.pt")
        built, checked, mismatched = [], 0, 0
        for row in rows:
            syms = context_symbols(row["context"])
            codec = TaskCodec(keywords, layout, **syms)
            tgt = torch.tensor([codec.encode(row["reference"]["segments"][0], length=cfg["canvas"])])
            canvas = torch.cat([torch.full_like(tgt[:, :1], MASK_ID), tgt[:, :-1]], 1)
            enc = lambda ctx: encode_one(  # noqa: E731
                serialize_context(row["request"], TaskContext.from_json(ctx), names=True),
                syms, tk, layout, dims)
            base = enc(row["context"])
            # data_cache_s5g holds this exam's tasks as `<id>+s6`; its bare
            # ids are S5 tasks that can share a name (R14 "The runs")
            key = row["id"] + "+s6" if row["id"] + "+s6" in idx else row["id"]
            if key in idx and checked < 50:
                checked += 1
                i = idx[key]
                for k in ("tool_tok", "field_tok", "const_tok", "req_tok"):
                    if not torch.equal(base[k].long(), held[k][i:i + 1].long()):
                        mismatched += 1
                        break
            for r, s in decisions(row):
                p = int((tgt[0] == layout.tool0 + syms["tools"].index(r)).nonzero()[0])
                b = {"task": row["id"], "world": row["world"], "slot": p,
                     "r": syms["tools"].index(r), "s": syms["tools"].index(s),
                     "lister": not next(t for t in row["context"]["tools"]
                                        if t["sym"] == r)["params"],
                     "canvas": canvas, "n_tool": len(syms["tools"]), "base": base,
                     "blank_ok": True}
                for cond in conds[1:]:
                    if cond == "blank":
                        bctx, b["blank_ok"] = blanked(row["context"], r, s)
                        b[cond] = enc(bctx)
                    else:
                        keys = {"swap": ("name", "desc"), "swap_name": ("name",),
                                "swap_desc": ("desc",)}[cond]
                        b[cond] = enc(swapped(row["context"], r, s, keys))
                built.append(b)
        print(f"\n{cache_name}: {len(built)} decisions; rebuilt inputs match the cache on "
              f"{checked - mismatched} of {checked} checked tasks")
        if mismatched:
            print("  MISMATCH: the probe does not see what the model was trained on; stopping")
            return 1
        for label, ckpt, _ in models:
            model = load_model(Path(ckpt), torch.device("cpu"))
            rec = defaultdict(list)
            for b in built:
                for cond in conds:
                    logits = model.decode(b[cond], b["canvas"])[0, b["slot"]]
                    tl = logits[layout.tool0:layout.tool0 + b["n_tool"]]
                    pr = torch.softmax(tl.float(), -1)
                    pick = int(tl.argmax())
                    rec[cond + "_R"].append(pick == b["r"])
                    rec[cond + "_S"].append(pick == b["s"])
                    rec[cond + "_pR"].append(float(pr[b["r"]]))
                    if cond == "base" and getattr(model.c, "split", False) and hasattr(model, "last_gate"):
                        rec["gate"].append(model.last_gate[0, b["slot"]].tolist())
                rec["lister"].append(b["lister"])
                rec["blank_ok"].append(b["blank_ok"])
            results[label] = rec

    def pct(xs):
        return f"{100 * sum(xs) / len(xs):5.1f}%" if xs else "   - "
    say = {"base": "the task as it is", "swap": "names and descriptions swapped",
           "swap_name": "names swapped", "swap_desc": "descriptions swapped",
           "blank": "entity words blanked to `item` (decisions where both texts changed)"}
    print("\nShare of decisions where the planner picks R (the tool the program needs; "
          "after a swap, the one with the right types) or S (its sibling; after a swap, "
          "the one with R's text). Gate = SPt's weights on its sig/desc/name scores.")
    n = len(next(iter(results.values()))["base_R"])
    print(f"n = {n} decisions per model")
    for cond in conds:
        print(f"\n{cond}: {say[cond]}")
        for label, rec in results.items():
            ix = [i for i, ok in enumerate(rec["blank_ok"]) if ok or cond != "blank"]
            g = rec.get("gate") if cond == "base" else None
            gs = ("   gate " + " / ".join(f"{sum(x[k] for x in g) / len(g):.2f}" for k in range(3))) if g else ""
            kinds = "   getters R {} / listers R {}".format(
                pct([rec[cond + "_R"][i] for i in ix if not rec["lister"][i]]),
                pct([rec[cond + "_R"][i] for i in ix if rec["lister"][i]]))
            print(f"  {label:22s} R {pct([rec[cond + '_R'][i] for i in ix])}  "
                  f"S {pct([rec[cond + '_S'][i] for i in ix])}  (n={len(ix)}){kinds}{gs}")
    Path(args.json).write_text(json.dumps(
        {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
