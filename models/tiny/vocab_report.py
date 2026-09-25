"""Grade stage checkpoints on the full exam's readable flip-slot decisions,
split by whether the words that decide each one were ever in training
(.claude/plans/vocab-push.md, "What gets reported").

  python vocab_report.py --cache data_cache_s6g --words train_words.json \
      --stages runs/stages_v1/stages_last.pt runs/stages_v2/stages_last.pt \
      --json runs/vocab_report.json

A decision's deciding words are the words of the right tool's description
that none of its look-alikes' descriptions contain. "Unseen" means at least
one of them never appears in the S6 training corpus (requests, descriptions
and names; --words, made by `--make-words`). The split is fixed by S6, not by
the general-English corpus, so it compares across tokenizers and with R11.

  python vocab_report.py --make-words ../../data/s6_train.jsonl --words train_words.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import torch

from stage_pretrain import build_head, forward, load_part, load_general, sibling_eval
from train import called_tools

WORD = re.compile(r"[a-z]+")


def make_words(corpus: Path, out: Path) -> None:
    seen = set()
    with open(corpus, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            seen.update(WORD.findall(r["request"].lower()))
            for t in r["context"]["tools"]:
                seen.update(WORD.findall(t["desc"].lower()))
                seen.update(WORD.findall((t.get("name") or "").lower().replace("_", " ")))
    out.write_text(json.dumps(sorted(seen)), encoding="utf-8")
    print(f"{len(seen)} training words -> {out}")


def decisions(cache: Path) -> tuple[dict, list]:
    """The exam's decoyed and flip rows, and per readable flip-slot decision
    (row, called column, twin columns, unseen-deciding-word flag later)."""
    meta = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    lay = meta["layout"]
    exam = load_part(cache, "holdout", None, ids=("+decoy", "+flip"))
    pos = called_tools(exam["tgt"], lay["n_kw"], lay["max_tool"])
    out = []
    for r in range(exam["tgt"].size(0)):
        n = int(exam["n_tool"][r])
        grp = exam["sig_group"][r, :n].long()
        for t in pos[r, :n].nonzero().flatten().tolist():
            members = (grp == grp[t]).nonzero().flatten().tolist()
            if len(members) >= 2 and bool(exam["flip_tool"][r, t]):
                out.append((r, t, members))
    return exam, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache")
    ap.add_argument("--stages", nargs="*", default=[])
    ap.add_argument("--words", required=True)
    ap.add_argument("--make-words", metavar="CORPUS")
    ap.add_argument("--json")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.make_words:
        make_words(Path(args.make_words), Path(args.words))
        return
    dev = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    cache = Path(args.cache)
    meta = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    pad = meta["in_pad"]
    seen = set(json.loads(Path(args.words).read_text(encoding="utf-8")))
    exam, decs = decisions(cache)
    rows = pickle.load(open(cache / "rows.pkl", "rb"))["holdout"]
    hmeta = json.loads((cache / "holdout_meta.json").read_text(encoding="utf-8"))
    keep = [i for i, m in enumerate(hmeta) if "+decoy" in m["task_id"] or "+flip" in m["task_id"]]
    teacher = torch.load(cache / "teacher.pt")["table"].float()
    gen = load_general(cache) if (cache / "general.pt").exists() else None

    unseen = []
    for r, t, members in decs:
        tools = sorted(rows[keep[r]]["context"]["tools"], key=lambda x: int(x["sym"][1:]))
        gold = set(WORD.findall(tools[t]["desc"].lower()))
        others = set()
        for m in members:
            if m != t:
                others |= set(WORD.findall(tools[m]["desc"].lower()))
        unseen.append(any(w not in seen for w in gold - others))

    def tally(picks):
        res = {}
        for name, sel in (("all", lambda u: True), ("seen", lambda u: not u),
                          ("unseen", lambda u: u)):
            hits = [p == t for p, (r, t, m), u in zip(picks, decs, unseen) if sel(u)]
            res[name] = {"acc": sum(hits) / max(len(hits), 1), "n": len(hits)}
        return res

    t_picks = []
    for r, t, members in decs:
        q = teacher[int(exam["t_req"][r])]
        sc = teacher[exam["t_desc"][r, members].long()] @ q
        t_picks.append(members[int(sc.argmax())])
    report = {"cache": str(cache), "teacher": tally(t_picks), "stages": {}}

    for path in args.stages:
        ck = torch.load(path, map_location="cpu")
        head = build_head(ck["cfg"], meta["in_vocab"], pad)
        head.load_state_dict(ck["head"])
        head.to(dev).eval()
        picks, bs = [], 64
        by_row = {}
        for k, (r, t, members) in enumerate(decs):
            by_row.setdefault(r, []).append(k)
        rows_idx = sorted(by_row)
        picks = [None] * len(decs)
        with torch.no_grad():
            for i in range(0, len(rows_idx), bs):
                rr = torch.tensor(rows_idx[i:i + bs])
                b = {k: v[rr].to(dev) for k, v in exam.items()}
                comb = forward(head, b, pad)[0].cpu()
                for j, r in enumerate(rows_idx[i:i + bs]):
                    for k in by_row[r]:
                        members = decs[k][2]
                        picks[k] = members[int(comb[j, members].argmax())]
        res = tally(picks)
        if gen is not None:
            res["dictionary_heldout"] = sibling_eval(head, gen, pad, dev)
        report["stages"][path] = res
        print(f"{path}: all {res['all']['acc']:.1%}  seen {res['seen']['acc']:.1%}  "
              f"unseen {res['unseen']['acc']:.1%}"
              + (f"  dictionary {res['dictionary_heldout']['acc_sib']:.1%}" if gen else ""),
              flush=True)
    T = report["teacher"]
    print(f"teacher: all {T['all']['acc']:.1%}  seen {T['seen']['acc']:.1%}  "
          f"unseen {T['unseen']['acc']:.1%}  (n {T['all']['n']}: {T['seen']['n']} seen, "
          f"{T['unseen']['n']} unseen)", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
