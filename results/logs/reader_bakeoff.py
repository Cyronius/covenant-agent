"""Reader bake-off (.claude/plans/npu-planner.md, phase 1): can a small
pretrained encoder, zero-shot, tell the right tool from its rivals on R20's
lookalike and decoy decisions?

  python results/logs/reader_bakeoff.py build   # decisions + texts -> $BAKEOFF_DIR
  node   results/logs/reader_bakeoff_embed.js   # ternlight vectors (needs TERNLIGHT_DIR)
  python results/logs/reader_bakeoff.py embed-hf sentence-transformers/all-MiniLM-L6-v2 ...
  python results/logs/reader_bakeoff.py score   # -> results/logs/reader_bakeoff/report.md

A decision is one tool the reference calls, in one of R20's classes
(llm_baseline.classify), with its rivals present in the task: same shape and
another thing for a lookalike, same signature for a decoy. A reader picks the
candidate whose text is closest to the request. Candidate text is the tool's
description, or its name and description ("+name").

The generators are scored on the same decisions from their programs, as the
share right among the times they committed to one of the candidates
(called the right tool and no rival, or a rival and not the right one), and
how often they committed at all. `BAKEOFF_DIR` (default: the scratchpad)
holds the vectors, which are too big for the repo.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import llm_baseline as L  # noqa: E402

WORK = Path(os.environ.get("BAKEOFF_DIR", HERE / "reader_bakeoff" / "work"))
OUT = HERE / "reader_bakeoff"
CLASSES = ("lookalike", "decoy, decidable", "decoy, undecidable", "decoy, unlabelled")
GENERATORS = {"27B, direct": "27B, direct (Cerebras)",
              "27B, reasons first": "27B, reasons first (Cerebras)",
              "A0 options-off": "A0 options-off",
              "SPt options-off": "SPt options-off",
              "SPt 50% S6": "SPt 50% S6",
              "A0 25% decoys+twin": "A0 25% decoys+twin (R19)",
              "A0 25% flip decoys": "A0 25% flip decoys",
              "A0+reader options-off": "A0+reader options-off",
              "A0+reader 25% decoys+twin": "A0+reader 25% decoys+twin",
              "A0+reader 25% flip decoys": "A0+reader 25% flip decoys"}


def rivals(tools: dict, r: str, k: str) -> list[str]:
    t = tools[r]
    if k.startswith("decoy"):
        return [s for s, x in tools.items() if s != r and L.entity(t) and L.signature(x) == L.signature(t)]
    return [s for s, x in tools.items() if s != r and L.shape(x) == L.shape(t)
            and L.entity(x) and L.entity(x) != L.entity(t)]


def build() -> None:
    ids = json.loads((L.OUT / "ids.json").read_text())
    decisions, texts = [], {}
    for exam, fn in L.EXAMS.items():
        want = set(ids[exam])
        for row in map(json.loads, open(fn)):
            if row["id"] not in want:
                continue
            tools = {t["sym"]: t for t in row["context"]["tools"]}
            ref_seq = set(re.findall(r"CALL (T\d+)", "\n".join(row["reference"]["segments"])))
            for r, k in L.classify(row):
                if k == "unique":
                    continue
                rv = [s for s in rivals(tools, r, k) if s not in ref_seq]
                if not rv:
                    continue
                cands = [r] + rv
                decisions.append({"exam": exam, "task": row["id"], "class": k, "right": r,
                                  "cands": cands, "request": row["request"],
                                  "desc": {s: tools[s].get("desc") or "" for s in cands},
                                  "name": {s: tools[s].get("name") or "" for s in cands}})
                texts[row["request"]] = 1
                for s in cands:
                    d, n = tools[s].get("desc") or "", tools[s].get("name") or ""
                    texts[d] = 1
                    texts[f"{n}: {d}"] = 1
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "decisions.json").write_text(json.dumps(decisions))
    (WORK / "texts.json").write_text(json.dumps(list(texts)))
    by = defaultdict(int)
    for d in decisions:
        by[(d["exam"], d["class"])] += 1
    print(f"{len(decisions)} decisions, {len(texts)} distinct texts -> {WORK}")
    for k, v in sorted(by.items()):
        print(f"   {k[0]:6s} {k[1]:20s} {v}")


def embed_hf(names: list[str]) -> None:
    """Full-precision reference readers through transformers, each pooled the
    way its model card says (bge: the first token; MiniLM and the rest: the
    mean over tokens), normalised."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    texts = json.loads((WORK / "texts.json").read_text())
    for name in names:
        tok, m = AutoTokenizer.from_pretrained(name), AutoModel.from_pretrained(name).eval()
        rows = []
        with torch.no_grad():
            for i in range(0, len(texts), 128):
                b = tok(texts[i:i + 128], padding=True, truncation=True, max_length=128, return_tensors="pt")
                h = m(**b).last_hidden_state
                if "bge" in name:
                    v = h[:, 0]
                else:
                    mask = b["attention_mask"].unsqueeze(-1).float()
                    v = (h * mask).sum(1) / mask.sum(1)
                rows.append(torch.nn.functional.normalize(v, dim=-1).numpy())
        tag = name.split("/")[-1]
        np.save(WORK / f"vec_{tag}.npy", np.concatenate(rows).astype(np.float32))
        n = sum(p.numel() for p in m.parameters())
        (WORK / f"vec_{tag}.meta.json").write_text(json.dumps({"params": n}))
        print(f"{tag}: {len(texts)} texts, {n / 1e6:.1f}M params")


def generator_choices(label: str) -> dict:
    """(exam, task, right) -> 'right' | 'rival' | None for one generator's programs."""
    out = {}
    for exam, path in L.SOURCES[label].items():
        probe = path if path.suffix == ".jsonl" else Path(f"{path}.jsonl")
        if not probe.exists():
            continue
        res = L.load_llm(path) if path.suffix == ".jsonl" else L.load_tiny(path)
        out[exam] = {L.base_id(t): set(re.findall(r"CALL (T\d+)", v["program"])) for t, v in res.items()}
    return out


def score() -> None:
    decisions = json.loads((WORK / "decisions.json").read_text())
    texts = json.loads((WORK / "texts.json").read_text())
    at = {t: i for i, t in enumerate(texts)}
    readers = sorted(p.stem[4:] for p in WORK.glob("vec_*.npy"))
    lines = ["| reader or model | params | " + " | ".join(CLASSES) + " |",
             "|---|---|" + "---|" * len(CLASSES)]
    chance = {c: [] for c in CLASSES}
    for d in decisions:
        chance[d["class"]].append(1 / len(d["cands"]))
    lines.append("| chance | — | " + " | ".join(
        f"{sum(v) / len(v):.1%}" if v else "—" for v in chance.values()) + " |")
    for rd in readers:
        vec = np.load(WORK / f"vec_{rd}.npy")
        meta = WORK / f"vec_{rd}.meta.json"
        params = f"{json.loads(meta.read_text())['params'] / 1e6:.1f}M" if meta.exists() else "?"
        for with_name in (False, True):
            acc = {c: [0, 0] for c in CLASSES}
            for d in decisions:
                q = vec[at[d["request"]]]
                key = (lambda s: f"{d['name'][s]}: {d['desc'][s]}") if with_name else (lambda s: d["desc"][s])
                sims = [float(q @ vec[at[key(s)]]) for s in d["cands"]]
                acc[d["class"]][0] += int(np.argmax(sims) == 0)
                acc[d["class"]][1] += 1
            lines.append(f"| {rd}{' +name' if with_name else ''} | {params} | " + " | ".join(
                f"{a / b:.1%} (n={b})" if b else "—" for a, b in acc.values()) + " |")
    for label, src_label in GENERATORS.items():
        ch = generator_choices(src_label)
        acc = {c: [0, 0, 0] for c in CLASSES}   # right, committed, all
        for d in decisions:
            called = ch.get(d["exam"], {}).get(d["task"])
            if called is None:
                continue
            right = d["right"] in called
            rival = any(s in called for s in d["cands"][1:])
            acc[d["class"]][2] += 1
            if right != rival:
                acc[d["class"]][1] += 1
                acc[d["class"]][0] += right
        lines.append(f"| {label} (generator) | — | " + " | ".join(
            f"{a / b:.1%} (committed {b}/{n})" if b else "—" for a, b, n in acc.values()) + " |")
    text = "\n".join(lines)
    OUT.mkdir(exist_ok=True)
    (OUT / "report.md").write_text(
        "# Reader bake-off: right tool among its rivals, zero-shot\n\n"
        "Chance is 1 / (number of candidates): about 40-50% here.\n\n" + text + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "embed-hf":
        embed_hf(sys.argv[2:])
    elif cmd == "score":
        score()
