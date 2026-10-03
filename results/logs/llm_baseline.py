"""A real LLM against the tiny planners on the same S6 exam tasks
(.claude/plans/llm-baseline.md).

  python results/logs/llm_baseline.py draw     # -> results/logs/llm_baseline/ids.json
  python results/logs/llm_baseline.py report   # every model in SOURCES that has output

Two measures, the same for every model:

- **goal**: the task's final state (the harness's verdict for an LLM run;
  every segment's `evaluate.py --score` verdict for a tiny planner).
- **tool choice**: for each tool the reference calls, did the program call
  it at all? Grammar-free, so an uncompiled LLM program still counts. Each
  reference call is put in one class:
    decoy, decidable    it has a decoy in the row and the request says which
    decoy, undecidable  it has a decoy and the request cannot say which
                        (results/logs/decoy_decidable.py)
    decoy, unlabelled   a same-signature tool of the same thing is in the
                        task, but the row's names do not match its theme
                        (meaningless-name rows, worlds without a theme)
    lookalike           a same-shape tool of another thing is in the task
                        (R18's pairs; cue_conflict_r18.shape/entity)
    unique              neither
  A wrong pick is the program's call at the same position, sorted into its
  decoy / the lookalike / a generic helper / another tool / no call there.
- **swapped**: at lookalike and decoy calls, the share where the program
  called a same-shape tool of another thing (lookalike) or a same-signature
  sibling (decoy) *and not* the reference's tool: R15's wrong-entity error.
  Unlike "called", it does not count a different but valid program shape
  (listing and filtering instead of fetching by id) as a wrong choice, which
  matters for a model never trained on the reference's program style.
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import decoy_decidable  # noqa: E402

OUT = HERE / "llm_baseline"
TINY = ROOT / "models/tiny/runs"
EXAMS = {"plain": ROOT / "data/s6_holdout.jsonl",
         "decoy": ROOT / "data/s6_holdout_decoy.jsonl",
         "flip": ROOT / "data/s6_flip.jsonl"}
# label -> {exam: output}; an LLM output is run_a's .jsonl, a tiny one is the
# gen-out stem (its .jsonl programs and .score.json verdicts)
SOURCES = {
    "27B, direct (Cerebras)": {e: OUT / f"qwen27b_k0_{e}.jsonl" for e in EXAMS},
    "27B, reasons first (Cerebras)": {e: OUT / f"qwen27b_think_{e}.jsonl" for e in ("plain", "decoy")},
    "A0 options-off": {"plain": TINY / "pod_s6split/out/s6off_A0_plain",
                       "decoy": TINY / "pod_s6split/out/s6off_A0_decoy",
                       "flip": TINY / "pod_s6split/out/s6off_A0_flip"},
    "A0 options-off + backoff": {"plain": TINY / "pod_s6split/out/s6off_A0_plain_bk3"},
    "SPt options-off": {"plain": TINY / "pod_s6off2/out/s6off_SPt_plain",
                        "decoy": TINY / "pod_s6off2/out/s6off_SPt_decoy",
                        "flip": TINY / "pod_s6off2/out/s6off_SPt_flip"},
    "SPt 25% S6": {"plain": TINY / "pod_s6frac/out/s6d25_SPt_plain",
                   "decoy": TINY / "pod_s6frac/out/s6d25_SPt_decoy",
                   "flip": TINY / "pod_s6frac/out/s6d25_SPt_flip"},
    "SPt 50% S6": {"plain": TINY / "pod_s6frac/out/s6d50_SPt_plain",
                   "decoy": TINY / "pod_s6frac/out/s6d50_SPt_decoy",
                   "flip": TINY / "pod_s6frac/out/s6d50_SPt_flip"},
    "A0 25% decoys+twin (R19)": {e: TINY / f"pod_s6opt/out/dtw25_A0_{e}" for e in EXAMS},
    # npu-planner.md phase 2: A0 with a frozen Ternlight-mini reader
    "A0 25% flip decoys": {e: TINY / f"pod_rd/out/fdc25_A0_{e}" for e in EXAMS},
    "A0+reader options-off": {e: TINY / f"pod_rd/out/s6off_RD_{e}" for e in EXAMS},
    "A0+reader options-off + backoff": {"plain": TINY / "pod_rd/out/s6off_RD_plain_bk3"},
    "A0+reader 25% decoys+twin": {e: TINY / f"pod_rd/out/dtw25_RD_{e}" for e in EXAMS},
    "A0+reader 25% flip decoys": {e: TINY / f"pod_rd/out/fdc25_RD_{e}" for e in EXAMS},
    "A0+reader 25% flip decoys + backoff": {"plain": TINY / "pod_rd/out/fdc25_RD_plain_bk3"},
}
CLASSES = ("unique", "lookalike", "decoy, decidable", "decoy, undecidable", "decoy, unlabelled")


def entity(t: dict) -> str | None:   # as cue_conflict_r18.entity
    m = re.search(r"OBJ:(\w+)", t.get("returns") or "")
    if m:
        return m.group(1)
    for p in t.get("params") or []:
        m = re.match(r"ID:(\w+)", p.get("type") or "")
        if m:
            return m.group(1)
    return None


def shape(t: dict) -> str:   # as cue_conflict_r18.shape
    blank = lambda s: re.sub(r"(ID|OBJ):\w+", r"\1:_", s or "")  # noqa: E731
    return json.dumps([[blank(p.get("type")) for p in t.get("params") or []],
                       blank(t.get("returns")), sorted(t.get("effects") or [])])


def signature(t: dict) -> str:
    return json.dumps([[p.get("type") for p in t.get("params") or []],
                       t.get("returns"), sorted(t.get("effects") or [])])


def base_id(tid: str) -> str:
    return tid.split("#")[0].replace("+decoy", "")


def draw() -> None:
    plain = [json.loads(line)["task_id"] for line in open(TINY / "pod_s6split/out/s6off_A0_plain.jsonl")]
    plain = list(dict.fromkeys(base_id(t) for t in plain))
    flip = [json.loads(line)["id"] for line in open(EXAMS["flip"])]
    rng = random.Random(20260928)
    p = sorted(rng.sample(plain, 300))
    OUT.mkdir(exist_ok=True)
    (OUT / "ids.json").write_text(json.dumps(
        {"plain": p, "decoy": p, "flip": sorted(rng.sample(flip, 300))}, indent=1))
    print(f"{len(plain)} plain/decoy tasks, {len(flip)} flip -> 300 each, {OUT / 'ids.json'}")


def load_llm(path: Path) -> dict:
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["task_id"]] = {"goal": bool(r["goal_success"]), "compile": bool(r["compile_ok"]),
                             "program": "\n".join(r.get("programs") or [])}
    return out


def load_tiny(stem: Path) -> dict:
    segs = defaultdict(list)
    for line in open(f"{stem}.jsonl"):
        r = json.loads(line)
        segs[base_id(r["task_id"])].append((r["task_id"], r["program"]))
    verdict = defaultdict(list)
    for r in json.load(open(f"{stem}.score.json"))["rows"]:
        verdict[base_id(r["task_id"])].append((r["goal"], r["compile"]))
    return {t: {"goal": all(g for g, _ in verdict[t]), "compile": all(c for _, c in verdict[t]),
                "program": "\n".join(p for _, p in sorted(s))}
            for t, s in segs.items()}


def classify(row: dict) -> list[tuple[str, str]]:
    """(reference tool sym, class) for each distinct tool the reference calls."""
    tools = {t["sym"]: t for t in row["context"]["tools"]}
    name_of = {t["sym"]: t.get("name") for t in row["context"]["tools"]}
    dec = {name: ok for name, _, ok in decoy_decidable.calls(row)}
    out = []
    for r in dict.fromkeys(re.findall(r"CALL (T\d+)", "\n".join(row["reference"]["segments"]))):
        if r not in tools:
            continue
        if name_of[r] in dec:
            k = "decoy, decidable" if dec[name_of[r]] else "decoy, undecidable"
        elif entity(tools[r]) and any(s != r and signature(t) == signature(tools[r])
                                      for s, t in tools.items()):
            k = "decoy, unlabelled"
        elif any(s != r and shape(t) == shape(tools[r]) and entity(t) and entity(t) != entity(tools[r])
                 for s, t in tools.items()):
            k = "lookalike"
        else:
            k = "unique"
        out.append((r, k))
    return out


def wrong_kind(row: dict, ref: str, pick: str | None) -> str:
    tools = {t["sym"]: t for t in row["context"]["tools"]}
    if pick is None or pick not in tools:
        return "no call there"
    a, b = tools[ref], tools[pick]
    if signature(a) == signature(b):
        return "its decoy"
    if shape(a) == shape(b) and entity(b) and entity(b) != entity(a):
        return "the lookalike"
    if entity(b) is None:
        return "a generic helper"
    return "another tool"


def report() -> None:
    ids = json.loads((OUT / "ids.json").read_text())
    exam_rows = {}
    for e, fn in EXAMS.items():
        want = set(ids[e])
        exam_rows[e] = {r["id"]: r for r in map(json.loads, open(fn)) if r["id"] in want}
    lines = []
    for exam in EXAMS:
        lines.append(f"\n## {exam} exam ({len(ids[exam])} tasks)\n")
        lines.append("| model | n | goal | compiles | " + " | ".join(f"called: {c}" for c in CLASSES)
                     + " | " + " | ".join(f"swapped: {c}" for c in CLASSES[1:])
                     + " | wrong picks at lookalike and decoy calls |")
        lines.append("|---|---|---|---|" + "---|" * (2 * len(CLASSES) - 1) + "---|")
        for label, src in SOURCES.items():
            path = src.get(exam)
            if path is None:
                continue
            probe = path if path.suffix == ".jsonl" else Path(f"{path}.jsonl")
            if not probe.exists():
                continue
            res = load_llm(path) if path.suffix == ".jsonl" else load_tiny(path)
            res = {base_id(t): v for t, v in res.items()}
            got = [t for t in ids[exam] if t in res]
            acc = {c: [0, 0] for c in CLASSES}
            swaps = {c: [0, 0] for c in CLASSES[1:]}
            wrong = Counter()
            for t in got:
                row = exam_rows[exam][t]
                tools = {x["sym"]: x for x in row["context"]["tools"]}
                called = re.findall(r"CALL (T\d+)", res[t]["program"])
                ref_seq = re.findall(r"CALL (T\d+)", "\n".join(row["reference"]["segments"]))
                for r, k in classify(row):
                    ok = r in called
                    acc[k][0] += ok
                    acc[k][1] += 1
                    if k != "unique":
                        # a rival the reference calls for its own sake is no swap
                        rival = [x for x in set(called) - set(ref_seq) if x in tools and (
                            signature(tools[x]) == signature(tools[r]) if k.startswith("decoy")
                            else shape(tools[x]) == shape(tools[r]) and entity(tools[x])
                            and entity(tools[x]) != entity(tools[r]))]
                        swaps[k][0] += bool(rival) and not ok
                        swaps[k][1] += 1
                    if not ok and k != "unique":
                        i = ref_seq.index(r)
                        wrong[wrong_kind(row, r, called[i] if i < len(called) else None)] += 1
            n = len(got)
            g = sum(res[t]["goal"] for t in got) / max(n, 1)
            c = sum(res[t]["compile"] for t in got) / max(n, 1)
            w = sum(wrong.values())
            lines.append(f"| {label} | {n} | {g:.1%} | {c:.1%} | " + " | ".join(
                f"{a / b:.1%} (n={b})" if b else "—" for a, b in acc.values()) + " | " + " | ".join(
                f"{a / b:.1%}" if b else "—" for a, b in swaps.values()) + " | " + (
                ", ".join(f"{k} {v / w:.0%}" for k, v in wrong.most_common()) if w else "—") + " |")
    text = "\n".join(lines)
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)
    (OUT / "report.md").write_text(text + "\n", encoding="utf-8")


def agree() -> None:
    """The LLM runs are scored by harness.run.run_task, the tiny ones by
    `evaluate.py --score`. Both must pass every reference program, and must
    agree on A0's single-segment plain programs."""
    sys.path.insert(0, str(ROOT))
    from data.gen.domains import register_domains
    from harness.run import run_task
    register_domains(str(ROOT / "data/gen/themes"))
    ids = json.loads((OUT / "ids.json").read_text())
    for exam in EXAMS:
        want = set(ids[exam])
        rows = [r for r in map(json.loads, open(EXAMS[exam])) if r["id"] in want]
        ok = sum(run_task(r, lambda req, ctx, i, reg, fb=None, r=r:
                          r["reference"]["segments"][i] if i < len(r["reference"]["segments"])
                          else None)["goal_success"] for r in rows)
        print(f"{exam}: reference programs pass run_task {ok}/{len(rows)}")
    a0 = load_tiny(TINY / "pod_s6split/out/s6off_A0_plain")
    rows = {r["id"]: r for r in map(json.loads, open(EXAMS["plain"]))}
    single = [t for t in ids["plain"] if len(rows[t]["reference"]["segments"]) == 1][:50]
    same = sum(run_task(rows[t], lambda req, ctx, i, reg, fb=None, t=t:
                        a0[t]["program"] if i == 0 else None)["goal_success"] == a0[t]["goal"]
               for t in single)
    print(f"A0's plain programs: run_task agrees with evaluate.py --score on {same}/{len(single)}")


if __name__ == "__main__":
    {"draw": draw, "report": report, "agree": agree}[sys.argv[1]]()
