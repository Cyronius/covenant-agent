"""The flip probe: the request is written for a decoy's action, so the decoy
is the answer (.claude/plans/description-reading.md step 1).

Everywhere else a decoy is never right, so a planner can score on a decoyed
exam by learning which *kind* of line is never the answer (results/R10.md
section 8), or by learning each slot's canonical action ("the one-argument
destructive tool is the delete"). Here one authored decoy of a flip slot
(`domains.FLIP_VERBS`: delete_child, set_bool, set_ref, send) swaps places
with the real tool: it takes the real tool's structure, sandbox impl and
request templates' role, so the request says "void the fares on segment X"
and the reference calls `voidSegFare`; the tool the theme calls real sits
beside it as the noop decoy. Only reading the request against both
descriptions gets it right. Scored on `same_tool`. Goal is consistent too
(the working impl moves with the promoted tool), but the impl is the slot's
-- a "void fares" that deletes -- so a state check says less than the call.

A corpus built with `data.gen --twin-roles` swaps these roles on three rows
in four already; this probe is the all-swapped slice, for a model trained
without them as much as with.

  python -m data.gen.flip_probe --n 400 --out data/s6_flip.jsonl

Holdout worlds only, typed surface, the default decoy rate. Rows carry the
`flip` tag and `provenance.flip = {slot, real, answer}`, and ids end
`+flip`.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from data.gen import __main__ as gen  # noqa: E402
from data.gen import domains  # noqa: E402
from data.gen.programs import SampleError  # noqa: E402
from harness.taskbuild import ReferenceError  # noqa: E402

# levels whose recipes act through a theme action slot
ACTION_LEVELS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 13, 14, 16)


def flipped(theme: dict, slot: str, k: int) -> dict:
    """The theme with decoy `k` of `slot` promoted to the working tool and
    the authored tool demoted to a decoy (`domains.swap_roles`)."""
    return domains.swap_roles(theme, {slot: k + 1})


def main() -> int:
    ap = argparse.ArgumentParser(prog="data.gen.flip_probe")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260923)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default=str(ROOT / "data/gen/themes"))
    ap.add_argument("--decoys", default="1:2")
    ap.add_argument("--opaque-names", type=float, default=0.0)
    ap.add_argument("--attempts", type=int, default=60)
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.decoys.split(":"))

    themes = {}
    for path in sorted(Path(args.domains).glob("*.json")):
        th = domains.load_theme(path)
        domains.register_theme(th)
        themes[th["domain"]] = th
    reserved = set(gen.RESERVED["worlds"])
    cands = [(w, slot, k) for w, th in sorted(themes.items()) if w in reserved
             for slot in domains.FLIP_VERBS
             for k in range(len(th["tools"][slot].get("decoys") or []))]
    if not cands:
        raise SystemExit("no holdout theme authors flip-slot decoys")
    rng = random.Random(args.seed)
    out = Path(args.out)
    written = skipped = 0
    seed = args.seed
    with out.open("w", encoding="utf-8", newline="\n") as fh:
        while written < args.n:
            world, slot, k = rng.choice(cands)
            th = flipped(themes[world], slot, k)
            answer = th["tools"][slot]["name"]
            domains.register_theme(th, record=False)
            task = None
            try:
                for _ in range(args.attempts):
                    seed += 1
                    try:
                        cand = gen.gen_one(rng.choice(ACTION_LEVELS), seed, True,
                                           "template", symbols="typed",
                                           enums=True, kinds=True,
                                           decoys=(lo, hi), world=world,
                                           opaque_rate=0.0)
                    except (SampleError, ReferenceError):
                        continue
                    if gen.is_noop(cand):
                        continue
                    called = {c.get("name") for c in cand["reference"]["call_log"]}
                    if answer in called and domains.request_fits(
                            cand["request"], slot, th["tools"][slot]):
                        task = cand
                        break
            finally:
                domains.register_theme(themes[world])
            if task is None:
                skipped += 1
                continue
            real = themes[world]["tools"][slot]["name"]
            task["id"] += "+flip"
            task["tags"] = sorted(set(task["tags"]) | {"flip"})
            task["provenance"]["flip"] = {"slot": slot, "real": real,
                                          "answer": answer}
            if args.opaque_names and random.Random(seed ^ 0x0A0E).random() < args.opaque_names:
                from harness.decoys import opaque_names
                ren = opaque_names(task, random.Random(seed ^ 0x0A0F))
                task["provenance"]["flip"].update(
                    real=ren.get(real, real), answer=ren[answer])
                task["tags"] = sorted(set(task["tags"]) | {"opaque-names"})
            fh.write(json.dumps(task) + "\n")
            written += 1
    print(f"wrote {written} flip rows -> {out} ({skipped} draws found no "
          f"task calling the promoted tool)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
