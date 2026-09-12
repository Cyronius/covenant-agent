"""Family E: recover from the error you were just shown.

  python -m data.gen.recovery --n 2000 --seed 42 --domains data/gen/themes \
      --out data/shards/recovery_0.jsonl

L8 already teaches `TRY`/`RETRY`, but as a shape rather than a decision: the
right continuation depends on *which* code came back. `NOT_FOUND` means find
the record another way, `PERMISSION_DENIED` means say you cannot, and
`RATE_LIMITED` means take the same call again inside a retry. R4 found the
react rule inert - the model does not read what happened last turn - and the
dungeon is the same thing with walls.

Each row is the turn *after* a failure. Its `input_text` is exactly what a
reactive planner sees at eval time (`baselines.qwen.run_a.build_prompt` with
`feedback`, minus the trailing PROGRAM: that make_sft adds back), its state
is the state the sandbox actually left, and its reference is the
continuation the code calls for. The failing first turn is not written out:
an ordinary correct-looking program is what levels 0-3 already teach, and it
is the second turn that is missing from the corpus.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from baselines.qwen.run_a import build_prompt  # noqa: E402
from core import diagnostics as dg  # noqa: E402
from core.ir import TaskContext  # noqa: E402
from core.pipeline import build  # noqa: E402
from data.gen import programs  # noqa: E402
from data.gen.programs import ConstAlloc, SampleError  # noqa: E402
from data.gen.worldgen import gen_state  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context, serialize_context  # noqa: E402
from harness.run import run_sandbox  # noqa: E402
from harness.taskbuild import ReferenceError, build_task  # noqa: E402
from runtime.worlds import get_world  # noqa: E402

GENERATOR_VERSION = "0.3.0"
LEVEL = 22

CODES = ["NOT_FOUND", "PERMISSION_DENIED", "RATE_LIMITED"]

RESERVED = json.loads(
    (ROOT / "data" / "holdout" / "reserved.json").read_text())
_DOMAIN_SPLIT = ROOT / "data" / "holdout" / "reserved_domains.json"
if _DOMAIN_SPLIT.exists():
    RESERVED["worlds"] = sorted(
        set(RESERVED["worlds"])
        | set(json.loads(_DOMAIN_SPLIT.read_text())["reserved_eval_domains"]))


def _draw(world_name: str, state: dict, rng: random.Random, code: str):
    """The failing first turn and the continuation the code calls for.

    Returns (request, constants, first_segment, continuation, injection).
    Nothing the first turn binds survives, so the continuation never reaches
    for a register - which is what lets it be an ordinary single-segment
    task once the feedback is in the prompt."""
    profile = programs.PROFILES[world_name]
    child = profile["v2"]["child"] if profile.get("v2") else None
    if child is None:
        raise SampleError("theme has no v2 block")
    prof = profile["entities"][child]
    name_field = prof.get("name_field")
    list_tool, get_tool = prof.get("list_tool"), prof.get("get_tool")
    if not (name_field and list_tool and get_tool):
        raise SampleError("theme cannot be looked up by name")
    actions = programs.visible_actions(prof, set(),
                                       exclude_kinds={"send_field"})
    if not actions:
        raise SampleError("no actions")

    alloc = ConstAlloc()
    rec, display, ref = programs._named_record(prof, child, state, rng, alloc)
    act = rng.choice(actions)
    noun = prof["noun"][0]

    if code == "NOT_FOUND":
        # the id lookup missed; the record may still be there under its name
        info = programs.build_action(act, child, {"<v>": "r0.@%s.id" % child},
                                     state, rng, alloc)
        first = (f"CALL @{get_tool} {ref} -> r0\n"
                 + programs.action_line(info, "r1" if info["dest"] else None)
                 + "\nSTOP\n")
        nref = alloc.get("STR", display,
                         f"the {noun} named, verbatim: {display}", kind="name")
        info2 = programs.build_action(act, child,
                                      {"<v>": "r2.@%s.id" % child},
                                      state, rng, alloc)
        cont = (f"CALL @{list_tool} -> r0\n"
                f"FILTER r0 @{child}.{name_field} EQ {nref} -> r1\n"
                f"IF EMPTY r1\n"
                f"  ABORT NOT_FOUND {nref}\n"
                f"FIRST r1 -> r2\n"
                + programs.action_line(info2, "r3" if info2["dest"] else None)
                + "\nSTOP\n")
        inj = [{"name": get_tool, "code": "NOT_FOUND", "times": -1}]
        cont_inj = list(inj)
    elif code == "PERMISSION_DENIED":
        info = programs.build_action(act, child, {"<v>": ref}, state, rng,
                                     alloc)
        first = programs.action_line(
            info, "r0" if info["dest"] else None) + "\nSTOP\n"
        cont = "ABORT UNSUPPORTED\n"
        inj = [{"name": info["tool"], "code": "PERMISSION_DENIED",
                "times": -1}]
        cont_inj = list(inj)
    else:  # RATE_LIMITED
        info = programs.build_action(act, child, {"<v>": ref}, state, rng,
                                     alloc)
        first = programs.action_line(
            info, "r0" if info["dest"] else None) + "\nSTOP\n"
        cont = ("TRY RETRY 3 -> r0\n"
                "  " + programs.action_line(info, "r1" if info["dest"] else None)
                + "\nSTOP\n")
        inj = [{"name": info["tool"], "code": "RATE_LIMITED", "times": 1}]
        # the continuation has to face it too, or the retry is decoration
        cont_inj = [{"name": info["tool"], "code": "RATE_LIMITED", "times": 1}]

    # build_action has already filled in the verb's value/name slots, so
    # only {obj} is left to place
    request = info["verb"].format(obj=f'the "{display}" {noun}')
    request = request[0].upper() + request[1:] + "."
    return request, alloc.items, first, cont, inj, cont_inj


def build_row(world_name: str, seed: int, rng: random.Random, code: str,
              symbols: str, enums: bool, kinds: bool) -> Optional[dict]:
    world = get_world(world_name)
    state = gen_state(world_name, rng, world["now"])
    request, constants, first, cont, inj, cont_inj = _draw(
        world_name, state, rng, code)
    if not kinds:
        constants = [{k: v for k, v in c.items() if k != "kind"}
                     for c in constants]
    ctx, sandbox_ctx = build_context(world, constants,
                                     random.Random(seed ^ 0x5EED),
                                     symbols=symbols, enums=enums)
    base_input = serialize_context(request, ctx)

    # run the failing turn for real: the error text, the calls that ran and
    # the state the sandbox left all have to be the genuine article
    first_res = build(resolve(first, ctx), ctx)
    if not first_res.compile_ok:
        raise SampleError("first turn does not compile")
    sres = run_sandbox({
        "js": first_res.js, "state": state, "tools": sandbox_ctx["tools"],
        "fields": sandbox_ctx["fields"], "constants": sandbox_ctx["constants"],
        "now": world["now"], "approval": True, "error_injection": inj,
        "initial_registers": {}})
    if sres.get("status") != "error":
        raise SampleError(f"first turn did not fail ({sres.get('status')})")
    err = sres.get("error") or {}
    if err.get("code") != code:
        raise SampleError(f"got {err.get('code')}, wanted {code}")

    rendered = dg.runtime_error(err.get("code"), sres.get("line", 0) or 0,
                                err.get("message", "")).render()
    feedback = {"error": rendered,
                "calls": [{"name": c.get("name"), "ok": c.get("ok", True)}
                          for c in sres.get("calls", [])],
                "registers": sres.get("registers") or {}}
    prompt = build_prompt(base_input, feedback["registers"],
                          [resolve(first, ctx).strip()], feedback)
    # make_sft adds the trailing PROGRAM: back, so drop it here rather than
    # ship a row with two of them
    input_text = prompt[:prompt.rindex("\nPROGRAM:")]

    segments = [resolve(cont, ctx)]
    aborts = cont.strip().startswith("ABORT")
    task = build_task(
        task_id=f"{world_name}_L{LEVEL}_{seed}_{code.lower()}", level=LEVEL,
        world_name=world_name, request=request, constants=constants,
        segments=segments, seed=seed,
        error_injection=cont_inj or None,
        expected_status="aborted" if aborts else "ok",
        state=sres.get("state", state),
        tags=["family:E", "recovery", f"code:{code.lower()}"]
             + (["abort"] if aborts else []),
        provenance={"generator_version": GENERATOR_VERSION,
                    "teacher": "template-v1", "seed": seed,
                    "recipe": f"recovery_{code.lower()}",
                    "failed_turn": resolve(first, ctx).strip()},
        prebuilt=(ctx, sandbox_ctx))
    task["input_text"] = input_text
    res = build(segments[0], TaskContext.from_json(task["context"]))
    task["effects"] = res.static_effects
    task["spec_version"] = "0.4.0"
    task["symbols"] = symbols
    return task


def main() -> None:
    ap = argparse.ArgumentParser(prog="data.gen.recovery")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--domains", default=None, metavar="DIR")
    ap.add_argument("--symbols", default="classic",
                    choices=["classic", "typed"])
    ap.add_argument("--enums", action="store_true")
    ap.add_argument("--kinds", action="store_true")
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--max-attempts", type=int, default=25)
    args = ap.parse_args()

    if args.domains:
        from data.gen.domains import register_domains
        print(f"registered {len(register_domains(args.domains))} domains")

    if args.holdout:
        pool = [w for w in RESERVED["worlds"] if w in programs.PROFILES]
    else:
        pool = sorted(set(programs.PROFILES) - set(RESERVED["worlds"]))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = failures = 0
    cursor = args.seed
    by_code: dict = {}
    with open(out, "w") as fh:
        while written < args.n:
            row = None
            for _ in range(args.max_attempts):
                cursor += 1
                rng = random.Random(cursor)
                code = CODES[written % len(CODES)]
                try:
                    row = build_row(rng.choice(pool), cursor, rng, code,
                                    args.symbols, args.enums, args.kinds)
                except (SampleError, ReferenceError):
                    failures += 1
                    row = None
                    continue
                break
            if row is None:
                print(f"FATAL: no recovery row in {args.max_attempts} "
                      f"attempts", file=sys.stderr)
                sys.exit(1)
            fh.write(json.dumps(row) + "\n")
            written += 1
            code = row["provenance"]["recipe"]
            by_code[code] = by_code.get(code, 0) + 1
            if written % 200 == 0:
                print(f"{written}/{args.n} ({failures} resamples)")
    print(f"wrote {written} rows ({failures} resamples) -> {out}")
    print("  by code:", dict(sorted(by_code.items())))


if __name__ == "__main__":
    main()
