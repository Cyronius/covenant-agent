"""Family D: ask, then act.

  python -m data.gen.askact --n 2000 --seed 42 --domains data/gen/themes \
      --out data/shards/askact_0.jsonl

`PAUSE` and `ABORT NEEDS_INFO` are already in the corpus, but only ever as
endpoints: nothing in 62,841 rows shows the turn *after* the question. The
real sessions are conversations and their largest routing bucket expects no
tool call at all, so the model has to learn both halves - stop and ask when a
required value is missing, and then act on the answer when it comes back.

Each draw writes rows against the same world and the same state:

  ask       the request is missing one required value; the reference is
            `ABORT NEEDS_INFO F<field>`, the field whose value nobody gave
  act       the same request with the exchange appended in one templated
            line, the answer now a constant; the reference does the work
  complete  the request that states the value up front, so no question is
            called for at all; same reference as `act`

That third row is the counterweight, and it is not optional. A corpus of
ask/act pairs alone is half abstentions, and over-abstention is exactly what
S2 did (results/S2.md): the model learned to decide from the shape of the
constant table rather than from the request. `--complete-share` sets how
often it is written; at the default of 1.0 the ask is a third of the rows,
not a half.

The question stays a template, not prose (the non-goals in the plan): what
varies is which value is missing and how the answer is worded back.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.ir import TaskContext  # noqa: E402
from core.pipeline import build  # noqa: E402
from data.gen import programs  # noqa: E402
from data.gen.programs import ConstAlloc, SampleError  # noqa: E402
from data.gen.worldgen import gen_state  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context, serialize_context  # noqa: E402
from harness.taskbuild import ReferenceError, build_task  # noqa: E402
from runtime.worlds import get_world  # noqa: E402

GENERATOR_VERSION = "0.3.0"
LEVEL = 21

RESERVED = json.loads(
    (ROOT / "data" / "holdout" / "reserved.json").read_text())
_DOMAIN_SPLIT = ROOT / "data" / "holdout" / "reserved_domains.json"
if _DOMAIN_SPLIT.exists():
    RESERVED["worlds"] = sorted(
        set(RESERVED["worlds"])
        | set(json.loads(_DOMAIN_SPLIT.read_text())["reserved_eval_domains"]))

# how the answer comes back. One line, templated - the model has to read the
# value out of it, not parse a conversation.
REPLIES = [
    'You asked {question} They said: "{answer}".',
    'You asked {question} The answer came back: "{answer}".',
    'You asked {question} They replied "{answer}".',
    '(You asked {question} Answer: "{answer}".)',
]

ASK_OPENERS = [
    "Set up a new {noun} for {who}.",
    "Add a {noun} for {who}.",
    "Open a {noun} against {who}.",
]
ENUM_OPENERS = [
    "Move {obj} along to where it should be now.",
    "Put {obj} into the right state.",
    "Update where {obj} has got to.",
]


def _reply_line(rng: random.Random, question: str, answer: str) -> str:
    return rng.choice(REPLIES).format(question=question, answer=answer)


def sample_pair(world_name: str, state: dict, now: int,
                rng: random.Random) -> Optional[tuple]:
    """(ask, act, complete) samples for one draw."""
    profile = programs.PROFILES[world_name]
    v2 = profile.get("v2")
    if not v2:
        raise SampleError("theme has no v2 block")
    child = v2["child"]
    prof = profile["entities"][child]
    flavour = "create" if (v2.get("name_field") and v2.get("titles")
                           and rng.random() < 0.6) else "enum"
    if flavour == "create":
        return _create_pair(world_name, profile, v2, prof, state, rng)
    return _enum_pair(world_name, profile, v2, prof, state, rng)


def _create_pair(world_name, profile, v2, prof, state, rng):
    """The new record's name is the missing value - spec §4's own example of
    what NEEDS_INFO points at."""
    child, parent = v2["child"], v2["parent"]
    parents = state["entities"].get(parent, [])
    titles = v2.get("titles") or []
    if not parents or not titles:
        raise SampleError("no parent or no titles")
    who = rng.choice(parents)
    title = rng.choice(titles)
    noun = prof["noun"][0]
    opener = rng.choice(ASK_OPENERS).format(noun=noun, who=who["name"])
    question = f"what the {noun} should be called."

    ask = ConstAlloc()
    ask.get(f"ID:{parent}", who["id"], f"{who['name']}'s {v2['parent_noun'][0]} id")
    ask_seg = f"ABORT NEEDS_INFO @{child}.{v2['name_field']}\n"

    act = ConstAlloc()
    pref = act.get(f"ID:{parent}", who["id"],
                   f"{who['name']}'s {v2['parent_noun'][0]} id")
    tref = act.get("STR", title, f"the {noun} name, verbatim: {title}",
                   kind="name")
    act_seg = f"CALL @{v2['create']} {pref} {tref} -> r0\nSTOP\n"
    act_request = opener + " " + _reply_line(rng, question, title)
    whole = (f'Set up a new {noun} for {who["name"]}, called "{title}".')
    return ({"request": opener, "segments": [ask_seg],
             "constants": ask.items, "status": "aborted",
             "tags": ["family:D", "ask", "needs_info", "abort"],
             "flavour": "create"},
            {"request": act_request, "segments": [act_seg],
             "constants": act.items, "status": "ok",
             "tags": ["family:D", "act", "answered"],
             "flavour": "create"},
            {"request": whole, "segments": [act_seg],
             "constants": act.items, "status": "ok",
             "tags": ["family:D", "act", "no_question_needed"],
             "flavour": "create"})


def _enum_pair(world_name, profile, v2, prof, state, rng):
    """Which state to move a record into is the missing value."""
    child = v2["child"]
    records = state["entities"].get(child, [])
    if not records:
        raise SampleError("no records")
    act_spec = next((a for a in prof["actions"]
                     if a.get("kind") == "set_enum"), None)
    if act_spec is None:
        raise SampleError("theme has no set_enum action")
    rec = rng.choice(records)
    value = rng.choice([v for v in v2["enum_values"]
                        if v != rec.get(v2["enum_field"])]
                       or v2["enum_values"])
    name_field = v2.get("name_field")
    display = rec.get(name_field) if name_field else rec["id"]
    noun = prof["noun"][0]
    obj = f'the "{display}" {noun}' if name_field else f"{noun} {rec['id']}"
    opener = rng.choice(ENUM_OPENERS).format(obj=obj)
    question = (f"which {v2['enum_field']} it should be "
                f"({', '.join(v2['enum_values'])}).")

    ask = ConstAlloc()
    _name_const(ask, rec, name_field, noun, display)
    ask_seg = f"ABORT NEEDS_INFO @{child}.{v2['enum_field']}\n"

    act = ConstAlloc()
    nref = _name_const(act, rec, name_field, noun, display)
    vref = act.get("STR", value, f"the {value} {v2['enum_field']}",
                   kind=f"enum:{child}.{v2['enum_field']}")
    if name_field and prof.get("list_tool"):
        act_seg = (f"CALL @{prof['list_tool']} -> r0\n"
                   f"FILTER r0 @{child}.{name_field} EQ {nref} -> r1\n"
                   f"FIRST r1 -> r2\n"
                   f"CALL @{act_spec['tool']} r2.@{child}.id {vref} -> r3\n"
                   "STOP\n")
    else:
        act_seg = f"CALL @{act_spec['tool']} {nref} {vref} -> r0\nSTOP\n"
    act_request = opener + " " + _reply_line(rng, question, value)
    whole = f"Move {obj} to {value}."
    return ({"request": opener, "segments": [ask_seg],
             "constants": ask.items, "status": "aborted",
             "tags": ["family:D", "ask", "needs_info", "abort"],
             "flavour": "enum"},
            {"request": act_request, "segments": [act_seg],
             "constants": act.items, "status": "ok",
             "tags": ["family:D", "act", "answered"],
             "flavour": "enum"},
            {"request": whole, "segments": [act_seg],
             "constants": act.items, "status": "ok",
             "tags": ["family:D", "act", "no_question_needed"],
             "flavour": "enum"})


def _name_const(alloc: ConstAlloc, rec: dict, name_field: Optional[str],
                noun: str, display: str) -> str:
    if name_field:
        return alloc.get("STR", display,
                         f"the {noun} named, verbatim: {display}", kind="name")
    return alloc.get(f"ID:{rec['id'].rsplit('_', 1)[0]}", rec["id"],
                     f"{noun} {display}")


def build_row(world_name: str, sample: dict, state: dict, seed: int,
              tag: str, symbols: str, enums: bool, kinds: bool) -> dict:
    world = get_world(world_name)
    constants = sample["constants"]
    if not kinds:
        constants = [{k: v for k, v in c.items() if k != "kind"}
                     for c in constants]
    ctx, sandbox_ctx = build_context(world, constants,
                                     random.Random(seed ^ 0x5EED),
                                     symbols=symbols, enums=enums)
    segments = [resolve(s, ctx) for s in sample["segments"]]
    task = build_task(
        task_id=f"{world_name}_L{LEVEL}_{seed}_{tag}", level=LEVEL,
        world_name=world_name, request=sample["request"],
        constants=constants, segments=segments, seed=seed,
        expected_status=sample["status"], state=state,
        tags=sorted(set(sample["tags"]) | {f"askact:{sample['flavour']}"}),
        provenance={"generator_version": GENERATOR_VERSION,
                    "teacher": "template-v1", "seed": seed,
                    "recipe": f"askact_{sample['flavour']}_{tag}",
                    "pair": tag},
        prebuilt=(ctx, sandbox_ctx))
    task["input_text"] = serialize_context(sample["request"], ctx)
    res = build(segments[0], TaskContext.from_json(task["context"]))
    task["effects"] = res.static_effects
    task["spec_version"] = "0.4.0"
    task["symbols"] = symbols
    return task


def main() -> None:
    ap = argparse.ArgumentParser(prog="data.gen.askact")
    ap.add_argument("--n", type=int, required=True,
                    help="draws (2 rows each, plus a complete one)")
    ap.add_argument("--complete-share", type=float, default=1.0,
                    help="how often a draw also writes the request that "
                         "states the value up front and needs no question - "
                         "the counterweight to over-abstention")
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
    written = pairs = failures = 0
    cursor = args.seed
    with open(out, "w") as fh:
        while pairs < args.n:
            rows: List[dict] = []
            for _ in range(args.max_attempts):
                cursor += 1
                rng = random.Random(cursor)
                world_name = rng.choice(pool)
                world = get_world(world_name)
                state = gen_state(world_name, rng, world["now"])
                try:
                    ask, act, whole = sample_pair(world_name, state,
                                                  world["now"], rng)
                    rows = [build_row(world_name, ask, state, cursor, "ask",
                                      args.symbols, args.enums, args.kinds),
                            build_row(world_name, act, state, cursor, "act",
                                      args.symbols, args.enums, args.kinds)]
                    if rng.random() < args.complete_share:
                        rows.append(build_row(world_name, whole, state,
                                              cursor, "complete",
                                              args.symbols, args.enums,
                                              args.kinds))
                except (SampleError, ReferenceError):
                    failures += 1
                    rows = []
                    continue
                break
            if not rows:
                print("FATAL: no ask/act pair in "
                      f"{args.max_attempts} attempts", file=sys.stderr)
                sys.exit(1)
            for row in rows:
                fh.write(json.dumps(row) + "\n")
                written += 1
            pairs += 1
            if pairs % 200 == 0:
                print(f"{pairs}/{args.n} pairs ({failures} resamples)")
    print(f"wrote {written} rows ({pairs} draws, {failures} resamples) "
          f"-> {out}")


if __name__ == "__main__":
    main()
