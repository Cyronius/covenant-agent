"""F4 CLI.

  python -m data.gen --level 3 --n 10000 --seed 42 --out data/L3.jsonl
  python -m data.gen --level all --n 1100 --seed 7 --out data/mix.jsonl
  python -m data.gen --level 5 --n 200 --holdout --out data/holdout/L5.jsonl

Each JSONL row is a harness-runnable task (see harness.run) extended with:
  input_text   the model-facing serialized context (request + schemas)
  effects      the reference program's static effect set
  provenance   seed, generator version, teacher, recipe frame, style, tags
Held-out worlds/tools (data/holdout/reserved.json) never appear unless
--holdout is passed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.ir import TaskContext  # noqa: E402
from core.pipeline import build  # noqa: E402
from data.gen import english, programs  # noqa: E402
from data.gen.worldgen import gen_state  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context, serialize_context  # noqa: E402
from harness.taskbuild import ReferenceError, build_task  # noqa: E402
from runtime.worlds import get_world  # noqa: E402

GENERATOR_VERSION = "0.2.0"
RESERVED = json.loads(
    (ROOT / "data" / "holdout" / "reserved.json").read_text())
_DOMAIN_SPLIT = ROOT / "data" / "holdout" / "reserved_domains.json"
if _DOMAIN_SPLIT.exists():
    RESERVED["worlds"] = sorted(
        set(RESERVED["worlds"])
        | set(json.loads(_DOMAIN_SPLIT.read_text())["reserved_eval_domains"]))


def gen_one(level: int, seed: int, holdout: bool, teacher: str,
            crowd: tuple | None = None, symbols: str = "classic",
            enums: bool = False, kinds: bool = False,
            decoys: tuple | None = None, decoy_nonsense: float = 0.15,
            inject_open: tuple | None = None) -> dict:
    rng = random.Random(seed)
    if holdout:
        pool = [w for w in RESERVED["worlds"] if w in programs.PROFILES]
        world_name = rng.choice(pool)
        holdout_tools = set()
    else:
        world_names = sorted(set(programs.PROFILES) - set(RESERVED["worlds"]))
        world_name = rng.choice(world_names)
        holdout_tools = set(RESERVED["tools"].get(world_name, []))
    world = get_world(world_name)
    decoy_names: list = []
    if decoys:
        # family B: siblings of every mutating tool, same signature, only
        # the description telling them apart
        # (.claude/plans/archive/task-families.md)
        from harness.decoys import decoy_world
        # its own RNG, so a --decoys run draws the *same* world, state and
        # program as the run without it: the pair is a clean A/B and the gap
        # between the two scores is the number the family is after
        world, decoy_names = decoy_world(world, random.Random(seed ^ 0xDEC0),
                                         per_tool=decoys,
                                         nonsense=decoy_nonsense)
    if crowd:
        from harness.crowding import crowd_world
        pool = (RESERVED["worlds"] if holdout
                else sorted(set(programs.PROFILES) - set(RESERVED["worlds"])))
        donor_names = [w for w in pool
                       if w != world_name and w in programs.PROFILES]
        donors = [get_world(n)
                  for n in rng.sample(donor_names,
                                      min(10, len(donor_names)))]
        world = crowd_world(world, donors, rng,
                            rng.randint(crowd[0], crowd[1]))
    n_injected = 0
    if inject_open:
        # imported open schemas as distractors: tool names from outside the
        # themed vocabulary, injected into a themed world whose program,
        # entities and ID:/OBJ: typing are untouched
        # (.claude/plans/imported-schemas-as-distractors.md)
        from data.gen.open_pool import load_pool
        from harness.crowding import crowd_world
        before = len(world["tools"])
        # its own RNG, like --decoys: --inject-open draws the *same* world,
        # state and program as the run without it, so the pair is a clean
        # A/B and the gap between the two scores is the number this is
        # after. --crowd, which consumes the shared rng, is not.
        irng = random.Random(seed ^ 0xFEED)
        world = crowd_world(world, [], irng,
                            irng.randint(inject_open[0], inject_open[1]),
                            foreign=load_pool(holdout=holdout))
        n_injected = len(world["tools"]) - before
    now = world["now"]
    state = gen_state(world_name, rng, now)

    sample = programs.sample_level(level, world_name, state, now, rng,
                                  holdout_tools)
    if teacher == "template":
        request, style = english.render(sample.frame, rng)
    else:
        raise NotImplementedError(
            f"teacher {teacher!r} not wired in; only 'template' is available")

    visible = None
    if holdout_tools:
        visible = [t["name"] for t in world["tools"]
                   if t["name"] not in holdout_tools]
    constants = sample.constants
    if not kinds:
        # the identical-prompt control: enum kinds are what --enums means,
        # name/text are what --kinds adds (spec 0.4.0 §2.2)
        constants = [{k: v for k, v in c.items() if k != "kind"}
                     for c in constants]
    ctx, sandbox_ctx = build_context(world, constants,
                                     random.Random(seed ^ 0x5EED), visible,
                                     symbols=symbols, enums=enums)
    segments = [resolve(s, ctx) for s in sample.segments]

    task = build_task(
        task_id=f"{world_name}_L{level}_{seed}", level=level,
        world_name=world_name, request=request,
        constants=constants, segments=segments, seed=seed,
        error_injection=sample.error_injection or None,
        # abstain recipes (level 11) reference ABORT; the sandbox reports it
        expected_status="aborted" if "abort" in sample.tags else "ok",
        state=state,
        tags=sorted(set(sample.tags + [f"style:{style}"])
                    | ({"holdout"} if holdout else set())),
        provenance={
            "generator_version": GENERATOR_VERSION,
            "teacher": "template-v1" if teacher == "template" else teacher,
            "seed": seed, "recipe": sample.frame.get("recipe"),
            "style": style, "frame": sample.frame,
        },
        prebuilt=(ctx, sandbox_ctx))
    if crowd or decoy_names or n_injected:
        # crowded, decoyed and injected contexts contain tools the native
        # world cannot rebuild a sandbox for — store the payload with the
        # task, or harness/run.py:101 KeyErrors in sandbox_from_context
        task["sandbox"] = sandbox_ctx
        task["tags"] = sorted(set(task["tags"])
                              | ({"crowded"} if crowd else set())
                              | ({"decoyed"} if decoy_names else set())
                              | ({"open-injected"} if n_injected else set()))
    if decoy_names:
        task["provenance"]["decoys"] = decoy_names
    if n_injected:
        task["provenance"]["open_distractors"] = n_injected

    # training-pair extras
    task["input_text"] = serialize_context(request, ctx)
    res = build(segments[0], TaskContext.from_json(task["context"]))
    task["effects"] = res.static_effects
    task["spec_version"] = "0.4.0"
    task["symbols"] = symbols
    return task


MUTATING_EFFECTS = {"WRITE", "DELETE", "SEND", "PAY", "EXTERNAL"}


def is_noop(task: dict) -> bool:
    """A task whose reference declares mutating effects yet leaves the world
    untouched (template artifact, e.g. "close the closed tickets")."""
    return (task["expected_status"] == "ok"
            and task["expected_state"] == task["state"]
            and bool(MUTATING_EFFECTS & set(task["effects"])))


def build_schedule(args) -> list[int]:
    if args.levels:
        weights = {}
        for part in args.levels.split(","):
            lvl, w = part.split(":")
            weights[int(lvl)] = float(w)
        total = sum(weights.values())
        quotas = {l: args.n * w / total for l, w in weights.items()}
        counts = {l: int(q) for l, q in quotas.items()}
        short = args.n - sum(counts.values())
        by_remainder = sorted(quotas, key=lambda l: quotas[l] - counts[l],
                              reverse=True)
        for l in by_remainder[:short]:
            counts[l] += 1
        schedule = [l for l in sorted(counts) for _ in range(counts[l])]
        random.Random(args.seed).shuffle(schedule)
        return schedule
    levels = (sorted(programs.RECIPES) if args.level == "all"
              else [int(args.level)])
    return [levels[i % len(levels)] for i in range(args.n)]


def main():
    ap = argparse.ArgumentParser(prog="data.gen")
    ap.add_argument("--level",
                    help="0-10, or 'all' for an even mix")
    ap.add_argument("--levels",
                    help="weighted mix 'lvl:weight,...' e.g. "
                         "'0:5,2:12,3:12'; overrides --level")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", action="store_true",
                    help="generate from reserved worlds/tools (R5 eval only)")
    ap.add_argument("--teacher", default="template")
    ap.add_argument("--max-attempts", type=int, default=25)
    ap.add_argument("--drop-noops", action="store_true",
                    help="resample tasks whose reference has mutating "
                         "effects but leaves the state unchanged")
    ap.add_argument("--domains", default=None, metavar="DIR",
                    help="register generated domain themes from DIR before "
                         "generating (S0 multi-domain worldgen)")
    ap.add_argument("--symbols", default="classic",
                    choices=["classic", "typed"],
                    help="constant symbols: C0.. (0.3.x) or the 0.4.0 typed "
                         "letters S/N/B/D/I")
    ap.add_argument("--enums", action="store_true",
                    help="emit the schema's enum values as constants for "
                         "every entity the visible tools touch (spec 0.4.0 "
                         "§2.3)")
    ap.add_argument("--kinds", action="store_true",
                    help="declare string kinds (name / text / enum) on STR "
                         "constants (spec 0.4.0 §2.2)")
    ap.add_argument("--crowd", default=None, metavar="MIN:MAX",
                    help="add MIN..MAX distractor tools from other domains "
                         "to each task's context (E-crowded)")
    ap.add_argument("--decoys", default=None, metavar="MIN:MAX",
                    help="family B: give every mutating tool MIN..MAX "
                         "siblings with the same signature and a "
                         "neighbouring description, so only the description "
                         "says which one the request means")
    ap.add_argument("--inject-open", default=None, metavar="MIN:MAX",
                    help="inject MIN..MAX imported open-schema tools "
                         "(data/open_pairs) into each themed world as "
                         "distractors; the program, entities and typing stay "
                         "the themed world's")
    ap.add_argument("--decoy-nonsense", type=float, default=0.15,
                    help="share of decoys named foo17 / operation_93, so the "
                         "name cannot carry the choice at all")
    ap.add_argument("--require-collisions", type=float, default=None,
                    metavar="PCT",
                    help="fail if more than PCT%% of reference CALLs name a "
                         "tool whose signature no sibling shares. A file "
                         "above the ceiling cannot teach or test description "
                         "reading (results/GROUNDING.md); every new corpus "
                         "and exam should pass this. Without it the "
                         "statistic is printed and a 100%% file warns.")
    args = ap.parse_args()
    crowd = None
    if args.crowd:
        lo, hi = args.crowd.split(":")
        crowd = (int(lo), int(hi))
    decoys = None
    if args.decoys:
        lo, hi = args.decoys.split(":")
        decoys = (int(lo), int(hi))
    inject_open = None
    if args.inject_open:
        lo, hi = args.inject_open.split(":")
        inject_open = (int(lo), int(hi))
    if args.domains:
        from data.gen.domains import register_domains
        registered = register_domains(args.domains)
        print(f"registered {len(registered)} generated domains")
    if not args.level and not args.levels:
        ap.error("one of --level / --levels is required")

    schedule = build_schedule(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    failures = 0
    noops = 0
    seed_cursor = args.seed
    with open(out, "w") as f:
        while written < args.n:
            level = schedule[written]
            task = None
            for _ in range(args.max_attempts):
                seed_cursor += 1
                try:
                    candidate = gen_one(level, seed_cursor, args.holdout,
                                        args.teacher, crowd=crowd,
                                        symbols=args.symbols,
                                        enums=args.enums, kinds=args.kinds,
                                        decoys=decoys,
                                        decoy_nonsense=args.decoy_nonsense,
                                        inject_open=inject_open)
                except (programs.SampleError, ReferenceError):
                    failures += 1
                    continue
                if args.drop_noops and is_noop(candidate):
                    noops += 1
                    continue
                task = candidate
                break
            if task is None:
                print(f"FATAL: level {level} failed "
                      f"{args.max_attempts} consecutive attempts",
                      file=sys.stderr)
                sys.exit(1)
            f.write(json.dumps(task) + "\n")
            written += 1
            if written % 200 == 0:
                print(f"{written}/{args.n} (resamples: {failures}, "
                      f"noops dropped: {noops})")
    print(f"wrote {written} tasks -> {out} "
          f"(resamples: {failures}, noops dropped: {noops})")
    report_signature_uniqueness(out, args.require_collisions)


def report_signature_uniqueness(out: Path, ceiling: float | None) -> None:
    """Can the typed signature alone pick the tool in this file?

    A file where it always can cannot teach description reading and cannot
    test it either, whatever accuracy it later produces — see
    `results/GROUNDING.md`, which found every suite in the tree at 100%
    except the two built with decoys. Printed with every generated file so
    the property is never again an unexamined one.
    """
    from harness.signature_uniqueness import measure_tasks

    r = measure_tasks(out)
    if r is None:
        return
    print(f"signature uniqueness: {r['full']:.1%} full, "
          f"{r['stripped']:.1%} types-only, over {r['calls']} reference CALLs "
          f"({r['tools_per_task']:.1f} tools/task)")
    if ceiling is not None and r["full"] * 100 > ceiling:
        print(f"FATAL: {r['full']:.1%} of reference CALLs name a tool no "
              f"sibling shares a signature with, over the {ceiling:.0f}% "
              f"ceiling. A model can pick every tool by type shape alone, so "
              f"nothing scored on this file says anything about grounding. "
              f"Add signature-identical siblings with --decoys MIN:MAX.",
              file=sys.stderr)
        sys.exit(1)
    if r["full"] >= 0.99:
        print("WARNING: the signature identifies the tool in essentially "
              "every call. This file cannot teach or test description "
              "reading; H4 and R5 claims taken on it are unsupported "
              "(results/GROUNDING.md). --decoys MIN:MAX adds the "
              "discrimination; --require-collisions makes this fatal.")


if __name__ == "__main__":
    main()
