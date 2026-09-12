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
            decoys: tuple | None = None, decoy_nonsense: float = 0.15) -> dict:
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
    if crowd or decoy_names:
        # crowded and decoyed contexts contain tools the native world cannot
        # rebuild a sandbox for — store the payload with the task
        task["sandbox"] = sandbox_ctx
        task["tags"] = sorted(set(task["tags"])
                              | ({"crowded"} if crowd else set())
                              | ({"decoyed"} if decoy_names else set()))
    if decoy_names:
        task["provenance"]["decoys"] = decoy_names

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
    ap.add_argument("--decoy-nonsense", type=float, default=0.15,
                    help="share of decoys named foo17 / operation_93, so the "
                         "name cannot carry the choice at all")
    args = ap.parse_args()
    crowd = None
    if args.crowd:
        lo, hi = args.crowd.split(":")
        crowd = (int(lo), int(hi))
    decoys = None
    if args.decoys:
        lo, hi = args.decoys.split(":")
        decoys = (int(lo), int(hi))
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
                                        decoy_nonsense=args.decoy_nonsense)
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


if __name__ == "__main__":
    main()
