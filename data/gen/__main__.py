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

GENERATOR_VERSION = "0.1.0"
RESERVED = json.loads(
    (ROOT / "data" / "holdout" / "reserved.json").read_text())


def gen_one(level: int, seed: int, holdout: bool, teacher: str) -> dict:
    rng = random.Random(seed)
    if holdout:
        world_name = rng.choice(RESERVED["worlds"])
        holdout_tools = set()
    else:
        world_names = sorted(set(programs.PROFILES) - set(RESERVED["worlds"]))
        world_name = rng.choice(world_names)
        holdout_tools = set(RESERVED["tools"].get(world_name, []))
    world = get_world(world_name)
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
    ctx, sandbox_ctx = build_context(world, sample.constants,
                                     random.Random(seed ^ 0x5EED), visible)
    segments = [resolve(s, ctx) for s in sample.segments]

    task = build_task(
        task_id=f"{world_name}_L{level}_{seed}", level=level,
        world_name=world_name, request=request,
        constants=sample.constants, segments=segments, seed=seed,
        error_injection=sample.error_injection or None,
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

    # training-pair extras
    task["input_text"] = serialize_context(request, ctx)
    res = build(segments[0], TaskContext.from_json(task["context"]))
    task["effects"] = res.static_effects
    return task


def main():
    ap = argparse.ArgumentParser(prog="data.gen")
    ap.add_argument("--level", required=True,
                    help="0-10, or 'all' for an even mix")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--holdout", action="store_true",
                    help="generate from reserved worlds/tools (R5 eval only)")
    ap.add_argument("--teacher", default="template")
    ap.add_argument("--max-attempts", type=int, default=25)
    args = ap.parse_args()

    levels = (list(range(11)) if args.level == "all"
              else [int(args.level)])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    failures = 0
    seed_cursor = args.seed
    with open(out, "w") as f:
        while written < args.n:
            level = levels[written % len(levels)]
            task = None
            for _ in range(args.max_attempts):
                seed_cursor += 1
                try:
                    task = gen_one(level, seed_cursor, args.holdout,
                                   args.teacher)
                    break
                except (programs.SampleError, ReferenceError):
                    failures += 1
            if task is None:
                print(f"FATAL: level {level} failed "
                      f"{args.max_attempts} consecutive attempts",
                      file=sys.stderr)
                sys.exit(1)
            f.write(json.dumps(task) + "\n")
            written += 1
            if written % 200 == 0:
                print(f"{written}/{args.n} (resamples: {failures})")
    print(f"wrote {written} tasks -> {out} (resamples: {failures})")


if __name__ == "__main__":
    main()
