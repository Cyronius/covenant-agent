"""Episode corpus for the decision families (A and C).

  python -m data.gen.episodes --world warehouse --episodes 200 --seed 42 \
      --out data/shards/warehouse_0.jsonl
  python -m data.gen.episodes --world all --episodes 60 --symbols typed \
      --enums --kinds --out data/shards/decision_0.jsonl

One row per turn, in the same task shape `data.gen` writes, so
`baselines.qwen.make_sft` converts it unchanged. The oracle labels every turn
and the world's own engine executes it, which is what makes a row's
`expected_state` a real derivation rather than an assertion.

The off-path part is the point (.claude/plans/archive/task-families.md §2).
An oracle plays perfectly and a model is off that path from its first wrong move; the
RPG showed it never comes back. So this does not record a golden play
through. At a random share of turns it executes something else - a legal move
the oracle would not have chosen, or an outright illegal one, so the next
turn's observation carries a real "Last turn: failed: ..." - and then asks
the oracle what it would do *from the state that produced*. Every turn is
labelled from the state the episode is actually in.

Held-out worlds (the dungeon, the house, the coursebuilder app) are refused
unless --holdout is passed, which is for building an exam, not a corpus.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.ir import TaskContext  # noqa: E402
from core.pipeline import build  # noqa: E402
from harness import decision  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context, serialize_context  # noqa: E402
from harness.run import run_sandbox  # noqa: E402
from harness.taskbuild import ReferenceError, build_task  # noqa: E402
from runtime.worlds import get_world  # noqa: E402

GENERATOR_VERSION = "0.3.0"
# family A is the decision worlds, family C the page apps; they share this
# builder because family C is family A in a screen's clothing
LEVEL_DECISION = 19
LEVEL_PAGE = 20
PAGE_WORLDS = {"app_settings", "app_checkout", "app_ticket",
               "app_coursebuilder"}

# turns whose reference would not execute, reported at the end of a run
DROPPED: list = []


def level_for(world: str) -> int:
    return LEVEL_PAGE if world in PAGE_WORLDS else LEVEL_DECISION


# --------------------------------------------------------------- actions

def candidate_actions(world_dict: dict, constants: List[dict],
                      rng: random.Random, n: int = 12) -> List[str]:
    """Well-typed one-call programs drawn from this turn's tools and
    constants, legal or not. Subtracting the world's own `legal_actions`
    leaves the moves that will fail at runtime - which is how a "Last turn:
    failed" line gets into the corpus at all."""
    by_type: dict = {}
    for i, c in enumerate(constants):
        by_type.setdefault(c["type"], []).append(i)
    out = []
    tools = [t for t in world_dict["tools"] if "READ" not in t["effects"]]
    for _ in range(n * 3):
        if len(out) >= n:
            break
        tool = rng.choice(tools)
        required = [p for p in tool["params"] if p.get("required", True)]
        if len(required) > 2:
            continue
        args = []
        ok = True
        for p in required:
            pool = by_type.get(p["type"])
            if not pool:
                ok = False
                break
            args.append(f"${rng.choice(pool)}")
        if not ok:
            continue
        line = f"CALL @{tool['name']}" + ("".join(f" {a}" for a in args))
        prog = line + "\nSTOP\n"
        if prog not in out:
            out.append(prog)
    return out


def pick_detour(module, world_dict: dict, state: dict, want: str,
                constants: List[dict], rng: random.Random,
                illegal_share: float) -> Optional[str]:
    """A move to execute instead of the oracle's, or None to stay on path."""
    legal = [a for a in module.legal_actions(state) if a != want]
    if rng.random() < illegal_share:
        legal_set = set(module.legal_actions(state))
        bad = [a for a in candidate_actions(world_dict, constants, rng)
               if a not in legal_set]
        if bad:
            return rng.choice(bad)
    return rng.choice(legal) if legal else None


# --------------------------------------------------------------- episodes

def run_episode(world: str, seed: int, rng: random.Random, *,
                symbols: str, enums: bool, kinds: bool,
                offpath: float, illegal_share: float,
                max_turns: Optional[int] = None) -> tuple:
    """Play one episode; return (task rows, outcome)."""
    module = decision.world_module(world)
    oracle = decision.oracle_module(world)
    world_dict = get_world(world)
    state = decision.sample_state(world, rng)
    budget = state.get("turn_budget", 3)
    cap = max_turns or state.get("max_turns", 30)

    rows = []
    turn_index = 0
    while state["status"] == "playing" and state["turn"] < cap:
        obs = module.observe(state)
        constants = obs.constants
        if not kinds:
            constants = [{k: v for k, v in c.items() if k != "kind"}
                         for c in constants]
        ctx, sandbox_ctx = build_context(
            world_dict, constants, random.Random(seed ^ (turn_index << 8)),
            symbols=symbols, enums=enums)
        want = oracle.plan_turn(state, budget=budget)

        row = _row(world, seed, turn_index, obs, constants, want, ctx,
                   sandbox_ctx, state, symbols)
        if row is not None:
            rows.append(row)

        detour = (pick_detour(module, world_dict, state, want, constants, rng,
                              illegal_share)
                  if rng.random() < offpath else None)
        played = detour or want
        state = _advance(state, played, ctx, sandbox_ctx, world_dict)
        if row is not None:
            row["provenance"]["off_path_move"] = bool(detour)
        turn_index += 1
    return rows, module.outcome(state)


def _row(world: str, seed: int, turn_index: int, obs, constants: List[dict],
         want: str, ctx, sandbox_ctx: dict, state: dict,
         symbols: str) -> Optional[dict]:
    """One (situation, oracle move) training pair. The oracle's program runs
    through the real pipeline to derive `expected_state`; a turn whose
    reference will not execute is dropped rather than shipped."""
    aborts = want.strip().startswith("ABORT")
    try:
        segments = [resolve(want, ctx)]
        task = build_task(
            task_id=f"{world}_ep{seed}_t{turn_index}",
            level=level_for(world), world_name=world,
            request=obs.request, constants=constants, segments=segments,
            seed=seed, expected_status="aborted" if aborts else "ok",
            state=copy.deepcopy(state),
            tags=sorted({"family:A" if world not in PAGE_WORLDS
                         else "family:C", "episode", f"world:{world}"}
                        | ({"abort"} if aborts else set())),
            provenance={
                "generator_version": GENERATOR_VERSION,
                "teacher": "oracle", "seed": seed, "recipe": "episode",
                "world": world, "turn": turn_index,
                "oracle_tool": want.strip().split("\n")[0],
            },
            prebuilt=(ctx, sandbox_ctx))
    except ReferenceError:
        # the oracle wrote something the sandbox would not execute from here;
        # drop the turn rather than ship a row whose label is a guess
        DROPPED.append(f"{world}#{seed}t{turn_index}")
        return None
    task["input_text"] = serialize_context(obs.request, ctx)
    res = build(segments[0], TaskContext.from_json(task["context"]))
    task["effects"] = res.static_effects
    task["spec_version"] = "0.4.0"
    task["symbols"] = symbols
    return task


def _advance(state: dict, program: str, ctx, sandbox_ctx: dict,
             world_dict: dict) -> dict:
    """Execute one turn for real - the world's post hook included, which the
    reference run in `_row` deliberately leaves out."""
    result = build(resolve(program, ctx), ctx)
    js = result.js if result.compile_ok else build("STOP\n", ctx).js
    sres = run_sandbox({
        "js": js, "state": state, "tools": sandbox_ctx["tools"],
        "fields": sandbox_ctx["fields"], "constants": sandbox_ctx["constants"],
        "now": world_dict["now"], "approval": True, "error_injection": [],
        "initial_registers": {}, "post_hook": world_dict["post_hook"],
    })
    return sres.get("state", state)


def main() -> None:
    ap = argparse.ArgumentParser(prog="data.gen.episodes")
    ap.add_argument("--world", default="all",
                    help="a decision world, or 'all' for every trainable one")
    ap.add_argument("--episodes", type=int, required=True,
                    help="episodes per world")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--offpath", type=float, default=0.25,
                    help="share of turns played with something other than "
                         "the oracle's move, so the next turn is labelled "
                         "from a state the golden path never reaches")
    ap.add_argument("--illegal", type=float, default=0.3,
                    help="share of those detours that are illegal moves, so "
                         "the corpus carries turns that open with a failure")
    ap.add_argument("--symbols", default="classic",
                    choices=["classic", "typed"])
    ap.add_argument("--enums", action="store_true")
    ap.add_argument("--kinds", action="store_true")
    ap.add_argument("--holdout", action="store_true",
                    help="allow the reserved worlds (exam building, never a "
                         "corpus)")
    args = ap.parse_args()

    if args.world == "all":
        worlds = list(decision.HELD_OUT) if args.holdout else decision.TRAINABLE
    else:
        worlds = [args.world]
    reserved = [w for w in worlds if w in decision.HELD_OUT]
    if reserved and not args.holdout:
        ap.error(f"{', '.join(reserved)} are held out; pass --holdout only "
                 f"to build an exam")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    wins = {w: 0 for w in worlds}
    with open(out, "w") as fh:
        for world in worlds:
            for i in range(args.episodes):
                seed = args.seed + i
                rng = random.Random((seed, world).__hash__() & 0xFFFFFFFF)
                rows, outcome = run_episode(
                    world, seed, rng, symbols=args.symbols, enums=args.enums,
                    kinds=args.kinds, offpath=args.offpath,
                    illegal_share=args.illegal)
                wins[world] += bool(outcome["won"])
                for row in rows:
                    fh.write(json.dumps(row) + "\n")
                    written += 1
                if (i + 1) % 20 == 0:
                    print(f"{world}: {i + 1}/{args.episodes} episodes, "
                          f"{written} turns", flush=True)
    print(f"wrote {written} turns -> {out}")
    if DROPPED:
        print(f"  {len(DROPPED)} turns dropped (the sandbox refused the "
              f"oracle's program), e.g. {DROPPED[:3]}")
    for world in worlds:
        print(f"  {world}: {wins[world]}/{args.episodes} episodes finished "
              f"(off-path detours make a clean finish the exception, not the "
              f"target)")


if __name__ == "__main__":
    main()
