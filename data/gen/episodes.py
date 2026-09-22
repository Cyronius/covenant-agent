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
    tools = [t for t in world_dict["tools"] if t["effects"]]
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


def call_parts(program: str) -> tuple:
    """('go', ('$0',)) for the first CALL of a one-call program."""
    for line in program.strip().splitlines():
        line = line.strip()
        if not line.startswith("CALL "):
            continue
        parts = line.split()
        return parts[1].lstrip("@"), tuple(parts[2:])
    return "", ()


def predictable_wrong(module, world_dict: dict, state: dict, want: str,
                      constants: List[dict], rng: random.Random,
                      verify=None) -> Optional[str]:
    """The wrong move a planner would actually make, not a random one.

    A random illegal move ("use the north way") teaches nothing about
    reacting, because the failure it produces has nothing to do with what
    the turn was trying to achieve — the oracle's next label is simply the
    plan it already had. The mistakes a planner makes are aimed at the
    right thing: the same verb on something that refuses it (walking
    through a shut way), or the right object with the verb that does not
    apply to it yet. The world then answers with a reason
    (`failed: the north way is shut - open it first`) and the oracle's next
    move is the one that reason calls for — so the row after the failure
    teaches "do what it just told you", which is the habit the dungeon
    shows the model has never been taught (`.claude/plans/
    general-agent-plan.md` Tier 1 item 1).
    """
    legal = set(module.legal_actions(state))
    verb, args = call_parts(want)
    if not verb:
        return None
    by_type: dict = {}
    for i, c in enumerate(constants):
        by_type.setdefault(c["type"], []).append(f"${i}")
    tools = {t["name"]: t for t in world_dict["tools"] if t["effects"]}

    def build_call(tool: dict, argv) -> Optional[str]:
        required = [p for p in tool["params"] if p.get("required", True)]
        if len(required) != len(argv):
            return None
        for p, a in zip(required, argv):
            if a not in by_type.get(p["type"], []):
                return None
        return f"CALL @{tool['name']}" + "".join(f" {a}" for a in argv) + \
            "\nSTOP\n"

    # the right object, the verb that does not apply to it yet ("walk
    # through the way that is shut")
    same_obj = [c for name, tool in sorted(tools.items()) if name != verb
                for c in [build_call(tool, args)] if c and c not in legal]
    # the right verb, aimed at something that refuses it
    same_verb = []
    if verb in tools:
        tool = tools[verb]
        required = [p for p in tool["params"] if p.get("required", True)]
        if len(required) == 1:
            for alt in by_type.get(required[0]["type"], []):
                if alt in args:
                    continue
                call = build_call(tool, (alt,))
                if call and call not in legal:
                    same_verb.append(call)
    # nothing aimed at the oracle's own move (a page app's first move opens
    # the one screen there is): aim at something else the turn really offers
    # — a verb that does not apply to an object the board actually has,
    # which is still the mistake of a planner reading the screen rather than
    # a call assembled out of nothing
    on_board = []
    for action in sorted(legal):
        for arg in call_parts(action)[1]:
            if not arg.startswith("$"):
                continue
            for name, tool in sorted(tools.items()):
                call = build_call(tool, (arg,))
                if call and call not in legal:
                    on_board.append(call)
    # order matters: measured over 38 failures, an aim at the right object
    # with the wrong verb is answered by the next label 7 times out of 8, one
    # aimed at any object the board offers 4 of 12, and the right verb at the
    # wrong object 0 of 15 — the last teaches nothing, so it goes last and
    # only survives when verification is off (`results/REACT.md`)
    pools = [(name, p) for name, p in (("same_obj", same_obj),
                                       ("on_board", on_board),
                                       ("same_verb", same_verb)) if p]
    if not pools:
        return None
    if verify is not None:
        # A failure only teaches when the next label answers it, and whether
        # it does is decidable here: play the candidate, read the oracle's
        # move from the state it produced, and keep the candidate whose
        # failure that move responds to. Three tries, then take the best
        # unverified aim rather than fall back to a random one.
        tried = 0
        for name, pool in pools:
            for cand in rng.sample(pool, len(pool)):
                if tried >= 8:
                    break
                tried += 1
                if verify(cand):
                    return cand, name
        # nothing aimed here would be answered: a random illegal move is no
        # worse and does not pretend to teach a reaction
        return None
    name, pool = pools[0]
    return rng.choice(pool), name


def pick_detour(module, world_dict: dict, state: dict, want: str,
                constants: List[dict], rng: random.Random,
                illegal_share: float,
                predictable_share: float = 1.0, verify=None) -> tuple:
    """(move to execute instead of the oracle's, which kind it is), or
    (None, None) to stay on path. The kind is recorded on the row so a
    failure's teaching value can be read against how it was produced."""
    legal = [a for a in module.legal_actions(state) if a != want]
    if rng.random() < illegal_share:
        if rng.random() < predictable_share:
            aimed = predictable_wrong(module, world_dict, state, want,
                                      constants, rng, verify)
            if aimed:
                move, branch = aimed
                return move, f"aimed:{branch}"
        legal_set = set(module.legal_actions(state))
        bad = [a for a in candidate_actions(world_dict, constants, rng)
               if a not in legal_set]
        if bad:
            return rng.choice(bad), "random_illegal"
    return (rng.choice(legal), "legal") if legal else (None, None)


# --------------------------------------------------------------- episodes

def run_episode(world: str, seed: int, rng: random.Random, *,
                symbols: str, enums: bool, kinds: bool,
                offpath: float, illegal_share: float,
                predictable_share: float = 1.0,
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

        def answered(candidate: str, _state=state) -> bool:
            """Would the oracle's next move answer this failure?

            Played here, for real, before the detour is chosen: a failure
            only earns its place in the corpus when the label that follows
            it is a response — the verb the refusal names, or the same
            object another way. Selecting for that is what makes the
            observation matter (plan step 3a); without it, an aimed move
            produces a failure the next label happens to ignore, which is
            measurably most of them (`results/REACT.md`).
            """
            probe = _advance(copy.deepcopy(_state), candidate, ctx,
                             sandbox_ctx, world_dict)
            failure = " ".join(str(e) for e in probe.get("log") or []
                               if str(e).startswith("failed:")).lower()
            if not failure:
                return False
            plan = oracle.plan_turn(probe, budget=budget)
            if plan.strip() == candidate.strip():
                return False            # never teach repeating what failed
            verb, args = call_parts(plan)
            bad_verb, bad_args = call_parts(candidate)
            stem, bad_stem = verb.split("_")[0], bad_verb.split("_")[0]
            if not stem or stem == bad_stem:
                return False
            return stem in failure or bool(set(args) & set(bad_args))

        detour, kind = (pick_detour(module, world_dict, state, want,
                                    constants, rng, illegal_share,
                                    predictable_share, verify=answered)
                        if rng.random() < offpath else (None, None))
        played = detour or want
        state = _advance(state, played, ctx, sandbox_ctx, world_dict)
        if row is not None:
            row["provenance"]["off_path_move"] = bool(detour)
            # what actually happened, so the next turn's label can be read
            # against it (`report_reaction`): a corpus that answers a
            # failure with the move that just failed teaches repeating it
            row["provenance"]["played_move"] = played.strip().splitlines()[0]
            row["provenance"]["detour_kind"] = kind or "on_path"
            row["provenance"]["played_failed"] = any(
                str(entry).startswith("failed:")
                for entry in state.get("log") or [])
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
    ap.add_argument("--predictable", type=float, default=1.0,
                    help="share of those illegal moves that try to be a "
                         "failure worth learning from: the right object with "
                         "the verb that does not apply to it yet, played "
                         "here first to confirm the oracle's next move "
                         "answers the refusal. Measured over 12 episodes a "
                         "world, an aimed failure is answered by the next "
                         "label 11 times out of 11 and a random one 1 time "
                         "in 25, which is why the default tries every time "
                         "and falls back to a random move only when no aim "
                         "would be answered (plan step 3a, results/"
                         "REACT.md). 0 reproduces the pre-2026-09-22 "
                         "corpora.")
    ap.add_argument("--symbols", default="classic",
                    choices=["classic", "typed"])
    ap.add_argument("--enums", action="store_true")
    ap.add_argument("--kinds", action="store_true")
    ap.add_argument("--holdout", action="store_true",
                    help="allow the reserved worlds (exam building, never a "
                         "corpus)")
    ap.add_argument("--require-collisions", type=float, default=50,
                    metavar="PCT",
                    help="fail (and write nothing) if more than PCT%% of "
                         "reference CALLs name a tool whose signature no "
                         "sibling shares (data.gen's ceiling, applied here "
                         "too). Default 50; see --allow-signature-unique.")
    ap.add_argument("--allow-signature-unique", action="store_true",
                    help="disable the ceiling above and stamp every row's "
                         "provenance with signature_unique_allowed — a "
                         "decision world with only a handful of tools (a "
                         "move is every tool there is) may need this for a "
                         "structural reason, not an accident.")
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
                    illegal_share=args.illegal,
                    predictable_share=args.predictable)
                wins[world] += bool(outcome["won"])
                for row in rows:
                    if args.allow_signature_unique:
                        row["provenance"]["signature_unique_allowed"] = True
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
    from data.gen.__main__ import report_signature_uniqueness
    report_signature_uniqueness(out, args.require_collisions,
                                args.allow_signature_unique)
    report_reaction(out)


def report_reaction(out) -> Optional[dict]:
    """After a failure, does the label do something else?

    The dungeon exam has the model repeat a move it was just told failed,
    103 turns out of 105 (`.claude/plans/general-agent-plan.md` Tier 1 item
    1). The reason is here, not in the model: a turn that opens with
    `Last turn: failed: ...` is labelled with the oracle's plan for a board
    the failure did not change, and until step 3a that failure was a random
    illegal move, so the label had no reason to answer it. This reads the
    written file and says what it teaches — printed beside signature
    uniqueness, not gated, because the right share is a judgment the
    corpus's other numbers do not settle.
    """
    played: dict = {}
    rows: list = []
    with Path(out).open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            prov = row.get("provenance") or {}
            key = (prov.get("world"), prov.get("seed"), prov.get("turn"))
            if None in key:
                continue
            played[key] = prov
            rows.append((key, prov.get("oracle_tool") or "",
                         failure_text(row.get("request") or "")))
    after = repeated = same_object = answered = 0
    by_branch: dict = {}
    for key, label, failure in rows:
        if not failure:
            continue
        after += 1
        prev = played.get((key[0], key[1], key[2] - 1)) or {}
        branch = prev.get("detour_kind") or "on_path"
        b = by_branch.setdefault(branch, {"n": 0, "answered": 0})
        b["n"] += 1
        failed = prev.get("played_move") or ""
        verb, args = call_parts(label)
        repeat = bool(failed) and call_parts(label) == call_parts(failed)
        if repeat:
            repeated += 1
        elif failed and set(args) & set(call_parts(failed)[1]):
            same_object += 1
        # the reason the world gave names the verb the label uses: "the
        # north way is shut - open it first" answered by `CALL @open`, "the
        # screen is not open" answered by `CALL @open`. The page apps need
        # this: there the answer acts on a *different* object (the screen)
        # than the move that failed (an element on it).
        stem = verb.split("_")[0]
        failed_stem = call_parts(failed)[0].split("_")[0]
        # "the north way is shut - open it first" after a failed `go`, then a
        # label that opens it. The verb has to differ from the one that
        # failed, or a world whose refusal echoes the verb it refused
        # ("cannot drive to shelf 0") would score every retry as an answer.
        named = bool(stem) and stem != failed_stem and stem in failure.lower()
        same_thing = bool(failed) and set(args) & set(call_parts(failed)[1])             and stem != failed_stem
        if not repeat and (named or same_thing):
            answered += 1
            b["answered"] += 1
    if not after:
        return None
    stats = {"after_failure": after, "repeated": repeated,
             "same_object": same_object, "answered": answered,
             "by_branch": by_branch}
    aimed = sum(b["n"] for name, b in by_branch.items()
                if name.startswith("aimed:"))
    stats["aimed"] = aimed
    print(f"reaction: of {after} turns that open with a failure "
          f"({aimed}, {aimed / after:.1%}, caused by an aimed move), "
          f"{repeated} ({repeated / after:.1%}) are labelled with the move "
          f"that just failed, {same_object} ({same_object / after:.1%}) act "
          f"on the same thing another way, and {answered} "
          f"({answered / after:.1%}) answer it - by name or on the same "
          f"object")
    for branch, b in sorted(by_branch.items()):
        print(f"    {branch:22s} {b['n']:4d} failures, "
              f"{b['answered']}/{b['n']} answered")
    return stats


def failure_text(request: str) -> str:
    """The `Last turn:` line of an observation, when it reports a failure."""
    for line in request.splitlines():
        if line.startswith("Last turn:") and "failed:" in line:
            return line[len("Last turn:"):].strip()
    return ""


if __name__ == "__main__":
    main()
