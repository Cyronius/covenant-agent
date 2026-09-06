"""E-rpg: episode runner for the `rpg` world (plan .claude/plans/rpg-demo-app.md).

One turn = one fresh single-segment program. Each turn the state is rendered
into a request plus constants (`rpg.observe`), symbols are reassigned
(`build_context`, fresh T/F/C as everywhere else), the planner writes a
program, and it runs through the real pipeline. Episodes end at the stairs,
at death, or at the turn cap.

`harness.run.run_task` is deliberately not reused: it needs a task's
`expected_state` and drives PAUSE continuations, and an episode has neither a
single correct trace nor a continuation. Scoring is a terminal predicate plus
a progress funnel, in the spirit of harness/real_suite.py's routing scorer.

  python -m harness.rpg_suite --planner oracle --episodes 3
  python -m harness.rpg_suite --model baselines/qwen/models/qwen3.5-0.8b-s2-q8.gguf \
      --episodes 10 --out results/logs/s2_e_rpg.jsonl
  python -m harness.rpg_suite --report results/logs/s2_e_rpg.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.pipeline import build  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context, serialize_context  # noqa: E402
from harness.run import run_sandbox  # noqa: E402
from harness import rpg_oracle  # noqa: E402
from runtime.worlds import get_world, rpg  # noqa: E402

WORLD = get_world("rpg")
DEFAULT_MAX_TURNS = 40

# A planner sees only the rendered request text and the turn index; it returns
# the program plus whatever generation stats it has.
#   plan(input_text, turn_idx, state, ctx) -> {text, tokens_in, tokens_out,
#                                              gen_ms, finish_reason}
# `ctx` is the turn's TaskContext: symbols are reassigned every turn, so a
# per-task grammar has to be rebuilt every turn too (harness/task_grammar.py).
Planner = Callable[[str, int, dict], dict]


def oracle_planner(max_actions: int) -> Planner:
    def plan(input_text, turn_idx, state, ctx):
        t0 = time.perf_counter()
        text = rpg_oracle.plan_turn(state, budget=max_actions)
        return {"text": text, "authoring": True, "tokens_in": 0,
                "tokens_out": 0, "gen_ms": (time.perf_counter() - t0) * 1000,
                "finish_reason": "oracle"}

    return plan


def gguf_planner(model: str, ctx: int, grammar_path: Optional[str],
                 max_tokens: int, template: str, threads: Optional[int]
                 ) -> Planner:
    from llama_cpp import Llama, LlamaGrammar

    from baselines.qwen.run_a import generate
    from harness import task_grammar

    base = Path(grammar_path).read_text() if grammar_path else None
    cache: dict = {}
    llm = Llama(model_path=model, n_ctx=ctx, n_threads=threads, verbose=False)

    def plan(input_text, turn_idx, state, task_ctx):
        grammar = None
        if base is not None:
            key = (tuple(sorted(task_ctx.tools)),
                   tuple(sorted(task_ctx.fields)),
                   tuple(sorted(task_ctx.constants)))
            if key not in cache:
                cache[key] = LlamaGrammar.from_string(
                    task_grammar.grammar_for_context(task_ctx, base),
                    verbose=False)
            grammar = cache[key]
        t0 = time.perf_counter()
        res = generate(llm, grammar, input_text + "\nPROGRAM:",
                       max_tokens, template)
        usage = res.get("usage") or {}
        return {"text": res["text"], "authoring": False,
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "gen_ms": (time.perf_counter() - t0) * 1000,
                "finish_reason": res.get("finish_reason")}

    return plan


def run_episode(planner: Planner, *, scenario: str = "keep", seed: int = 0,
                max_turns: int = DEFAULT_MAX_TURNS, max_actions: int = 3,
                verbose: bool = False) -> dict:
    """Play one game. Returns the episode row (JSONL-ready)."""
    state = rpg.new_state(scenario)
    state["turn_budget"] = max_actions
    rng = random.Random(seed)

    turns = []
    counters = {"calls": 0, "invalid_calls": 0, "compile_failures": 0,
                "abstains": 0, "pauses": 0, "runtime_errors": 0,
                "budget_hits": 0}
    uses = {"TRY": 0, "PARALLEL": 0, "FOREACH": 0, "IF": 0}

    while state["status"] == "playing" and state["turn"] < max_turns:
        obs = rpg.observe(state)
        state["memory"] = obs.memory
        ctx, sandbox_ctx = build_context(WORLD, obs.constants, rng)
        input_text = serialize_context(obs.request, ctx)

        gen = planner(input_text, state["turn"], state, ctx)
        text = gen["text"]
        if gen.get("authoring"):
            text = resolve(text, ctx)
        for kw in uses:
            uses[kw] += sum(1 for ln in text.splitlines()
                            if ln.strip().startswith(kw))

        turn = {"turn": state["turn"], "request": obs.request,
                "program": gen["text"], "tokens_in": gen.get("tokens_in", 0),
                "tokens_out": gen.get("tokens_out", 0),
                "gen_ms": round(gen.get("gen_ms", 0.0), 1),
                "finish_reason": gen.get("finish_reason"),
                "hp": state["entities"]["player"][0]["hp"],
                "pos": [state["entities"]["player"][0]["x"],
                        state["entities"]["player"][0]["y"]]}

        result = build(text, ctx)
        if not result.compile_ok:
            counters["compile_failures"] += 1
            turn.update(status="static_error",
                        diagnostics=result.rendered_diagnostics()[:3],
                        calls=[])
            turns.append(turn)
            state = _idle_turn(state, ctx, sandbox_ctx)
            if verbose:
                _echo(turn)
            continue

        sres = run_sandbox({
            "js": result.js, "state": state,
            "tools": sandbox_ctx["tools"], "fields": sandbox_ctx["fields"],
            "constants": sandbox_ctx["constants"], "now": WORLD["now"],
            "approval": True, "error_injection": [], "initial_registers": {},
            "post_hook": WORLD["post_hook"],
        })
        state = sres.get("state", state)
        calls = sres.get("calls", [])
        counters["calls"] += len(calls)
        counters["invalid_calls"] += sum(1 for c in calls if not c["ok"])
        status = sres.get("status")
        if status == "aborted":
            counters["abstains"] += 1
        elif status == "paused":
            counters["pauses"] += 1
        elif status == "error":
            counters["runtime_errors"] += 1
            if (sres.get("error") or {}).get("code") == "RATE_LIMITED":
                counters["budget_hits"] += 1

        turn.update(status=status, diagnostics=[],
                    calls=[{"name": c["name"], "args": c["args"],
                            "ok": c["ok"], "error": c["error"]} for c in calls],
                    error=sres.get("error"), reason=sres.get("reason"))
        turns.append(turn)
        if verbose:
            _echo(turn)

    out = rpg.outcome(state)
    tokens_out = [t["tokens_out"] for t in turns if t["tokens_out"]]
    gen_ms = [t["gen_ms"] for t in turns if t["gen_ms"]]
    return {
        "scenario": scenario, "seed": seed, "max_actions": max_actions,
        "won": out["won"], "dead": out["dead"],
        "status": out["status"], "turns": len(turns),
        "hp_end": out["hp"], "key_taken": out["key_taken"],
        "door_opened": out["door_opened"], "enemies_left": out["enemies_left"],
        **counters,
        "uses": uses,
        "tokens_in_p50": statistics.median(
            [t["tokens_in"] for t in turns if t["tokens_in"]] or [0]),
        "tokens_out_p50": statistics.median(tokens_out or [0]),
        "gen_ms_p50": round(statistics.median(gen_ms or [0]), 1),
        "finish_reasons": _tally(t["finish_reason"] for t in turns),
        "turn_log": turns,
    }


def _idle_turn(state: dict, ctx, sandbox_ctx: dict) -> dict:
    """A turn where nothing executed (the program did not compile) still
    passes: run the enemy phase with an empty program so a model that cannot
    compile does not get a free pass on the clock."""
    empty = build("STOP\n", ctx)
    sres = run_sandbox({
        "js": empty.js, "state": state, "tools": sandbox_ctx["tools"],
        "fields": sandbox_ctx["fields"], "constants": sandbox_ctx["constants"],
        "now": WORLD["now"], "approval": True, "error_injection": [],
        "initial_registers": {}, "post_hook": WORLD["post_hook"],
    })
    return sres.get("state", state)


def _tally(values) -> dict:
    out: dict = {}
    for v in values:
        key = str(v)
        out[key] = out.get(key, 0) + 1
    return out


def _echo(turn: dict) -> None:
    calls = ", ".join(
        f"{c['name']}({', '.join(map(str, c['args']))}){'' if c['ok'] else ' X'}"
        for c in turn.get("calls", [])) or "-"
    print(f"  t{turn['turn']:>2} hp={turn['hp']} {tuple(turn['pos'])} "
          f"{turn['status']:<13} {calls}", flush=True)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def report(path: Path) -> None:
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    if not rows:
        print("no episodes")
        return
    n = len(rows)
    won = [r for r in rows if r["won"]]
    print(f"{path.name}: {n} episodes, model={rows[0].get('model')} "
          f"condition={rows[0].get('condition')}")
    print(f"  won            {len(won)}/{n} ({len(won) / n:.0%})")
    print(f"  died           {sum(r['dead'] for r in rows)}/{n}")
    if won:
        print(f"  turns to win   p50 {statistics.median(r['turns'] for r in won):.0f}")
    print("  funnel         key %d/%d  door %d/%d  exit %d/%d" % (
        sum(r["key_taken"] for r in rows), n,
        sum(r["door_opened"] for r in rows), n, len(won), n))
    total_calls = sum(r["calls"] for r in rows)
    print(f"  calls          {total_calls} ({sum(r['invalid_calls'] for r in rows)} invalid)")
    print(f"  turns          {sum(r['turns'] for r in rows)} "
          f"({sum(r['compile_failures'] for r in rows)} did not compile, "
          f"{sum(r['abstains'] for r in rows)} abstained, "
          f"{sum(r['budget_hits'] for r in rows)} hit the action budget)")
    print(f"  tokens_out p50 {statistics.median(r['tokens_out_p50'] for r in rows):.0f}"
          f"   gen_ms p50 {statistics.median(r['gen_ms_p50'] for r in rows):.0f}")
    reasons: dict = {}
    for r in rows:
        for k, v in (r.get("finish_reasons") or {}).items():
            reasons[k] = reasons.get(k, 0) + v
    print(f"  finish_reason  {reasons}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="harness.rpg_suite")
    ap.add_argument("--planner", default="model", choices=["model", "oracle"])
    ap.add_argument("--model", help="GGUF path (planner=model)")
    ap.add_argument("--template", default="qwen", choices=["qwen", "chat"],
                    help="qwen: the ChatML markup our checkpoints were tuned "
                         "on; chat: the GGUF's own chat template")
    ap.add_argument("--grammar", default="baselines/qwen/agent_core.gbnf")
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0,
                    help="first episode's symbol seed; episode i uses seed+i "
                         "so every model plays the same paired games")
    ap.add_argument("--scenario", default="keep")
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--max-actions", type=int, default=3)
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--report", metavar="RESULTS_JSONL")
    args = ap.parse_args()

    if args.report:
        report(Path(args.report))
        return

    if args.planner == "oracle":
        planner = oracle_planner(args.max_actions)
        model_name, condition = "oracle", "oracle"
    else:
        if not args.model:
            ap.error("--model is required unless --planner oracle")
        planner = gguf_planner(
            args.model, args.ctx,
            None if args.no_grammar else args.grammar,
            args.max_tokens, args.template, args.threads)
        model_name = Path(args.model).name
        condition = ("grammar-task" if not args.no_grammar
                     else "unconstrained") + f"/{args.template}"

    meta = {"model": model_name, "condition": condition,
            "git_sha": git_sha(), "world": "rpg",
            "max_turns": args.max_turns, "template": args.template}
    out_f = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_f = open(args.out, "w")

    rows = []
    for i in range(args.episodes):
        seed = args.seed + i
        if not args.quiet:
            print(f"episode {i + 1}/{args.episodes} (seed {seed})", flush=True)
        row = run_episode(planner, scenario=args.scenario, seed=seed,
                          max_turns=args.max_turns,
                          max_actions=args.max_actions,
                          verbose=not args.quiet)
        row.update(meta)
        rows.append(row)
        if not args.quiet:
            print(f"  -> {row['status']} in {row['turns']} turns "
                  f"(key={row['key_taken']} door={row['door_opened']})",
                  flush=True)
        if out_f:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()
    if out_f:
        out_f.close()
        report(Path(args.out))
    else:
        won = sum(r["won"] for r in rows)
        print(f"{won}/{len(rows)} won")
    sys.exit(0)


if __name__ == "__main__":
    main()
