"""Episode runner for the decision worlds (family A / C).

Started as E-rpg for the dungeon (.claude/plans/rpg-demo-app.md) and now runs
any world in `harness.decision`: the dungeon, its three training siblings,
the held-out house, and the page apps
(.claude/plans/archive/task-families.md). `--world` picks; the default is
still rpg, so every stored E-rpg command line means what it always did.

One turn = one fresh single-segment program. Each turn the state is rendered
into a request plus constants (the world's `observe`), symbols are reassigned
(`build_context`, fresh T/F/C as everywhere else), the planner writes a
program, and it runs through the real pipeline. Episodes end at the world's
goal, at its failure state, or at the turn cap.

`harness.run.run_task` is deliberately not reused: it needs a task's
`expected_state` and drives PAUSE continuations, and an episode has neither a
single correct trace nor a continuation. Scoring is a terminal predicate, a
progress funnel, and — the numbers that move long before an episode is won —
a per-turn score against the oracle asked from the same state: did the
model's first call name the tool the oracle would have called with the same
argument, and did the whole turn match (task-families.md §4). Read the whole
turn: scoring only the opening call gives full marks to a model that sprays
every option, which is exactly what S3 does on the dungeon.

  python -m harness.rpg_suite --planner oracle --episodes 3
  python -m harness.rpg_suite --world warehouse --planner oracle --episodes 3
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
from harness import decision  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context, serialize_context  # noqa: E402
from harness.run import run_sandbox  # noqa: E402
from runtime.worlds import get_world  # noqa: E402

DEFAULT_MAX_TURNS = 40

# A planner sees only the rendered request text and the turn index; it returns
# the program plus whatever generation stats it has.
#   plan(input_text, turn_idx, state, ctx) -> {text, tokens_in, tokens_out,
#                                              gen_ms, finish_reason}
# `ctx` is the turn's TaskContext: symbols are reassigned every turn, so a
# per-task grammar has to be rebuilt every turn too (harness/task_grammar.py).
Planner = Callable[[str, int, dict], dict]


def oracle_planner(max_actions: int, world: str = "rpg") -> Planner:
    oracle = decision.oracle_module(world)

    def plan(input_text, turn_idx, state, ctx):
        t0 = time.perf_counter()
        text = oracle.plan_turn(state, budget=max_actions)
        return {"text": text, "authoring": True, "tokens_in": 0,
                "tokens_out": 0, "gen_ms": (time.perf_counter() - t0) * 1000,
                "finish_reason": "oracle"}

    return plan


def gguf_planner(model: str, ctx: int, grammar_path: Optional[str],
                 max_tokens: int, template: str, threads: Optional[int],
                 symbols: str = "classic", kinds: bool = False,
                 gpu_layers: int = 0) -> Planner:
    from llama_cpp import Llama, LlamaGrammar

    from baselines.qwen.run_a import SYSTEM, generate, typed_system
    from harness import task_grammar

    base = Path(grammar_path).read_text() if grammar_path else None
    typed = symbols == "typed" or kinds
    system = typed_system(SYSTEM) if symbols == "typed" else SYSTEM
    cache: dict = {}
    llm = Llama(model_path=model, n_ctx=ctx, n_threads=threads,
                n_gpu_layers=gpu_layers, verbose=False)

    def plan(input_text, turn_idx, state, task_ctx):
        grammar = None
        if base is not None:
            # the typed grammar builders read the stored-task shape, and a
            # turn's symbol table is rebuilt every turn, so go through to_json
            task = {"context": task_ctx.to_json()}
            key = (task_grammar.typed_signature(task) if typed
                   else task_grammar.symbol_signature(task))
            if key not in cache:
                cache[key] = LlamaGrammar.from_string(
                    task_grammar.grammar_for_task(task, base, typed=typed,
                                                  kinds=kinds),
                    verbose=False)
            grammar = cache[key]
        t0 = time.perf_counter()
        res = generate(llm, grammar, input_text + "\nPROGRAM:",
                       max_tokens, template, system=system)
        usage = res.get("usage") or {}
        return {"text": res["text"], "authoring": False,
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0),
                "gen_ms": (time.perf_counter() - t0) * 1000,
                "finish_reason": res.get("finish_reason")}

    return plan


def server_planner(url: str, grammar_path: Optional[str], max_tokens: int,
                   symbols: str = "classic", kinds: bool = False) -> Planner:
    """Same prompt, grammar and greedy decoding as `gguf_planner`, generated
    by a llama.cpp server instead of in-process. This box has no CUDA card
    and the installed llama-cpp-python is a CPU build, so the only GPU path
    here is LM Studio's bundled Vulkan `llama-server.exe` against the iGPU:
    ~2 s a turn against ~23 s in process, same programs token for token.

      llama-server.exe -m <gguf> -c 4096 -ngl 99 --port 8077
      python -m harness.rpg_suite --server http://127.0.0.1:8077 ...
    """
    import urllib.request

    from baselines.qwen.run_a import SYSTEM, typed_system
    from harness import task_grammar

    base = Path(grammar_path).read_text() if grammar_path else None
    typed = symbols == "typed" or kinds
    system = typed_system(SYSTEM) if symbols == "typed" else SYSTEM
    cache: dict = {}

    def plan(input_text, turn_idx, state, task_ctx):
        grammar = None
        if base is not None:
            task = {"context": task_ctx.to_json()}
            key = (task_grammar.typed_signature(task) if typed
                   else task_grammar.symbol_signature(task))
            if key not in cache:
                cache[key] = task_grammar.grammar_for_task(
                    task, base, typed=typed, kinds=kinds)
            grammar = cache[key]
        user = input_text + "\nPROGRAM:"
        prompt = (f"<|im_start|>system\n{system}<|im_end|>\n"
                  f"<|im_start|>user\n{user}<|im_end|>\n"
                  f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
        payload = {"prompt": prompt, "n_predict": max_tokens,
                   "temperature": 0, "stop": ["<|im_end|>"],
                   "cache_prompt": True}
        if grammar:
            payload["grammar"] = grammar
        t0 = time.perf_counter()
        req = urllib.request.Request(
            url.rstrip("/") + "/completion", json.dumps(payload).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as fh:
            res = json.load(fh)
        return {"text": res["content"].strip(), "authoring": False,
                "tokens_in": res.get("tokens_evaluated", 0),
                "tokens_out": res.get("tokens_predicted", 0),
                "gen_ms": (time.perf_counter() - t0) * 1000,
                # llama-server says eos | limit | word; the in-process path
                # says stop | length, and the tallies are compared
                "finish_reason": ("length" if res.get("stop_type") == "limit"
                                  else "stop")}

    return plan


def oracle_move(state: dict, constants: list, world: str,
                budget: int) -> dict:
    """What the oracle would do from *this* state: {tool, args, calls}, where
    `calls` is the whole turn and `tool`/`args` are its first call.

    The label for the per-turn score. It reads the oracle's authoring text
    rather than executing it — running it would mutate the state the model is
    about to act on."""
    oracle = decision.oracle_module(world)
    try:
        text = oracle.plan_turn(state, budget=budget)
    except Exception:  # noqa: BLE001 — a stuck oracle must not kill the run
        return {"tool": None, "args": [], "calls": []}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines and lines[0].startswith("ABORT"):
        return {"tool": "ABORT", "args": [], "calls": [("ABORT", [])]}
    calls = []
    for line in lines:
        if not line.startswith("CALL @"):
            continue
        parts = line[len("CALL @"):].split()
        args = []
        for tok in parts[1:]:
            if tok == "->":
                break
            if tok.startswith("$") and tok[1:].isdigit():
                args.append(constants[int(tok[1:])]["value"])
            else:
                args.append(tok)
        calls.append((parts[0], args))
    if not calls:
        return {"tool": None, "args": [], "calls": []}
    return {"tool": calls[0][0], "args": calls[0][1], "calls": calls}


def _score_turn(turn: dict, want: dict) -> None:
    """Two numbers, because one of them is gameable.

    `tool_match`/`arg_match` are the plan's own §4 measure: did the model's
    *first* call name the oracle's tool, with the oracle's argument. Measured
    alone it flatters a model that sprays — S3 on the dungeon calls `move` in
    all four directions every turn, so its first call is the oracle's move
    and it scores 100% while winning nothing. `turn_match` is the whole
    turn's call sequence against the oracle's, which spraying cannot pass.
    """
    calls = turn.get("calls") or []
    aborted = turn.get("status") == "aborted"
    got = [("ABORT", [])] if (aborted and not calls) else [
        (c["name"], list(c["args"])) for c in calls]
    turn["oracle_tool"] = want["tool"]
    turn["oracle_args"] = want["args"]
    turn["oracle_calls"] = [list(c) for c in want["calls"]]
    turn["tool_match"] = bool(want["tool"]) and bool(got) \
        and got[0][0] == want["tool"]
    turn["arg_match"] = bool(turn["tool_match"]
                             and got[0][1] == list(want["args"]))
    turn["turn_match"] = bool(want["calls"]) and got == [
        (t, list(a)) for t, a in want["calls"]]


def run_episode(planner: Planner, *, scenario: Optional[str] = None,
                seed: int = 0, max_turns: int = DEFAULT_MAX_TURNS,
                max_actions: int = 3, verbose: bool = False,
                symbols: str = "classic", enums: bool = False,
                world: str = "rpg") -> dict:
    """Play one game. Returns the episode row (JSONL-ready)."""
    module = decision.world_module(world)
    WORLD = get_world(world)
    scenario = scenario or decision.scenarios(world)[0]
    state = module.new_state(scenario)
    state["turn_budget"] = max_actions
    rng = random.Random(seed)

    turns = []
    counters = {"calls": 0, "invalid_calls": 0, "compile_failures": 0,
                "abstains": 0, "pauses": 0, "runtime_errors": 0,
                "budget_hits": 0}
    uses = {"TRY": 0, "PARALLEL": 0, "FOREACH": 0, "IF": 0}

    while state["status"] == "playing" and state["turn"] < max_turns:
        obs = module.observe(state)
        if hasattr(obs, "memory"):
            state["memory"] = obs.memory
        ctx, sandbox_ctx = build_context(WORLD, obs.constants, rng,
                                         symbols=symbols, enums=enums)
        input_text = serialize_context(obs.request, ctx)
        want = oracle_move(state, obs.constants, world, max_actions)

        gen = planner(input_text, state["turn"], state, ctx)
        text = gen["text"]
        if gen.get("authoring"):
            text = resolve(text, ctx)
        for kw in uses:
            uses[kw] += sum(1 for ln in text.splitlines()
                            if ln.strip().startswith(kw))

        # the world's own scalars before the turn ran (the dungeon's hp, the
        # robot's battery, the bankroll): what a breakdown reads afterwards
        before = {k: v for k, v in module.outcome(state).items()
                  if k != "funnel"}
        turn = {"turn": state["turn"], "request": obs.request,
                "program": gen["text"], "tokens_in": gen.get("tokens_in", 0),
                "tokens_out": gen.get("tokens_out", 0),
                "gen_ms": round(gen.get("gen_ms", 0.0), 1),
                "finish_reason": gen.get("finish_reason"),
                "before": before}

        result = build(text, ctx)
        if not result.compile_ok:
            counters["compile_failures"] += 1
            turn.update(status="static_error",
                        diagnostics=result.rendered_diagnostics()[:3],
                        calls=[])
            _score_turn(turn, want)
            turns.append(turn)
            state = _idle_turn(state, ctx, sandbox_ctx, WORLD)
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
        _score_turn(turn, want)
        turns.append(turn)
        if verbose:
            _echo(turn)

    out = module.outcome(state)
    tokens_out = [t["tokens_out"] for t in turns if t["tokens_out"]]
    gen_ms = [t["gen_ms"] for t in turns if t["gen_ms"]]
    scored = [t for t in turns if t.get("oracle_tool")]
    row = {
        "world": world, "scenario": scenario, "seed": seed,
        "max_actions": max_actions,
        "won": out["won"], "dead": out["dead"],
        "status": out["status"], "turns": len(turns),
        "funnel": out.get("funnel", {}),
        **counters,
        "uses": uses,
        "scored_turns": len(scored),
        "tool_match": sum(t["tool_match"] for t in scored),
        "arg_match": sum(t["arg_match"] for t in scored),
        "turn_match": sum(t["turn_match"] for t in scored),
        "tokens_in_p50": statistics.median(
            [t["tokens_in"] for t in turns if t["tokens_in"]] or [0]),
        "tokens_out_p50": statistics.median(tokens_out or [0]),
        "gen_ms_p50": round(statistics.median(gen_ms or [0]), 1),
        "finish_reasons": _tally(t["finish_reason"] for t in turns),
        "turn_log": turns,
    }
    # the dungeon's own columns, so stored E-rpg logs and their reports keep
    # meaning what they meant
    for key in ("hp", "key_taken", "door_opened", "enemies_left"):
        if key in out:
            row["hp_end" if key == "hp" else key] = out[key]
    return row


def _idle_turn(state: dict, ctx, sandbox_ctx: dict, WORLD: dict) -> dict:
    """A turn where nothing executed (the program did not compile) still
    passes: run the world's post hook with an empty program so a model that
    cannot compile does not get a free pass on the clock."""
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
    want = turn.get("oracle_tool") or "-"
    mark = "==" if turn.get("turn_match") else (
        "~" if turn.get("arg_match") else "!=")
    print(f"  t{turn['turn']:>2} {turn['status']:<13} {calls}"
          f"   [{mark} oracle {want}"
          f"({', '.join(map(str, turn.get('oracle_args') or []))})]", flush=True)


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return "unknown"


def _funnel(rows: list) -> dict:
    """Stage counts across the episodes. New rows carry a `funnel` dict; the
    stored E-rpg logs predate it and carry the dungeon's own columns."""
    if rows[0].get("funnel"):
        keys = list(rows[0]["funnel"])
        return {k: sum(bool(r.get("funnel", {}).get(k)) for r in rows)
                for k in keys}
    if "key_taken" in rows[0]:
        return {"key": sum(r["key_taken"] for r in rows),
                "door": sum(r["door_opened"] for r in rows),
                "exit": sum(r["won"] for r in rows)}
    return {}


def report(path: Path) -> None:
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    if not rows:
        print("no episodes")
        return
    n = len(rows)
    won = [r for r in rows if r["won"]]
    print(f"{path.name}: {n} episodes, world={rows[0].get('world', 'rpg')} "
          f"model={rows[0].get('model')} condition={rows[0].get('condition')}")
    print(f"  won            {len(won)}/{n} ({len(won) / n:.0%})")
    print(f"  died           {sum(r['dead'] for r in rows)}/{n}")
    if won:
        print(f"  turns to win   p50 {statistics.median(r['turns'] for r in won):.0f}")
    stages = _funnel(rows)
    if stages:
        print("  funnel         " + "  ".join(
            f"{k} {v}/{n}" for k, v in stages.items()))
    scored = sum(r.get("scored_turns", 0) for r in rows)
    if scored:
        tools = sum(r.get("tool_match", 0) for r in rows)
        args = sum(r.get("arg_match", 0) for r in rows)
        whole = sum(r.get("turn_match", 0) for r in rows)
        # the numbers that move before an episode is ever won. Read the last
        # one: the first two only look at the opening call, and a model that
        # sprays every option passes them without deciding anything.
        print(f"  per-turn       first tool {tools}/{scored} "
              f"({tools / scored:.0%})   first tool+arg {args}/{scored} "
              f"({args / scored:.0%})   whole turn {whole}/{scored} "
              f"({whole / scored:.0%})")
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
    ap.add_argument("--world", default="rpg",
                    choices=sorted(decision.DECISION_WORLDS),
                    help="which decision world to play; rpg is the dungeon "
                         "and stays the default so stored E-rpg commands "
                         "mean what they always did")
    ap.add_argument("--planner", default="model", choices=["model", "oracle"])
    ap.add_argument("--model", help="GGUF path (planner=model)")
    ap.add_argument("--server", metavar="URL",
                    help="generate through a llama.cpp server at URL instead "
                         "of in process (the GPU path on this box; qwen "
                         "template only). --model then only names the row.")
    ap.add_argument("--template", default="qwen", choices=["qwen", "chat"],
                    help="qwen: the ChatML markup our checkpoints were tuned "
                         "on; chat: the GGUF's own chat template")
    ap.add_argument("--grammar", default="baselines/qwen/agent_core.gbnf")
    ap.add_argument("--no-grammar", action="store_true")
    # spec 0.4.0 surface. It has to match the checkpoint: a typed-trained
    # model scored on classic prompts looks broadly broken (results/R6.md §0).
    ap.add_argument("--symbols", choices=["classic", "typed"], default="classic")
    ap.add_argument("--enums", action="store_true")
    ap.add_argument("--kinds", action="store_true")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--gpu-layers", type=int, default=0,
                    help="in-process offload; -1 is all layers. The "
                         "--server path ignores it.")
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0,
                    help="first episode's symbol seed; episode i uses seed+i "
                         "so every model plays the same paired games")
    ap.add_argument("--scenario", default=None,
                    help="named game; the world's first scenario by default")
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--max-actions", type=int, default=None,
                    help="actions per turn; the world's own budget by default")
    ap.add_argument("--out", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--report", metavar="RESULTS_JSONL")
    args = ap.parse_args()

    if args.report:
        report(Path(args.report))
        return

    if args.max_actions is None:
        args.max_actions = decision.default_actions(args.world)
    if args.planner == "oracle":
        planner = oracle_planner(args.max_actions, args.world)
        model_name, condition = "oracle", "oracle"
    else:
        if not args.model:
            ap.error("--model is required unless --planner oracle")
        if args.server:
            if args.template != "qwen":
                ap.error("--server only builds the qwen template")
            planner = server_planner(
                args.server, None if args.no_grammar else args.grammar,
                args.max_tokens, symbols=args.symbols, kinds=args.kinds)
        else:
            planner = gguf_planner(
                args.model, args.ctx,
                None if args.no_grammar else args.grammar,
                args.max_tokens, args.template, args.threads,
                symbols=args.symbols, kinds=args.kinds,
                gpu_layers=args.gpu_layers)
        model_name = Path(args.model).name
        condition = ("grammar-task" if not args.no_grammar
                     else "unconstrained") + f"/{args.template}"
    condition += ("+letters" if args.symbols == "typed" else "") + (
        "+enums" if args.enums else "") + ("+kinds" if args.kinds else "")

    meta = {"model": model_name, "condition": condition,
            "git_sha": git_sha(), "world": args.world,
            "max_turns": args.max_turns, "template": args.template,
            "symbols": args.symbols, "enums": args.enums, "kinds": args.kinds}
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
                          verbose=not args.quiet, world=args.world,
                          symbols=args.symbols, enums=args.enums)
        row.update(meta)
        rows.append(row)
        if not args.quiet:
            stages = " ".join(f"{k}={int(bool(v))}"
                              for k, v in (row.get("funnel") or {}).items())
            print(f"  -> {row['status']} in {row['turns']} turns  {stages}",
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
