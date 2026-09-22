"""Imported open schemas injected into themed worlds as distractors.

Plan: .claude/plans/imported-schemas-as-distractors.md. The bet is that
7,263 tool names the encoder has never read, sitting in the distractor slot
of a themed world, move unseen-world binding transfer. What these tests
cover is the part that is silent when it breaks:

  - a distractor the reference program can reach is a mislabelled row, and
    nothing downstream would say so;
  - an injected row without its own `sandbox` payload KeyErrors in
    `sandbox_from_context` only when someone finally evaluates it
    (harness/run.py:101);
  - an injection that consumes the shared RNG silently moves the world, the
    state and the program, so the plain/injected pair stops being an A/B and
    the gap between the two scores stops meaning anything.

The pod arms are what settle the bet; these keep the corpus honest.
"""
import random
import sys
from pathlib import Path

import pytest

from core.ir import TaskContext
from data.gen.__main__ import gen_one
from data.gen.open_pool import (HOLDOUT_PCT, MAX_PARAMS, _side, fit_params,
                                load_pool, tools_of_row)
from harness.crowding import crowd_world
from harness.run import reference_planner, run_task
from runtime.worlds import get_world

ROOT = Path(__file__).resolve().parent.parent
TINY = ROOT / "models" / "tiny"

LEVEL = 3
SEEDS = [20260919 + i for i in range(12)]


@pytest.fixture(scope="module")
def themed():
    """Register the generated themes, then put the registry back.

    `register_theme` shadows a same-named world unless it has a post_hook,
    and `data/gen/themes/coursebuilder.json` is a 9-tool theme over the
    36-tool hand-written world `tests/test_real_suite.py` asserts on. That
    only bites when a registering module runs first, so the teardown is not
    optional here.
    """
    from data.gen import domains, worldgen
    from runtime import worlds as registry
    live = (registry.WORLDS, domains.PROFILES, worldgen.GENERATORS,
            domains.TEXT_BANKS)
    saved = [dict(d) for d in live]
    domains.register_domains(str(ROOT / "data" / "gen" / "themes"))
    yield True
    # restored in place: other modules hold their own references to these
    # dicts, so rebinding the names here would put nothing back
    for d, snapshot in zip(live, saved):
        d.clear()
        d.update(snapshot)


@pytest.fixture(scope="module")
def pool():
    return load_pool(holdout=False)


# --------------------------------------------------------------- the pool

def test_pool_is_world_form(pool):
    """Every pooled tool is what build_context and the sandbox consume."""
    assert len(pool) > 4000
    for tool in pool[:500]:
        assert tool["name"] and isinstance(tool["desc"], str)
        # the external stub is what makes a wrong-but-well-typed call
        # execute instead of being statically impossible
        assert tool["impl"] == {"op": "external", "kind": tool["name"]}
        assert tool["returns"] is None
        names = [p["name"] for p in tool["params"]]
        assert len(names) == len(set(names)), tool["name"]
        for p in tool["params"]:
            assert p["type"] in ("STR", "INT", "BOOL")
            assert isinstance(p["required"], bool)
            # no entity annotation: that is why a pooled tool can never
            # collide with a themed world on an entity
            assert "field" not in p


def test_pool_split_is_disjoint_and_stable():
    """A holdout world's distractors must be unseen too, and eight parallel
    generator shards have to agree on which side a tool falls."""
    train = {t["name"] for t in load_pool(holdout=False)}
    held = {t["name"] for t in load_pool(holdout=True)}
    assert not (train & held)
    # 7,263 distinct names, less the 6 whose required parameters alone
    # overflow a schema line
    assert len(train) + len(held) == 7257
    frac = 100 * len(held) / (len(train) + len(held))
    assert abs(frac - HOLDOUT_PCT) < 2
    # a content hash, never the salted builtin: the same answer next process
    assert _side("search_movie") == 13
    assert _side("get_weather") == 78


def test_parameter_lists_fit_a_schema_line():
    """A tool line renders one slot per parameter and `prep.py` refuses to
    truncate one past --max-line (64 tokens). Measured, 7 parameters is the
    longest list that always fits; without the cap 46% of injected rows stop
    a prep run, after the whole corpus has been generated."""
    for holdout in (False, True):
        for tool in load_pool(holdout=holdout):
            assert len(tool["params"]) <= MAX_PARAMS, tool["name"]


def test_fit_params_trims_optionals_and_keeps_required():
    req = [{"name": f"r{i}", "required": True} for i in range(3)]
    opt = [{"name": f"o{i}", "required": False} for i in range(9)]
    kept = fit_params(req + opt, cap=7)
    assert [p["name"] for p in kept] == ["r0", "r1", "r2", "o0", "o1", "o2", "o3"]
    # interleaved: every required one survives, wherever it sits
    mixed = [opt[0], req[0], opt[1], req[1], opt[2], req[2], opt[3], opt[4]]
    kept = fit_params(mixed, cap=4)
    assert [p["name"] for p in kept] == ["o0", "r0", "r1", "r2"]
    # under the cap, nothing is touched
    assert fit_params(req, cap=7) is req
    # required alone over the cap is the only way out of the pool
    assert fit_params([dict(p, required=True) for p in req + opt], cap=7) is None


def test_descriptions_survive_the_line_serializer():
    """`serialize_context` writes one line per tool and per field and
    `models/tiny/prep.py` parses that back by line, so a newline inside a
    foreign description splits a tool line in two and prep refuses the row
    ("unrecognized context line") — after the whole corpus is generated.
    Four of the 7,263 imported descriptions carry one."""
    for holdout in (False, True):
        for tool in load_pool(holdout=holdout):
            texts = [tool["desc"]] + [p["desc"] for p in tool["params"]]
            for text in texts:
                assert "\n" not in text and "\r" not in text, tool["name"]
                assert " :: " not in text, tool["name"]
                assert text == text.strip()


def test_tools_of_row_recovers_param_names():
    """The param name lives on the row's FieldDecl, not on its ToolDecl."""
    row = {
        "context": {
            "tools": [{"sym": "T0", "name": "get_quote", "desc": "A quote",
                       "params": [{"sym": "F3", "type": "STR",
                                   "required": True, "desc": "ticker"}],
                       "returns": None, "effects": []}],
            "fields": [{"sym": "F3", "entity": None, "name": "symbol",
                        "type": "STR", "desc": "ticker"}],
            "constants": [],
        }
    }
    tool, = tools_of_row(row)
    assert tool["params"] == [{"name": "symbol", "type": "STR",
                               "required": True, "desc": "ticker"}]


# ----------------------------------------------------------- crowd_world

def test_foreign_injection_is_append_only(themed, pool):
    world = get_world("bookstore")
    before = [t["name"] for t in world["tools"]]
    merged = crowd_world(world, [], random.Random(0), 42, foreign=pool)
    after = [t["name"] for t in merged["tools"]]
    assert after[:len(before)] == before, "native tools must keep their order"
    assert len(after) == len(before) + 42
    assert len(set(after)) == len(after), "names must stay distinct"
    # entity-free distractors declare nothing, so the table is untouched
    assert merged["entities"] == world["entities"]
    assert [t["name"] for t in get_world("bookstore")["tools"]] == before


def test_foreign_name_collisions_are_skipped(themed):
    """Drop on collision, never rename: harness/crowding.py's policy, and
    with 41% of imported occurrences polysemous it is load-bearing."""
    world = get_world("bookstore")
    clash = dict(world["tools"][0])
    probe = {"name": "zz_unique_probe", "desc": "d", "params": [],
             "returns": None, "effects": [],
             "impl": {"op": "external", "kind": "zz_unique_probe"}}
    merged = crowd_world(world, [], random.Random(0), 2, foreign=[clash, probe])
    names = [t["name"] for t in merged["tools"]]
    assert names.count(clash["name"]) == 1
    assert "zz_unique_probe" in names


# ------------------------------------------------ generated injected rows

@pytest.fixture(scope="module")
def pairs(themed):
    """(plain, injected) at the same seed: the A/B the corpus rests on."""
    out = []
    for seed in SEEDS:
        plain = gen_one(LEVEL, seed, False, "template", symbols="typed",
                        enums=True, kinds=True)
        inj = gen_one(LEVEL, seed, False, "template", symbols="typed",
                      enums=True, kinds=True, inject_open=(42, 42))
        out.append((plain, inj))
    return out


def test_injection_does_not_move_the_task(pairs):
    """--inject-open derives its own RNG, so the pair differs only in the
    distractors. --crowd consumes the shared one and does not have this."""
    for plain, inj in pairs:
        assert inj["id"] == plain["id"]
        assert inj["world"] == plain["world"]
        assert inj["request"] == plain["request"]
        assert inj["state"] == plain["state"]
        assert inj["expected_status"] == plain["expected_status"]
        assert inj["provenance"]["frame"] == plain["provenance"]["frame"]
        assert inj["effects"] == plain["effects"]


def test_injected_rows_carry_their_sandbox(pairs):
    for plain, inj in pairs:
        assert "sandbox" in inj, "harness/run.py:101 KeyErrors without it"
        assert "open-injected" in inj["tags"]
        assert inj["provenance"]["open_distractors"] == 42
        assert {t["sym"] for t in inj["sandbox"]["tools"]} == \
            set(TaskContext.from_json(inj["context"]).tools)
        assert "sandbox" not in plain


def test_world_grows_only_in_the_distractor_slot(pairs):
    for plain, inj in pairs:
        assert len(inj["context"]["tools"]) == len(plain["context"]["tools"]) + 42
        assert len(inj["context"]["constants"]) == \
            len(plain["context"]["constants"])
        # fields grow too: every distractor param is an unlinked field slot
        assert len(inj["context"]["fields"]) > len(plain["context"]["fields"])


def test_reference_never_reaches_a_distractor(pairs):
    """A distractor the program can call is a mislabelled row."""
    for plain, inj in pairs:
        native = {t["name"] for t in get_world(plain["world"])["tools"]}
        ctx = TaskContext.from_json(inj["context"])
        for segment in inj["reference"]["segments"]:
            for token in segment.split():
                tool = ctx.tools.get(token)
                if tool is not None:
                    assert tool.name in native, \
                        f"{inj['id']} calls the distractor {tool.name}"


def test_injected_rows_still_execute(pairs):
    """Typecheck, compile and run in the sandbox, as the plain row does."""
    for plain, inj in pairs:
        row = run_task(inj, reference_planner(inj))
        assert row["goal_success"], (inj["id"], row["status"],
                                     row.get("diagnostics"))
        assert row["status"] == run_task(plain, reference_planner(plain))["status"]


def test_holdout_rows_draw_only_holdout_distractors(themed):
    """The silent one: a held-out world whose distractors are the ones
    training saw stops measuring novel schema reading, and says nothing."""
    from harness.context import COMPUTE_TOOLS

    train = {t["name"] for t in load_pool(holdout=False)}
    held = {t["name"] for t in load_pool(holdout=True)}
    compute = {t["name"] for t in COMPUTE_TOOLS}
    for seed in SEEDS[:4]:
        row = gen_one(LEVEL, seed, True, "template", symbols="typed",
                      enums=True, kinds=True, inject_open=(42, 42))
        native = {t["name"] for t in get_world(row["world"])["tools"]}
        injected = {t["name"] for t in row["context"]["tools"]} - native - compute
        assert injected <= held
        assert not injected & train


def test_crowd_and_injection_compose(themed):
    """Themed donors and foreign distractors in one world, both tagged."""
    row = gen_one(LEVEL, SEEDS[0], False, "template", symbols="typed",
                  enums=True, kinds=True, crowd=(15, 15), inject_open=(42, 42))
    assert {"crowded", "open-injected"} <= set(row["tags"])
    assert row["provenance"]["open_distractors"] == 42
    assert "sandbox" in row
    assert run_task(row, reference_planner(row))["goal_success"]


# ------------------------------------------- the unseen-world split itself

def test_reserved_eval_worlds_are_disjoint_from_the_training_pool(themed):
    """The split that measures generalisation is only unseen if `--holdout`
    and the default draw from disjoint pools. This is silent when it breaks:
    a reserved theme that leaks into training turns the unseen-world number
    into a seen-world number and nothing says so.

    42 reserved worlds the generator can build, against 104 training ones.
    R3, R7 and R8 measured on one world carved out of the training pool
    instead (`bookstore`, 0.96% of the data) while these sat unused.
    """
    from data.gen import programs
    from data.gen.__main__ import RESERVED
    reserved = set(RESERVED["worlds"])
    buildable = {w for w in reserved if w in programs.PROFILES}
    training = set(programs.PROFILES) - reserved
    assert len(buildable) == 42, sorted(buildable)
    assert len(training) == 104
    assert not (buildable & training)

    held = {gen_one(LEVEL, s, True, "template", symbols="typed", enums=True,
                    kinds=True)["world"] for s in SEEDS}
    seen = {gen_one(LEVEL, s, False, "template", symbols="typed", enums=True,
                    kinds=True)["world"] for s in SEEDS}
    assert held <= buildable
    assert not (seen & reserved)
    assert not (held & seen)


# -------------------------------------------------- the holdout widening

def _example(world, i):
    sys.path.insert(0, str(TINY))
    from corpus import Example
    return Example(task_id=f"{world}_{i}", level=1, world=world,
                   source="", target="", row={})


def test_split_holds_out_a_set_of_worlds():
    sys.path.insert(0, str(TINY))
    from corpus import split
    ex = [_example(w, i) for w in ("a", "b", "c", "d") for i in range(50)]
    tr, va, te, ho = split(ex, seed=0, holdout_worlds={"b", "d"})
    assert {e.world for e in ho} == {"b", "d"}
    assert len(ho) == 100
    assert not {e.world for e in tr + va + te} & {"b", "d"}
    # a bare string still means one world, and no argument means none
    assert len(split(ex, seed=0, holdout_worlds="b")[3]) == 50
    assert split(ex, seed=0)[3] == []


def test_holdout_draw_defaults_to_the_historical_pick():
    sys.path.insert(0, str(TINY))
    from prep import pick_holdout_worlds
    counts = {f"w{i}": 100 - i for i in range(11)}
    ranked = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
    # an existing cache has to rebuild byte for byte
    assert pick_holdout_worlds(counts, 1, None) == [ranked[len(ranked) // 2]]
    band = set(ranked[len(ranked) // 4:(3 * len(ranked)) // 4])
    drawn = [pick_holdout_worlds(counts, 3, s) for s in (0, 1, 2)]
    for d in drawn:
        assert len(d) == 3 == len(set(d))
        assert set(d) <= band, "the draw stays inside the mid-sized band"
    assert pick_holdout_worlds(counts, 3, 0) == drawn[0], "reproducible"
    assert len({tuple(d) for d in drawn}) > 1, "and it moves with the seed"
