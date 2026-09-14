"""Generator recipes for the 0.4.0 stdlib (`.claude/plans/spec-0.4.0.md` §4).

The A/B on the 27B (results/R5.md) confirmed `MOST`/`LEAST`/`EMPTY` in
context; these tests pin what the corpus teaches: argmax/argmin by count are
one instruction, the "did anything match" branch is `EMPTY`, and every STR
constant declares a kind.
"""
import pytest

import data.gen.__main__ as gen_main
import runtime.worlds
from data.gen import programs
from data.gen.domains import register_domains
from runtime.worlds import get_world


@pytest.fixture(scope="module", autouse=True)
def themes():
    """Register the generated domains for this module only. The theme set
    contains a `coursebuilder` theme that shadows the real-sessions eval
    world of the same name in `runtime.worlds.WORLDS`, so leaving it
    registered breaks whatever runs next (tests/test_real_suite.py)."""
    worlds = dict(runtime.worlds.WORLDS)
    profiles = dict(programs.PROFILES)
    register_domains("data/gen/themes")
    yield
    runtime.worlds.WORLDS.clear()
    runtime.worlds.WORLDS.update(worlds)
    programs.PROFILES.clear()
    programs.PROFILES.update(profiles)


def _sample(recipe, seed, **kw):
    """First sample of `recipe` at or after `seed`, as a built task."""
    level = kw.pop("level", 4)
    for s in range(seed, seed + 400):
        try:
            t = gen_main.gen_one(level, s, False, "template", **kw)
        except Exception:
            continue
        if t["provenance"]["recipe"] == recipe:
            return t
    raise AssertionError(f"no {recipe} sample in 400 seeds from {seed}")


def _program(task):
    return task["reference"]["segments"][0]


# ---------------------------------------------------------------- MOST/LEAST
def test_argmax_uses_most_not_the_loop():
    p = _program(_sample("argmax_count", 1000))
    assert "MOST " in p
    assert "FOREACH" not in p and "LET " not in p and "COUNT" not in p


def test_argmin_uses_least_with_a_candidate_list():
    p = _program(_sample("argmin_count", 1000))
    least = [l for l in p.splitlines() if l.startswith("LEAST ")]
    assert len(least) == 1
    # LEAST <reg> <field> <candidates> -> <reg>: four tokens before the arrow
    head = least[0].split("->")[0].split()
    assert len(head) == 4, least[0]
    assert head[3].startswith("r")


def test_argmin_answer_can_be_a_candidate_with_no_matches():
    """The point of the candidate list: "the fewest" is usually someone
    absent from the filtered list entirely (spec 0.4.0 §4)."""
    seen = []
    orig = programs._extreme_counts

    def spy(state, o, i, link, clauses):
        r = orig(state, o, i, link, clauses)
        seen.append(r)
        return r

    programs._extreme_counts = spy
    try:
        zero = 0
        for s in range(2000, 2200):
            seen.clear()
            try:
                t = gen_main.gen_one(4, s, False, "template")
            except Exception:
                continue
            if t["provenance"]["recipe"] != "argmin_count":
                continue
            over_all, _ = seen[-1]
            if min(over_all.values()) == 0:
                zero += 1
        assert zero > 0
    finally:
        programs._extreme_counts = orig


def test_extreme_winner_is_unique():
    """A tie has no single right answer in English; the sampler rejects it."""
    seen = []
    orig = programs._extreme_counts

    def spy(state, o, i, link, clauses):
        r = orig(state, o, i, link, clauses)
        seen.append(r)
        return r

    programs._extreme_counts = spy
    try:
        checked = 0
        for s in range(3000, 3120):
            seen.clear()
            try:
                t = gen_main.gen_one(4, s, False, "template")
            except Exception:
                continue
            recipe = t["provenance"]["recipe"]
            if recipe not in ("argmax_count", "argmin_count") or not seen:
                continue
            over_all, present = seen[-1]
            pool = over_all if recipe == "argmin_count" else present
            best = min(pool.values()) if recipe == "argmin_count" \
                else max(pool.values())
            assert sum(1 for v in pool.values() if v == best) == 1, t["id"]
            checked += 1
        assert checked > 5
    finally:
        programs._extreme_counts = orig


# -------------------------------------------------------------------- EMPTY
def test_checked_abort_guards_with_empty():
    t = _sample("direct", 4000, level=11)
    p = _program(t)
    if "FILTER" not in p:          # entities with no name field: bare ABORT
        pytest.skip("bare-abort flavor")
    assert "IF EMPTY r1" in p
    assert "COUNT" not in p
    assert "ABORT NOT_FOUND" in p


def test_parallel_count_branch_uses_empty():
    for s in range(5000, 5400):
        try:
            t = gen_main.gen_one(7, s, False, "template")
        except Exception:
            continue
        p = _program(t)
        if "IF " not in p:
            continue              # the FOREACH variant
        assert "IF NOT EMPTY" in p
        assert "COUNT" not in p
        return
    raise AssertionError("no count-variant parallel sample")


# -------------------------------------------------------------------- kinds
def test_generated_str_constants_declare_a_kind():
    checked = 0
    for s in range(6000, 6060):
        try:
            t = gen_main.gen_one(s % 12, s, False, "template",
                                 symbols="typed", enums=True, kinds=True)
        except Exception:
            continue
        for c in t["context"]["constants"]:
            if c["type"] == "STR":
                assert c.get("kind"), (t["id"], c)
                checked += 1
    assert checked > 20


def test_enum_constants_carry_the_field_they_belong_to():
    t = _sample("direct", 7000, level=11, symbols="typed", enums=True,
                kinds=True)
    enums = [c for c in t["context"]["constants"]
             if c.get("kind", "").startswith("enum:")]
    assert enums
    for c in enums:
        entity, field = c["kind"][5:].split(".")
        world = get_world(t["world"])
        assert c["value"] in world["enums"][(entity, field)]


def test_surface_flags_switch_the_symbol_table():
    for seed in range(8001, 8100):
        try:
            classic = gen_main.gen_one(4, seed, False, "template")
        except Exception:
            continue
        typed = gen_main.gen_one(4, seed, False, "template", symbols="typed",
                                 enums=True, kinds=True)
        break
    else:
        raise AssertionError("no L4 sample")
    assert all(c["sym"].startswith("C")
               for c in classic["context"]["constants"])
    assert all(c["sym"][0] in "SNBDI"
               for c in typed["context"]["constants"])
    # same task, same program shape -- only the symbols move
    assert classic["request"] == typed["request"]
    assert classic["symbols"] == "classic" and typed["symbols"] == "typed"
    assert typed["spec_version"] == "0.4.0"


def test_kinds_off_strips_name_and_text_but_keeps_enum_declarations():
    t = _sample("direct", 7000, level=11, symbols="typed", enums=True,
                kinds=False)
    kinds = {c.get("kind", "") for c in t["context"]["constants"]}
    assert not (kinds - {""} - {k for k in kinds if k.startswith("enum:")})


# ------------------------------------------------------- nothing else moved
def test_generation_still_covers_every_level():
    """Every level still produces a task: the generator resamples on
    SampleError (a world without the surface a recipe needs), so the check
    is that some seed works, not that every seed does."""
    for level in sorted(programs.RECIPES):
        for seed in range(9000, 9060):
            try:
                gen_main.gen_one(level, seed, False, "template",
                                 symbols="typed", enums=True, kinds=True)
                break
            except Exception:
                continue
        else:
            raise AssertionError(f"level {level} produced nothing")


# ------------------------------------------- reading a field off the winner
def test_argmax_question_refetches_the_winner():
    """R6 §0.2: 2,016 of 2,024 MOST rows pass the winner's id straight to a
    tool, so the model never learns that a *field* of the winner needs the
    record back. The question form does exactly that."""
    p = _program(_sample("argmax_which", 1000))
    lines = p.splitlines()
    most = next(i for i, l in enumerate(lines) if l.startswith("MOST "))
    assert lines[most + 1].startswith("FILTER ") and " EQ r" in lines[most + 1]
    assert lines[most + 2].startswith("FIRST ")
    assert lines[most + 3].startswith("GET ")
    assert lines[-1].startswith("RETURN ")


# ------------------------------------------------ an abort with no referent
def test_some_not_found_aborts_have_nothing_to_check_with():
    """The L11 exam's 30 tasks name a record whose name reached no constant,
    so the only faithful program is a bare abort. The corpus allocated the
    name almost every time (2,597 checked against 195 bare) and the model
    learned to reach for a referent it does not have."""
    bare = seen = 0
    for seed in range(3000, 3080):
        try:
            t = gen_main.gen_one(11, seed, False, "template")
        except Exception:
            continue
        seen += 1
        if _program(t).strip() == "ABORT NOT_FOUND":
            bare += 1
            assert not any(c.get("kind") == "name"
                           for c in t["context"]["constants"])
    assert seen >= 40 and bare / seen > 0.25, f"{bare}/{seen} bare"


# ------------------------------------------------------ IN (spec 0.6.0) L19
def test_parents_with_a_matching_child_bind_the_set_with_in():
    t = _sample("parents_with", 5000, level=19)
    p = _program(t)
    assert "MAP " in p and " IN r" in p
    # one pass over the parents, so a parent with three matching children is
    # messaged once - which the FOREACH-over-children form got wrong
    assert p.count("FOREACH") == 1


def test_parents_without_uses_a_negated_clause():
    p = _program(_sample("parents_without", 5000, level=19))
    assert "NOT " in p and " IN r" in p
