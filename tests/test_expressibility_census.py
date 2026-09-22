"""The expressibility census's two offline passes (plan step 2c).

The census produces the demand number the admission rule's first test asks
for (`spec/agent_core.md` §11), so what has to hold is that a probe fires on
wording only that candidate could answer and stays quiet on the wording that
looks like it but isn't — "the top 10 issues" is a list prefix, "in the last
24 hours" is a time window — and that the workaround detector recognises the
three-line join without calling every nested FOREACH one.
"""
import pytest

from core.ir import (Clause, Filter, Foreach, If, Pred, Program, Reg,
                     RegField, Select, Sort, Stop)
from harness.expressibility_census import (SCAFFOLD, label_request, walk,
                                           workarounds)


@pytest.mark.parametrize("request_text,candidate", [
    ("Give me the top 10 issues that operators see", "take"),
    ("Duplicate the first 2 elements in this lesson", "take"),
    ("show me the three most overdue invoices", "take"),
    ("list the distinct statuses on the board", "unique"),
    ("how many different assignees have open cards", "unique"),
    ("count the open tickets per customer", "group_join"),
    ("give me a breakdown of spend", "group_join"),
    ("archive them unless they are pinned and recent", "predicate_group"),
    ("flag neither the drafts nor the archived ones", "predicate_group"),
    ("do the same for Bob", "followup"),
    ("make it cartoon-style instead of realistic", "followup"),
    ("what is the average handling time", "aggregate_tool"),
])
def test_the_probes_fire_on_wording_only_that_candidate_answers(request_text,
                                                               candidate):
    assert candidate in label_request(request_text), request_text


@pytest.mark.parametrize("request_text,candidate", [
    # a time window, not the prefix of a sorted list
    ("show me everything from the last 24 hours", "take"),
    ("what changed in the first 30 minutes", "take"),
    # FIRST already takes one element
    ("open the first card", "take"),
    # a plain filter, not a table keyed by anything
    ("archive each card that is done", "group_join"),
    # "I mean" is not an average
    ("I mean lesson titles only", "aggregate_tool"),
])
def test_the_probes_stay_quiet_on_wording_that_looks_like_it(request_text,
                                                            candidate):
    assert candidate not in label_request(request_text), request_text


def test_the_product_s_own_continuity_line_is_recognised():
    """It reads exactly like a human follow-up and would otherwise carry the
    followup count by itself (70 of 1,544 real requests)."""
    line = ("Take font size on accordian titles to 22pt. Earlier modules in "
            "this run updated 6 titles. Apply the same intent and tone to "
            "this module.")
    assert SCAFFOLD.search(line)
    assert "followup" in label_request(line)
    assert not SCAFFOLD.search("do the same for Bob")


def _join_program():
    """The ~3-line join: iterate the outer list, filter the inner one by the
    loop variable (results/R1.md's Tier C entry)."""
    inner = Filter(
        src=Reg(1),
        pred=Pred(clauses=(Clause(False, "F4", "EQ", RegField(3, "F2")),),
                  ops=()),
        dst=Reg(4))
    return Program(body=[Foreach(src=Reg(2), var=Reg(3), body=[inner]),
                         Stop()])


def test_the_join_workaround_is_recognised():
    assert workarounds(_join_program()) == ["group_join"]


def test_a_nested_foreach_whose_filter_ignores_the_loop_variable_is_not_a_join():
    inner = Filter(src=Reg(1),
                   pred=Pred(clauses=(Clause(False, "F4", "EQ", Reg(9)),),
                             ops=()),
                   dst=Reg(4))
    prog = Program(body=[Foreach(src=Reg(2), var=Reg(3), body=[inner])])
    assert workarounds(prog) == []


def test_two_selects_off_one_sorted_register_is_a_hand_written_prefix():
    prog = Program(body=[
        Sort(src=Reg(0), field="F1", dir="DESC", dst=Reg(1)),
        Select(src=Reg(1), idx=0, dst=Reg(2)),
        Select(src=Reg(1), idx=1, dst=Reg(3)),
    ])
    assert workarounds(prog) == ["take"]


def test_one_select_off_a_sorted_register_is_just_an_ordinal():
    prog = Program(body=[
        Sort(src=Reg(0), field="F1", dir="ASC", dst=Reg(1)),
        Select(src=Reg(1), idx=1, dst=Reg(2)),
    ])
    assert workarounds(prog) == []


def test_walk_descends_into_every_block_carrying_instruction():
    deep = Stop()
    prog = Program(body=[If(cond=Pred(clauses=(), ops=()),
                            then=[Foreach(src=Reg(0), var=Reg(1),
                                          body=[deep])],
                            els=None)])
    assert deep in list(walk(prog.body))
