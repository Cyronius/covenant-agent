"""Typechecker + effect checker: every structured diagnostic fires."""
import random

from core.pipeline import build
from harness.context import build_context
from harness.authoring import resolve
from runtime.worlds import get_world


def _ctx():
    world = get_world("kanban")
    constants = [
        {"type": "ID:card", "value": "card_1", "desc": "card 1"},
        {"type": "STR", "value": "done", "desc": "done"},
        {"type": "BOOL", "value": False, "desc": "false"},
    ]
    return build_context(world, constants, random.Random(5))[0]


def _codes(src, ctx=None):
    ctx = ctx or _ctx()
    res = build(resolve(src, ctx), ctx)
    return [d.code for d in res.diagnostics], res


def test_clean_program():
    codes, res = _codes(
        "CALL @list_cards -> r0\n"
        "FILTER r0 @card.status EQ $1 -> r1\n"
        "FOREACH r1 -> r2\n"
        "  CALL @archive_card r2 -> r3\n"
        "STOP\n")
    assert codes == [] and res.compile_ok


def test_unbound_register():
    codes, _ = _codes("COUNT r5 -> r0\nSTOP\n")
    assert codes == ["UNBOUND"]


def test_unbound_constant():
    codes, _ = _codes("LET C9 -> r0\nSTOP\n")
    assert codes == ["UNBOUND"]


def test_unknown_tool():
    codes, _ = _codes("CALL T99 -> r0\nSTOP\n")
    assert codes == ["UNKNOWN_TOOL"]


def test_unknown_field():
    ctx = _ctx()
    # a user-entity field used against a card list element
    codes, _ = _codes(
        "CALL @list_cards -> r0\n"
        "FILTER r0 @user.email EQ $1 -> r1\n"
        "STOP\n", ctx)
    assert codes == ["UNKNOWN_FIELD"]


def test_missing_arg():
    codes, _ = _codes("CALL @archive_card -> r0\nSTOP\n")
    assert codes == ["MISSING_ARG"]


def test_type_error_arg():
    codes, _ = _codes("CALL @archive_card $1 -> r0\nSTOP\n")  # STR into ID:card
    assert codes == ["TYPE_ERROR"]


def test_type_error_cmp():
    codes, _ = _codes(
        "CALL @get_card $0 -> r0\n"
        "GET r0.@card.title -> r1\n"
        "IF r1 LT NOW\n"
        "  STOP\n"
        "STOP\n")
    assert "TYPE_ERROR" in codes


def test_unreachable():
    codes, _ = _codes("STOP\nCALL @list_cards -> r0\n")
    assert codes == ["UNREACHABLE"]


def test_unreachable_after_pause():
    codes, _ = _codes("CALL @list_cards -> r0\nPAUSE\nSTOP\n")
    assert codes == ["UNREACHABLE"]


def test_effect_undeclared():
    codes, _ = _codes("EFFECTS READ\nCALL @delete_card $0\nSTOP\n")
    assert codes == ["EFFECT_UNDECLARED"]


def test_effects_computed():
    _, res = _codes("CALL @delete_card $0\nCALL @list_cards -> r0\nSTOP\n")
    assert res.static_effects == ["DELETE", "READ"]


def test_if_branch_binding_merge():
    # r1 bound in only one arm: reading it after the IF is UNBOUND
    codes, _ = _codes(
        "CALL @get_card $0 -> r0\n"
        "IF r0.@card.archived EQ $2\n"
        "  LET $1 -> r1\n"
        "ELSE\n"
        "  CALL @list_cards -> r2\n"
        "COUNT r2 -> r3\n"
        "STOP\n")
    assert "UNBOUND" in codes


def test_parallel_no_intra_block_reads():
    codes, _ = _codes(
        "PARALLEL\n"
        "  CALL @list_cards -> r0\n"
        "  CALL @get_card r0.@card.id -> r1\n"
        "STOP\n")
    assert "UNBOUND" in codes


def test_pause_env_reported():
    _, res = _codes("CALL @list_cards -> r0\nPAUSE\n")
    assert res.pause_envs and res.pause_envs[0]["r0"] == "LIST OBJ:card"


def test_unreachable_after_abort():
    codes, _ = _codes("ABORT UNSUPPORTED\nCALL @list_cards -> r0\n")
    assert codes == ["UNREACHABLE"]


def test_abort_inside_if_is_a_valid_early_exit():
    codes, res = _codes(
        "CALL @list_cards -> r0\n"
        "FIRST r0 -> r1\n"
        "IF r1 EQ NULL\n"
        "  ABORT NOT_FOUND\n"
        "CALL @delete_card r1\n"
        "STOP\n")
    assert codes == []
    assert 'rt.abort("NOT_FOUND", [])' in res.js


def _fmt_ctx():
    world = get_world("kanban")
    constants = [
        {"type": "ID:card", "value": "card_1", "desc": "card 1"},
        {"type": "STR", "value": "Copy of {0}", "desc": "title template"},
        {"type": "STR", "value": "{0} due {1}", "desc": "two-slot template"},
        {"type": "INT", "value": 3, "desc": "three"},
    ]
    return build_context(world, constants, random.Random(7))[0]


def test_format_slot_count_and_result_type():
    ctx = _fmt_ctx()
    codes, res = _codes(
        "CALL @get_card $0 -> r0\n"
        "FORMAT $1 r0.@card.title -> r1\n"
        "CALL @create_card r1 r0.@card.due r0.@card.assignee -> r2\n"
        "STOP\n", ctx)
    assert codes == []
    assert 'rt.format(rt.constant("C1")' in res.js
    codes, _ = _codes("CALL @get_card $0 -> r0\nFORMAT $1 r0.@card.title r0.@card.due -> r1\nSTOP\n", ctx)
    assert codes == ["TYPE_ERROR"]  # one slot, two operands
    codes, _ = _codes("FORMAT $3 -> r1\nSTOP\n", ctx)
    assert codes == ["TYPE_ERROR"]  # template must be STR
    codes, _ = _codes("CALL @get_card $0 -> r0\nFORMAT $2 r0.@card.title r0 -> r1\nSTOP\n", ctx)
    assert codes == ["TYPE_ERROR"]  # OBJ operand not renderable


def test_null_required_arg():
    """An explicit NULL where the schema wants a value is a missing argument,
    not a value of every type (results/RPG.md: the grammar offered NULL as
    the only operand for a slot with no matching constant, and 25 of 66 RPG
    calls were NULL calls that could only fail in the sandbox)."""
    codes, _ = _codes("CALL @archive_card NULL -> r0\nSTOP\n")
    assert codes == ["MISSING_ARG"]


def test_null_optional_arg_is_fine():
    world = get_world("rpg")
    ctx = build_context(world, [{"type": "ID:item", "value": "item_1",
                                 "desc": "a potion"}], random.Random(3))[0]
    codes, res = _codes("CALL @use_item $0 NULL\nSTOP\n", ctx)
    assert codes == [] and res.compile_ok


def test_filter_field_on_the_right_resolves_against_the_element():
    # spec 0.5.0: both fields of a FILTER clause belong to the element's entity
    codes, res = _codes(
        "CALL @list_cards -> r0\n"
        "FILTER r0 @card.due LT @card.due AND @card.title EQ @card.status -> r1\n"
        "COUNT r1 -> r2\n"
        "STOP\n")
    assert codes == [] and res.compile_ok
    assert 'rt.fld(_x, "' in res.js and res.js.count("rt.fld(_x") == 4


def test_filter_field_on_the_right_wrong_entity():
    codes, _ = _codes(
        "CALL @list_cards -> r0\n"
        "FILTER r0 @card.due LT @user.email -> r1\n"
        "STOP\n")
    assert codes == ["UNKNOWN_FIELD"]


def test_filter_field_on_the_right_type_error():
    codes, _ = _codes(
        "CALL @list_cards -> r0\n"
        "FILTER r0 @card.due LT @card.title -> r1\n"  # TIME vs STR
        "STOP\n")
    assert codes == ["TYPE_ERROR"]
