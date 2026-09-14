"""Parser: structure, block discipline, structured errors (never exceptions)."""
from core.parser import parse
from core import ir


def _diag_codes(text):
    prog, diags = parse(text)
    return prog, [d.code for d in diags]


def test_minimal_program():
    prog, diags = parse("CALL T0 -> r0\nSTOP\n")
    assert diags == []
    assert isinstance(prog.body[0], ir.Call)
    assert prog.body[0].dst == ir.Reg(0)
    assert prog.effects_decl is None


def test_effects_header():
    prog, diags = parse("EFFECTS READ WRITE\nSTOP\n")
    assert diags == []
    assert prog.effects_decl == ["READ", "WRITE"]


def test_bad_effect_name():
    prog, codes = _diag_codes("EFFECTS READ FROB\nSTOP\n")
    assert prog is None and codes == ["PARSE_ERROR"]


def test_nested_blocks():
    text = ("FOREACH r0 -> r1\n"
            "  IF r1.F0 EQ C0\n"
            "    CALL T1 r1\n"
            "STOP\n")
    prog, diags = parse(text)
    assert diags == []
    fe = prog.body[0]
    assert isinstance(fe, ir.Foreach)
    assert isinstance(fe.body[0], ir.If)


def test_else_binds_to_if():
    text = ("IF r0 EQ C0\n  CALL T0\nELSE\n  CALL T1\nSTOP\n")
    prog, diags = parse(text)
    assert diags == []
    assert prog.body[0].els is not None


def test_else_without_if():
    prog, codes = _diag_codes("ELSE\n  CALL T0\nSTOP\n")
    assert prog is None and codes == ["PARSE_ERROR"]


def test_parallel_rejects_non_call():
    prog, codes = _diag_codes("PARALLEL\n  LET C0 -> r0\nSTOP\n")
    assert prog is None and codes == ["PARSE_ERROR"]


def test_tabs_illegal():
    prog, codes = _diag_codes("CALL T0\n\tCALL T1\n")
    assert prog is None and codes == ["PARSE_ERROR"]


def test_odd_indent_illegal():
    prog, codes = _diag_codes("FOREACH r0 -> r1\n   CALL T0\n")
    assert prog is None and codes == ["PARSE_ERROR"]


def test_block_head_requires_body():
    prog, codes = _diag_codes("FOREACH r0 -> r1\nSTOP\n")
    assert prog is None and codes == ["PARSE_ERROR"]


def test_register_out_of_range():
    prog, codes = _diag_codes("LET C0 -> r16\n")
    assert prog is None and codes == ["PARSE_ERROR"]


def test_try_retry():
    prog, diags = parse("TRY RETRY 3 -> r0\n  CALL T0\nSTOP\n")
    assert diags == []
    assert prog.body[0].retry == 3


def test_comments_and_blank_lines():
    prog, diags = parse("# header\n\nCALL T0 -> r0  # trailing\nSTOP\n")
    assert diags == []
    assert len(prog.body) == 2


def test_pred_parsing():
    prog, diags = parse("FILTER r0 F1 EQ C0 AND NOT F2 LT NOW OR F3 CONTAINS C1 -> r1\nSTOP\n")
    assert diags == []
    pred = prog.body[0].pred
    assert pred.ops == ("AND", "OR")
    assert pred.clauses[1].neg


def test_abort_parses_with_reason():
    prog, diags = parse("ABORT NOT_FOUND\n")
    assert diags == []
    assert isinstance(prog.body[0], ir.Abort)
    assert prog.body[0].reason == "NOT_FOUND"


def test_abort_rejects_unknown_reason():
    prog, codes = _diag_codes("ABORT BECAUSE\n")
    assert prog is None and codes == ["PARSE_ERROR"]
    prog, codes = _diag_codes("ABORT\n")
    assert prog is None and codes == ["PARSE_ERROR"]


def test_format_parses():
    prog, diags = parse("FORMAT C1 r0.F2 r3 -> r4\n")
    assert diags == []
    f = prog.body[0]
    assert isinstance(f, ir.Format) and f.template == "C1" and len(f.ops) == 2
    assert f.dst == ir.Reg(4)


def test_format_needs_template_and_arrow():
    prog, codes = _diag_codes("FORMAT r0 -> r1\n")
    assert prog is None and codes == ["PARSE_ERROR"]
    prog, codes = _diag_codes("FORMAT C1 r0\n")
    assert prog is None and codes == ["PARSE_ERROR"]


def test_filter_clause_takes_a_field_on_the_right():
    # spec 0.5.0: a FILTER clause may compare two fields of the element;
    # an IF condition may not name a bare field
    prog, diags = parse("FILTER r0 F1 LT F2 AND F3 EQ C0 -> r1\nSTOP\n")
    assert diags == []
    cl = prog.body[0].pred.clauses
    assert cl[0].right == ir.ElemField("F2")
    assert isinstance(cl[1].right, ir.Const)
    assert str(cl[0]) == "F1 LT F2"
    _, codes = _diag_codes("IF r0.F1 LT F2\n  STOP\nSTOP\n")
    assert codes == ["PARSE_ERROR"]


def test_in_parses_with_a_register_on_the_right():
    prog, diags = parse("FILTER r0 F1 IN r2 -> r1\nSTOP\n")
    assert diags == []
    cl = prog.body[0].pred.clauses[0]
    assert (cl.left, cl.cmp, str(cl.right)) == ("F1", "IN", "r2")
