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
