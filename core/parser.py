"""Agent Core parser: text -> AST (spec §3).

parse(text) returns (Program | None, [Diagnostic]). It never raises on bad
input; every syntax problem is a structured PARSE_ERROR. Parsing stops at the
first syntax error (a partial AST is not useful downstream).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

from . import diagnostics as dg
from .ir import (ABORT_REASONS, CMPS, EFFECTS, NUM_REGISTERS, Abort, Call,
                 Clause, Const, Count,
                 Filter, First, Foreach, Format, Get, If, IntLit, Let, MapF,
                 Now, Null,
                 Parallel, Pause, Pred, Program, Reg, RegField, Return, Select,
                 SetF, Sort, Stop, Try)

_REG_RE = re.compile(r"^r(\d{1,2})$")
_REGFIELD_RE = re.compile(r"^r(\d{1,2})\.(F\d+)$")
_TOOL_RE = re.compile(r"^T\d+$")
_FIELD_RE = re.compile(r"^F\d+$")
_CONST_RE = re.compile(r"^C\d+$")
_INT_RE = re.compile(r"^\d+$")

BLOCK_HEADS = ("FOREACH", "IF", "ELSE", "PARALLEL", "TRY")


class _ParseFail(Exception):
    def __init__(self, diag: dg.Diagnostic):
        self.diag = diag


def _reg(tok: str, line: int) -> Reg:
    m = _REG_RE.match(tok)
    if not m or int(m.group(1)) >= NUM_REGISTERS:
        raise _ParseFail(dg.parse_error(line, f"expected register, got {tok!r}"))
    return Reg(int(m.group(1)))


def _field(tok: str, line: int) -> str:
    if not _FIELD_RE.match(tok):
        raise _ParseFail(dg.parse_error(line, f"expected field symbol, got {tok!r}"))
    return tok


def _operand(tok: str, line: int):
    m = _REGFIELD_RE.match(tok)
    if m:
        n = int(m.group(1))
        if n >= NUM_REGISTERS:
            raise _ParseFail(dg.parse_error(line, f"bad register in {tok!r}"))
        return RegField(n, m.group(2))
    if _REG_RE.match(tok):
        return _reg(tok, line)
    if _CONST_RE.match(tok):
        return Const(tok)
    if tok == "NOW":
        return Now()
    if tok == "NULL":
        return Null()
    if _INT_RE.match(tok):
        return IntLit(int(tok))
    raise _ParseFail(dg.parse_error(line, f"expected operand, got {tok!r}"))


def _arrow_dst(toks: List[str], line: int) -> Tuple[List[str], Reg]:
    """Split off a trailing '-> rN'. Errors if absent."""
    if len(toks) < 2 or toks[-2] != "->":
        raise _ParseFail(dg.parse_error(line, "expected '-> rN'"))
    return toks[:-2], _reg(toks[-1], line)


def _parse_pred(toks: List[str], line: int, field_left: bool) -> Pred:
    """Parse `clause (AND|OR clause)*`. field_left selects FILTER-style
    (left is a field symbol) vs IF-style (left is an operand)."""
    clauses, ops = [], []
    i = 0
    while True:
        neg = False
        if i < len(toks) and toks[i] == "NOT":
            neg = True
            i += 1
        if i >= len(toks):
            raise _ParseFail(dg.parse_error(line, "expected comparison clause"))
        left = _field(toks[i], line) if field_left else _operand(toks[i], line)
        i += 1
        if i >= len(toks) or toks[i] not in CMPS:
            got = toks[i] if i < len(toks) else "end of line"
            raise _ParseFail(dg.parse_error(line, f"expected comparator, got {got!r}"))
        cmp = toks[i]
        i += 1
        if i >= len(toks):
            raise _ParseFail(dg.parse_error(line, "expected right operand"))
        right = _operand(toks[i], line)
        i += 1
        clauses.append(Clause(neg, left, cmp, right))
        if i == len(toks):
            return Pred(tuple(clauses), tuple(ops))
        if toks[i] not in ("AND", "OR"):
            raise _ParseFail(dg.parse_error(line, f"expected AND/OR, got {toks[i]!r}"))
        ops.append(toks[i])
        i += 1


def _parse_instr(toks: List[str], line: int):
    op = toks[0]
    rest = toks[1:]
    if op == "LET":
        rest, dst = _arrow_dst(rest, line)
        if len(rest) != 1:
            raise _ParseFail(dg.parse_error(line, "LET takes one operand"))
        return Let(_operand(rest[0], line), dst, line=line)
    if op == "GET":
        rest, dst = _arrow_dst(rest, line)
        if len(rest) != 1:
            raise _ParseFail(dg.parse_error(line, "GET takes rN.Fk"))
        src = _operand(rest[0], line)
        if not isinstance(src, RegField):
            raise _ParseFail(dg.parse_error(line, "GET takes rN.Fk"))
        return Get(Reg(src.n), src.field, dst, line=line)
    if op == "SET":
        rest, dst = _arrow_dst(rest, line)
        if len(rest) != 3:
            raise _ParseFail(dg.parse_error(line, "SET takes rN Fk operand"))
        return SetF(_reg(rest[0], line), _field(rest[1], line),
                    _operand(rest[2], line), dst, line=line)
    if op == "CALL":
        dst = None
        if "->" in rest:
            rest, dst = _arrow_dst(rest, line)
        if not rest or not _TOOL_RE.match(rest[0]):
            got = rest[0] if rest else "end of line"
            raise _ParseFail(dg.parse_error(line, f"expected tool symbol, got {got!r}"))
        args = tuple(_operand(t, line) for t in rest[1:])
        return Call(rest[0], args, dst, line=line)
    if op == "FORMAT":
        rest, dst = _arrow_dst(rest, line)
        if not rest or not _CONST_RE.match(rest[0]):
            raise _ParseFail(dg.parse_error(line, "FORMAT takes Ck template then operands"))
        return Format(rest[0], tuple(_operand(t, line) for t in rest[1:]), dst, line=line)
    if op == "FILTER":
        rest, dst = _arrow_dst(rest, line)
        if len(rest) < 4:
            raise _ParseFail(dg.parse_error(line, "FILTER takes rN predicate"))
        src = _reg(rest[0], line)
        return Filter(src, _parse_pred(rest[1:], line, field_left=True), dst, line=line)
    if op == "MAP":
        rest, dst = _arrow_dst(rest, line)
        if len(rest) != 2:
            raise _ParseFail(dg.parse_error(line, "MAP takes rN Fk"))
        return MapF(_reg(rest[0], line), _field(rest[1], line), dst, line=line)
    if op == "COUNT":
        rest, dst = _arrow_dst(rest, line)
        if len(rest) != 1:
            raise _ParseFail(dg.parse_error(line, "COUNT takes rN"))
        return Count(_reg(rest[0], line), dst, line=line)
    if op == "SORT":
        rest, dst = _arrow_dst(rest, line)
        if len(rest) != 3 or rest[2] not in ("ASC", "DESC"):
            raise _ParseFail(dg.parse_error(line, "SORT takes rN Fk ASC|DESC"))
        return Sort(_reg(rest[0], line), _field(rest[1], line), rest[2], dst, line=line)
    if op == "SELECT":
        rest, dst = _arrow_dst(rest, line)
        if len(rest) != 2:
            raise _ParseFail(dg.parse_error(line, "SELECT takes rN index"))
        return Select(_reg(rest[0], line), _operand(rest[1], line), dst, line=line)
    if op == "FIRST":
        rest, dst = _arrow_dst(rest, line)
        if len(rest) != 1:
            raise _ParseFail(dg.parse_error(line, "FIRST takes rN"))
        return First(_reg(rest[0], line), dst, line=line)
    if op == "FOREACH":
        rest, var = _arrow_dst(rest, line)
        if len(rest) != 1:
            raise _ParseFail(dg.parse_error(line, "FOREACH takes rN -> rM"))
        return Foreach(_reg(rest[0], line), var, [], line=line)
    if op == "IF":
        if not rest:
            raise _ParseFail(dg.parse_error(line, "IF needs a condition"))
        return If(_parse_pred(rest, line, field_left=False), [], None, line=line)
    if op == "ELSE":
        if rest:
            raise _ParseFail(dg.parse_error(line, "ELSE takes no arguments"))
        return If(None, [], None, line=line)  # placeholder; folded into prior IF
    if op == "PARALLEL":
        if rest:
            raise _ParseFail(dg.parse_error(line, "PARALLEL takes no arguments"))
        return Parallel([], line=line)
    if op == "TRY":
        retry = 0
        if rest[:1] == ["RETRY"]:
            if len(rest) < 2 or not _INT_RE.match(rest[1]):
                raise _ParseFail(dg.parse_error(line, "RETRY needs an integer count"))
            retry = int(rest[1])
            rest = rest[2:]
        rest, dst = _arrow_dst(rest, line)
        if rest:
            raise _ParseFail(dg.parse_error(line, "TRY takes [RETRY n] -> rN"))
        return Try(retry, dst, [], line=line)
    if op == "RETURN":
        if len(rest) != 1:
            raise _ParseFail(dg.parse_error(line, "RETURN takes one operand"))
        return Return(_operand(rest[0], line), line=line)
    if op == "STOP":
        if rest:
            raise _ParseFail(dg.parse_error(line, "STOP takes no arguments"))
        return Stop(line=line)
    if op == "PAUSE":
        if rest:
            raise _ParseFail(dg.parse_error(line, "PAUSE takes no arguments"))
        return Pause(line=line)
    if op == "ABORT":
        if len(rest) != 1 or rest[0] not in ABORT_REASONS:
            raise _ParseFail(dg.parse_error(
                line, "ABORT takes one reason: " + "|".join(ABORT_REASONS)))
        return Abort(rest[0], line=line)
    raise _ParseFail(dg.parse_error(line, f"unknown instruction {op!r}"))


def _lines(text: str):
    """Yield (lineno, indent_level, tokens) for meaningful lines."""
    out = []
    for i, raw in enumerate(text.split("\n"), start=1):
        if "\t" in raw:
            raise _ParseFail(dg.parse_error(i, "tabs are illegal; indent with 2 spaces"))
        code = raw.split("#", 1)[0].rstrip()
        if not code.strip():
            continue
        stripped = code.lstrip(" ")
        n_spaces = len(code) - len(stripped)
        if n_spaces % 2 != 0:
            raise _ParseFail(dg.parse_error(i, "indentation must be a multiple of 2 spaces"))
        out.append((i, n_spaces // 2, stripped.split(" ")))
    return out


def _parse_block(lines, pos: int, level: int):
    """Parse lines at exactly `level`; recurse into deeper blocks."""
    body = []
    while pos < len(lines):
        lineno, lvl, toks = lines[pos]
        if lvl < level:
            break
        if lvl > level:
            raise _ParseFail(dg.parse_error(lineno, "unexpected indent"))
        instr = _parse_instr(toks, lineno)
        pos += 1
        is_else = isinstance(instr, If) and instr.cond is None
        if toks[0] in BLOCK_HEADS:
            child, pos = _parse_block(lines, pos, level + 1)
            if not child:
                raise _ParseFail(dg.parse_error(lineno, f"{toks[0]} requires an indented body"))
            if is_else:
                prev = body[-1] if body else None
                if not (isinstance(prev, If) and prev.cond is not None and prev.els is None):
                    raise _ParseFail(dg.parse_error(lineno, "ELSE without matching IF"))
                prev.els = child
                continue
            if isinstance(instr, Foreach):
                instr.body = child
            elif isinstance(instr, If):
                instr.then = child
            elif isinstance(instr, Try):
                instr.body = child
            elif isinstance(instr, Parallel):
                for c in child:
                    if not isinstance(c, Call):
                        raise _ParseFail(dg.parse_error(
                            c.line, "PARALLEL body must contain only CALL lines"))
                instr.calls = child
        body.append(instr)
    return body, pos


def parse(text: str) -> Tuple[Optional[Program], List[dg.Diagnostic]]:
    try:
        lines = _lines(text)
        if not lines:
            return None, [dg.parse_error(1, "empty program")]
        effects_decl = None
        if lines[0][2][0] == "EFFECTS":
            lineno, lvl, toks = lines[0]
            if lvl != 0:
                raise _ParseFail(dg.parse_error(lineno, "EFFECTS must be unindented"))
            decl = toks[1:]
            if not decl:
                raise _ParseFail(dg.parse_error(lineno, "EFFECTS needs at least one effect"))
            for e in decl:
                if e not in EFFECTS:
                    raise _ParseFail(dg.parse_error(lineno, f"unknown effect {e!r}"))
            effects_decl = decl
            lines = lines[1:]
            if not lines:
                raise _ParseFail(dg.parse_error(lineno, "program has no instructions"))
        body, pos = _parse_block(lines, 0, 0)
        if pos != len(lines):
            raise _ParseFail(dg.parse_error(lines[pos][0], "unexpected indent"))
        return Program(effects_decl, body), []
    except _ParseFail as e:
        return None, [e.diag]
