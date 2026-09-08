"""Per-task GBNF: enumerate the symbols a prompt actually declares.

`baselines/qwen/agent_core.gbnf` spells every symbol as `"F" num` with
`num ::= [0-9] [0-9]?`. That caps decodable symbols at 99. Crowded contexts
declare 150+ field symbols, so on those tasks the correct answer is
*unspellable*: llama.cpp masks the third digit and the sampler commits to a
two-digit prefix (`F14` for `F142`), which the typechecker then reports as
`UNKNOWN_FIELD`. 122 of 300 `e_crowded_v2` tasks need an `F>=100` and all 122
failed; the other 178 scored 97.2%. See results/S2.md §S3.

This module builds the grammar per task instead: `tool`/`field`/`const` are
replaced by a digit trie over exactly the symbols that task's context
declares. Nothing else changes. Two faults die at once — the unreachable tail
above 99, and undeclared-but-in-range symbols, which can no longer be decoded
at all. This is PLAN.md §5's condition C4 restricted to what a context-free
grammar can see (the symbol table); it cannot know a register's entity, so
per-entity field validity stays the typechecker's job.

  from harness.task_grammar import grammar_for_task
  gbnf = grammar_for_task(task)          # task dict, as stored in the suites
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
BASE_GRAMMAR = ROOT / "baselines" / "qwen" / "agent_core.gbnf"

# Used when a context declares none of a kind (a Level 0 task may have no
# constants): fall back to the base file's open form rather than emit a rule
# that nothing can satisfy.
_OPEN_NUM = "[0-9] [0-9]? [0-9]?"


# -- digit trie ---------------------------------------------------------

def _trie_insert(node: dict, s: str) -> None:
    for ch in s:
        node = node.setdefault("kids", {}).setdefault(ch, {})
    node["end"] = True


def _trie_render(node: dict) -> str:
    alts = []
    for ch in sorted(node.get("kids", {})):
        kid = node["kids"][ch]
        if not kid.get("kids"):
            alts.append('"%s"' % ch)
        else:
            inner = "(%s)" % _trie_render(kid)
            alts.append('"%s" %s?' % (ch, inner) if kid.get("end")
                        else '"%s" %s' % (ch, inner))
    return " | ".join(alts)


def digit_trie_expr(nums: Iterable[int]) -> str:
    """A GBNF expression matching exactly the given numbers, as a trie.

    Flat alternation over 150+ literals leaves that many grammar stacks live
    at every field position; a trie keeps it to the branching factor (<=10).
    """
    nums = sorted({int(n) for n in nums})
    if not nums:
        return _OPEN_NUM
    root: dict = {}
    for n in nums:
        _trie_insert(root, str(n))
    return _trie_render(root)


# -- grammar assembly ---------------------------------------------------

def _nums(syms: Iterable[str]) -> List[int]:
    return [int(s[1:]) for s in syms]


def const_expr(consts: Iterable[str]) -> str:
    """A GBNF expression over constant symbols, one digit trie per letter:
    `"C" (...)` for a 0.3.x table, `"S" (...) | "I" (...)` for 0.4.0
    typed letters (spec §1). An empty table keeps the base file's open C."""
    by_letter: Dict[str, List[int]] = {}
    for s in consts:
        by_letter.setdefault(s[0], []).append(int(s[1:]))
    if not by_letter:
        return '"C" (%s)' % _OPEN_NUM
    return " | ".join('"%s" (%s)' % (L, digit_trie_expr(ns))
                      for L, ns in sorted(by_letter.items()))


def symbol_rules(tools: Iterable[str], fields: Iterable[str],
                 consts: Iterable[str]) -> Dict[str, str]:
    """{rule name -> right-hand side} for the three symbol rules."""
    return {
        "tool": '"T" (%s)' % digit_trie_expr(_nums(tools)),
        "field": '"F" (%s)' % digit_trie_expr(_nums(fields)),
        "const": const_expr(consts),
    }


def _replace_rules(base: str, rules: Dict[str, str]) -> str:
    out = base
    for name, rhs in rules.items():
        pat = re.compile(r"^%s\s*::=.*$" % name, re.MULTILINE)
        if not pat.search(out):
            raise ValueError("base grammar has no `%s` rule" % name)
        out = pat.sub(lambda m, r=rhs, n=name: "%s ::= %s" % (n, r),
                      out, count=1)
    return out


def load_base(path: Optional[Path] = None) -> str:
    return (path or BASE_GRAMMAR).read_text()


def grammar_for_symbols(tools: Iterable[str], fields: Iterable[str],
                        consts: Iterable[str], base: Optional[str] = None
                        ) -> str:
    return _replace_rules(base if base is not None else load_base(),
                          symbol_rules(tools, fields, consts))


def without_stdlib(grammar: str) -> str:
    """The 0.3.x surface: no MOST/LEAST, no EMPTY. The control arm of the
    spec 0.4.0 step-0b A/B (`run_a.py --no-stdlib`)."""
    out = grammar.replace(" | mostin", "")
    out = out.replace('(operand " " cmp " " operand | "EMPTY " reg)',
                      'operand " " cmp " " operand')
    return out


def grammar_for_task(task: dict, base: Optional[str] = None,
                     typed: bool = False, stdlib: bool = True,
                     kinds: bool = False) -> str:
    """`task` as the suites store it: task['context'] holds the symbol table.
    `typed` adds the per-tool typed-slot CALL rules (PLAN.md §5 C4);
    `stdlib=False` removes the 0.4.0 instructions (MOST/LEAST/EMPTY);
    `kinds` (spec 0.4.0 §2.2, implies typed) makes FILTER clauses, SET and
    CALL slots kind-aware: an enum field admits only its enum constants and
    a plain STR slot admits no enum constant."""
    ctx = task["context"]
    out = grammar_for_symbols(
        [t["sym"] for t in ctx.get("tools", [])],
        [f["sym"] for f in ctx.get("fields", [])],
        [c["sym"] for c in ctx.get("constants", [])],
        base=base)
    if typed or kinds:
        call_rhs, extra = typed_call_rules(task, kinds=kinds)
        out = _replace_rules(out, {"call": call_rhs}) + "\n" + extra
    if kinds:
        clause_rhs, set_rhs, extra = kind_field_rules(task)
        out = _replace_rules(out, {"clause": clause_rhs, "set": set_rhs}) + "\n" + extra
    if not stdlib:
        out = without_stdlib(out)
    return out


def grammar_for_context(ctx, base: Optional[str] = None) -> str:
    """`ctx` is a core.ir.TaskContext (harness.context.build_context)."""
    return grammar_for_symbols(ctx.tools, ctx.fields, ctx.constants, base=base)


def symbol_signature(task: dict) -> tuple:
    """Cache key: two tasks with the same symbol table share a grammar."""
    ctx = task["context"]
    return (tuple(sorted(_nums(t["sym"] for t in ctx.get("tools", [])))),
            tuple(sorted(_nums(f["sym"] for f in ctx.get("fields", [])))),
            tuple(sorted(c["sym"] for c in ctx.get("constants", []))))


# -- verification -------------------------------------------------------

def accepts(expr: str, s: str) -> bool:
    """Does `expr` -- the subset of GBNF digit_trie_expr emits (string
    literals, `|`, grouping, `?`) -- match `s` exactly? Lets a test assert
    against the emitted grammar text rather than the builder's internals."""
    return any(end == len(s) for end in _match_alt(expr.strip(), s, 0))


def _split_top(expr: str) -> List[str]:
    parts, depth, cur = [], 0, ""
    for ch in expr:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "|" and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += ch
    parts.append(cur)
    return [p.strip() for p in parts]


def _match_alt(expr: str, s: str, pos: int) -> List[int]:
    ends: List[int] = []
    for alt in _split_top(expr):
        ends.extend(_match_seq(alt, s, pos))
    return ends


def _tokens(seq: str) -> List[tuple]:
    """[(kind, body, optional)] for one alternative; kind is 'lit' or 'grp'."""
    out, i = [], 0
    while i < len(seq):
        ch = seq[i]
        if ch.isspace():
            i += 1
        elif ch == '"':
            j = seq.index('"', i + 1)
            out.append(("lit", seq[i + 1:j], False))
            i = j + 1
        elif ch == "(":
            depth, j = 1, i + 1
            while depth:
                if seq[j] == "(":
                    depth += 1
                elif seq[j] == ")":
                    depth -= 1
                j += 1
            opt = j < len(seq) and seq[j] == "?"
            out.append(("grp", seq[i + 1:j - 1], opt))
            i = j + (1 if opt else 0)
        else:
            raise ValueError("unsupported grammar fragment: %r" % seq)
    return out


def _match_seq(seq: str, s: str, pos: int) -> List[int]:
    positions = [pos]
    for kind, body, opt in _tokens(seq):
        nxt: List[int] = []
        for p in positions:
            if kind == "lit":
                if s.startswith(body, p):
                    nxt.append(p + len(body))
            else:
                if opt:
                    nxt.append(p)
                nxt.extend(_match_alt(body, s, p))
        positions = nxt
        if not positions:
            return []
    return positions


# -- typed slots: PLAN.md §5 condition C4 ------------------------------------
#
# The symbol rules above make an undeclared symbol undecodable. These make a
# mistyped ARGUMENT undecodable: one CALL rule per tool, each parameter slot
# admitting only registers, field accesses whose declared type fits, and
# constants whose type fits — so `CALL T5 <ID:user> <STR>` cannot take a TIME
# constant in slot one, and a required parameter cannot be omitted. Registers
# stay open (a context-free grammar cannot know what bound r3); the
# typechecker still owns that. Motivating result: Qwen3.8-27B's 11 of 15
# diagnostics were CALL-argument TYPE_ERRORs, four of them repeated after
# being told (results/S2.md). Measure goal_success, not compile: a grammar
# that forbids the wrong symbol makes the sampler pick the best legal one.

from core.ir import parse_type  # noqa: E402

_NUMERIC = {"INT", "FLOAT"}


def _compatible(want, have) -> bool:
    """Mirror of core/typecheck.py _Checker._compatible."""
    if have[0] == "NULL" or want[0] == "NULL":
        return True
    if want[0] == "ID" and have[0] in ("ID", "OBJ"):
        return want[1] == have[1]
    if want[0] in _NUMERIC and have[0] in _NUMERIC:
        return True
    if want[0] == "LIST" and have[0] == "LIST":
        return _compatible(want[1], have[1])
    return want == have


def _type_id(tstr: str) -> str:
    # llama.cpp's GBNF lexer allows only [a-zA-Z0-9-] in rule names; an
    # underscore makes `op_ID_user` parse as `op` and the sampler then
    # dereferences a null grammar (access violation, 2026-09-07).
    # entity names carry underscores too (`barrel_lot`); they never carry
    # hyphens, so the mapping is collision-free
    return tstr.replace(":", "-").replace(" ", "-").replace("_", "-")


def _enum_fields(ctx: dict) -> Dict[str, List[str]]:
    """{field sym -> [enum constant syms]} from the constants' kinds
    (`enum:<entity>.<field>`, spec 0.4.0 §2.2)."""
    by_ef = {(f.get("entity"), f["name"]): f["sym"] for f in ctx.get("fields", [])}
    out: Dict[str, List[str]] = {}
    for c in ctx.get("constants", []):
        k = c.get("kind", "")
        if k.startswith("enum:") and "." in k:
            entity, fname = k[5:].split(".", 1)
            fsym = by_ef.get((entity, fname))
            if fsym:
                out.setdefault(fsym, []).append(c["sym"])
    return out


def _operand_alts(want, fields, consts, fsym: Optional[str], enum_fields,
                  kinds: bool, op_rules: Dict[str, str], tag: str) -> str:
    """Operand alternatives for a slot of type `want`; with `kinds` the slot
    `fsym` narrows the constants: an enum field admits only its own enum
    constants, a plain STR slot admits none of them."""
    alts = ["reg"]
    fsyms = [int(s[1:]) for s, t in fields if _compatible(want, t)]
    if fsyms:
        op_rules[f"cf-{tag}"] = f'"F" ({digit_trie_expr(fsyms)})'
        alts.append(f'reg "." cf-{tag}')
    csyms = [s for s, t in consts if _compatible(want, t)]
    if kinds and want[0] == "STR":
        enum_syms = set(sum(enum_fields.values(), []))
        if fsym in enum_fields:
            csyms = [s for s in csyms if s in enum_fields[fsym]]
        else:
            csyms = [s for s in csyms if s not in enum_syms]
    if csyms:
        op_rules[f"cc-{tag}"] = const_expr(csyms)
        alts.append(f"cc-{tag}")
    alts.append('"NULL"')
    if want[0] == "TIME":
        alts.append('"NOW"')
    if want[0] in _NUMERIC:
        alts.append("num")
    return " | ".join(alts)


def typed_call_rules(task: dict, kinds: bool = False) -> tuple:
    """(rhs for `call`, extra GBNF rules). One `callTn` per tool; one
    `op-<type>` operand class per distinct parameter type — or, with
    `kinds`, per parameter slot whose field is an enum."""
    ctx = task["context"]
    consts = [(c["sym"], parse_type(c["type"])) for c in ctx.get("constants", [])]
    fields = [(f["sym"], parse_type(f["type"])) for f in ctx.get("fields", [])]
    enum_fields = _enum_fields(ctx) if kinds else {}
    op_rules: Dict[str, str] = {}
    lines: List[str] = []

    def op_class(tstr: str, psym: Optional[str]) -> str:
        want = parse_type(tstr)
        tid = _type_id(tstr)
        if kinds and want[0] == "STR" and psym in enum_fields:
            name = f"op-{tid}-{psym}"
        else:
            name = f"op-{tid}"
        if name in op_rules:
            return name
        op_rules[name] = _operand_alts(want, fields, consts, psym, enum_fields,
                                       kinds, op_rules, name[3:])
        return name

    call_alts = []
    for t in ctx.get("tools", []):
        params = t["params"]
        req = [p for p in params if p.get("required", True)]
        opt = [p for p in params if not p.get("required", True)]
        body = f'"CALL {t["sym"]}"'
        for p in req:
            body += f' " " {op_class(p["type"], p.get("sym"))}'
        # optional params are positional and trailing: nested optionals
        tail = ""
        for p in reversed(opt):
            tail = f' (" " {op_class(p["type"], p.get("sym"))}{tail})?'
        body += tail + " (arrow)?"
        rname = f"call{t['sym']}"
        lines.append(f"{rname} ::= {body}")
        call_alts.append(rname)
    for name, rhs in op_rules.items():
        lines.append(f"{name} ::= {rhs}")
    return " | ".join(call_alts) if call_alts else '"CALL " tool (arrow)?', "\n".join(lines)


def kind_field_rules(task: dict) -> tuple:
    """(rhs for `clause`, rhs for `set`, extra rules): one FILTER clause and
    one SET form per field, each admitting only the operands its kind
    allows (spec 0.4.0 §2.2). Non-STR fields keep type-compatible operands."""
    ctx = task["context"]
    consts = [(c["sym"], parse_type(c["type"])) for c in ctx.get("constants", [])]
    fields = [(f["sym"], parse_type(f["type"])) for f in ctx.get("fields", [])]
    enum_fields = _enum_fields(ctx)
    op_rules: Dict[str, str] = {}
    clause_alts, set_alts = [], []
    for fsym, ftype in fields:
        tag = f"f{fsym[1:]}"
        rhs = _operand_alts(ftype, fields, consts, fsym, enum_fields, True,
                            op_rules, tag)
        op_rules[f"opf-{tag}"] = rhs
        clause_alts.append(f'"{fsym} " cmp " " opf-{tag}')
        set_alts.append(f'"{fsym} " opf-{tag}')
    lines = [f"{n} ::= {r}" for n, r in op_rules.items()]
    clause = '("NOT ")? (%s)' % " | ".join(clause_alts) if clause_alts else \
        '("NOT ")? field " " cmp " " operand'
    setr = '"SET " reg " " (%s) arrow' % " | ".join(set_alts) if set_alts else \
        '"SET " reg " " field " " operand arrow'
    return clause, setr, "\n".join(lines)


def typed_signature(task: dict) -> tuple:
    """Cache key for typed grammars: symbol table plus every type and kind."""
    ctx = task["context"]
    return (tuple((t["sym"], tuple((p["type"], p.get("required", True), p.get("sym"))
                                   for p in t["params"]))
                  for t in ctx.get("tools", [])),
            tuple(sorted((f["sym"], f["type"], f.get("entity"), f["name"])
                         for f in ctx.get("fields", []))),
            tuple(sorted((c["sym"], c["type"], c.get("kind", ""))
                         for c in ctx.get("constants", []))))


_GBNF_NAME = re.compile(r"^[a-zA-Z0-9-]+$")


def check_gbnf_names(text: str) -> List[str]:
    """Rule names llama.cpp's lexer would reject. `LlamaGrammar.from_string`
    prints the parse error and returns anyway, so this is the check."""
    bad = []
    for line in text.splitlines():
        if "::=" in line and not line.lstrip().startswith("#"):
            name = line.split("::=")[0].strip()
            if not _GBNF_NAME.match(name):
                bad.append(name)
    return bad
