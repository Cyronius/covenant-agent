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


def symbol_rules(tools: Iterable[str], fields: Iterable[str],
                 consts: Iterable[str]) -> Dict[str, str]:
    """{rule name -> right-hand side} for the three symbol rules."""
    return {
        "tool": '"T" (%s)' % digit_trie_expr(_nums(tools)),
        "field": '"F" (%s)' % digit_trie_expr(_nums(fields)),
        "const": '"C" (%s)' % digit_trie_expr(_nums(consts)),
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


def grammar_for_task(task: dict, base: Optional[str] = None,
                     typed: bool = False) -> str:
    """`task` as the suites store it: task['context'] holds the symbol table.
    `typed` adds the per-tool typed-slot CALL rules (PLAN.md §5 C4)."""
    ctx = task["context"]
    out = grammar_for_symbols(
        [t["sym"] for t in ctx.get("tools", [])],
        [f["sym"] for f in ctx.get("fields", [])],
        [c["sym"] for c in ctx.get("constants", [])],
        base=base)
    if typed:
        call_rhs, extra = typed_call_rules(task)
        out = _replace_rules(out, {"call": call_rhs}) + "\n" + extra
    return out


def grammar_for_context(ctx, base: Optional[str] = None) -> str:
    """`ctx` is a core.ir.TaskContext (harness.context.build_context)."""
    return grammar_for_symbols(ctx.tools, ctx.fields, ctx.constants, base=base)


def symbol_signature(task: dict) -> tuple:
    """Cache key: two tasks with the same symbol table share a grammar."""
    ctx = task["context"]
    return (tuple(sorted(_nums(t["sym"] for t in ctx.get("tools", [])))),
            tuple(sorted(_nums(f["sym"] for f in ctx.get("fields", [])))),
            tuple(sorted(_nums(c["sym"] for c in ctx.get("constants", [])))))


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
    return tstr.replace(":", "_").replace(" ", "_")


def typed_call_rules(task: dict) -> tuple:
    """(rhs for `call`, extra GBNF rules). One `callTn` per tool; one
    `op_<type>` operand class per distinct parameter type."""
    ctx = task["context"]
    consts = [(c["sym"], parse_type(c["type"])) for c in ctx.get("constants", [])]
    fields = [(f["sym"], parse_type(f["type"])) for f in ctx.get("fields", [])]
    op_rules: Dict[str, str] = {}
    lines: List[str] = []

    def op_class(tstr: str) -> str:
        tid = _type_id(tstr)
        name = f"op_{tid}"
        if name in op_rules:
            return name
        want = parse_type(tstr)
        alts = ["reg"]
        fsyms = [int(s[1:]) for s, t in fields if _compatible(want, t)]
        if fsyms:
            op_rules[f"cf_{tid}"] = f'"F" ({digit_trie_expr(fsyms)})'
            alts.append(f'reg "." cf_{tid}')
        csyms = [int(s[1:]) for s, t in consts if _compatible(want, t)]
        if csyms:
            op_rules[f"cc_{tid}"] = f'"C" ({digit_trie_expr(csyms)})'
            alts.append(f"cc_{tid}")
        alts.append('"NULL"')
        if want[0] == "TIME":
            alts.append('"NOW"')
        if want[0] in _NUMERIC:
            alts.append("num")
        op_rules[name] = " | ".join(alts)
        return name

    call_alts = []
    for t in ctx.get("tools", []):
        params = t["params"]
        req = [p for p in params if p.get("required", True)]
        opt = [p for p in params if not p.get("required", True)]
        body = f'"CALL {t["sym"]}"'
        for p in req:
            body += f' " " {op_class(p["type"])}'
        # optional params are positional and trailing: nested optionals
        tail = ""
        for p in reversed(opt):
            tail = f' (" " {op_class(p["type"])}{tail})?'
        body += tail + " (arrow)?"
        rname = f"call{t['sym']}"
        lines.append(f"{rname} ::= {body}")
        call_alts.append(rname)
    for name, rhs in op_rules.items():
        lines.append(f"{name} ::= {rhs}")
    return " | ".join(call_alts) if call_alts else '"CALL " tool (arrow)?', "\n".join(lines)


def typed_signature(task: dict) -> tuple:
    """Cache key for typed grammars: symbol table plus every type."""
    ctx = task["context"]
    return (tuple((t["sym"], tuple((p["type"], p.get("required", True))
                                   for p in t["params"]))
                  for t in ctx.get("tools", [])),
            tuple(sorted((f["sym"], f["type"]) for f in ctx.get("fields", []))),
            tuple(sorted((c["sym"], c["type"]) for c in ctx.get("constants", []))))
