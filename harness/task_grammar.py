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


def grammar_for_task(task: dict, base: Optional[str] = None) -> str:
    """`task` as the suites store it: task['context'] holds the symbol table."""
    ctx = task["context"]
    return grammar_for_symbols(
        [t["sym"] for t in ctx.get("tools", [])],
        [f["sym"] for f in ctx.get("fields", [])],
        [c["sym"] for c in ctx.get("constants", [])],
        base=base)


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
