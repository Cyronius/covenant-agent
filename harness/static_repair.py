"""Deterministic repair of provably-meaningless program fragments.

The K2 repair round (baselines/qwen/run_a.py) hands the checker's diagnostics
back to the model and asks it to rewrite. On the demo surface that does not
work: the tuned checkpoint answers a diagnostic it cannot act on by
abstaining, so "didn't compile" becomes `ABORT NEEDS_INFO` (measured on all
three demo failures, 2026-09-15). This module fixes the fragments that need
no judgement at all, before the model is asked anything.

Only provably-dead fragments are touched. Both rules remove text that cannot
change what a correct program computes:

  1. A FILTER conjunct naming a field the filtered entity does not have.
     The checker already calls this UNKNOWN_FIELD; it can only ever be an
     error, never a predicate, so dropping it is not a guess at intent.
     `FILTER r0 F20 EQ B0 AND F1 EQ F15` (F1/F15 are user fields, r0 holds
     customers) -> `FILTER r0 F20 EQ B0`.

  2. Two un-negated EQ conjuncts on the same field with different constants.
     The conjunction is unsatisfiable, so the FILTER returns empty whatever
     the data — `F13 EQ B0 AND F13 EQ B1` is delinquent=true AND
     delinquent=false. The later conjunct is dropped: the model writes the
     clause it was asked for first and pads afterwards (both observed cases
     had the intended predicate in front).

Neither rule invents a predicate, and a FILTER whose every conjunct dies is
removed entirely, its destination register rewritten to its source.

This is a stopgap for a corpus defect, not a fix for it: 97.2% of S5 rows
that call a list tool then FILTER, so the model pads a one-predicate request
into two. `results/logs/` has the measurement. Repairs are counted so a run
reports how often it leaned on this.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

# FILTER <src> <predicate> -> <dst>
_FILTER = re.compile(r'^(?P<indent>\s*)FILTER\s+(?P<src>r\d+)\s+'
                     r'(?P<pred>.+?)\s*->\s*(?P<dst>r\d+)\s*$')
_UNKNOWN_FIELD = re.compile(r'^UNKNOWN_FIELD\s+(r\d+)\s+(F\d+)')
# a conjunct: optional NOT, a field, a comparator, an operand
_CONJUNCT = re.compile(r'^(?P<not>NOT\s+)?(?P<lhs>F\d+)\s+(?P<cmp>\S+)\s+'
                       r'(?P<rhs>.+)$')


def _split_conjuncts(pred: str) -> Optional[List[str]]:
    """AND-joined conjuncts, or None when the predicate uses OR — mixing the
    two makes dropping a term unsound, so those are left alone."""
    if re.search(r'\bOR\b', pred):
        return None
    return [c.strip() for c in re.split(r'\s+AND\s+', pred.strip()) if c.strip()]


def _drop_filter_line(lines: List[str], idx: int, src: str, dst: str) -> None:
    """Remove a FILTER whose predicate is entirely dead, aliasing its
    destination register to its source in the lines that follow."""
    del lines[idx]
    for j in range(idx, len(lines)):
        lines[j] = re.sub(r'(?<![A-Za-z0-9])%s(?![0-9])' % dst, src, lines[j])


def _apply_unknown_fields(text: str, diagnostics: List[str]) -> Tuple[str, int]:
    """Rule 1. Driven by the checker's own UNKNOWN_FIELD output, so it needs
    no second implementation of register typing."""
    dead = {}
    for d in diagnostics:
        m = _UNKNOWN_FIELD.match(d.strip())
        if m:
            dead.setdefault(m.group(1), set()).add(m.group(2))
    if not dead:
        return text, 0
    lines = text.split("\n")
    fixed = 0
    i = 0
    while i < len(lines):
        m = _FILTER.match(lines[i])
        if not m or m.group('src') not in dead:
            i += 1
            continue
        bad = dead[m.group('src')]
        parts = _split_conjuncts(m.group('pred'))
        if parts is None:
            i += 1
            continue
        keep = [c for c in parts
                if not any(re.search(r'(?<![A-Za-z0-9])%s(?![0-9])' % f, c)
                           for f in bad)]
        if len(keep) == len(parts):
            i += 1
            continue
        fixed += len(parts) - len(keep)
        if keep:
            lines[i] = "%sFILTER %s %s -> %s" % (
                m.group('indent'), m.group('src'), " AND ".join(keep), m.group('dst'))
            i += 1
        else:
            _drop_filter_line(lines, i, m.group('src'), m.group('dst'))
    return "\n".join(lines), fixed


def _apply_contradictions(text: str) -> Tuple[str, int]:
    """Rule 2. Static and self-contained: an unsatisfiable AND needs no
    context to recognise."""
    lines = text.split("\n")
    fixed = 0
    i = 0
    while i < len(lines):
        m = _FILTER.match(lines[i])
        if not m:
            i += 1
            continue
        parts = _split_conjuncts(m.group('pred'))
        if parts is None or len(parts) < 2:
            i += 1
            continue
        keep, seen = [], {}
        for c in parts:
            cm = _CONJUNCT.match(c)
            if not cm or cm.group('not') or cm.group('cmp') != 'EQ':
                keep.append(c)
                continue
            lhs, rhs = cm.group('lhs'), cm.group('rhs').strip()
            # only constants: a register or field operand can hold anything
            if not re.fullmatch(r'[CSNBDI]\d+', rhs):
                keep.append(c)
                continue
            if lhs in seen and seen[lhs] != rhs:
                fixed += 1          # unsatisfiable against the earlier term
                continue
            seen[lhs] = rhs
            keep.append(c)
        if not fixed or len(keep) == len(parts):
            i += 1
            continue
        if keep:
            lines[i] = "%sFILTER %s %s -> %s" % (
                m.group('indent'), m.group('src'), " AND ".join(keep), m.group('dst'))
            i += 1
        else:
            _drop_filter_line(lines, i, m.group('src'), m.group('dst'))
    return "\n".join(lines), fixed


def repair(text: str, diagnostics: Optional[List[str]] = None) -> Tuple[str, int]:
    """(repaired program, number of fragments removed). Returns the text
    unchanged, and 0, when nothing is provably dead."""
    out, n1 = _apply_unknown_fields(text, diagnostics or [])
    out, n2 = _apply_contradictions(out)
    return out, n1 + n2
