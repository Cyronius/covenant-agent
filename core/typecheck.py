"""Agent Core typechecker (spec §5, §6, §9).

check(program, ctx) -> TypecheckResult with structured diagnostics. Tracks
register bindings per program point, validates tool calls against schemas,
field symbols against entities, and reachability.

Unknown constant symbols are reported as UNBOUND <sym> (the symbol has no
binding in the task context) — see spec §9.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional

from . import diagnostics as dg
from .ir import (ABORT_REF_KINDS, Abort, Format, Call, Clause, Const, Count, Filter, First, Foreach, Get, If,
                 IntLit, Let, MapF, Now, Null, Parallel, Pause, Pred, Program,
                 Reg, RegField, Return, Select, SetF, Sort, Stop, TaskContext,
                 Try, Type, format_type)

NUMERIC = ("INT", "TIME")


@dataclass
class TypecheckResult:
    ok: bool
    diagnostics: List[dg.Diagnostic]
    # environment (reg name -> type string) at each PAUSE, in source order
    pause_envs: List[Dict[str, str]] = dc_field(default_factory=list)
    # environment at normal termination points (informational)
    final_env: Dict[str, str] = dc_field(default_factory=dict)


class _Checker:
    def __init__(self, ctx: TaskContext):
        self.ctx = ctx
        self.diags: List[dg.Diagnostic] = []
        self.pause_envs: List[Dict[str, str]] = []

    # -- helpers ---------------------------------------------------------
    def _compatible(self, expected: Type, got: Type) -> bool:
        if got[0] == "NULL" or expected[0] == "NULL":
            return True
        if expected[0] == "ID" and got[0] in ("ID", "OBJ"):
            return expected[1] == got[1]
        if expected[0] in NUMERIC and got[0] in NUMERIC:
            return True
        if expected[0] == "LIST" and got[0] == "LIST":
            return self._compatible(expected[1], got[1])
        return expected == got

    def _field_decl(self, sym: str, entity: str, line: int, reg_repr: str):
        f = self.ctx.fields.get(sym)
        if f is None or f.entity != entity:
            self.diags.append(dg.unknown_field(line, reg_repr, sym))
            return None
        return f

    def _operand_type(self, op, env: dict, line: int) -> Optional[Type]:
        if isinstance(op, Reg):
            t = env.get(op.n)
            if t is None:
                self.diags.append(dg.unbound(line, str(op)))
            return t
        if isinstance(op, RegField):
            base = env.get(op.n)
            if base is None:
                self.diags.append(dg.unbound(line, f"r{op.n}"))
                return None
            if base[0] != "OBJ":
                self.diags.append(dg.type_error(
                    line, "OBJ:<entity>", format_type(base)))
                return None
            f = self._field_decl(op.field, base[1], line, f"r{op.n}")
            return f.type if f else None
        if isinstance(op, Const):
            c = self.ctx.constants.get(op.sym)
            if c is None:
                self.diags.append(dg.unbound(line, op.sym))
                return None
            return c.type
        if isinstance(op, Now):
            return ("TIME",)
        if isinstance(op, Null):
            return ("NULL",)
        if isinstance(op, IntLit):
            return ("INT",)
        raise AssertionError(op)

    def _check_cmp(self, cmp: str, lt: Optional[Type], rt: Optional[Type],
                   line: int):
        if lt is None or rt is None:
            return
        if cmp in ("LT", "GT"):
            for t in (lt, rt):
                if t[0] not in NUMERIC and t[0] != "NULL":
                    self.diags.append(dg.type_error(
                        line, "INT|TIME", format_type(t)))
            return
        if cmp == "CONTAINS":
            if lt[0] == "STR":
                if rt[0] not in ("STR", "NULL"):
                    self.diags.append(dg.type_error(line, "STR", format_type(rt)))
            elif lt[0] == "LIST":
                if not self._compatible(lt[1], rt):
                    self.diags.append(dg.type_error(
                        line, format_type(lt[1]), format_type(rt)))
            elif lt[0] != "NULL":
                self.diags.append(dg.type_error(
                    line, "STR|LIST", format_type(lt)))
            return
        # EQ
        if not (self._compatible(lt, rt) or self._compatible(rt, lt)):
            self.diags.append(dg.type_error(line, format_type(lt), format_type(rt)))

    def _list_elem_entity(self, reg: Reg, env: dict, line: int) -> Optional[str]:
        """For FILTER/MAP/SORT: register must hold LIST(OBJ(e)); return e."""
        t = env.get(reg.n)
        if t is None:
            self.diags.append(dg.unbound(line, str(reg)))
            return None
        if t[0] != "LIST" or t[1][0] != "OBJ":
            self.diags.append(dg.type_error(
                line, "LIST OBJ:<entity>", format_type(t)))
            return None
        return t[1][1]

    def _check_filter_pred(self, pred: Pred, entity: str, reg: Reg,
                           env: dict, line: int):
        for cl in pred.clauses:
            f = self._field_decl(cl.left, entity, line, str(reg))
            rt = self._operand_type(cl.right, env, line)
            self._check_cmp(cl.cmp, f.type if f else None, rt, line)

    def _check_cond(self, cond: Pred, env: dict, line: int):
        for cl in cond.clauses:
            lt = self._operand_type(cl.left, env, line)
            rt = self._operand_type(cl.right, env, line)
            self._check_cmp(cl.cmp, lt, rt, line)

    # -- block walk ------------------------------------------------------
    def check_block(self, body: list, env: dict) -> bool:
        """Check instructions; mutate env. Returns True if the block always
        terminates the program (ends in RETURN/STOP/PAUSE on all paths)."""
        terminated = False
        for instr in body:
            if terminated:
                self.diags.append(dg.unreachable(instr.line))
                continue
            terminated = self.check_instr(instr, env)
        return terminated

    def check_instr(self, instr, env: dict) -> bool:
        ln = instr.line
        if isinstance(instr, Let):
            t = self._operand_type(instr.op, env, ln)
            if t is not None:
                env[instr.dst.n] = t
            return False
        if isinstance(instr, Get):
            t = self._operand_type(RegField(instr.src.n, instr.field), env, ln)
            if t is not None:
                env[instr.dst.n] = t
            return False
        if isinstance(instr, SetF):
            base = env.get(instr.src.n)
            if base is None:
                self.diags.append(dg.unbound(ln, str(instr.src)))
                return False
            if base[0] != "OBJ":
                self.diags.append(dg.type_error(
                    ln, "OBJ:<entity>", format_type(base)))
                return False
            f = self._field_decl(instr.field, base[1], ln, str(instr.src))
            vt = self._operand_type(instr.op, env, ln)
            if f and vt and not self._compatible(f.type, vt):
                self.diags.append(dg.type_error(
                    ln, format_type(f.type), format_type(vt)))
            env[instr.dst.n] = base
            return False
        if isinstance(instr, Call):
            self._check_call(instr, env)
            return False
        if isinstance(instr, Filter):
            entity = self._list_elem_entity(instr.src, env, ln)
            if entity:
                self._check_filter_pred(instr.pred, entity, instr.src, env, ln)
                env[instr.dst.n] = env[instr.src.n]
            return False
        if isinstance(instr, MapF):
            entity = self._list_elem_entity(instr.src, env, ln)
            if entity:
                f = self._field_decl(instr.field, entity, ln, str(instr.src))
                if f:
                    env[instr.dst.n] = ("LIST", f.type)
            return False
        if isinstance(instr, Format):
            c = self.ctx.constants.get(instr.template)
            if c is None:
                self.diags.append(dg.unbound(ln, instr.template))
            elif c.type != ("STR",):
                self.diags.append(dg.type_error(ln, "STR", format_type(c.type)))
            else:
                slots = set(re.findall(r"\{(\d+)\}", str(c.value)))
                want = {str(i) for i in range(len(instr.ops))}
                if slots != want:
                    self.diags.append(dg.type_error(
                        ln, f"template with {len(instr.ops)} slots",
                        f"{len(slots)} slots"))
            kinds = []
            for op in instr.ops:
                t = self._operand_type(op, env, ln)
                if t is not None and t[0] not in ("STR", "INT", "TIME", "ID"):
                    self.diags.append(dg.type_error(
                        ln, "STR|INT|TIME|ID", format_type(t)))
                kinds.append(t[0] if t else "STR")
            # the runtime has no types; tell the compiler how to render each slot
            instr.kinds = tuple(kinds)
            env[instr.dst.n] = ("STR",)
            return False
        if isinstance(instr, Count):
            t = env.get(instr.src.n)
            if t is None:
                self.diags.append(dg.unbound(ln, str(instr.src)))
            elif t[0] != "LIST":
                self.diags.append(dg.type_error(ln, "LIST", format_type(t)))
            env[instr.dst.n] = ("INT",)
            return False
        if isinstance(instr, Sort):
            entity = self._list_elem_entity(instr.src, env, ln)
            if entity:
                f = self._field_decl(instr.field, entity, ln, str(instr.src))
                if f and f.type[0] not in ("INT", "TIME", "STR", "BOOL"):
                    self.diags.append(dg.type_error(
                        ln, "INT|TIME|STR|BOOL", format_type(f.type)))
                env[instr.dst.n] = env[instr.src.n]
            return False
        if isinstance(instr, Select):
            t = env.get(instr.src.n)
            it = self._operand_type(instr.idx, env, ln)
            if t is None:
                self.diags.append(dg.unbound(ln, str(instr.src)))
            elif t[0] != "LIST":
                self.diags.append(dg.type_error(ln, "LIST", format_type(t)))
            else:
                env[instr.dst.n] = t[1]
            if it is not None and it[0] != "INT":
                self.diags.append(dg.type_error(ln, "INT", format_type(it)))
            return False
        if isinstance(instr, First):
            t = env.get(instr.src.n)
            if t is None:
                self.diags.append(dg.unbound(ln, str(instr.src)))
            elif t[0] != "LIST":
                self.diags.append(dg.type_error(ln, "LIST", format_type(t)))
            else:
                env[instr.dst.n] = t[1]
            return False
        if isinstance(instr, Foreach):
            t = env.get(instr.src.n)
            if t is None:
                self.diags.append(dg.unbound(ln, str(instr.src)))
                return False
            if t[0] != "LIST":
                self.diags.append(dg.type_error(ln, "LIST", format_type(t)))
                return False
            env[instr.var.n] = t[1]
            self.check_block(instr.body, env)
            return False
        if isinstance(instr, If):
            self._check_cond(instr.cond, env, ln)
            env_then = dict(env)
            t_then = self.check_block(instr.then, env_then)
            if instr.els is not None:
                env_else = dict(env)
                t_else = self.check_block(instr.els, env_else)
                # merge: keep bindings equal in both surviving arms
                survivors = []
                if not t_then:
                    survivors.append(env_then)
                if not t_else:
                    survivors.append(env_else)
                if not survivors:
                    return True
                merged = dict(survivors[0])
                for other in survivors[1:]:
                    for k in list(merged):
                        if other.get(k) != merged[k]:
                            del merged[k]
                env.clear()
                env.update(merged)
                return False
            # no else: only bindings made before the IF are guaranteed
            return False
        if isinstance(instr, Parallel):
            dsts = {c.dst.n for c in instr.calls if c.dst is not None}
            for c in instr.calls:
                for a in c.args:
                    if isinstance(a, (Reg, RegField)) and a.n in dsts:
                        self.diags.append(dg.unbound(c.line, f"r{a.n}"))
                self._check_call(c, env, defer_bind=True)
            for c in instr.calls:
                if c.dst is not None:
                    tool = self.ctx.tools.get(c.tool)
                    env[c.dst.n] = tool.returns if tool and tool.returns \
                        else ("NULL",)
            return False
        if isinstance(instr, Try):
            # bindings inside TRY escape (spec §4); dst is STATUS
            self.check_block(instr.body, env)
            env[instr.dst.n] = ("STATUS",)
            return False
        if isinstance(instr, Return):
            self._operand_type(instr.op, env, ln)
            return True
        if isinstance(instr, Stop):
            return True
        if isinstance(instr, Abort):
            self._check_abort(instr)
            return True
        if isinstance(instr, Pause):
            self.pause_envs.append(
                {f"r{n}": format_type(t) for n, t in sorted(env.items())})
            return True
        raise AssertionError(instr)

    def _check_abort(self, instr: Abort):
        # spec §4: referents are symbols of the kind the reason admits, and
        # must be declared. A wrong kind is a TYPE_ERROR whose "expected" is
        # the reason with its admitted kinds, e.g. `NEEDS_INFO:F`.
        allowed = ABORT_REF_KINDS[instr.reason]
        for ref in instr.refs:
            kind = ref[0]
            if kind not in allowed:
                self.diags.append(dg.type_error(
                    instr.line, f"{instr.reason}:{allowed or 'none'}", ref))
                continue
            if kind == "T" and ref not in self.ctx.tools:
                self.diags.append(dg.unknown_tool(instr.line, ref))
            elif kind == "F" and ref not in self.ctx.fields:
                self.diags.append(dg.unknown_field(instr.line, "ABORT", ref))
            elif kind == "C" and ref not in self.ctx.constants:
                self.diags.append(dg.unbound(instr.line, ref))

    def _check_call(self, call: Call, env: dict, defer_bind: bool = False):
        ln = call.line
        tool = self.ctx.tools.get(call.tool)
        if tool is None:
            self.diags.append(dg.unknown_tool(ln, call.tool))
            return
        for i, p in enumerate(tool.params):
            if i >= len(call.args):
                if p.required:
                    self.diags.append(dg.missing_arg(ln, tool.sym, p.sym))
                continue
            at = self._operand_type(call.args[i], env, ln)
            if at is not None and not self._compatible(p.type, at):
                self.diags.append(dg.type_error(
                    ln, format_type(p.type), format_type(at)))
        if len(call.args) > len(tool.params):
            self.diags.append(dg.type_error(
                ln, f"args:{len(tool.params)}", f"args:{len(call.args)}"))
        if call.dst is not None and not defer_bind:
            env[call.dst.n] = tool.returns if tool.returns else ("NULL",)


def check(program: Program, ctx: TaskContext) -> TypecheckResult:
    checker = _Checker(ctx)
    env = {}
    for rname, t in ctx.initial_registers.items():
        n = int(rname[1:])
        env[n] = t
    checker.check_block(program.body, env)
    return TypecheckResult(
        ok=not checker.diags,
        diagnostics=checker.diags,
        pause_envs=checker.pause_envs,
        final_env={f"r{n}": format_type(t) for n, t in sorted(env.items())},
    )
