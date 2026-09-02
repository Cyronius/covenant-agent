"""Agent Core AST -> JavaScript (spec §10).

Deterministic: the same AST always produces byte-identical JS. The emitted
code contains no free identifiers besides `rt`; tool/field/constant symbols
are resolved by the runtime, so the JS is independent of the task context.

Emitted runtime surface: rt.call, rt.pause, rt.ret, rt.stop, rt.initial,
rt.constant, rt.now, rt.cmp, rt.fld, rt.setF, rt.mapF, rt.count, rt.sortBy,
rt.select, rt.first, rt.isToolError.
"""
from __future__ import annotations

from .ir import (Abort, Format, Call, Const, Count, Filter, First, Foreach, Get, If, IntLit,
                 Let, MapF, Now, Null, Parallel, Pause, Pred, Program, Reg,
                 RegField, Return, Select, SetF, Sort, Stop, TaskContext, Try)


def _collect_regs(body: list, regs: set):
    def op_regs(op):
        if isinstance(op, (Reg, RegField)):
            regs.add(op.n)

    def pred_regs(p: Pred):
        for cl in p.clauses:
            if not isinstance(cl.left, str):
                op_regs(cl.left)
            op_regs(cl.right)

    for instr in body:
        for attr in ("dst", "src", "var"):
            v = getattr(instr, attr, None)
            if isinstance(v, Reg):
                regs.add(v.n)
        if isinstance(instr, Let):
            op_regs(instr.op)
        elif isinstance(instr, SetF):
            op_regs(instr.op)
        elif isinstance(instr, Call):
            for a in instr.args:
                op_regs(a)
        elif isinstance(instr, Filter):
            pred_regs(instr.pred)
        elif isinstance(instr, Select):
            op_regs(instr.idx)
        elif isinstance(instr, Foreach):
            _collect_regs(instr.body, regs)
        elif isinstance(instr, If):
            pred_regs(instr.cond)
            _collect_regs(instr.then, regs)
            if instr.els is not None:
                _collect_regs(instr.els, regs)
        elif isinstance(instr, Parallel):
            _collect_regs(instr.calls, regs)
        elif isinstance(instr, Try):
            _collect_regs(instr.body, regs)
        elif isinstance(instr, Return):
            op_regs(instr.op)


class _Emitter:
    def __init__(self, program: Program, ctx: TaskContext):
        self.program = program
        self.ctx = ctx
        self.lines: list = []
        self.try_counter = 0
        self.par_counter = 0
        self.assigned: set = set()
        for rname in ctx.initial_registers:
            self.assigned.add(int(rname[1:]))

    def out(self, depth: int, s: str):
        self.lines.append("  " * depth + s)

    # -- expressions -----------------------------------------------------
    def operand(self, op) -> str:
        if isinstance(op, Reg):
            return f"r{op.n}"
        if isinstance(op, RegField):
            return f'rt.fld(r{op.n}, "{op.field}")'
        if isinstance(op, Const):
            return f'rt.constant("{op.sym}")'
        if isinstance(op, Now):
            return "rt.now()"
        if isinstance(op, Null):
            return "null"
        if isinstance(op, IntLit):
            return str(op.v)
        raise AssertionError(op)

    def pred(self, p: Pred, elem: str) -> str:
        """elem is the JS identifier for the FILTER element, or None for IF."""
        parts = []
        for i, cl in enumerate(p.clauses):
            if isinstance(cl.left, str):
                left = f'rt.fld({elem}, "{cl.left}")'
            else:
                left = self.operand(cl.left)
            e = f'rt.cmp("{cl.cmp}", {left}, {self.operand(cl.right)})'
            if cl.neg:
                e = f"!{e}"
            if i > 0:
                parts.append("&&" if p.ops[i - 1] == "AND" else "||")
            parts.append(e)
        return " ".join(parts)

    def call_expr(self, c: Call) -> str:
        args = ", ".join(self.operand(a) for a in c.args)
        return f'rt.call("{c.tool}", [{args}])'

    # -- statements ------------------------------------------------------
    def block(self, body: list, depth: int):
        for instr in body:
            self.instr(instr, depth)

    def _assign(self, dst: Reg) -> str:
        self.assigned.add(dst.n)
        return f"r{dst.n} = "

    def instr(self, instr, depth: int):
        if isinstance(instr, Let):
            self.out(depth, f"{self._assign(instr.dst)}{self.operand(instr.op)};")
        elif isinstance(instr, Get):
            self.out(depth, f'{self._assign(instr.dst)}rt.fld(r{instr.src.n}, "{instr.field}");')
        elif isinstance(instr, SetF):
            self.out(depth, f'{self._assign(instr.dst)}rt.setF(r{instr.src.n}, "{instr.field}", {self.operand(instr.op)});')
        elif isinstance(instr, Call):
            prefix = self._assign(instr.dst) if instr.dst is not None else ""
            self.out(depth, f"{prefix}await {self.call_expr(instr)};")
        elif isinstance(instr, Filter):
            p = self.pred(instr.pred, "_x")
            self.out(depth, f"{self._assign(instr.dst)}(r{instr.src.n}).filter((_x) => ({p}));")
        elif isinstance(instr, MapF):
            self.out(depth, f'{self._assign(instr.dst)}rt.mapF(r{instr.src.n}, "{instr.field}");')
        elif isinstance(instr, Format):
            vals = ", ".join(self.operand(o) for o in instr.ops)
            kinds = ", ".join(f'"{k}"' for k in getattr(instr, "kinds", ()))
            self.out(depth, f'{self._assign(instr.dst)}rt.format(rt.constant("{instr.template}"), [{vals}], [{kinds}]);')
        elif isinstance(instr, Count):
            self.out(depth, f"{self._assign(instr.dst)}rt.count(r{instr.src.n});")
        elif isinstance(instr, Sort):
            self.out(depth, f'{self._assign(instr.dst)}rt.sortBy(r{instr.src.n}, "{instr.field}", "{instr.dir}");')
        elif isinstance(instr, Select):
            self.out(depth, f"{self._assign(instr.dst)}rt.select(r{instr.src.n}, {self.operand(instr.idx)});")
        elif isinstance(instr, First):
            self.out(depth, f"{self._assign(instr.dst)}rt.first(r{instr.src.n});")
        elif isinstance(instr, Foreach):
            self.assigned.add(instr.var.n)
            self.out(depth, f"for (r{instr.var.n} of r{instr.src.n}) {{")
            self.block(instr.body, depth + 1)
            self.out(depth, "}")
        elif isinstance(instr, If):
            self.out(depth, f"if ({self.pred(instr.cond, None)}) {{")
            self.block(instr.then, depth + 1)
            if instr.els is not None:
                self.out(depth, "} else {")
                self.block(instr.els, depth + 1)
            self.out(depth, "}")
        elif isinstance(instr, Parallel):
            v = f"_p{self.par_counter}"
            self.par_counter += 1
            exprs = ", ".join(self.call_expr(c) for c in instr.calls)
            self.out(depth, f"const {v} = await Promise.all([{exprs}]);")
            for i, c in enumerate(instr.calls):
                if c.dst is not None:
                    self.out(depth, f"{self._assign(c.dst)}{v}[{i}];")
        elif isinstance(instr, Try):
            a = f"_a{self.try_counter}"
            self.try_counter += 1
            self.assigned.add(instr.dst.n)
            self.out(depth, f'r{instr.dst.n} = "OK";')
            self.out(depth, f"for (let {a} = 0; {a} <= {instr.retry}; {a}++) {{")
            self.out(depth + 1, "try {")
            self.block(instr.body, depth + 2)
            self.out(depth + 2, f'r{instr.dst.n} = "OK";')
            self.out(depth + 2, "break;")
            self.out(depth + 1, "} catch (_e) {")
            self.out(depth + 2, "if (!rt.isToolError(_e)) { throw _e; }")
            self.out(depth + 2, f"r{instr.dst.n} = _e.code;")
            self.out(depth + 1, "}")
            self.out(depth, "}")
        elif isinstance(instr, Return):
            self.out(depth, f"return rt.ret({self.operand(instr.op)});")
        elif isinstance(instr, Stop):
            self.out(depth, "return rt.stop();")
        elif isinstance(instr, Abort):
            self.out(depth, f'return rt.abort("{instr.reason}");')
        elif isinstance(instr, Pause):
            regs = sorted(self.assigned)
            pairs = ", ".join(f"r{n}: r{n}" for n in regs)
            self.out(depth, f"return rt.pause({{{pairs}}});")
        else:
            raise AssertionError(instr)

    def emit(self) -> str:
        regs: set = set(self.assigned)
        _collect_regs(self.program.body, regs)
        decl = ", ".join(f"r{n}" for n in sorted(regs))
        self.out(0, "async function main(rt) {")
        if decl:
            self.out(1, f"let {decl};")
        for rname in sorted(self.ctx.initial_registers,
                            key=lambda r: int(r[1:])):
            self.out(1, f'{rname} = rt.initial("{rname}");')
        self.block(self.program.body, 1)
        last = self.program.body[-1] if self.program.body else None
        if not isinstance(last, (Return, Stop, Pause, Abort)):
            self.out(1, "return rt.stop();")
        self.out(0, "}")
        return "\n".join(self.lines) + "\n"


def compile_program(program: Program, ctx: TaskContext) -> str:
    """AST -> JS source. Deterministic; raises nothing on well-typed input."""
    return _Emitter(program, ctx).emit()
