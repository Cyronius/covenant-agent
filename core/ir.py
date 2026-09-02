"""Agent Core AST and task-context declarations.

The AST mirrors spec/agent_core.md §3. Every node carries the 1-based source
line so diagnostics and UNREACHABLE reports can point at real lines.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

EFFECTS = ("READ", "WRITE", "DELETE", "SEND", "PAY", "EXTERNAL")
DESTRUCTIVE_EFFECTS = ("DELETE", "SEND", "PAY")
CMPS = ("EQ", "LT", "GT", "CONTAINS")
ERROR_CODES = (
    "NOT_FOUND", "PERMISSION_DENIED", "RATE_LIMITED", "INVALID_ARGUMENT",
    "PARTIAL_DATA", "INDEX_OUT_OF_RANGE",
)
NUM_REGISTERS = 16


# ---------------------------------------------------------------- types
# A type is a tuple: ('INT',) ('STR',) ('BOOL',) ('TIME',) ('STATUS',)
# ('NULL',) ('ID', entity) ('OBJ', entity) ('LIST', elemtype)
Type = tuple


def parse_type(s: str) -> Type:
    """Parse the schema type syntax: INT, STR, ID:card, OBJ:card, LIST OBJ:card."""
    s = s.strip()
    if s.startswith("LIST "):
        return ("LIST", parse_type(s[5:]))
    if ":" in s:
        kind, entity = s.split(":", 1)
        if kind not in ("ID", "OBJ"):
            raise ValueError(f"bad type: {s}")
        return (kind, entity)
    if s not in ("INT", "STR", "BOOL", "TIME", "STATUS", "NULL"):
        raise ValueError(f"bad type: {s}")
    return (s,)


def format_type(t: Type) -> str:
    if t[0] == "LIST":
        return "LIST " + format_type(t[1])
    if t[0] in ("ID", "OBJ"):
        return f"{t[0]}:{t[1]}"
    return t[0]


# ---------------------------------------------------------------- operands
@dataclass(frozen=True)
class Reg:
    n: int

    def __str__(self) -> str:
        return f"r{self.n}"


@dataclass(frozen=True)
class RegField:
    n: int
    field: str  # 'F3'

    def __str__(self) -> str:
        return f"r{self.n}.{self.field}"


@dataclass(frozen=True)
class Const:
    sym: str  # 'C3'

    def __str__(self) -> str:
        return self.sym


@dataclass(frozen=True)
class Now:
    def __str__(self) -> str:
        return "NOW"


@dataclass(frozen=True)
class Null:
    def __str__(self) -> str:
        return "NULL"


@dataclass(frozen=True)
class IntLit:
    v: int

    def __str__(self) -> str:
        return str(self.v)


Operand = Union[Reg, RegField, Const, Now, Null, IntLit]


# ---------------------------------------------------------------- predicates
@dataclass(frozen=True)
class Clause:
    """One comparison. In FILTER predicates `left` is a field symbol (str);
    in IF conditions it is an Operand."""
    neg: bool
    left: Union[str, Operand]
    cmp: str
    right: Operand

    def __str__(self) -> str:
        neg = "NOT " if self.neg else ""
        return f"{neg}{self.left} {self.cmp} {self.right}"


@dataclass(frozen=True)
class Pred:
    """clauses[0] (ops[0]) clauses[1] (ops[1]) clauses[2] ... ; ops are AND/OR.
    AND binds tighter than OR (matches JS && / || precedence)."""
    clauses: tuple
    ops: tuple

    def __str__(self) -> str:
        parts = [str(self.clauses[0])]
        for op, cl in zip(self.ops, self.clauses[1:]):
            parts.append(op)
            parts.append(str(cl))
        return " ".join(parts)


# ---------------------------------------------------------------- instructions
@dataclass
class Instr:
    line: int = field(default=0, kw_only=True)


@dataclass
class Let(Instr):
    op: Operand
    dst: Reg


@dataclass
class Get(Instr):
    src: Reg
    field: str
    dst: Reg


@dataclass
class SetF(Instr):
    src: Reg
    field: str
    op: Operand
    dst: Reg


@dataclass
class Call(Instr):
    tool: str  # 'T4'
    args: tuple
    dst: Optional[Reg]


@dataclass
class Format(Instr):
    """spec §4 FORMAT: fill a STR template constant's {0},{1},... slots with
    operand values -> STR. The only way a program produces new text, and it
    still emits none: the template is a constant, the values are data."""
    template: str  # 'C3'
    ops: tuple
    dst: Reg


@dataclass
class Filter(Instr):
    src: Reg
    pred: Pred
    dst: Reg


@dataclass
class MapF(Instr):
    src: Reg
    field: str
    dst: Reg


@dataclass
class Count(Instr):
    src: Reg
    dst: Reg


@dataclass
class Sort(Instr):
    src: Reg
    field: str
    dir: str  # ASC | DESC
    dst: Reg


@dataclass
class Select(Instr):
    src: Reg
    idx: Operand
    dst: Reg


@dataclass
class First(Instr):
    src: Reg
    dst: Reg


@dataclass
class Foreach(Instr):
    src: Reg
    var: Reg
    body: list


@dataclass
class If(Instr):
    cond: Pred
    then: list
    els: Optional[list]


@dataclass
class Parallel(Instr):
    calls: list  # of Call


@dataclass
class Try(Instr):
    retry: int
    dst: Reg
    body: list


@dataclass
class Return(Instr):
    op: Operand


@dataclass
class Stop(Instr):
    pass


@dataclass
class Pause(Instr):
    pass


# spec §4 ABORT: the planner declines to act. Reasons are a closed enum so
# the value channel stays out of the token stream.
ABORT_REASONS = ("NOT_FOUND", "AMBIGUOUS", "UNSUPPORTED", "NEEDS_INFO")


@dataclass
class Abort(Instr):
    reason: str


@dataclass
class Program:
    effects_decl: Optional[list]  # list of effect names, or None if no header
    body: list


# ---------------------------------------------------------------- task context
@dataclass
class ToolParam:
    sym: str          # field symbol used for this parameter ('F0')
    type: Type
    required: bool = True
    desc: str = ""


@dataclass
class ToolDecl:
    sym: str          # 'T4'
    name: str         # internal tool name, never shown to the model
    desc: str
    params: list      # of ToolParam
    returns: Optional[Type]
    effects: list     # of effect names


@dataclass
class FieldDecl:
    sym: str          # 'F7'
    entity: Optional[str]  # None for non-entity tool params
    name: str         # real field name in world state
    type: Type
    desc: str = ""


@dataclass
class ConstDecl:
    sym: str          # 'C3'
    type: Type
    value: object
    desc: str = ""


@dataclass
class TaskContext:
    tools: dict       # sym -> ToolDecl
    fields: dict      # sym -> FieldDecl
    constants: dict   # sym -> ConstDecl
    # Pre-bound registers for a continuation segment after PAUSE: reg name -> Type
    initial_registers: dict = field(default_factory=dict)

    def fields_of_entity(self, entity: str) -> dict:
        return {s: f for s, f in self.fields.items() if f.entity == entity}

    def to_json(self) -> dict:
        return {
            "tools": [
                {
                    "sym": t.sym, "name": t.name, "desc": t.desc,
                    "params": [
                        {"sym": p.sym, "type": format_type(p.type),
                         "required": p.required, "desc": p.desc}
                        for p in t.params
                    ],
                    "returns": format_type(t.returns) if t.returns else None,
                    "effects": list(t.effects),
                }
                for t in self.tools.values()
            ],
            "fields": [
                {"sym": f.sym, "entity": f.entity, "name": f.name,
                 "type": format_type(f.type), "desc": f.desc}
                for f in self.fields.values()
            ],
            "constants": [
                {"sym": c.sym, "type": format_type(c.type), "value": c.value,
                 "desc": c.desc}
                for c in self.constants.values()
            ],
            "initial_registers": {
                r: format_type(t) for r, t in self.initial_registers.items()
            },
        }

    @staticmethod
    def from_json(d: dict) -> "TaskContext":
        tools = {}
        for t in d.get("tools", []):
            tools[t["sym"]] = ToolDecl(
                sym=t["sym"], name=t.get("name", t["sym"]), desc=t.get("desc", ""),
                params=[
                    ToolParam(sym=p["sym"], type=parse_type(p["type"]),
                              required=p.get("required", True),
                              desc=p.get("desc", ""))
                    for p in t.get("params", [])
                ],
                returns=parse_type(t["returns"]) if t.get("returns") else None,
                effects=list(t.get("effects", [])),
            )
        fields = {}
        for f in d.get("fields", []):
            fields[f["sym"]] = FieldDecl(
                sym=f["sym"], entity=f.get("entity"), name=f["name"],
                type=parse_type(f["type"]), desc=f.get("desc", ""))
        constants = {}
        for c in d.get("constants", []):
            constants[c["sym"]] = ConstDecl(
                sym=c["sym"], type=parse_type(c["type"]), value=c["value"],
                desc=c.get("desc", ""))
        init = {r: parse_type(t)
                for r, t in d.get("initial_registers", {}).items()}
        return TaskContext(tools=tools, fields=fields, constants=constants,
                           initial_registers=init)
