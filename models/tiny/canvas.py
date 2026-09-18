"""The structural canvas: keywords plus pointers, and the renderer back to IR text.

Under structural binding (`.claude/plans/npu-native-planner.md`, decision 7) a
canvas slot holds one of two things:

  a keyword   one of a few dozen fixed rows: IR instructions, effect names,
              comparators, structural tokens (NL, IND, ->), PAD, MASK, digits.
  a pointer   an index into THIS task's declared symbols: tool i, field j,
              constant k, or a register. There is no `T23` row anywhere. A
              symbol the task does not declare cannot be produced, because
              the vector it would point at does not exist.

Both live in one joint id space so the model can put a single softmax over
them. The layout is

    [ keywords (K) | tools (max_tool) | fields (max_field) | consts (max_const) | registers (32) ]

and a per-task mask hides the tool/field/const rows beyond what the task
declares. PAD is joint id 0 and MASK is joint id 1, as in the flat vocabulary,
so `mask_canvas` and the samplers need no change.

Registers are 32 targets: `r0`..`r15` (an operand) and `r0.`..`r15.` (the
receiver of a field access). A field access `r0.F6` is two slots, the marked
receiver `r0.` then the field pointer `F6`, exactly as `corpus.program_tokens`
splits it for the flat vocabulary too (`.claude/plans/canvas-field-access-split.md`).
The marker is what makes rendering a total function with no grammar
knowledge: `FILTER r0 F1` (plain register, then a field operand) and
`GET r0. F6` (receiver, then its field) differ by the marker alone, so
`render` is just `corpus.detokenize` over the per-slot surface tokens. A
dangling `r0.` renders verbatim and is the compiler's to reject.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch

from corpus import detokenize, program_tokens
from core.ir import ABORT_REASONS, CMPS, EFFECTS, NUM_REGISTERS  # noqa: E402  corpus put covenant on sys.path

PAD, MASK = "PAD", "MASK"

# Slot kinds, by joint-id range. Order matters: `Layout.kind_tensor` bucketizes.
KINDS = ("kw", "tool", "field", "const", "reg")
KW, TOOL, FIELD, CONST, REG = range(5)

# Instructions of spec/agent_core.md section 3, plus the header.
INSTRUCTIONS = ("EFFECTS", "LET", "GET", "SET", "CALL", "FORMAT", "FILTER", "MAP",
                "COUNT", "MOST", "LEAST", "SORT", "SELECT", "FIRST", "FOREACH",
                "IF", "ELSE", "PARALLEL", "TRY", "RETRY", "RETURN", "STOP",
                "PAUSE", "ABORT")
OPERAND_WORDS = ("EMPTY", "NOT", "AND", "OR", "NOW", "NULL", "ASC", "DESC")
STRUCTURAL = ("NL", "IND", "->")

_REG = re.compile(r"^r(\d{1,2})$")
_REGDOT = re.compile(r"^r(\d{1,2})\.$")
_TOOL = re.compile(r"^T\d+$")
_FIELD = re.compile(r"^F\d+$")
_CONST = re.compile(r"^[CSNBDI]\d+$")


class CanvasError(ValueError):
    """A program form the canvas cannot represent or round-trip."""


def base_keywords() -> list[str]:
    """The fixed keyword table, before corpus extras. PAD=0, MASK=1."""
    out = [PAD, MASK, *STRUCTURAL, *INSTRUCTIONS, *EFFECTS, *CMPS, *OPERAND_WORDS,
           *ABORT_REASONS, *[str(i) for i in range(10)]]
    assert len(out) == len(set(out)), "duplicate keyword"
    return out


def is_symbol(tok: str) -> bool:
    return bool(_REG.match(tok) or _REGDOT.match(tok) or _TOOL.match(tok)
                or _FIELD.match(tok) or _CONST.match(tok))


def build_keywords(programs: list[str]) -> list[str]:
    """Fixed table plus any non-symbol token the corpus uses that it lacks.

    Extras are appended so the fixed rows keep stable ids; a corpus that needs
    one is reported by the caller (prep.py prints them), because a keyword the
    fixed table does not know is a spec conversation, not a workaround.
    """
    kws = base_keywords()
    have = set(kws)
    extra: set[str] = set()
    for p in programs:
        for t in program_tokens(p):
            if t not in have and not is_symbol(t):
                extra.add(t)
    return kws + sorted(extra)


@dataclass(frozen=True)
class Layout:
    """Joint id ranges. Sizes are per-cache constants, not per task.

    `n_reg` counts both register forms: the first NUM_REGISTERS rows are the
    plain operands `rN`, the next NUM_REGISTERS are the receivers `rN.`.
    """
    n_kw: int
    max_tool: int
    max_field: int
    max_const: int
    n_reg: int = 2 * NUM_REGISTERS

    @property
    def tool0(self) -> int:
        return self.n_kw

    @property
    def field0(self) -> int:
        return self.tool0 + self.max_tool

    @property
    def const0(self) -> int:
        return self.field0 + self.max_field

    @property
    def reg0(self) -> int:
        return self.const0 + self.max_const

    @property
    def regdot0(self) -> int:
        return self.reg0 + NUM_REGISTERS

    @property
    def size(self) -> int:
        return self.reg0 + self.n_reg

    @property
    def n_ptr(self) -> int:
        return self.size - self.n_kw

    def offset(self, kind: int) -> int:
        return (0, self.tool0, self.field0, self.const0, self.reg0)[kind]

    def kind_of(self, jid: int) -> int:
        if jid < self.tool0:
            return KW
        if jid < self.field0:
            return TOOL
        if jid < self.const0:
            return FIELD
        if jid < self.reg0:
            return CONST
        return REG

    def kind_tensor(self, ids: torch.Tensor) -> torch.Tensor:
        """Kind index (0..4) for every joint id in `ids`."""
        bounds = torch.tensor([self.tool0, self.field0, self.const0, self.reg0],
                              device=ids.device)
        return torch.bucketize(ids, bounds, right=True)

    def to_dict(self) -> dict:
        return {"n_kw": self.n_kw, "max_tool": self.max_tool, "max_field": self.max_field,
                "max_const": self.max_const, "n_reg": self.n_reg}

    @classmethod
    def from_dict(cls, d: dict) -> "Layout":
        return cls(**{k: d[k] for k in ("n_kw", "max_tool", "max_field", "max_const", "n_reg")})


class TaskCodec:
    """Program text <-> canvas ids, bound to one task's declared symbols.

    Duck-compatible with `tok.OutVocab` where the samplers care: `.pad`,
    `.mask`, `decode(ids)` (one surface token per slot) and `render(ids)`
    (program text).
    """

    def __init__(self, keywords: list[str], layout: Layout,
                 tools: list[str], fields: list[str], consts: list[str]):
        if len(tools) > layout.max_tool or len(fields) > layout.max_field \
                or len(consts) > layout.max_const:
            raise CanvasError(f"task declares {len(tools)} tools, {len(fields)} fields, "
                              f"{len(consts)} constants; layout allows "
                              f"{layout.max_tool}/{layout.max_field}/{layout.max_const}")
        self.keywords = keywords
        self.layout = layout
        self.kw_id = {k: i for i, k in enumerate(keywords)}
        self.tools, self.fields, self.consts = list(tools), list(fields), list(consts)
        self.tool_id = {s: i for i, s in enumerate(tools)}
        self.field_id = {s: i for i, s in enumerate(fields)}
        self.const_id = {s: i for i, s in enumerate(consts)}

    pad = 0
    mask = 1

    def __len__(self) -> int:
        return self.layout.size

    @property
    def syms(self) -> dict:
        return {"tools": self.tools, "fields": self.fields, "consts": self.consts}

    # -- inverse: text -> ids --------------------------------------------

    def encode_tokens(self, text: str) -> list[int]:
        L = self.layout
        ids: list[int] = []
        for t in program_tokens(text):        # already splits r0.F6 -> r0. F6
            m = _REGDOT.match(t)
            if m:
                n = int(m.group(1))
                if n >= NUM_REGISTERS:
                    raise CanvasError(f"register out of range: {t!r}")
                ids.append(L.regdot0 + n)
                continue
            m = _REG.match(t)
            if m:
                n = int(m.group(1))
                if n >= NUM_REGISTERS:
                    raise CanvasError(f"register out of range: {t!r}")
                ids.append(L.reg0 + n)
            elif _TOOL.match(t):
                if t not in self.tool_id:
                    raise CanvasError(f"undeclared tool {t!r}")
                ids.append(L.tool0 + self.tool_id[t])
            elif _FIELD.match(t):
                if t not in self.field_id:
                    raise CanvasError(f"undeclared field {t!r}")
                ids.append(L.field0 + self.field_id[t])
            elif _CONST.match(t):
                if t not in self.const_id:
                    raise CanvasError(f"undeclared constant {t!r}")
                ids.append(L.const0 + self.const_id[t])
            elif t in self.kw_id:
                ids.append(self.kw_id[t])
            else:
                raise CanvasError(f"token {t!r} is not a keyword this canvas knows")
        return ids

    def encode(self, text: str, length: int | None = None, verify: bool = True) -> list[int]:
        """Canvas ids for a program, padded to `length`. Refuses to truncate.

        With `verify`, renders the ids back and demands byte-identity with the
        normalized input text, so a form the split does not cover fails here
        rather than corrupting a training target.
        """
        ids = self.encode_tokens(text)
        if verify:
            want = detokenize(program_tokens(text))
            got = self.render(ids)
            if got != want:
                raise CanvasError("render(encode(text)) differs from text:\n"
                                  f"--- want\n{want}--- got\n{got}")
        if length is not None:
            if len(ids) > length:
                raise CanvasError(f"program needs {len(ids)} slots, canvas has {length}")
            ids = ids + [self.pad] * (length - len(ids))
        return ids

    # -- forward: ids -> surface tokens -> text -------------------------

    def token(self, jid: int) -> str:
        """The surface token for one joint id. Absent pointers render as
        `?T` etc., which the compiler rejects; they never occur in targets."""
        L = self.layout
        k = L.kind_of(jid)
        i = jid - L.offset(k)
        if k == KW:
            return self.keywords[i]
        if k == TOOL:
            return self.tools[i] if i < len(self.tools) else "?T"
        if k == FIELD:
            return self.fields[i] if i < len(self.fields) else "?F"
        if k == CONST:
            return self.consts[i] if i < len(self.consts) else "?C"
        return f"r{i}" if i < NUM_REGISTERS else f"r{i - NUM_REGISTERS}."

    def decode(self, ids: list[int]) -> list[str]:
        return [self.token(int(i)) for i in ids]

    def kinds(self, ids: list[int]) -> list[str]:
        return [KINDS[self.layout.kind_of(int(i))] for i in ids]

    def render(self, ids: list[int]) -> str:
        """Canvas ids to program text. Total: a garbage canvas renders to
        garbage text and the compiler says so. The receiver marker is what
        re-joins `r0.` with `F6`; see corpus.detokenize."""
        return detokenize(self.decode(ids))

    # -- the grammar's keyword half, per task ---------------------------

    def keyword_mask(self, declared_effects: set[str]) -> list[bool]:
        """Which keyword rows this task may emit: everything except MASK, and
        an effect name only if some declared tool carries that effect. The
        pointer half of the grammar needs no mask: undeclared symbols have
        no row at all."""
        eff = set(EFFECTS)
        return [k != MASK and (k not in eff or k in declared_effects)
                for k in self.keywords]


def save_keywords(kws: list[str], path: Path) -> None:
    path.write_text(json.dumps(kws), encoding="utf-8")


def load_keywords(path: Path) -> list[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def context_symbols(ctx: dict) -> dict:
    """The per-task symbol lists in the order the serializer prints them,
    which is the order the region-A vectors take. `ctx` is the task's
    `context` dict as stored in the corpus."""
    tools = sorted((t["sym"] for t in ctx.get("tools", [])), key=lambda s: int(s[1:]))
    fields = sorted((f["sym"] for f in ctx.get("fields", [])), key=lambda s: int(s[1:]))
    consts = sorted((c["sym"] for c in ctx.get("constants", [])),
                    key=lambda s: (s[0], int(s[1:])))
    return {"tools": tools, "fields": fields, "consts": consts}


if __name__ == "__main__":
    from corpus import COVENANT, load

    ex = load(COVENANT / "data" / "s5_plain.jsonl", limit=3000)
    kws = build_keywords([e.target for e in ex])
    print(f"{len(kws)} keywords; extras beyond the fixed table: {kws[len(base_keywords()):]}")
    lay = Layout(len(kws), 18, 23, 10)
    lens = []
    for e in ex:
        codec = TaskCodec(kws, lay, **context_symbols(e.row["context"]))
        ids = codec.encode(e.target)
        lens.append(len(ids))
    lens.sort()
    print(f"{len(ex)} programs round-trip; slots p50 {lens[len(lens)//2]} "
          f"p95 {lens[int(len(lens)*.95)]} max {lens[-1]}")
    e = next(x for x in ex if "." in x.target)
    codec = TaskCodec(kws, lay, **context_symbols(e.row["context"]))
    ids = codec.encode(e.target)
    print(list(zip(codec.kinds(ids), codec.decode(ids))))
    # A dangling receiver must render verbatim, not be repaired.
    dangling = codec.encode_tokens("GET r0.F1 -> r1\n")[:2] + [codec.kw_id["NL"]]
    print("dangling receiver renders as:", repr(codec.render(dangling)))
