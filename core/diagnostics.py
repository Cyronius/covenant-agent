"""Structured diagnostics (spec §9). These strings are model feedback in R4,
so the rendered form is stable and part of the contract."""
from __future__ import annotations

from dataclasses import dataclass

CODES = (
    "PARSE_ERROR", "UNBOUND", "TYPE_ERROR", "UNKNOWN_TOOL", "UNKNOWN_FIELD",
    "MISSING_ARG", "UNREACHABLE", "EFFECT_UNDECLARED",
)


@dataclass(frozen=True)
class Diagnostic:
    code: str
    args: tuple
    line: int  # 1-based source line, 0 when not applicable

    def render(self) -> str:
        if self.code == "PARSE_ERROR":
            return f"PARSE_ERROR line:{self.line} {' '.join(map(str, self.args))}"
        if self.code == "TYPE_ERROR":
            expected, got = self.args
            return f"TYPE_ERROR line:{self.line} {expected} {got}"
        if self.code == "UNREACHABLE":
            return f"UNREACHABLE {self.line}"
        return f"{self.code} {' '.join(map(str, self.args))}"

    def __str__(self) -> str:
        return self.render()


def parse_error(line: int, detail: str) -> Diagnostic:
    return Diagnostic("PARSE_ERROR", (detail,), line)


def unbound(line: int, reg: str) -> Diagnostic:
    return Diagnostic("UNBOUND", (reg,), line)


def type_error(line: int, expected: str, got: str) -> Diagnostic:
    return Diagnostic("TYPE_ERROR", (expected, got), line)


def unknown_tool(line: int, sym: str) -> Diagnostic:
    return Diagnostic("UNKNOWN_TOOL", (sym,), line)


def unknown_field(line: int, reg: str, sym: str) -> Diagnostic:
    return Diagnostic("UNKNOWN_FIELD", (reg, sym), line)


def missing_arg(line: int, tool: str, param: str) -> Diagnostic:
    return Diagnostic("MISSING_ARG", (tool, param), line)


def unreachable(line: int) -> Diagnostic:
    return Diagnostic("UNREACHABLE", (), line)


def effect_undeclared(effect: str) -> Diagnostic:
    return Diagnostic("EFFECT_UNDECLARED", (effect,), 0)
