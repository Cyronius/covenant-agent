# Agent Core IR — Specification (F1)

**Version:** 0.1.0
**Status:** Foundation draft. Every change to this document must land in the same
commit as the matching changes to `core/` (parser, typechecker, effects, compiler),
`data/gen/`, and `spec/examples/`, with round-trip tests passing.

Agent Core is a restricted, register-based IR emitted by the planner model and
deterministically compiled to JavaScript. It is deliberately small: the model's
job is tool orchestration, not programming.

---

## 1. Lexical structure

- A program is a sequence of **lines**, UTF-8 text.
- Blocks are expressed by **indentation**: exactly two spaces per nesting level.
  Tabs are illegal.
- `#` starts a comment to end of line. Blank lines are ignored.
- Tokens are separated by single spaces.

### Symbols

| Form | Meaning |
|------|---------|
| `r0` … `r15` | Registers. Fixed set of 16. Never generated variable names. |
| `T0`, `T1`, … | Tool symbols. Assigned **per request** by the input serializer. Tool names are never tokenized permanently; the tool's meaning is carried by its description and schema in the input context. |
| `F0`, `F1`, … | Field symbols. Assigned per request to `(entity, field)` pairs. Tool parameters reference field symbols where the parameter corresponds to an entity field, and fresh `F` symbols otherwise. |
| `C0`, `C1`, … | Constant symbols. Values are held by the runtime binding supplied with the task input (extracted from the request by the serializer; exact in synthetic data). |
| `NOW` | The current time, bound by the runtime. Type `TIME`. |
| `NULL` | The null value. |

## 2. Types

```
INT  STR  BOOL  TIME  ID(entity)  OBJ(entity)  LIST(elem)  STATUS  NULL
```

- `TIME` is an integer Unix timestamp (seconds). Comparisons use `LT`/`GT`.
- `ID(e)` is an opaque identifier of an entity `e`.
- `OBJ(e)` is a record whose fields are declared by the world schema for `e`.
- `LIST(t)` is a homogeneous list.
- `STATUS` is the result of a `TRY` block: the string `OK` or an error code
  (see §8).
- Constants carry the type declared in their binding.

## 3. Grammar (EBNF)

```ebnf
program     = [effects_line] block ;
effects_line= "EFFECTS" effect { effect } NL ;
effect      = "READ" | "WRITE" | "DELETE" | "SEND" | "PAY" | "EXTERNAL" ;
block       = { line } ;
line        = instr NL [ body ] ;
body        = INDENT block DEDENT ;              (* only after block heads *)

instr       = let | get | set | call | filter | map | count | sort
            | select | first | foreach | if | else | parallel | try
            | return | stop | pause ;

let         = "LET" operand "->" reg ;
get         = "GET" reg "." field "->" reg ;
set         = "SET" reg field operand "->" reg ;
call        = "CALL" tool { operand } [ "->" reg ] ;
filter      = "FILTER" reg pred "->" reg ;
map         = "MAP" reg field "->" reg ;
count       = "COUNT" reg "->" reg ;
sort        = "SORT" reg field dir "->" reg ;
dir         = "ASC" | "DESC" ;
select      = "SELECT" reg operand "->" reg ;
first       = "FIRST" reg "->" reg ;
foreach     = "FOREACH" reg "->" reg ;           (* block head *)
if          = "IF" cond ;                        (* block head *)
else        = "ELSE" ;                           (* block head; must follow IF at same level *)
parallel    = "PARALLEL" ;                       (* block head; body is CALL lines only *)
try         = "TRY" [ "RETRY" int ] "->" reg ;   (* block head *)
return      = "RETURN" operand ;
stop        = "STOP" ;
pause       = "PAUSE" ;

operand     = reg | reg "." field | const | "NOW" | "NULL" | int ;
reg         = "r0" … "r15" ;
tool        = "T" int ;
field       = "F" int ;
const       = "C" int ;

pred        = clause { ("AND" | "OR") clause } ;   (* AND binds tighter than OR *)
clause      = [ "NOT" ] field cmp operand ;
cond        = ccl { ("AND" | "OR") ccl } ;
ccl         = [ "NOT" ] operand cmp operand ;
cmp         = "EQ" | "LT" | "GT" | "CONTAINS" ;
int         = digit { digit } ;
```

Notes:
- Bare integer literals are permitted only as `SELECT` indices and `RETRY`
  counts and as comparison operands; all task-level values (names, dates,
  amounts, enum values) must come through `C` symbols so that the value channel
  stays out of the token stream.
- `ASC`/`DESC` are modifiers of `SORT`, not standalone primitives. They were part
  of the initial spec draft; without a direction, ordinal tasks
  ("second oldest") are inexpressible.

## 4. Instruction semantics

| Instruction | Semantics |
|---|---|
| `LET x -> r` | Bind value of operand `x` to `r`. |
| `GET r0.F3 -> r1` | Field extraction. `r0` must hold `OBJ(e)` and `F3` a field of `e`. |
| `SET r0 F3 x -> r1` | `r1` = copy of `r0` with field `F3` set to `x`. Pure; persistence only happens through tools. |
| `CALL T2 a b -> r` | Invoke tool `T2` with positional args matching the tool schema's parameter order. Result bound to `r` if present, else discarded. Errors: see §8. |
| `FILTER r0 p -> r1` | `r0 : LIST(OBJ(e))`; keep elements satisfying predicate `p`, whose field symbols resolve against `e`. |
| `MAP r0 F3 -> r1` | Project field `F3` over `LIST(OBJ(e))` → `LIST(field type)`. |
| `COUNT r0 -> r1` | Length of a list → `INT`. |
| `SORT r0 F3 ASC -> r1` | Stable sort of `LIST(OBJ(e))` by field. `ASC`/`DESC`. |
| `SELECT r0 i -> r1` | `i`-th element (0-based, `INT` operand). Out of range → runtime error `INDEX_OUT_OF_RANGE`. |
| `FIRST r0 -> r1` | First element. Empty list → `NULL` bound to `r1`. |
| `FOREACH r0 -> r1` | For each element of `r0` (in order), bind it to `r1` and run the body. `r1` remains bound to the last element after the loop (or is untouched when the list is empty — reading it after an possibly-empty loop is a typecheck warning, not an error). |
| `IF c` / `ELSE` | Standard branch. `ELSE` must immediately follow the `IF` body at the same indentation. |
| `PARALLEL` | Body must be only `CALL` lines. Calls are issued concurrently; all results are bound when the block ends. No result register may be read inside the block. |
| `TRY [RETRY n] -> r` | Run body. On a tool error inside, abort the body, and (if `RETRY n` and attempts remain) re-run it from the top, up to `n` additional attempts. `r` gets `OK` or the last error code (`STATUS`). Execution continues after the block. |
| `RETURN x` | Terminate program, final value `x`. |
| `STOP` | Terminate program, no value. |
| `PAUSE` | Terminate this **program segment**, returning all bound registers to the harness. The planner is re-invoked with the register state and emits a fresh continuation program whose registers arrive pre-bound. `PAUSE` is a terminator: any following line at the same or an outer level is `UNREACHABLE`. |

### Execution boundary (`PAUSE`)

`PAUSE` is the default execution model, not an ablation. A task run is a
sequence of segments:

```
segment 1 → PAUSE (registers out) → planner → segment 2 (registers in) → … → STOP/RETURN
```

A continuation segment's typechecking environment is seeded with the returned
register types; its input context includes the serialized register values.

## 5. Registers

- 16 registers, `r0`–`r15`. Rebinding is allowed; the typechecker tracks the
  current type per program point.
- Reading an unbound register is `UNBOUND <reg>`.
- Registers bound inside an `IF` arm are only considered bound after the branch
  if **both** arms bind them (or the read is inside the arm).
- A `FOREACH` loop variable is an ordinary register.

## 6. Tools and schemas

Each tool in the task input declares:

```json
{
  "sym": "T4",
  "desc": "Archive a card on the board",
  "params": [ { "sym": "F0", "type": "ID:card", "required": true, "desc": "card id" } ],
  "returns": "OBJ:card",
  "effects": ["WRITE"]
}
```

- `effects` ⊆ `{READ, WRITE, DELETE, SEND, PAY, EXTERNAL}`.
- Positional `CALL` arguments map to `params` in order. Missing required
  parameter → `MISSING_ARG`; wrong type → `TYPE_ERROR`.
- `ID:e` accepts an `ID(e)` value or an `OBJ(e)` (auto-narrowed to its id by
  the compiler); this keeps programs short (`CALL T9 r2 …` where `r2` is the
  loop element).

## 7. Effects

- A program's **static effect set** is the union of the declared effects of
  every tool that appears in a `CALL`, reachable or not.
- The optional `EFFECTS` header is a declaration. If present, the static effect
  set must be a subset of the declaration; each violation is
  `EFFECT_UNDECLARED <effect>`. If absent, no declaration check occurs.
- The runtime **effect gate** (see `runtime/sandbox.js`) blocks any `DELETE`,
  `SEND`, or `PAY` call unless the run carries an approval token; a blocked
  call halts the run with status `EFFECT_BLOCKED`. This is deterministic and
  independent of the model.

## 8. Errors

Tool calls can fail with runtime error codes:

```
NOT_FOUND  PERMISSION_DENIED  RATE_LIMITED  INVALID_ARGUMENT  PARTIAL_DATA
INDEX_OUT_OF_RANGE
```

- Inside `TRY`: the error aborts the body (after retries) and becomes the
  block's `STATUS` result.
- Outside `TRY`: the error halts the program; the run reports `exec_ok = false`
  with the code.

## 9. Static diagnostics

Structured diagnostics, used verbatim as model feedback in R4:

```
UNBOUND <reg>
TYPE_ERROR <site> <expected> <got>
UNKNOWN_TOOL <sym>
UNKNOWN_FIELD <reg> <sym>
MISSING_ARG <tool> <field>
UNREACHABLE <line>
EFFECT_UNDECLARED <effect>
```

`<site>` is `line:<n>` with the 1-based source line. Parsers report syntax
errors as structured `PARSE_ERROR line:<n> <detail>`, never Python exceptions.

## 10. Compilation

`core/compile.py` lowers the AST to a single `async function main(rt)` in
JavaScript. Requirements:

- **Deterministic**: the same AST must produce byte-identical JS.
- All effects go through the runtime object `rt` (`rt.call`, `rt.pause`);
  the emitted code contains no free identifiers besides `rt`.
- Registers compile to `let r0, …` locals; only registers used by the program
  are declared, in numeric order.
- `PARALLEL` compiles to `Promise.all` over `rt.call` invocations.
- `TRY`/`RETRY` compiles to a loop with `try`/`catch` on `rt.ToolError`.
- `PAUSE` compiles to `return rt.pause({r0: r0, …})` over the registers bound
  at that point.

## 11. Worked example

Request: *"Archive every overdue card, then message Bob about each one."*

Input context (abridged): `T4` = list cards → `LIST(OBJ(card))`; `F7` = card.due
(`TIME`); `T9` = send message(user id, text) `SEND`; `F0` = card.id; `C3` = the
message text; `C1` = Bob's user id.

```
EFFECTS READ WRITE SEND
CALL T4 -> r0
FILTER r0 F7 LT NOW -> r1
FOREACH r1 -> r2
  CALL T6 r2 -> r3
  CALL T9 C1 C3
STOP
```

Static effect set: `{READ, WRITE, SEND}` — covered by the declaration; the
`SEND` calls execute only when the run carries an approval token.

## 12. Change control

Candidate missing primitives discovered during R1 are **listed for review** in
`results/R1.md`, never added directly. Any grammar change bumps the version at
the top of this file.
