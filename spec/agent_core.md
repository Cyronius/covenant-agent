# Agent Core IR — Specification (F1)

**Version:** 0.6.0 (0.6.0, 2026-09-12: `IN` - membership with the list on the right, legal in a `FILTER` clause; `CONTAINS` is substring only, §2/§3/§4/§12; 0.5.0, 2026-09-11: a `FILTER` clause may compare against a second field of the element, §3/§4/§12; 0.4.0, 2026-09-08: `MOST`/`LEAST`, `EMPTY`, normalized STR comparison, §2/§3/§4/§12; typed constant letters and constant kinds, §1/§6 — landing in the same series; 0.3.1, 2026-09-08: runtime errors as a segment boundary, §4/§9; 0.3.0, 2026-09-07: `ABORT` referents, §3/§4/§9; 0.2.0, 2026-09-02: `ABORT` terminator and `FORMAT`, §3/§4/§8)
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
| `S0`, `N0`, `B0`, `D0`, `I0`, … | Constant symbols (0.4.0). The letter is the base type — `S` STR, `N` INT, `B` BOOL, `D` TIME, `I` ID (the entity is in the declaration) — numbered per letter in declaration order. Values are held by the runtime binding supplied with the task input (extracted from the request by the serializer; exact in synthetic data). An `S` declaration carries a **kind**: `name` (a lookup key), `text` (content passed along), or `enum <entity>.<field>` (one of that field's declared values; the serializer emits every enum value of every entity the visible tools touch, whether or not the request spells it). Kinds are declaration metadata, not types: the typechecker ignores them; the per-task grammar and the corpus use them. `C0`, `C1`, … is the 0.3.x form, still parsed. |
| `NOW` | The current time, bound by the runtime. Type `TIME`. |
| `NULL` | The null value. Legal as an operand in a comparison, a `SET`, and an optional `CALL` slot; in a required `CALL` slot it is `MISSING_ARG` (§6). |

## 2. Types

```
INT  STR  BOOL  TIME  ID(entity)  OBJ(entity)  LIST(elem)  STATUS  NULL
```

- `TIME` is an integer Unix timestamp (seconds). Comparisons use `LT`/`GT`.
- `STR` comparison (`EQ`, `CONTAINS`, `IN`) is **case-folded and trimmed** (0.4.0):
  "cyrus" equals "Cyrus". Ids and enum values are canonical and unaffected.
  The model never chooses this, so it cannot get it wrong.
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

instr       = let | get | set | call | format | filter | map | count | sort
            | most | select | first | foreach | if | else | parallel | try
            | return | stop | pause | abort ;

let         = "LET" operand "->" reg ;
get         = "GET" reg "." field "->" reg ;
set         = "SET" reg field operand "->" reg ;
call        = "CALL" tool { operand } [ "->" reg ] ;
format      = "FORMAT" const { operand } "->" reg ;
filter      = "FILTER" reg pred "->" reg ;
map         = "MAP" reg field "->" reg ;
count       = "COUNT" reg "->" reg ;
sort        = "SORT" reg field dir "->" reg ;
dir         = "ASC" | "DESC" ;
most        = ( "MOST" | "LEAST" ) reg field [ reg ] "->" reg ;
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
abort       = "ABORT" reason { symbol } ;      (* at most two symbols *)
reason      = "NOT_FOUND" | "AMBIGUOUS" | "UNSUPPORTED" | "NEEDS_INFO" ;
symbol      = tool | field | const ;

operand     = reg | reg "." field | const | "NOW" | "NULL" | int ;
reg         = "r0" … "r15" ;
tool        = "T" int ;
field       = "F" int ;
const       = "C" int ;

pred        = clause { ("AND" | "OR") clause } ;   (* AND binds tighter than OR *)
clause      = [ "NOT" ] field cmp ( operand | field ) ;   (* right-hand field: same element, 0.5.0 *)
cond        = ccl { ("AND" | "OR") ccl } ;
ccl         = [ "NOT" ] ( operand cmp operand | "EMPTY" reg ) ;
cmp         = "EQ" | "LT" | "GT" | "CONTAINS" | "IN" ;   (* IN: right is a LIST, 0.6.0 *)
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
| `FORMAT C3 a b -> r1` | `C3` must be a `STR` constant whose value contains slots `{0}`, `{1}`, … — exactly one per operand. `r1` = the template with each slot replaced by the rendered operand (`STR` as is, `INT` as digits, `TIME` as an ISO date, `ID` as the id). Operands must be `STR`, `INT`, `TIME` or `ID(e)`. This is the only way a program produces new text, and it emits none: the template is a constant supplied by the serializer, the values are data. |
| `FILTER r0 p -> r1` | `r0 : LIST(OBJ(e))`; keep elements satisfying predicate `p`, whose field symbols resolve against `e` — on either side of a clause. `FILTER r0 F1 LT F2 -> r1` keeps the elements whose own `F1` is below their own `F2` ("over budget", "understaffed", "past its own deadline"), so the set is a value that `COUNT`, `SORT`, `FIRST` and `RETURN` can take. Before 0.5.0 that comparison was legal only in an `IF` inside a `FOREACH`, which can act on each element but never bind the set. `FILTER r0 F1 IN r2 -> r3` (0.6.0) keeps the elements whose `F1` appears in the list `r2`, so an intersection is a value too. |
| `MAP r0 F3 -> r1` | Project field `F3` over `LIST(OBJ(e))` → `LIST(field type)`. |
| `COUNT r0 -> r1` | Length of a list → `INT`. |
| `SORT r0 F3 ASC -> r1` | Stable sort of `LIST(OBJ(e))` by field. `ASC`/`DESC`. |
| `MOST r0 F3 [r2] -> r1` / `LEAST` | `r0 : LIST(OBJ(e))`, `F3` a field of `e`. `r1` = the value of `F3` shared by the most (fewest) elements of `r0`; type of `F3`; ties to the first key seen; `NULL` when there are no keys. With the optional candidate list `r2 : LIST(OBJ(e'))` — legal only when `F3 : ID(e')` — the keys are the candidates' ids in candidate order, **zeros included**, and elements keyed elsewhere are ignored. That is what makes `LEAST` answer "the user with the fewest overdue cards" when the answer has none. Without it `LEAST` is least-among-present. Added in 0.4.0 as the named form of the 14-line count-per-key loop a capable model would not find (results/R4.md). |
| `SELECT r0 i -> r1` | `i`-th element (0-based, `INT` operand). Out of range → runtime error `INDEX_OUT_OF_RANGE`. |
| `FIRST r0 -> r1` | First element. Empty list → `NULL` bound to `r1`. |
| `FOREACH r0 -> r1` | For each element of `r0` (in order), bind it to `r1` and run the body. `r1` remains bound to the last element after the loop (or is untouched when the list is empty — reading it after an possibly-empty loop is a typecheck warning, not an error). |
| `x IN r` | Membership: `r` must hold a `LIST` whose element type is compatible with `x`, records narrowing to their ids as they do under `EQ`. The only membership form since 0.6.0 — `CONTAINS` is substring on `STR` and nothing else. `IN` reads the same in an `IF` condition and in a `FILTER` clause, where it is the only way to put a list on the right. |
| `IF c` / `ELSE` | Standard branch. `ELSE` must immediately follow the `IF` body at the same indentation. A clause is `x cmp y` or the unary `EMPTY r` (true for an empty list or `NULL`; `r` must hold a `LIST`), so "did the lookup match anything" needs no numeric constant. |
| `PARALLEL` | Body must be only `CALL` lines. Calls are issued concurrently; all results are bound when the block ends. No result register may be read inside the block. |
| `TRY [RETRY n] -> r` | Run body. On a tool error inside, abort the body, and (if `RETRY n` and attempts remain) re-run it from the top, up to `n` additional attempts. `r` gets `OK` or the last error code (`STATUS`). Execution continues after the block. |
| `RETURN x` | Terminate program, final value `x`. |
| `STOP` | Terminate program, no value. |
| `ABORT reason` | Terminate program **without carrying out the request**, reporting why. Reads that already ran are fine; the point is that no further action is taken. A terminator: any following line at the same or an outer level is `UNREACHABLE`. Run status `aborted`, with the reason. |
| `PAUSE` | Terminate this **program segment**, returning all bound registers to the harness. The planner is re-invoked with the register state and emits a fresh continuation program whose registers arrive pre-bound. `PAUSE` is a terminator: any following line at the same or an outer level is `UNREACHABLE`. |

### Abstaining (`ABORT`)

The planner may only reference values that exist in its context, and may
only call tools that are listed. When a request cannot be carried out
faithfully, the correct program declines rather than guesses:

| reason | when |
|---|---|
| reason | when | referent |
|---|---|---|
| `NOT_FOUND C` | the request names a thing (a person, a record) and nothing matches it | the constant carrying the name |
| `UNSUPPORTED [C]` | the request needs an action no listed tool performs | optionally the constant carrying the request fragment that asks for it — there is no symbol for an absent verb, but the serializer usually extracted the words |
| `NEEDS_INFO F` | a required value (a title, an amount, a date) is neither in the request nor derivable from the context | the field or parameter whose value is missing |
| `AMBIGUOUS a [b]` | the request could refer to several things and the context does not disambiguate | what would resolve it: the candidate symbols when the choice is among symbols (`T2 T5`), or the field whose value would select among records (`F4`) |

**Referents.** A reason says what kind of failure; the referent says what it
is about, in the program's own symbol vocabulary. (`UNSUPPORTED` admitted
none until the 2026-09-07 referent probe, where Qwen3.8-27B twice wrote
`ABORT UNSUPPORTED C8` with `C8` the request text — "create standalone html
files" — and was rejected for it. The instinct is right; it is admitted.) — up to two `T`/`F`/`C`
symbols, never a register or a literal. The referent is what makes an abort
*checkable* (`NOT_FOUND C2` can be tested against the world; `NEEDS_INFO F3`
against the constant table) and *actionable* (a UI can ask for `F3` by its
description). Referents are optional in the grammar so earlier programs stay
valid, and expected wherever the reason admits one; a referent of the wrong
kind for its reason is a `TYPE_ERROR`, an undeclared one is `UNKNOWN_TOOL` /
`UNKNOWN_FIELD ABORT <sym>` / `UNBOUND <sym>`.

`NOT_FOUND` is a claim about the world, and the planner cannot see the
world statically. The faithful form is check-then-decline: fetch, filter on
the named value, and `ABORT NOT_FOUND C` inside the `IF` that finds nothing —
`ABORT` is legal inside an `IF` body for exactly this. A bare first-line
`ABORT NOT_FOUND` asserts absence the planner has not observed.

The reason is a closed enum and the referents are symbols, so the value
channel stays out of the token stream. An abstain task is scored correct
when the status *and* the reason match the reference (`correct_abstain`);
where the reference carries referents, `abort_referent_match` records
whether those matched too, and does not gate `correct_abstain`.

### Execution boundary (`PAUSE`)

`PAUSE` is the default execution model, not an ablation. A task run is a
sequence of segments:

```
segment 1 → PAUSE (registers out) → planner → segment 2 (registers in) → … → STOP/RETURN
```

A continuation segment's typechecking environment is seeded with the returned
register types; its input context includes the serialized register values.

A segment that raises outside `TRY` is a boundary of the same kind when the
harness runs reactively (`run_task(react_on_error=True)`, R4): the sandbox
reports the failing line and the registers bound so far, the state is what
the completed calls left (the raising call did not write), and the planner
is re-invoked with the rendered `RUNTIME` diagnostic (§9), the calls that
ran, and those registers pre-bound — typed from the failed program's final
environment. Without the reactive flag the error ends the task.

## 5. Registers

- 16 registers, `r0`–`r15`. Rebinding is allowed; the typechecker tracks the
  current type per program point.
- Reading an unbound register is `UNBOUND <reg>`.
- Registers bound inside an `IF` arm are only considered bound after the branch
  if **both** arms bind them (or the read is inside the arm).
- A `FOREACH` loop variable is an ordinary register.

## 6. Tools and schemas

A tool line in the model-facing input shows each parameter as its slot
letter with the parameter's field symbol: `T5 (I:user=F11 S=F0) -> - [SEND]`
takes a user id then a string (0.4.0). Under 0.3.x symbols the same line
reads `T5 (F11:ID:user F0:STR)`.

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
  parameter → `MISSING_ARG`; wrong type → `TYPE_ERROR`. An explicit `NULL`
  in a required slot is the absence of a value, so it is also
  `MISSING_ARG`; `NULL` belongs in an optional slot. Only the literal
  counts — a register that holds null still flows, which is what `MOST`
  with no keys and a result-less tool produce.
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
- `ABORT` reasons are not tool errors: they are the planner's own verdict,
  emitted statically, and never raised by a tool.

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
ABORT_UNFOUNDED <reason> <sym> <detail>
RUNTIME <code> line:<n> <detail>
```

`RUNTIME` is not static either: it is a sandbox error outside `TRY`, with
the tool error code (§8) or `JS_ERROR`, the 1-based source line of the
instruction that raised (0 if none ran), and the sandbox's message — e.g.
`RUNTIME INDEX_OUT_OF_RANGE line:3 index 0 of 0`. It is the failed
segment's verdict in a reactive run (§4, execution boundary).

`ABORT_UNFOUNDED` is not a static diagnostic: the harness emits it after
checking an abort's referent against the task (`harness/abort_check.py`) —
`NOT_FOUND C2` when a record matches `C2`'s value, `NEEDS_INFO F3` when a
constant of `F3`'s type is present, `AMBIGUOUS T5` when only one candidate
is named. It confirms or denies what the program itself asserted and reveals
nothing about the reference, which is what lets a repair round consume it.

`<site>` is `line:<n>` with the 1-based source line. Parsers report syntax
errors as structured `PARSE_ERROR line:<n> <detail>`, never Python exceptions.

`UNBOUND` covers registers *and* constant symbols: a `C` symbol with no
binding in the task context is an unbound symbol (there is no separate
`UNKNOWN_CONST` diagnostic).

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
- **Iframe/worker-portable** (SandwichTS-compatible): the emitted JS must run
  unchanged as an async function handed only its tool stubs — no ambient
  authority (no DOM, storage, network, `require`, globals), all effects via
  the injected `rt`, `PAUSE` mappable to a MessagePort round-trip, and no
  unbounded loops that would defeat a watchdog kill-switch. Only the `rt`
  adapter may differ between the Node `vm` harness and a browser worker.

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

### Admitting a named composite (the "standard library")

Every instruction is the name of one `rt.*` function, and each name costs
a grammar production, a spec row, a typecheck rule, a place in the small
model's vocabulary and corpus rows to teach it. A composite gets a name
only when all four hold:

1. a request class in the corpus needs it;
2. the composite is four or more lines in the IR;
3. a capable model fails to find the composite (measured, with reasoning
   on — results/R4.md is the template);
4. it types with the existing types.

`MOST`/`LEAST` and `EMPTY` (0.4.0) pass all four. String operations and
date arithmetic fail (1) by design: the model never emits text or computes
cutoffs, and that is a safety property, not a gap.

The same four tests apply, by analogy, to widening an existing production.
The 0.5.0 `FILTER` clause (a field on the right) is the one case so far: it
names no new instruction and no new type (4); the `FOREACH`/`IF` form it
replaces is not longer but *incapable* — it cannot bind the set, so nothing
downstream can count, sort or return it (2, stronger than asked); the
request class ("which are understaffed", "how many are over budget") is
unserved in the corpus because it was inexpressible (1); and test 3 is
vacuous rather than measured — there was no long form for a capable model to
find or miss, which `harness/schedule_probe.py` shows by construction. What
a model run *can* still say is whether the form gets used once taught, and
that is the S5 corpus's question (`results/FAMILIES.md` §4).
The other half of that gap — membership of a field in another list — is
0.6.0's `IN`, and it reads against the four tests the same way: no new type
or instruction (4), a `FOREACH`/`IF` workaround that cannot bind the
intersection so nothing can sort or count it (2), a request class that was
unserved because it was unsayable (1), and test 3 vacuous for the same
reason 0.5.0's was. Admitting `IN` alongside `CONTAINS`'s list arm would have
left two spellings of one job, so the list arm went: `CONTAINS` is substring
on `STR` and nothing else. Nothing depended on it: no entity field in any world
is `LIST`-typed — 141 theme worlds built through `data.gen.domains` plus the
14 registered ones, every field `STR`, `BOOL`, `TIME` or `ID` — and
`CONTAINS` appears in 0 of 56,000 S4c references.

Candidate missing primitives discovered during R1 are **listed for review** in
`results/R1.md`, never added directly. Any grammar change bumps the version at
the top of this file.
