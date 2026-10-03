---
name: covenant-ir
description: Always emit Covenant Agent Core IR — the register-based program language this repo's planner produces (spec/agent_core.md) — instead of prose, pseudocode, or JS, whenever writing a reference program, worked example, task trace, generator template, or any demonstration of what the planner should output. Triggers on requests to write/trace/debug a program for a task, add a spec example, show what the model should emit, or explain a plan in executable form.
---

# Covenant Agent Core IR

This is a condensed, load-bearing copy of `spec/agent_core.md` (currently
version 0.8.0). **The spec file is canonical** — if this skill and the spec
ever disagree, the spec wins and this file is stale; re-derive from it rather
than trust memory. Every grammar change bumps the spec version and must land
with matching changes to `core/`, `data/gen/`, `spec/examples/` (PLAN.md §14).

## When to use this

Any time you would otherwise reach for prose steps, pseudocode, or hand-rolled
JS to describe what a task's program does — write real Agent Core IR instead.
That covers:

- a reference program for a new `spec/examples/*.ac` file
- a worked example inside a plan, result write-up, or explanation
- tracing what the planner should emit for a given request + tool set
- debugging a failing task by hand-writing the correct program
- generator templates (`data/gen/`) before they're rendered

Do **not** invent new instructions, opcodes, or syntax to make an example
read more naturally — if the language can't say it, that's a finding (see
§12 admission test below), not a license to freelance.

## Two forms — pick the right one

| Form | Symbols | Where it's used | Example |
|---|---|---|---|
| **Authoring form** | `@tool_name`, `@entity.field`, `@tool.param` (unlinked param), `$0 $1 …` (constants by declared position) | Hand-written examples (`spec/examples/*.ac`), generator templates, plans, explanations | `CALL @archive_card $0 -> r1` |
| **Serialized form** | `T0 T1…` (tools), `F0 F1…` (fields), `C0/S0/N0/B0/D0/I0…` (typed constants) | What the parser/typechecker/model actually see, assigned fresh per request | `CALL T4 S0 -> r1` |

**Default to authoring form** when writing anything a human will read or that
will be resolved later (`harness/authoring.py:resolve`) — it's what every
`spec/examples/*.ac` file uses and it doesn't require you to fabricate a
plausible symbol table. Only use serialized form when the point is to show
exactly what the parser/model consumes (e.g. explaining a `T5 (I:user=F11
S=F0) -> - [SEND]` tool line, or a diagnostic that names a `T`/`F`/`C` symbol).

Authoring-form file header convention (see any `spec/examples/*.ac`):

```
# <name>  (level <n>, world <world>)
# request: <the English request, verbatim>
# $0: <TYPE> = '<value>' — <human label>
# $1: ...
# expected status: <only if not the default "ok">
# injected errors: [...]   (only if the example exercises TRY/RETRY)

<program body>
```

## Lexical structure

- A program is a sequence of **lines**. Blocks are **indentation**, exactly
  two spaces per level. Tabs are illegal.
- `#` starts a comment to end of line. Blank lines ignored. Tokens separated
  by single spaces.

| Symbol | Meaning |
|---|---|
| `r0`…`r15` | Registers — fixed set of 16, never named variables |
| `T0,T1,…` | Tool symbols (serialized form only) |
| `F0,F1,…` | Field symbols — entity fields, or unlinked tool params |
| `S0,N0,B0,D0,I0,…` | Typed constants — `S`=STR, `N`=INT, `B`=BOOL, `D`=TIME, `I`=ID; `C0,C1,…` is the older untyped form, still legal |
| `NOW` | Current time, type `TIME` |
| `NULL` | Null — legal in a comparison or optional `CALL` slot; in a **required** slot it's `MISSING_ARG` |

Bare integer literals are legal **only** as `SELECT` indices, `RETRY` counts,
and comparison operands — every other value (names, dates, amounts, enum
values) must be a constant symbol so the value channel stays out of the
token stream. Never write a literal string or free text into a program body
except as the operand list to `FORMAT`, and even there the template itself
is a constant.

## Types

```
INT  FLOAT  STR  BOOL  TIME  ID(entity)  OBJ(entity)  LIST(elem)  STATUS  NULL
```

- `TIME` = Unix seconds; compare with `LT`/`GT`.
- `FLOAT` exists only for `avg`'s return type — nothing else produces or
  expects it.
- `STR` comparisons (`EQ`, `CONTAINS`, `IN`) are case-folded and trimmed.
  Ids/enum values are canonical, unaffected.
- `ID(e)` = opaque id of entity `e`. `OBJ(e)` = full record. A tool param
  typed `ID:e` also accepts an `OBJ(e)` register (auto-narrowed to its id).
- `LIST(t)` = homogeneous list. `STATUS` = result of a `TRY` block (`OK` or
  an error code).

## Grammar (EBNF)

```ebnf
program     = block ;
block       = { line } ;
line        = instr NL [ body ] ;
body        = INDENT block DEDENT ;              (* only after block heads *)

instr       = let | get | call | format | filter | map | count | sort
            | most | select | first | foreach | if | else | parallel | try
            | return | stop | pause | abort ;

let         = "LET" operand "->" reg ;
get         = "GET" reg "." field "->" reg ;
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
tool        = "T" int ;    (* @tool_name in authoring form *)
field       = "F" int ;    (* @entity.field or @tool.param in authoring form *)
const       = "C" int ;    (* $n in authoring form *)

pred        = clause { ("AND" | "OR") clause } ;   (* AND binds tighter than OR *)
clause      = [ "NOT" ] field cmp ( operand | field ) ;   (* right-hand field: same element *)
cond        = ccl { ("AND" | "OR") ccl } ;
ccl         = [ "NOT" ] ( operand cmp operand | "EMPTY" reg ) ;
cmp         = "EQ" | "LT" | "GT" | "CONTAINS" | "IN" ;   (* IN: right is a LIST *)
int         = digit { digit } ;
```

## Instructions: semantics and when to reach for each

| Instr | Does | Use it when |
|---|---|---|
| `LET x -> r` | Bind an operand into a register | You need a value in a register-only slot (e.g. `FILTER`'s `IN` right side), or you'll reference it by register repeatedly. Skip it for a one-shot `CALL` arg — pass the constant/`r.field` directly. |
| `GET r0.F3 -> r1` | Extract one field from an `OBJ` into its own register | The field is read more than once, or is needed after the source object falls out of scope (e.g. before a later `FOREACH`). For a single use, prefer the inline `r0.F3` operand form instead of a standalone `GET`. |
| `CALL T2 a b -> r` | Invoke a tool, positional args per schema | The only way to touch the world. Omit `-> r` when the result is never read (e.g. `send_message`, `archive_card`). |
| `FORMAT C3 a b -> r1` | Fill a `STR` constant's `{0} {1} …` slots with rendered operands | The **only** way to produce new text. The template is always a constant supplied in context — never author free text yourself. If a request needs generated prose beyond templating, that's a tool call (e.g. `@write_text`), not `FORMAT`. |
| `FILTER r0 p -> r1` | Keep elements of `LIST(OBJ(e))` matching a predicate | Narrowing a list to a value you'll `COUNT`/`SORT`/`RETURN`/pass on. Predicate can compare a field to a constant, to another field of the *same* element ("its own due date vs its own budget"), or `IN` a second list (intersection). Prefer this over `FOREACH`+`IF` whenever the result needs to be bound as a set — the loop form can act per-element but can't produce a value downstream. |
| `MAP r0 F3 -> r1` | Project one field over a list | Feeding `sum`/`avg`/`max`/`min` (which take `LIST INT`), or building a `LIST` for an `IN` check. |
| `COUNT r0 -> r1` | List length → `INT` | "how many". |
| `SORT r0 F3 ASC/DESC -> r1` | Stable sort by field | Paired with `SELECT`/`FIRST` for ordinal requests. |
| `SELECT r0 i -> r1` | `i`-th element (0-based) | "the Nth …" after a `SORT`. Out of range → `INDEX_OUT_OF_RANGE`. |
| `FIRST r0 -> r1` | First element, `NULL` if empty | "the oldest/newest/top …" — pairs with `SORT`. |
| `MOST/LEAST r0 F3 [r2] -> r1` | Value of `F3` shared by most/fewest elements | Grouping requests: "who has the most overdue cards". **Pass the candidate list `r2`** whenever the true answer could be zero-among-candidates (e.g. "fewest cards" should be able to name someone with none) — without it, `LEAST` only considers keys actually present. |
| `FOREACH r0 -> r1` | Iterate, binding each element to `r1` | Per-element side effects (a `CALL` per item) or per-element branching. Not for building a derived set — use `FILTER`/`MAP` for that. |
| `IF c` / `ELSE` | Branch on a comparison or `EMPTY reg` | A single decision point (e.g. one `ABORT` vs proceed, or "if any overdue, notify"). Keep multi-element narrowing in `FILTER`, not here. |
| `PARALLEL` | Concurrent `CALL`s only, no reads inside the block | The request implies simultaneous/independent actions, or two independent fetches with no ordering dependency. All results bind when the block ends. |
| `TRY [RETRY n] -> r` | Run body, retry on tool error, bind `STATUS` | The request implies flakiness/retry ("the API's been flaky, retry"), or you want to branch on success/failure afterward. |
| `RETURN x` | Terminate with a value | The request asks a question — the program's job is to answer it. |
| `STOP` | Terminate, no value | The request asks for an action, nothing to report back. |
| `PAUSE` | End this segment, return bound registers, planner re-invoked | The task needs mid-flight input (approval, a follow-up choice) before continuing. Terminator — nothing after it at the same/outer level. |
| `ABORT reason [refs]` | Decline, no further action taken | See below — never guess when the request can't be faithfully carried out. |

`x IN r` (membership) works identically in an `IF` condition and a `FILTER`
clause — it's the only way to put a list on the right of a comparison.
`CONTAINS` is substring-on-`STR` only; nothing else.

## `ABORT` — declining instead of guessing

| reason | when | referent (≤2 symbols: `T`/`F`/`C`, never a register/literal) |
|---|---|---|
| `NOT_FOUND C` | request names a thing and nothing matches | the constant carrying the name |
| `UNSUPPORTED [C]` | no listed tool performs the needed action | optionally the constant carrying the request fragment |
| `NEEDS_INFO F` | a required value is missing and not derivable | the field/param whose value is missing |
| `AMBIGUOUS a [b]` | request could mean several things, context doesn't disambiguate | candidate symbols (`T2 T5`) or the field that would disambiguate |

- `NOT_FOUND` is a claim about the world the planner can't see statically —
  the faithful form is **check-then-decline**: fetch, `FILTER` on the named
  value, `ABORT NOT_FOUND C` *inside* the `IF` that finds nothing. A bare
  first-line `ABORT NOT_FOUND` (no prior fetch) asserts absence you never
  observed — don't write that.
- Referents make the abort checkable (`NOT_FOUND C2` testable against the
  world) and actionable (a UI can ask for `F3` by description). Include one
  whenever the reason admits it.

## `PAUSE` — the execution boundary

`PAUSE` is the default model, not an ablation:

```
segment 1 → PAUSE (registers out) → planner → segment 2 (registers in) → … → STOP/RETURN
```

A continuation segment's registers arrive pre-bound and pre-typed. A runtime
error outside `TRY` is a boundary of the same kind in a reactive run — the
planner is re-invoked with the `RUNTIME` diagnostic and the registers bound
so far.

## Registers

16 total (`r0`–`r15`), rebinding allowed. Reading an unbound register is
`UNBOUND <reg>`. A register bound inside only one arm of an `IF` is
considered bound after the branch only if **both** arms bind it.

## Tools and effects

A tool declares `sym, name, desc, params[], returns, effects`. `effects` is
a subset of three composable properties (0.7.0 — not the old closed word
list): `mutates`, `irreversible`, `external`. Rendered on the model-facing
tool line as a compact code: `[M]` `[M!]` `[!X]` `[M!X]` `[X]` `[]` (letters
in fixed order, no separator).

- A program's static effect set = union of the effect properties of every
  tool it `CALL`s, reachable or not.
- Runtime gate: any `irreversible` call is blocked without an approval token
  (`EFFECT_BLOCKED`). Too many `mutates` calls in an unapproved run trips
  `BULK_WRITE`. Both deterministic, independent of the model — you don't
  need to work around them in a program, just know a gated call may not
  execute even though the program is correct.

`sum`, `avg`, `max`, `min` are always-present tools (not opcodes) over
`LIST INT` → `INT` (`avg` → `FLOAT`). Empty list: `sum`→`0`, others→`NULL`.
Reach for `MAP` then `CALL @sum`/`@avg`/`@max`/`@min` for totals/averages —
never invent a `SUM` instruction.

## Errors and diagnostics (for context, not to author by hand)

Tool runtime errors: `NOT_FOUND PERMISSION_DENIED RATE_LIMITED
INVALID_ARGUMENT PARTIAL_DATA INDEX_OUT_OF_RANGE`. Inside `TRY` these become
the block's `STATUS`; outside, they halt the program.

Static diagnostics you may see reported against a program: `UNBOUND`,
`TYPE_ERROR`, `UNKNOWN_TOOL`, `UNKNOWN_FIELD`, `MISSING_ARG`,
`UNREACHABLE`, `EFFECT_UNDECLARED`, `ABORT_UNFOUNDED`, `RUNTIME`,
`PARSE_ERROR`. If you're hand-writing an example and something wouldn't
typecheck, fix the program — don't route around the diagnostic.

## Worked example

Request: *"Archive every overdue card, then message Bob about each one."*

```
CALL @list_cards -> r0
FILTER r0 @card.due LT NOW -> r1
FOREACH r1 -> r2
  CALL @archive_card r2.@card.id -> r3
  CALL @send_message $0 $1
STOP
```

`$0` = Bob's user id, `$1` = the message text. Static effect set =
`{mutates, irreversible, external}` — the union of `archive_card`
(`mutates`) and `send_message` (`irreversible`+`external`); `send_message`
only actually fires under an approval token.

## Change control

Do not propose new instructions to make an example read more naturally. A
composite earns a name only when **all four** hold (spec §12): (1) a request
class in the corpus needs it, (2) it's 4+ lines of existing IR, (3) a capable
model demonstrably fails to find it (measured), (4) it types with existing
types — and even then, ask "can this be a tool instead?" first, since
anything over a fixed `LIST INT`/plain-value signature with no field-as-value
requirement is a tool, not a grammar change. If you hit something the
language genuinely can't express, say so and point at the gap — don't paper
over it with invented syntax.
