# BFCL function-definition bank

**What it is.** The Berkeley Function Calling Leaderboard (BFCL) tests whether a
model calls the right function with the right arguments. Every test ships
JSON-schema function definitions (name, description, parameters). This bank
is a sample of those definitions, to serve as a distractor pool of tool schemas
(plan, "Also, from BFCL and ToolSandbox").

- **Source:** https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard
- **Revision:** git commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` of `ShishirPatil/gorilla`
  (sparse checkout of `berkeley-function-call-leaderboard/bfcl_eval/data`)
- **Licence:** Apache-2.0 (repo root `LICENSE`, copy in `LICENSE.upstream`). No NOTICE file upstream.
- **Pulled:** 2026-09-29.

## What was taken (562 rows in `bank.jsonl`, all `kind: function`)

`text` is the function description. `meta` has `name`, `parameters` (JSON
schema as upstream), `category` and `file`.

| category | rows | how chosen |
|---|---|---|
| `multi_turn_api` | 162 | **All** functions in `multi_turn_func_doc/*.json`: BFCL's stateful mock APIs (file system, messaging, posting, tickets, trading, travel booking, vehicle control, math, memory stores, web search). `meta.api` names the API and `meta.response` has its return schema. They are coherent tool sets, useful as whole worlds. |
| `simple_python` | 150 | seeded sample (`random.Random(0)`) of the 399 unique definitions |
| `multiple` | 80 | sample of 462 |
| `irrelevance` | 60 | sample of 238 |
| `parallel` | 30 | sample of 197 |
| `parallel_multiple` | 30 | sample of 474 |
| `simple_java` | 30 | sample of 100 |
| `simple_javascript` | 20 | sample of 50 |

Upstream uses `"type": "dict"` for objects and language-specific types in the
Java and JavaScript sets. They were kept verbatim; normalise at use.

## Deliberately not taken

- User questions, ground-truth calls (`possible_answer/`) and multi-turn
  conversations. The distractor pool needs schemas only, and the plan does not
  import labelled pairs (`data/open_pairs/README.md`: flat pairs were six times
  easier than our corpus).
- The `live_*` categories. Their functions and queries were contributed by users
  and include real companies' APIs, so their provenance is less clear than the
  BFCL team's own definitions.
- `memory_prereq_conversation/`, `format_sensitivity`, and `unused_datasets/`.

## Attribution

> Function definitions from the Berkeley Function Calling Leaderboard (Patil et
> al., "The Berkeley Function Calling Leaderboard (BFCL): From Tool Use to
> Agentic Evaluation of Large Language Models", ICML 2025; Gorilla project),
> https://github.com/ShishirPatil/gorilla, Apache License 2.0.
