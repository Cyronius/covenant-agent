# Imported open function-calling pairs

Converted by `data/gen/convert_open.py` from three Apache-2.0, ungated
sources. What was taken is the request English and the tool schemas; the
program is a mechanical derivation re-verified through our own typechecker
and sandbox, not the dataset's label.

## Inventory, verified 2026-09-19

| file | rows | what it is |
|---|---|---|
| `glaive_12k.jsonl` | 4,347 | glaiveai/glaive-function-calling-v2 |
| `hermes_full.jsonl` | 349 | NousResearch/hermes-function-calling-v1 |
| `toolace_4k.jsonl` | 2,197 | Team-ACE/ToolACE |
| **total** | **6,893** | **3,235 distinct tool sets, 7,263 distinct tool names** |
| `b3_draw.jsonl` | 1,349 | **not a source.** A re-draw of rows already in the three above (`data/gen/draw_open.py`), kept only because the SFT mixes in `results/logs/finish_s*_corpus.sh` cite it by name as their open-data sample. |
| `*_pilot.jsonl` | — | exact subsets of their full files; they contribute nothing |

Counting `b3_draw.jsonl` and the pilots as sources is what produced the
inflated 8,242-row / ~4,300-tool-set / ~10,000-name figures that circulated
before this was checked. Per-file counts also double-count: 4,344 tool sets
and 10,111 names are sums, and the deduplicated unions are 25% and 28%
lower. `data/gen/open_pool.py:SOURCES` is the one list that decides what
counts, and it holds the three files above.

## What these rows are for

**Distractors, not worlds, and not training targets.** A tf-idf baseline over
request-vs-tool text scores 94.2% on these worlds as-is and 77.0% after
crowding them to 60 tools, against 13.4% on `s5_plain`: every imported
parameter is a bare scalar, every `returns` is null and every field carries
`entity: null`, so there is no `ID:`/`OBJ:` structure for tool selection to
go through. They are six times easier than the corpus the tiny model already
trains on, and crowding them does not change that.

Their value is the 7,263 tool names, which intersect the themed vocabulary in
8. So they enter the corpus through `python -m data.gen --inject-open MIN:MAX`,
as distractors inside themed worlds whose program, entities and typing are
untouched. See `.claude/plans/imported-schemas-as-distractors.md` and
`data/gen/open_pool.py`.

Two properties of foreign text that themed schemas never have, and that stop
a `models/tiny/prep.py` run after the corpus is already generated, are
normalised at the pool boundary: four descriptions contain a newline, which
splits a tool line in two, and 159 tools declare more than seven parameters,
which overflows a schema line (`--max-line 64` tokens; measured, 7 is the
longest list that always fits). Optional parameters past the seventh are
trimmed rather than the tool dropped, so 7,257 of the 7,263 names survive.

The rows themselves remain a candidate non-template-request eval suite — every
themed request is rendered by `data/gen/english.py` templates and these carry
real human phrasing — but nothing is built for that yet. The length worry does
not survive measurement: tokenized with the structural cache's own tokenizer,
**89.3% of the 6,893 requests already fit `prep.py --max-req 128`** (glaive p50
22 tokens and 100% under the cap, ToolACE p50 52 and 81.7% under). Only Hermes
is long — p50 311 tokens, 3.7% under — and it is 349 rows. So the suite wants a
length filter, not a raised cap: filtering costs 739 rows and leaves 6,154.

They also cannot be run as tasks as they stand: the converted rows carry
`context` / `reference` / `request` but no `world`, `state` or `sandbox`, so
`harness/run.py` has nothing to execute them against. That is the work the
suite is, and it is not done.
