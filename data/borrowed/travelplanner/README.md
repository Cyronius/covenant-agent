# TravelPlanner wording bank

**What it is.** TravelPlanner is a trip-planning benchmark. An agent uses search
tools (flights, stays, restaurants, attractions, distances) to build a
multi-day plan under a budget and "hard" user constraints (house rule, cuisine,
room type, transport). Plans are also checked against "commonsense" rules (no
repeated restaurants, minimum nights, and so on). This bank holds the tool
vocabulary and the constraint types, values and violation wording, for a
future `trip` world (plan, kind 7).

- **Source:** https://github.com/OSU-NLP-Group/TravelPlanner
- **Revision:** git commit `e52c87f4ac348a3410c46dc3553c519db5ec5e23` (2026-05-23)
- **Licence:** everything here comes from the **code**, which is MIT, Copyright (c)
  2024 OSU Natural Language Processing. Copy in `LICENSE.upstream`. The
  dataset (queries, reference plans, database) is CC-BY-4.0 and none of it was
  taken.
- **Pulled:** 2026-09-29, shallow git clone.

## What was taken (76 rows in `bank.jsonl`)

| kind | rows | from |
|---|---|---|
| `tool` | 8 | `agents/prompts.py` (`ZEROSHOT_REACT_INSTRUCTION`): FlightSearch, GoogleDistanceMatrix, AccommodationSearch, RestaurantSearch, AttractionSearch, CitySearch, NotebookWrite, Planner. `text` is the description; `meta` has the signature, parameter descriptions and the upstream example. |
| `hard_constraint` | 5 | budget, house rule, cuisine, room type, transportation, with value sets from `utils/query_element_selection.py` (house rule: parties, smoking, children under 10, visitors, pets; cuisine: 7 cuisines; room type: shared, not shared, private, entire; transportation: no flight, no self-driving). The `note` text is our summary of the code. |
| `difficulty_level` | 3 | easy, medium and hard: group sizes, and how many local constraints are drawn from which types. |
| `trip_shape` | 1 | Days to cities visited (3→1, 5→2, 7→3). |
| `commonsense_constraint` | 10 | Check names with the paper's terms (Within Sandbox, Complete Information, Within Current City, Reasonable City Route, Diverse Restaurants, Diverse Attractions, Non-conf. Transportation, Minimum Nights Stay). Two helper checks that `eval.py` doesn't score are marked `scored_in_eval_py: false`. |
| `violation_message` | 41 | Every live `return False, f"..."` message in `evaluation/commonsense_constraint.py` and `evaluation/hard_constraint.py`, read from the AST, with its placeholders verbatim (e.g. `The room type should be {question['local_constraint']['room type']}.`). |
| `plan_field` | 8 | Plan line labels (Day, Current City, Transportation, Breakfast, Attraction, Lunch, Dinner, Accommodation). |

## Deliberately not taken

- **The queries** (`osunlp/TravelPlanner` on Hugging Face and `finetuning_data/`).
  They were written by GPT-4, so they are RED.
- **Database records** (`database/`: flights, accommodations, restaurants,
  attractions, distances, cities): scraped or derived real-world data.
- Reference plans, annotations, the planner's worked example (real flight
  numbers and venue names), and prompt text beyond the tool list.

## Attribution

> Tool vocabulary and constraint definitions adapted from the TravelPlanner code
> (Xie et al., "TravelPlanner: A Benchmark for Real-World Planning with Language
> Agents", ICML 2024), https://github.com/OSU-NLP-Group/TravelPlanner, MIT
> License, Copyright (c) 2024 OSU Natural Language Processing.
