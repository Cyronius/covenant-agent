# WebShop wording bank

**What it is.** WebShop is a simulated shop: an agent reads a shopping instruction
("i need 12 inch blue hair extensions that are made from natural hair, and price
lower than 40.00 dollars"), then searches, opens items, picks options and buys,
using only `search[...]` and `click[...]`. This bank holds the instruction
wording, the goal templates, the action and button vocabulary, and the page
list. It feeds a future `shop_search` page world (plan, kind 5).

- **Source:** https://github.com/princeton-nlp/WebShop
- **Revision:** git commit `64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd` (2024-09-05)
- **Licence:** MIT for the code, Copyright (c) 2023 Princeton Natural Language
  Processing. Copy in `LICENSE.upstream`. The human instructions were written by
  Amazon Mechanical Turk workers and ship inside the MIT repo
  (`baseline_models/data/items_human_ins.json`). **YELLOW** in the licence audit:
  instructions and structure only, no product data.
- **Pulled:** 2026-09-29, shallow git clone.

## What was taken (3020 rows in `bank.jsonl`)

| kind | rows | from |
|---|---|---|
| `goal_template` | 2 | `web_agent_site/engine/goal.py`: how a goal is phrased. Human goals are `{instruction}, and price lower than {price_upper} dollars`. Synthetic goals add `with {option}: {value}, and ...` clauses. |
| `action` | 2 | `search[keywords]` and `click[value]` (`web_agent_site/envs/web_agent_text_env.py`) |
| `button` | 8 | Fixed clickables from `engine.py`: Back to Search, Next >, < Prev, Buy Now, Description, Features, Reviews, Attributes. The `role` text is our own summary. |
| `page` | 8 | Page types from `web_agent_site/templates/` (search, results, item, description, features, review, attributes, done). Names only. |
| `human_instruction` | 3000 | A seeded random sample (`random.Random(0)`) of the 11,724 unique MTurk instructions in `items_human_ins.json`. `meta`: `instruction_attributes` and `instruction_options`, the attribute and option phrases the worker was asked to mention. 44 have empty attributes; upstream skips those when it builds goals. |

## Deliberately not taken

- All product data: titles, descriptions, bullet points, reviews, prices and
  images (the scraped Amazon files `items_shuffle*.json` and `items_ins_v2*.json`
  are downloaded separately upstream and were not downloaded here).
- From the instruction file: ASINs (Amazon product IDs), MTurk `worker_id` and
  `assignment_id`, and the product's full attribute list.
- The other 8,724 instructions (a size cap; re-run with a different sample if
  more are needed), `human_goals.json`, the search-query maps, and imitation
  trajectories (`il_trajs_finalized_images.zip`).

**Note:** instructions sometimes name the scraped product's brand or model
("cosycost usb microphone"). Rewrite them with our own catalogue before use.

## Attribution

> Shopping instructions, goal templates and action vocabulary adapted from
> WebShop (Yao et al., "WebShop: Towards Scalable Real-World Web Interaction
> with Grounded Language Agents", NeurIPS 2022),
> https://github.com/princeton-nlp/WebShop, MIT License, Copyright (c) 2023
> Princeton Natural Language Processing.
