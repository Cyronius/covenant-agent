# WebArena wording bank

**What it is.** WebArena is a web-agent benchmark on self-hosted copies of a
shop, a shop admin panel, a forum, GitLab, a map and a wiki. Its 812 tasks are
built from human-written intent templates with slots
(`Show me products under ${{price}} in "{{product_category}}" category`). This
bank holds those templates and the agent action vocabulary, for future page
and form worlds (plan, kind 5).

- **Source:** https://github.com/web-arena-x/webarena
- **Revision:** git commit `dce04686a56253aefba7b18a4fa0937cf1dc987b` (2025-11-26)
- **Licence:** Apache-2.0. Copy in `LICENSE.upstream`. No NOTICE file upstream.
- **Pulled:** 2026-09-29, shallow git clone.

## What was taken (235 rows in `bank.jsonl`)

| kind | rows | from |
|---|---|---|
| `intent_template` | 223 | `config_files/test.raw.json`, field `intent_template`. There are 241 distinct template strings under 190 `intent_template_id`s; 18 were dropped (below). `meta`: `template_id`, `sites`, `slots` (names only), `n_instances`, and `site_names_to_rewrite`. |
| `action` | 12 | The action space in `agent/prompts/raw/p_cot_id_actree_2s.py` (`click [id]`, `type [id] [content] [press_enter_after=0\|1]`, `hover`, `press`, `scroll`, `new_tab`, `tab_focus`, `close_tab`, `goto`, `go_back`, `go_forward`, `stop [answer]`), with category and description. |

`site_names_to_rewrite` lists names of WebArena's own sites and fixtures that
appear in the literal template text: "One Stop Market" and its variants,
GitLab, Reddit, r/books, and project names such as "AutoAGI". 44 templates carry
at least one. Swap in our own entities before use.

## Deliberately not taken

- **Slot values** (`instantiation_dict`), rendered `intent`s, reference answers,
  evaluators and start URLs. The values are real product, subreddit, repo and
  place names from the site content.
- **18 templates that name real brands or places in their fixed text** (no slot
  to abstract them): Nike, Sony, Anker, Oral B, EYZUTAK, XBox, Nintendo Switch,
  Hyatt, Pittsburgh, Philadelphia, CMU, Carnegie Mellon, Massachusetts, the
  Declaration of Independence, midjourney. Template ids 19 (one variant), 35, 47,
  49, 77, 120, 171, 204, 210, 666, 781, 782 and 1356.
- All site content: product catalogue, forum posts, repos, map data, and the
  accessibility-tree observations in the prompt examples.

**Caution:** WebArena is an evaluation set, and these templates come from its
test config. If we ever report a WebArena score, rows built from this bank
contaminate it.

## Attribution

> Intent templates and action vocabulary adapted from WebArena (Zhou et al.,
> "WebArena: A Realistic Web Environment for Building Autonomous Agents",
> ICLR 2024), https://github.com/web-arena-x/webarena, Apache License 2.0.
