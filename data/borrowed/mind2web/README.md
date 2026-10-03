# Mind2Web wording bank (train split only)

**What it is.** Mind2Web is a set of crowd-written web tasks on 137 real
websites ("Check for pickup restaurant available in Boston, NY on March 18, 5pm
with just one guest"), each with the recorded action sequence. This bank holds
the **train split's** task text, site and domain labels, and the short action
strings. It feeds future page and form worlds (plan, kind 5).

- **Source:** https://huggingface.co/datasets/osunlp/Mind2Web
- **Revision:** dataset `main` at `17ece8eb89862368edc0cc806acee6fca5163474`.
  The rows were read from the Hub's automatic parquet conversion
  (`refs/convert/parquet` at `eabe74c3532cf3a35ff02913cece5341bd1ca0d5`,
  `default/partial-train/*.parquet`, 8 files, 1009 rows), reading only the
  selected columns. *Unverified:* that the parquet conversion was built from that
  exact `main` commit. The Hub labels it "partial", but its row count (1009)
  matches the paper's train split.
- **Licence:** CC-BY-4.0 (dataset card). Attribution below is required.
- **Pulled:** 2026-09-29.

## What was taken (1014 rows in `bank.jsonl`)

| kind | rows | contents |
|---|---|---|
| `task` | 1009 | `confirmed_task` as `text`. `meta`: `split: "train"`, `annotation_id`, `website` (73 sites), `domain` (Travel 467, Shopping 281, Entertainment 261), `subdomain`, and `action_reprs`, the list of short action strings such as `[combobox]  Reservation type -> SELECT: Pickup`. |
| `action_op` | 5 | The operation vocabulary seen in `action_reprs` (CLICK, TYPE, SELECT, HOVER, ENTER), with counts and five sampled examples each. |

## Deliberately not taken

- **The test splits** (`test.zip`: cross-task, cross-website, cross-domain).
  Mind2Web asks that they not be used for training, and they carry canary
  strings. Nothing from `test.zip` was downloaded.
- Page captures: `raw_html`, `cleaned_html`, candidate elements
  (`pos_candidates`, `neg_candidates`), screenshots and `scores_all_data.pkl`.
- Per-action element IDs and structured `operation` objects. Only the
  human-readable `action_reprs` strings were kept.

**Note:** task text names real sites, businesses and places, and some tasks
contain made-up personal details ("allan.smith@gmail.com"). Rewrite with our
own entities before any row reaches the corpus.

## Attribution

> Task descriptions from Mind2Web (Deng et al., "Mind2Web: Towards a Generalist
> Agent for the Web", NeurIPS 2023 Datasets and Benchmarks), train split,
> https://huggingface.co/datasets/osunlp/Mind2Web, licensed CC BY 4.0
> (https://creativecommons.org/licenses/by/4.0/). Changes: only selected text
> fields were extracted, and they were rewritten for our worlds.
