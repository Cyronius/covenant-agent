"""General-English text for teaching the reading stages vocabulary.

The stages have only ever read our own generated descriptions, so a word
the themes never used (*caddie*, *hoist*) reaches them as bytes. This builds
the public, permissively licensed text that the vocabulary push trains on
(`.claude/plans/vocab-push.md` parts B and C), from four sources and no
others:

  wordnet     every noun/verb/adjective/adverb synset's gloss and examples
              (WordNet licence: permissive, notice required)
  google      every `description` in every Google API Discovery document
              (Apache-2.0)
  aws         every `documentation` string in botocore's service models,
              latest API version per service (Apache-2.0)
  open_pairs  tool names and descriptions from the three imported
              function-calling sets (Apache-2.0, data/open_pairs/README.md)

Outputs, all under `--out`:

  texts.jsonl     {"text", "kind", "source"}  kind: gloss|example|api_desc|tool_desc
  names.jsonl     {"text", "source"}          raw identifiers, split later by
                                              the tokenizer code
  siblings.jsonl  WordNet co-hyponym families: the dictionary's own twin
                  decision, with answers that do not come from the teacher
  README.md       sources, licences, counts, and the leakage check

The rule that keeps this honest: nothing from the 42 exam worlds enters.
Every text that shares a lowercased 8-word run with an exam request, an exam
tool or parameter description, or a reserved world's theme file is dropped
before anything is written, and the drop counts go in the README.

  python -m data.gen.general_corpus --out data/general

Google documents are cached under `<out>/raw/google/`, so a rerun is offline.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import html
import json
import random
import re
import time
import urllib.request
import zlib
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent

# The three imported files and only them: `b3_draw.jsonl` re-draws rows
# already in these and `*_pilot.jsonl` are subsets (data/open_pairs/README.md).
OPEN_PAIRS = ("glaive_12k.jsonl", "hermes_full.jsonl", "toolace_4k.jsonl")

# The exam: 8,400 rows over the 42 reserved worlds, and their theme files.
EXAM_ROWS = ROOT / "data" / "s6_holdout_both.jsonl"
THEMES = ROOT / "data" / "gen" / "themes"

DISCOVERY = "https://discovery.googleapis.com/discovery/v1/apis"
FETCH_WORKERS = 12
FETCH_TIMEOUT = 30

MIN_WORDS, MAX_WORDS = 3, 60
MIN_LETTER_SHARE = 0.6
LEAK_RUN = 8                      # words in a shared run that counts as leakage
MIN_FAMILY, MAX_FAMILY = 2, 12
HELDOUT_MOD = 10                  # crc32(parent) % 10 == 0 -> heldout (~10%)

# --- cleaning ---------------------------------------------------------------

_WS = re.compile(r"\s+")
_URL = re.compile(r"(?:https?://|www\.)\S+")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")        # [text](url) -> text
_MD_MARK = re.compile(r"`+|\*\*|(?<!\w)__|__(?!\w)|^#+\s*|^\s*[-*]\s+", re.M)
_TAG = re.compile(r"<[^>]+>")
# Block-level HTML ends a thought even when the author left out the period
# (botocore's <li> items rarely have one), so it separates texts outright.
_BLOCK = re.compile(r"</?(?:p|li|ul|ol|br|div|h\d|dl|dt|dd|table|tr|note|"
                    r"important)\b[^>]*>", re.I)
_SENT = re.compile(r"(?<=\.)\s+")
_WORD = re.compile(r"[a-z0-9]+")
_LOWER3 = re.compile(r"\b[a-z]{3,}\b")


def strip_html(text: str) -> List[str]:
    """botocore documentation as plain-text blocks, one per paragraph/item.

    Some models escape their markup twice (`&lt;p&gt;`), so entities are
    resolved until stable before tags are removed; otherwise the tags
    survive into the text as literal `<p>`.
    """
    for _ in range(3):
        unescaped = html.unescape(text)
        if unescaped == text:
            break
        text = unescaped
    return [_TAG.sub(" ", b) for b in _BLOCK.split(text)]


def strip_markdown(text: str) -> str:
    return _MD_MARK.sub(" ", _MD_LINK.sub(r"\1", text))


def acceptable(text: str) -> bool:
    """Mostly letters, not just URLs, and at least one real lowercase word."""
    chars = text.replace(" ", "")
    if not chars or not _URL.sub("", text).strip():
        return False
    if sum(c.isalpha() for c in chars) / len(chars) < MIN_LETTER_SHARE:
        return False
    return bool(_LOWER3.search(text))


def chunks(text: str) -> List[str]:
    """Cleaned 3-60 word texts from one raw description.

    Short descriptions are kept whole; longer ones are split at sentence
    boundaries and each sentence judged on its own.
    """
    text = _WS.sub(" ", strip_markdown(text)).strip()
    words = len(text.split())
    pieces = [text] if words <= MAX_WORDS else _SENT.split(text)
    return [p for p in pieces
            if MIN_WORDS <= len(p.split()) <= MAX_WORDS and acceptable(p)]


def norm_key(text: str) -> str:
    return _WS.sub(" ", text).strip().lower()


def runs(text: str, n: int = LEAK_RUN) -> Iterator[Tuple[str, ...]]:
    words = _WORD.findall(text.lower())
    return (tuple(words[i:i + n]) for i in range(len(words) - n + 1))


# --- sources ----------------------------------------------------------------
# Each yields (kind, raw_text) for texts and ("name", identifier) for names;
# cleaning and dedupe happen once, in build().

def wordnet_items() -> Iterator[Tuple[str, str]]:
    from nltk.corpus import wordnet as wn
    for syn in wn.all_synsets():          # n, v, a, s (satellite adj), r
        yield "gloss", syn.definition()
        for ex in syn.examples():
            yield "example", ex
        for lemma in syn.lemma_names():
            yield "name", lemma.replace("_", " ")


def _walk_discovery(node, out: List[Tuple[str, str]]) -> None:
    """Every description in a Discovery document, plus method ids and
    parameter names. Generic over the tree so schemas, nested properties,
    `items` and `additionalProperties` are all reached."""
    if isinstance(node, dict):
        for key, val in node.items():
            if key == "description" and isinstance(val, str):
                out.append(("api_desc", val))
            elif key == "parameters" and isinstance(val, dict):
                out.extend(("name", p) for p in val)
            elif key == "methods" and isinstance(val, dict):
                out.extend(("name", m["id"]) for m in val.values()
                           if isinstance(m, dict) and isinstance(m.get("id"), str))
            _walk_discovery(val, out)
    elif isinstance(node, list):
        for val in node:
            _walk_discovery(val, out)


def _fetch_json(url: str, cache: Path):
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    req = urllib.request.Request(url, headers={"User-Agent": "covenant-agent"})
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT) as resp:
        raw = resp.read().decode("utf-8")
    doc = json.loads(raw)
    cache.write_text(raw, encoding="utf-8")
    return doc


def google_items(raw_dir: Path, log: Dict[str, int]) -> Iterator[Tuple[str, str]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    directory = _fetch_json(DISCOVERY, raw_dir / "_directory.json")
    items = [it for it in directory.get("items", []) if it.get("discoveryRestUrl")]

    def get(it):
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", it["id"])
        try:
            return _fetch_json(it["discoveryRestUrl"], raw_dir / f"{safe}.json")
        except Exception:                       # network, 404, bad JSON: skip
            return None

    with cf.ThreadPoolExecutor(FETCH_WORKERS) as pool:
        docs = list(pool.map(get, items))
    log["google_apis"] = len(items)
    log["google_failed"] = sum(d is None for d in docs)
    out: List[Tuple[str, str]] = []
    for doc in docs:
        if doc is not None:
            _walk_discovery(doc, out)
    yield from out


def _botocore_models() -> List[Path]:
    """service-2.json(.gz) for the latest API version of each service.
    Version directories are ISO dates, so the lexical max is the newest."""
    import botocore
    data = Path(botocore.__file__).parent / "data"
    models = []
    for svc in sorted(p for p in data.iterdir() if p.is_dir()):
        versions = sorted(v for v in svc.iterdir() if v.is_dir()
                          and ((v / "service-2.json").exists()
                               or (v / "service-2.json.gz").exists()))
        if versions:
            v = versions[-1]
            models.append(v / "service-2.json" if (v / "service-2.json").exists()
                          else v / "service-2.json.gz")
    return models


def aws_items(log: Dict[str, int]) -> Iterator[Tuple[str, str]]:
    models = _botocore_models()
    log["aws_services"] = len(models)
    for path in models:
        raw = (gzip.decompress(path.read_bytes()) if path.suffix == ".gz"
               else path.read_bytes())
        model = json.loads(raw.decode("utf-8"))
        docs = [model.get("documentation", "")]
        for op_name, op in model.get("operations", {}).items():
            yield "name", op_name
            docs.append(op.get("documentation", ""))
        for shape in model.get("shapes", {}).values():
            docs.append(shape.get("documentation", ""))
            for member_name, member in shape.get("members", {}).items():
                yield "name", member_name
                docs.append(member.get("documentation", ""))
        for doc in docs:
            for block in strip_html(doc or ""):
                yield "api_desc", block


def open_pairs_items() -> Iterator[Tuple[str, str]]:
    for fname in OPEN_PAIRS:
        with open(ROOT / "data" / "open_pairs" / fname, encoding="utf-8") as f:
            for line in f:
                for tool in json.loads(line)["context"]["tools"]:
                    yield "name", tool["name"]
                    yield "tool_desc", tool.get("desc") or ""
                    for p in tool.get("params", []):
                        yield "tool_desc", p.get("desc") or ""


# --- WordNet sibling families (plan part C) --------------------------------

def _contains_lemma(gloss: str, lemma: str) -> bool:
    return re.search(r"\b" + re.escape(lemma.lower()) + r"\b", gloss.lower()) is not None


def sibling_families() -> Iterator[dict]:
    """Co-hyponyms under one parent, the shape of our flip slots.

    A member whose gloss names its own lemma would be answerable by string
    match, and members sharing a lemma cannot be told apart by name, so both
    are dropped before the family is sized.
    """
    from nltk.corpus import wordnet as wn
    for pos in ("n", "v"):
        for parent in wn.all_synsets(pos):
            members = []
            for h in sorted(parent.hyponyms(), key=lambda s: s.name()):
                lemma = h.lemma_names()[0].replace("_", " ")
                gloss = h.definition()
                if _contains_lemma(gloss, lemma):
                    continue
                members.append({"synset": h.name(), "lemma": lemma,
                                "gloss": gloss, "examples": list(h.examples())})
            # Two siblings with the same first lemma (discard.n.02 and
            # discard.n.03 under abandonment.n.03) make the pick unanswerable.
            lemmas = Counter(m["lemma"].lower() for m in members)
            members = [m for m in members if lemmas[m["lemma"].lower()] == 1]
            if len(members) > MAX_FAMILY:
                keep = set(random.Random(parent.name()).sample(
                    range(len(members)), MAX_FAMILY))
                members = [m for i, m in enumerate(members) if i in keep]
            if len(members) < MIN_FAMILY:
                continue
            split = ("heldout" if zlib.crc32(parent.name().encode()) % HELDOUT_MOD == 0
                     else "train")
            yield {"parent": parent.name(), "pos": pos, "members": members,
                   "split": split}


# --- the exam, for the leakage check ---------------------------------------

def _strings(node) -> Iterator[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, dict):
        for v in node.values():
            yield from _strings(v)
    elif isinstance(node, list):
        for v in node:
            yield from _strings(v)


def exam_text() -> Tuple[List[str], List[str], List[str]]:
    """(all exam strings, exam tool descriptions, worlds without a theme file).

    Requests, tool descriptions and (stricter than needed) parameter
    descriptions from every exam row, plus every string value in each exam
    world's theme file.
    """
    texts, tool_descs, worlds = [], [], set()
    with open(EXAM_ROWS, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            worlds.add(row["world"])
            texts.append(row["request"])
            for tool in row["context"]["tools"]:
                tool_descs.append(tool.get("desc") or "")
                texts.extend(p.get("desc") or "" for p in tool.get("params", []))
    texts.extend(tool_descs)
    missing = []
    for w in sorted(worlds):
        path = next((p for p in (THEMES / f"gen_{w}.json", THEMES / f"{w}.json")
                     if p.exists()), None)
        if path is None:
            missing.append(w)
            continue
        texts.extend(_strings(json.loads(path.read_text(encoding="utf-8"))))
    return texts, tool_descs, missing


# --- build -------------------------------------------------------------------

def _split_identifier(name: str) -> List[str]:
    """camelCase / snake_case / spaced identifier -> lowercased words, only
    for the exam-word coverage figure (the tokenizer does its own split)."""
    return _WORD.findall(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name).lower())


def build(out: Path) -> dict:
    t0 = time.time()
    log: Dict[str, int] = {}
    sources = (("wordnet", wordnet_items()),
               ("google", google_items(out / "raw" / "google", log)),
               ("aws", aws_items(log)),
               ("open_pairs", open_pairs_items()))

    # Exact dedupe across sources, first source wins (the order above).
    texts: List[dict] = []
    names: List[dict] = []
    seen_text: Set[str] = set()
    seen_name: Set[str] = set()
    for source, items in sources:
        for kind, raw in items:
            if kind == "name":
                name = raw.strip()
                if name and name not in seen_name:
                    seen_name.add(name)
                    names.append({"text": name, "source": source})
                continue
            for text in chunks(raw):
                key = norm_key(text)
                if key not in seen_text:
                    seen_text.add(key)
                    texts.append({"text": text, "kind": kind, "source": source})
        print(f"  {source}: {sum(t['source'] == source for t in texts):,} texts, "
              f"{sum(n['source'] == source for n in names):,} names "
              f"({time.time() - t0:.0f}s)", flush=True)

    families = list(sibling_families())

    # Leakage: no corpus text may share an 8-word run with the exam.
    exam, exam_tool_descs, no_theme = exam_text()
    exam_runs = {r for t in exam for r in runs(t)}
    leaks = lambda t: any(r in exam_runs for r in runs(t))   # noqa: E731
    leaked = [t for t in texts if leaks(t["text"])]
    dropped = Counter(t["source"] for t in leaked)
    dropped_examples = leaked[:10]
    leaked_ids = {id(t) for t in leaked}
    texts = [t for t in texts if id(t) not in leaked_ids]
    # The sibling glosses and examples are texts too; hold them to the rule.
    sib_dropped = 0
    for fam in families:
        kept = []
        for m in fam["members"]:
            if leaks(m["gloss"]):
                sib_dropped += 1
                continue
            m["examples"] = [e for e in m["examples"] if not leaks(e)]
            kept.append(m)
        fam["members"] = kept
    families = [f for f in families if len(f["members"]) >= MIN_FAMILY]

    # For information: how much of the exam's description vocabulary the
    # corpus covers at all.
    exam_words = {w for d in exam_tool_descs for w in _WORD.findall(d.lower())}
    text_words = {w for t in texts for w in _WORD.findall(t["text"].lower())}
    name_words = {w for n in names for w in _split_identifier(n["text"])}
    coverage = {
        "exam_words": len(exam_words),
        "in_texts": len(exam_words & text_words),
        "in_texts_or_names": len(exam_words & (text_words | name_words)),
        "missing_sample": sorted(exam_words - text_words - name_words)[:60],
    }

    out.mkdir(parents=True, exist_ok=True)
    for fname, rows in (("texts.jsonl", texts), ("names.jsonl", names),
                        ("siblings.jsonl", families)):
        with open(out / fname, "w", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "texts": Counter((t["source"], t["kind"]) for t in texts),
        "names": Counter(n["source"] for n in names),
        "families": Counter(f["split"] for f in families),
        "family_pos": Counter((f["pos"], f["split"]) for f in families),
        "members": Counter(f["split"] for f in families
                           for _ in f["members"]),
        "dropped": dropped, "dropped_examples": dropped_examples,
        "sib_dropped": sib_dropped, "exam_runs": len(exam_runs),
        "exam_strings": len(exam), "no_theme": no_theme,
        "coverage": coverage, "log": log, "seconds": time.time() - t0,
    }
    (out / "README.md").write_text(readme(stats), encoding="utf-8", newline="\n")
    return stats


def _wordnet_licence() -> str:
    import nltk
    with nltk.data.find("corpora/wordnet/LICENSE").open() as f:
        return f.read().decode("utf-8").rstrip()


def readme(s: dict) -> str:
    rows = sorted(s["texts"].items())
    by_source = Counter()
    for (src, _), n in rows:
        by_source[src] += n
    cov = s["coverage"]
    fam, mem = s["families"], s["members"]
    L = [
        "# General-English corpus",
        "",
        "Built by `python -m data.gen.general_corpus --out data/general` "
        "(`data/gen/general_corpus.py`). Generated, not hand-edited; rerun to "
        "refresh. See `.claude/plans/vocab-push.md` parts B and C.",
        "",
        "## Sources and licences",
        "",
        "| source | what was taken | licence |",
        "|---|---|---|",
        "| `wordnet` | WordNet 3.0 via nltk: every synset's gloss and example "
        "sentences (n, v, adj, adv), lemma names, co-hyponym families | "
        "WordNet 3.0 licence (permissive, notice required; reproduced below) |",
        f"| `google` | Google API Discovery: every `description` in "
        f"{s['log'].get('google_apis', 0) - s['log'].get('google_failed', 0)} of "
        f"{s['log'].get('google_apis', 0)} documents, method ids, parameter names "
        "| Apache-2.0 |",
        f"| `aws` | botocore service models, latest API version of each of "
        f"{s['log'].get('aws_services', 0)} services: operation, shape and "
        "member `documentation` (HTML stripped), operation and member names "
        "| Apache-2.0 |",
        "| `open_pairs` | `glaive_12k`, `hermes_full`, `toolace_4k` tool names, "
        "tool and parameter descriptions | Apache-2.0 (`data/open_pairs/README.md`) |",
        "",
        "Nothing else: no Wikipedia, xLAM, APIGen or ToolBench.",
        "",
        "## Files",
        "",
        "- `texts.jsonl` `{text, kind, source}`: 3-60 word texts, markdown and "
        "HTML stripped, mostly-letter, exact-deduplicated (case and whitespace "
        "insensitive) across sources in the order wordnet, google, aws, "
        "open_pairs.",
        "- `names.jsonl` `{text, source}`: raw identifiers (WordNet lemmas with "
        "spaces for underscores), exact-deduplicated. Exempt from the leakage "
        "check.",
        "- `siblings.jsonl` `{parent, pos, members[{synset, lemma, gloss, "
        "examples}], split}`: every noun/verb synset's direct hyponyms, members "
        "whose gloss contains their own lemma or whose lemma repeats a "
        "sibling's dropped, 2-12 per family (a "
        "seeded sample of 12 when larger), `heldout` when "
        "`crc32(parent) % 10 == 0`.",
        "- `raw/google/`: the cached Discovery documents.",
        "",
        "## Counts",
        "",
        "| source | kind | texts |",
        "|---|---|---:|",
        *[f"| {src} | {kind} | {n:,} |" for (src, kind), n in rows],
        f"| **all** | | **{sum(by_source.values()):,}** |",
        "",
        "| source | names |",
        "|---|---:|",
        *[f"| {src} | {n:,} |" for src, n in sorted(s["names"].items())],
        f"| **all** | **{sum(s['names'].values()):,}** |",
        "",
        f"Sibling families: {fam['train']:,} train / {fam['heldout']:,} heldout "
        f"({mem['train']:,} / {mem['heldout']:,} members). By part of speech: "
        + ", ".join(f"{pos} {split} {n:,}"
                    for (pos, split), n in sorted(s["family_pos"].items())) + ".",
        "",
        "## Leakage check",
        "",
        f"Exam text: the {s['exam_strings']:,} strings of "
        "`data/s6_holdout_both.jsonl` (requests, tool descriptions, parameter "
        "descriptions) and every string value of the exam worlds' theme files "
        f"({s['exam_runs']:,} distinct lowercased {LEAK_RUN}-word runs). "
        + (f"Worlds with no theme file (their text is in the rows): "
           f"{', '.join(s['no_theme'])}. " if s["no_theme"] else "")
        + "A corpus text sharing any run was dropped before writing.",
        "",
        "| source | texts dropped |",
        "|---|---:|",
        *[f"| {src} | {s['dropped'].get(src, 0):,} |"
          for src in ("wordnet", "google", "aws", "open_pairs")],
        "",
        f"Sibling members dropped for the same reason: {s['sib_dropped']:,}.",
        "",
    ]
    if s["dropped_examples"]:
        L += ["Dropped texts (first 10):", ""]
        L += [f"- [{t['source']}] {t['text']}" for t in s["dropped_examples"]]
        L += [""]
    L += [
        "## Exam-word coverage (information only)",
        "",
        f"Of the {cov['exam_words']:,} distinct words in exam tool "
        f"descriptions, {cov['in_texts']:,} "
        f"({cov['in_texts'] / max(cov['exam_words'], 1):.1%}) appear in "
        f"`texts.jsonl` and {cov['in_texts_or_names']:,} "
        f"({cov['in_texts_or_names'] / max(cov['exam_words'], 1):.1%}) in "
        "texts or split names. Words are the exam's words, not its sentences; "
        "the check above is what keeps sentences out.",
        "",
        "Missing (sample): " + ", ".join(cov["missing_sample"]),
        "",
        "## WordNet licence",
        "",
        "```",
        _wordnet_licence(),
        "```",
        "",
    ]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/general")
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    s = build(out)
    print(f"texts {sum(s['texts'].values()):,}  names {sum(s['names'].values()):,}  "
          f"families {dict(s['families'])}  dropped {dict(s['dropped'])}  "
          f"coverage {s['coverage']['in_texts']}/{s['coverage']['exam_words']}  "
          f"{s['seconds']:.0f}s")


if __name__ == "__main__":
    main()
