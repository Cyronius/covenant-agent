"""Demo-style clutter in a row's constants (`data.gen --clutter`).

Generated rows list almost only the constants their program uses, and they
word them so the used ones stand out:

  - an enum value the sampler picked is "the done status"; the other values
    of that enum, added by `harness.context.enum_constants`, are
    `card.status "todo"`. Over 8,000 fdc25 rows the first wording was used
    92% of the time and the second 0%;
  - ID, BOOL and message constants were used 93-96% of the time.

The demo host (server/dev_server.py `constants_from_board`) lists every
person on the board, true and false, every status as "the X status", five
canned messages and the request itself, and a program uses one or two of
them. A planner trained on the generated rows puts all of them in
(results/R23.md). This module makes a row look like that:

  1. every enum value of the row is worded the same way, in a style drawn
     per row (one of them is the demo's "the done status");
  2. records the request does not name are added as ID constants, described
     the way the row's own ID constants are;
  3. true/false, canned messages, and the request as a writer brief are
     added where the row has none of their kind;
  4. the list is shuffled, so no position says which constant is used.

The reference program is written against constant positions ($k), so the
shuffle rewrites them; nothing else in the row changes.
"""
from __future__ import annotations

import random
import re

# the demo's five (server/dev_server.py GENERIC_MESSAGES) and more like them,
# so the planner cannot learn these five strings as "never used"
CANNED_MESSAGES = [
    "This needs your attention.",
    "Heads up — this is overdue.",
    "Following up on this — any update?",
    "Please take a look when you get a chance.",
    "Reminder: this is due soon.",
    "Can you confirm this is still on track?",
    "Quick nudge on this one.",
    "Thanks — this is done.",
    "Please review when you can.",
    "Checking in on the status of this.",
    "This is blocked; can you help?",
    "Just a reminder about this.",
    "Let me know if you have questions.",
    "Please update this when it changes.",
    "FYI — this changed today.",
]

BRIEF_DESC = "the request itself, verbatim (brief for write_text)"


def _sp(s: str, spaced: bool) -> str:
    return s.replace("_", " ") if spaced else s


# (entity, field, value, spaced) -> description; `spaced` is drawn once per
# row, so every value of a row is worded exactly alike
ENUM_STYLES = [
    lambda e, f, v, sp: f"the {_sp(v, sp)} {_sp(f, sp)}",       # the demo's
    lambda e, f, v, sp: f'{e}.{f} "{v}"',                       # enum_constants'
    lambda e, f, v, sp: f"{_sp(v, sp)}",
    lambda e, f, v, sp: f"{_sp(f, sp)} {_sp(v, sp)}",
    lambda e, f, v, sp: f"{_sp(f, sp)}: {v}",
    lambda e, f, v, sp: f"{_sp(e, sp)} {_sp(f, sp)} {_sp(v, sp)}",
    lambda e, f, v, sp: f'"{v}"',
    lambda e, f, v, sp: f"{_sp(v, sp)} ({_sp(f, sp)})",
]

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> set:
    return {w for w in _WORD.findall(str(text).lower()) if len(w) >= 3}


def _enum_map(world: dict, tools: list) -> dict:
    """lower(value) -> (entity, field, value) over the enums of entities the
    visible tools touch, as harness.context._kinded sees them."""
    from harness.context import _touched_entities
    touched = _touched_entities(world, tools)
    out: dict = {}
    for (entity, fname), values in sorted(world.get("enums", {}).items(),
                                          key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if entity in touched:
            for v in values:
                out.setdefault(str(v).strip().lower(), (entity, fname, v))
    return out


def _reword_enums(consts: list, world: dict, tools: list, rng: random.Random) -> list:
    pick, spaced = rng.choice(ENUM_STYLES), rng.random() < 0.5
    style = lambda e, f, v: pick(e, f, v, spaced)  # noqa: E731
    emap = _enum_map(world, tools)
    have = set()
    out = []
    for c in consts:
        c = dict(c)
        if c.get("type") == "STR":
            hit = emap.get(str(c["value"]).strip().lower())
            if hit and (not c.get("kind") or c["kind"].startswith("enum:")):
                e, f, v = hit
                c["desc"] = style(e, f, str(c["value"]))
                c["kind"] = f"enum:{e}.{f}"
                have.add(str(c["value"]).strip().lower())
        out.append(c)
    # the rest of every touched enum, worded the same way; build_context's
    # enum_constants then finds them all present and adds nothing
    for low, (e, f, v) in emap.items():
        if low not in have:
            out.append({"type": "STR", "value": v, "desc": style(e, f, str(v)),
                        "kind": f"enum:{e}.{f}"})
            have.add(low)
    return out


def _id_template(c: dict, rec: dict):
    """How this ID constant describes its record: (field, template) with the
    record's value of `field` cut out, or None."""
    desc = c["desc"]
    best = None
    for f, v in rec.items():
        if f == "id" or not isinstance(v, str) or len(v) < 2:
            continue
        if v in desc and (best is None or len(v) > len(rec[best])):
            best = f
    if best:
        return best, desc.replace(rec[best], "{}", 1)
    num = str(rec["id"]).rsplit("_", 1)[-1]
    if num.isdigit() and re.search(rf"\b{num}\b", desc):
        return "#num", re.sub(rf"\b{num}\b", "{}", desc, count=1)
    return None


def _fill(field: str, template: str, rec: dict) -> str | None:
    if field == "#num":
        num = str(rec["id"]).rsplit("_", 1)[-1]
        return template.format(num) if num.isdigit() else None
    v = rec.get(field)
    return template.format(v) if isinstance(v, str) and v else None


def _distractor_ids(consts: list, state: dict, request: str, rng: random.Random,
                    k: int) -> list:
    ents = state.get("entities", {})
    used_ids = {str(c["value"]) for c in consts if str(c.get("type", "")).startswith("ID:")}
    used_desc = {c["desc"].strip().lower() for c in consts}
    req = _words(request)
    shapes = []                                     # (entity, field, template)
    for c in consts:
        t = str(c.get("type", ""))
        if not t.startswith("ID:"):
            continue
        ent = t[3:]
        rec = next((r for r in ents.get(ent, []) if str(r.get("id")) == str(c["value"])), None)
        shape = _id_template(c, rec) if rec else None
        if shape:
            shapes.append((ent, *shape))
    if not shapes:
        # a row with no ID constant: the demo lists every person anyway, so
        # list some records of an entity that has a name
        named = [(e, f) for e, recs in sorted(ents.items()) if recs
                 for f in ("name", "title") if isinstance(recs[0].get(f), str)]
        if not named:
            return []
        e, f = rng.choice(named)
        tmpl = rng.choice(["{}", "{}'s " + e.replace("_", " ") + " id", "{}'s id"])
        shapes = [(e, f, tmpl)]
    out = []
    for _ in range(k * 4):
        if len(out) >= k:
            break
        ent, field, tmpl = rng.choice(shapes)
        cands = [r for r in ents.get(ent, []) if str(r.get("id")) not in used_ids]
        if not cands:
            continue
        rec = rng.choice(cands)
        desc = _fill(field, tmpl, rec)
        if not desc or desc.strip().lower() in used_desc:
            continue
        # a record the request names (even by one word) is not clutter: it
        # could be what an abstain or ambiguity task is about
        if field != "#num" and _words(rec.get(field, "")) & req:
            continue
        out.append({"type": f"ID:{ent}", "value": rec["id"], "desc": desc})
        used_ids.add(str(rec["id"]))
        used_desc.add(desc.strip().lower())
    return out


def _texty(c: dict) -> bool:
    d = c.get("desc", "").lower()
    return c.get("type") == "STR" and bool(re.search(r"message|notice|brief|text \"", d))


def clutter(constants: list, segments: list, world: dict, tools: list, state: dict,
            request: str, rng: random.Random) -> tuple[list, list, dict]:
    """(constants, segments in authoring form) -> the cluttered pair, and
    what was added, for provenance."""
    consts = _reword_enums(constants, world, tools, rng)
    n_enum_added = len(consts) - len(constants)
    added = {"ids": 0, "bools": 0, "messages": 0, "brief": 0,
             "enum_values": n_enum_added}
    extra = []
    if rng.random() < 0.7:
        extra += _distractor_ids(consts, state, request, rng, rng.randint(1, 5))
        added["ids"] = len(extra)
    if rng.random() < 0.5:
        have = {c["value"] for c in consts if c.get("type") == "BOOL"}
        for v in (True, False):
            if v not in have:
                extra.append({"type": "BOOL", "value": v, "desc": str(v).lower()})
                added["bools"] += 1
    if not any(_texty(c) for c in consts):
        if rng.random() < 0.5:
            for m in rng.sample(CANNED_MESSAGES, rng.randint(1, 5)):
                extra.append({"type": "STR", "value": m, "desc": f"message: {m}",
                              "kind": "text"})
                added["messages"] += 1
        if rng.random() < 0.4:
            extra.append({"type": "STR", "value": request, "desc": BRIEF_DESC,
                          "kind": "text"})
            added["brief"] = 1
    consts = consts + extra
    # ConstAlloc deduplicates on (type, value); keep that true after adding
    seen, uniq = set(), []
    for c in consts:
        key = (c["type"], repr(c["value"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    # shuffle, and point every $k in the program at the constant's new place
    order = list(range(len(uniq)))
    rng.shuffle(order)
    new_pos = {old: new for new, old in enumerate(order)}
    shuffled = [uniq[old] for old in order]
    n_orig = len(constants)

    def remap(m):
        k = int(m.group(1))
        if k >= n_orig:
            raise ValueError(f"${k} past the sampler's {n_orig} constants")
        return f"${new_pos[k]}"
    segs = [re.sub(r"\$(\d+)", remap, s) for s in segments]
    return shuffled, segs, added
