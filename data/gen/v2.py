"""Theme surface v2 + recipes L12–L18 (plan .claude/plans/lane-c-retrain.md
§C1/§C2): the request shapes the real sessions have and the S1 corpus did
not.

Surface (added to every theme by data/gen/domains.py, rule-derived):
  create_<child>(<parent>, <name>, <enum>?)          create from literals
  <child>_note entity + list/add/update/delete note  two-id (container + item)
  set_<child>_image(<child>, image)                  apply a generated asset
  write_text(brief, data) -> STR      [EXTERNAL]     the writer tool (A8)
  generate_image(prompt, style?) -> STR [EXTERNAL]   the image tool
  search_docs(query) -> STR           [READ]         knowledge search

Recipes:
  12 format_reminder   get -> FORMAT -> send            (templated text)
  13 create            create_<child> literals [+ one set_* on the result]
  14 report            list [FILTER] [COUNT] -> RETURN
  15 content           write_text -> send | generate_image -> set_image
  16 abort_v2          ABORT AMBIGUOUS | NEEDS_INFO | UNSUPPORTED
  17 search            search_docs -> RETURN | -> send
  18 note_edit         add note | list notes -> FILTER title -> update/delete
"""
from __future__ import annotations

import random
from typing import Optional

from .programs import (GenSample, SampleError, _named_record, _records,
                       _sample_clauses, action_line, build_action,
                       clause_expr, matches, visible_actions)

V2_RECIPES = {"format_reminder", "create", "report", "content_send",
              "content_image", "abort_ambiguous", "abort_needs_info",
              "abort_unsupported", "search", "note_add", "note_edit",
              "note_delete"}

# Domain-neutral text pools. Kept short and plain so nothing here reads as
# a real person or company.
BRIEFS = [
    "summarize them and ask for a plan", "give a short status update",
    "say which ones need attention this week", "ask for an update by Friday",
    "thank them and list what is still open", "flag the ones that look stuck",
    "keep it to three sentences", "be friendly but firm about the deadline",
]
IMAGE_PROMPTS = [
    "a sunrise over the harbour", "a clean line-art icon of a checklist",
    "a watercolour map of the district", "a photorealistic office desk",
    "a cartoon delivery van", "a minimalist chart on a whiteboard",
    "a team high-fiving in a warehouse", "a calm forest path in autumn",
]
IMAGE_STYLES = ["cartoon", "photorealistic", "watercolour", "line art", "flat"]
QUESTIONS = [
    "how do I reset a password", "what is the refund policy",
    "where are the export settings", "how are duplicates handled",
    "what does the escalation process look like", "how long is data retained",
    "which browsers are supported", "how do I add a second approver",
    "what counts as an overdue item", "how do I change the default currency",
]
NOTE_TITLES = ["Follow-up", "Context", "Blocker", "Decision", "Handover",
               "Risks", "Next steps", "Customer ask", "Budget", "Timeline"]
NOTE_TEXTS = [
    "Waiting on a reply from the client.", "Needs a second review.",
    "Agreed in the weekly sync.", "Escalate if nothing moves by Friday.",
    "Costs are within the approved range.", "Owner changed last week.",
    "Check the numbers before sending.", "Nothing further to do here.",
]
UNSUPPORTED = [
    ("export {np} to a spreadsheet", "export"), ("print {obj}", "print"),
    ("translate {obj} into Spanish", "translate"),
    ("merge {obj} with its duplicates", "merge"), ("fax {obj} to accounting", "fax"),
    ("undo the last change to {obj}", "undo"),
    ("book a meeting room to discuss {obj}", "book"),
]
CREATE_VERBS = ["create", "add", "open", "set up", "make", "log", "register"]
REPORT_LIST = ["list {q} {np}", "show me {q} {np}", "which {np} are there",
               "what {np} do we have", "give me the {np}", "pull up {q} {np}"]
REPORT_COUNT = ["how many {np} do we have", "count the {np}",
                "how many {np} are there", "tell me the number of {np}"]


def _v2(profile) -> dict:
    v = profile.get("v2")
    if not v:
        raise SampleError("no v2 surface")
    return v


def _new_title(v2, state, rng) -> str:
    used = {r.get(v2["name_field"]) for r in _records(state, v2["child"])}
    pool = [t for t in v2["titles"] if t not in used]
    if not pool:
        raise SampleError("no unused titles")
    return rng.choice(pool)


def _parent(v2, state, rng, alloc):
    recs = _records(state, v2["parent"])
    if not recs:
        raise SampleError("no parents")
    r = rng.choice(recs)
    return r, alloc.get(f"ID:{v2['parent']}", r["id"], f"{r['name']}'s {v2['parent_noun'][0]} id")


# ------------------------------------------------------------ L12 format
def sample_format(world, profile, state, now, rng, alloc, holdout):
    v2 = _v2(profile)
    child, prof = v2["child"], profile["entities"][v2["child"]]
    rec, display, ref = _named_record(prof, child, state, rng, alloc)
    slot0 = v2["name_field"] or v2["enum_field"]
    tf = v2["time_field"]
    words0 = v2["name_field"] or v2["enum_field"]
    tpl = "Reminder: {0} is due {1}."
    tref = alloc.get("STR", tpl,
                     f"message template: {tpl} (fill {{0}} with the {prof['noun'][0]}'s "
                     f"{words0}, {{1}} with its {tf.replace('_', ' ')})")
    seg = (f"CALL @{prof['get_tool']} {ref} -> r0\n"
           f"FORMAT {tref} r0.@{child}.{slot0} r0.@{child}.{tf} -> r1\n"
           f"CALL @{v2['send']} r0.@{child}.{v2['ref_field']} r1\nSTOP\n")
    frame = {"recipe": "format_reminder", "noun": prof["noun"], "record": display,
             "parent_noun": v2["parent_noun"], "slot0": words0.replace("_", " "),
             "slot1": tf.replace("_", " ")}
    return GenSample(frame, [seg], alloc.items, tags=["format"])


# ------------------------------------------------------------ L13 create
def sample_create(world, profile, state, now, rng, alloc, holdout):
    v2 = _v2(profile)
    child, prof = v2["child"], profile["entities"][v2["child"]]
    parent, pref = _parent(v2, state, rng, alloc)
    args = [pref]
    title = None
    if v2["name_field"]:
        title = _new_title(v2, state, rng)
        args.append(alloc.get("STR", title, f'the text "{title}"'))
    enum_phrase = None
    if not v2["name_field"] or rng.random() < 0.5:
        val = rng.choice(v2["enum_values"])
        args.append(alloc.get("STR", val, f"the {val} {v2['enum_field']}"))
        enum_phrase = v2["enum_phrases"][val][0]
    follow = None
    if rng.random() < 0.4:
        # a follow-up that keeps the record: set_enum / set_bool, never delete
        acts = [a for a in visible_actions(prof, holdout, exclude_kinds={"send_field", "set_ref"})
                if a["tool"] != v2["create"] and a.get("dest")]
        if acts:
            follow = build_action(rng.choice(acts), child, {"<v>": f"r0.@{child}.id"},
                                  state, rng, alloc)
    seg = f"CALL @{v2['create']} {' '.join(args)} -> r0\n"
    if follow:
        seg += action_line(follow, "r1" if follow["dest"] else None) + "\n"
    seg += "STOP\n"
    frame = {"recipe": "create", "noun": prof["noun"], "title": title,
             "parent_name": parent["name"], "enum_phrase": enum_phrase,
             "follow": follow, "parent_noun": v2["parent_noun"]}
    return GenSample(frame, [seg], alloc.items, tags=["create"])


# ------------------------------------------------------------ L14 report
def sample_report(world, profile, state, now, rng, alloc, holdout):
    v2 = _v2(profile)
    child, prof = v2["child"], profile["entities"][v2["child"]]
    n = rng.choice([0, 1, 1, 2])
    clauses = []
    if n:
        for _ in range(6):
            trial = alloc.clone()
            clauses = _sample_clauses(prof, child, state, now, rng, trial, n, False)
            if matches(state, child, clauses):
                alloc.adopt(trial)
                break
        else:
            raise SampleError("no matching records")
    count = rng.random() < 0.4
    seg = f"CALL @{prof['list_tool']} -> r0\n"
    reg = "r0"
    if clauses:
        seg += f"FILTER r0 {clause_expr(clauses)} -> r1\n"
        reg = "r1"
    if count:
        seg += f"COUNT {reg} -> r2\n"
        reg = "r2"
    seg += f"RETURN {reg}\n"
    frame = {"recipe": "report", "noun": prof["noun"], "count": count,
             "clauses": [{"phrase": c["phrase"], "neg": False} for c in clauses]}
    return GenSample(frame, [seg], alloc.items, tags=["report"])


# ------------------------------------------------------------ L15 content
def sample_content(world, profile, state, now, rng, alloc, holdout):
    v2 = _v2(profile)
    child, prof = v2["child"], profile["entities"][v2["child"]]
    if rng.random() < 0.5:
        # write_text over a filtered list, sent to a parent
        for _ in range(6):
            trial = alloc.clone()
            clauses = _sample_clauses(prof, child, state, now, rng, trial, 1, False)
            if matches(state, child, clauses):
                alloc.adopt(trial)
                break
        else:
            raise SampleError("no matching records")
        parent, pref = _parent(v2, state, rng, alloc)
        brief = rng.choice(BRIEFS)
        bref = alloc.get("STR", brief, f"the brief, verbatim: {brief}")
        seg = (f"CALL @{prof['list_tool']} -> r0\n"
               f"FILTER r0 {clause_expr(clauses)} -> r1\n"
               f"CALL @{v2['writer']} {bref} r1 -> r2\n"
               f"CALL @{v2['send']} {pref} r2\nSTOP\n")
        frame = {"recipe": "content_send", "noun": prof["noun"], "brief": brief,
                 "parent_name": parent["name"],
                 "clauses": [{"phrase": c["phrase"], "neg": False} for c in clauses]}
        return GenSample(frame, [seg], alloc.items, tags=["content", "writer"])
    rec, display, ref = _named_record(prof, child, state, rng, alloc)
    prompt = rng.choice(IMAGE_PROMPTS)
    pref = alloc.get("STR", prompt, f"the image prompt, verbatim: {prompt}")
    style = rng.choice(IMAGE_STYLES) if rng.random() < 0.4 else None
    sref = alloc.get("STR", style, f"image style: {style}") if style else ""
    seg = (f"CALL @{v2['image']} {pref}{(' ' + sref) if sref else ''} -> r0\n"
           f"CALL @{v2['set_image']} {ref} r0\nSTOP\n")
    frame = {"recipe": "content_image", "noun": prof["noun"], "record": display,
             "prompt": prompt, "style": style}
    return GenSample(frame, [seg], alloc.items, tags=["content", "image"])


# ------------------------------------------------------------ L16 abort
def sample_abort_v2(world, profile, state, now, rng, alloc, holdout):
    v2 = _v2(profile)
    child, prof = v2["child"], profile["entities"][v2["child"]]
    flavour = rng.choice(["ambiguous", "ambiguous", "needs_info", "unsupported", "unsupported"])
    if flavour == "ambiguous":
        if len(_records(state, child)) < 2:
            raise SampleError("ambiguous needs 2+ records")
        acts = visible_actions(prof, holdout, exclude_kinds={"send_field"})
        if not acts:
            raise SampleError("no actions")
        info = build_action(rng.choice(acts), child, {"<v>": "r0"}, state, rng, alloc)
        frame = {"recipe": "abort_ambiguous", "noun": prof["noun"], "action": info}
        # spec §4 0.3.0: the referent is the field whose value would pick one
        nf = v2.get("name_field")
        seg = f"ABORT AMBIGUOUS @{child}.{nf}\n" if nf else "ABORT AMBIGUOUS\n"
        return GenSample(frame, [seg], alloc.items, tags=["abort", "ambiguous"])
    if flavour == "needs_info":
        if not v2["name_field"]:
            raise SampleError("needs_info needs a name field")
        parent, _ = _parent(v2, state, rng, alloc)
        frame = {"recipe": "abort_needs_info", "noun": prof["noun"],
                 "parent_name": parent["name"], "parent_noun": v2["parent_noun"]}
        # spec §4 0.3.0: what is missing is the new record's name
        seg = f"ABORT NEEDS_INFO @{child}.{v2['name_field']}\n"
        return GenSample(frame, [seg], alloc.items, tags=["abort", "needs_info"])
    rec, display, ref = _named_record(prof, child, state, rng, alloc)
    template, verb = rng.choice(UNSUPPORTED)
    frame = {"recipe": "abort_unsupported", "noun": prof["noun"], "record": display,
             "template": template, "verb": verb}
    return GenSample(frame, ["ABORT UNSUPPORTED\n"], alloc.items, tags=["abort", "unsupported"])


# ------------------------------------------------------------ L17 search
def sample_search(world, profile, state, now, rng, alloc, holdout):
    v2 = _v2(profile)
    q = rng.choice(QUESTIONS)
    qref = alloc.get("STR", q, f"the question, verbatim: {q}")
    if rng.random() < 0.6:
        seg = f"CALL @{v2['search']} {qref} -> r0\nRETURN r0\n"
        frame = {"recipe": "search", "question": q, "send_to": None}
        return GenSample(frame, [seg], alloc.items, tags=["search", "report"])
    parent, pref = _parent(v2, state, rng, alloc)
    seg = (f"CALL @{v2['search']} {qref} -> r0\n"
           f"CALL @{v2['send']} {pref} r0\nSTOP\n")
    frame = {"recipe": "search", "question": q, "send_to": parent["name"]}
    return GenSample(frame, [seg], alloc.items, tags=["search"])


# ------------------------------------------------------------ L18 notes
def sample_note(world, profile, state, now, rng, alloc, holdout):
    v2 = _v2(profile)
    child, prof = v2["child"], profile["entities"][v2["child"]]
    note = v2["note_entity"]
    variant = rng.choice(["add", "edit", "delete"])
    if variant == "add":
        rec, display, ref = _named_record(prof, child, state, rng, alloc)
        title = rng.choice(NOTE_TITLES)
        text = rng.choice(NOTE_TEXTS)
        tref = alloc.get("STR", title, f'the text "{title}"')
        xref = alloc.get("STR", text, f'the text "{text}"')
        seg = f"CALL @{v2['add_note']} {ref} {xref} {tref} -> r0\nSTOP\n"
        frame = {"recipe": "note_add", "noun": prof["noun"], "record": display,
                 "title": title, "text": text}
        return GenSample(frame, [seg], alloc.items, tags=["note", "create"])
    notes = _records(state, note)
    if not notes:
        raise SampleError("no notes")
    n = rng.choice(notes)
    rec = next(r for r in _records(state, child) if r["id"] == n[child])
    nf = prof.get("name_field")
    display = rec[nf] if nf else prof["ref_word"].format(n=rec["id"].split("_")[-1])
    ref = alloc.get(f"ID:{child}", rec["id"], display)
    tref = alloc.get("STR", n["title"], f'the text "{n["title"]}"')
    head = (f"CALL @{v2['list_notes']} {ref} -> r0\n"
            f"FILTER r0 @{note}.title EQ {tref} -> r1\n"
            f"FOREACH r1 -> r2\n")
    if variant == "edit":
        text = rng.choice([t for t in NOTE_TEXTS if t != n["text"]])
        xref = alloc.get("STR", text, f'the text "{text}"')
        seg = head + f"  CALL @{v2['update_note']} {ref} r2.@{note}.id {xref} -> r3\nSTOP\n"
        frame = {"recipe": "note_edit", "noun": prof["noun"], "record": display,
                 "title": n["title"], "text": text}
        return GenSample(frame, [seg], alloc.items, tags=["note", "two_id"])
    seg = head + f"  CALL @{v2['delete_note']} {ref} r2.@{note}.id\nSTOP\n"
    frame = {"recipe": "note_delete", "noun": prof["noun"], "record": display,
             "title": n["title"]}
    return GenSample(frame, [seg], alloc.items, tags=["note", "two_id"])


RECIPES_V2 = {
    12: [("format_reminder", lambda *a: sample_format(*a))],
    13: [("create", lambda *a: sample_create(*a))],
    14: [("report", lambda *a: sample_report(*a))],
    15: [("content", lambda *a: sample_content(*a))],
    16: [("abort_v2", lambda *a: sample_abort_v2(*a))],
    17: [("search", lambda *a: sample_search(*a))],
    18: [("note", lambda *a: sample_note(*a))],
}


# ------------------------------------------------------------ English
def render_v2(frame: dict, rng: random.Random, np_fn, obj_fn, wrap, every) -> str:
    """Core sentence for a v2 frame; the caller wraps it in a style."""
    r = frame["recipe"]
    if r == "format_reminder":
        obj = obj_fn(frame, rng)
        who = f"the {frame['parent_noun'][0]} of {obj}"
        return rng.choice([
            f'send {who} a reminder that reads "Reminder: <{frame["slot0"]}> is due <{frame["slot1"]}>", filled in from the record',
            f'text {who}: "Reminder: … is due …" with {obj}\'s {frame["slot0"]} and {frame["slot1"]} filled in',
            f'remind {who} using the template "Reminder: {{0}} is due {{1}}." — {{0}} is its {frame["slot0"]}, {{1}} its {frame["slot1"]}',
        ])
    if r == "create":
        verb = rng.choice(CREATE_VERBS)
        noun = frame["noun"][0]
        parts = [f"{verb} a"]
        if frame["enum_phrase"] and not frame["title"]:
            parts.append(frame["enum_phrase"])
        parts.append(noun)
        if frame["title"]:
            parts.append(rng.choice(["called", "titled", "named"]) + f' "{frame["title"]}"')
        parts.append(f"for {frame['parent_name']}")
        core = " ".join(parts)
        if frame["enum_phrase"] and frame["title"]:
            core += rng.choice([f", marked {frame['enum_phrase']}", f" and set it to {frame['enum_phrase']}",
                                f" ({frame['enum_phrase']})"])
        if frame["follow"]:
            core += rng.choice([", then ", " and then ", "; after that "]) + frame["follow"]["verb"].format(obj="it")
        return core
    if r == "report":
        np = np_fn(frame["noun"], frame["clauses"], rng)
        if frame["count"]:
            return rng.choice(REPORT_COUNT).format(np=np)
        return rng.choice(REPORT_LIST).format(np=np, q=rng.choice(every))
    if r == "content_send":
        np = np_fn(frame["noun"], frame["clauses"], rng)
        return rng.choice([
            f"write {frame['parent_name']} a message about the {np}: {frame['brief']}",
            f"draft a note to {frame['parent_name']} covering the {np} — {frame['brief']}",
            f"put together a short message for {frame['parent_name']} on the {np} ({frame['brief']}) and send it",
        ])
    if r == "content_image":
        obj = obj_fn(frame, rng)
        style = f" in {frame['style']} style" if frame["style"] else ""
        return rng.choice([
            f"make an image of {frame['prompt']}{style} for {obj}",
            f"give {obj} a picture of {frame['prompt']}{style}",
            f"generate {obj} an illustration{style}: {frame['prompt']}",
        ])
    if r == "abort_ambiguous":
        return frame["action"]["verb"].format(obj=f"the {frame['noun'][0]}")
    if r == "abort_needs_info":
        return rng.choice(CREATE_VERBS) + f" a {frame['noun'][0]} for {frame['parent_name']}"
    if r == "abort_unsupported":
        obj = obj_fn(frame, rng)
        return frame["template"].format(obj=obj, np=f"all the {frame['noun'][1]}")
    if r == "search":
        q = frame["question"]
        if frame["send_to"]:
            return rng.choice([f"look up {q} and text {frame['send_to']} the answer",
                               f"find out {q}, then send it to {frame['send_to']}"])
        return rng.choice([f"{q}?", f"{q[0].upper() + q[1:]}?", f"can you tell me {q}?",
                           f"look up {q}"])
    if r == "note_add":
        obj = obj_fn(frame, rng)
        return rng.choice([
            f'add a note to {obj} titled "{frame["title"]}" saying "{frame["text"]}"',
            f'attach a "{frame["title"]}" note to {obj}: "{frame["text"]}"',
            f'put a note on {obj} — title "{frame["title"]}", text "{frame["text"]}"',
        ])
    if r == "note_edit":
        obj = obj_fn(frame, rng)
        return rng.choice([
            f'on {obj}, change the "{frame["title"]}" note to say "{frame["text"]}"',
            f'update the note titled "{frame["title"]}" on {obj} to read "{frame["text"]}"',
        ])
    if r == "note_delete":
        obj = obj_fn(frame, rng)
        return rng.choice([
            f'delete the "{frame["title"]}" note from {obj}',
            f'remove the note titled "{frame["title"]}" on {obj}',
        ])
    raise ValueError(f"unknown v2 recipe {r}")


from . import programs as _programs  # noqa: E402
_programs.RECIPES.update(RECIPES_V2)

