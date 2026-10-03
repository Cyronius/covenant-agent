"""Family B: tools that can only be told apart by their descriptions.

Today a tool choice rides on verb habits the model has seen thousands of
times. R5 §2 caught it reaching for `setTeeTimeStatus` where the request said
cancel, and the dungeon is that same failure with five tools instead of two -
five tools whose names are all plausible and whose descriptions are the only
thing that separates them.

`decoy_world` gives every tool in a world one to four siblings with the *same
signature* and a neighbouring description: `archive_card` beside `snooze_card`
and `escalate_card`, all `(ID:card) -> OBJ:card [M]`. Only the description
says which one the request means. On a slice of them the names are adversarial
(`foo17`, `operation_93`, R5's own), so the name cannot carry the choice at
all.

The decoys are `noop` impls (runtime/sandbox.js): they accept their arguments
and produce nothing useful, so a model that picks one is scored wrong on the
task's state rather than corrupting the world in some other way. They are
never in a reference program.

READ and EXTERNAL were out of scope until 2026-09-20, and that left 51.8% of
reference calls in the 42-world holdout signature-unique on their own --
`results/GROUNDING.md` section 5 called READ "the wall". What held it up was a
scoring question, not a generator one. A decoy READ has to hand back a value
of the declared type while doing nothing, and for `LIST OBJ:x` the natural
nothing is an empty list -- which is also the legitimate trigger for the
check-then-decline recipe. A model that picks a decoy list tool, sees nothing
and emits `ABORT NOT_FOUND` therefore reaches the same observable outcome as
one that read the right tool's description, and outcome scoring cannot tell
them apart.

It is the call log that tells them apart, and it was simply never read.
`harness/metrics.py` now carries `decoy_called` -- recorded, not gated, so
every number already in `results/` keeps its meaning -- and
`models/tiny/diagnose.py` carries `chance_tool_sig`, the chance line for a
model that infers the type shape and picks uniformly inside the collision
group. With those two, a decoy READ is scorable and the wall comes down: the
holdout goes from 53.1% signature-unique to single digits.

What each noop hands back, by declared return type (`_return_shape`):

  LIST OBJ:x   `[]`, so a FOREACH over it does nothing
  STR          `""`, which fails the state check against real generated text
  OBJ:x, ID:x  a NOT_FOUND ToolError -- null is not of the declared type, and
               returning the real record would make the decoy *work*, which
               is the shortcut this family exists to remove

  crowding (harness/crowding.py) is the neighbouring pressure and does a
  different job: it buries the right tool in 30-80 foreign ones. A decoy sits
  next to the right tool with the same shape. Both can be on at once.
"""
from __future__ import annotations

import copy
import random
import warnings
from typing import Dict, List, Optional, Tuple

MUTATING = {"mutates", "irreversible"}

# verb -> what a tool by that name would actually do. Banked by property
# (spec 0.7.0 §7): a decoy whose description is a copy operation while its
# properties say external+irreversible is a tell, and the point of the
# family is that the description is the only thing that separates the tools.
DECOY_OPS = {
    "mutates": {
        "snooze": "Hide {a_noun} until a date you set, then bring it back to "
                  "the list unchanged.",
        "escalate": "Pass {a_noun} up to the next tier and mark it as needing "
                    "someone senior.",
        "flag": "Mark {a_noun} for review at the next audit. Changes nothing "
                "else about it.",
        "pin": "Keep {a_noun} at the top of the list for everyone who opens "
               "it.",
        "duplicate": "Make a copy of {a_noun}, with the same details and a "
                     "new id.",
        "lock": "Stop anyone but an owner editing {a_noun} until it is "
                "unlocked again.",
        "watch": "Follow {a_noun} so its changes turn up in your feed.",
        "silence": "Stop {a_noun} sending any more notifications, without "
                   "changing its state.",
        "defer": "Push {a_noun} to the back of the queue without touching "
                 "its status.",
        "retire": "Take {a_noun} out of use while keeping it readable for "
                  "reporting.",
        "promote": "Move {a_noun} into the featured set shown on the front "
                   "page.",
        "quarantine": "Hold {a_noun} aside pending a compliance check.",
        "reindex": "Rebuild the search index entry for {a_noun}.",
        "recheck": "Run the validation rules over {a_noun} again and record "
                   "the result.",
    },
    "irreversible+mutates": {
        "purge": "Strip the attachments and history off {a_noun}, leaving "
                 "the record itself in place.",
        "expire": "End {a_noun}'s current term now; it stays on file as "
                  "expired.",
        "revoke": "Withdraw the approvals on {a_noun} so it has to go round "
                  "again.",
        "shred": "Erase the personal details on {a_noun} for a data request, "
                 "keeping the shell.",
        "clear_notes": "Remove every note left on {a_noun}, and nothing "
                       "else.",
        "discard_draft": "Throw away the unsaved draft of {a_noun}; the "
                         "saved one is untouched.",
    },
    "external+irreversible": {
        "remind": "Send a reminder about {a_noun} to whoever it is assigned "
                  "to.",
        "nudge": "Send a short chase-up about {a_noun} on the quiet channel.",
        "broadcast": "Announce {a_noun} to everyone on the team channel.",
        "digest": "Add {a_noun} to tonight's summary mail instead of sending "
                  "now.",
        "acknowledge": "Send the standard we-have-received-it reply about "
                       "{a_noun}.",
        "escalate_to": "Page the on-call about {a_noun}.",
    },
    "external+irreversible+mutates": {
        "authorize": "Put a hold on the funds for {a_noun} without taking "
                     "them yet.",
        "refund": "Return an earlier payment on {a_noun}.",
        "void": "Cancel an authorization on {a_noun} before it settles.",
    },
    # A tool with neither `mutates` nor `irreversible` (a pure read, or a
    # harmless external call) is banked by return shape as well, because
    # "list every" and "fetch the one" are not interchangeable descriptions
    # and a decoy whose description does not fit its own return type is a
    # tell. The key is `none:LIST` / `none:OBJ` / `none:STR` / `external:STR`
    # -- see `_bank_key`.
    "none:LIST": {
        "archived": "List the archived {noun} records -- the ones taken out "
                    "of use, not the live set.",
        "stale": "List the {noun} records nobody has touched since the last "
                 "review.",
        "flagged": "List the {noun} records marked for review at the next "
                   "audit.",
        "drafts": "List the unsaved {noun} drafts; the saved ones are not "
                  "included.",
        "deleted": "List the {noun} records deleted in the last 30 days, for "
                   "recovery.",
        "imported": "List the {noun} records that came in from a partner "
                    "feed rather than from here.",
        "queued": "List the {noun} records still waiting to be imported, "
                  "before they go live.",
        "sampled": "List a random sample of {noun} records for spot-checking.",
    },
    "none:OBJ": {
        "archived_copy": "Fetch the archived copy of {a_noun}, as it stood "
                         "when it was retired.",
        "snapshot": "Fetch {a_noun} as it stood at the last audit, not as it "
                    "is now.",
        "draft_of": "Fetch the unsaved draft of {a_noun}; the saved one is "
                    "untouched.",
        "redacted": "Fetch {a_noun} with the personal details stripped, for "
                    "sharing outside the team.",
        "upstream": "Fetch the partner feed's copy of {a_noun} rather than "
                    "ours.",
    },
    # no entity behind these, so they are named by verb alone
    "none:STR": {
        "glossary": "Look a term up in the internal glossary rather than in "
                    "the documentation.",
        "changelog": "Look a term up in the change history rather than in "
                     "the documentation.",
        "faq": "Look a term up in the customer-facing FAQ.",
        "archived_docs": "Look a term up in last year's documentation set.",
    },
    "external:STR": {
        "outline_only": "Produce a heading outline of the text rather than "
                        "the text itself.",
        "summarize": "Produce a one-line summary of the material rather than "
                     "the full text.",
        "translate": "Produce the text in the reader's language rather than "
                     "the house one.",
        "held_draft": "Draft the text and hold it for review instead of "
                      "returning it ready to use.",
    },
}

# R5's adversarial names: nothing in them points at what the tool does
NONSENSE_NAMES = ["foo17", "operation_93", "do_thing_4", "x_op_12", "fn_204",
                  "task_77", "handler_31", "proc_58", "bar_9", "op_146"]


def _noun_for(tool: dict, world: dict) -> Tuple[str, str]:
    """(noun, entity) the tool's first id-ish param points at."""
    for p in tool["params"]:
        if p.get("field"):
            entity = p["field"][0]
            return entity.replace("_", " "), entity
        if p["type"].startswith("ID:"):
            entity = p["type"][3:]
            return entity.replace("_", " "), entity
    ret = tool.get("returns") or ""
    for token in ret.replace("LIST ", "").split():
        if token.startswith(("OBJ:", "ID:")):
            entity = token.split(":", 1)[1]
            return entity.replace("_", " "), entity
    return "record", ""


def _article(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


def _return_shape(tool: dict) -> Optional[str]:
    """LIST / OBJ / ID / STR, or None for a return a noop cannot fake."""
    ret = (tool.get("returns") or "").strip()
    if ret.startswith("LIST"):
        return "LIST"
    for prefix in ("OBJ:", "ID:"):
        if ret.startswith(prefix):
            return prefix[:-1]
    return "STR" if ret == "STR" else None


def _bank_key(tool: dict) -> Optional[str]:
    """Which description bank a sibling of this tool draws from.

    The tool's full property set picks it (spec 0.7.0 §7's `mutates`/
    `irreversible`/`external`), so a decoy never describes a copy while its
    properties say external+irreversible. A tool with neither `mutates` nor
    `irreversible` (a pure read, or a harmless external call) is keyed by
    return shape too: "list every" and "fetch the one" are not
    interchangeable, and a description that does not fit its own return type
    is a tell.
    """
    props = set(tool["effects"])
    if props & {"mutates", "irreversible"}:
        key = "+".join(sorted(props))
        return key if key in DECOY_OPS else None
    shape = _return_shape(tool)
    prefix = "external" if "external" in props else "none"
    key = f"{prefix}:{shape}" if shape else None
    return key if key in DECOY_OPS else None


_BANK_WARNED: set = set()


def _warn_bank(world: dict, tool: dict) -> None:
    """Once per tool: this sibling set comes from the fixed bank, whose
    style a description-only classifier separates from real descriptions at
    AUC 1.000 (results/R10.md section 8). Loud, because an exam built on it
    cannot carry a reading claim."""
    key = (world.get("name"), tool["name"])
    if key in _BANK_WARNED:
        return
    _BANK_WARNED.add(key)
    warnings.warn(f"decoys for {key[0]}.{key[1]} drawn from the template "
                  "bank: the theme authors none (harness/decoy_audit.py "
                  "will flag the style)", stacklevel=3)


def decoy_world(world: dict, rng: random.Random,
                per_tool: Tuple[int, int] = (2, 4),
                nonsense: float = 0.15,
                only: Optional[set] = None) -> Tuple[dict, List[str]]:
    """A copy of `world` with description-only siblings beside every tool
    `_bank_key` can bank, or only beside the tools named in `only`
    (data.gen --decoy-slots). Returns (world, decoy names)."""
    merged = dict(world)
    merged["tools"] = [copy.deepcopy(t) for t in world["tools"]]
    taken = {t["name"] for t in merged["tools"]}
    added: List[str] = []

    targets = [(t, k) for t in world["tools"]
               if ((k := _bank_key(t)) is not None or t.get("authored_decoys"))
               and (only is None or t["name"] in only)]
    for tool, bank_key in targets:
        noun, entity = _noun_for(tool, world)
        mutating = bool(MUTATING & set(tool["effects"]))
        authored = tool.get("authored_decoys")
        if authored:
            # the theme's own siblings, written in its desc_style and naming
            # style (THEME_SCHEMA.md, "Decoys"), so neither the description
            # nor the name says which side of the twin decision is real.
            # Opaque names are a row-level draw (`opaque_names`), never a
            # per-decoy one: a nonsense name only decoys carry is a tell.
            picks = rng.sample(range(len(authored)),
                               min(len(authored), rng.randint(*per_tool)))
            drawn = [(authored[i]["name"], authored[i]["desc"])
                     for i in sorted(picks)]
        else:
            _warn_bank(world, tool)
            bank = DECOY_OPS[bank_key]
            verbs = rng.sample(sorted(bank), min(len(bank),
                                                 rng.randint(*per_tool)))
            # a none:STR or external:STR tool has no entity behind it, so
            # the verb stands alone rather than dragging `_record` along
            drawn = [((rng.choice(NONSENSE_NAMES) if rng.random() < nonsense
                       else (f"{verb}_{entity}" if entity else verb)),
                      bank[verb].format(a_noun=_article(noun), noun=noun))
                     for verb in verbs]
        for name, desc in drawn:
            if name in taken:
                continue
            taken.add(name)
            decoy = {
                "name": name,
                "desc": desc,
                "params": copy.deepcopy(tool["params"]),
                "returns": tool.get("returns"),
                "effects": list(tool["effects"]),
                "impl": {"op": "noop"},
                # a param with no entity field behind it (a message body, a
                # note) gets a fresh F symbol per tool name, so without this
                # the decoy's free-text slot reads `S=F6` against the
                # original's `S=F5` and the signature stops colliding --
                # 97.9% of SEND calls stayed unique that way. build_context
                # gives a decoy's unlinked params its original's symbols.
                "decoy_of": tool["name"],
            }
            id_param = next((i for i, p in enumerate(decoy["params"])
                             if p.get("field") and p["field"][1] == "id"), None)
            if mutating and id_param is not None and decoy["returns"]:
                # a mutating tool that hands a record back has to hand back
                # the record it was given: "I changed nothing, here it is"
                decoy["impl"] = {"op": "noop",
                                 "entity": decoy["params"][id_param]["field"][0],
                                 "id_param": id_param}
            elif not mutating:
                # a non-mutating decoy must not return the real record --
                # that would make it *work*, and the collision would teach
                # nothing. It
                # hands back the empty value of its declared type instead,
                # or NOT_FOUND where there is no empty value (module docstring)
                decoy["impl"] = {"op": "noop", "empty": _return_shape(tool)}
            merged["tools"].append(decoy)
            added.append(name)
    return merged, added


# Row-level opaque names (.claude/plans/description-reading.md step 1): on a
# drawn share of rows *every* tool -- real, decoy and compute block alike --
# is renamed to something that says nothing, so description reading stays
# measurable on its own once names are model input (spec 0.8.0). Per-decoy
# nonsense names (`NONSENSE_NAMES` above) only ever landed on decoys, which
# made "has a nonsense name" a decoy tell.
OPAQUE_STEMS = ("foo", "op_", "fn_", "x_op_", "handler_", "proc_", "task_",
                "bar_", "do_thing_", "act_", "cmd_", "step_", "routine_",
                "job_", "call_")


def opaque_names(task: dict, rng: random.Random) -> Dict[str, str]:
    """Rename every tool in a built task, in place, everywhere a tool *name*
    is stored (context, sandbox, call log, error injection, provenance).
    Sandbox `impl` fields are left alone: `kind`/`fn` there name an engine
    function, not a tool. Returns the old -> new map."""
    names = [t["name"] for t in task["context"]["tools"]]
    new, taken = {}, set()
    for n in names:
        while True:
            cand = f"{rng.choice(OPAQUE_STEMS)}{rng.randint(1, 999)}"
            if cand not in taken:
                break
        taken.add(cand)
        new[n] = cand
    for t in task["context"]["tools"]:
        t["name"] = new[t["name"]]
    for t in (task.get("sandbox") or {}).get("tools", []):
        t["name"] = new.get(t["name"], t["name"])
    for c in (task.get("reference") or {}).get("call_log", []):
        if c.get("name") in new:
            c["name"] = new[c["name"]]
    for e in task.get("error_injection") or []:
        if isinstance(e, dict) and e.get("name") in new:
            e["name"] = new[e["name"]]
    prov = task.get("provenance") or {}
    for key in ("decoys", "flip_slot_tools"):
        if prov.get(key):
            prov[key] = [new.get(n, n) for n in prov[key]]

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "tool" and isinstance(v, str) and v in new:
                    o[k] = new[v]
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(prov.get("frame"))
    return new
