"""Family B: tools that can only be told apart by their descriptions.

Today a tool choice rides on verb habits the model has seen thousands of
times. R5 §2 caught it reaching for `setTeeTimeStatus` where the request said
cancel, and the dungeon is that same failure with five tools instead of two -
five tools whose names are all plausible and whose descriptions are the only
thing that separates them.

`decoy_world` gives every mutating tool in a world two to four siblings with
the *same signature* and a neighbouring description: `archive_card` beside
`snooze_card` and `escalate_card`, all `(ID:card) -> OBJ:card [WRITE]`. Only
the description says which one the request means. On a slice of them the
names are adversarial (`foo17`, `operation_93`, R5's own), so the name cannot
carry the choice at all.

The decoys are `noop` impls (runtime/sandbox.js): they accept their arguments
and change nothing, so a model that picks one is scored wrong on the task's
state rather than corrupting the world in some other way. They are never in a
reference program.

  crowding (harness/crowding.py) is the neighbouring pressure and does a
  different job: it buries the right tool in 30-80 foreign ones. A decoy sits
  next to the right tool with the same shape. Both can be on at once.
"""
from __future__ import annotations

import copy
import random
from typing import List, Tuple

MUTATING = {"WRITE", "DELETE", "SEND", "PAY"}

# verb -> what a tool by that name would actually do. Banked by effect: a
# decoy whose description is a copy operation while its effect line says SEND
# is a tell, and the point of the family is that the description is the only
# thing that separates the tools.
DECOY_OPS = {
    "WRITE": {
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
    "DELETE": {
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
    "SEND": {
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
    "PAY": {
        "authorize": "Put a hold on the funds for {a_noun} without taking "
                     "them yet.",
        "refund": "Return an earlier payment on {a_noun}.",
        "void": "Cancel an authorization on {a_noun} before it settles.",
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


def decoy_world(world: dict, rng: random.Random,
                per_tool: Tuple[int, int] = (2, 4),
                nonsense: float = 0.15) -> Tuple[dict, List[str]]:
    """A copy of `world` with description-only siblings beside every mutating
    tool. Returns (world, decoy names)."""
    merged = dict(world)
    merged["tools"] = [copy.deepcopy(t) for t in world["tools"]]
    taken = {t["name"] for t in merged["tools"]}
    added: List[str] = []

    targets = [t for t in world["tools"] if MUTATING & set(t["effects"])]
    for tool in targets:
        noun, entity = _noun_for(tool, world)
        # the strongest effect the tool declares picks the bank, so a decoy
        # never describes a copy while its effect line says SEND
        bank_key = next(e for e in ("PAY", "SEND", "DELETE", "WRITE")
                        if e in tool["effects"])
        bank = DECOY_OPS[bank_key]
        verbs = rng.sample(sorted(bank), min(len(bank),
                                             rng.randint(*per_tool)))
        for verb in verbs:
            name = (rng.choice(NONSENSE_NAMES) if rng.random() < nonsense
                    else f"{verb}_{entity or 'record'}")
            if name in taken:
                continue
            taken.add(name)
            decoy = {
                "name": name,
                "desc": bank[verb].format(a_noun=_article(noun)),
                "params": copy.deepcopy(tool["params"]),
                "returns": tool.get("returns"),
                "effects": list(tool["effects"]),
                "impl": {"op": "noop"},
            }
            # keep the return type honest: a tool that hands a record back
            # has to hand back the record it was given
            id_param = next((i for i, p in enumerate(decoy["params"])
                             if p.get("field") and p["field"][1] == "id"), None)
            if id_param is not None and decoy["returns"]:
                decoy["impl"] = {"op": "noop",
                                 "entity": decoy["params"][id_param]["field"][0],
                                 "id_param": id_param}
            merged["tools"].append(decoy)
            added.append(name)
    return merged, added
