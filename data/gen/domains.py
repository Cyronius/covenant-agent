"""S0: compile domain theme packs into WORLD + PROFILE + state generator.

A theme pack (data/gen/THEME_SCHEMA.md) supplies vocabulary; this module
supplies structure. `register_domains(dir)` compiles every theme JSON and
mutates the live registries (runtime.worlds.WORLDS, worldgen.GENERATORS,
profiles.PROFILES/TEXT_BANKS) so the existing recipes, harness, and CLI
work on generated domains unchanged.

Bank keys are namespaced `{domain}:{bank}` to avoid collisions.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from . import worldgen
from .profiles import PROFILES, TEXT_BANKS
from runtime import worlds as worlds_registry

DAY = 86400
NOW = 1_760_000_000


# --------------------------------------------------------------- WORLD dict
def _child_fields(theme: dict) -> dict:
    c = theme["child"]
    fields = {"id": f"ID:{c['entity']}"}
    if c.get("name_field"):
        fields[c["name_field"]] = "STR"
    fields[c["enum"]["field"]] = "STR"
    for b in c["bools"]:
        fields[b["field"]] = "BOOL"
    for t in c["times"]:
        fields[t["field"]] = "TIME"
    if c.get("number"):
        # The one INT field a theme may declare, so a themed world has
        # something the compute tools (`sum`/`avg`/`max`/`min`, spec 0.7.0
        # §6) can be called on. Before this, every themed field was
        # STR/BOOL/TIME/ID and the aggregate arm of L21 fired on 1.7% of
        # themed rows (`results/REFLEX.md` §6).
        fields[c["number"]["field"]] = "INT"
    fields[c["ref_field"]] = f"ID:{theme['parent']['entity']}"
    fields["image"] = "STR"
    return fields


def _v2_names(theme: dict) -> dict:
    """Surface v2 tool names, rule-derived from the theme (plan
    lane-c-retrain §C1). The three generic tools draw from small pools so
    the corpus sees a few spellings of the same role."""
    child = theme["child"]["entity"]
    h = sum(ord(ch) for ch in theme["domain"])
    names = {
        "create": f"create_{child}",
        "list_notes": f"list_{child}_notes", "add_note": f"add_{child}_note",
        "update_note": f"update_{child}_note", "delete_note": f"delete_{child}_note",
        "set_image": f"set_{child}_image",
        "writer": ["write_text", "draft_text", "compose_text"][h % 3],
        "image": ["generate_image", "make_image", "render_image"][(h // 3) % 3],
        "search": ["search_docs", "search_help", "lookup_docs"][(h // 9) % 3],
    }
    # a theme's own `v2` block renames them in its naming style (see
    # `_v2_text`)
    for slot, over in (theme.get("v2") or {}).items():
        names[slot] = over["name"]
    return names


# The compiler's nine generic tools, in `_v2_tools` order.
V2_SLOTS = ("create", "list_notes", "add_note", "update_note", "delete_note",
            "set_image", "writer", "image", "search")


def _v2_text(theme: dict, slot: str, default: str) -> str:
    """A generic tool's description: the theme's own, when it authors one.

    The defaults are one template per slot across every world, so a
    description-only classifier learns `Attach a note to a` as "real" from
    the training worlds and carries it to any unseen one -- the same
    familiarity shortcut the bank decoys handed it the other way
    (results/R10.md section 8). A theme that authors its decoys authors
    these too, in its own desc_style, so neither side of a twin decision is
    recognisable as a template.
    """
    over = (theme.get("v2") or {}).get(slot)
    return over["desc"] if over else default


def _authored(slot_spec: dict | None) -> dict:
    """The authored decoys of one tool slot, carried on the world tool so
    `harness.decoys.decoy_world` can draw them (empty when there are none)."""
    ds = (slot_spec or {}).get("decoys")
    return {"authored_decoys": [dict(d) for d in ds]} if ds else {}


def _v2_tools(theme: dict) -> list:
    c, p = theme["child"], theme["parent"]
    child, parent = c["entity"], p["entity"]
    note = f"{child}_note"
    n = _v2_names(theme)
    noun = c["noun"][0]
    enum_f = c["enum"]["field"]
    nf = c.get("name_field")

    def idp(entity, desc, pname=None):
        return {"name": pname or entity, "type": f"ID:{entity}", "desc": desc,
                "field": [entity, "id"]}

    create_params = [idp(parent, f"the {p['noun'][0]} it belongs to")]
    create_fields = [c["ref_field"]]
    if nf:
        create_params.append({"name": nf, "type": "STR", "desc": f"the {noun}'s {nf}",
                              "field": [child, nf]})
        create_fields.append(nf)
    create_params.append({"name": enum_f, "type": "STR", "desc": f"initial {enum_f}",
                          "required": False, "field": [child, enum_f]})
    create_fields.append(enum_f)
    defaults = {enum_f: c["enum"]["values"][0], "image": ""}
    if nf:
        defaults[nf] = f"New {noun}"
    for b in c["bools"]:
        defaults[b["field"]] = False
    for t in c["times"]:
        defaults[t["field"]] = "$now"
    if c.get("number"):
        # created records carry the midpoint, so a create inside a task does
        # not move an average much; no tool sets this field (see
        # THEME_SCHEMA.md's `number`)
        num = c["number"]
        defaults[num["field"]] = (num["min"] + num["max"]) // 2
    tools = [
        {"name": n["create"], "desc": _v2_text(theme, "create", f"Create a new {noun} for a {p['noun'][0]}."),
         "params": create_params, "returns": f"OBJ:{child}", "effects": ["mutates"],
         "impl": {"op": "create", "entity": child, "param_fields": create_fields,
                  "defaults": defaults}},
        {"name": n["list_notes"], "desc": _v2_text(theme, "list_notes", f"List the notes attached to one {noun}."),
         "params": [idp(child, f"the {noun}")],
         "returns": f"LIST OBJ:{note}", "effects": [],
         "impl": {"op": "list_by", "entity": note, "field": child, "id_param": 0}},
        {"name": n["add_note"], "desc": _v2_text(theme, "add_note", f"Attach a note to a {noun}."),
         "params": [idp(child, f"the {noun}"),
                    {"name": "text", "type": "STR", "desc": "note text", "field": [note, "text"]},
                    {"name": "title", "type": "STR", "desc": "note title", "required": False,
                     "field": [note, "title"]}],
         "returns": f"OBJ:{note}", "effects": ["mutates"],
         "impl": {"op": "create", "entity": note, "param_fields": [child, "text", "title"],
                  "defaults": {"title": "Note", "text": ""}}},
        {"name": n["update_note"], "desc": _v2_text(theme, "update_note", f"Change the text (and optionally the title) of one of a {noun}'s notes."),
         "params": [idp(child, f"the {noun}"), idp(note, "the note", "note"),
                    {"name": "text", "type": "STR", "desc": "new text", "field": [note, "text"]},
                    {"name": "title", "type": "STR", "desc": "new title", "required": False,
                     "field": [note, "title"]}],
         "returns": f"OBJ:{note}", "effects": ["mutates"],
         "impl": {"op": "update", "entity": note, "id_param": 1,
                  "set_from_params": {"text": 2, "title": 3}}},
        {"name": n["delete_note"], "desc": _v2_text(theme, "delete_note", f"Remove a note from a {noun}."),
         "params": [idp(child, f"the {noun}"), idp(note, "the note to remove", "note")],
         "returns": None, "effects": ["mutates", "irreversible"],
         "impl": {"op": "delete", "entity": note, "id_param": 1}},
        {"name": n["set_image"], "desc": _v2_text(theme, "set_image", f"Set the image shown on a {noun}."),
         "params": [idp(child, f"the {noun}"),
                    {"name": "image", "type": "STR", "desc": "image URL", "field": [child, "image"]}],
         "returns": f"OBJ:{child}", "effects": ["mutates"],
         "impl": {"op": "update", "entity": child, "id_param": 0, "set_from_params": {"image": 1}}},
        {"name": n["writer"], "desc": _v2_text(theme, "writer", "Write a short message from a brief (what to say, in the requester's words) and the records it should mention. Returns the text."),
         "params": [{"name": "brief", "type": "STR", "desc": "what to write"},
                    {"name": "data", "type": f"LIST OBJ:{child}", "desc": f"{c['noun'][1]} the message is about"}],
         "returns": "STR", "effects": ["external"],
         "impl": {"op": "external", "kind": "write_text"}},
        {"name": n["image"], "desc": _v2_text(theme, "image", "Generate an image from a prompt and return its URL."),
         "params": [{"name": "prompt", "type": "STR", "desc": "what the image shows"},
                    {"name": "style", "type": "STR", "desc": "visual style", "required": False}],
         "returns": "STR", "effects": ["external"],
         "impl": {"op": "external", "kind": "generate_image"}},
        {"name": n["search"], "desc": _v2_text(theme, "search", "Search the help docs and knowledge base with a question; returns the best passage."),
         "params": [{"name": "query", "type": "STR", "desc": "the question"}],
         "returns": "STR", "effects": [],
         "impl": {"op": "external", "kind": "search_docs"}},
    ]
    v2 = theme.get("v2") or {}
    return [dict(t, **_authored(v2.get(slot)))
            for slot, t in zip(V2_SLOTS, tools)]


def _parent_fields(theme: dict) -> dict:
    p = theme["parent"]
    fields = {"id": f"ID:{p['entity']}", "name": "STR"}
    if p.get("contact_field"):
        fields[p["contact_field"]] = "STR"
    return fields


# The theme's nine authored tool slots, in `build_world` order.
THEME_SLOTS = ("list_child", "get_child", "list_parent", "get_parent",
               "delete_child", "set_enum", "set_bool", "set_ref", "send")


def build_world(theme: dict) -> dict:
    c, p, tools = theme["child"], theme["parent"], theme["tools"]
    child, parent = c["entity"], p["entity"]
    enum_f = c["enum"]["field"]
    sb = theme["tools"]["set_bool"]
    bool_f = c["bools"][sb["bool_index"]]["field"]

    def idp(entity, desc):
        return {"name": entity, "type": f"ID:{entity}", "desc": desc,
                "field": [entity, "id"]}

    world_tools = [
        {"name": tools["list_child"]["name"], "desc": tools["list_child"]["desc"],
         "params": [], "returns": f"LIST OBJ:{child}", "effects": [],
         "impl": {"op": "list", "entity": child}},
        {"name": tools["get_child"]["name"], "desc": tools["get_child"]["desc"],
         "params": [idp(child, f"the {c['noun'][0]} to fetch")],
         "returns": f"OBJ:{child}", "effects": [],
         "impl": {"op": "get", "entity": child, "id_param": 0}},
        {"name": tools["list_parent"]["name"], "desc": tools["list_parent"]["desc"],
         "params": [], "returns": f"LIST OBJ:{parent}", "effects": [],
         "impl": {"op": "list", "entity": parent}},
        {"name": tools["get_parent"]["name"], "desc": tools["get_parent"]["desc"],
         "params": [idp(parent, f"the {p['noun'][0]} to fetch")],
         "returns": f"OBJ:{parent}", "effects": [],
         "impl": {"op": "get", "entity": parent, "id_param": 0}},
        {"name": tools["delete_child"]["name"], "desc": tools["delete_child"]["desc"],
         "params": [idp(child, f"{c['noun'][0]} to delete")],
         "returns": None, "effects": ["mutates", "irreversible"],
         "impl": {"op": "delete", "entity": child, "id_param": 0}},
        {"name": tools["set_enum"]["name"], "desc": tools["set_enum"]["desc"],
         "params": [idp(child, f"the {c['noun'][0]}"),
                    {"name": enum_f, "type": "STR", "desc": f"new {enum_f}",
                     "field": [child, enum_f]}],
         "returns": f"OBJ:{child}", "effects": ["mutates"],
         "impl": {"op": "update", "entity": child, "id_param": 0,
                  "set_from_params": {enum_f: 1}}},
        {"name": sb["name"], "desc": sb["desc"],
         "params": [idp(child, f"the {c['noun'][0]}")],
         "returns": f"OBJ:{child}", "effects": ["mutates"],
         "impl": {"op": "update", "entity": child, "id_param": 0,
                  "set_const": {bool_f: sb["value"]}}},
        {"name": tools["set_ref"]["name"], "desc": tools["set_ref"]["desc"],
         "params": [idp(child, f"the {c['noun'][0]}"),
                    {"name": parent, "type": f"ID:{parent}",
                     "desc": f"new {p['noun'][0]}",
                     "field": [child, c["ref_field"]]}],
         "returns": f"OBJ:{child}", "effects": ["mutates"],
         "impl": {"op": "update", "entity": child, "id_param": 0,
                  "set_from_params": {c["ref_field"]: 1}}},
        {"name": tools["send"]["name"], "desc": tools["send"]["desc"],
         "params": [idp(parent, "recipient"),
                    {"name": "text", "type": "STR", "desc": "message text"}],
         "returns": None, "effects": ["irreversible", "external"],
         "impl": {"op": "send", "channel": "message",
                  "param_map": ["to", "text"]}},
    ]
    world_tools = [dict(t, **_authored(tools[slot]))
                   for slot, t in zip(THEME_SLOTS, world_tools)]
    note = f"{child}_note"
    return {
        "name": theme["domain"], "now": NOW,
        "entities": {child: _child_fields(theme),
                     parent: _parent_fields(theme),
                     note: {"id": f"ID:{note}", child: f"ID:{child}",
                            "title": "STR", "text": "STR"}},
        "enums": {(child, enum_f): list(c["enum"]["values"])},
        "tools": world_tools + _v2_tools(theme),
        "default_state": {"entities": {child: [], parent: [], note: []},
                          "outbox": [], "payments": []},
    }


# ------------------------------------------------------------- PROFILE dict
def _bank(theme: dict, name: str) -> str:
    return f"{theme['domain']}:{name}"


def _compile_clause(cl: dict, theme: dict) -> dict:
    c = theme["child"]
    if "enum_value" in cl:
        return {"kind": "enum", "field": c["enum"]["field"],
                "value": cl["enum_value"]}
    if "bool_index" in cl:
        b = c["bools"][cl["bool_index"]]
        return {"kind": "bool", "field": b["field"], "value": cl["value"]}
    t = c["times"][cl["time_index"]]
    if "days" in cl:
        return {"kind": "time_cutoff", "field": t["field"],
                "days": cl["days"]}
    return {"kind": "time_now", "field": t["field"], "op": cl.get("op", "LT")}


def build_profile(theme: dict) -> dict:
    c, p, tools = theme["child"], theme["parent"], theme["tools"]
    child, parent = c["entity"], p["entity"]
    enum = c["enum"]
    parent_id_desc = f"{{name}}'s {p['noun'][0]} id"

    filters = [{"kind": "enum", "field": enum["field"],
                "values": list(enum["values"]),
                "phrases": {v: tuple(ph) for v, ph in enum["phrases"].items()},
                "desc": f"the {{v}} {enum['field']}"}]
    for b in c["bools"]:
        filters.append({"kind": "bool", "field": b["field"],
                        "true_phrase": b["true_phrase"],
                        "false_phrase": b["false_phrase"],
                        "placement": b["placement"]})
    for t in c["times"]:
        if t["kind"] == "time_now":
            filters.append({"kind": "time_now", "field": t["field"],
                            "lt_phrase": tuple(t["lt_phrase"]),
                            "gt_phrase": tuple(t["gt_phrase"])})
        else:
            filters.append({"kind": "time_cutoff", "field": t["field"],
                            "days": tuple(t.get("days", (20, 240))),
                            "phrase": t["phrase"], "desc": "{d} days ago"})
    filters.append({"kind": "ref", "field": c["ref_field"],
                    "ref_entity": parent, "name_field": "name",
                    "phrase": c["ref_phrase"], "placement": "post",
                    "desc": parent_id_desc})

    sb = tools["set_bool"]
    actions = [
        {"tool": tools["delete_child"]["name"], "dest": False,
         "args": ["<v>"], "verbs": list(tools["delete_child"]["verbs"])},
        {"tool": tools["set_enum"]["name"], "dest": True, "kind": "set_enum",
         "args": ["<v>", {"const_enum": {"field": enum["field"],
                                         "values": list(enum["values"]),
                                         "desc": f"the {{v}} {enum['field']}"}}],
         "verbs": list(tools["set_enum"]["verbs"]),
         "value_phrases": dict(tools["set_enum"]["value_phrases"])},
        {"tool": sb["name"], "dest": True,
         "args": ["<v>"], "verbs": list(sb["verbs"])},
        {"tool": tools["set_ref"]["name"], "dest": True, "kind": "set_ref",
         "args": ["<v>", {"const_ref": {"entity": parent,
                                        "name_field": "name",
                                        "desc": parent_id_desc}}],
         "verbs": list(tools["set_ref"]["verbs"]),
         "to_template": tools["set_ref"]["to_template"]},
        {"tool": tools["send"]["name"], "dest": False, "kind": "send_field",
         "args": [{"field": c["ref_field"]},
                  {"const_text": {"bank": _bank(theme, "child_msgs")}}],
         "verbs": list(tools["send"]["child_verbs"])},
    ]

    sort = theme["sort"]
    pf = sort["pre_filter"]
    if "enum_value" in pf:
        pre_filter = {"kind": "enum", "field": enum["field"],
                      "value": pf["enum_value"], "phrase": pf["phrase"]}
    else:
        b = c["bools"][pf["bool_index"]]
        pre_filter = {"kind": "bool", "field": b["field"],
                      "value": pf["value"], "phrase": pf["phrase"]}

    slot_to_tool = {"delete_child": tools["delete_child"]["name"],
                    "set_bool": sb["name"]}
    ambiguous = [{"request": list(a["request"]), "hint": a["hint"],
                  "entity": child,
                  "clauses": [_compile_clause(cl, theme)
                              for cl in a["clauses"]],
                  "action_tool": slot_to_tool[a["action"]]}
                 for a in theme["ambiguous"]]

    return {
        "entities": {child: {
            "noun": tuple(c["noun"]),
            "list_tool": tools["list_child"]["name"],
            "get_tool": tools["get_child"]["name"],
            "name_field": c.get("name_field"),
            "ref_word": f"{c['noun'][0]} {{n}}",
            "filters": filters,
            "actions": actions,
            # {INT field -> how a request says it}. The aggregate arm of
            # L21 asks "what is the total {phrase} across the {plural}",
            # and a field name read literally ("page count minutes") is not
            # English. Absent for a theme with no `number`.
            "measures": ({c["number"]["field"]: c["number"]["noun"]}
                         if c.get("number") else {}),
        }},
        "direct_send": {"tool": tools["send"]["name"],
                        "target": {"const_ref": {"entity": parent,
                                                 "name_field": "name",
                                                 "desc": parent_id_desc}},
                        "text_bank": _bank(theme, "child_msgs"),
                        "verbs": list(tools["send"]["direct_verbs"])},
        "pairs": [{"outer": parent,
                   "outer_list": tools["list_parent"]["name"],
                   "outer_get": tools["get_parent"]["name"],
                   "outer_noun": tuple(p["noun"]), "outer_name": "name",
                   "inner": child, "link_field": c["ref_field"],
                   "notify": {"tool": tools["send"]["name"],
                              "args": ["<outer>",
                                       {"const_text": {
                                           "bank": _bank(theme, "agg_msgs")}}],
                              "verbs": list(tools["send"]["pair_verbs"])}}],
        "sorts": [{"entity": child,
                   "field": c["times"][sort["time_index"]]["field"],
                   "dir": sort["dir"], "sup_phrase": sort["sup_phrase"],
                   "ord_phrase": sort["ord_phrase"],
                   "pre_filter": pre_filter}],
        "v2": {
            **_v2_names(theme),
            "child": child, "parent": parent, "note_entity": f"{child}_note",
            "parent_noun": tuple(p["noun"]), "ref_field": c["ref_field"],
            "name_field": c.get("name_field"), "titles": list(c.get("titles") or []),
            "enum_field": enum["field"], "enum_values": list(enum["values"]),
            "enum_phrases": {v: tuple(ph) for v, ph in enum["phrases"].items()},
            "time_field": c["times"][theme["sort"]["time_index"]]["field"],
            "send": tools["send"]["name"],
        },
        "fallback_notify": {"tool": tools["send"]["name"],
                            "target": {"const_ref": {"entity": parent,
                                                     "name_field": "name",
                                                     "desc": parent_id_desc}},
                            "text_bank": _bank(theme, "fail_msgs"),
                            "verbs": ["let {name} know", "tell {name}"]},
        "ambiguous": ambiguous,
    }


# --------------------------------------------------------- state generation
def make_state_gen(theme: dict):
    c, p = theme["child"], theme["parent"]
    child, parent = c["entity"], p["entity"]

    def gen(rng: random.Random, now: int) -> dict:
        parents = []
        used = set()
        n_parents = rng.randint(2, 4)
        while len(parents) < n_parents:
            if p["name_style"] == "company":
                name = rng.choice(worldgen.COMPANY_NAMES)
            else:
                name = (f"{rng.choice(worldgen.FIRST_NAMES)} "
                        f"{rng.choice(worldgen.LAST_NAMES)}")
            if name in used:
                continue
            used.add(name)
            rec = {"id": f"{parent}_{len(parents) + 1}", "name": name}
            cf = p.get("contact_field")
            if cf == "email":
                rec[cf] = name.split()[0].lower() + f"@{theme['domain']}.test"
            elif cf:
                rec[cf] = f"+1-555-{rng.randint(100, 999)}-{rng.randint(1000, 9999)}"
            parents.append(rec)

        children = []
        n_children = rng.randint(4, 9)
        titles = (rng.sample(c["titles"], n_children)
                  if c.get("name_field") else None)
        for i in range(n_children):
            rec = {"id": f"{child}_{i + 1}"}
            if titles:
                rec[c["name_field"]] = titles[i]
            rec[c["enum"]["field"]] = rng.choice(c["enum"]["values"])
            for b in c["bools"]:
                rec[b["field"]] = rng.random() < 0.3
            for t in c["times"]:
                if t["kind"] == "time_now":
                    rec[t["field"]] = now + rng.randint(-20, 20) * DAY
                else:
                    rec[t["field"]] = now - rng.randint(5, 300) * DAY
            if c.get("number"):
                num = c["number"]
                rec[num["field"]] = rng.randint(num["min"], num["max"])
            rec[c["ref_field"]] = rng.choice(parents)["id"]
            rec["image"] = "" if rng.random() < 0.7 else \
                f"https://cdn.{theme['domain']}.test/img/{i + 1}.jpg"
            children.append(rec)
        from .v2 import NOTE_TEXTS, NOTE_TITLES
        notes = []
        for ch in children:
            for _ in range(rng.choice([0, 0, 1, 1, 2, 3])):
                notes.append({"id": f"{child}_note_{len(notes) + 1}", child: ch["id"],
                              "title": rng.choice(NOTE_TITLES),
                              "text": rng.choice(NOTE_TEXTS)})
        return {"entities": {parent: parents, child: children,
                             f"{child}_note": notes},
                "outbox": [], "payments": []}

    return gen


# ------------------------------------------------------------- registration
def load_theme(path: Path) -> dict:
    theme = json.loads(path.read_text(encoding="utf-8"))
    validate_theme(theme)
    return theme


def validate_theme(theme: dict) -> None:
    c = theme["child"]
    assert theme["domain"].replace("_", "").isalnum(), "bad domain id"
    assert 1 <= len(c["bools"]) <= 3, "need 1-3 bools"
    assert 2 <= len(c["enum"]["values"]) <= 4, "enum needs 2-4 values"
    assert set(c["enum"]["phrases"]) == set(c["enum"]["values"]), \
        "enum phrases must cover values"
    assert 1 <= len(c["times"]) <= 2, "need 1-2 time fields"
    kinds = [t["kind"] for t in c["times"]]
    assert kinds.count("time_now") <= 1 and kinds.count("time_cutoff") <= 1
    if c.get("name_field"):
        assert len(c["titles"]) >= 12, "need >=12 titles"
    if c.get("number"):
        num = c["number"]
        assert set(num) <= {"field", "noun", "min", "max"}, \
            "number takes field/noun/min/max only"
        assert num["field"] and num["noun"], "number needs a field and a noun"
        assert isinstance(num["min"], int) and isinstance(num["max"], int), \
            "number bounds must be integers (the field is INT)"
        assert 0 <= num["min"] < num["max"], "number needs min < max, min >= 0"
        # against the field set the theme would have without it: a collision
        # here is silent otherwise, because `_child_fields` just overwrites
        # the other field's type with INT
        bare = {k: v for k, v in c.items() if k != "number"}
        taken = set(_child_fields({**theme, "child": bare}))
        assert num["field"] not in taken, \
            f"number field {num['field']!r} collides with another child field"
    st = theme["sort"]
    assert 0 <= st["time_index"] < len(c["times"]), "sort.time_index invalid"
    sb = theme["tools"]["set_bool"]
    assert 0 <= sb["bool_index"] < len(c["bools"])
    for a in theme["ambiguous"]:
        assert a["action"] in ("delete_child", "set_bool"), \
            "ambiguous action must be a single-arg tool slot"
    for bank in ("child_msgs", "agg_msgs", "fail_msgs"):
        assert len(theme["banks"][bank]) >= 3, f"bank {bank} too small"
    names = [t["name"] for t in
             (theme["tools"][k] for k in theme["tools"])]
    assert len(names) == len(set(names)), "duplicate tool names"
    v2 = set(_v2_names(theme).values())
    assert not (v2 & set(names)), f"theme tool names collide with v2 surface: {v2 & set(names)}"
    _validate_authored(theme)


# Which slots a flip probe can swap a decoy into (data/gen/flip_probe.py),
# and the request templates such a decoy must carry to be swappable: the
# keys the profile reads for that slot, each template carrying the
# placeholders the real slot's do.
FLIP_VERBS = {
    "delete_child": {"verbs": ("{obj}",)},
    "set_bool": {"verbs": ("{obj}",)},
    "set_ref": {"verbs": ("{obj}", "{to_name}")},
    "send": {"child_verbs": ("{obj}",), "direct_verbs": ("{name}",),
             "pair_verbs": ()},
}
_RESERVED_NAMES = {"sum", "avg", "max", "min"}


def swap_roles(theme: dict, choice: dict) -> dict:
    """The theme with, per flip slot in `choice`, sibling `choice[slot]`
    wired up as the working tool: 0 is the tool as authored, i > 0 its
    decoy i-1. The working tool takes the sibling's name, description and
    request templates; every other sibling, the authored tool included,
    becomes that slot's decoys. Structure and sandbox impl stay the slot's,
    so a program calling the working tool is the right program for a request
    written with its templates."""
    t = json.loads(json.dumps(theme))
    for slot, i in choice.items():
        real = t["tools"][slot]
        sibs = [{"name": real["name"], "desc": real["desc"],
                 **{k: real[k] for k in FLIP_VERBS[slot]}}] + real["decoys"]
        ans = sibs[i]
        t["tools"][slot] = {**real, "name": ans["name"], "desc": ans["desc"],
                            **{k: list(ans[k]) for k in FLIP_VERBS[slot]},
                            "decoys": [dict(x) for j, x in enumerate(sibs)
                                       if j != i]}
    return t


def role_choice(theme: dict, rng: random.Random) -> dict:
    """Uniform twin roles: each flip slot's working tool drawn uniformly
    from its siblings, the authored one included, so across a corpus no
    sibling's text is likelier to be the answer than another's
    (harness/decoy_audit.py). {} when the theme authors no flip decoys."""
    out = {}
    for slot in FLIP_VERBS:
        ds = theme["tools"][slot].get("decoys") or []
        if ds:
            out[slot] = rng.randrange(len(ds) + 1)
    return out


def request_fits(request: str, slot: str, spec: dict) -> bool:
    """Does the request carry one of this slot's request templates? Its
    longest literal chunk, case-insensitive; a bare pair verb must appear as
    a word. A recipe with its own fixed phrasing for the slot (the vague L9
    requests, the fallback "let {name} know") fails this for a swapped-in
    sibling, and the row is resampled rather than taught."""
    import re
    req = request.lower()
    for key in FLIP_VERBS[slot]:
        for tpl in spec.get(key) or []:
            if key == "pair_verbs":
                if re.search(rf"{re.escape(tpl.lower())}", req):
                    return True
                continue
            chunks = [c.strip() for c in re.split(r"\{[a-z_]+\}", tpl.lower())]
            lit = max(chunks, key=len) if chunks else ""
            if lit and lit in req:
                return True
    return False


def _validate_authored(theme: dict) -> None:
    """The optional authored-decoy blocks (THEME_SCHEMA.md, "Decoys")."""
    v2 = theme.get("v2")
    if v2 is not None:
        assert set(v2) == set(V2_SLOTS), \
            f"v2 block must cover exactly {V2_SLOTS}, got {sorted(v2)}"
        for slot, over in v2.items():
            assert over.get("name") and over.get("desc"), \
                f"v2.{slot} needs a name and a desc"
    specs = [(f"tools.{k}", v) for k, v in theme["tools"].items()]
    specs += [(f"v2.{k}", v) for k, v in (v2 or {}).items()]
    taken = set(_v2_names(theme).values()) | {
        v["name"] for v in theme["tools"].values()}
    assert len(taken) == len(theme["tools"]) + len(V2_SLOTS), \
        "v2 names collide with each other or with the theme's tools"
    assert not (taken & _RESERVED_NAMES), \
        f"tool names collide with the compute block: {taken & _RESERVED_NAMES}"
    for where, spec in specs:
        ds = spec.get("decoys")
        if ds is None:
            continue
        assert 2 <= len(ds) <= 4, f"{where}: 2-4 decoys, got {len(ds)}"
        slot = where.split(".", 1)[1]
        for d in ds:
            assert d.get("name") and d.get("desc"), \
                f"{where}: every decoy needs a name and a desc"
            assert d["name"] not in taken | _RESERVED_NAMES, \
                f"{where}: decoy name {d['name']!r} is taken"
            taken.add(d["name"])
            if where.startswith("tools.") and slot in FLIP_VERBS:
                for key, holes in FLIP_VERBS[slot].items():
                    vs = d.get(key)
                    assert vs and len(vs) >= 2, \
                        f"{where}: decoy {d['name']!r} needs >=2 {key}"
                    for v in vs:
                        for h in holes:
                            assert h in v, \
                                f"{where}: {d['name']!r} {key} {v!r} lacks {h}"


# domain -> the theme as authored, for anything that re-registers a variant of
# it (`role_theme`, data/gen/flip_probe.py) and has to put it back
THEMES: dict = {}


def register_theme(theme: dict, record: bool = True) -> str:
    name = theme["domain"]
    if record:
        THEMES[name] = theme
    existing = worlds_registry.WORLDS.get(name)
    if existing is not None and existing.get("post_hook"):
        # a CRUD theme cannot stand in for a world whose rules live in an
        # engine, and shadowing one is silent: the corpus keeps saying
        # world=X while X now means something else entirely
        raise ValueError(
            f"theme domain {name!r} would shadow the engine world of the "
            f"same name - rename one of them")
    worlds_registry.WORLDS[name] = build_world(theme)
    PROFILES[name] = build_profile(theme)
    worldgen.GENERATORS[name] = make_state_gen(theme)
    for bank, texts in theme["banks"].items():
        TEXT_BANKS[_bank(theme, bank)] = list(texts)
    return name


def register_domains(theme_dir: str | Path) -> list[str]:
    """Load, validate, and register every theme in a directory."""
    names = []
    for path in sorted(Path(theme_dir).glob("*.json")):
        names.append(register_theme(load_theme(path)))
    return names


# ------------------------------------------------------------ smoke testing
def smoke_theme(name: str, attempts_per_level: int = 80) -> list[str]:
    """Generate + execute a reference task at every level for a registered
    domain. Returns a list of failure strings (empty = pass)."""
    import random as _random

    from data.gen import english, programs
    from data.gen.worldgen import gen_state
    from harness.authoring import resolve
    from harness.context import build_context
    from harness.run import reference_planner, run_task
    from harness.taskbuild import ReferenceError, build_task
    from runtime.worlds import get_world

    failures = []
    world = get_world(name)
    for level in sorted(programs.RECIPES):
        last_err = "NO_SAMPLE"
        for attempt in range(attempts_per_level):
            seed = hash((name, level, attempt)) & 0xFFFFFF
            rng = _random.Random(seed)
            state = gen_state(name, rng, world["now"])
            try:
                sample = programs.sample_level(level, name, state,
                                               world["now"], rng, set())
                request, _ = english.render(sample.frame, rng)
                ctx, sandbox_ctx = build_context(
                    world, sample.constants, _random.Random(seed ^ 0x5EED),
                    None)
                segments = [resolve(s, ctx) for s in sample.segments]
                task = build_task(
                    task_id=f"{name}_L{level}_{seed}", level=level,
                    world_name=name, request=request,
                    constants=sample.constants, segments=segments, seed=seed,
                    error_injection=sample.error_injection or None,
                    expected_status="aborted" if "abort" in sample.tags else "ok",
                    state=state, tags=sample.tags, provenance={},
                    prebuilt=(ctx, sandbox_ctx))
            except (programs.SampleError, ReferenceError) as e:
                last_err = f"sample: {e}"
                continue
            except Exception as e:  # noqa: BLE001 — smoke must report, not die
                last_err = f"gen-crash: {type(e).__name__}: {e}"
                break
            row = run_task(task, reference_planner(task))
            if row["goal_success"]:
                last_err = None
            else:
                last_err = (f"L{level} {row['status']} "
                            f"{row['diagnostics'][:2]}")
            break
        if last_err:
            failures.append(f"L{level}: {last_err}")
    return failures


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="data.gen.domains")
    ap.add_argument("--validate", metavar="DIR",
                    help="register every theme in DIR and smoke-test all "
                         "11 levels per domain")
    ap.add_argument("--attempts", type=int, default=80)
    ap.add_argument("--check", nargs="+", metavar="FILE",
                    help="load, validate and compile these theme files only "
                         "(no smoke test): the quick check for authoring")
    args = ap.parse_args()
    if args.check:
        bad = 0
        for f in args.check:
            try:
                theme = load_theme(Path(f))
                build_world(theme)
                build_profile(theme)
                n = sum(len(v.get("decoys") or []) for v in
                        list(theme["tools"].values())
                        + list((theme.get("v2") or {}).values()))
                print(f"OK     {f}: {n} decoys, v2 "
                      f"{'authored' if theme.get('v2') else 'default'}")
            except Exception as e:  # noqa: BLE001
                bad += 1
                print(f"REJECT {f}: {e}")
        raise SystemExit(1 if bad else 0)
    if not args.validate:
        ap.error("--validate DIR required")
    passed, failed = [], []
    for path in sorted(Path(args.validate).glob("*.json")):
        try:
            name = register_theme(load_theme(path))
        except Exception as e:  # noqa: BLE001
            failed.append((path.name, [f"load/validate: {e}"]))
            print(f"REJECT {path.name}: load/validate: {e}", flush=True)
            continue
        fails = smoke_theme(name, args.attempts)
        if fails:
            failed.append((path.name, fails))
            print(f"REJECT {path.name}: {fails}", flush=True)
        else:
            passed.append(path.name)
            print(f"PASS   {path.name}", flush=True)
    print(f"\n{len(passed)} passed, {len(failed)} rejected")
    if failed:
        print("rejected:", [f[0] for f in failed])


if __name__ == "__main__":
    main()
