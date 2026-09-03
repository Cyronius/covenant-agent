"""Course-builder world tools from the real agent's schemas (plan
.claude/plans/real-sessions-eval-suite.md §2).

Inputs (committable; schemas carry no user data):
  data/schemas/mobi_frontend_tools.json   lm-admin tool maps (unbound chat)
  data/schemas/mobi_backend_tools.json    lm-python-functions factories
Output:
  data/schemas/coursebuilder_tools.json   loaded by runtime/worlds/coursebuilder.py

Typing rules follow data/gen/convert_open.py: string/int/bool params keep
their type (enum values go into the desc), arrays and objects are dropped
from the signature — except `props`, which is flattened into the optional
STR params the eval slice's writes actually carry (heading, text,
backgroundColor, fontFamily, image, title, altText). `columnId` /
`moduleId` / `subItemId` become ID:element / ID:module / ID:sub_item with
field links. Authored content is the writer tool's job: `write_text`,
`generate_image`, `edit_image` are EXTERNAL.

Implementation descriptors: entity CRUD where the tool is one (list / get /
create / update / delete on module, element, sub_item); everything else is
an `external` stub. State equality is not scored by the real-session
suite, routing is, so a stub is enough for the long tail.

  python -m data.gen.world_from_schemas            # writes the JSON, prints a summary
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

FRONTEND = ROOT / "data" / "schemas" / "mobi_frontend_tools.json"
BACKEND = ROOT / "data" / "schemas" / "mobi_backend_tools.json"
OUT = ROOT / "data" / "schemas" / "coursebuilder_tools.json"

ID_PARAMS = {
    "columnId": ("element", "id"), "sourceColumnId": ("element", "id"),
    "targetColumnId": ("element", "id"), "moduleId": ("module", "id"),
    "subItemId": ("sub_item", "id"), "courseId": ("course", "id"),
    "sourceCourseId": ("course", "id"),
}
# props.<key> -> element / sub_item field
ELEMENT_PROPS = ["heading", "text", "backgroundColor", "fontFamily", "image"]
SUB_ITEM_PROPS = ["title", "text", "image"]
PROP_ALIASES = {"paragraph": "text", "altText": None, "title": "heading"}

# world tool -> (source kind, source name) ; None = hand-written below
SOURCES = {
    "list_modules": None,
    "get_module": None,
    "list_elements": ("frontend", "get_module_elements"),
    "get_element": ("frontend", "get_element_details"),
    "list_sub_items": ("frontend", "get_sub_items"),
    "get_course_theme": ("frontend", "get_course_theme"),
    "get_available_fonts": ("frontend", "get_available_fonts"),
    "get_element_schema": ("frontend", "get_element_schema"),
    "get_course_metadata": ("frontend", "get_course_metadata"),
    "get_course_outline": ("frontend", "get_course_outline"),
    "find_courses": ("frontend", "find_courses"),
    "search_help": ("backend", "knowledge_base_search"),
    "add_module": ("frontend", "add_module"),
    "rename_module": ("frontend", "rename_module"),
    "delete_module": ("frontend", "delete_module"),
    "add_element": ("frontend", "add_element"),
    "update_element": ("frontend", "update_element"),
    "delete_element": ("frontend", "delete_element"),
    "move_element": ("frontend", "move_element"),
    "duplicate_element": ("frontend", "duplicate_element"),
    "transform_element": ("frontend", "transform_element"),
    "add_sub_item": ("frontend", "add_sub_item"),
    "update_sub_item": ("frontend", "update_sub_item"),
    "delete_sub_item": ("frontend", "delete_sub_item"),
    "update_course_metadata": ("frontend", "update_course_metadata"),
    "update_course_styles": ("frontend", "update_course_styles"),
    "apply_to_lessons": ("frontend", "apply_to_lessons"),
    "generate_lesson_content": ("frontend", "generate_lesson_content"),
    "add_course_to_curriculum": ("frontend", "add_course_to_curriculum"),
    "course_search_and_create": ("backend", "course_search_and_create"),
    "start_free_trial": ("backend", "start_free_trial"),
    "submit_contact_form": ("backend", "contact_us"),
    "export_document": ("backend", "export_document"),
    "generate_image": ("backend", "generate_image"),
    "edit_image": ("backend", "edit_image"),
    "write_text": None,
}
HAND_DESC = {
    "list_modules": "List every module (lesson or quiz) in the course, in order.",
    "get_module": "Fetch one module by id.",
    "search_help": "Search the help and knowledge bases with a question; returns the best passage.",
    "write_text": "Write prose from a brief (what to write, in the requester's words) and the elements it is about. Returns the text.",
    "add_course_to_curriculum": "Add a course to the curriculum.",
}
# Params the JSON-schema rules drop (arrays/objects) but the planner needs a
# literal for: the topic of a course to create, the brief for a lesson, the
# style values. STR stand-ins, appended after the converted params.
EXTRA_PARAMS = {
    "course_search_and_create": [{"name": "topic", "type": "STR", "desc": "topic of the course to find or create", "required": True}],
    "generate_lesson_content": [{"name": "moduleId", "type": "ID:module", "desc": "the lesson to fill", "required": True, "field": ["module", "id"]},
                                {"name": "brief", "type": "STR", "desc": "what the lesson should cover", "required": True}],
    "update_course_styles": [{"name": "fontFamily", "type": "STR", "desc": "styles.fontFamily", "required": False},
                             {"name": "backgroundColor", "type": "STR", "desc": "styles.backgroundColor", "required": False},
                             {"name": "headingColor", "type": "STR", "desc": "styles.headingColor", "required": False}],
    "apply_to_lessons": [{"name": "moduleId", "type": "ID:module", "desc": "one lesson to apply it to (omit = every lesson)", "required": False, "field": ["module", "id"]}],
}
HAND_PARAMS = {
    "list_modules": [],
    "get_module": [{"name": "moduleId", "type": "ID:module", "desc": "the module", "required": True,
                    "field": ["module", "id"]}],
    "write_text": [{"name": "brief", "type": "STR", "desc": "what to write, in the requester's words",
                    "required": True},
                   {"name": "data", "type": "LIST OBJ:element", "desc": "elements the text is about",
                    "required": True}],
}
# Effects: verb rule with explicit overrides.
EFFECTS = {
    "submit_contact_form": ["SEND"], "write_text": ["EXTERNAL"],
    "generate_image": ["EXTERNAL"], "edit_image": ["EXTERNAL"],
    "search_help": ["READ"], "export_document": ["WRITE"],
    "start_free_trial": ["WRITE"], "course_search_and_create": ["WRITE"],
}
RETURNS = {
    "list_modules": "LIST OBJ:module", "get_module": "OBJ:module",
    "list_elements": "LIST OBJ:element", "get_element": "OBJ:element",
    "list_sub_items": "LIST OBJ:sub_item", "get_course_theme": "STR",
    "get_available_fonts": "STR", "get_element_schema": "STR",
    "get_course_metadata": "OBJ:course", "get_course_outline": "LIST OBJ:module",
    "find_courses": "LIST OBJ:course", "search_help": "STR",
    "add_module": "OBJ:module", "rename_module": "OBJ:module",
    "add_element": "OBJ:element", "update_element": "OBJ:element",
    "move_element": "OBJ:element", "duplicate_element": "OBJ:element",
    "transform_element": "OBJ:element", "add_sub_item": "OBJ:sub_item",
    "update_sub_item": "OBJ:sub_item", "update_course_metadata": "OBJ:course",
    "write_text": "STR", "generate_image": "STR", "edit_image": "STR",
    "export_document": "STR",
}


def load_dumps():
    fe = json.loads(FRONTEND.read_text(encoding="utf-8")) if FRONTEND.exists() else {}
    be = json.loads(BACKEND.read_text(encoding="utf-8")) if BACKEND.exists() else {}
    fe_tools = {t["name"]: t for t in (fe.get("unbound_chat") or fe.get("tools") or [])}
    be_tools = {t["name"]: t for t in (be.get("tools") or [])}
    for t in list(be_tools.values()):     # the DB name is what the LLM saw
        if t.get("db_name"):
            be_tools[t["db_name"]] = t
    return fe_tools, be_tools, {"frontend": fe.get("source"), "backend": be.get("source")}


def ir_type(schema: dict):
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), None)
    if t in ("string",):
        return "STR"
    if t in ("integer", "number"):
        return "INT"
    if t == "boolean":
        return "BOOL"
    return None


def desc_of(schema: dict, name: str) -> str:
    d = str(schema.get("description") or name).strip()
    if schema.get("enum"):
        d += " (" + " | ".join(str(v) for v in schema["enum"]) + ")"
    return d[:120]


def short_desc(desc: str, limit: int = 160) -> str:
    """First sentence (the real descriptions run to a paragraph of agent
    guidance; the planner prompt wants one line per tool)."""
    d = " ".join(desc.split())
    m = re.match(r"(.+?[.!?])(\s|$)", d)
    first = m.group(1) if m else d
    if len(first) > limit:
        first = first[:limit - 1].rsplit(" ", 1)[0] + "…"
    return first


def effect_for(name: str) -> list:
    if name in EFFECTS:
        return EFFECTS[name]
    if name.startswith(("list_", "get_", "find_", "search_")):
        return ["READ"]
    if name.startswith("delete_"):
        return ["DELETE"]
    return ["WRITE"]


def convert(name: str, src: dict | None) -> dict:
    if src is None and name in HAND_PARAMS:
        params = [dict(p) for p in HAND_PARAMS[name]]
        return {"name": name, "desc": HAND_DESC[name], "params": params,
                "returns": RETURNS.get(name), "effects": effect_for(name),
                "impl": impl_for(name, params)}
    schema = ((src or {}).get("parameters") or {})
    props = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    params = []
    flattened = []
    for pname, pschema in props.items():
        if pname in ID_PARAMS:
            ent, fld = ID_PARAMS[pname]
            params.append({"name": pname, "type": f"ID:{ent}", "desc": desc_of(pschema, pname),
                           "required": pname in required, "field": [ent, fld]})
            continue
        if pname == "props":
            keys = SUB_ITEM_PROPS if "sub_item" in name else ELEMENT_PROPS
            ent = "sub_item" if "sub_item" in name else "element"
            for k in keys:
                params.append({"name": k, "type": "STR", "desc": f"props.{k}", "required": False,
                               "field": [ent, k]})
                flattened.append(k)
            continue
        ty = ir_type(pschema)
        if ty is None:
            continue   # arrays / objects other than props: not in the signature
        p = {"name": pname, "type": ty, "desc": desc_of(pschema, pname), "required": pname in required}
        params.append(p)
    params += [dict(p) for p in EXTRA_PARAMS.get(name, [])]
    # required first, keeping source order within each group
    params = [p for p in params if p["required"]] + [p for p in params if not p["required"]]
    desc = HAND_DESC.get(name) or str((src or {}).get("description") or name).strip()
    desc = short_desc(desc)
    tool = {"name": name, "desc": desc, "params": params,
            "returns": RETURNS.get(name), "effects": effect_for(name),
            "impl": impl_for(name, params)}
    if flattened:
        tool["flattened_props"] = flattened
    if src is None and name not in HAND_DESC:
        tool["note"] = "no source schema found; signature is empty"
    return tool


def _idx(params, pname):
    return next(i for i, p in enumerate(params) if p["name"] == pname)


def impl_for(name: str, params: list) -> dict:
    names = [p["name"] for p in params]
    if name == "list_modules":
        return {"op": "list", "entity": "module"}
    if name == "get_module":
        return {"op": "get", "entity": "module", "id_param": _idx(params, "moduleId")}
    if name == "list_elements":
        return {"op": "list", "entity": "element"}
    if name == "get_element":
        return {"op": "get", "entity": "element", "id_param": _idx(params, "columnId")}
    if name == "list_sub_items":
        return {"op": "list", "entity": "sub_item"}
    if name == "add_module":
        return {"op": "create", "entity": "module",
                "param_fields": [{"type": "type", "name": "name"}.get(n, None) for n in names],
                "defaults": {"name": "New module", "type": "lesson", "position": 99}}
    if name == "rename_module":
        return {"op": "update", "entity": "module", "id_param": _idx(params, "moduleId"),
                "set_from_params": {"name": _idx(params, "name")}}
    if name == "delete_module":
        return {"op": "delete", "entity": "module", "id_param": _idx(params, "moduleId")}
    if name == "add_element":
        pf = [n if n in ELEMENT_PROPS or n == "type" else ("module" if n == "moduleId" else None)
              for n in names]
        return {"op": "create", "entity": "element", "param_fields": pf,
                "defaults": {"module": "module_1", "type": "paragraph", "heading": "", "text": "",
                             "backgroundColor": "", "fontFamily": "", "image": "", "position": 99}}
    if name in ("update_element", "transform_element", "move_element"):
        sfp = {n: i for i, n in enumerate(names) if n in ELEMENT_PROPS}
        if "newType" in names:
            sfp["type"] = _idx(params, "newType")
        return {"op": "update", "entity": "element", "id_param": _idx(params, "columnId"),
                "set_from_params": sfp}
    if name == "delete_element":
        return {"op": "delete", "entity": "element", "id_param": _idx(params, "columnId")}
    if name == "add_sub_item":
        pf = [n if n in SUB_ITEM_PROPS else ("element" if n == "columnId" else None) for n in names]
        return {"op": "create", "entity": "sub_item", "param_fields": pf,
                "defaults": {"element": "element_5", "title": "", "text": "", "image": ""}}
    if name == "update_sub_item":
        return {"op": "update", "entity": "sub_item", "id_param": _idx(params, "subItemId"),
                "set_from_params": {n: i for i, n in enumerate(names) if n in SUB_ITEM_PROPS}}
    if name == "delete_sub_item":
        return {"op": "delete", "entity": "sub_item", "id_param": _idx(params, "subItemId")}
    if name == "submit_contact_form":
        return {"op": "send", "channel": "contact", "param_map": names}
    return {"op": "external", "kind": name}


def main() -> None:
    fe, be, sources = load_dumps()
    tools = []
    missing = []
    for name, src in SOURCES.items():
        s = None
        if src:
            kind, sname = src
            s = (fe if kind == "frontend" else be).get(sname)
            if s is None:
                missing.append(f"{name} <- {kind}:{sname}")
        tools.append(convert(name, s))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "generated": _dt.date.today().isoformat(), "sources": sources,
        "tools": tools}, indent=1), encoding="utf-8")
    print(f"{len(tools)} tools -> {OUT}")
    for t in tools:
        ps = " ".join(f"{p['name']}:{p['type']}{'' if p['required'] else '?'}" for p in t["params"])
        print(f"  {t['name']:26s} [{','.join(t['effects'])}] ({ps}) -> {t['returns']}")
    if missing:
        print("no source schema (hand-written signature):")
        for m in missing:
            print("  ", m)


if __name__ == "__main__":
    main()
