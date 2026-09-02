"""Level-targeted program sampling (F4): grammar-guided construction of valid
Agent Core programs (authoring form) plus the semantic frame the English
renderer consumes.

Every sampler returns a GenSample:
  frame            what the program does, for the renderer + provenance
  segments         authoring-form program segments (PAUSE-separated)
  constants        ordered constant specs ({type, value, desc})
  error_injection  sandbox injections (L8)
  tags             adversarial/phenomenon tags
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field as dc_field
from typing import List, Optional

from .profiles import DAY, PROFILES, TEXT_BANKS

PAIR_OUTER_GET = {"kanban": "get_user", "crm": "get_customer",
                  "projects": "get_member"}


@dataclass
class GenSample:
    frame: dict
    segments: List[str]
    constants: List[dict]
    error_injection: List[dict] = dc_field(default_factory=list)
    tags: List[str] = dc_field(default_factory=list)


class ConstAlloc:
    def __init__(self):
        self.items: List[dict] = []
        self._index = {}

    def get(self, type_: str, value, desc: str) -> str:
        key = (type_, json.dumps(value, sort_keys=True))
        if key in self._index:
            return f"${self._index[key]}"
        i = len(self.items)
        self.items.append({"type": type_, "value": value, "desc": desc})
        self._index[key] = i
        return f"${i}"

    def clone(self) -> "ConstAlloc":
        """Speculative allocation: indices stay aligned with the parent, so
        clause expressions minted in the clone stay valid after adopt()."""
        c = ConstAlloc()
        c.items = list(self.items)
        c._index = dict(self._index)
        return c

    def adopt(self, other: "ConstAlloc"):
        self.items[:] = other.items
        self._index = dict(other._index)


class SampleError(Exception):
    pass


# ------------------------------------------------------------------ clauses
def _records(state, entity):
    return state["entities"].get(entity, [])


def sample_clause(spec: dict, entity: str, state: dict, now: int,
                  rng: random.Random, alloc: ConstAlloc) -> dict:
    """Instantiate one filter clause. Returns a dict with:
    expr (authoring), sem {field, op, value}, phrase (text, placement),
    tags."""
    f = spec["field"]
    if spec["kind"] == "enum":
        v = rng.choice(spec["values"])
        ref = alloc.get("STR", v, spec["desc"].format(v=v))
        return {"expr": f"@{entity}.{f} EQ {ref}",
                "sem": {"field": f, "op": "EQ", "value": v},
                "phrase": spec["phrases"][v], "tags": []}
    if spec["kind"] == "bool":
        v = rng.random() < 0.5
        ref = alloc.get("BOOL", v, "true" if v else "false")
        phrase = spec["true_phrase"] if v else spec["false_phrase"]
        return {"expr": f"@{entity}.{f} EQ {ref}",
                "sem": {"field": f, "op": "EQ", "value": v},
                "phrase": (phrase, spec.get("placement", "pre")), "tags": []}
    if spec["kind"] == "time_now":
        op = "LT" if rng.random() < 0.8 else "GT"
        phrase = spec["lt_phrase"] if op == "LT" else spec["gt_phrase"]
        return {"expr": f"@{entity}.{f} {op} NOW",
                "sem": {"field": f, "op": op, "value": now},
                "phrase": phrase, "tags": ["adv:temporal"]}
    if spec["kind"] == "time_cutoff":
        lo, hi = spec["days"]
        d = rng.randint(lo, hi)
        cutoff = now - d * DAY
        ref = alloc.get("TIME", cutoff, spec["desc"].format(d=d))
        return {"expr": f"@{entity}.{f} LT {ref}",
                "sem": {"field": f, "op": "LT", "value": cutoff},
                "phrase": (spec["phrase"].format(d=d), "post"),
                "tags": ["adv:temporal"]}
    if spec["kind"] == "int_cmp":
        lo, hi = spec["range"]
        p = rng.randint(lo, hi)
        op = "LT" if rng.random() < 0.6 else "GT"
        ref = alloc.get("INT", p, spec["desc"].format(p=p))
        phrase = (spec["lt_phrase"] if op == "LT"
                  else spec["gt_phrase"]).format(p=p)
        return {"expr": f"@{entity}.{f} {op} {ref}",
                "sem": {"field": f, "op": op, "value": p},
                "phrase": (phrase, "post"), "tags": ["adv:quantifier"]}
    if spec["kind"] == "ref":
        recs = _records(state, spec["ref_entity"])
        if not recs:
            raise SampleError("no ref records")
        r = rng.choice(recs)
        name = r[spec["name_field"]]
        ref = alloc.get(f"ID:{spec['ref_entity']}", r["id"],
                        spec["desc"].format(name=name))
        return {"expr": f"@{entity}.{f} EQ {ref}",
                "sem": {"field": f, "op": "EQ", "value": r["id"]},
                "phrase": (spec["phrase"].format(name=name), "post"),
                "tags": ["adv:reference"]}
    raise SampleError(f"unknown filter kind {spec['kind']}")


def fixed_clause(c: dict, entity: str, profile_entity: dict, now: int,
                 alloc: ConstAlloc) -> dict:
    """Clause with a fixed value (ambiguous-scope recipes)."""
    spec = next(s for s in profile_entity["filters"]
                if s["field"] == c["field"] and s["kind"] == c["kind"])
    f = c["field"]
    if c["kind"] == "enum":
        ref = alloc.get("STR", c["value"], spec["desc"].format(v=c["value"]))
        return {"expr": f"@{entity}.{f} EQ {ref}",
                "sem": {"field": f, "op": "EQ", "value": c["value"]}}
    if c["kind"] == "bool":
        ref = alloc.get("BOOL", c["value"],
                        "true" if c["value"] else "false")
        return {"expr": f"@{entity}.{f} EQ {ref}",
                "sem": {"field": f, "op": "EQ", "value": c["value"]}}
    if c["kind"] == "time_now":
        return {"expr": f"@{entity}.{f} {c['op']} NOW",
                "sem": {"field": f, "op": c["op"], "value": now}}
    if c["kind"] == "time_cutoff":
        cutoff = now - c["days"] * DAY
        ref = alloc.get("TIME", cutoff, spec["desc"].format(d=c["days"]))
        return {"expr": f"@{entity}.{f} LT {ref}",
                "sem": {"field": f, "op": "LT", "value": cutoff}}
    raise SampleError(f"bad fixed clause {c}")


def eval_sem(rec: dict, sem: dict) -> bool:
    v = rec.get(sem["field"])
    if sem["op"] == "EQ":
        return v == sem["value"]
    if sem["op"] == "LT":
        return v < sem["value"]
    if sem["op"] == "GT":
        return v > sem["value"]
    raise SampleError(sem["op"])


def matches(state, entity, clauses) -> list:
    out = []
    for rec in _records(state, entity):
        ok = True
        for c in clauses:
            hit = eval_sem(rec, c["sem"])
            if c.get("neg"):
                hit = not hit
            if not hit:
                ok = False
                break
        if ok:
            out.append(rec)
    return out


def clause_expr(clauses) -> str:
    parts = []
    for i, c in enumerate(clauses):
        if i > 0:
            parts.append("AND")
        if c.get("neg"):
            parts.append("NOT")
        parts.append(c["expr"])
    return " ".join(parts)


# ------------------------------------------------------------------ actions
def build_action(action: dict, entity: str, subst: dict, state: dict,
                 rng: random.Random, alloc: ConstAlloc,
                 outer_entity: Optional[str] = None) -> dict:
    """Resolve an action spec into arg expressions + phrase info."""
    exprs = []
    info = {"tool": action["tool"], "to_value": "", "to_name": ""}
    for a in action["args"]:
        if a == "<v>":
            exprs.append(subst["<v>"])
        elif a == "<outer>":
            exprs.append(subst["<outer>"])
        elif isinstance(a, dict) and "field" in a:
            exprs.append(f"{subst['<v>']}.@{entity}.{a['field']}")
        elif isinstance(a, dict) and "outer_field" in a:
            exprs.append(
                f"{subst['<outer>']}.@{outer_entity}.{a['outer_field']}")
        elif isinstance(a, dict) and "const_text" in a:
            text = rng.choice(TEXT_BANKS[a["const_text"]["bank"]])
            exprs.append(alloc.get("STR", text, "message text"))
        elif isinstance(a, dict) and "const_text_addr" in a:
            spec = a["const_text_addr"]
            exprs.append(alloc.get("STR", spec["value"], spec["desc"]))
        elif isinstance(a, dict) and "const_enum" in a:
            spec = a["const_enum"]
            v = rng.choice(spec["values"])
            info["enum_value"] = v
            info["to_value"] = action.get("value_phrases", {}).get(v, v)
            exprs.append(alloc.get("STR", v, spec["desc"].format(v=v)))
        elif isinstance(a, dict) and "const_int" in a:
            spec = a["const_int"]
            v = rng.randint(*spec["range"])
            info["int_value"] = v
            info["to_value"] = action.get(
                "value_template", "{v}").format(v=v)
            exprs.append(alloc.get("INT", v, spec["desc"].format(v=v)))
        elif isinstance(a, dict) and "const_bool" in a:
            spec = a["const_bool"]
            exprs.append(alloc.get("BOOL", spec["value"], spec["desc"]))
        elif isinstance(a, dict) and "const_ref" in a:
            spec = a["const_ref"]
            recs = _records(state, spec["entity"])
            if not recs:
                raise SampleError("no ref records for action")
            r = rng.choice(recs)
            info["ref_name"] = r[spec["name_field"]]
            info["to_name"] = action.get(
                "to_template", "{name}").format(name=r[spec["name_field"]])
            exprs.append(alloc.get(f"ID:{spec['entity']}", r["id"],
                                   spec["desc"].format(
                                       name=r[spec["name_field"]])))
        else:
            raise SampleError(f"bad action arg {a}")
    verb = rng.choice(action["verbs"]).format(
        obj="{obj}", to_value=info["to_value"],
        to_name=info["to_name"]).strip()
    info["verb"] = verb
    info["dest"] = action.get("dest", False)
    info["arg_exprs"] = exprs
    return info


def action_line(info: dict, reg: Optional[str]) -> str:
    args = " ".join(info["arg_exprs"])
    dst = f" -> {reg}" if info["dest"] and reg else ""
    return f"CALL @{info['tool']}{args and ' ' + args}{dst}"


def visible_actions(entity_prof: dict, holdout_tools: set,
                    kinds: Optional[set] = None,
                    exclude_kinds: Optional[set] = None) -> list:
    out = []
    for a in entity_prof["actions"]:
        if a["tool"] in holdout_tools:
            continue
        k = a.get("kind", "simple")
        if kinds is not None and k not in kinds:
            continue
        if exclude_kinds and k in exclude_kinds:
            continue
        out.append(a)
    return out


def _named_record(entity_prof, entity, state, rng, alloc):
    recs = _records(state, entity)
    if not recs:
        raise SampleError("no records")
    r = rng.choice(recs)
    nf = entity_prof.get("name_field")
    display = r[nf] if nf else entity_prof["ref_word"].format(
        n=r["id"].split("_")[-1])
    ref = alloc.get(f"ID:{entity}", r["id"], f"{display}")
    return r, display, ref


# ------------------------------------------------------------------ recipes
def sample_direct(world, profile, state, now, rng, alloc, holdout, level=0):
    entity = rng.choice(sorted(profile["entities"]))
    prof = profile["entities"][entity]
    acts = visible_actions(prof, holdout, exclude_kinds={"send_field"})
    if rng.random() < 0.25 and profile.get("direct_send"):
        ds = profile["direct_send"]
        tgt = ds["target"]
        if "const_ref" in tgt:
            spec = tgt["const_ref"]
            recs = _records(state, spec["entity"])
            r = rng.choice(recs)
            name = r[spec["name_field"]]
            tref = alloc.get(f"ID:{spec['entity']}", r["id"],
                             spec["desc"].format(name=name))
        else:
            spec = tgt["const_text_addr"]
            name = spec["desc"]
            tref = alloc.get("STR", spec["value"], spec["desc"])
        text = rng.choice(TEXT_BANKS[ds["text_bank"]])
        cref = alloc.get("STR", text, "message text")
        verb = rng.choice(ds["verbs"]).format(name=name)
        seg = f"CALL @{ds['tool']} {tref} {cref}\nSTOP\n"
        frame = {"recipe": "direct_send", "verb": verb, "text": text}
        return GenSample(frame, [seg], alloc.items)
    if not acts:
        raise SampleError("no direct actions")
    act = rng.choice(acts)
    rec, display, ref = _named_record(prof, entity, state, rng, alloc)
    info = build_action(act, entity, {"<v>": ref}, state, rng, alloc)
    reg = "r0" if info["dest"] else None
    seg = action_line(info, reg) + "\nSTOP\n"
    frame = {"recipe": "direct", "entity": entity, "noun": prof["noun"],
             "record": display, "action": info}
    return GenSample(frame, [seg], alloc.items)


def sample_chain(world, profile, state, now, rng, alloc, holdout):
    cands = [(e, p) for e, p in sorted(profile["entities"].items())
             if p.get("get_tool")
             and visible_actions(p, holdout, kinds={"send_field"})]
    if not cands:
        raise SampleError("no chain entity")
    entity, prof = rng.choice(cands)
    act = rng.choice(visible_actions(prof, holdout, kinds={"send_field"}))
    rec, display, ref = _named_record(prof, entity, state, rng, alloc)
    info = build_action(act, entity, {"<v>": "r0"}, state, rng, alloc)
    seg = (f"CALL @{prof['get_tool']} {ref} -> r0\n"
           + action_line(info, None) + "\nSTOP\n")
    frame = {"recipe": "chain", "entity": entity, "noun": prof["noun"],
             "record": display, "action": info}
    return GenSample(frame, [seg], alloc.items, tags=["adv:reference"])


def _sample_clauses(prof, entity, state, now, rng, alloc, n, force_neg):
    specs = list(prof["filters"])
    rng.shuffle(specs)
    clauses = []
    for spec in specs:
        if len(clauses) == n:
            break
        try:
            c = sample_clause(spec, entity, state, now, rng, alloc)
        except SampleError:
            continue
        clauses.append(c)
    if len(clauses) < n:
        raise SampleError("not enough clauses")
    if force_neg:
        # negate one clause whose phrase reads naturally as an exception
        c = rng.choice(clauses)
        c["neg"] = True
        c["tags"] = c.get("tags", []) + ["adv:negation", "adv:exception"]
    return clauses


def sample_filter_act(world, profile, state, now, rng, alloc, holdout,
                      n_clauses, force_neg):
    entity = rng.choice(sorted(profile["entities"]))
    prof = profile["entities"][entity]
    acts = visible_actions(prof, holdout)
    if not acts:
        raise SampleError("no actions")
    for _ in range(6):
        trial = alloc.clone()
        clauses = _sample_clauses(prof, entity, state, now, rng, trial,
                                  n_clauses, force_neg)
        if matches(state, entity, clauses):
            break
    else:
        raise SampleError("no matching records")
    alloc.adopt(trial)
    act = rng.choice(acts)
    info = build_action(act, entity, {"<v>": "r2"}, state, rng, alloc)
    body = action_line(info, "r3" if info["dest"] else None)
    seg = (f"CALL @{prof['list_tool']} -> r0\n"
           f"FILTER r0 {clause_expr(clauses)} -> r1\n"
           f"FOREACH r1 -> r2\n"
           f"  {body}\n"
           "STOP\n")
    tags = sorted({t for c in clauses for t in c.get("tags", [])})
    frame = {"recipe": "filter_act", "entity": entity, "noun": prof["noun"],
             "clauses": [{"phrase": c["phrase"], "neg": c.get("neg", False)}
                         for c in clauses],
             "action": info}
    return GenSample(frame, [seg], alloc.items, tags=tags)


def sample_argmax(world, profile, state, now, rng, alloc, holdout):
    pair = rng.choice(profile["pairs"])
    inner_prof = profile["entities"][pair["inner"]]
    trial = alloc.clone()
    inner_clauses = _sample_clauses(inner_prof, pair["inner"], state, now,
                                    rng, trial, 1, False)
    alloc.adopt(trial)
    o, i, link = pair["outer"], pair["inner"], pair["link_field"]
    notify = build_action(pair["notify"], i, {"<outer>": "r3"}, state, rng,
                          alloc, outer_entity=o)
    seg = (f"CALL @{pair['outer_list']} -> r0\n"
           f"CALL @{inner_prof['list_tool']} -> r1\n"
           f"FILTER r1 {clause_expr(inner_clauses)} -> r2\n"
           f"FIRST r0 -> r3\n"
           f"FILTER r2 @{i}.{link} EQ r3.@{o}.id -> r4\n"
           f"COUNT r4 -> r5\n"
           f"FOREACH r0 -> r6\n"
           f"  FILTER r2 @{i}.{link} EQ r6.@{o}.id -> r7\n"
           f"  COUNT r7 -> r8\n"
           f"  IF r8 GT r5\n"
           f"    LET r8 -> r5\n"
           f"    LET r6 -> r3\n"
           + action_line(notify, None) + "\nSTOP\n")
    frame = {"recipe": "argmax_count", "outer_noun": pair["outer_noun"],
             "inner_noun": inner_prof["noun"],
             "clauses": [{"phrase": c["phrase"], "neg": False}
                         for c in inner_clauses],
             "action": notify}
    return GenSample(frame, [seg], alloc.items, tags=["adv:quantifier"])


def sample_sort(world, profile, state, now, rng, alloc, holdout,
                allow_ordinal=True):
    sort = rng.choice(profile["sorts"])
    entity = sort["entity"]
    prof = profile["entities"][entity]
    pf = sort["pre_filter"]
    clause = fixed_clause(
        {"kind": pf["kind"], "field": pf["field"],
         "value": pf.get("value"), "op": pf.get("op", "EQ")},
        entity, prof, now, alloc)
    pool = matches(state, entity, [clause])
    k = 0
    tags = ["adv:temporal" if "TIME" in str(sort["field"]) else
            "adv:quantifier"]
    if allow_ordinal and len(pool) > 2 and rng.random() < 0.35:
        k = rng.randint(1, min(2, len(pool) - 1))
        tags.append("adv:ordinal")
    if not pool or k >= len(pool):
        raise SampleError("sort pool too small")
    acts = visible_actions(prof, holdout)
    act = rng.choice(acts)
    info = build_action(act, entity, {"<v>": "r3"}, state, rng, alloc)
    pick = (f"FIRST r2 -> r3" if k == 0 else f"SELECT r2 {k} -> r3")
    seg = (f"CALL @{prof['list_tool']} -> r0\n"
           f"FILTER r0 {clause['expr']} -> r1\n"
           f"SORT r1 @{entity}.{sort['field']} {sort['dir']} -> r2\n"
           f"{pick}\n"
           + action_line(info, "r4" if info["dest"] else None) + "\nSTOP\n")
    ords = {1: "second", 2: "third"}
    np = (sort["sup_phrase"] if k == 0
          else sort["ord_phrase"].format(ord=ords[k]))
    frame = {"recipe": "extreme_sort", "np": np, "action": info}
    return GenSample(frame, [seg], alloc.items, tags=tags)


def sample_branch(world, profile, state, now, rng, alloc, holdout):
    cands = [(e, p) for e, p in sorted(profile["entities"].items())
             if p.get("get_tool")]
    if not cands:
        raise SampleError("no branch entity")
    entity, prof = rng.choice(cands)
    specs = [s for s in prof["filters"] if s["kind"] in ("bool", "enum")]
    if not specs:
        raise SampleError("no branch condition")
    spec = rng.choice(specs)
    cond = sample_clause(spec, entity, state, now, rng, alloc)
    acts = visible_actions(prof, holdout)
    if len(acts) < 2:
        raise SampleError("need two actions")
    a1, a2 = rng.sample(acts, 2)
    rec, display, ref = _named_record(prof, entity, state, rng, alloc)
    i1 = build_action(a1, entity, {"<v>": "r0"}, state, rng, alloc)
    i2 = build_action(a2, entity, {"<v>": "r0"}, state, rng, alloc)
    cond_rhs = cond["expr"].split(" EQ ")[1] if " EQ " in cond["expr"] \
        else cond["expr"].split(" ")[-1]
    seg = (f"CALL @{prof['get_tool']} {ref} -> r0\n"
           f"GET r0.@{entity}.{cond['sem']['field']} -> r1\n"
           f"IF r1 EQ {cond_rhs}\n"
           f"  {action_line(i1, 'r2' if i1['dest'] else None)}\n"
           f"ELSE\n"
           f"  {action_line(i2, 'r3' if i2['dest'] else None)}\n"
           "STOP\n")
    frame = {"recipe": "branch", "entity": entity, "noun": prof["noun"],
             "record": display, "cond_phrase": cond["phrase"],
             "then": i1, "else": i2}
    return GenSample(frame, [seg], alloc.items, tags=["adv:negation"])


def sample_nested(world, profile, state, now, rng, alloc, holdout):
    pair = rng.choice(profile["pairs"])
    o, i, link = pair["outer"], pair["inner"], pair["link_field"]
    inner_prof = profile["entities"][i]
    outer_prof = profile["entities"].get(o)
    outer_clauses = []
    if outer_prof and outer_prof.get("filters") and rng.random() < 0.9:
        trial = alloc.clone()
        outer_clauses = _sample_clauses(outer_prof, o, state, now, rng,
                                        trial, 1, False)
        if not matches(state, o, outer_clauses):
            outer_clauses = []
        else:
            alloc.adopt(trial)
    trial = alloc.clone()
    inner_clauses = _sample_clauses(inner_prof, i, state, now, rng, trial,
                                    1, False)
    alloc.adopt(trial)
    acts = visible_actions(inner_prof, holdout,
                           exclude_kinds={"send_field"})
    if not acts:
        raise SampleError("no inner actions")
    act = rng.choice(acts)
    info = build_action(act, i, {"<v>": "r5"}, state, rng, alloc)
    outer_f = (f"FILTER r0 {clause_expr(outer_clauses)} -> r2\n"
               if outer_clauses else "LET r0 -> r2\n")
    seg = (f"CALL @{pair['outer_list']} -> r0\n"
           f"CALL @{inner_prof['list_tool']} -> r1\n"
           + outer_f +
           f"FOREACH r2 -> r3\n"
           f"  FILTER r1 @{i}.{link} EQ r3.@{o}.id"
           f" AND {clause_expr(inner_clauses)} -> r4\n"
           f"  FOREACH r4 -> r5\n"
           f"    {action_line(info, 'r6' if info['dest'] else None)}\n"
           "STOP\n")
    frame = {"recipe": "nested", "outer_noun": pair["outer_noun"],
             "inner_noun": inner_prof["noun"],
             "outer_clauses": [{"phrase": c["phrase"], "neg": False}
                               for c in outer_clauses],
             "inner_clauses": [{"phrase": c["phrase"], "neg": False}
                               for c in inner_clauses],
             "action": info}
    return GenSample(frame, [seg], alloc.items, tags=["adv:reference"])


def sample_parallel(world, profile, state, now, rng, alloc, holdout):
    pair = rng.choice(profile["pairs"])
    o, i, link = pair["outer"], pair["inner"], pair["link_field"]
    inner_prof = profile["entities"][i]
    get_tool = pair.get("outer_get") or PAIR_OUTER_GET[world]
    recs = _records(state, o)
    if not recs:
        raise SampleError("no outer records")
    rec = rng.choice(recs)
    name = rec[pair["outer_name"]]
    ref = alloc.get(f"ID:{o}", rec["id"], f"{name}'s id")
    trial = alloc.clone()
    inner_clauses = _sample_clauses(inner_prof, i, state, now, rng, trial,
                                    1, False)
    alloc.adopt(trial)
    variant = rng.choice(["foreach", "count"])
    if variant == "foreach":
        acts = visible_actions(inner_prof, holdout,
                               exclude_kinds={"send_field"})
        if not acts:
            raise SampleError("no actions")
        act = rng.choice(acts)
        info = build_action(act, i, {"<v>": "r3"}, state, rng, alloc)
        tail = (f"FOREACH r2 -> r3\n"
                f"  {action_line(info, 'r4' if info['dest'] else None)}\n"
                "STOP\n")
    else:
        info = build_action(pair["notify"], i, {"<outer>": "r0"}, state,
                            rng, alloc, outer_entity=o)
        tail = (f"COUNT r2 -> r3\n"
                f"IF r3 GT 0\n"
                f"  {action_line(info, None)}\n"
                "STOP\n")
    seg = (f"PARALLEL\n"
           f"  CALL @{get_tool} {ref} -> r0\n"
           f"  CALL @{inner_prof['list_tool']} -> r1\n"
           f"FILTER r1 @{i}.{link} EQ {ref}"
           f" AND {clause_expr(inner_clauses)} -> r2\n"
           + tail)
    frame = {"recipe": "parallel", "variant": variant, "name": name,
             "outer_noun": pair["outer_noun"],
             "inner_noun": inner_prof["noun"],
             "clauses": [{"phrase": c["phrase"], "neg": False}
                         for c in inner_clauses],
             "action": info}
    tags = ["adv:quantifier"] if variant == "count" else []
    return GenSample(frame, [seg], alloc.items, tags=tags)


def sample_recovery(world, profile, state, now, rng, alloc, holdout):
    variant = rng.choice(["retry", "fallback"])
    if variant == "retry":
        entity = rng.choice(sorted(profile["entities"]))
        prof = profile["entities"][entity]
        acts = visible_actions(prof, holdout, exclude_kinds={"send_field"})
        if not acts:
            raise SampleError("no actions")
        act = rng.choice(acts)
        rec, display, ref = _named_record(prof, entity, state, rng, alloc)
        info = build_action(act, entity, {"<v>": ref}, state, rng, alloc)
        retries = rng.randint(2, 3)
        times = rng.randint(1, retries)
        seg = (f"TRY RETRY {retries} -> r0\n"
               f"  {action_line(info, 'r1' if info['dest'] else None)}\n"
               "STOP\n")
        frame = {"recipe": "recovery_retry", "entity": entity,
                 "noun": prof["noun"], "record": display, "action": info}
        inj = [{"name": info["tool"], "code": "RATE_LIMITED",
                "times": times}]
        return GenSample(frame, [seg], alloc.items, error_injection=inj)
    # fallback
    cands = [(e, p) for e, p in sorted(profile["entities"].items())
             if p.get("get_tool")
             and visible_actions(p, holdout, kinds={"send_field"})]
    if not cands:
        raise SampleError("no fallback entity")
    entity, prof = rng.choice(cands)
    act = rng.choice(visible_actions(prof, holdout, kinds={"send_field"}))
    rec, display, ref = _named_record(prof, entity, state, rng, alloc)
    ok_ref = alloc.get("STATUS", "OK", "the OK status")
    info = build_action(act, entity, {"<v>": "r1"}, state, rng, alloc)
    fb = profile["fallback_notify"]
    tgt = fb["target"]
    if "const_ref" in tgt:
        spec = tgt["const_ref"]
        recs = _records(state, spec["entity"])
        r = rng.choice(recs)
        fname = r[spec["name_field"]]
        tref = alloc.get(f"ID:{spec['entity']}", r["id"],
                         spec["desc"].format(name=fname))
    else:
        spec = tgt["const_text_addr"]
        fname = spec["desc"]
        tref = alloc.get("STR", spec["value"], spec["desc"])
    text = rng.choice(TEXT_BANKS[fb["text_bank"]])
    fb_text = alloc.get("STR", text, "failure notice text")
    fb_verb = rng.choice(fb["verbs"]).format(name=fname)
    code = rng.choice(["NOT_FOUND", "PERMISSION_DENIED"])
    seg = (f"TRY -> r0\n"
           f"  CALL @{prof['get_tool']} {ref} -> r1\n"
           f"IF r0 EQ {ok_ref}\n"
           f"  {action_line(info, None)}\n"
           f"ELSE\n"
           f"  CALL @{fb['tool']} {tref} {fb_text}\n"
           "STOP\n")
    frame = {"recipe": "recovery_fallback", "entity": entity,
             "noun": prof["noun"], "record": display, "action": info,
             "fb_verb": fb_verb}
    inj = [{"name": prof["get_tool"], "code": code, "times": -1}]
    return GenSample(frame, [seg], alloc.items, error_injection=inj)


def sample_ambiguous(world, profile, state, now, rng, alloc, holdout):
    spec = rng.choice(profile["ambiguous"])
    entity = spec["entity"]
    prof = profile["entities"][entity]
    clauses = [fixed_clause(c, entity, prof, now, alloc)
               for c in spec["clauses"]]
    act_spec = next(a for a in prof["actions"]
                    if a["tool"] == spec["action_tool"])
    if act_spec["tool"] in holdout:
        raise SampleError("ambiguous action is holdout")
    info = build_action(act_spec, entity, {"<v>": "r2"}, state, rng, alloc)
    seg = (f"CALL @{prof['list_tool']} -> r0\n"
           f"FILTER r0 {clause_expr(clauses)} -> r1\n"
           f"FOREACH r1 -> r2\n"
           f"  {action_line(info, 'r3' if info['dest'] else None)}\n"
           "STOP\n")
    frame = {"recipe": "ambiguous", "request": rng.choice(spec["request"]),
             "hint": spec["hint"]}
    return GenSample(frame, [seg], alloc.items, tags=["ambiguous"])


def sample_pause(world, profile, state, now, rng, alloc, holdout):
    base = sample_filter_act(world, profile, state, now, rng, alloc,
                             holdout, rng.choice([1, 2]), False)
    lines = base.segments[0].rstrip("\n").split("\n")
    # discovery = list + filter; execution = the rest
    seg1 = "\n".join(lines[:2]) + "\nPAUSE\n"
    seg2 = "\n".join(lines[2:]) + "\n"
    base.segments = [seg1, seg2]
    base.frame = dict(base.frame, recipe="pause_" + base.frame["recipe"])
    return base


# Names that never appear in any generated world state: a request naming one
# of these has no matching constant, and the reference program abstains.
ABSENT_NAMES = ["Cyrus", "Marguerite", "Tobias", "Ingrid", "Rafael", "Wen",
                "Oluwaseun", "Katarina", "Desmond", "Priyanka"]
# For entities named by a title-like field ("title", "item", "route", ...)
# rather than a person name: neutral titles the worldgen never produces.
ABSENT_TITLES = ["Q3 budget review", "Northwind onboarding", "Harbor lights",
                 "Legacy migration", "Winter retrospective", "Blue corridor",
                 "Sunset audit", "Pilot phase two", "Orchard survey",
                 "Meridian handover"]


def sample_abort(world, profile, state, now, rng, alloc, holdout):
    """Level 11, NOT_FOUND flavor: a direct action on a record that does not
    exist. Same frame shape as `direct`, so the English renders naturally,
    but no ID constant is allocated for the record and the reference is
    `ABORT NOT_FOUND`. (UNSUPPORTED / NEEDS_INFO / AMBIGUOUS flavors: see
    harness/curriculum.py L11 examples; generator coverage is Lane C.)"""
    entity = rng.choice(sorted(profile["entities"]))
    prof = profile["entities"][entity]
    acts = visible_actions(prof, holdout, exclude_kinds={"send_field"})
    if not acts:
        raise SampleError("no direct actions")
    act = rng.choice(acts)
    nf = prof.get("name_field")
    if nf:
        existing = {r.get(nf) for r in _records(state, entity)}
        source = ABSENT_NAMES if nf == "name" else ABSENT_TITLES
        pool = [n for n in source if n not in existing]
        display = rng.choice(pool)
    else:
        ids = [int(r["id"].split("_")[-1]) for r in _records(state, entity)
               if r["id"].split("_")[-1].isdigit()]
        display = prof["ref_word"].format(n=(max(ids) if ids else 0) + rng.randint(7, 40))
    # build_action may still allocate the action's own constants (an enum
    # value, a message) — those are legitimately in the request.
    info = build_action(act, entity, {"<v>": "r0"}, state, rng, alloc)
    frame = {"recipe": "direct", "entity": entity, "noun": prof["noun"],
             "record": display, "action": info}
    return GenSample(frame, ["ABORT NOT_FOUND\n"], alloc.items,
                     tags=["abort", "not_found"])


RECIPES = {
    0: [("direct", lambda *a: sample_direct(*a))],
    1: [("chain", lambda *a: sample_chain(*a))],
    2: [("filter_act",
         lambda *a: sample_filter_act(*a, n_clauses=1, force_neg=False))],
    3: [("filter_act3",
         lambda *a: sample_filter_act(
             *a, n_clauses=random.Random().randint(2, 3), force_neg=True))],
    4: [("argmax", lambda *a: sample_argmax(*a)),
        ("sort", lambda *a: sample_sort(*a))],
    5: [("branch", lambda *a: sample_branch(*a))],
    6: [("nested", lambda *a: sample_nested(*a))],
    7: [("parallel", lambda *a: sample_parallel(*a))],
    8: [("recovery", lambda *a: sample_recovery(*a))],
    9: [("ambiguous", lambda *a: sample_ambiguous(*a))],
    10: [("pause", lambda *a: sample_pause(*a))],
    11: [("abort", lambda *a: sample_abort(*a))],
}


def sample_level(level: int, world: str, state: dict, now: int,
                 rng: random.Random, holdout_tools: set) -> GenSample:
    alloc = ConstAlloc()
    name, fn = rng.choice(RECIPES[level])
    if level == 3:
        return sample_filter_act(world, PROFILES[world], state, now, rng,
                                 alloc, holdout_tools,
                                 n_clauses=rng.randint(2, 3), force_neg=True)
    return fn(world, PROFILES[world], state, now, rng, alloc, holdout_tools)
