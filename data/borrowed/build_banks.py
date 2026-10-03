"""Build data/borrowed/<source>/bank.jsonl from local clones in %TEMP%/borrowed_src.

Usage: python build_banks.py <source> [<source> ...]   (or: all)
Clones each upstream repo into %TEMP%/borrowed_src first (see each
source README for the revision); the banks are this script's output.
"""
import ast
import collections
import glob
import json
import os
import random
import re
import subprocess
import sys

SRC = os.path.join(os.environ.get("TEMP", "/tmp"), "borrowed_src")
OUT = os.path.dirname(os.path.abspath(__file__))


def git_rev(repo):
    return subprocess.check_output(["git", "-C", os.path.join(SRC, repo), "rev-parse", "HEAD"], text=True).strip()


def write_bank(source, licence, url, revision, items):
    d = os.path.join(OUT, source)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, "bank.jsonl")
    seen = set()
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for kind, text, meta in items:
            key = (kind, text, json.dumps(meta, sort_keys=True))
            if key in seen:
                continue
            seen.add(key)
            row = {"source": source, "licence": licence, "url": url, "revision": revision,
                   "kind": kind, "text": text, "meta": meta}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    kinds = collections.Counter(k for k, _, _ in seen)
    print(f"{source}: {n} rows, {os.path.getsize(path)/1e6:.2f} MB, kinds={dict(kinds)}")


# ---------------------------------------------------------------- ALFWorld
def alfworld():
    repo = os.path.join(SRC, "alfworld")
    rev = git_rev("alfworld")
    sys.path.insert(0, os.path.join(repo, "alfworld", "gen"))
    import constants as C
    import goal_library as G

    six = ["pick_and_place_simple", "pick_two_obj_and_place", "look_at_obj_in_light",
           "pick_clean_then_place_in_recep", "pick_heat_then_place_in_recep",
           "pick_cool_then_place_in_recep"]
    items = []
    gl = "alfworld/gen/goal_library.py"
    for t in six:
        for variant in (t, t + "_slice"):
            if variant not in G.gdict:
                continue
            g = G.gdict[variant]
            pddl = re.sub(r"[ \t]+", " ", g["pddl"]).strip()
            items.append(("task_type", variant, {
                "task_type": t, "sliced": variant.endswith("_slice"),
                "templates": g["templates"], "pddl_goal": pddl,
                "valid_scene_types": sorted(C.GOALS_VALID.get(t, [])),
                "file": gl}))
            for tpl in g["templates"]:
                items.append(("task_template", tpl, {
                    "task_type": t, "sliced": variant.endswith("_slice"),
                    "slots": re.findall(r"\{(\w+)\}", tpl), "file": gl}))

    # action + feedback templates from the TextWorld grammar
    tw = open(os.path.join(repo, "alfworld", "data", "alfred.twl2"), encoding="utf-8").read()
    for m in re.finditer(r'action (\w+) \{\s*template :: "([^"]+)";', tw):
        name, tpl = m.group(1), m.group(2)
        verb = tpl.split()[0]
        items.append(("action_template", tpl, {
            "action": name, "verb": verb,
            "slots": re.findall(r"\{(\w+)", tpl), "file": "alfworld/data/alfred.twl2"}))
    # feedback rhs strings (observation wording)
    for m in re.finditer(r'"([\w.()\-, ]+?)":\s*\[(.*?)\n\s*\]', tw, re.S):
        rule, body = m.group(1), m.group(2)
        for rm in re.finditer(r'(?:"condition":\s*"([^"]*)",\s*)?"rhs":\s*"((?:[^"\\]|\\.)*)"', body):
            cond, rhs = rm.group(1), rm.group(2)
            if not rhs.strip() or rhs in ("TODO", ", ", ", and ", "nothing") or rhs.startswith("\\nAvailable"):
                continue
            items.append(("feedback_template", rhs.encode().decode("unicode_escape"), {
                "rule": rule, "condition": cond, "file": "alfworld/data/alfred.twl2"}))

    # vocabulary
    const = "alfworld/gen/constants.py"
    movable = set(C.MOVABLE_RECEPTACLES)
    static = set(C.RECEPTACLES) - movable
    accepts = {r: sorted(v) for r, v in C.VAL_RECEPTACLE_OBJECTS.items()}
    pickupable = set().union(*C.VAL_RECEPTACLE_OBJECTS.values())
    afford = collections.defaultdict(list)
    for a, objs in C.VAL_ACTION_OBJECTS.items():
        for o in objs:
            afford[o].append(a.lower())
    for r in sorted(C.RECEPTACLES | {"Sink", "Bathtub"}):
        items.append(("receptacle", r.lower(), {
            "class": r, "movable": r in movable, "openable": r in C.OPENABLE_CLASS_SET,
            "accepts": accepts.get(r, []), "file": const}))
    for o in sorted(set(C.OBJECTS_WSLICED)):
        if o in static:
            continue
        items.append(("object", o.lower(), {
            "class": o, "pickupable": o in pickupable, "movable_receptacle": o in movable,
            "affordances": sorted(afford.get(o, [])),
            "can_go_in": sorted(r for r, v in C.VAL_RECEPTACLE_OBJECTS.items() if o in v),
            "file": const}))
    write_bank("alfworld", "MIT", "https://github.com/alfworld/alfworld", rev, items)


# ---------------------------------------------------------------- WebArena
WA_DROP = ["Nike", "Sony", "Anker", "Oral B", "EYZUTAK", "XBox", "Nintendo", "Hyatt",
           "Pittsburgh", "pittsburgh", "Philadelphia", "CMU", "Carnegie Mellon", "massachusetts",
           "Declaration of Independence", "midjourney"]
WA_SITE_NAMES = ["One Stop Market", "one stop market", "OneStopShopping", "OneStopShop", "OneStopMarket",
                 "GitLab", "Gitlab", "gitlab", "Reddit", "reddit", "r/books", "/f/pics", "DIY subreddit",
                 "games subreddit", "on Map", "AutoAGI", "awesome-llms", "llm_bulk_inference",
                 '"planner"', "web_arena", "AC-DC Adapter", "dotfile"]


def webarena():
    repo = os.path.join(SRC, "webarena")
    rev = git_rev("webarena")
    d = json.load(open(os.path.join(repo, "config_files", "test.raw.json"), encoding="utf-8"))
    groups = collections.OrderedDict()
    for x in d:
        k = (x["intent_template_id"], x["intent_template"])
        g = groups.setdefault(k, {"sites": set(), "n": 0})
        g["sites"].add(tuple(x["sites"]))
        g["n"] += 1
    items, dropped = [], []
    for (tid, tpl), g in sorted(groups.items()):
        hits = [w for w in WA_DROP if w in tpl]
        if hits:
            dropped.append((tid, tpl, hits))
            continue
        sites = sorted({s for t in g["sites"] for s in t})
        items.append(("intent_template", tpl, {
            "template_id": tid, "sites": sites,
            "slots": re.findall(r"\{\{(.+?)\}\}", tpl),
            "n_instances": g["n"],
            "site_names_to_rewrite": [w for w in WA_SITE_NAMES if w in tpl],
            "file": "config_files/test.raw.json"}))
    # action vocabulary from the prompt
    p = open(os.path.join(repo, "agent", "prompts", "raw", "p_cot_id_actree_2s.py"), encoding="utf-8").read()
    cat = None
    for line in p.split("\n"):
        if line.endswith("Actions:"):
            cat = line[:-1].strip()
        m = re.match(r"`([^`]+)`:\s*(.*)", line)
        if m:
            items.append(("action", m.group(1), {"category": cat, "description": m.group(2).strip(),
                                                 "file": "agent/prompts/raw/p_cot_id_actree_2s.py"}))
    write_bank("webarena", "Apache-2.0", "https://github.com/web-arena-x/webarena", rev, items)
    print("  dropped templates:", len(dropped))
    for x in dropped:
        print("   ", x)


# ---------------------------------------------------------------- OSWorld
def osworld():
    repo = os.path.join(SRC, "OSWorld")
    rev = git_rev("OSWorld")
    ex = os.path.join(repo, "evaluation_examples")
    test_all = json.load(open(os.path.join(ex, "test_all.json")))
    in_all = {i for v in test_all.values() for i in v}
    items, os_dropped = [], []
    for f in sorted(glob.glob(os.path.join(ex, "examples", "*", "*.json"))):
        dom = os.path.basename(os.path.dirname(f))
        t = json.load(open(f, encoding="utf-8"))
        cfg_types = sorted({c.get("type") for c in t.get("config", []) if isinstance(c, dict)})
        ev = t.get("evaluator", {})
        src_s = t.get("source") or ""
        src_s = " ".join(src_s) if isinstance(src_s, list) else str(src_s)
        upstream = next((u for u in ("GAIA", "Mind2Web", "NL2Bash", "SheetCopilot") if u.lower() in src_s.lower()), None)
        if upstream in ("GAIA", "Mind2Web"):
            os_dropped.append((dom, t["id"], upstream, t["instruction"][:80]))
            continue
        items.append(("task", t["instruction"].strip(), {
            "upstream_dataset": upstream,
            "domain": dom, "id": t["id"], "related_apps": t.get("related_apps", []),
            "snapshot": t.get("snapshot"), "idea_source": t.get("source"),
            "infeasible": ev.get("func") == "infeasible",
            "in_test_all": t["id"] in in_all,
            "setup_uses_files": any(c in cfg_types for c in ("download", "open", "upload_file")),
            "file": f"evaluation_examples/examples/{dom}/{os.path.basename(f)}"}))
    write_bank("osworld", "Apache-2.0", "https://github.com/xlang-ai/OSWorld", rev, items)
    print("  dropped:", len(os_dropped))
    for x in os_dropped:
        print("   ", x)
    print("  upstream kept:", collections.Counter(m["upstream_dataset"] for _, _, m in items))


# ---------------------------------------------------------------- Mind2Web
def mind2web():
    rows = json.load(open(os.path.join(SRC, "m2w", "train_cols.json"), encoding="utf-8"))
    rev = "main@17ece8eb89862368edc0cc806acee6fca5163474 (parquet: refs/convert/parquet@eabe74c3532cf3a35ff02913cece5341bd1ca0d5)"
    items = []
    ops = collections.defaultdict(list)
    for r in rows:
        items.append(("task", r["confirmed_task"].strip(), {
            "split": "train", "annotation_id": r["annotation_id"], "website": r["website"],
            "domain": r["domain"], "subdomain": r["subdomain"],
            "action_reprs": r["action_reprs"]}))
        for a in r["action_reprs"]:
            m = re.search(r"->\s*([A-Z]+)", a)
            if m:
                ops[m.group(1)].append(a)
    rng = random.Random(0)
    for op, ex in sorted(ops.items()):
        items.append(("action_op", op, {"count": len(ex), "examples": rng.sample(ex, min(5, len(ex)))}))
    write_bank("mind2web", "CC-BY-4.0", "https://huggingface.co/datasets/osunlp/Mind2Web", rev, items)


# ---------------------------------------------------------------- WebShop
def webshop():
    repo = os.path.join(SRC, "WebShop")
    rev = git_rev("WebShop")
    items = []
    goal = "web_agent_site/engine/goal.py"
    items.append(("goal_template", "{instruction}, and price lower than {price_upper} dollars", {
        "use": "human goals: MTurk instruction with trailing period stripped, plus a price cap",
        "slots": ["instruction", "price_upper"], "file": goal}))
    items.append(("goal_template", "{instruction_text} with {option_name}: {option_value}, and {option_name}: {option_value}, and price lower than {price_upper} dollars", {
        "use": "synthetic goals: option clauses joined by ', and ', each optional; price clause optional",
        "slots": ["instruction_text", "option_name", "option_value", "price_upper"], "file": goal}))
    env = "web_agent_site/envs/web_agent_text_env.py"
    items.append(("action", "search[keywords]", {"description": "type a search query on the search page", "file": env}))
    items.append(("action", "click[value]", {"description": "click a button, a result, or an option value", "file": env}))
    eng = "web_agent_site/engine/engine.py"
    for b, role in [("Back to Search", "return to the search page"), ("Next >", "next results page"),
                    ("< Prev", "previous results page / leave a sub-page"), ("Buy Now", "purchase and end the episode"),
                    ("Description", "item sub-page"), ("Features", "item sub-page"),
                    ("Reviews", "item sub-page"), ("Attributes", "item sub-page")]:
        items.append(("button", b, {"role": role, "file": eng}))
    for pg in sorted(glob.glob(os.path.join(repo, "web_agent_site", "templates", "*.html"))):
        name = os.path.basename(pg)[:-5]
        items.append(("page", name, {"file": "web_agent_site/templates/" + os.path.basename(pg)}))
    # human instructions (MTurk), sampled; no worker/assignment IDs, no ASINs
    d = json.load(open(os.path.join(repo, "baseline_models", "data", "items_human_ins.json"), encoding="utf-8"))
    flat, seen_t = [], set()
    for asin in sorted(d):
        for ins in d[asin]:
            t = (ins.get("instruction") or "").strip()
            if t and t.lower() not in seen_t:
                seen_t.add(t.lower())
                flat.append((t, ins.get("instruction_attributes", []), ins.get("instruction_options", [])))
    rng = random.Random(0)
    sample = rng.sample(flat, 3000)
    for t, a, o in sample:
        items.append(("human_instruction", t, {"instruction_attributes": a, "instruction_options": o,
                                               "file": "baseline_models/data/items_human_ins.json"}))
    print(f"  webshop human instructions: {len(flat)} unique, sampled 3000")
    write_bank("webshop", "MIT", "https://github.com/princeton-nlp/WebShop", rev, items)


# ---------------------------------------------------------------- TravelPlanner
def travelplanner():
    repo = os.path.join(SRC, "TravelPlanner")
    rev = git_rev("TravelPlanner")
    items = []
    p = open(os.path.join(repo, "agents", "prompts.py"), encoding="utf-8").read()
    zs = p.split('ZEROSHOT_REACT_INSTRUCTION = """', 1)[1].split('"""', 1)[0]
    for m in re.finditer(r"\((\d)\) (\w+)\[([^\]]*)\]:?\n(.*?)(?=\n\(\d\) |\nYou should use|\nEach action)", zs, re.S):
        num, name, args, body = m.groups()
        desc = re.search(r"Description: (.*)", body)
        ex = re.search(r"Example: (.*)", body)
        params = []
        pm = re.search(r"Parameters?:\s*(.*?)(?=\nExample:)", body, re.S)
        if pm:
            for line in pm.group(1).strip().split("\n"):
                line = line.strip()
                mm = re.match(r"([\w ]+?)\s*[:\-\u2013]\s+(.*)", line)
                if mm:
                    params.append({"name": mm.group(1).strip(), "description": mm.group(2).strip()})
        items.append(("tool", desc.group(1).strip() if desc else "", {
            "name": name, "signature": f"{name}[{args}]", "parameters": params,
            "example": ex.group(1).strip() if ex else None, "file": "agents/prompts.py"}))
    # hard constraints: types and value sets (utils/query_element_selection.py), messages (evaluation/hard_constraint.py)
    qes = "utils/query_element_selection.py"
    hard = [
        ("budget", "Budget", None, "total plan cost must be <= budget; budget rounded to hundreds, derived from a cost estimate x people x level factor"),
        ("house rule", "Room Rule", ["parties", "smoking", "children under 10", "visitors", "pets"], "accommodation must allow the named activity"),
        ("cuisine", "Cuisine", ["Chinese", "American", "Italian", "Mexican", "Indian", "Mediterranean", "French"], "every listed cuisine must appear among the meals (medium picks 2, hard picks 4)"),
        ("room type", "Room Type", ["shared room", "not shared room", "private room", "entire room"], "shared/not shared only when people <= 2"),
        ("transportation", "Transportation", ["no flight", "no self-driving"], "forbidden transport mode"),
    ]
    for key, paper, values, note in hard:
        items.append(("hard_constraint", key, {"paper_term": paper, "values": values, "note": note, "file": qes}))
    # levels
    items.append(("difficulty_level", "easy", {"people": [1], "local_constraints": 0, "budget": True, "file": qes}))
    items.append(("difficulty_level", "medium", {"people": [2, 3, 4, 5, 6, 7, 8], "local_constraints": 1,
                                                 "choose_from": ["house rule", "cuisine", "room type"], "budget": True, "file": qes}))
    items.append(("difficulty_level", "hard", {"people": [2, 3, 4, 5, 6, 7, 8], "local_constraints": 3,
                                               "choose_from": ["house rule", "cuisine", "room type", "transportation"], "budget": True, "file": qes}))
    items.append(("trip_shape", "days -> visiting_city_number", {"mapping": {"3": 1, "5": 2, "7": 3},
                                                                 "note": "3-day trips go city to city; 5/7-day trips go city to a state and visit 2/3 of its cities; dates within 2022-03",
                                                                 "file": qes}))
    # constraint violation messages (both evaluators), via AST
    terms = {'is_valid_information_in_current_city': 'Within Current City', 'is_valid_information_in_sandbox': 'Within Sandbox',
             'is_reasonable_visiting_city': 'Reasonable City Route', 'is_valid_restaurants': 'Diverse Restaurants',
             'is_valid_transportation': 'Non-conf. Transportation', 'is_valid_attractions': 'Diverse Attractions',
             'is_valid_accommodaton': 'Minimum Nights Stay', 'is_not_absent': 'Complete Information',
             'is_valid_room_rule': 'Room Rule', 'is_valid_cuisine': 'Cuisine', 'is_valid_room_type': 'Room Type'}
    for fname, group in [("evaluation/commonsense_constraint.py", "commonsense"), ("evaluation/hard_constraint.py", "hard")]:
        src = open(os.path.join(repo, fname), encoding="utf-8").read()
        tree = ast.parse(src)
        for fn in tree.body:
            if not isinstance(fn, ast.FunctionDef) or not fn.name.startswith("is_"):
                continue
            term = terms.get(fn.name)
            if group == "hard" and fn.name == "is_valid_transportation":
                term = "Transportation"
            if group == "commonsense" and fn.name in ("is_valid_city_sequence",):
                continue
            if group == "commonsense":
                items.append(("commonsense_constraint", term or fn.name, {"function": fn.name, "scored_in_eval_py": term is not None, "file": fname}))
            for node in ast.walk(fn):
                if isinstance(node, ast.Return) and isinstance(node.value, ast.Tuple) and len(node.value.elts) == 2:
                    a, b = node.value.elts
                    if isinstance(a, ast.Constant) and a.value is False and isinstance(b, ast.JoinedStr):
                        seg = ast.get_source_segment(src, b)
                        text = re.sub(r"^f", "", seg).strip("\"'")
                        items.append(("violation_message", text, {"group": group, "function": fn.name,
                                                                  "paper_term": term, "file": fname}))
    # plan fields
    for fld in ["Day", "Current City", "Transportation", "Breakfast", "Attraction", "Lunch", "Dinner", "Accommodation"]:
        items.append(("plan_field", fld, {"file": "agents/prompts.py (PLANNER_INSTRUCTION example format)"}))
    write_bank("travelplanner", "MIT (code)", "https://github.com/OSU-NLP-Group/TravelPlanner", rev, items)


# ---------------------------------------------------------------- TheAgentCompany
PORTS = {":3000": "rocketchat", ":8929": "gitlab", ":8092": "owncloud", ":8091": "plane"}


def theagentcompany():
    import yaml
    repo = os.path.join(SRC, "TheAgentCompany")
    rev = git_rev("TheAgentCompany")
    items = []
    for d in sorted(glob.glob(os.path.join(repo, "workspaces", "tasks", "*"))):
        name = os.path.basename(d)
        tm = os.path.join(d, "task.md")
        if name == "example" or not os.path.exists(tm):
            continue
        text = open(tm, encoding="utf-8").read().strip()
        deps = []
        dy = os.path.join(d, "dependencies.yml")
        if os.path.exists(dy):
            y = yaml.safe_load(open(dy, encoding="utf-8"))
            deps = sorted(set(y or []))
        mentioned = sorted({s for p, s in PORTS.items() if p in text} |
                           {s for s in ("rocketchat", "gitlab", "owncloud", "plane") if s in text.lower().replace(" ", "")})
        items.append(("task", text, {"task": name, "category": name.split("-")[0], "services": deps,
                                     "services_mentioned": mentioned, "has_npc_scenarios": os.path.exists(os.path.join(d, "scenarios.json")),
                                     "file": f"workspaces/tasks/{name}/task.md"}))
    write_bank("theagentcompany", "MIT", "https://github.com/TheAgentCompany/TheAgentCompany", rev, items)


# ---------------------------------------------------------------- BFCL
def bfcl():
    base = os.path.join(SRC, "gorilla", "berkeley-function-call-leaderboard", "bfcl_eval", "data")
    rev = git_rev("gorilla")
    rel = "berkeley-function-call-leaderboard/bfcl_eval/data/"
    items = []
    for f in sorted(glob.glob(os.path.join(base, "multi_turn_func_doc", "*.json"))):
        api = os.path.basename(f)[:-5]
        for line in open(f, encoding="utf-8"):
            if line.strip():
                fn = json.loads(line)
                items.append(("function", fn.get("description", ""), {
                    "name": fn["name"], "parameters": fn.get("parameters"), "response": fn.get("response"),
                    "category": "multi_turn_api", "api": api, "file": rel + "multi_turn_func_doc/" + os.path.basename(f)}))
    quotas = {"simple_python": 150, "multiple": 80, "parallel": 30, "parallel_multiple": 30,
              "simple_java": 30, "simple_javascript": 20, "irrelevance": 60}
    rng = random.Random(0)
    for cat, q in quotas.items():
        fn_pool = {}
        fp = os.path.join(base, f"BFCL_v4_{cat}.json")
        for line in open(fp, encoding="utf-8"):
            if not line.strip():
                continue
            ex = json.loads(line)
            for fn in ex["function"]:
                key = (fn["name"], fn.get("description", ""))
                fn_pool.setdefault(key, fn)
        keys = sorted(fn_pool)
        for k in rng.sample(keys, min(q, len(keys))):
            fn = fn_pool[k]
            items.append(("function", fn.get("description", ""), {
                "name": fn["name"], "parameters": fn.get("parameters"), "category": cat,
                "file": rel + f"BFCL_v4_{cat}.json"}))
        print(f"  bfcl {cat}: pool {len(keys)}, took {min(q, len(keys))}")
    write_bank("bfcl", "Apache-2.0", "https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard", rev, items)


if __name__ == "__main__":
    fns = {"alfworld": alfworld, "webarena": webarena, "osworld": osworld, "mind2web": mind2web,
           "webshop": webshop, "travelplanner": travelplanner, "theagentcompany": theagentcompany, "bfcl": bfcl}
    names = list(fns) if sys.argv[1:] == ["all"] else sys.argv[1:]
    for n in names:
        fns[n]()
