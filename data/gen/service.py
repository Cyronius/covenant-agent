"""Customer service under a written policy: task rows from tau2-bench.

  python -m data.gen.service --domain retail --out data/service_retail.jsonl
  python -m data.gen.service --domain airline --out data/service_airline.jsonl
  python -m data.gen.service --domain telecom --holdout \
      --out data/holdout/e_service_telecom.jsonl
  python -m data.gen.service --extract PATH/TO/tau2-bench   # re-vendor

Source: tau2-bench (github.com/sierra-research/tau2-bench, MIT), its
official TRAIN split only, vendored under data/borrowed/tau2/ with the
commit it came from (README.md there). Nothing from the test split is
copied; the test split is read once, at extraction, only to list the users
and records its tasks touch so the template draws below never use them.

Rows come in the task shape data/gen/episodes.py writes (build_task +
serialize_context + static effects), from two sources:

  tau2 task     each train task's expected write actions, mapped onto our
                tools and re-derived against our world: a task becomes a
                whole-task row (when its brief fits) and one row per write.
                Train tasks whose point is a refusal (airline mostly) are
                mapped by hand in AIRLINE_POLICY_TASKS.
  template      the same kinds of request drawn over other records of the
                tau2 database, so a few hundred rows exist per domain.

Around every permitted row the generator writes the variants that teach
the policy as a decision rather than an accident:

  forbidden     the same request in a world where the policy refuses it (the
                order was already delivered; the variant is out of stock;
                the refund names a card that did not pay; the booking is two
                weeks old). The reference is `ABORT UNSUPPORTED <constant>`
                and the generator confirms the permitted program really is
                refused there - by runtime/engines/service.js, with a
                PERMISSION_DENIED whose message names the rule - before it
                keeps the row.
  ask / act     a request missing one value the policy needs (why cancel;
                which card pays the difference): `ABORT NEEDS_INFO <field>`,
                then the same request with the answer appended, in
                data/gen/askact.py's shape.

The situation lives in the constants' descriptions (order status, balance,
availability, when a booking was made), the request is a one-line
first-person brief within the tiny planner's 128-token budget
(data/gen/brief_budget.check_brief), and every reference executes in the
sandbox before it is written.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.ir import TaskContext  # noqa: E402
from core.pipeline import build  # noqa: E402
from data.gen.brief_budget import brief_tokens, check_brief  # noqa: E402
from harness.abort_check import check_abort  # noqa: E402
from harness.authoring import resolve  # noqa: E402
from harness.context import build_context, serialize_context  # noqa: E402
from harness.run import run_sandbox  # noqa: E402
from harness.taskbuild import ReferenceError, build_task  # noqa: E402
from runtime.worlds import service as SW  # noqa: E402

GENERATOR_VERSION = "0.1.0"
LEVEL = 25
BORROWED = ROOT / "data" / "borrowed" / "tau2"
TAU2_COMMIT = "5bfa7e37b36656b37dc6d022156be6563c1007f3"
TAU2_URL = "https://github.com/sierra-research/tau2-bench"

# how an answer comes back (data/gen/askact.py's templates, same shape)
REPLIES = [
    'You asked {question} They said: "{answer}".',
    'You asked {question} The answer came back: "{answer}".',
    'You asked {question} They replied "{answer}".',
    '(You asked {question} Answer: "{answer}".)',
]


class Skip(Exception):
    """A draw or task that cannot become a row; the message says why."""


def dollars(cents: int) -> str:
    return f"${cents / 100:,.2f}"


def cents(x) -> int:
    return int(round(float(x) * 100))


# ================================================================ drafts

class Draft:
    """One row before symbols: a request, constants keyed by name, and a
    program whose constants are written `{key}`."""

    def __init__(self, world: str, request: str = ""):
        self.world = world
        self.request = request
        self.consts: Dict[str, dict] = {}
        self._by_value: Dict[tuple, str] = {}
        self.program = ""
        self.status = "ok"
        self.state: dict = {}
        self.tags: set = set()
        self.prov: dict = {}
        self.regs = 0

    def const(self, key: str, type_: str, value, desc: str,
              kind: str = "") -> str:
        """Declare a constant (once); returns its `{key}` placeholder. Two
        keys with the same type and value are one constant, as
        data.gen.programs.ConstAlloc has it."""
        vkey = (type_, json.dumps(value, sort_keys=True))
        if vkey in self._by_value and self._by_value[vkey] in self.consts:
            return "{" + self._by_value[vkey] + "}"
        key = re.sub(r"[\s{}]", "_", key)
        base, n = key, 2
        while key in self.consts:       # same key, another value
            key, n = f"{base}_{n}", n + 1
        c = {"type": type_, "value": value, "desc": desc}
        if kind:
            c["kind"] = kind
        self.consts[key] = c
        self._by_value[vkey] = key
        return "{" + key + "}"

    def reg(self) -> str:
        r = f"r{self.regs}"
        self.regs += 1
        if self.regs > 15:
            raise Skip("program needs more than 16 registers")
        return r

    def copy(self) -> "Draft":
        d = Draft(self.world, self.request)
        d.consts = copy.deepcopy(self.consts)
        d._by_value = dict(self._by_value)
        d.program = self.program
        d.status = self.status
        d.state = copy.deepcopy(self.state)
        d.tags = set(self.tags)
        d.prov = copy.deepcopy(self.prov)
        d.regs = self.regs
        return d


def _order(draft: Draft, rng: random.Random) -> Tuple[List[dict], Dict[str, str]]:
    """Shuffle the constants (a position must carry nothing) and map each
    key to its `$i`."""
    keys = list(draft.consts)
    rng.shuffle(keys)
    consts = [dict(draft.consts[k]) for k in keys]
    return consts, {k: f"${i}" for i, k in enumerate(keys)}


def _fill(program: str, index: Dict[str, str]) -> str:
    def sub(m):
        if m.group(1) not in index:
            raise Skip(f"program names undeclared constant {m.group(1)}")
        return index[m.group(1)]
    return re.sub(r"\{([^{}\s]+)\}", sub, program)


def probe(draft: Draft, program: Optional[str] = None) -> dict:
    """Run a program (the draft's own by default) against the draft's state
    and return the sandbox result - used to confirm the policy refuses the
    permitted program in a forbidden variant."""
    from runtime.worlds import get_world
    world = get_world(draft.world)
    consts, index = _order(draft, random.Random(0))
    ctx, sb = build_context(world, consts, random.Random(0))
    text = resolve(_fill(program or draft.program, index), ctx)
    res = build(text, ctx)
    if not res.compile_ok:
        return {"status": "static_error",
                "error": {"code": "STATIC",
                          "message": str(res.rendered_diagnostics()[:3])}}
    return run_sandbox({
        "js": res.js, "state": draft.state, "tools": sb["tools"],
        "fields": sb["fields"], "constants": sb["constants"],
        "now": world["now"], "approval": True, "error_injection": [],
        "initial_registers": {},
    })


def finish(draft: Draft, task_id: str, seed: int, *, symbols: str,
           enums: bool, kinds: bool, decoys: Optional[tuple] = None) -> dict:
    """The draft as a task row: symbols assigned, reference executed, input
    serialized. Raises Skip when the brief is over budget or the reference
    does not reach its expected status."""
    from runtime.worlds import get_world
    try:
        check_brief(draft.request)
    except ValueError as e:
        raise Skip(f"brief: {e}")
    world = get_world(draft.world)
    decoy_names: List[str] = []
    if decoys:
        from harness.decoys import decoy_world
        # only the siblings authored in runtime/worlds/service.py: the fixed
        # bank's style is a tell (results/R10.md section 8)
        authored = {t["name"] for t in world["tools"]
                    if t.get("authored_decoys")}
        world, decoy_names = decoy_world(world, random.Random(seed ^ 0xDEC0),
                                         per_tool=decoys, nonsense=0.0,
                                         only=authored)
    rng = random.Random(seed)
    consts, index = _order(draft, rng)
    if not kinds:
        consts = [{k: v for k, v in c.items() if k != "kind"} for c in consts]
    ctx, sandbox_ctx = build_context(world, consts, random.Random(seed ^ 0x5E4),
                                     symbols=symbols, enums=enums)
    try:
        segment = resolve(_fill(draft.program, index), ctx)
    except Exception as e:  # ResolveError
        raise Skip(f"resolve: {e}")
    if draft.status == "aborted":
        m = re.match(r"\s*ABORT (\w+)((?: \S+)*)", segment)
        if m:
            refs = m.group(2).split()
            msg = check_abort(ctx, draft.state, m.group(1), refs)
            if msg:
                raise Skip(f"unfounded abort: {msg}")
    try:
        task = build_task(
            task_id=task_id, level=LEVEL, world_name=draft.world,
            request=draft.request, constants=consts, segments=[segment],
            seed=seed, expected_status=draft.status,
            state=copy.deepcopy(draft.state),
            tags=sorted(draft.tags | {"family:service",
                                      f"world:{draft.world}"}
                        | ({"abort"} if draft.status == "aborted" else set())),
            provenance=dict(draft.prov, generator_version=GENERATOR_VERSION,
                            source=TAU2_URL, source_commit=TAU2_COMMIT,
                            licence="MIT", seed=seed),
            prebuilt=(ctx, sandbox_ctx))
    except ReferenceError as e:
        raise Skip(f"reference: {e}")
    if decoy_names:
        # the decoys are not in the registered world: the row carries its
        # own sandbox payload, as data.gen's decoyed rows do
        task["sandbox"] = sandbox_ctx
        task["tags"] = sorted(set(task["tags"]) | {"decoyed"})
        task["provenance"]["decoys"] = decoy_names
    task["input_text"] = serialize_context(draft.request, ctx)
    res = build(segment, TaskContext.from_json(task["context"]))
    task["effects"] = res.static_effects
    task["spec_version"] = "0.4.0"
    task["symbols"] = symbols
    return task


def refused(draft: Draft) -> str:
    """The rule the engine names when it refuses the draft's program, or ''
    when the program runs (or fails for some other reason)."""
    sres = probe(draft)
    err = sres.get("error") or {}
    if sres.get("status") == "error" and err.get("code") == "PERMISSION_DENIED":
        return str(err.get("message", ""))
    return ""


def as_abort(draft: Draft, ref_key: str, reason: str = "UNSUPPORTED") -> Draft:
    d = draft.copy()
    d.program = f"ABORT {reason} {{{ref_key}}}\n" if ref_key else \
        f"ABORT {reason}\n"
    d.status = "aborted"
    return d


# ============================================================ extraction

_UID = re.compile(r"\b[a-z]+_[a-z]+_\d{3,5}\b")


def extract(src: Path) -> None:
    """Vendor the train split, the policies and the database records the
    generator draws on from a tau2-bench checkout into data/borrowed/tau2/.
    The test split is scanned once for the user/order/reservation ids its
    tasks touch; those users are left out of the template pool (and listed
    in denylist.json), and no test task is copied."""
    src = Path(src)
    dom = src / "data" / "tau2" / "domains"
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    rng = random.Random(20260929)
    BORROWED.mkdir(parents=True, exist_ok=True)
    (BORROWED / "LICENSE").write_text((src / "LICENSE").read_text(
        encoding="utf-8"), encoding="utf-8")
    report = {"source": TAU2_URL, "commit": commit, "domains": {}}

    for d in ("retail", "airline", "telecom"):
        out = BORROWED / d
        out.mkdir(exist_ok=True)
        split = json.loads((dom / d / "split_tasks.json").read_text())
        tasks = json.loads((dom / d / "tasks.json").read_text(
            encoding="utf-8"))
        by_id = {t["id"]: t for t in tasks}
        train = [by_id[i] for i in split["train"]]
        (out / "tasks_train.json").write_text(
            json.dumps(train, indent=1, ensure_ascii=False), encoding="utf-8")
        test_text = " ".join(json.dumps(by_id[i]) for i in split["test"])
        train_text = " ".join(json.dumps(t) for t in train)
        policy = "main_policy.md" if d == "telecom" else "policy.md"
        (out / "policy.md").write_text((dom / d / policy).read_text(
            encoding="utf-8"), encoding="utf-8")
        rep = {"train_tasks": len(train), "test_tasks_copied": 0,
               "files": [f"data/tau2/domains/{d}/split_tasks.json (train ids)",
                         f"data/tau2/domains/{d}/tasks.json (train tasks)",
                         f"data/tau2/domains/{d}/{policy}"]}
        if d == "telecom":
            (out / "db.toml").write_text((dom / d / "db.toml").read_text(
                encoding="utf-8"), encoding="utf-8")
            rep["files"].append("data/tau2/domains/telecom/db.toml")
            report["domains"][d] = rep
            continue
        db = json.loads((dom / d / "db.json").read_text(encoding="utf-8"))
        users = db["users"]
        deny = sorted({u for u in _UID.findall(test_text) if u in users})
        train_users = sorted({u for u in _UID.findall(train_text)
                              if u in users})
        if d == "retail":
            # train tasks name users by name + zip as often as by id
            for t in train:
                for a in t["evaluation_criteria"]["actions"] or []:
                    oid = a["arguments"].get("order_id")
                    if oid in db["orders"]:
                        train_users.append(db["orders"][oid]["user_id"])
            for t in split["test"]:
                for a in by_id[t]["evaluation_criteria"]["actions"] or []:
                    oid = a["arguments"].get("order_id")
                    if oid in db["orders"]:
                        deny.append(db["orders"][oid]["user_id"])
        else:
            for t in train:
                for a in t["evaluation_criteria"]["actions"] or []:
                    rid = a["arguments"].get("reservation_id")
                    if rid in db["reservations"]:
                        train_users.append(db["reservations"][rid]["user_id"])
            for t in split["test"]:
                for a in by_id[t]["evaluation_criteria"]["actions"] or []:
                    rid = a["arguments"].get("reservation_id")
                    if rid in db["reservations"]:
                        deny.append(db["reservations"][rid]["user_id"])
        deny = sorted(set(deny))
        train_users = sorted(set(train_users))
        pool = sorted(set(users) - set(deny) - set(train_users))
        sampled = sorted(rng.sample(pool, min(220, len(pool))))
        keep = set(train_users) | set(sampled)
        ext: dict = {"users": {u: users[u] for u in sorted(keep)},
                     "template_users": sampled, "train_users": train_users}
        if d == "retail":
            ext["products"] = db["products"]
            ext["orders"] = {o: v for o, v in db["orders"].items()
                             if v["user_id"] in keep}
        else:
            ext["reservations"] = {r: v for r, v in db["reservations"].items()
                                   if v["user_id"] in keep}
            flights = {}
            for num, f in db["flights"].items():
                dates = {}
                for date, info in f["dates"].items():
                    dates[date] = {"status": info["status"],
                                   "prices": info.get("prices")}
                flights[num] = {k: f[k] for k in (
                    "origin", "destination", "flight_number",
                    "scheduled_departure_time_est",
                    "scheduled_arrival_time_est")}
                flights[num]["dates"] = dates
            ext["flights"] = flights
        (out / "db_extract.json").write_text(
            json.dumps(ext, separators=(",", ":")), encoding="utf-8")
        (out / "denylist.json").write_text(json.dumps(
            {"comment": "users touched by tau2's TEST split; never drawn",
             "users": deny}, indent=1), encoding="utf-8")
        rep["files"].append(f"data/tau2/domains/{d}/db.json (subset)")
        rep.update(test_users_denied=len(deny), train_users=len(train_users),
                   template_users=len(sampled))
        report["domains"][d] = rep
    (BORROWED / "SOURCE.json").write_text(json.dumps(report, indent=1),
                                          encoding="utf-8")
    print(json.dumps(report, indent=1))


def load(domain: str, name: str):
    p = BORROWED / domain / name
    if not p.exists():
        raise SystemExit(f"{p} missing: run python -m data.gen.service "
                         f"--extract PATH/TO/tau2-bench first")
    return json.loads(p.read_text(encoding="utf-8"))


# ================================================================ retail

PM_KIND = {"gift_card": "gift_card", "credit_card": "credit_card",
           "paypal": "paypal"}
REASON_PHRASE = {"no longer needed": ["I no longer need it",
                                      "it's no longer needed",
                                      "I don't need it anymore"],
                 "ordered by mistake": ["I ordered it by mistake",
                                        "it was ordered by mistake",
                                        "I placed it by mistake"]}
BAD_REASONS = ["I found it cheaper elsewhere", "the delivery is too slow",
               "I want to reorder it with a coupon",
               "the seller took too long to ship"]
_PAY_CLAUSE = re.compile(r"(, paying any difference with my [^.;]*|, with the "
                         r"difference on my [^.;]*|; use my [^.;]* for the "
                         r"difference)\.")


def _twin_suffix(pm: dict, state: Optional[dict],
                 same_balance: bool = False) -> str:
    """' ending 6644' when the customer holds another method of this kind
    (and, for certificates, of this amount)."""
    if state is None:
        return ""
    twins = [p for p in state["entities"]["payment_method"]
             if p["user"] == pm["user"] and p["kind"] == pm["kind"]
             and p["id"] != pm["id"]
             and (not same_balance or p["balance"] == pm["balance"])]
    return f" ending {pm['id'][-4:]}" if twins else ""


def options(opts: dict) -> str:
    return ", ".join(f"{k}: {v}" for k, v in opts.items())


def parse_options(text: str) -> dict:
    out = {}
    for part in text.split(", "):
        if ": " in part:
            k, v = part.split(": ", 1)
            out[k] = v
    return out


def addr_text(a: dict) -> str:
    line = a["address1"] + (f", {a['address2']}" if a.get("address2") else "")
    return f"{line}, {a['city']}, {a['state']} {a['zip']}"


def join_and(xs: List[str]) -> str:
    if len(xs) <= 1:
        return "".join(xs)
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def _retail_task_10(dom, run, rng, prov):
    """Refund each of two orders to the other order's payment method: the
    policy allows only the original method or a gift card."""
    uid = "mia_garcia_4516"
    state = dom.base_state(uid, rng)
    pairs = [("#W5490111", "paypal_9497703"),
             ("#W7387996", "credit_card_3124723")]
    for oid, pm in pairs:
        items = [it["id"] for it in dom.items_of(state, oid)]
        d = dom.build(uid, [("return", oid, items, pm)], rng, "email",
                      copy.deepcopy(state), refund_mode="named")
        run.abort_only(d, f"pm_{pm}", prov, expect_refusal=True)


def _retail_task_50(dom, run, rng, prov):
    """Undo a cancellation: no tool reinstates a cancelled order."""
    uid = next((u for u, v in dom.db["users"].items()
                if v["name"] == {"first_name": "Chen", "last_name": "Smith"}
                and v["address"]["zip"] == "32278"), None)
    if uid is None:
        run.dropped["tau2 task: user not in extract"] += 1
        return
    state = dom.base_state(uid, rng)
    gone = [o for o in state["entities"]["order"]
            if o["user"] == uid and o["status"] == "cancelled"]
    if not gone:
        run.dropped["tau2 task: no cancelled order"] += 1
        return
    o = gone[0]
    d = Draft(dom.world)
    d.state = state
    u = dom.rec(state, "user", uid)
    who, _ = dom.identity(d, u, "namezip", rng)
    dom.add_pms(d, uid)
    ref = dom.add_order(d, o["id"])
    d.request = (f"{who} Please undo the cancellation of my order {o['id']} "
                 f"and ship everything in it as soon as possible.")
    d.program = f"ABORT UNSUPPORTED {ref}\n"
    d.status = "aborted"
    d.prov.update(prov, variant="forbidden_tau2",
                  policy_reason="no tool reinstates a cancelled order")
    d.tags |= {"service", "policy_forbidden", "no_tool"}
    run.emit(d, "forbid")


RETAIL_POLICY_TASKS = {"10": _retail_task_10, "50": _retail_task_50}


class Retail:
    """tau2's retail database as our state, and the drafts built on it."""

    world = "service_retail"

    def __init__(self):
        self.db = load("retail", "db_extract.json")
        self.tasks = load("retail", "tasks_train.json")
        self.products = self.db["products"]

    # ------------------------------------------------------------ state
    @staticmethod
    def pm_record(uid: str, pm: dict) -> dict:
        return {"id": pm["id"], "user": uid, "kind": PM_KIND[pm["source"]],
                "balance": cents(pm.get("balance", 0)),
                "last_four": pm.get("last_four", "")}

    @staticmethod
    def item_ids(order: dict) -> List[str]:
        seen: Counter = Counter()
        out = []
        for it in order["items"]:
            seen[it["item_id"]] += 1
            n = seen[it["item_id"]]
            out.append(f"{order['order_id']}/{it['item_id']}"
                       + (f"#{n}" if n > 1 else ""))
        return out

    def state_for(self, uids: List[str]) -> dict:
        ents = {k: [] for k in SW.RETAIL["default_state"]["entities"]}
        products = set()
        for uid in uids:
            u = self.db["users"][uid]
            a = u["address"]
            ents["user"].append({
                "id": uid, "first_name": u["name"]["first_name"],
                "last_name": u["name"]["last_name"], "email": u["email"],
                "address1": a["address1"], "address2": a["address2"],
                "city": a["city"], "state": a["state"], "zip": a["zip"]})
            for pm in u["payment_methods"].values():
                ents["payment_method"].append(self.pm_record(uid, pm))
            for oid in u["orders"]:
                o = self.db["orders"].get(oid)
                if o is None:
                    continue
                a = o["address"]
                ents["order"].append({
                    "id": oid, "user": uid, "status": o["status"],
                    "address1": a["address1"], "address2": a["address2"],
                    "city": a["city"], "state": a["state"], "zip": a["zip"],
                    "payment_method":
                        o["payment_history"][0]["payment_method_id"],
                    "total": sum(cents(i["price"]) for i in o["items"]),
                    "cancel_reason": None,
                    "locked": o["status"] in ("pending (item modified)",
                                              "exchange requested",
                                              "return requested")})
                for rid, it in zip(self.item_ids(o), o["items"]):
                    products.add(it["product_id"])
                    ents["order_item"].append({
                        "id": rid, "order": oid, "product": it["product_id"],
                        "variant": it["item_id"], "name": it["name"],
                        "price": cents(it["price"]), "status": "ordered",
                        "new_variant": None, "payment_method": None})
        for pid in sorted(products):
            p = self.products[pid]
            ents["product"].append({"id": pid, "name": p["name"]})
            for vid, v in p["variants"].items():
                ents["variant"].append({
                    "id": vid, "product": pid,
                    "options": options(v["options"]),
                    "available": bool(v["available"]),
                    "price": cents(v["price"])})
        return {"entities": ents, "outbox": [], "payments": []}

    # ------------------------------------------------------------ lookups
    @staticmethod
    def rec(state: dict, entity: str, id_: str) -> dict:
        for r in state["entities"][entity]:
            if r["id"] == id_:
                return r
        raise Skip(f"{entity} {id_} not in state")

    @staticmethod
    def items_of(state: dict, oid: str) -> List[dict]:
        return [r for r in state["entities"]["order_item"]
                if r["order"] == oid]

    @staticmethod
    def pms_of(state: dict, uid: str) -> List[dict]:
        return [r for r in state["entities"]["payment_method"]
                if r["user"] == uid]

    @staticmethod
    def variants_of(state: dict, pid: str) -> List[dict]:
        return [r for r in state["entities"]["variant"] if r["product"] == pid]

    @staticmethod
    def product_name(state: dict, pid: str) -> str:
        return next(p["name"] for p in state["entities"]["product"]
                    if p["id"] == pid)

    # ------------------------------------------------------------ phrases
    @staticmethod
    def pm_short(pm: dict, state: Optional[dict] = None) -> str:
        """How the customer names a payment method: by kind, and by the end
        of its id when they hold two of that kind."""
        if pm["kind"] == "gift_card":
            return "gift card" + _twin_suffix(pm, state)
        if pm["kind"] == "paypal":
            return "PayPal"
        return f"credit card ending {pm['last_four']}"

    @staticmethod
    def pm_desc(pm: dict) -> str:
        if pm["kind"] == "gift_card":
            return f"gift card {pm['id']}, balance {dollars(pm['balance'])}"
        if pm["kind"] == "paypal":
            return f"PayPal account {pm['id']}"
        return f"credit card {pm['id']} ending {pm['last_four']}"

    def order_desc(self, state: dict, o: dict) -> str:
        n = len(self.items_of(state, o["id"]))
        pay = self.rec(state, "payment_method", o["payment_method"])
        return (f"order {o['id']}: {o['status']}, {n} item"
                f"{'s' if n != 1 else ''}, {dollars(o['total'])}, paid with "
                f"{self.pm_short(pay, state)}")

    @staticmethod
    def item_desc(it: dict, v: dict) -> str:
        return (f"the {it['name']} ({v['options']}) in order {it['order']}, "
                f"{dollars(it['price'])}")

    @staticmethod
    def variant_desc(name: str, v: dict) -> str:
        return (f"{name} option {v['options']} (item {v['id']}), "
                f"{dollars(v['price'])}, "
                f"{'available' if v['available'] else 'out of stock'}")

    def item_phrase(self, state: dict, it: dict) -> str:
        """'the Water Bottle', with options when the order holds two."""
        twins = [x for x in self.items_of(state, it["order"])
                 if x["name"] == it["name"]]
        if len(twins) > 1:
            v = self.rec(state, "variant", it["variant"])
            return f"the {it['name']} ({v['options']})"
        return f"the {it['name']}"

    @staticmethod
    def diff_phrase(old: dict, new: dict) -> str:
        a, b = parse_options(old["options"]), parse_options(new["options"])
        diff = [f"{k} {b[k]}" for k in b if a.get(k) != b[k]]
        return ", ".join(diff) or new["options"]

    # ------------------------------------------------------------ drafts
    def identity(self, d: Draft, u: dict, mode: str, rng) -> Tuple[str, str]:
        """(sentence, how the program reaches the user id: a constant, or a
        lookup CALL whose result register is written {USER})."""
        name = f"{u['first_name']} {u['last_name']}"
        if mode == "email":
            e = d.const("email", "STR", u["email"],
                        f"your email address, verbatim: {u['email']}", "name")
            return (rng.choice([f"I'm {name} ({u['email']}).",
                                f"Hi, {name} here, email {u['email']}."]),
                    f"CALL @find_user_id_by_email {e} -> {{USER}}\n")
        if mode == "uid":
            ref = d.const("user", "ID:user", u["id"], f"{name}'s user id")
            return (rng.choice([f"I'm {name}, user id {u['id']}.",
                                f"This is {name} ({u['id']})."]), ref)
        f = d.const("first", "STR", u["first_name"],
                    f"your first name: {u['first_name']}", "name")
        l_ = d.const("last", "STR", u["last_name"],
                     f"your last name: {u['last_name']}", "name")
        z = d.const("zip", "STR", u["zip"], f"your zip code: {u['zip']}",
                    "name")
        return (rng.choice([f"I'm {name}, zip {u['zip']}.",
                            f"{name} here, zip code {u['zip']}."]),
                f"CALL @find_user_id_by_name_zip {f} {l_} {z} -> {{USER}}\n")

    def add_pms(self, d: Draft, uid: str) -> None:
        for pm in self.pms_of(d.state, uid):
            d.const(f"pm_{pm['id']}", "ID:payment_method", pm["id"],
                    self.pm_desc(pm))

    def add_order(self, d: Draft, oid: str, with_items=True) -> str:
        o = self.rec(d.state, "order", oid)
        ref = d.const(f"o_{oid}", "ID:order", oid,
                      self.order_desc(d.state, o))
        if with_items:
            for it in self.items_of(d.state, oid):
                v = self.rec(d.state, "variant", it["variant"])
                d.const(f"it_{it['id']}", "ID:order_item", it["id"],
                        self.item_desc(it, v))
        return ref

    def add_variants(self, d: Draft, it: dict, target: str, rng,
                     n_extra: int = 2) -> str:
        """The target variant plus a few siblings of the same product."""
        v = self.rec(d.state, "variant", target)
        name = self.product_name(d.state, v["product"])
        ref = d.const(f"v_{target}", "ID:variant", target,
                      self.variant_desc(name, v))
        sibs = [x for x in self.variants_of(d.state, v["product"])
                if x["id"] not in (target, it["variant"])]
        for x in rng.sample(sibs, min(n_extra, len(sibs))):
            d.const(f"v_{x['id']}", "ID:variant", x["id"],
                    self.variant_desc(name, x))
        return ref

    def addr_consts(self, d: Draft, key: str, addr: dict) -> List[str]:
        labels = {"address1": "street address",
                  "address2": "second address line", "city": "city",
                  "state": "state", "zip": "zip code"}
        refs = []
        for f in ("address1", "address2", "city", "state", "zip"):
            val = addr.get(f, "")
            desc = (f"new {labels[f]}: {val}" if val else
                    f"new {labels[f]}: none (empty)")
            refs.append(d.const(f"{key}_{f}", "STR", val, desc, "text"))
        return refs

    def distractor_users(self, uid: str, rng, n: int) -> List[str]:
        pool = [u for u in self.db["template_users"] if u != uid]
        return rng.sample(pool, n)

    def base_state(self, uid: str, rng) -> dict:
        return self.state_for([uid] + self.distractor_users(uid, rng, 1))

    def build(self, uid: str, actions: List[tuple], rng: random.Random,
              ident: str, state: dict,
              refund_mode: Optional[str] = None) -> Draft:
        """A draft for one customer's writes. actions:
          ('cancel', oid, reason)            ('address', oid, addr)
          ('user_address', addr)             ('payment', oid, pm)
          ('modify'|'exchange', oid, [(item, variant)], pm)
          ('return', oid, [item], pm)"""
        d = Draft(self.world)
        d.state = state
        u = self.rec(d.state, "user", uid)
        who, user_ref = self.identity(d, u, ident, rng)
        parts, lines = [], []
        user_reg = None
        self.add_pms(d, uid)
        for act in actions:
            kind, rest = act[0], act[1:]
            if kind == "cancel":
                oid, reason = rest
                o_ref = self.add_order(d, oid, with_items=False)
                r = d.const("reason", "STR", reason,
                            f"cancellation reason '{reason}'",
                            "enum:order.cancel_reason"
                            if reason in SW.RETAIL_CANCEL_REASONS else "text")
                for x in SW.RETAIL_CANCEL_REASONS:
                    d.const(f"reason_{x[:4]}", "STR", x,
                            f"cancellation reason '{x}'",
                            "enum:order.cancel_reason")
                phrase = (rng.choice(REASON_PHRASE[reason])
                          if reason in REASON_PHRASE else reason)
                parts.append(rng.choice([
                    f"Please cancel order {oid}; {phrase}.",
                    f"Cancel my order {oid}, {phrase}.",
                    f"I want order {oid} cancelled because {phrase}."]))
                lines.append(f"CALL @cancel_pending_order {o_ref} {r}\n")
            elif kind == "address":
                oid, addr = rest
                o_ref = self.add_order(d, oid, with_items=False)
                refs = self.addr_consts(d, "new", addr)
                parts.append(rng.choice([
                    f"Ship order {oid} to {addr_text(addr)} instead.",
                    f"Change the shipping address of order {oid} to "
                    f"{addr_text(addr)}."]))
                lines.append(f"CALL @modify_pending_order_address {o_ref} "
                             + " ".join(refs) + "\n")
            elif kind == "user_address":
                (addr,) = rest
                refs = self.addr_consts(d, "new", addr)
                if user_reg is None:
                    if user_ref.startswith("CALL"):
                        user_reg = d.reg()
                        lines.append(user_ref.replace("{USER}", user_reg))
                    else:
                        user_reg = user_ref
                parts.append(rng.choice([
                    f"Update my default address to {addr_text(addr)}.",
                    f"My default address should be {addr_text(addr)}."]))
                lines.append(f"CALL @modify_user_address {user_reg} "
                             + " ".join(refs) + "\n")
            elif kind == "payment":
                oid, pmid = rest
                o_ref = self.add_order(d, oid, with_items=False)
                pm = self.rec(d.state, "payment_method", pmid)
                parts.append(rng.choice([
                    f"Pay for order {oid} with my {self.pm_short(pm, d.state)} "
                    f"instead.",
                    f"Switch the payment on order {oid} to my "
                    f"{self.pm_short(pm, d.state)}."]))
                lines.append(f"CALL @modify_pending_order_payment {o_ref} "
                             f"{{pm_{pmid}}}\n")
            elif kind in ("modify", "exchange"):
                oid, pairs, pmid = rest
                self.add_order(d, oid)
                pm = self.rec(d.state, "payment_method", pmid)
                swaps = []
                tool = ("modify_pending_order_items" if kind == "modify"
                        else "exchange_delivered_order_items")
                for item_id, vid in pairs:
                    it = self.rec(d.state, "order_item", item_id)
                    old = self.rec(d.state, "variant", it["variant"])
                    v_ref = self.add_variants(d, it, vid, rng)
                    new = self.rec(d.state, "variant", vid)
                    if new["product"] != it["product"]:
                        pname = self.product_name(d.state, new["product"])
                        swaps.append(f"{self.item_phrase(d.state, it)} for "
                                     f"the {pname} ({new['options']})")
                    else:
                        swaps.append(f"{self.item_phrase(d.state, it)} for "
                                     f"the {self.diff_phrase(old, new)} one")
                    lines.append(f"CALL @{tool} {{it_{item_id}}} {v_ref} "
                                 f"{{pm_{pmid}}}\n")
                what = join_and(swaps)
                pay = rng.choice([
                    f", paying any difference with my {self.pm_short(pm, d.state)}",
                    f", with the difference on my {self.pm_short(pm, d.state)}",
                    f"; use my {self.pm_short(pm, d.state)} for the difference"])
                if kind == "modify":
                    parts.append(rng.choice([
                        f"In my pending order {oid}, change {what}{pay}.",
                        f"Before order {oid} ships, switch {what}{pay}."]))
                else:
                    parts.append(rng.choice([
                        f"Exchange {what} from order {oid}{pay}.",
                        f"From my delivered order {oid}, swap {what}{pay}."]))
            elif kind == "return":
                oid, item_ids, pmid = rest
                o_ref = self.add_order(d, oid)
                o = self.rec(d.state, "order", oid)
                pm = self.rec(d.state, "payment_method", pmid)
                names = join_and([self.item_phrase(
                    d.state, self.rec(d.state, "order_item", i))
                    for i in item_ids])
                mode = refund_mode or (
                    "original" if pmid == o["payment_method"]
                    and rng.random() < 0.5 else "named")
                if mode == "original" and pmid == o["payment_method"]:
                    r = d.reg()
                    lines.append(f"CALL @get_order_details {o_ref} -> {r}\n")
                    pay_ref = f"{r}.@order.payment_method"
                    refund = rng.choice(["to the original payment method",
                                         "to whatever I paid with",
                                         "the way I paid"])
                else:
                    pay_ref = f"{{pm_{pmid}}}"
                    refund = f"to my {self.pm_short(pm, d.state)}"
                for i in item_ids:
                    lines.append(f"CALL @return_delivered_order_items "
                                 f"{{it_{i}}} {pay_ref}\n")
                parts.append(rng.choice([
                    f"Return {names} from order {oid} and refund {refund}.",
                    f"I'd like to return {names} from order {oid}, refunded "
                    f"{refund}."]))
            else:
                raise Skip(f"unknown retail action {kind}")
        # a distractor order of the same customer
        others = [o for o in d.state["entities"]["order"]
                  if o["user"] == uid and f"o_{o['id']}" not in d.consts]
        for o in rng.sample(others, min(1, len(others))):
            d.const(f"o_{o['id']}", "ID:order", o["id"],
                    self.order_desc(d.state, o))
        d.request = " ".join([who] + parts)
        d.program = "".join(lines) + "STOP\n"
        d.prov["actions"] = [a[0] for a in actions]
        return d

    NATURAL_REFUSALS = False

    def base_state_for(self, uid: str, actions, rng) -> dict:
        return self.base_state(uid, rng)

    def natural_ref(self, act, msg) -> str:
        return f"o_{act[1]}"

    def tau2_task(self, run: "Run", task: dict, rng: random.Random) -> None:
        prov = {"source_kind": "tau2", "tau2_task": task["id"],
                "file": "data/tau2/domains/retail/tasks.json (train)",
                "recipe": "tau2_task", "teacher": "tau2-expected-actions"}
        if task["id"] in RETAIL_POLICY_TASKS:
            RETAIL_POLICY_TASKS[task["id"]](self, run, rng, prov)
            return
        try:
            uid, acts = self.task_actions(task)
        except Skip as e:
            run.dropped[f"tau2 task: {_reason(str(e))}"] += 1
            return
        run.group(uid, acts, self.ident_mode(task, uid), rng, prov,
                  self.base_state(uid, rng))

    # ------------------------------------------------------------ tau2 tasks
    @staticmethod
    def ident_mode(task: dict, uid: str) -> str:
        known = task["user_scenario"]["instructions"].get("known_info") or ""
        if "@" in known:
            return "email"
        if uid in known:
            return "uid"
        return "namezip"

    def task_actions(self, task: dict) -> Tuple[str, List[tuple]]:
        """tau2's expected writes as our actions (item ids mapped to our
        order_item ids)."""
        acts = []
        uid = None
        used: set = set()
        for a in task["evaluation_criteria"]["actions"] or []:
            n, args = a["name"], a["arguments"]
            if n.startswith(("get_", "find_", "list_", "calc")):
                continue
            if n == "transfer_to_human_agents":
                raise Skip("transfer_to_human_agents (a refusal whose "
                           "scenario is not mapped)")
            oid = args.get("order_id")
            o = None
            if oid:
                o = self.db["orders"].get(oid)
                if o is None:
                    raise Skip(f"order {oid} not in extract")
                uid = o["user_id"]
            if n == "cancel_pending_order":
                acts.append(("cancel", oid, args["reason"]))
            elif n == "modify_pending_order_address":
                acts.append(("address", oid, {k: args[k] for k in (
                    "address1", "address2", "city", "state", "zip")}))
            elif n == "modify_user_address":
                uid = args["user_id"]
                acts.append(("user_address", {k: args[k] for k in (
                    "address1", "address2", "city", "state", "zip")}))
            elif n == "modify_pending_order_payment":
                acts.append(("payment", oid, args["payment_method_id"]))
            elif n in ("modify_pending_order_items",
                       "exchange_delivered_order_items",
                       "return_delivered_order_items"):
                rids = self.item_ids(o)
                mapped = []
                for iid in args["item_ids"]:
                    rid = next((r for r, it in zip(rids, o["items"])
                                if it["item_id"] == iid and r not in used),
                               None)
                    if rid is None:
                        raise Skip(f"item {iid} not in order {oid}")
                    used.add(rid)
                    mapped.append(rid)
                pm = args["payment_method_id"]
                if n == "return_delivered_order_items":
                    acts.append(("return", oid, mapped, pm))
                else:
                    kind = "modify" if n.startswith("modify") else "exchange"
                    acts.append((kind, oid, list(zip(
                        mapped, args["new_item_ids"])), pm))
            else:
                raise Skip(f"unmapped tau2 action {n}")
        if not acts:
            raise Skip("no write actions (an information or refusal task)")
        return uid, acts

    # ------------------------------------------------------------ templates
    def template(self, rng: random.Random) -> Tuple[str, List[tuple], str]:
        """One DB-drawn request of a kind the tau2 train tasks ask for."""
        for _ in range(50):
            uid = rng.choice(self.db["template_users"])
            u = self.db["users"][uid]
            orders = [self.db["orders"][o] for o in u["orders"]
                      if o in self.db["orders"]]
            kind = rng.choices(
                ["cancel", "address", "user_address", "payment", "modify",
                 "exchange", "return"], [3, 2, 1, 1, 3, 3, 4])[0]
            want = {"cancel": "pending", "address": "pending",
                    "payment": "pending", "modify": "pending",
                    "exchange": "delivered", "return": "delivered"}.get(kind)
            pool = [o for o in orders if o["status"] == want] if want else []
            if want and not pool:
                continue
            o = rng.choice(pool) if pool else None
            pms = list(u["payment_methods"].values())
            ident = rng.choice(["email", "namezip", "uid"])
            if kind == "cancel":
                return uid, [("cancel", o["order_id"], rng.choice(
                    SW.RETAIL_CANCEL_REASONS))], ident
            if kind in ("address", "user_address"):
                other = self.db["users"][rng.choice(self.db["template_users"])]
                addr = {k: other["address"][k] for k in (
                    "address1", "address2", "city", "state", "zip")}
                if kind == "address":
                    return uid, [("address", o["order_id"], addr)], ident
                return uid, [("user_address", addr)], ident
            if kind == "payment":
                cur = o["payment_history"][0]["payment_method_id"]
                total = sum(cents(i["price"]) for i in o["items"])
                alt = [p for p in pms if p["id"] != cur and (
                    p["source"] != "gift_card"
                    or cents(p.get("balance", 0)) >= total)]
                if not alt or len(o["payment_history"]) != 1:
                    continue
                return uid, [("payment", o["order_id"],
                              rng.choice(alt)["id"])], ident
            rids = self.item_ids(o)
            idx = list(range(len(rids)))
            if kind == "return":
                k = rng.choice([1, 1, 2])
                chosen = sorted(rng.sample(idx, min(k, len(idx))))
                cur = o["payment_history"][0]["payment_method_id"]
                ok = [p["id"] for p in pms if p["id"] == cur
                      or p["source"] == "gift_card"]
                return uid, [("return", o["order_id"],
                              [rids[i] for i in chosen], rng.choice(ok))], ident
            pairs = []
            for i in rng.sample(idx, min(rng.choice([1, 1, 2]), len(idx))):
                it = o["items"][i]
                sib = [vid for vid, v in
                       self.products[it["product_id"]]["variants"].items()
                       if v["available"] and vid != it["item_id"]]
                if sib:
                    pairs.append((rids[i], rng.choice(sib)))
            if not pairs:
                continue
            pay = rng.choice(pms)["id"]
            return uid, [(kind, o["order_id"], pairs, pay)], ident
        raise Skip("no template draw")

    # ------------------------------------------------------------ variants
    def forbidden(self, uid: str, act: tuple, ident: str,
                  rng: random.Random, state: dict):
        """A variant the policy refuses: (would-be draft, the constant key the
        refusal is about, the perturbation) or None."""
        kind = act[0]
        options_ = []
        if kind in ("cancel", "address", "payment", "modify"):
            options_.append("status")
        if kind in ("exchange", "return"):
            options_ += ["status", "once"]
        if kind in ("modify", "exchange"):
            options_ += ["unavailable", "other_product", "gift_balance"]
        if kind == "return":
            options_ += ["refund_method", "refund_method"]
        if kind == "cancel":
            options_ += ["bad_reason", "bad_reason"]
        if kind == "payment":
            options_.append("gift_balance")
        rng.shuffle(options_)
        for how in options_:
            try:
                out = self._perturb(how, uid, act, ident, rng,
                                    copy.deepcopy(state))
            except Skip:
                continue
            if out is not None:
                return out
        return None

    def _perturb(self, how, uid, act, ident, rng, st):
        kind = act[0]
        oid = act[1] if kind != "user_address" else None
        o = self.rec(st, "order", oid) if oid else None
        refund_mode = None
        new_act = act
        if how == "status":
            if kind in ("exchange", "return"):
                o["status"] = rng.choice(["pending", "processed"])
            elif kind == "cancel":
                o["status"] = rng.choice(["processed", "delivered",
                                          "pending (item modified)"])
            else:
                o["status"] = rng.choice(["processed", "delivered"])
            ref = f"o_{oid}"
        elif how == "once":
            o["status"] = ("return requested" if kind == "exchange"
                           else "exchange requested")
            o["locked"] = True
            ref = f"o_{oid}"
        elif how == "unavailable":
            item_id, vid = act[2][0]
            self.rec(st, "variant", vid)["available"] = False
            ref = f"v_{vid}"
            new_act = (kind, oid, act[2][:1], act[3])
        elif how == "other_product":
            item_id, _ = act[2][0]
            it = self.rec(st, "order_item", item_id)
            pool = [v for v in st["entities"]["variant"]
                    if v["product"] != it["product"] and v["available"]]
            if not pool:
                raise Skip("no other product")
            v = rng.choice(pool)
            ref = f"v_{v['id']}"
            new_act = (kind, oid, [(item_id, v["id"])], act[3])
        elif how == "gift_balance":
            gifts = [p for p in self.pms_of(st, uid)
                     if p["kind"] == "gift_card"]
            if not gifts:
                raise Skip("no gift card")
            g = rng.choice(gifts)
            if kind == "payment":
                if g["id"] == o["payment_method"]:
                    raise Skip("gift card already pays")
                need = o["total"]
                new_act = ("payment", oid, g["id"])
            else:
                item_id, vid = act[2][0]
                it = self.rec(st, "order_item", item_id)
                v = self.rec(st, "variant", vid)
                if v["price"] - it["price"] <= 100:
                    pricier = [x for x in self.variants_of(st, it["product"])
                               if x["available"] and x["id"] != it["variant"]
                               and x["price"] - it["price"] > 100]
                    if not pricier:
                        raise Skip("no pricier variant")
                    v = rng.choice(pricier)
                need = v["price"] - it["price"]
                new_act = (kind, oid, [(item_id, v["id"])], g["id"])
            g["balance"] = max(0, need - rng.choice(
                [100, 500, 1000, max(1, need // 2)]))
            ref = f"pm_{g['id']}"
        elif how == "refund_method":
            bad = [p for p in self.pms_of(st, uid)
                   if p["id"] != o["payment_method"]
                   and p["kind"] != "gift_card"]
            if not bad:
                raise Skip("no other non-gift method")
            p = rng.choice(bad)
            new_act = ("return", oid, act[2], p["id"])
            ref = f"pm_{p['id']}"
            refund_mode = "named"
        elif how == "bad_reason":
            new_act = ("cancel", oid, rng.choice(BAD_REASONS))
            ref = "reason"
        else:
            return None
        d = self.build(uid, [new_act], rng, ident, st,
                       refund_mode=refund_mode)
        return d, ref, how

    def ask(self, uid: str, act: tuple, ident: str, rng, state: dict):
        """(ask, act) drafts when one value the policy needs is left out, or
        None."""
        kind = act[0]
        if kind == "cancel" and act[2] in SW.RETAIL_CANCEL_REASONS:
            seed = rng.random()
            full = self.build(uid, [act], random.Random(seed), ident,
                              copy.deepcopy(state))
            q = full.copy()
            for k in [k for k in q.consts if k.startswith("reason")]:
                del q.consts[k]
            who = q.request[:q.request.index(act[1])].rsplit(".", 1)[0] + "."
            q.request = who + " " + rng.choice([
                f"Please cancel my order {act[1]}.",
                f"I need order {act[1]} cancelled."])
            q.program = "ABORT NEEDS_INFO @order.cancel_reason\n"
            q.status = "aborted"
            a = full.copy()
            a.request = q.request + " " + rng.choice(REPLIES).format(
                question="why they want to cancel ('no longer needed' or "
                         "'ordered by mistake').", answer=act[2])
            return q, a
        if kind in ("modify", "exchange") and len(
                self.pms_of(state, uid)) >= 2:
            full = self.build(uid, [act], rng, ident, copy.deepcopy(state))
            q = full.copy()
            for k in [k for k in q.consts if k.startswith("pm_")]:
                del q.consts[k]
            q.request = _PAY_CLAUSE.sub(".", q.request)
            if q.request == full.request:
                return None
            q.program = "ABORT NEEDS_INFO @payment_method.id\n"
            q.status = "aborted"
            pm = self.rec(state, "payment_method", act[3])
            a = full.copy()
            a.request = q.request + " " + rng.choice(REPLIES).format(
                question="which payment method pays or receives the price "
                         "difference.", answer=f"my {self.pm_short(pm, state)}")
            return q, a
        return None


# ================================================================ airline

CABIN_WORDS = {"basic_economy": "basic economy", "economy": "economy",
               "business": "business"}
AIR_REASON_PHRASE = {
    "change of plan": ["my plans changed", "I no longer need the trip",
                       "my plans have changed"],
    "health": ["I'm sick", "I'm unwell and can't travel",
               "I have a health problem"],
    "weather": ["a storm is forecast", "the weather makes it unsafe"],
    "airline cancelled flight": ["the airline cancelled my flight"],
    "other": ["it clashes with a friend's birthday",
              "I'd simply rather not go", "something else came up"],
}
UNSUPPORTED_ASKS = [
    ("remove_passenger", "Please remove passenger {pax} from reservation "
                         "{rid}; they're not coming."),
    ("add_insurance", "I'd like to add travel insurance to reservation "
                      "{rid} now."),
    ("refund_insurance", "Refund the travel insurance on reservation {rid} "
                         "but keep the flights."),
    ("change_destination", "Change the destination of reservation {rid} to "
                           "{other}, same dates."),
]

# Train tasks whose expected outcome is a refusal (tau2 labels them by
# nl_assertions, not actions): our reading of each scenario, as the request
# the customer makes. A would-be program the engine must refuse, or a request
# no tool performs.
AIRLINE_POLICY_TASKS = {
    "0": [("cancel", "EHGLP3", "other")],
    "1": [("cancel", "Q69X3R", "change of plan")],
    "5": [("certificate", "3JA7XV", 20000)],
    "9": [("cancel", "IFOYYZ", "change of plan"),
          ("cancel", "NQNU5R", "change of plan")],
    "10": [("unsupported", "4NQLHD", "upgrade only the IAH to SEA flight of "
            "reservation {rid} to business")],
    "11": [("unsupported", "GV1N64", "remove passenger Sophia from "
            "reservation {rid}")],
    "27": [("certificate", "M61CQM", 15000)],
    "28": [("cancel", "SI5UKW", "change of plan")],
    "36": [("flight_by_days", "EUJUY6", 0, 2)],
    "41": [("cancel", "UDMOP1", "change of plan"),
           ("cancel", "4XGCCM", "change of plan")],
    "43": [("cancel", "D1EW9B", "change of plan"),
           ("cancel", "9HBUV8", "change of plan")],
    "46": [("unsupported", "H8Q05L", "refund the insurance on reservation "
            "{rid} without cancelling it")],
    "47": [("cancel", "H8Q05L", "other")],
    "49": [("cancel", "3RK2T9", "health")],
}
# the reason a tau2 cancel_reservation (which takes none) was given, read
# from the scenario
AIRLINE_CANCEL_REASON = {"7": "health", "14": "change of plan",
                         "23": "change of plan", "39": "change of plan",
                         "42": "change of plan"}


def ago(seconds: int) -> str:
    h = seconds // 3600
    if h < 48:
        return f"{h} hour{'s' if h != 1 else ''} ago"
    return f"{h // 24} days ago"


class Airline:
    world = "service_airline"
    NATURAL_REFUSALS = True

    def __init__(self):
        self.db = load("airline", "db_extract.json")
        self.tasks = load("airline", "tasks_train.json")
        self.now = SW.AIRLINE_NOW
        # template users' reservations with a delayed or cancelled flight:
        # the only ones a compensation request can be about
        self.disrupted = [
            (u, rid) for u in self.db["template_users"]
            for rid in self.db["users"][u]["reservations"]
            if rid in self.db["reservations"] and any(
                self._status(f) in ("delayed", "cancelled")
                for f in self.db["reservations"][rid]["flights"])]

    def _status(self, f: dict) -> Optional[str]:
        return (self.db["flights"].get(f["flight_number"], {})
                .get("dates", {}).get(f["date"], {}).get("status"))

    # ------------------------------------------------------------ state
    @staticmethod
    def fid(num: str, date: str) -> str:
        return f"{num}@{date}"

    def flight_rec(self, num: str, date: str) -> Optional[dict]:
        f = self.db["flights"].get(num)
        if not f or date not in f["dates"]:
            return None
        info = f["dates"][date]
        prices = info.get("prices") or {}
        return {"id": self.fid(num, date), "number": num,
                "origin": f["origin"], "destination": f["destination"],
                "date": date, "status": info["status"],
                "departs": f["scheduled_departure_time_est"][:5],
                "price_basic_economy": cents(prices.get("basic_economy", 0)),
                "price_economy": cents(prices.get("economy", 0)),
                "price_business": cents(prices.get("business", 0))}

    def state_for(self, uids: List[str], extra_flights=()) -> dict:
        ents = {k: [] for k in SW.AIRLINE["default_state"]["entities"]}
        flights = set(extra_flights)
        for uid in uids:
            u = self.db["users"][uid]
            ents["user"].append({
                "id": uid, "first_name": u["name"]["first_name"],
                "last_name": u["name"]["last_name"], "email": u["email"],
                "membership": u["membership"]})
            for pm in u["payment_methods"].values():
                ents["payment_method"].append({
                    "id": pm["id"], "user": uid, "kind": pm["source"],
                    "balance": cents(pm.get("amount", 0)),
                    "last_four": pm.get("last_four", "")})
            for rid in u["reservations"]:
                r = self.db["reservations"].get(rid)
                if r is None:
                    continue
                ents["reservation"].append({
                    "id": rid, "user": uid, "origin": r["origin"],
                    "destination": r["destination"],
                    "trip_type": r["flight_type"], "cabin": r["cabin"],
                    "passenger_count": len(r["passengers"]),
                    "total_baggages": r["total_baggages"],
                    "nonfree_baggages": r["nonfree_baggages"],
                    "insurance": r["insurance"] == "yes",
                    "created": SW._epoch(r["created_at"]),
                    "status": "cancelled" if r.get("status") == "cancelled"
                    else "active",
                    "payment_method": r["payment_history"][0]["payment_id"],
                    "cancel_reason": None, "changed": False})
                for i, f in enumerate(r["flights"]):
                    fid = self.fid(f["flight_number"], f["date"])
                    flights.add(fid)
                    ents["segment"].append({
                        "id": f"{rid}-{i + 1}", "reservation": rid,
                        "flight": fid, "origin": f["origin"],
                        "destination": f["destination"], "date": f["date"],
                        "price": cents(f["price"])})
                for i, p in enumerate(r["passengers"]):
                    ents["passenger"].append({
                        "id": f"{rid}-p{i + 1}", "reservation": rid,
                        "first_name": p["first_name"],
                        "last_name": p["last_name"], "dob": p["dob"]})
        for fid in sorted(flights):
            num, date = fid.split("@")
            rec = self.flight_rec(num, date)
            if rec:
                ents["flight"].append(rec)
        return {"entities": ents, "outbox": [], "payments": []}

    def base_state(self, uid: str, rng, extra_flights=()) -> dict:
        pool = [u for u in self.db["template_users"] if u != uid]
        return self.state_for([uid, rng.choice(pool)], extra_flights)

    rec = staticmethod(Retail.rec)

    @staticmethod
    def segs(state, rid):
        return [s for s in state["entities"]["segment"]
                if s["reservation"] == rid]

    @staticmethod
    def pax(state, rid):
        return [p for p in state["entities"]["passenger"]
                if p["reservation"] == rid]

    def fl(self, state, fid) -> Optional[dict]:
        return next((f for f in state["entities"]["flight"]
                     if f["id"] == fid), None)

    # ------------------------------------------------------------ phrases
    @staticmethod
    def pm_short(pm: dict, state: Optional[dict] = None) -> str:
        if pm["kind"] == "gift_card":
            return "gift card" + _twin_suffix(pm, state)
        if pm["kind"] == "certificate":
            return (f"{dollars(pm['balance'])} travel certificate"
                    + _twin_suffix(pm, state, same_balance=True))
        return f"credit card ending {pm['last_four']}"

    @staticmethod
    def pm_desc(pm: dict) -> str:
        if pm["kind"] == "gift_card":
            return f"gift card {pm['id']}, balance {dollars(pm['balance'])}"
        if pm["kind"] == "certificate":
            return (f"travel certificate {pm['id']}, "
                    f"{dollars(pm['balance'])}")
        return f"credit card {pm['id']} ending {pm['last_four']}"

    def res_desc(self, state, r) -> str:
        segs = self.segs(state, r["id"])
        flown = any((self.fl(state, s["flight"]) or {}).get("status") in
                    ("landed", "flying") for s in segs)
        flags = [f"flight {s['flight'].split('@')[0]} on {s['date']} "
                 f"{self.fl(state, s['flight'])['status']}" for s in segs
                 if (self.fl(state, s["flight"]) or {}).get("status") in
                 ("delayed", "cancelled")]
        n = r["passenger_count"]
        return (f"reservation {r['id']}: {r['origin']} to {r['destination']} "
                f"{r['trip_type'].replace('_', ' ')}, "
                f"{CABIN_WORDS[r['cabin']]}, {n} passenger{'s' if n > 1 else ''}"
                f", booked {ago(self.now - r['created'])}, "
                f"{'with' if r['insurance'] else 'no'} travel insurance, "
                f"{'partly flown' if flown else 'not yet flown'}"
                + (f", {r['status']}" if r["status"] != "active" else "")
                + ("; " + "; ".join(flags) if flags else ""))

    def seg_desc(self, state, s) -> str:
        f = self.fl(state, s["flight"]) or {}
        return (f"the {s['origin']} to {s['destination']} flight "
                f"{s['flight'].split('@')[0]} on {s['date']} in reservation "
                f"{s['reservation']} ({f.get('status', 'unknown')})")

    @staticmethod
    def flight_desc(f, cabin) -> str:
        price = f.get(f"price_{cabin}", 0)
        return (f"flight {f['number']} {f['origin']} to {f['destination']} "
                f"on {f['date']}, departs {f['departs']}, {f['status']}"
                + (f", {CABIN_WORDS[cabin]} {dollars(price)}" if price else ""))

    # ------------------------------------------------------------ drafts
    def add_res(self, d, rid) -> str:
        r = self.rec(d.state, "reservation", rid)
        return d.const(f"r_{rid}", "ID:reservation", rid,
                       self.res_desc(d.state, r))

    def add_pms(self, d, uid):
        for pm in [p for p in d.state["entities"]["payment_method"]
                   if p["user"] == uid]:
            d.const(f"pm_{pm['id']}", "ID:payment_method", pm["id"],
                    self.pm_desc(pm))

    def build(self, uid, actions, rng, ident, state) -> Draft:
        """actions:
          ('cancel', rid, reason)          ('cabin', rid, cabin, pm)
          ('flight', rid, [(segment, flight)], pm)
          ('bags', rid, total, pm)
          ('passenger', rid, [(pax, first, last, dob)])
          ('certificate', rid, amount)     ('unsupported', rid, text)"""
        d = Draft(self.world)
        d.state = state
        u = self.rec(state, "user", uid)
        name = f"{u['first_name']} {u['last_name']}"
        d.const("user", "ID:user", uid, f"{name}'s user id")
        who = rng.choice([f"I'm {name}, user id {uid}.",
                          f"This is {name} ({uid})."])
        self.add_pms(d, uid)
        parts, lines = [], []
        for act in actions:
            kind, rid = act[0], act[1]
            r_ref = self.add_res(d, rid)
            r = self.rec(state, "reservation", rid)
            if kind == "cancel":
                reason = act[2]
                ref = d.const("reason", "STR", reason,
                              f"cancellation reason '{reason}'",
                              "enum:reservation.cancel_reason")
                for x in rng.sample(SW.AIRLINE_CANCEL_REASONS, 2):
                    d.const(f"reason_{x[:5]}", "STR", x,
                            f"cancellation reason '{x}'",
                            "enum:reservation.cancel_reason")
                phrase = rng.choice(AIR_REASON_PHRASE[reason])
                parts.append(rng.choice([
                    f"Please cancel reservation {rid}; {phrase}.",
                    f"Cancel my reservation {rid}, {phrase}.",
                    f"I need reservation {rid} cancelled because {phrase}."]))
                lines.append(f"CALL @cancel_reservation {r_ref} {ref}\n")
            elif kind == "cabin":
                cabin, pmid = act[2], act[3]
                pm = self.rec(state, "payment_method", pmid)
                c_ref = d.const(f"cabin_{cabin}", "STR", cabin,
                                f"cabin class '{cabin}'",
                                "enum:reservation.cabin")
                for x in SW.AIRLINE_CABINS:
                    d.const(f"cabin_{x}", "STR", x, f"cabin class '{x}'",
                            "enum:reservation.cabin")
                up = (SW.AIRLINE_CABINS.index(cabin)
                      > SW.AIRLINE_CABINS.index(r["cabin"]))
                verb = "Upgrade" if up else "Move"
                parts.append(rng.choice([
                    f"{verb} reservation {rid} to {CABIN_WORDS[cabin]} for "
                    f"everyone, using my {self.pm_short(pm, d.state)}.",
                    f"Please change the cabin on {rid} to "
                    f"{CABIN_WORDS[cabin]}; pay or refund via my "
                    f"{self.pm_short(pm, d.state)}."]))
                lines.append(f"CALL @update_reservation_cabin {r_ref} "
                             f"{c_ref} {{pm_{pmid}}}\n")
            elif kind == "flight":
                pairs, pmid = act[2], act[3]
                pm = self.rec(state, "payment_method", pmid)
                moves = []
                for sid, fid in pairs:
                    s = self.rec(state, "segment", sid)
                    f = self.fl(state, fid)
                    s_ref = d.const(f"s_{sid}", "ID:segment", sid,
                                    self.seg_desc(state, s))
                    f_ref = d.const(f"f_{fid}", "ID:flight", fid,
                                    self.flight_desc(f, r["cabin"]))
                    for alt in [x for x in state["entities"]["flight"]
                                if x["origin"] == f["origin"]
                                and x["destination"] == f["destination"]
                                and x["id"] not in (fid, s["flight"])][:2]:
                        d.const(f"f_{alt['id']}", "ID:flight", alt["id"],
                                self.flight_desc(alt, r["cabin"]))
                    moves.append(f"my {s['origin']} to {s['destination']} "
                                 f"flight on {s['date']} to {f['number']} on "
                                 f"{f['date']}")
                    lines.append(f"CALL @update_reservation_flights {s_ref} "
                                 f"{f_ref} {{pm_{pmid}}}\n")
                for s in self.segs(state, rid):
                    d.const(f"s_{s['id']}", "ID:segment", s["id"],
                            self.seg_desc(state, s))
                parts.append(f"On reservation {rid}, move {join_and(moves)}; "
                             f"use my {self.pm_short(pm, d.state)} for any difference.")
            elif kind == "bags":
                total, pmid = act[2], act[3]
                pm = self.rec(state, "payment_method", pmid)
                cur = r["total_baggages"]
                t_ref = d.const("bags_total", "INT", total,
                                f"{total} checked bags in total")
                if total > cur:
                    d.const("bags_add", "INT", total - cur,
                            f"{total - cur} more checked bag"
                            f"{'s' if total - cur > 1 else ''}")
                    ask_ = rng.choice([
                        f"add {total - cur} checked bag"
                        f"{'s' if total - cur > 1 else ''} to reservation "
                        f"{rid} (I have {cur} now)",
                        f"bring reservation {rid} to {total} checked bags"])
                else:
                    ask_ = (f"bring reservation {rid} down to {total} checked "
                            f"bag{'s' if total != 1 else ''}")
                parts.append(f"Please {ask_}; charge any fee to my "
                             f"{self.pm_short(pm, d.state)}.")
                lines.append(f"CALL @update_reservation_baggages {r_ref} "
                             f"{t_ref} {{pm_{pmid}}}\n")
            elif kind == "passenger":
                changes = []
                for pid, first, last, dob in act[2]:
                    p = self.rec(state, "passenger", pid)
                    p_ref = d.const(f"p_{pid}", "ID:passenger", pid,
                                    f"passenger {p['first_name']} "
                                    f"{p['last_name']} (born {p['dob']}) on "
                                    f"reservation {rid}")
                    fr = d.const(f"first_{pid}", "STR", first,
                                 f"first name {first}", "name")
                    lr = d.const(f"last_{pid}", "STR", last,
                                 f"last name {last}", "name")
                    br = d.const(f"dob_{pid}", "STR", dob,
                                 f"date of birth {dob}", "text")
                    changes.append(f"{p['first_name']} {p['last_name']} to "
                                   f"{first} {last} (born {dob})")
                    lines.append(f"CALL @update_reservation_passengers "
                                 f"{p_ref} {fr} {lr} {br}\n")
                for p in self.pax(state, rid):
                    d.const(f"p_{p['id']}", "ID:passenger", p["id"],
                            f"passenger {p['first_name']} {p['last_name']} "
                            f"(born {p['dob']}) on reservation {rid}")
                parts.append(f"On reservation {rid}, change passenger "
                             f"{join_and(changes)}.")
            elif kind == "certificate":
                amount = act[2]
                n = r["passenger_count"]
                a_ref = d.const("amount", "INT", amount,
                                f"{dollars(amount)} certificate")
                for x in {10000 * n, 5000 * n, 10000, 5000} - {amount}:
                    d.const(f"amount_{x}", "INT", x,
                            f"{dollars(x)} certificate")
                segs = self.segs(state, rid)
                bad = [s for s in segs if (self.fl(state, s["flight"]) or {})
                       .get("status") in ("delayed", "cancelled")]
                what = (f"flight {bad[0]['flight'].split('@')[0]} was "
                        f"{self.fl(state, bad[0]['flight'])['status']}"
                        if bad else "my flight was disrupted")
                parts.append(rng.choice([
                    f"My {what} on reservation {rid}; I'd like the "
                    f"compensation certificate.",
                    f"On reservation {rid} my {what}. Please send me "
                    f"compensation."]))
                lines.append(f"CALL @send_certificate {r_ref} {a_ref}\n")
            elif kind == "unsupported":
                text = act[2]
                parts.append(text)
                d.const("wish", "STR", text, "your request, verbatim", "text")
                lines.append("ABORT UNSUPPORTED {wish}\n")
            else:
                raise Skip(f"unknown airline action {kind}")
        others = [x for x in state["entities"]["reservation"]
                  if x["user"] == uid and f"r_{x['id']}" not in d.consts]
        for x in rng.sample(others, min(2, len(others))):
            self.add_res(d, x["id"])
        d.request = " ".join([who] + parts)
        d.program = "".join(lines)
        if not d.program.rstrip().endswith(("STOP", "ABORT UNSUPPORTED "
                                            "{wish}")):
            d.program += "STOP\n"
        d.prov["actions"] = [a[0] for a in actions]
        return d

    def natural_ref(self, act, msg) -> str:
        if "certificate cannot pay" in msg or "gift card" in msg:
            return f"pm_{act[-1]}"
        if act[0] == "bags" and "not removed" in msg:
            return "bags_total"
        if act[0] == "flight" and ("not available" in msg or "origin" in msg):
            return f"f_{act[2][0][1]}"
        if act[0] == "flight" and "departed" in msg:
            return f"s_{act[2][0][0]}"
        return f"r_{act[1]}"

    # ------------------------------------------------------------ tau2
    def tau2_task(self, run: "Run", task: dict, rng: random.Random) -> None:
        for task in [task]:
            tid = task["id"]
            prov = {"source_kind": "tau2", "tau2_task": tid,
                    "file": "data/tau2/domains/airline/tasks.json (train)",
                    "recipe": "tau2_task", "teacher": "tau2-expected-actions"}
            known = task["user_scenario"]["instructions"].get(
                "known_info") or ""
            m = _UID.search(known)
            uid = m.group(0) if m else None
            if uid not in self.db["users"]:
                run.dropped["tau2 task: user id not stated"] += 1
                continue
            if tid in AIRLINE_POLICY_TASKS:
                for act in AIRLINE_POLICY_TASKS[tid]:
                    self.policy_row(run, uid, act, rng, prov)
            try:
                acts = self.task_actions(task, uid)
            except Skip as e:
                run.dropped[f"tau2 task: {_reason(str(e))}"] += 1
                continue
            if not acts:
                if tid not in AIRLINE_POLICY_TASKS:
                    run.dropped["tau2 task: no writes and no policy "
                                "mapping (information task)"] += 1
                continue
            extra = [f for a in acts if a[0] == "flight"
                     for _, f in a[2]]
            run.group(uid, acts, "uid", rng, prov,
                      self.base_state(uid, rng, extra), natural_ok="if_whole")

    def policy_row(self, run, uid, act, rng, prov) -> None:
        kind, rid = act[0], act[1]
        state = self.base_state(uid, rng)
        if kind == "unsupported":
            text = act[2].format(rid=rid)
            text = "I'd like to " + text + "."
            d = self.build(uid, [("unsupported", rid, text[0].upper()
                                  + text[1:])], rng, "uid", state)
            d.status = "aborted"
            d.prov.update(prov, variant="forbidden_tau2", policy_reason=
                          "no tool performs this")
            d.tags |= {"service", "policy_forbidden", "no_tool"}
            run.emit(d, "forbid")
            return
        if kind == "flight_by_days":
            idx, days = act[2], act[3]
            s = self.segs(state, rid)[idx]
            num = s["flight"].split("@")[0]
            day = int(s["date"][-2:]) + days
            new = self.fid(num, f"{s['date'][:-2]}{day:02d}")
            state = self.base_state(uid, rng, [new])
            pms = [p for p in state["entities"]["payment_method"]
                   if p["user"] == uid and p["kind"] != "certificate"]
            act = ("flight", rid, [(s["id"], new)], pms[0]["id"])
        elif kind == "certificate":
            act = ("certificate", rid, act[2])
        d = self.build(uid, [act], rng, "uid", state)
        run.abort_only(d, self.natural_ref(act, refused(d) or ""), prov,
                       expect_refusal=True)

    def task_actions(self, task, uid) -> List[tuple]:
        acts = []
        for a in task["evaluation_criteria"]["actions"] or []:
            n, args = a["name"], a["arguments"]
            if n.startswith(("get_", "search_", "list_", "calc")):
                continue
            if n == "book_reservation":
                raise Skip("book_reservation (booking is not in this world)")
            rid = args.get("reservation_id")
            r = self.db["reservations"].get(rid)
            if r is None:
                raise Skip("reservation not in extract")
            if n == "cancel_reservation":
                reason = AIRLINE_CANCEL_REASON.get(task["id"])
                if reason is None:
                    raise Skip("cancel with no reason mapped")
                acts.append(("cancel", rid, reason))
            elif n == "update_reservation_flights":
                old = [(f["flight_number"], f["date"]) for f in r["flights"]]
                new = [(f["flight_number"], f["date"]) for f in args["flights"]]
                if old == new and args["cabin"] != r["cabin"]:
                    acts.append(("cabin", rid, args["cabin"],
                                 args["payment_id"]))
                elif len(old) == len(new) and args["cabin"] == r["cabin"]:
                    pairs = []
                    for i, (o_, n_) in enumerate(zip(old, new)):
                        if o_ != n_:
                            pairs.append((f"{rid}-{i + 1}", self.fid(*n_)))
                    acts.append(("flight", rid, pairs, args["payment_id"]))
                else:
                    raise Skip("flight change that alters segment count or "
                               "cabin at once")
            elif n == "update_reservation_baggages":
                acts.append(("bags", rid, args["total_baggages"],
                             args["payment_id"]))
            elif n == "update_reservation_passengers":
                olds = r["passengers"]
                if len(olds) != len(args["passengers"]):
                    raise Skip("passenger count change")
                ch = [(f"{rid}-p{i + 1}", p["first_name"], p["last_name"],
                       p["dob"]) for i, (o_, p) in
                      enumerate(zip(olds, args["passengers"])) if o_ != p]
                acts.append(("passenger", rid, ch))
            elif n == "send_certificate":
                raise Skip("send_certificate by user id")
            else:
                raise Skip(f"unmapped tau2 action {n}")
        return acts

    # ------------------------------------------------------------ templates
    def template(self, rng):
        for _ in range(50):
            uid = rng.choice(self.db["template_users"])
            u = self.db["users"][uid]
            res = [self.db["reservations"][x] for x in u["reservations"]
                   if x in self.db["reservations"]]
            if not res:
                continue
            r = rng.choice(res)
            rid = r["reservation_id"]
            pms = list(u["payment_methods"].values())
            payers = [p for p in pms if p["source"] != "certificate"]
            kind = rng.choices(["cancel", "cabin", "flight", "bags",
                                "passenger", "certificate", "unsupported"],
                               [5, 3, 3, 3, 2, 2, 1])[0]
            if kind == "cancel":
                reasons = list(SW.AIRLINE_CANCEL_REASONS)
                reasons.remove("airline cancelled flight")
                if any((self.db["flights"].get(f["flight_number"], {})
                        .get("dates", {}).get(f["date"], {}).get("status"))
                       == "cancelled" for f in r["flights"]):
                    reasons.append("airline cancelled flight")
                return uid, [("cancel", rid, rng.choice(reasons))], "uid"
            if kind == "cabin":
                cabin = rng.choice([c for c in SW.AIRLINE_CABINS
                                    if c != r["cabin"]])
                pm = rng.choice(payers if rng.random() < 0.85 else pms)
                return uid, [("cabin", rid, cabin, pm["id"])], "uid"
            if kind == "flight":
                i = rng.randrange(len(r["flights"]))
                f = r["flights"][i]
                alts = []
                for num, fl in self.db["flights"].items():
                    if (fl["origin"], fl["destination"]) != (
                            f["origin"], f["destination"]):
                        continue
                    for date in fl["dates"]:
                        if abs(int(date[-2:]) - int(f["date"][-2:])) <= 1 \
                                and (num, date) != (f["flight_number"],
                                                    f["date"]):
                            alts.append(self.fid(num, date))
                if not alts or not payers:
                    continue
                return uid, [("flight", rid, [(f"{rid}-{i + 1}",
                                               rng.choice(alts))],
                              rng.choice(payers)["id"])], "uid"
            if kind == "bags":
                cur = r["total_baggages"]
                total = cur + rng.choice([1, 1, 2, 3]) if (
                    cur == 0 or rng.random() < 0.8) else cur - 1
                if not payers:
                    continue
                return uid, [("bags", rid, total,
                              rng.choice(payers)["id"])], "uid"
            if kind == "passenger":
                i = rng.randrange(len(r["passengers"]))
                p = r["passengers"][i]
                other = self.db["users"][rng.choice(
                    self.db["template_users"])]["name"]
                first, last = rng.choice([
                    (p["first_name"], other["last_name"]),
                    (other["first_name"], p["last_name"]),
                    (other["first_name"], other["last_name"])])
                return uid, [("passenger", rid, [(f"{rid}-p{i + 1}", first,
                                                  last, p["dob"])])], "uid"
            if kind == "certificate":
                if not self.disrupted:
                    continue
                uid, rid = rng.choice(self.disrupted)
                r = self.db["reservations"][rid]
                bad = [self._status(f) for f in r["flights"]]
                n = len(r["passengers"])
                if "cancelled" in bad:
                    return uid, [("certificate", rid, 10000 * n)], "uid"
                if "delayed" in bad:
                    if rng.random() < 0.6:
                        pre = ("cancel", rid, rng.choice(
                            ["change of plan", "health"]))
                        return uid, [pre, ("certificate", rid, 5000 * n)], \
                            "uid"
                    return uid, [("certificate", rid, 5000 * n)], "uid"
                continue
            if kind == "unsupported":
                name, tmpl = rng.choice(UNSUPPORTED_ASKS)
                p = r["passengers"][0]
                other = rng.choice([a for a in ("JFK", "SFO", "ORD", "SEA",
                                                "MIA") if a != r["destination"]])
                return uid, [("unsupported", rid, tmpl.format(
                    rid=rid, pax=f"{p['first_name']} {p['last_name']}",
                    other=other))], "uid"
        raise Skip("no template draw")

    def base_state_for(self, uid, actions, rng):
        extra = [f for a in actions if a[0] == "flight" for _, f in a[2]]
        return self.base_state(uid, rng, extra)

    # ------------------------------------------------------------ variants
    def forbidden(self, uid, act, ident, rng, state):
        kind, rid = act[0], act[1]
        st = copy.deepcopy(state)
        hows = []
        if kind in ("cancel", "cabin", "flight"):
            hows.append("flown")
        if kind == "cancel":
            hows.append("old_booking")
        if kind == "flight":
            hows += ["basic_economy", "unavailable"]
        if kind in ("cabin", "flight", "bags"):
            hows.append("certificate_pay")
        if kind == "bags":
            hows.append("remove_bags")
        if kind == "certificate":
            hows.append("not_eligible")
        rng.shuffle(hows)
        for how in hows:
            s2 = copy.deepcopy(st)
            r2 = self.rec(s2, "reservation", rid)
            new = act
            if how == "flown":
                f = self.fl(s2, self.segs(s2, rid)[0]["flight"])
                if f is None:
                    continue
                f["status"] = "landed"
                ref = f"r_{rid}"
            elif how == "old_booking":
                r2["created"] = self.now - rng.choice([3, 6, 11]) * 86400
                r2["insurance"] = False
                if r2["cabin"] == "business":
                    r2["cabin"] = "economy"
                ref = f"r_{rid}"
            elif how == "basic_economy":
                r2["cabin"] = "basic_economy"
                ref = f"r_{rid}"
            elif how == "unavailable":
                fid = act[2][0][1]
                f = self.fl(s2, fid)
                f["status"] = rng.choice(["on time", "delayed", "cancelled"])
                ref = f"f_{fid}"
            elif how == "certificate_pay":
                certs = [p for p in s2["entities"]["payment_method"]
                         if p["user"] == uid and p["kind"] == "certificate"]
                if not certs:
                    continue
                c = rng.choice(certs)
                new = tuple(list(act[:-1]) + [c["id"]])
                ref = f"pm_{c['id']}"
            elif how == "remove_bags":
                r2["total_baggages"] = act[2] + rng.choice([1, 2])
                ref = "bags_total"
            elif how == "not_eligible":
                self.rec(s2, "user", uid)["membership"] = "regular"
                r2["insurance"] = False
                if r2["cabin"] == "business":
                    r2["cabin"] = "economy"
                ref = f"r_{rid}"
            else:
                continue
            try:
                d = self.build(uid, [new], rng, ident, s2)
            except Skip:
                continue
            return d, ref, how
        return None

    def ask(self, uid, act, ident, rng, state):
        kind = act[0]
        if kind == "cancel":
            full = self.build(uid, [act], rng, ident, copy.deepcopy(state))
            q = full.copy()
            for k in [k for k in q.consts if k.startswith("reason")]:
                del q.consts[k]
            who = q.request[:q.request.index(act[1])].rsplit(".", 1)[0] + "."
            q.request = who + " " + rng.choice([
                f"Please cancel reservation {act[1]}.",
                f"I want to cancel my reservation {act[1]}."])
            q.program = "ABORT NEEDS_INFO @reservation.cancel_reason\n"
            q.status = "aborted"
            a = full.copy()
            a.request = q.request + " " + rng.choice(REPLIES).format(
                question="the reason for cancelling (change of plan, airline "
                         "cancelled flight, health, weather or other).",
                answer=act[2])
            return q, a
        return None


# ================================================================ telecom

COMPLAINT = {
    "mms_issue": ["I can't send picture messages (MMS) from my messaging "
                  "app", "my MMS messages won't send",
                  "picture messages fail to send"],
    "mobile_data_issue": ["my mobile data isn't working properly",
                          "mobile data is down or crawling on my phone",
                          "I can't get a decent mobile data connection"],
    "service_issue": ["my phone shows no service", "I have no cell service "
                      "at all", "my phone can't connect to the network"],
}
PHONE_DEFAULT = {"airplane_mode": False, "network_mode": "4g_5g_preferred",
                 "wifi_calling": False, "apn": "default", "mobile_data": True,
                 "data_saver": False, "roaming": False, "vpn": False,
                 "sim": "seated", "abroad": False, "sms_permission": True,
                 "storage_permission": True, "needs_reboot": False}
# tau2 user-side and agent-side actions -> our tools (arguments by kind)
TELECOM_TOOL = {
    "toggle_airplane_mode": "phone", "toggle_wifi_calling": "phone",
    "reset_apn_settings": "phone", "reboot_device": "phone",
    "toggle_data": "phone", "reseat_sim_card": "phone",
    "toggle_roaming": "phone", "disconnect_vpn": "phone",
    "toggle_data_saver_mode": "phone",
    "set_network_mode_preference": "phone_mode",
    "grant_app_permission": "phone_perm", "refuel_data": "line_gb",
    "enable_roaming": "line", "resume_line": "line",
    "send_payment_request": "bill", "make_payment": "bill",
}


class Telecom:
    world = "service_telecom"

    def __init__(self):
        import tomllib
        self.db = tomllib.loads((BORROWED / "telecom" / "db.toml").read_text(
            encoding="utf-8"))
        self.tasks = load("telecom", "tasks_train.json")

    def state_for(self, task: dict) -> Tuple[dict, dict]:
        """Our state for a tau2 telecom task: the database's customer, lines,
        plans and bills, then the task's initialization actions applied."""
        db = self.db
        cust = next(c for c in db["customers"] if c["customer_id"] == "C1001")
        ents = {k: [] for k in SW.TELECOM["default_state"]["entities"]}
        ents["customer"].append({
            "id": cust["customer_id"], "name": cust["full_name"],
            "phone_number": cust["phone_number"],
            "dob": cust["date_of_birth"], "status": cust["account_status"]})
        for p in db["plans"]:
            ents["plan"].append({
                "id": p["plan_id"], "name": p["name"],
                "data_limit_gb": int(p["data_limit_gb"]),
                "refuel_price_per_gb": cents(p["data_refueling_price_per_gb"]),
                "monthly_price": cents(p["price_per_month"])})
        for line in db["lines"]:
            if line["line_id"] not in cust["line_ids"]:
                continue
            ents["line"].append({
                "id": line["line_id"], "customer": cust["customer_id"],
                "phone_number": line["phone_number"],
                "status": line["status"], "plan": line["plan_id"],
                "data_used_mb": int(round(line["data_used_gb"] * 1000)),
                "data_refuel_gb": int(line["data_refueling_gb"]),
                "roaming_enabled": bool(line["roaming_enabled"]),
                "contract_end": line["contract_end_date"]})
        for b in db["bills"]:
            if b["customer_id"] == cust["customer_id"]:
                ents["bill"].append({
                    "id": b["bill_id"], "customer": b["customer_id"],
                    "amount": cents(b["total_due"]), "status": b["status"],
                    "due": b["due_date"]})
        phone = dict(PHONE_DEFAULT, id="phone_L1002", line="L1002")
        ents["phone"].append(phone)
        facts = {"abroad": None}
        line = next(x for x in ents["line"] if x["id"] == "L1002")
        for a in task["initial_state"]["initialization_actions"]:
            fn, args = a["func_name"], a["arguments"]
            if fn == "set_user_location":
                phone["abroad"] = bool(args["abroad"])
            elif fn == "set_network_mode_preference":
                phone["network_mode"] = args["mode"]
            elif fn == "turn_airplane_mode_on":
                phone["airplane_mode"] = True
            elif fn == "set_wifi_calling":
                phone["wifi_calling"] = bool(args["enabled"])
            elif fn == "break_apn_mms_setting":
                phone["apn"] = "MMS settings broken"
            elif fn == "break_apn_settings":
                phone["apn"] = "broken"
            elif fn == "remove_app_permission":
                phone[f"{args['permission']}_permission"] = False
            elif fn == "turn_data_off":
                phone["mobile_data"] = False
            elif fn == "unseat_sim_card":
                phone["sim"] = "missing"
            elif fn == "lock_sim_card":
                phone["sim"] = "pin_locked"
            elif fn == "turn_roaming_off":
                phone["roaming"] = False
            elif fn == "turn_roaming_on":
                phone["roaming"] = True
            elif fn == "turn_data_saver_mode_on":
                phone["data_saver"] = True
            elif fn == "break_vpn":
                phone["vpn"] = True
            elif fn == "set_data_usage":
                line["data_used_mb"] = int(round(args["data_used_gb"] * 1000))
            elif fn == "disable_roaming":
                line["roaming_enabled"] = False
            elif fn == "enable_roaming":
                line["roaming_enabled"] = True
            elif fn == "suspend_line_for_overdue_bill":
                plan = next(p for p in ents["plan"] if p["id"] == line["plan"])
                ents["bill"].append({
                    "id": args["new_bill_id"], "customer": "C1001",
                    "amount": plan["monthly_price"], "status": "Overdue",
                    "due": "2025-01-15"})
                line["status"] = "Suspended"
                if args["contract_ended"]:
                    line["contract_end"] = "2025-01-31"
            elif fn in ("set_user_info", "simulate_network_search"):
                pass
            else:
                raise Skip(f"unmapped telecom setup {fn}")
        return {"entities": ents, "outbox": [], "payments": []}, facts

    # ------------------------------------------------------------ phrases
    @staticmethod
    def on(b: bool) -> str:
        return "on" if b else "off"

    def phone_desc(self, p: dict) -> str:
        perms = []
        for k in ("sms", "storage"):
            perms.append(f"messaging {k} permission "
                         f"{'granted' if p[f'{k}_permission'] else 'denied'}")
        return (f"your phone: airplane mode {self.on(p['airplane_mode'])}, "
                f"network mode {p['network_mode']}, mobile data "
                f"{self.on(p['mobile_data'])}, data saver "
                f"{self.on(p['data_saver'])}, roaming switch "
                f"{self.on(p['roaming'])}, VPN "
                f"{'connected' if p['vpn'] else 'off'}, Wi-Fi calling "
                f"{self.on(p['wifi_calling'])}, SIM {p['sim']}, APN "
                f"{p['apn']}, " + ", ".join(perms)
                + (", you are abroad" if p["abroad"] else ""))

    @staticmethod
    def line_desc(state, line) -> str:
        plan = next(p for p in state["entities"]["plan"]
                    if p["id"] == line["plan"])
        return (f"line {line['id']} ({line['phone_number']}): "
                f"{line['status']}, {plan['name']}, "
                f"{line['data_used_mb'] / 1000:.1f} GB used of "
                f"{plan['data_limit_gb']} GB"
                + (f" (+{line['data_refuel_gb']} GB refuelled)"
                   if line["data_refuel_gb"] else "")
                + f", roaming {'enabled' if line['roaming_enabled'] else 'disabled'}"
                f", contract ends {line['contract_end']}")

    @staticmethod
    def bill_desc(b) -> str:
        return (f"bill {b['id']}: {dollars(b['amount'])}, {b['status']}, "
                f"due {b['due']}")

    # ------------------------------------------------------------ drafts
    def build(self, task: dict, state: dict, rng, gb_want: Optional[int] = None,
              only: Optional[List[str]] = None) -> Draft:
        d = Draft(self.world)
        d.state = state
        ents = state["entities"]
        cust = ents["customer"][0]
        phone = ents["phone"][0]
        line = next(x for x in ents["line"] if x["id"] == phone["line"])
        d.const("customer", "ID:customer", cust["id"],
                f"{cust['name']}, customer {cust['id']}")
        p_ref = d.const("phone", "ID:phone", phone["id"],
                        self.phone_desc(phone))
        l_ref = d.const("line", "ID:line", line["id"],
                        self.line_desc(state, line))
        for other in ents["line"]:
            d.const(f"line_{other['id']}", "ID:line", other["id"],
                    self.line_desc(state, other))
        for b in ents["bill"]:
            d.const(f"bill_{b['id']}", "ID:bill", b["id"], self.bill_desc(b))
        for m in SW.TELECOM_NETWORK_MODES:
            d.const(f"mode_{m}", "STR", m, f"network mode '{m}'",
                    "enum:phone.network_mode")
        app = d.const("app", "STR", "messaging", "the messaging app", "name")
        perm = {k: d.const(f"perm_{k}", "STR", k, f"the '{k}' permission",
                           "name") for k in ("sms", "storage")}
        d.const("app_camera", "STR", "camera", "the camera app", "name")
        instr = task["user_scenario"]["instructions"]
        text = (instr.get("task_instructions") or "") + " " + (
            instr.get("reason_for_call") or "")
        m = re.search(r"refuel (\d+(?:\.\d+)?) ?GB", text)
        gb = gb_want if gb_want is not None else (
            int(float(m.group(1))) if m else None)
        if gb:
            g_ref = d.const("gb", "INT", gb, f"{gb} GB of data, the most "
                            f"you'd pay for" if gb_want is None else
                            f"{gb} GB of data")
        d.const("gb_1", "INT", 1, "1 GB of data")
        lines, rng_ = [], rng
        new_bill = next((b for b in ents["bill"] if b["status"] == "Overdue"),
                        None)
        for a in task["evaluation_criteria"]["actions"] or []:
            n, args = a["name"], a["arguments"]
            if only is not None and n not in only:
                continue
            if n == "transfer_to_human_agents":
                raise Skip("transfer")
            shape = TELECOM_TOOL.get(n)
            if shape is None:
                raise Skip(f"unmapped telecom action {n}")
            if shape == "phone":
                lines.append(f"CALL @{n} {p_ref}\n")
            elif shape == "phone_mode":
                lines.append(f"CALL @{n} {p_ref} {{mode_{args['mode']}}}\n")
            elif shape == "phone_perm":
                lines.append(f"CALL @{n} {p_ref} {app} "
                             f"{perm[args['permission']]}\n")
            elif shape == "line_gb":
                if not gb:
                    raise Skip("refuel with no amount in the scenario")
                lines.append(f"CALL @{n} {l_ref} {g_ref}\n")
            elif shape == "line":
                lines.append(f"CALL @{n} {l_ref}\n")
            elif shape == "bill":
                lines.append(f"CALL @{n} {{bill_{new_bill['id']}}}\n")
        cat = task["id"][1:task["id"].index("]")]
        where = (f", abroad in {m.group(1)}" if (m := re.search(
            r"abroad in ([A-Z][a-z]+)", instr.get("known_info") or ""))
            else "")
        pay = (f" I'll pay for up to {gb} GB of extra data." if gb and
               gb_want is None else "")
        if gb_want is not None:
            pay = f" Please add {gb_want} GB of data to my line."
        d.request = (f"I'm {cust['name']} ({line['phone_number']}{where}); "
                     f"{rng_.choice(COMPLAINT[cat])}. Please fix it.{pay}")
        d.program = "".join(lines) + "STOP\n"
        d.prov["issues"] = task["id"][task["id"].index("]") + 1:
                                      task["id"].index("[PERSONA")].split("|")
        return d

    def tau2_task(self, run: "Run", task: dict, rng: random.Random) -> None:
        for task in [task]:
            prov = {"source_kind": "tau2", "tau2_task": task["id"],
                    "file": "data/tau2/domains/telecom/tasks.json (train)",
                    "recipe": "tau2_task", "teacher": "tau2-expected-actions"}
            try:
                state, _ = self.state_for(task)
            except Skip as e:
                run.dropped[_reason(str(e))] += 1
                continue
            names = [a["name"] for a in
                     task["evaluation_criteria"]["actions"] or []]
            if names == ["transfer_to_human_agents"]:
                # the SIM is PIN-locked or the contract ended: tau2's agent
                # hands over; ours declines, pointing at what blocks it
                t2 = dict(task, evaluation_criteria={"actions": []})
                d = self.build(t2, state, rng)
                why = ("phone" if "lock_sim_card_pin" in task["id"]
                       else "line")
                ab = as_abort(d, why)
                ab.prov.update(prov, variant="forbidden_tau2",
                               policy_reason="tau2 expects a transfer to a "
                                             "human agent")
                ab.tags |= {"service", "policy_forbidden", "perturb:none"}
                run.emit(ab, "forbid")
                continue
            try:
                d = self.build(task, state, rng)
            except Skip as e:
                run.dropped[_reason(str(e))] += 1
                continue
            d.prov.update(prov, variant="act", scope="whole")
            d.tags |= {"service", "act"}
            if run.emit(d, "act") is None:
                continue
            if "refuel_data" in names and rng.random() < 0.5:
                # the same outage, asking for more data than policy allows
                gb = rng.choice([3, 4, 5, 10])
                t2 = dict(task, evaluation_criteria={"actions": [
                    {"name": "refuel_data", "arguments": {}}]})
                w = self.build(t2, copy.deepcopy(state), rng, gb_want=gb)
                run.abort_only(w, "gb", dict(prov, perturbation="refuel_cap"),
                               expect_refusal=True)



# ================================================================ driver

class Run:
    """Collects the rows of one job (a tau2 task or a template draw) and the
    account of what was dropped and why. Jobs run in parallel threads (the
    work is node subprocesses) and are merged in job order, so a run is
    deterministic for a seed whatever the thread timing."""

    def __init__(self, dom, job: int, opts: dict):
        self.dom = dom
        self.job = job
        self.opts = opts
        self.rows: List[dict] = []
        self.dropped: Counter = Counter()
        self.k = 0

    def emit(self, draft: Draft, tag: str) -> Optional[dict]:
        self.k += 1
        tid = f"{draft.world}_{tag}_j{self.job}_{self.k}"
        o = self.opts
        try:
            row = finish(draft, tid, o["seed"] * 1_000_003 + self.job * 101
                         + self.k, symbols=o["symbols"], enums=o["enums"],
                         kinds=o["kinds"], decoys=o["decoys"])
        except Skip as e:
            self.dropped[_reason(str(e))] += 1
            return None
        self.rows.append(row)
        return row

    def group(self, uid: str, actions: List[tuple], ident: str,
              rng: random.Random, prov: dict, state: dict,
              natural_ok: bool = False) -> None:
        """The act row(s) for one customer's request, then the variants."""
        dom = self.dom
        singles = [[a] for a in actions]
        todo = ([actions] if len(actions) > 1 else []) + singles
        whole_ok = None
        for acts in todo:
            try:
                d = dom.build(uid, acts, random.Random(rng.random()), ident,
                              copy.deepcopy(state))
            except Skip as e:
                self.dropped[_reason(str(e))] += 1
                continue
            if d.program.lstrip().startswith("ABORT"):
                if len(acts) == 1:
                    d.status = "aborted"
                    d.prov.update(prov, variant="forbidden_no_tool",
                                  policy_reason="no tool performs this")
                    d.tags |= {"service", "policy_forbidden", "no_tool"}
                    self.emit(d, "forbid")
                continue
            d.prov.update(prov, variant="act",
                          scope="whole" if len(acts) > 1 else "single")
            d.tags |= {"service", "act"}
            # the reference run inside finish() is the check that the act
            # executes; only a failure costs a second sandbox run, to ask
            # whether the policy refused it
            row = self.emit(d, "act")
            if len(acts) > 1:
                whole_ok = row is not None
            if row is None:
                msg = refused(d)
                # "if_whole" (tau2 tasks): a write refused on its own is a
                # real refusal only when the task's whole sequence runs -
                # tau2's upgrade-then-cancel; if the whole fails too, our
                # engine and tau2's label disagree and the row is dropped
                allowed = (natural_ok is True or (
                    natural_ok == "if_whole" and whole_ok))
                if msg and allowed and len(acts) == 1:
                    self.dropped_last_as_natural()
                    ab = as_abort(d, dom.natural_ref(acts[0], msg))
                    ab.prov.update(variant="forbidden_natural",
                                   policy_reason=msg)
                    ab.tags = (ab.tags - {"act"}) | {"policy_forbidden",
                                                     "perturb:none"}
                    self.emit(ab, "forbid")
                elif msg and len(acts) == 1 and natural_ok == "if_whole":
                    self.dropped_last_as_natural()
                    self.dropped["tau2 write refused by our engine "
                                 "(label disagreement)"] += 1
                elif msg:
                    self.dropped_last_as_natural()
                    self.dropped["act refused by policy (" + (
                        "whole-task sequence" if len(acts) > 1 else
                        "single write; this domain keeps no natural "
                        "refusals") + ")"] += 1
                continue
            if len(acts) != 1:
                continue
            act = acts[0]
            if rng.random() < self.opts["forbid_share"]:
                self.forbidden(uid, act, ident, rng, prov, state)
            if rng.random() < self.opts["ask_share"]:
                self.ask(uid, act, ident, rng, prov, state)

    def dropped_last_as_natural(self) -> None:
        """The act that failed was a natural refusal, not a drop: take the
        drop just counted back."""
        for why in list(self.dropped):
            if why.startswith("reference:"):
                self.dropped[why] -= 1
                if self.dropped[why] <= 0:
                    del self.dropped[why]
                return

    def forbidden(self, uid, act, ident, rng, prov, state) -> None:
        out = self.dom.forbidden(uid, act, ident, random.Random(rng.random()),
                                 state)
        if out is None:
            self.dropped["no forbidden variant applies"] += 1
            return
        would, ref, how = out
        msg = refused(would)
        if not msg:
            self.dropped[f"forbidden variant not refused ({how})"] += 1
            return
        ab = as_abort(would, ref)
        ab.prov.update(prov, variant="forbidden", perturbation=how,
                       policy_reason=msg)
        ab.tags |= {"service", "policy_forbidden", f"perturb:{how}"}
        self.emit(ab, "forbid")

    def ask(self, uid, act, ident, rng, prov, state) -> None:
        out = self.dom.ask(uid, act, ident, random.Random(rng.random()), state)
        if out is None:
            return
        q, a = out
        q.prov.update(prov, variant="ask")
        q.tags |= {"service", "ask", "needs_info"}
        a.prov.update(prov, variant="answered")
        a.tags |= {"service", "act", "answered"}
        if self.emit(q, "ask") is not None:
            self.emit(a, "answered")

    def abort_only(self, draft: Draft, ref: str, prov: dict,
                   expect_refusal: bool) -> None:
        """A tau2 task whose point is a refusal. When there is a program to
        refuse, our engine must refuse it, or the row is dropped as a
        disagreement between the engine and tau2's labels."""
        if expect_refusal:
            msg = refused(draft)
            if not msg:
                self.dropped["tau2 refusal not refused by our engine"] += 1
                return
            prov = dict(prov, policy_reason=msg)
        ab = as_abort(draft, ref)
        ab.prov.update(prov, variant=prov.get("variant", "forbidden_tau2"))
        ab.tags |= {"service", "policy_forbidden", "perturb:none"}
        self.emit(ab, "forbid")


def _reason(text: str) -> str:
    """A drop reason with the per-row detail cut off, so they count."""
    text = re.sub(r"'[^']*'|\"[^\"]*\"", "'...'", text)
    text = re.sub(r"#W\d+|\b[A-Z0-9]{6}\b|\d{3,}", "N", text)
    return text[:120]


def _jobs_parallel(fns: List[Callable], workers: int) -> List[Run]:
    from concurrent.futures import ThreadPoolExecutor
    if workers <= 1:
        return [f() for f in fns]
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(lambda f: f(), fns))


def generate(domain: str, *, limit: int, templates: int, seed: int,
             symbols: str = "classic", enums: bool = False,
             kinds: bool = False, forbid_share: float = 0.6,
             ask_share: float = 0.3, decoys: Optional[tuple] = None,
             workers: int = 8, tau2_limit: Optional[int] = None,
             abort_cap: float = 0.4) -> dict:
    """Rows for one domain: every tau2 train task (or the first
    `tau2_limit`), then template draws in batches until `limit` rows or
    `templates` draws."""
    SW.register()
    dom = {"retail": Retail, "airline": Airline, "telecom": Telecom}[domain]()
    opts = dict(seed=seed, symbols=symbols, enums=enums, kinds=kinds,
                forbid_share=forbid_share, ask_share=ask_share,
                decoys=decoys)
    tasks = dom.tasks[:tau2_limit] if tau2_limit is not None else dom.tasks
    if not hasattr(dom, "template"):
        templates = 0          # the telecom exam is tau2's tasks only

    def tau2_job(i, task):
        def run_it():
            run = Run(dom, i, opts)
            dom.tau2_task(run, task, random.Random(seed * 7919 + i))
            return run
        return run_it

    def template_job(i):
        def run_it():
            run = Run(dom, i, opts)
            rng = random.Random(seed * 104729 + i)
            try:
                uid, actions, ident = dom.template(rng)
            except Skip as e:
                run.dropped[_reason(str(e))] += 1
                return run
            state = dom.base_state_for(uid, actions, rng)
            run.group(uid, actions, ident, rng,
                      {"source_kind": "template",
                       "recipe": f"tau2_template:{actions[0][0]}",
                       "file": f"data/borrowed/tau2/{domain}/db_extract.json",
                       "teacher": "tau2-template"},
                      state, natural_ok=getattr(dom, "NATURAL_REFUSALS",
                                                False))
            return run
        return run_it

    runs_ = _jobs_parallel([tau2_job(i, t) for i, t in enumerate(tasks)],
                           workers)
    rows = [r for run in runs_ for r in run.rows]
    dropped: Counter = Counter()
    for run in runs_:
        dropped.update(run.dropped)
    job = 10_000
    batch = max(8, workers * 4)
    while templates and len(rows) < limit and job - 10_000 < templates:
        fns = [template_job(job + k) for k in range(batch)]
        job += batch
        for run in _jobs_parallel(fns, workers):
            dropped.update(run.dropped)
            for row in run.rows:
                aborts = sum(r["expected_status"] == "aborted" for r in rows)
                forbidden = str(row["provenance"].get("variant", "")
                                ).startswith("forbidden")
                if (forbidden and abort_cap < 1
                        and aborts + 1 > abort_cap * (len(rows) + 1)):
                    # template refusals come cheap (most old bookings cannot
                    # be cancelled); past the cap they would teach "refuse"
                    # as the prior rather than as a decision
                    dropped["template ABORT row over --abort-cap"] += 1
                    continue
                rows.append(row)
    return {"rows": rows[:limit], "dropped": dropped}


def report(result: dict, out: Path) -> dict:
    rows = result["rows"]
    aborts = sum(r["expected_status"] == "aborted" for r in rows)
    toks = [brief_tokens(r["request"]) for r in rows]
    variants = Counter(r["provenance"].get("variant") for r in rows)
    sources = Counter(r["provenance"].get("source_kind") for r in rows)
    reasons = Counter(re.match(r"\s*ABORT (\w+)", r["reference"]["segments"][0])
                      .group(1) for r in rows
                      if r["expected_status"] == "aborted")
    print(f"wrote {len(rows)} rows -> {out}")
    print(f"  sources: {dict(sources)}")
    print(f"  variants: {dict(variants)}")
    print(f"  ABORT rows: {aborts} ({aborts / max(1, len(rows)):.1%}) "
          f"{dict(reasons)}")
    if toks:
        print(f"  brief tokens: min {min(toks)}, median "
              f"{sorted(toks)[len(toks) // 2]}, max {max(toks)} "
              f"(budget 128)")
    dropped = result["dropped"]
    if dropped:
        print(f"  dropped {sum(dropped.values())} drafts:")
        for why, n in dropped.most_common(25):
            print(f"    {n:4d}  {why}")
    return {"rows": len(rows), "aborts": aborts,
            "max_tokens": max(toks) if toks else 0}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="data.gen.service")
    ap.add_argument("--domain", choices=["retail", "airline", "telecom"])
    ap.add_argument("--out")
    ap.add_argument("--limit", type=int, default=400,
                    help="rows to write at most")
    ap.add_argument("--templates", type=int, default=None,
                    help="template draws on top of the tau2 tasks (default: "
                         "as many as --limit needs, up to 10x it; 0 for "
                         "tau2 tasks only)")
    ap.add_argument("--tau2-limit", type=int, default=None,
                    help="only the first N tau2 train tasks")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel sandbox threads")
    ap.add_argument("--symbols", default="classic",
                    choices=["classic", "typed"])
    ap.add_argument("--enums", action="store_true")
    ap.add_argument("--kinds", action="store_true")
    ap.add_argument("--decoys", default=None, metavar="MIN:MAX",
                    help="authored description-only siblings per tool "
                         "(harness/decoys.py), stored with the row's own "
                         "sandbox payload")
    ap.add_argument("--forbid-share", type=float, default=0.6,
                    help="share of permitted single-action rows that also "
                         "get a policy-forbidden twin")
    ap.add_argument("--ask-share", type=float, default=0.3,
                    help="share that also get an ask/answered pair")
    ap.add_argument("--abort-cap", type=float, default=0.4,
                    help="template ABORT rows are skipped while they would "
                         "push the file's ABORT share over this (tau2-task "
                         "rows are always kept)")
    ap.add_argument("--holdout", action="store_true",
                    help="allow the held-out telecom world (exam building)")
    ap.add_argument("--require-collisions", type=float, default=50,
                    metavar="PCT")
    ap.add_argument("--allow-signature-unique", action="store_true")
    ap.add_argument("--extract", metavar="TAU2_CHECKOUT",
                    help="re-vendor data/borrowed/tau2 from a checkout")
    args = ap.parse_args(argv)
    if args.extract:
        extract(Path(args.extract))
        return
    if not args.domain or not args.out:
        ap.error("--domain and --out are required")
    world = f"service_{args.domain}"
    if world in SW.HELD_OUT and not args.holdout:
        ap.error(f"{world} is held out; pass --holdout only to build an exam")
    if world not in SW.HELD_OUT and args.holdout:
        ap.error(f"{world} is trainable; --holdout is for the telecom exam")
    decoys = (tuple(int(x) for x in args.decoys.split(":"))
              if args.decoys else None)
    templates = (args.templates if args.templates is not None
                 else 10 * args.limit)
    result = generate(args.domain, limit=args.limit, templates=templates,
                      seed=args.seed, symbols=args.symbols, enums=args.enums,
                      kinds=args.kinds, forbid_share=args.forbid_share,
                      ask_share=args.ask_share, decoys=decoys,
                      workers=args.workers, tau2_limit=args.tau2_limit,
                      abort_cap=args.abort_cap)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for row in result["rows"]:
            if args.allow_signature_unique:
                row["provenance"]["signature_unique_allowed"] = True
            fh.write(json.dumps(row) + "\n")
    report(result, out)
    from data.gen.__main__ import report_signature_uniqueness
    report_signature_uniqueness(out, args.require_collisions,
                                args.allow_signature_unique)


if __name__ == "__main__":
    main()
