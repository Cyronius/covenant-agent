"""The customer-service worlds borrowed from tau2-bench (runtime/worlds/
service.py, runtime/engines/service.js, data/gen/service.py).

A small generation per domain is replayed through the real harness: every
reference executes and reaches its expected state, every policy-forbidden
variant aborts, every brief fits the tiny planner's budget. The engine's
policy rules are then checked one call at a time.
"""
import copy
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from core.pipeline import build
from data.gen import service as G
from data.gen.brief_budget import check_brief
from harness.abort_check import check_abort
from harness.authoring import resolve
from harness.context import build_context
from harness.run import reference_planner, run_sandbox, run_task
from core.ir import TaskContext
from runtime.worlds import service as SW

SW.register()


@pytest.fixture(scope="module")
def rows():
    out = {}
    common = dict(seed=11, symbols="typed", enums=True, kinds=True,
                  workers=8, forbid_share=1.0, ask_share=0.5)
    out["retail"] = G.generate("retail", limit=40, templates=8, tau2_limit=6,
                               decoys=(1, 2), **common)["rows"]
    out["airline"] = G.generate("airline", limit=40, templates=16,
                                tau2_limit=10, **common)["rows"]
    out["telecom"] = G.generate("telecom", limit=12, templates=0,
                                tau2_limit=8, **common)["rows"]
    return out


def _all(rows):
    return [r for dom in rows.values() for r in dom]


def test_every_domain_yields_act_and_abort_rows(rows):
    for dom, rs in rows.items():
        assert any(r["expected_status"] == "ok" for r in rs), dom
        assert any(r["expected_status"] == "aborted" for r in rs), dom


def test_references_replay_to_expected_state(rows):
    """The stored reference, run through harness.run with the row's own
    context, reaches expected_state with the expected status."""
    tasks = _all(rows)
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(lambda t: run_task(t, reference_planner(t)),
                              tasks))
    bad = [(t["id"], m["status"]) for t, m in zip(tasks, results)
           if not m["goal_success"]]
    assert not bad, bad[:5]


def test_forbidden_variants_abort_with_a_named_rule(rows):
    forbidden = [r for r in _all(rows) if "policy_forbidden" in r["tags"]]
    assert forbidden
    for r in forbidden:
        assert r["expected_status"] == "aborted"
        assert r["reference"]["segments"][0].startswith("ABORT UNSUPPORTED")
        reason = r["provenance"].get("policy_reason", "")
        # the engine named the rule, or no tool performs the request
        assert reason.startswith("policy:") or "no tool" in reason \
            or "transfer" in reason, (r["id"], reason)


def test_abort_referents_are_founded(rows):
    for r in _all(rows):
        if r["expected_status"] != "aborted":
            continue
        ctx = TaskContext.from_json(r["context"])
        head = r["reference"]["segments"][0].split()
        assert check_abort(ctx, r["state"], head[1], head[2:]) is None, r["id"]


def test_briefs_fit_the_tiny_planner(rows):
    for r in _all(rows):
        check_brief(r["request"])
        assert r["input_text"].startswith("REQUEST: ")


def test_provenance_is_train_split_only(rows):
    train = {d: {t["id"] for t in G.load(d, "tasks_train.json")}
             for d in ("retail", "airline", "telecom")}
    for dom, rs in rows.items():
        for r in rs:
            p = r["provenance"]
            assert p["source_commit"] == G.TAU2_COMMIT and p["licence"] == "MIT"
            if p.get("source_kind") == "tau2":
                assert p["tau2_task"] in train[dom]


def test_telecom_is_held_out():
    assert "service_telecom" in SW.HELD_OUT
    with pytest.raises(SystemExit):
        G.main(["--domain", "telecom", "--out", "unused.jsonl"])


# ------------------------------------------------------------ engine rules

def _call(world_name: str, state: dict, tool: str, args: list) -> dict:
    """One CALL of `tool` with constant arguments, run in the sandbox."""
    world = SW.RETAIL if world_name == "retail" else (
        SW.AIRLINE if world_name == "airline" else SW.TELECOM)
    spec = next(t for t in world["tools"] if t["name"] == tool)
    consts = [{"type": p["type"], "value": v, "desc": p["name"]}
              for p, v in zip(spec["params"], args)]
    ctx, sb = build_context(world, consts, random.Random(0))
    prog = resolve(f"CALL @{tool} " + " ".join(f"${i}" for i in
                                               range(len(args))) + "\nSTOP\n",
                   ctx)
    res = build(prog, ctx)
    assert res.compile_ok, res.rendered_diagnostics()
    return run_sandbox({"js": res.js, "state": copy.deepcopy(state),
                        "tools": sb["tools"], "fields": sb["fields"],
                        "constants": sb["constants"], "now": world["now"],
                        "approval": True, "error_injection": [],
                        "initial_registers": {}})


def _retail_state():
    return {"entities": {
        "user": [{"id": "u1", "first_name": "A", "last_name": "B",
                  "email": "a@x", "address1": "", "address2": "",
                  "city": "", "state": "", "zip": "1"}],
        "payment_method": [
            {"id": "cc1", "user": "u1", "kind": "credit_card", "balance": 0,
             "last_four": "1111"},
            {"id": "cc2", "user": "u1", "kind": "credit_card", "balance": 0,
             "last_four": "2222"},
            {"id": "gc1", "user": "u1", "kind": "gift_card", "balance": 500,
             "last_four": ""}],
        "product": [{"id": "p1", "name": "Lamp"}, {"id": "p2", "name": "Mug"}],
        "variant": [
            {"id": "v1", "product": "p1", "options": "color: red",
             "available": True, "price": 1000},
            {"id": "v2", "product": "p1", "options": "color: blue",
             "available": True, "price": 3000},
            {"id": "v3", "product": "p2", "options": "size: L",
             "available": True, "price": 900}],
        "order": [
            {"id": "#W1", "user": "u1", "status": "pending", "address1": "",
             "address2": "", "city": "", "state": "", "zip": "",
             "payment_method": "gc1", "total": 1000, "cancel_reason": None,
             "locked": False},
            {"id": "#W2", "user": "u1", "status": "delivered", "address1": "",
             "address2": "", "city": "", "state": "", "zip": "",
             "payment_method": "cc1", "total": 1000, "cancel_reason": None,
             "locked": False}],
        "order_item": [
            {"id": "#W1/v1", "order": "#W1", "product": "p1", "variant": "v1",
             "name": "Lamp", "price": 1000, "status": "ordered",
             "new_variant": None, "payment_method": None},
            {"id": "#W2/v1", "order": "#W2", "product": "p1", "variant": "v1",
             "name": "Lamp", "price": 1000, "status": "ordered",
             "new_variant": None, "payment_method": None}]},
        "outbox": [], "payments": []}


def test_retail_cancel_rules():
    st = _retail_state()
    r = _call("retail", st, "cancel_pending_order", ["#W2", "no longer needed"])
    assert r["error"]["code"] == "PERMISSION_DENIED"
    assert "only a pending order can be cancelled" in r["error"]["message"]
    r = _call("retail", st, "cancel_pending_order",
              ["#W1", "found it cheaper"])
    assert "'no longer needed' or 'ordered by mistake'" in r["error"]["message"]
    r = _call("retail", st, "cancel_pending_order", ["#W1", "ordered by mistake"])
    assert r["status"] == "ok"
    order = next(o for o in r["state"]["entities"]["order"] if o["id"] == "#W1")
    gift = next(p for p in r["state"]["entities"]["payment_method"]
                if p["id"] == "gc1")
    assert order["status"] == "cancelled" and gift["balance"] == 1500


def test_retail_refund_and_exchange_rules():
    st = _retail_state()
    r = _call("retail", st, "return_delivered_order_items", ["#W2/v1", "cc2"])
    assert "original payment method or to a gift card" in r["error"]["message"]
    assert _call("retail", st, "return_delivered_order_items",
                 ["#W2/v1", "gc1"])["status"] == "ok"
    r = _call("retail", st, "exchange_delivered_order_items",
              ["#W2/v1", "v3", "cc1"])
    assert "same product" in r["error"]["message"]
    # $20 more than the item, a $5 gift card cannot cover it
    r = _call("retail", st, "exchange_delivered_order_items",
              ["#W2/v1", "v2", "gc1"])
    assert "does not cover the price difference" in r["error"]["message"]
    r = _call("retail", st, "modify_pending_order_items",
              ["#W1/v1", "v2", "cc1"])
    assert r["status"] == "ok"
    o = next(o for o in r["state"]["entities"]["order"] if o["id"] == "#W1")
    assert o["status"] == "pending (item modified)" and o["total"] == 3000


def _airline_state(cabin="economy", created_days=10, insurance=False,
                   membership="gold", flight_status="available"):
    now = SW.AIRLINE_NOW
    return {"entities": {
        "user": [{"id": "u1", "first_name": "A", "last_name": "B",
                  "email": "a@x", "membership": membership}],
        "payment_method": [{"id": "cc1", "user": "u1", "kind": "credit_card",
                            "balance": 0, "last_four": "1111"},
                           {"id": "cert1", "user": "u1", "kind": "certificate",
                            "balance": 10000, "last_four": ""}],
        "reservation": [{"id": "R1", "user": "u1", "origin": "JFK",
                         "destination": "SFO", "trip_type": "one_way",
                         "cabin": cabin, "passenger_count": 2,
                         "total_baggages": 1, "nonfree_baggages": 0,
                         "insurance": insurance,
                         "created": now - created_days * 86400,
                         "status": "active", "payment_method": "cc1",
                         "cancel_reason": None, "changed": False}],
        "segment": [{"id": "R1-1", "reservation": "R1",
                     "flight": "HAT1@2024-05-20", "origin": "JFK",
                     "destination": "SFO", "date": "2024-05-20",
                     "price": 10000}],
        "flight": [{"id": "HAT1@2024-05-20", "number": "HAT1",
                    "origin": "JFK", "destination": "SFO",
                    "date": "2024-05-20", "status": flight_status,
                    "departs": "08:00", "price_basic_economy": 5000,
                    "price_economy": 10000, "price_business": 30000}],
        "passenger": []}, "outbox": [], "payments": []}


def test_airline_cancellation_policy():
    r = _call("airline", _airline_state(), "cancel_reservation",
              ["R1", "change of plan"])
    assert "within 24 hours" in r["error"]["message"]
    for ok in (_airline_state(created_days=0),
               _airline_state(cabin="business"),
               _airline_state(insurance=True)):
        reason = "health" if ok["entities"]["reservation"][0]["insurance"] \
            else "change of plan"
        assert _call("airline", ok, "cancel_reservation",
                     ["R1", reason])["status"] == "ok"
    r = _call("airline", _airline_state(insurance=True), "cancel_reservation",
              ["R1", "other"])
    assert r["error"]["code"] == "PERMISSION_DENIED"
    r = _call("airline", _airline_state(cabin="business",
                                        flight_status="landed"),
              "cancel_reservation", ["R1", "change of plan"])
    assert "already been flown" in r["error"]["message"]


def test_airline_bags_and_payment_rules():
    # gold economy: 3 free per passenger, 2 passengers -> 6 free
    r = _call("airline", _airline_state(), "update_reservation_baggages",
              ["R1", 6, "cc1"])
    res = r["state"]["entities"]["reservation"][0]
    assert r["status"] == "ok" and res["nonfree_baggages"] == 0
    r = _call("airline", _airline_state(membership="regular"),
              "update_reservation_baggages", ["R1", 3, "cc1"])
    assert r["state"]["payments"] == [{"payment_method": "cc1",
                                       "amount": 5000}]
    r = _call("airline", _airline_state(), "update_reservation_baggages",
              ["R1", 0, "cc1"])
    assert "added but not removed" in r["error"]["message"]
    r = _call("airline", _airline_state(), "update_reservation_cabin",
              ["R1", "business", "cert1"])
    assert "certificate cannot pay" in r["error"]["message"]


def test_telecom_refuel_and_resume_rules():
    st = {"entities": {
        "customer": [{"id": "C1", "name": "J", "phone_number": "5",
                      "dob": "", "status": "Active"}],
        "plan": [], "phone": [{"id": "ph", "line": "L1", "needs_reboot": False,
                               "apn": "default"}],
        "line": [{"id": "L1", "customer": "C1", "phone_number": "5",
                  "status": "Suspended", "plan": "P", "data_used_mb": 0,
                  "data_refuel_gb": 0, "roaming_enabled": False,
                  "contract_end": "2026-12-31"}],
        "bill": [{"id": "B1", "customer": "C1", "amount": 6500,
                  "status": "Overdue", "due": "2025-01-15"}]},
        "outbox": [], "payments": []}
    r = _call("telecom", st, "resume_line", ["L1"])
    assert "overdue bill must be paid" in r["error"]["message"]
    paid = copy.deepcopy(st)
    paid["entities"]["bill"][0]["status"] = "Paid"
    r = _call("telecom", paid, "resume_line", ["L1"])
    assert r["status"] == "ok"
    assert r["state"]["entities"]["phone"][0]["needs_reboot"] is True
    ended = copy.deepcopy(paid)
    ended["entities"]["line"][0]["contract_end"] = "2025-01-31"
    assert "contract ended" in _call("telecom", ended, "resume_line",
                                     ["L1"])["error"]["message"]
    active = copy.deepcopy(paid)
    active["entities"]["line"][0]["status"] = "Active"
    assert "at most 2 GB" in _call("telecom", active, "refuel_data",
                                   ["L1", 3])["error"]["message"]
    assert _call("telecom", active, "refuel_data",
                 ["L1", 2])["status"] == "ok"
