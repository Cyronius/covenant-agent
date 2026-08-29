"""Random world states (F4): same schemas as runtime/worlds, randomized
records. Deterministic given the RNG."""
from __future__ import annotations

import random

DAY = 86400

FIRST_NAMES = ["Ada", "Bob", "Carol", "Dev", "Elif", "Femi", "Gus", "Hana",
               "Ivan", "June", "Kai", "Lena", "Mo", "Nia", "Otto", "Priya",
               "Quinn", "Rosa", "Sam", "Ty", "Uma", "Vik", "Wren", "Yara"]
LAST_NAMES = ["Adler", "Brooks", "Chen", "Diaz", "Endo", "Farah", "Gray",
              "Hale", "Ito", "Jonas", "Kerr", "Lund", "Mora", "Nakai",
              "Ortiz", "Patel", "Quist", "Reyes", "Sato", "Tran"]
CARD_TITLES = ["Fix login bug", "Write Q3 report", "Renew TLS cert",
               "Update onboarding doc", "Plan offsite", "Migrate database",
               "Refactor billing", "Design landing page", "Audit permissions",
               "Ship mobile beta", "Clean up backlog", "Review vendor quote",
               "Draft press release", "Patch CVE", "Tune search ranking"]
COMPANY_NAMES = ["Northwind", "Initech", "Globex", "Umbrella", "Hooli",
                 "Stark Labs", "Wayne Tech", "Acme", "Soylent", "Vandelay",
                 "Wonka Ops", "Tyrell", "Cyberdyne", "Aperture", "Oscorp"]
PROJECT_NAMES = ["Website refresh", "Mobile app", "Data warehouse",
                 "Legacy CMS", "Billing rewrite", "Search upgrade",
                 "Analytics dashboard", "Partner portal", "Auth overhaul",
                 "Design system"]


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _email(name: str, domain: str) -> str:
    return name.split()[0].lower() + "@" + domain


def gen_kanban_state(rng: random.Random, now: int) -> dict:
    n_users = rng.randint(2, 4)
    users = []
    used = set()
    while len(users) < n_users:
        name = _person(rng)
        if name in used:
            continue
        used.add(name)
        uid = f"user_{len(users) + 1}"
        users.append({"id": uid, "name": name,
                      "email": _email(name, "acme.test")})
    n_cards = rng.randint(4, 9)
    titles = rng.sample(CARD_TITLES, n_cards)
    cards = []
    for i in range(n_cards):
        cards.append({
            "id": f"card_{i + 1}",
            "title": titles[i],
            "status": rng.choice(["todo", "doing", "done"]),
            "assignee": rng.choice(users)["id"],
            "due": now + rng.randint(-20, 20) * DAY,
            "urgent": rng.random() < 0.3,
            "archived": rng.random() < 0.2,
            "created": now - rng.randint(5, 60) * DAY,
        })
    return {"entities": {"user": users, "card": cards},
            "outbox": [], "payments": []}


def gen_crm_state(rng: random.Random, now: int) -> dict:
    staff = []
    used = set()
    while len(staff) < rng.randint(2, 3):
        name = _person(rng)
        if name in used:
            continue
        used.add(name)
        staff.append({"id": f"user_{len(staff) + 1}", "name": name,
                      "email": _email(name, "corp.test")})
    customers = []
    names = rng.sample(COMPANY_NAMES, rng.randint(3, 6))
    for i, cname in enumerate(names):
        customers.append({
            "id": f"customer_{i + 1}", "name": cname,
            "email": "ops@" + cname.split()[0].lower() + ".test",
            "manager": rng.choice(staff)["id"],
            "delinquent": rng.random() < 0.3,
            "plan": rng.choice(["basic", "pro", "enterprise"]),
            "signup": now - rng.randint(30, 900) * DAY,
        })
    tickets = []
    for i in range(rng.randint(3, 8)):
        tickets.append({
            "id": f"ticket_{i + 1}",
            "customer": rng.choice(customers)["id"],
            "status": rng.choice(["open", "open", "closed"]),
            "priority": rng.randint(1, 3),
            "opened": now - rng.randint(1, 90) * DAY,
        })
    invoices = []
    for i in range(rng.randint(2, 6)):
        invoices.append({
            "id": f"invoice_{i + 1}",
            "customer": rng.choice(customers)["id"],
            "amount": rng.randint(20, 2000) * 100,
            "paid": rng.random() < 0.5,
            "due": now + rng.randint(-45, 45) * DAY,
        })
    return {"entities": {"user": staff, "customer": customers,
                         "ticket": tickets, "invoice": invoices},
            "outbox": [], "payments": []}


def gen_projects_state(rng: random.Random, now: int) -> dict:
    members = []
    used = set()
    n_members = rng.randint(3, 5)
    while len(members) < n_members:
        name = _person(rng)
        if name in used:
            continue
        used.add(name)
        members.append({"id": f"user_{len(members) + 1}", "name": name,
                        "email": _email(name, "studio.test"),
                        "active": rng.random() < 0.7})
    for m in members:
        m["manager"] = rng.choice(members)["id"]
    projects = []
    names = rng.sample(PROJECT_NAMES, rng.randint(3, 6))
    for i, pname in enumerate(names):
        projects.append({
            "id": f"project_{i + 1}", "name": pname,
            "owner": rng.choice(members)["id"],
            "status": rng.choice(["active", "active", "archived"]),
            "last_activity": now - rng.randint(1, 500) * DAY,
        })
    return {"entities": {"user": members, "project": projects},
            "outbox": [], "payments": []}


GENERATORS = {
    "kanban": gen_kanban_state,
    "crm": gen_crm_state,
    "projects": gen_projects_state,
}


def gen_state(world_name: str, rng: random.Random, now: int) -> dict:
    return GENERATORS[world_name](rng, now)
