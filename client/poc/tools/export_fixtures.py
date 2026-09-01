"""Export a small, diverse set of curriculum tasks as browser-POC fixtures.

  python client/poc/tools/export_fixtures.py

Reads data/curriculum_tasks.jsonl, picks a handful of tasks spanning
difficulty levels, and writes client/poc/fixtures/tasks.json — each entry
carrying only what the browser client needs to build a prompt
(task_id/level/world/request/input_text). The reference program (answer key)
and world `state` are deliberately NOT included here: they stay server-side,
looked up by task_id by the (separately-owned) validate server.

Restriction: this POC's first cut only covers single-segment tasks, i.e.
tasks whose reference program never hits PAUSE and needs a second
model-generated segment. Multi-segment tasks (the L10_* PAUSE/continuation
tasks in the curriculum set) are skipped here; round-tripping a PAUSE through
the browser client is out of scope for this pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from core.ir import TaskContext  # noqa: E402
from harness.context import serialize_context  # noqa: E402

TASKS_PATH = ROOT / "data" / "curriculum_tasks.jsonl"
OUT_PATH = ROOT / "client" / "poc" / "fixtures" / "tasks.json"

# Hand-picked for diversity across difficulty levels while keeping the set
# small: a direct call (L0), a chained lookup (L1), boolean-composed
# filtering (L2 and L3, per the task's requested coverage), an aggregation
# with running-max tracking (L4), a nested-filter multi-step write (L6),
# PARALLEL CALLs with a conditional follow-up (L7), and TRY/IF/ELSE
# error-fallback (L8). All are single-segment (see module docstring).
SELECTED_IDS = [
    "L0_kanban_delete",
    "L1_crm_email_lookup",
    "L2_crm_close_customer_tickets",
    "L3_kanban_bob_overdue",
    "L4_crm_most_open_tickets",
    "L6_projects_transfer_inactive",
    "L7_kanban_parallel_check",
    "L8_crm_fallback_email",
]


def load_tasks(path: Path) -> dict:
    tasks = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            t = json.loads(line)
            tasks[t["id"]] = t
    return tasks


def main():
    tasks = load_tasks(TASKS_PATH)

    fixtures = []
    for task_id in SELECTED_IDS:
        task = tasks[task_id]
        n_segments = len(task["reference"]["segments"])
        if n_segments > 1:
            raise ValueError(
                f"{task_id} has {n_segments} reference segments (needs a "
                "PAUSE round-trip); this POC's first cut is single-segment "
                "tasks only")
        ctx = TaskContext.from_json(task["context"])
        input_text = serialize_context(task["request"], ctx)
        fixtures.append({
            "task_id": task["id"],
            "level": task["level"],
            "world": task["world"],
            "request": task["request"],
            "input_text": input_text,
        })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(fixtures, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(fixtures)} fixtures to {OUT_PATH}")


if __name__ == "__main__":
    main()
