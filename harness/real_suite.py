"""E-real-sessions: the frozen real-request slice as an eval suite (plan
.claude/plans/real-sessions-eval-suite.md).

Real turns carry no reference program, so the suite scores *routing*: did
the planner call the same course-builder tools the frontier agent called,
and did it abstain when the frontier called nothing. Rows come from
data/real_sessions/eval_candidates.jsonl (strict-clean requests + frontier
tool names) and run against the `coursebuilder` world.

  python -m harness.real_suite build                  # -> data/holdout/e_real_sessions.jsonl
  python -m baselines.qwen.run_a --model ... --tasks data/holdout/e_real_sessions.jsonl --out results/...jsonl
  python -m harness.real_suite score results/...jsonl # routing metrics over the run

Name map: the frontier tool names (May–Sep 2026, several agents) -> the
world's tool names. Names that map to None are UI or crawl tools with no
planner equivalent; they drop out of the expected set.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CANDIDATES = ROOT / "data" / "real_sessions" / "eval_candidates.jsonl"
OUT = ROOT / "data" / "holdout" / "e_real_sessions.jsonl"

# frontier name -> world tool (None = no planner equivalent)
NAME_MAP = {
    # reads
    "get_course_outline": "get_course_outline",
    "get_module_elements": "list_elements",
    "get_element_details": "get_element",
    "get_sub_items": "list_sub_items",
    "get_element_schema": "get_element_schema",
    "get_course_theme": "get_course_theme",
    "get_course_styles": "get_course_theme",
    "get_available_fonts": "get_available_fonts",
    "get_course_metadata": "get_course_metadata",
    "find_courses": "find_courses",
    "get_module_from_course": "get_course_outline",
    "get_element_from_course": "get_element",
    "list_notebooks": "search_help",
    "list_sources": "search_help",
    "get_document": "search_help",
    "query_knowledge_base_raw": "search_help",
    "query_notebook_raw": "search_help",
    # writes
    "add_module": "add_module",
    "rename_module": "rename_module",
    "delete_module": "delete_module",
    "add_element": "add_element",
    "update_element": "update_element",
    "batch_update_elements": "update_element",
    "delete_element": "delete_element",
    "move_element": "move_element",
    "duplicate_element": "duplicate_element",
    "transform_element": "transform_element",
    "replace_element": "transform_element",      # renamed (MOBI-TRANSFORM-ELEMENT)
    "merge_elements": "update_element",
    "add_sub_item": "add_sub_item",
    "update_sub_item": "update_sub_item",
    "batch_update_sub_items": "update_sub_item",
    "delete_sub_item": "delete_sub_item",
    "update_course_metadata": "update_course_metadata",
    "update_course_styles": "update_course_styles",
    "apply_to_lessons": "apply_to_lessons",
    "generate_lesson_content": "generate_lesson_content",
    "course_search_and_create": "course_search_and_create",
    "soco_create_courses": "course_search_and_create",
    "start_free_trial": "start_free_trial",
    "submit_contact_form": "submit_contact_form",
    "contact_us": "submit_contact_form",
    "show_contact_form": "submit_contact_form",
    "export_document": "export_document",
    "export_spreadsheet": "export_document",
    "download_file": "export_document",           # retired 2026-08-24
    "download_conversation": "export_document",
    "enhance_text": "write_text",
    # generative
    "generate_image": "generate_image",
    "generate_image_variants": "generate_image",
    "edit_image": "edit_image",
    "analyze_image": "get_element",
    # curriculum agent (mobi--curriculum-)
    "add_course_to_curriculum": "add_course_to_curriculum",
    "find_courses_for_curriculum": "find_courses",
    "get_curriculum_contents": "get_course_outline",
    "update_curriculum_metadata": "update_course_metadata",
    "update_curriculum_settings": "update_course_metadata",
    "get_course_settings": "get_course_metadata",
    # UI / crawl / account / assignment rules: no planner equivalent
    "navigate_to_module": None,
    "get_assignment_rules": None,
    "find_learners_for_assignment": None,
    "select_element": None,
    "show_calendar_widget": None,
    "get_user_info": None,
    "open_tool_library": None,
    "fetch_page": None,
    "save_page": None,
    "finish_crawl": None,
    "list_site_urls": None,
    "get_assignment_rule_values": None,
}
# Knowledge bases are named by the user (sanitize_name(kb.name) in
# lm-python-functions/agent/knowledge_base_tool.py): LEARNER_HOW_TO,
# railway_notebook, test_notebook_5 … Anything not in the map and not shaped
# like a CRUD/UI tool is one of them.
_CRUD_SHAPE = re.compile(r"^(get|list|add|update|delete|remove|move|show|find|open|select|"
                         r"navigate|set|create|rename|duplicate|transform|merge|apply|generate|"
                         r"edit|export|download|submit|start|fetch|save|finish|batch|query)_")

READ_TOOLS = {
    "list_modules", "get_module", "list_elements", "get_element", "list_sub_items",
    "get_course_theme", "get_available_fonts", "get_element_schema",
    "get_course_metadata", "find_courses", "get_course_outline", "search_help",
}
WRITE_TOOLS_EXTRA = {"add_course_to_curriculum"}  # world tools with no CRUD-shaped frontier twin
CONTENT_TOOLS = {"write_text", "generate_image", "edit_image"}


def map_name(frontier: str) -> str | None:
    if frontier in NAME_MAP:
        return NAME_MAP[frontier]
    if frontier and not _CRUD_SHAPE.match(frontier):
        return "search_help"       # a user-named knowledge base
    return None


def expected_tools(frontier_calls: list) -> list:
    """Mapped, de-duplicated, order-preserving world tool names."""
    out = []
    for n in frontier_calls:
        m = map_name(n)
        if m and m not in out:
            out.append(m)
    return out


def score_row(row: dict, task: dict) -> dict:
    """Routing metrics for one run row against its task's expectations."""
    called = [c["name"] for c in row.get("calls", []) if c.get("ok", True)]
    called_set = set(called)
    exp = set(task["expected_tools"])
    exp_status = task.get("expected_status", "ok")
    aborted = row["status"] == "aborted"
    unmapped = exp_status == "ok" and not exp
    out = {
        "task_id": row["task_id"],
        "compile_ok": row["compile_ok"],
        "unmapped": unmapped,
        "correct_abstain": (aborted and row.get("abort_reason") != "NOT_FOUND")
        if exp_status == "aborted" else (not aborted),
        "route_match": None, "route_write_match": None, "content_routed": None,
        "unnecessary_destructive": row.get("unnecessary_destructive", 0),
    }
    if exp_status == "ok" and not unmapped:
        # a program that did not compile or run routed nothing, even when the
        # expected write set is empty (read-only turns)
        ran = row["compile_ok"] and row["status"] in ("ok", "effect_blocked")
        out["route_match"] = ran and called_set == exp
        out["route_write_match"] = ran and (called_set - READ_TOOLS) == (exp - READ_TOOLS)
    if task.get("content_bearing"):
        out["content_routed"] = bool(called_set & CONTENT_TOOLS)
    return out


def summarize(scored: list) -> dict:
    def rate(key, subset=None):
        vals = [s[key] for s in (subset or scored) if s[key] is not None]
        return (sum(vals) / len(vals), len(vals)) if vals else (None, 0)
    return {
        "n": len(scored),
        "unmapped": sum(s["unmapped"] for s in scored),
        "compile_ok": rate("compile_ok"),
        "route_match": rate("route_match"),
        "route_write_match": rate("route_write_match"),
        "correct_abstain": rate("correct_abstain"),
        "content_routed": rate("content_routed"),
        "unnecessary_destructive_total": sum(s["unnecessary_destructive"] for s in scored),
    }


def cmd_score(args) -> None:
    tasks = {t["id"]: t for t in (json.loads(l) for l in open(args.tasks, encoding="utf-8"))}
    rows = [json.loads(l) for l in open(args.run, encoding="utf-8")]
    scored = [score_row(r, tasks[r["task_id"]]) for r in rows if r["task_id"] in tasks]
    summ = summarize(scored)
    print(json.dumps(summ, indent=1))
    by = collections.defaultdict(list)
    for s, r in zip(scored, rows):
        by[tasks[r["task_id"]].get("routing_primary", "?")].append(s)
    for k, v in sorted(by.items(), key=lambda kv: -len(kv[1])):
        m = summarize(v)
        print(f"  {k:10s} n={m['n']:3d} compile={m['compile_ok'][0]:.2f} "
              f"route_write={m['route_write_match'][0] if m['route_write_match'][0] is not None else float('nan'):.2f} "
              f"abstain={m['correct_abstain'][0]:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="harness.real_suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=20260902)
    b.add_argument("--symbols", choices=["classic", "typed"],
                   default="classic",
                   help="0.4.0 typed constant letters (spec §2.1)")
    b.add_argument("--enums", action="store_true",
                   help="emit the schema's enum values as constants (§2.3)")
    b.add_argument("--kinds", action="store_true",
                   help="declare string kinds on STR constants (§2.2)")
    s = sub.add_parser("score")
    s.add_argument("run")
    s.add_argument("--tasks", default=str(OUT))
    args = ap.parse_args()
    if args.cmd == "score":
        cmd_score(args)
    else:
        from harness.real_suite_build import cmd_build  # depends on the coursebuilder world
        cmd_build(args)


if __name__ == "__main__":
    main()
