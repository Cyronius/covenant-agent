"""Routing suite: name map coverage, expected sets, scorer (plan
.claude/plans/real-sessions-eval-suite.md §1, §3)."""
import json
from pathlib import Path

import pytest

from harness.real_suite import (READ_TOOLS, expected_tools, map_name, score_row,
                                summarize)

INVENTORY = Path(__file__).resolve().parent.parent / "data" / "real_sessions" / "tool_inventory.json"


def test_every_frequent_frontier_tool_is_mapped_or_deliberately_dropped():
    # the inventory is gitignored; skip when the pull is not on this machine
    if not INVENTORY.exists():
        pytest.skip("no real-session inventory here")
    inv = json.loads(INVENTORY.read_text(encoding="utf-8"))
    from harness.real_suite import NAME_MAP
    unmapped = [n for n, e in inv.items() if e["turns"] >= 3
                and n not in NAME_MAP and map_name(n) is None]
    assert unmapped == [], unmapped


def test_knowledge_bases_map_to_search_help():
    assert map_name("LEARNER_HOW_TO") == "search_help"
    assert map_name("How_it_s_made") == "search_help"
    assert map_name("query_knowledge_base_raw") == "search_help"


def test_renamed_and_batch_tools_fold():
    assert expected_tools(["get_course_theme", "batch_update_elements", "batch_update_elements"]) \
        == ["get_course_theme", "update_element"]
    assert expected_tools(["replace_element"]) == ["transform_element"]
    assert expected_tools(["select_element", "fetch_page"]) == []


def _row(calls, status="ok", reason=None, compile_ok=True):
    return {"task_id": "t", "status": status, "abort_reason": reason, "compile_ok": compile_ok,
            "calls": [{"name": c, "ok": True} for c in calls], "unnecessary_destructive": 0}


def test_route_match_ignores_reads_only_in_the_write_variant():
    task = {"expected_tools": ["get_course_theme", "add_element"], "expected_status": "ok"}
    s = score_row(_row(["add_element"]), task)
    assert s["route_match"] is False and s["route_write_match"] is True
    s = score_row(_row(["get_course_theme", "add_element"]), task)
    assert s["route_match"] is True


def test_abstain_scoring():
    task = {"expected_tools": [], "expected_status": "aborted"}
    assert score_row(_row([], "aborted", "AMBIGUOUS"), task)["correct_abstain"] is True
    assert score_row(_row([], "aborted", "NOT_FOUND"), task)["correct_abstain"] is False
    assert score_row(_row(["add_element"]), task)["correct_abstain"] is False
    task = {"expected_tools": ["add_element"], "expected_status": "ok"}
    assert score_row(_row([], "aborted", "AMBIGUOUS"), task)["correct_abstain"] is False


def test_unmapped_turns_are_excluded_from_routing():
    task = {"expected_tools": [], "expected_status": "ok"}
    s = score_row(_row(["add_element"]), task)
    assert s["unmapped"] and s["route_match"] is None
    assert summarize([s])["route_match"] == (None, 0)


def test_content_routed_needs_a_content_tool():
    task = {"expected_tools": ["update_element"], "expected_status": "ok", "content_bearing": True}
    assert score_row(_row(["update_element"]), task)["content_routed"] is False
    assert score_row(_row(["write_text", "update_element"]), task)["content_routed"] is True
    assert "write_text" not in READ_TOOLS
