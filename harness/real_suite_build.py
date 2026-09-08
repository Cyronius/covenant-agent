"""Build data/holdout/e_real_sessions.jsonl from the frozen candidates
(harness/real_suite.py `build`; plan real-sessions-eval-suite §3).

Each row is a harness task: the request verbatim, a coursebuilder context
built the same way the demo builds its kanban contexts (fixed constants +
literals from the request + the request as the writer brief), the world's
default state, and the routing expectations the scorer reads
(`expected_tools`, `expected_status`, `content_bearing`). Only request
text and tool names come from the candidates — no arguments, results,
account or session ids.
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "server"))

from baselines.qwen.run_a import SYSTEM, build_prompt  # noqa: E402,F401
from dev_server import literals_from_request  # noqa: E402
from harness.context import build_context, serialize_context  # noqa: E402
from harness.real_suite import CANDIDATES, OUT, expected_tools  # noqa: E402
from runtime.worlds import get_world  # noqa: E402

_HEX = re.compile(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b")
_RGB = re.compile(r"rgb\(\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\)")
_INT = re.compile(r"\b(\d{1,3})\b")
# phrases people use -> element type strings the real palette has
ELEMENT_TYPES = [
    (re.compile(r"\baccordi[oa]ns?\b", re.I), "accordion"),
    (re.compile(r"\bflip ?cards?\b", re.I), "flipCards"),
    (re.compile(r"\bcarousels?\b|\bslides?h?o?w?\b", re.I), "carousel"),
    (re.compile(r"\btabs?\b", re.I), "tabs"),
    (re.compile(r"\bhotspots?\b", re.I), "hotspot"),
    (re.compile(r"\bquiz(zes)?\b|\bmultiple[- ]choice\b", re.I), "multipleChoice"),
    (re.compile(r"\bknowledge ?checks?\b", re.I), "knowledgeCheckMultipleChoice"),
    (re.compile(r"\bskill ?checks?\b", re.I), "skillcheckOpenResponse"),
    (re.compile(r"\bimages?\b|\bpictures?\b|\bphotos?\b", re.I), "image"),
    (re.compile(r"\bvideos?\b", re.I), "video"),
    (re.compile(r"\bheadings?\b|\btitles?\b", re.I), "heading1"),
    (re.compile(r"\bparagraphs?\b|\btext\b", re.I), "paragraph"),
    (re.compile(r"\bbullet(ed)? ?(points|list)\b|\blist\b", re.I), "bulletedList"),
    (re.compile(r"\btables?\b", re.I), "table"),
    (re.compile(r"\bhighlight\b|\bcall ?out\b", re.I), "highlight1"),
    (re.compile(r"\bembed(ded)?\b|\bhtml\b|\bcustom code\b", re.I), "customCode"),
    (re.compile(r"\bquotes?\b", re.I), "quote1"),
    (re.compile(r"\bdivider\b|\bspacer\b", re.I), "divider"),
]


def constants_for(request: str, world: dict) -> list:
    """Fixed course-builder constants first (stable indices), then request
    literals, then the request itself as the writer brief (last)."""
    st = world["default_state"]["entities"]
    out = [
        {"type": "ID:module", "value": "module_2", "desc": "the current lesson (the module open in the editor)"},
        {"type": "ID:element", "value": "element_4", "desc": "the selected element (a paragraph in the current lesson)"},
        {"type": "ID:course", "value": st["course"][0]["id"], "desc": "this course"},
        {"type": "BOOL", "value": True, "desc": "true"},
        {"type": "BOOL", "value": False, "desc": "false"},
        {"type": "STR", "value": "lesson", "desc": "module type: lesson",
         "kind": "enum:module.type"},
        {"type": "STR", "value": "quiz", "desc": "module type: quiz",
         "kind": "enum:module.type"},
        {"type": "STR", "value": "bottom",
         "desc": "position: bottom (end of the lesson)", "kind": "text"},
    ]
    seen_types = set()
    for rx, ty in ELEMENT_TYPES:
        if rx.search(request) and ty not in seen_types:
            seen_types.add(ty)
            # element.type is not declared in world["enums"] -- the palette
            # has ~18 values and emitting all of them for every request is
            # the cost R5 measured -- but the kind still narrows the slot
            out.append({"type": "STR", "value": ty,
                        "desc": f"element type: {ty}",
                        "kind": "enum:element.type"})
    for m in _HEX.finditer(request):
        out.append({"type": "STR", "value": m.group(0),
                    "desc": f"the colour {m.group(0)}", "kind": "text"})
    for m in _RGB.finditer(request):
        out.append({"type": "STR", "value": m.group(0),
                    "desc": f"the colour {m.group(0)}", "kind": "text"})
    for m in _INT.finditer(request):
        v = int(m.group(1))
        if 1 <= v <= 100 and not any(c["type"] == "INT" and c["value"] == v for c in out):
            out.append({"type": "INT", "value": v, "desc": f"the number {v}"})
    out.extend(literals_from_request(request, world["now"]))
    out.append({"type": "STR", "value": request,
                "desc": "the request itself, verbatim (brief for write_text / generate_image / apply_to_lessons)",
                "kind": "text"})
    return out


def build_rows(candidates: list, seed: int, symbols: str = "classic",
               enums: bool = False, kinds: bool = False) -> list:
    """`symbols`/`enums`/`kinds` select the 0.4.0 surface (spec 0.4.0
    §2.1/§2.2/§2.3). Off by default so the stored suite rebuilds byte for
    byte; `constants_for` always writes the kinds and this strips them."""
    world = get_world("coursebuilder")
    if not world["tools"]:
        raise SystemExit("coursebuilder world has no tools: run python -m data.gen.world_from_schemas first")
    rows = []
    for i, c in enumerate(candidates):
        request = c["request"]
        constants = constants_for(request, world)
        if not kinds:
            constants = [{k: v for k, v in c.items() if k != "kind"}
                         for c in constants]
        ctx, _ = build_context(world, constants,
                               random.Random(seed * 1000003 + i),
                               symbols=symbols, enums=enums)
        primary = c["routing"]["primary"]
        exp = expected_tools([n for n in c["frontier_tool_calls"] if n])
        rows.append({
            "id": f"real_{c['turn_id']}",
            "level": 0,
            "world": "coursebuilder",
            "request": request,
            "context": ctx.to_json(),
            "input_text": serialize_context(request, ctx),
            "state": world["default_state"],
            "expected_state": world["default_state"],   # unused: routing is scored, not state
            "expected_status": "aborted" if primary == "none" else "ok",
            "now": world["now"],
            "approval": True,                           # let every call run so the route is visible
            "reference": {},
            "expected_tools": exp,
            "routing_primary": primary,
            "multi_step": c["routing"]["multi_step"],
            "content_bearing": c["routing"]["content_bearing"],
            "teachability": c["teachability_guess"],
            "tags": ["real", c["agent"], primary] + (["unmapped"] if primary != "none" and not exp else []),
            "symbols": symbols,
            "spec_version": "0.4.0",
        })
    return rows


def cmd_build(args) -> None:
    cands = [json.loads(l) for l in open(CANDIDATES, encoding="utf-8")]
    rows = build_rows(cands, args.seed, symbols=args.symbols,
                      enums=args.enums, kinds=args.kinds)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    n_ab = sum(r["expected_status"] == "aborted" for r in rows)
    n_un = sum("unmapped" in r["tags"] for r in rows)
    lens = sorted(len(r["input_text"]) for r in rows)
    print(f"{len(rows)} rows -> {OUT}: {n_ab} expect abstain, {n_un} unmapped; "
          f"input_text chars p50 {lens[len(lens)//2]} max {lens[-1]}")
