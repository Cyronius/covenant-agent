"""The tiny planner in the demo app (.claude/plans/tiny-planner-demo.md step 5).

Both checks need local artifacts that are not in git -- a tiny checkpoint
and its cache (models/tiny/runs/, data_cache_*), and the ELECTRA reader's
files under models/tiny/reader (electra_reader.py) -- and skip where those
are missing.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TINY = ROOT / "models" / "tiny"
READER = TINY / "reader" / "embed" / "n" / "head.pt"
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(TINY))


def _need(*paths):
    missing = [p for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"local artifact missing: {missing[0]}")


def test_the_live_reader_reproduces_the_cache_table():
    """A demo context is read live; training read the same texts from
    reader.pt. If the two disagree the planner sees different inputs in the
    app than it was trained on, and nothing would say so."""
    import torch
    from live_reader import open_reader
    from prep import QUERY_PREFIX
    cache = TINY / "data_cache_c0e"
    _need(READER, cache / "reader.pt", cache / "teacher.pt")
    table = torch.load(cache / "reader.pt")["table"].float()
    texts = torch.load(cache / "teacher.pt")["texts"]
    idx = [0, 1, 7, 1000, len(texts) - 1]
    got = open_reader("electra").vectors([t[len(QUERY_PREFIX):] if t.startswith(QUERY_PREFIX) else t
                                          for t in (texts[i] for i in idx)])
    # the table was read on a GPU (TF32), the live path on the CPU
    assert (got * table[idx]).sum(-1).min().item() > 0.995


@pytest.mark.parametrize("request_text", ["list all the cards",
                                          "delete all the cards owned by bob",
                                          "archive everything that is done"])
def test_the_server_backend_returns_a_program_that_compiles(request_text):
    """POST /plan with a tiny checkpoint selected: the demo's own kanban
    prompt in, a program the compiler accepts out (backoff's compile check
    runs inside, so a failure here is a wiring fault, not a model miss)."""
    import tiny_planner
    ckpt, cache = tiny_planner.TINY_MODELS["tiny:c0_ESC"]
    _need(READER, TINY / ckpt, TINY / cache / "config.json")
    import dev_server
    from core.ir import TaskContext
    from core.pipeline import build
    from harness.demo_suite import demo_state
    kp = dev_server.handle_kanban_prompt({"request": request_text, "state": demo_state()})
    before = dev_server.PLANNER.tiny_name or str(dev_server.PLANNER.model_path)
    dev_server.PLANNER.select("tiny:c0_ESC")
    try:
        out = dev_server.handle_plan({"request": request_text, "context": kp["context"]})
    finally:
        dev_server.PLANNER.select(before)
    assert "error" not in out, out
    assert build(out["text"], TaskContext.from_json(kp["context"])).compile_ok, out["text"]


def test_a_request_with_line_breaks_is_read_as_one_line():
    """The serialized context is line-oriented; the dungeon's grid used to
    reach the parser as schema lines and kill the turn
    (.claude/plans/rpg-exits-perception.md part 2)."""
    import tiny_planner
    ckpt, cache = tiny_planner.TINY_MODELS["tiny:c0_ESC"]
    _need(READER, TINY / ckpt, TINY / cache / "config.json")
    import dev_server
    from harness.demo_suite import demo_state
    kp = dev_server.handle_kanban_prompt({"request": "list all the cards",
                                          "state": demo_state()})
    out = tiny_planner.TinyPlanner("tiny:c0_ESC").plan(
        "list all\n  the cards\n", kp["context"])
    assert out["source"].splitlines()[0] == "REQUEST: list all the cards"
    assert out["truncated"] is False


def test_the_dungeon_brief_fits_the_request_budget_at_its_longest():
    """Everything remembered, out of view, and a full turn of events: the
    longest brief the dungeon can write still reaches the planner whole."""
    import json
    from tokenizers import Tokenizer
    from runtime.worlds import rpg
    cache = TINY / "data_cache_clt"
    _need(cache / "in_tok.json")
    tk = Tokenizer.from_file(str(cache / "in_tok.json"))
    tk.no_truncation()
    budget = json.loads((cache / "config.json").read_text())["max_req"]
    st = rpg.new_state()
    st["entities"]["player"][0].update(x=1, y=1, hp=10)
    st["memory"] = ["enemy_1", "enemy_2", "item_1", "item_2", "door_1"]
    st.update(turn=39, log=["failed: cannot move south: a wall"] * 3
              + ["the goblin hits you for 2"] * 2 + ["you died"])
    st["entities"]["enemy"][0].update(x=9, y=7)     # both goblins out of view
    brief = rpg.observe(st, exits=True, paths=True).brief
    assert len(tk.encode(brief).ids) <= budget, brief
