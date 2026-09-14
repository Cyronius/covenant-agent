"""English request rendering (F4).

Template renderer with three styles (formal / casual / terse) and phrasing
variety per frame. This is the default *teacher stand-in*: the plan calls for
a teacher model to write requests; data.gen exposes a --teacher hook, and
until one is wired in, pairs are tagged teacher=template-v1 so template-only
corpora are identifiable (flagged in results/foundation.md).

render(frame, rng) -> (request_text, style_tag)
"""
from __future__ import annotations

import random

STYLES = ["formal", "casual", "terse"]

EVERY = ["every", "all the", "each of the", "any"]


def _np(noun_pair, clauses, rng, plural=True):
    """Noun phrase from clause phrase material."""
    noun = noun_pair[1] if plural else noun_pair[0]
    pre, post, negs = [], [], []
    for c in clauses:
        text, place = c["phrase"]
        if c.get("neg"):
            negs.append((text, place))
        elif place == "pre":
            pre.append(text)
        else:
            post.append(text)
    out = (" ".join(pre) + " " if pre else "") + noun
    if post:
        out += " " + " ".join(post) if len(post) == 1 else (
            " " + ", ".join(post))
    for text, place in negs:
        form = rng.choice([
            f"except the {text} ones",
            f"but leave the {text} ones alone",
            f"as long as they aren't {text}",
        ]) if place == "pre" else rng.choice([
            f"except those {text}",
            f"unless they are {text}",
        ])
        out += ", " + form
    return out


def _wrap(core: str, style: str, rng) -> str:
    core = core[0].lower() + core[1:]
    if core.startswith(("if ", "check ", "for every ")):
        # conditional / iterative cores read badly behind "Please"
        return core[0].upper() + core[1:] + ("" if style == "terse" else ".")
    if style == "formal":
        return rng.choice(["Please ", "I need you to ", "Could you "]) \
            + core + "."
    if style == "casual":
        return rng.choice([f"Can you {core}?", f"Go ahead and {core}.",
                           f"Hey — {core}, thanks!"])
    return core  # terse


def _obj_for(frame, rng) -> str:
    noun = frame["noun"][0]
    rec = frame["record"]
    if rec.startswith(noun + " ") or rec.startswith("ticket ") \
            or rec.startswith("invoice "):
        return rec  # "card 3" style
    if noun in ("customer",):
        return rec
    return f'the "{rec}" {noun}'


def render(frame: dict, rng: random.Random):
    style = rng.choice(STYLES)
    r = frame["recipe"]
    if r == "ambiguous":
        return frame["request"], style
    from .v2 import V2_RECIPES, render_v2  # lazy: v2 imports programs
    if r in V2_RECIPES:
        core = render_v2(frame, rng, _np, _obj_for, _wrap, EVERY)
        low = core.lower().rstrip("?")
        if low.startswith(("how ", "which ", "what ", "where ", "can you ", "tell me ")):
            # questions stay questions; no "Please" / "Can you" wrapper
            return low[0].upper() + low[1:] + ("?" if style != "terse" else ""), style
        if low.startswith("on "):
            return core[0].upper() + core[1:] + ("" if style == "terse" else "."), style
        return _wrap(core, style, rng), style

    if r == "direct":
        core = frame["action"]["verb"].format(obj=_obj_for(frame, rng))
        return _wrap(core, style, rng), style
    if r == "direct_send":
        return _wrap(frame["verb"], style, rng), style
    if r == "chain":
        obj = _obj_for(frame, rng)
        act = frame["action"]["verb"].format(obj="it")
        core = rng.choice([f"look up {obj} and {act}",
                           f"find {obj}, then {act}"])
        return _wrap(core, style, rng), style
    if r in ("filter_act", "pause_filter_act"):
        np = _np(frame["noun"], frame["clauses"], rng)
        q = rng.choice(EVERY)
        act = frame["action"]["verb"]
        if r == "pause_filter_act":
            core = rng.choice([
                f"find {q} {np}, report back to me, then "
                + act.format(obj="each of them"),
                f"work out which {np} we have, check in first, then "
                + act.format(obj="them"),
            ])
        else:
            core = act.format(obj=f"{q} {np}")
        return _wrap(core, style, rng), style
    if r in ("argmax_count", "argmin_count"):
        np = _np(frame["inner_noun"], frame["clauses"], rng)
        if r == "argmax_count":
            sup = rng.choice(["the most", "the highest number of",
                              "the largest number of"])
        else:
            # "fewest" includes candidates with none at all -- the reason
            # LEAST carries the candidate list (spec 0.4.0 §4)
            sup = rng.choice(["the fewest", "the least", "the lowest number of"])
        winner = f"the {frame['outer_noun'][0]} with {sup} {np}"
        core = f"{frame['action']['verb']} {winner}"
        return _wrap(core, style, rng), style
    if r in ("argmax_which", "argmin_which"):
        np = _np(frame["inner_noun"], frame["clauses"], rng)
        sup = rng.choice(["the most", "the highest number of"]) \
            if r == "argmax_which" else \
            rng.choice(["the fewest", "the lowest number of"])
        core, end = rng.choice([
            (f"which {frame['outer_noun'][0]} has {sup} {np}", "?"),
            (f"who has {sup} {np}", "?"),
            (f"tell me which {frame['outer_noun'][0]} has {sup} {np}", "."),
        ])
        return core[0].upper() + core[1:] + ("" if style == "terse" else end), style
    if r == "extreme_sort":
        core = frame["action"]["verb"].format(obj=frame["np"])
        return _wrap(core, style, rng), style
    if r == "branch":
        obj = _obj_for(frame, rng)
        cond = frame["cond_phrase"][0]
        t = frame["then"]["verb"].format(obj="it")
        e = frame["else"]["verb"].format(obj="it")
        core = rng.choice([
            f"if {obj} is {cond}, {t}; otherwise {e}",
            f"check {obj}: {t} if it is {cond}, and {e} if not",
        ])
        return _wrap(core, style, rng), style
    if r == "nested":
        onp = _np(frame["outer_noun"], frame["outer_clauses"], rng,
                  plural=False)
        inp = _np(frame["inner_noun"], frame["inner_clauses"], rng)
        act = frame["action"]["verb"].format(obj=f"each of their {inp}")
        core = f"for every {onp}, {act}"
        return _wrap(core, style, rng), style
    if r == "parallel":
        name = frame["name"]
        inp = _np(frame["inner_noun"], frame["clauses"], rng)
        if frame["variant"] == "foreach":
            act = frame["action"]["verb"].format(obj=f"{name}'s {inp}")
            core = (f"fetch {name}'s record and the "
                    f"{frame['inner_noun'][1]} list together, then {act}")
        else:
            core = (f"pull up {name} and the {frame['inner_noun'][1]} at "
                    f"the same time; if {name} has any {inp}, "
                    f"{frame['action']['verb']} them about it")
        return _wrap(core, style, rng), style
    if r == "recovery_retry":
        act = frame["action"]["verb"].format(obj=_obj_for(frame, rng))
        core = rng.choice([
            f"{act} — the API has been flaky, so retry if it fails",
            f"{act}, and try again a couple of times if it errors out",
        ])
        return _wrap(core, style, rng), style
    if r == "recovery_fallback":
        obj = _obj_for(frame, rng)
        act = frame["action"]["verb"].format(obj="it")
        core = (f"fetch {obj} and {act}; if that fails, "
                f"{frame['fb_verb']}")
        return _wrap(core, style, rng), style
    raise ValueError(f"unknown recipe {r}")
