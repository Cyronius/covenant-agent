"""Can each sampler reproduce a program it is told the answer to?

A bug in either sampler would be invisible in training loss and would show up
only as one arm looking worse than the other. That is the most dangerous kind of
bug in a comparison, because it produces a confident wrong result rather than an
error.

So: replace the model with an oracle that returns the target as one-hot logits,
and check that each sampler recovers the target exactly. Anything less than an
exact match is a bug in the sampler, not in a model. Under structural binding
this also exercises the receiver rule in `sample.local_mask`: an oracle target
never violates it, so a mask that blocked a legal target would fail here.

    python test_samplers.py --cache data_cache_struct
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from evaluate import Split
from sample import (ar_backoff, ar_sample, diffusion_sample, local_mask, local_sets,
                    repair_targets, slots_for_lines, slots_for_register)


class Oracle(nn.Module):
    """Stands in for a perfectly trained model.

    Returns the target at every slot regardless of what the canvas holds, which
    is what a model that had learned the task completely would do. Confidence is
    uniform, so the cosine schedule -- not the scores -- decides fill order, and
    any slot the sampler forgets to fill stays visible in the output.
    """

    def __init__(self, target: torch.Tensor, vocab: int, canvas: int, causal: bool):
        super().__init__()
        self.target = target
        self.vocab = vocab

        class _C:
            pass
        self.c = _C()
        self.c.canvas = canvas
        self.c.causal = causal

    def encode_inputs(self, inputs):
        """The samplers cache this, so the stand-in has to offer it too."""
        return None

    def decode(self, inputs, canvas, mem=None, loops=None):
        b, n = canvas.shape
        logits = torch.zeros(b, n, self.vocab)
        logits.scatter_(2, self.target[:, :n].unsqueeze(-1), 10.0)
        return logits


class Detour(Oracle):
    """The oracle, except that at one slot after CALL it prefers a wrong tool
    and ranks the target's second -- the shape of R15's same-shape swap, which
    ar_backoff exists to undo."""

    def __init__(self, target: torch.Tensor, vocab: int, canvas: int, slot: int, wrong: int):
        super().__init__(target, vocab, canvas, causal=True)
        self.slot, self.wrong = slot, wrong

    def decode(self, inputs, canvas, mem=None, loops=None):
        logits = super().decode(inputs, canvas, mem, loops)
        logits[:, self.slot, int(self.target[0, self.slot])] = 5.0
        logits[:, self.slot, self.wrong] = 10.0
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="data_cache_struct")
    ap.add_argument("--n", type=int, default=25)
    args = ap.parse_args()

    split = Split(Path(args.cache), "val", torch.device("cpu"))
    canvas_len = split.d["tgt"].shape[1]
    print(f"binding: {split.binding}")

    failures = []
    backoff_checked = 0
    n = min(args.n, len(split))
    for i in range(n):
        tgt = split.target(i)
        ov = split.codec(i)
        inputs = {"tgt": tgt}                 # the oracle ignores inputs; a device is all it needs

        # The diffusion sampler, at several step counts. One step means filling
        # the whole canvas at once, which an oracle should still get right.
        for steps in (1, 2, 4, 8, 16):
            model = Oracle(tgt, len(ov), canvas_len, causal=False)
            out, tr = diffusion_sample(model, inputs, ov, steps=steps)
            if not torch.equal(out, tgt):
                bad = (out != tgt).nonzero()[:, 1].tolist()
                failures.append(f"diffusion steps={steps} row={i} slots={bad[:6]}")
            if -1 in tr.unmask_step:
                failures.append(f"diffusion steps={steps} row={i} left slots unfilled")

        # The control. It stops at the first PAD, so compare only up to there.
        model = Oracle(tgt, len(ov), canvas_len, causal=True)
        out, tr = ar_sample(model, inputs, ov)
        end = int((tgt[0] != ov.pad).sum())
        if not torch.equal(out[0, :end], tgt[0, :end]):
            failures.append(f"ar row={i} differs within the program body")
        if int(out[0, end]) != ov.pad:
            failures.append(f"ar row={i} did not stop at the program end")
        # The control must not be charged fewer passes than tokens it emitted.
        if tr.passes < end:
            failures.append(f"ar row={i} passes {tr.passes} < {end} emitted tokens")

        # Text round-trip: what the scorer will compile has to match the target.
        if ov.render(out[0].tolist()) != ov.render(tgt[0].tolist()):
            failures.append(f"ar row={i} text round-trip differs")

        # Tool backoff. The compiler stand-in accepts only the target program.
        want = ov.render(tgt[0].tolist())
        model = Oracle(tgt, len(ov), canvas_len, causal=True)
        out, tr = ar_backoff(model, inputs, ov, lambda text: text == want, tries=3)
        if ov.render(out[0].tolist()) != want or tr.tries != 1 or tr.backoff_at is not None:
            failures.append(f"backoff row={i} touched a program that compiled")
        toks = ov.decode(tgt[0].tolist())
        tools = local_sets(ov)[6].nonzero().squeeze(1).tolist()
        for s in [k for k in range(1, end) if toks[k - 1] == "CALL"][:1]:
            wrong = next((t for t in tools if t != int(tgt[0, s])), None)
            if wrong is None:
                continue
            backoff_checked += 1
            model = Detour(tgt, len(ov), canvas_len, s, wrong)
            out, tr = ar_backoff(model, inputs, ov, lambda text: text == want, tries=3)
            if ov.render(out[0].tolist()) != want:
                failures.append(f"backoff row={i} slot={s} did not recover the target")
            elif tr.tries != 2 or not tr.backoff_at or tr.backoff_at[0] != s:
                failures.append(f"backoff row={i} slot={s} recovered it at the wrong "
                                f"place: tries={tr.tries} at={tr.backoff_at}")
            # Nothing compiles: the greedy program comes back, within budget.
            greedy, _ = ar_sample(model, inputs, ov)
            out, tr = ar_backoff(model, inputs, ov, lambda text: False, tries=3)
            if not torch.equal(out, greedy) or not 1 < tr.tries <= 4:
                failures.append(f"backoff row={i} slot={s} did not fall back to greedy "
                                f"within budget: tries={tr.tries}")

        # The receiver rule must never block the reference at any slot.
        blocked = local_mask(tgt, ov)[0]
        hit = blocked.gather(1, tgt[0].unsqueeze(1)).squeeze(1)
        if hit.any():
            failures.append(f"row={i} local_mask blocks the reference at slots "
                            f"{hit.nonzero().squeeze(1).tolist()[:6]}")

    # Line attribution, which is what the repair loop steers by. A diagnostic
    # blaming line k must select exactly the slots of line k.
    ids = split.target(0)[0].tolist()
    ov = split.codec(0)
    toks = ov.decode(ids)
    n_lines = sum(1 for t in toks if t == "NL")
    for line in range(1, n_lines + 1):
        slots = slots_for_lines(ids, ov, {line})
        if not slots:
            failures.append(f"line {line} of {n_lines} mapped to no slots")
            continue
        got = [toks[s] for s in slots]
        if got[-1] != "NL":
            failures.append(f"line {line} does not end at a newline: {got}")
    # An UNBOUND diagnostic must blame the receiver form `r0.` as well as `r0`.
    for i in range(n):
        ids = split.target(i)[0].tolist()
        ov = split.codec(i)
        toks = ov.decode(ids)
        for r in {t.rstrip(".") for t in toks if t.startswith("r") and t[1:].rstrip(".").isdigit()}:
            want = [k for k, t in enumerate(toks) if t == r or t == r + "."]
            if slots_for_register(ids, ov, r) != want:
                failures.append(f"row={i} slots_for_register({r}) misses the receiver form")
    # And an unattributable failure must fall back to the whole canvas rather
    # than selecting nothing, which used to make the repair loop a no-op.
    empty = [ov.pad] * canvas_len
    if not repair_targets(empty, ov, ["PARSE_ERROR line:1 empty program"]) == []:
        pass  # attribution finds nothing here; repair() supplies the fallback

    print(f"checked {n} programs x 5 step counts, plus {n_lines} line attributions, "
          f"plus tool backoff on {backoff_checked} detoured CALL slots")
    if not backoff_checked:
        failures.append("no program with a CALL and two tools: tool backoff went unchecked")
    if failures:
        print(f"\n{len(failures)} FAILURES:")
        for f in failures[:15]:
            print("  " + f)
        raise SystemExit(1)
    print("\nPASS -- both samplers reproduce a known answer exactly")


if __name__ == "__main__":
    main()
