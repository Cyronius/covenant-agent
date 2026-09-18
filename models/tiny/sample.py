"""Turn a trained model into programs.

The diffusion sampler starts from an all-masked canvas and unmasks the most
confident slots on a cosine schedule. The control decodes left to right. Both
count forward passes, because that is the axis the two arms have to be compared
on: the control spends one pass per token no matter what, while the diffusion
arm spends however many steps it is given.

Both encode the input exactly once. The input runs about 1100 tokens against a
64-slot canvas, so the encoder dominates a full forward pass, and re-encoding per
step would bury the difference between the arms under a cost they share. Caching
it makes the pass count measure decoder work, which is the thing that actually
differs. It also removes the main way this comparison could have been unfair,
since the control takes one pass per token and would have paid that encoder cost
sixty-four times.

The compiler loop is the part worth paying attention to. After sampling, the
program goes through covenant-agent's real parser and typechecker. Diagnostics
carry line numbers, so a failure points at the slots that caused it, those slots
go back to MASK, and the model fills them again with everything else held still.
That is a critic that cannot be fooled, which is a luxury prose does not have.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from corpus import detokenize
from model import CanvasModel
from tok import OutVocab

LINE_RE = re.compile(r"line:(\d+)")
REG_RE = re.compile(r"\br(\d+)\b")


@dataclass
class Trace:
    """What the sampler did, for the fill-order probe and the cost accounting."""
    passes: int = 0
    steps: int = 0
    repairs: int = 0
    unmask_step: list[int] = field(default_factory=list)   # per slot, -1 if never
    compiled: bool = False
    diagnostics: list[str] = field(default_factory=list)


@torch.no_grad()
def diffusion_sample(model: CanvasModel, src, pad, ov: OutVocab, steps: int = 8,
                     temperature: float = 0.0, trace: Trace | None = None, sym=None):
    """Cosine-schedule confidence unmasking. Batch of one, for clarity.

    temperature 0 takes the argmax. That is the opposite of what the prose
    decoder wants, and deliberately so: here the target is near-deterministic
    given the input, and a compiler decides whether the answer is right, so the
    most likely program is the one to want.
    """
    n = model.c.canvas
    device = src.device
    canvas = torch.full((1, n), ov.mask, dtype=torch.long, device=device)
    filled = torch.zeros(1, n, dtype=torch.bool, device=device)
    tr = trace or Trace()
    tr.unmask_step = [-1] * n
    mem = model.encode(src, pad)          # once: the encoder dwarfs the decoder

    for s in range(steps):
        logits = model(src, pad, canvas, mem=mem, sym=sym)
        tr.passes += 1
        tr.steps += 1
        probs = F.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        if temperature > 0:
            pred = torch.multinomial(
                F.softmax(logits.float()[0] / temperature, dim=-1), 1).squeeze(1).unsqueeze(0)
            conf = probs[0].gather(1, pred[0].unsqueeze(1)).squeeze(1).unsqueeze(0)

        # How many slots should be filled after this step, on a cosine schedule.
        target_filled = n if s == steps - 1 else int(
            n * (1 - math.cos(math.pi / 2 * (s + 1) / steps)))
        need = target_filled - int(filled.sum())
        if need <= 0:
            continue
        cand = conf.masked_fill(filled, -1.0)
        idx = cand[0].topk(min(need, int((~filled).sum()))).indices
        canvas[0, idx] = pred[0, idx]
        filled[0, idx] = True
        for i in idx.tolist():
            tr.unmask_step[i] = s
        if filled.all():
            break

    return canvas, tr


@torch.no_grad()
def ar_sample(model: CanvasModel, src, pad, ov: OutVocab, trace: Trace | None = None,
              sym=None):
    """Greedy left to right. Stops at the first PAD, which is the trained stop."""
    n = model.c.canvas
    device = src.device
    canvas = torch.full((1, n), ov.mask, dtype=torch.long, device=device)
    tr = trace or Trace()
    tr.unmask_step = [-1] * n
    out = torch.full((1, n), ov.pad, dtype=torch.long, device=device)
    mem = model.encode(src, pad)          # once, same as the diffusion arm
    for i in range(n):
        logits = model(src, pad, canvas, mem=mem, sym=sym)
        tr.passes += 1
        tok = int(logits[0, i].argmax())
        out[0, i] = tok
        tr.unmask_step[i] = i
        if tok == ov.pad:
            break
        if i + 1 < n:
            canvas[0, i + 1] = tok
    return out, tr


def slots_for_lines(ids: list[int], ov: OutVocab, lines: set[int]) -> list[int]:
    """Map 1-based program line numbers back to canvas slot indices.

    A line is a run of slots ending in NL. Indentation tokens belong to the line
    they precede, which is what makes a FOREACH body repairable as a unit.
    """
    toks = ov.decode(ids)
    out, cur, line_no = [], [], 1
    for i, t in enumerate(toks):
        if t == "PAD":
            break
        cur.append(i)
        if t == "NL":
            if line_no in lines:
                out.extend(cur)
            cur, line_no = [], line_no + 1
    if cur and line_no in lines:
        out.extend(cur)
    return out


def slots_for_register(ids: list[int], ov: OutVocab, reg: str) -> list[int]:
    toks = ov.decode(ids)
    return [i for i, t in enumerate(toks) if t == reg]


def repair_targets(ids: list[int], ov: OutVocab, diagnostics: list[str]) -> list[int]:
    """Which slots a set of diagnostics blames.

    A diagnostic with a line number blames that line. An unbound register also
    blames every slot that mentions it anywhere, because the fix may be to bind
    it earlier rather than to stop reading it. When nothing is attributable,
    blame the whole canvas and start over.
    """
    lines: set[int] = set()
    slots: set[int] = set()
    for d in diagnostics:
        m = LINE_RE.search(d)
        if m:
            lines.add(int(m.group(1)))
        if d.startswith("UNBOUND"):
            for r in REG_RE.findall(d):
                slots.update(slots_for_register(ids, ov, f"r{r}"))
        if d.startswith("PARSE_ERROR") and not m:
            return list(range(len(ids)))
    slots.update(slots_for_lines(ids, ov, lines))
    return sorted(slots)


@torch.no_grad()
def repair(model: CanvasModel, src, pad, ov: OutVocab, canvas: torch.Tensor,
           build_fn, ctx, rounds: int = 2, steps: int = 4, trace: Trace | None = None,
           sym=None):
    """Remask what the compiler blamed, refill, repeat.

    build_fn is covenant-agent's pipeline.build. Everything the compiler did not
    blame stays frozen, so a repair round is a small conditional denoising
    problem rather than a fresh generation.
    """
    tr = trace or Trace()
    ids = canvas[0].tolist()
    res = build_fn(detokenize(ov.decode(ids)), ctx)
    tr.compiled = res.compile_ok
    tr.diagnostics = res.rendered_diagnostics() if not res.compile_ok else []
    if res.compile_ok:
        return canvas, tr

    for _ in range(rounds):
        targets = repair_targets(canvas[0].tolist(), ov, tr.diagnostics)
        if not targets:
            # The compiler rejected the program but nothing could be blamed --
            # an empty program is the common case, since there are no slots to
            # point at. Giving up here would silently skip the repair budget,
            # so start the whole canvas over instead.
            targets = list(range(canvas.size(1)))
        idx = torch.tensor(targets, device=canvas.device)
        canvas[0, idx] = ov.mask
        filled = torch.ones_like(canvas, dtype=torch.bool)
        filled[0, idx] = False
        mem = model.encode(src, pad)
        for s in range(steps):
            logits = model(src, pad, canvas, mem=mem, sym=sym)
            tr.passes += 1
            conf, pred = F.softmax(logits.float(), dim=-1).max(dim=-1)
            need = int((~filled).sum()) if s == steps - 1 else max(
                1, int((~filled).sum()) // (steps - s))
            cand = conf.masked_fill(filled, -1.0)
            pick = cand[0].topk(min(need, int((~filled).sum()))).indices
            canvas[0, pick] = pred[0, pick]
            filled[0, pick] = True
            if filled.all():
                break
        tr.repairs += 1
        res = build_fn(detokenize(ov.decode(canvas[0].tolist())), ctx)
        tr.compiled = res.compile_ok
        tr.diagnostics = res.rendered_diagnostics() if not res.compile_ok else []
        if res.compile_ok:
            break
    return canvas, tr


def to_text(canvas: torch.Tensor, ov: OutVocab) -> str:
    return detokenize(ov.decode(canvas[0].tolist()))
