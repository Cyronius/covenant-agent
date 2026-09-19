"""Turn a trained model into programs.

The diffusion sampler starts from an all-masked canvas and unmasks the most
confident slots on a cosine schedule. The control decodes left to right. Both
count forward passes, because that is the axis the two arms have to be compared
on: the control spends one pass per token no matter what, while the diffusion
arm spends however many steps it is given.

Both encode the input exactly once. Under the flat binding the input runs about
1100 tokens against a 64-slot canvas, so the encoder dominates a full forward
pass, and re-encoding per step would bury the difference between the arms under
a cost they share. Caching it makes the pass count measure decoder work, which
is the thing that actually differs. Under structural binding the cached thing
is region A (`model.Memory`), the same object the NPU design keeps on-chip.

The model is used through two calls so that both bindings fit one sampler:
`mem = model.encode_inputs(inputs)` and `logits = model.decode(inputs, canvas,
mem=mem)`, where `inputs` is a dict of whatever tensors the binding needs.
`ov` is the flat `OutVocab` or the structural `TaskCodec`; the sampler uses
only `.pad`, `.mask`, `decode(ids)` and `render(ids)`.

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


# -- the grammar's local half: a receiver is followed by a field -----------

CMP = ("EQ", "LT", "GT", "CONTAINS", "IN")     # spec/agent_core.md:99
JOIN = ("AND", "OR")


def local_sets(ov):
    """(receiver, field, cmp, join, not) as bool tensors over the codec's id
    space. Cached on the codec, which is per task for the structural binding."""
    cached = getattr(ov, "_local_sets", None)
    if cached is None:
        toks = ov.decode(list(range(len(ov))))
        recv = torch.tensor([bool(re.fullmatch(r"r\d+\.", t)) for t in toks])
        fld = torch.tensor([bool(re.fullmatch(r"F\d+", t)) for t in toks])
        cmp_ = torch.tensor([t in CMP for t in toks])
        join = torch.tensor([t in JOIN for t in toks])
        neg = torch.tensor([t == "NOT" for t in toks])
        cached = (recv, fld, cmp_, join, neg)
        try:
            ov._local_sets = cached
        except AttributeError:
            pass
    return cached


def local_mask(canvas: torch.Tensor, ov) -> torch.Tensor:
    """(1, C, V) True where a joint id is illegal at that slot given the slots
    already on the canvas. Finite-state rules over immediate neighbours, and
    nothing else (`.claude/plans/canvas-field-access-split.md` section 7):

      after a receiver `rN.`   the slot must hold a field
      before a filled non-field the slot must not hold a receiver
      after AND or OR          the slot is not AND, OR, or a comparator
      after NOT                the slot is not AND, OR, NOT, or a comparator
      after a comparator       the slot is not AND, OR, NOT, or a comparator
      before a comparator      the slot is not AND, OR, NOT, or a comparator
      before AND or OR         the slot is not AND, OR, or a comparator
      before NOT               the slot is not NOT or a comparator

    The predicate rules are the grammar's own productions read locally
    (`spec/agent_core.md:95-99`): `pred = clause {("AND"|"OR") clause}` and
    `clause = ["NOT"] field cmp (operand|field)`, plus the `cond` form for `IF`,
    which differs only in allowing an operand or `EMPTY` where `pred` wants a
    field. Every rule above holds under both, so no line-head scope is needed:
    each one says only that two operators cannot sit next to each other, and
    `AND NOT` -- the one legal operator pair -- is allowed.

    They exist because a left-to-right decoder cannot break them and a parallel
    one can. Measured on the step-1 runs (`results/R7.md` section 3): the control
    arm produced no ungrammatical predicate in 1,415 programs, while the
    diffusion arm broke these rules in a third of level-3 programs -- `AND AND`,
    a comparator where a field belongs -- because each slot's distribution was
    computed before its neighbour committed. This is decision 8's mask carrying
    the part of the grammar that is about adjacency rather than about which
    symbols exist.

    Everything entity-aware (a field of the entity in r0, a register bound
    before use) stays with the host typechecker in `repair`.
    """
    recv, fld, cmp_, join, neg = (t.to(canvas.device) for t in local_sets(ov))
    tok = canvas[0]
    n = tok.numel()
    blocked = torch.zeros(1, n, recv.numel(), dtype=torch.bool, device=canvas.device)
    filled = tok != ov.mask

    def shifted(sel, by):
        """True at slot i when the neighbour `by` away is filled and in `sel`."""
        out = torch.zeros(n, dtype=torch.bool, device=canvas.device)
        if by == 1:                                   # the slot before
            out[1:] = sel[tok[:-1]] & filled[:-1]
        else:                                         # the slot after
            out[:-1] = sel[tok[1:]] & filled[1:]
        return out

    after_recv = shifted(recv, 1)
    blocked[0, after_recv] = ~fld
    before_nonfield = torch.zeros(n, dtype=torch.bool, device=canvas.device)
    before_nonfield[:-1] = filled[1:] & ~fld[tok[1:]]
    blocked[0, before_nonfield] |= recv

    # Two operators cannot be adjacent. AND NOT is the exception and is legal.
    op = cmp_ | join | neg
    for sel, by, ban in ((join, 1, join | cmp_),      # AND/OR then ...
                         (neg, 1, op),                # NOT then ...
                         (cmp_, 1, op),               # a comparator then ...
                         (cmp_, -1, op),              # ... then a comparator
                         (join, -1, join | cmp_),     # ... then AND/OR
                         (neg, -1, neg | cmp_)):      # ... then NOT
        blocked[0, shifted(sel, by)] |= ban
    return blocked


# -- samplers ----------------------------------------------------------------

@torch.no_grad()
def diffusion_sample(model, inputs: dict, ov, steps: int = 8, temperature: float = 0.0,
                     trace: Trace | None = None, threshold: float = 0.0):
    """Confidence unmasking. Batch of one, for clarity.

    temperature 0 takes the argmax. That is the opposite of what the prose
    decoder wants, and deliberately so: here the target is near-deterministic
    given the input, and a compiler decides whether the answer is right, so the
    most likely program is the one to want.

    Two schedules, and what `steps` means differs between them:

      threshold = 0   the cosine schedule. `steps` passes, each committing the
                      slots the schedule calls for whether or not the model is
                      sure of them. Every program costs exactly `steps` passes.
      threshold > 0   commit every unfilled slot the model is at least this
                      confident of, and at least the best one so it cannot
                      stall; `steps` becomes a cap rather than a count, and a
                      program costs as many passes as it needs.

    The second exists because of what step 2 measured. On compound filter
    predicates (level 3) goal success is 0% at 8 passes and 7 to 16% at 32,
    while looping the block -- eight times the arithmetic per pass, free on the
    NPU -- does nothing for them. So what a pass buys is not refinement, it is a
    smaller commit granularity: at 32 passes over a 64-slot canvas the sampler
    commits about two slots per pass, each seeing the last. Paying that price on
    every program is the waste the threshold removes, because most programs have
    no predicate in them.
    """
    n = model.c.canvas
    device = next(iter(inputs.values())).device
    canvas = torch.full((1, n), ov.mask, dtype=torch.long, device=device)
    filled = torch.zeros(1, n, dtype=torch.bool, device=device)
    tr = trace or Trace()
    tr.unmask_step = [-1] * n
    mem = model.encode_inputs(inputs)      # once: the encoder dwarfs the decoder

    for s in range(steps):
        logits = model.decode(inputs, canvas, mem=mem)
        logits = logits.masked_fill(local_mask(canvas, ov), float("-inf"))
        tr.passes += 1
        tr.steps += 1
        probs = F.softmax(logits.float(), dim=-1)
        conf, pred = probs.max(dim=-1)
        if temperature > 0:
            pred = torch.multinomial(
                F.softmax(logits.float()[0] / temperature, dim=-1), 1).squeeze(1).unsqueeze(0)
            conf = probs[0].gather(1, pred[0].unsqueeze(1)).squeeze(1).unsqueeze(0)

        cand = conf.masked_fill(filled, -1.0)
        if threshold > 0:
            # Everything the model is sure of, and never nothing: a pass that
            # commits no slot would loop until the cap with the canvas unchanged.
            take = (~filled[0]) & (conf[0] >= threshold)
            if s == steps - 1 or not bool(take.any()):
                take = ~filled[0] if s == steps - 1 else torch.zeros_like(take)
                if not bool(take.any()):
                    take[int(cand[0].argmax())] = True
            idx = take.nonzero(as_tuple=True)[0]
        else:
            # How many slots should be filled after this step, on a cosine schedule.
            target_filled = n if s == steps - 1 else int(
                n * (1 - math.cos(math.pi / 2 * (s + 1) / steps)))
            need = target_filled - int(filled.sum())
            if need <= 0:
                continue
            idx = cand[0].topk(min(need, int((~filled).sum()))).indices
        # Commit in descending confidence, re-checking the local rules against
        # what THIS step has already committed.
        #
        # Committing the whole set at once is what let `AND AND` through. The
        # mask above is computed from the canvas as the step began, so for two
        # slots filled in the same step neither sees the other, and the last step
        # of an 8-step schedule commits about 13 slots together -- exactly the
        # low-confidence ones, which is where a predicate's operators live. The
        # loop costs no forward pass: the model's logits are reused and only the
        # mask is recomputed.
        for i in idx[conf[0, idx].argsort(descending=True)].tolist():
            row = logits[0, i].masked_fill(local_mask(canvas, ov)[0, i], float("-inf"))
            tok = int(row.argmax()) if bool(torch.isfinite(row).any()) else int(pred[0, i])
            canvas[0, i] = tok
            filled[0, i] = True
            tr.unmask_step[i] = s
        if filled.all():
            break

    return canvas, tr


@torch.no_grad()
def ar_sample(model, inputs: dict, ov, trace: Trace | None = None):
    """Greedy left to right. Stops at the first PAD, which is the trained stop."""
    n = model.c.canvas
    device = next(iter(inputs.values())).device
    canvas = torch.full((1, n), ov.mask, dtype=torch.long, device=device)
    tr = trace or Trace()
    tr.unmask_step = [-1] * n
    out = torch.full((1, n), ov.pad, dtype=torch.long, device=device)
    mem = model.encode_inputs(inputs)      # once, same as the diffusion arm
    for i in range(n):
        logits = model.decode(inputs, canvas, mem=mem)
        tr.passes += 1
        row = logits[0, i]
        if i > 0:
            # The receiver rule, left to right: after `rN.` comes a field.
            row = row.masked_fill(local_mask(out[:, :i + 1], ov)[0, i], float("-inf"))
        tok = int(row.argmax())
        out[0, i] = tok
        tr.unmask_step[i] = i
        if tok == ov.pad:
            break
        if i + 1 < n:
            canvas[0, i + 1] = tok
    return out, tr


# -- the compiler in the loop -------------------------------------------------

def slots_for_lines(ids: list[int], ov, lines: set[int]) -> list[int]:
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


def slots_for_register(ids: list[int], ov, reg: str) -> list[int]:
    """Every slot that names `reg`, as an operand (`r0`) or as the receiver
    of a field access (`r0.`); an UNBOUND diagnostic blames both forms."""
    toks = ov.decode(ids)
    return [i for i, t in enumerate(toks) if t == reg or t == reg + "."]


def repair_targets(ids: list[int], ov, diagnostics: list[str]) -> list[int]:
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
def repair(model, inputs: dict, ov, canvas: torch.Tensor, build_fn, ctx,
           rounds: int = 2, steps: int = 4, trace: Trace | None = None):
    """Remask what the compiler blamed, refill, repeat.

    build_fn is covenant-agent's pipeline.build. Everything the compiler did not
    blame stays frozen, so a repair round is a small conditional denoising
    problem rather than a fresh generation.
    """
    tr = trace or Trace()
    ids = canvas[0].tolist()
    res = build_fn(ov.render(ids), ctx)
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
        mem = model.encode_inputs(inputs)
        for s in range(steps):
            logits = model.decode(inputs, canvas, mem=mem)
            logits = logits.masked_fill(local_mask(canvas, ov), float("-inf"))
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
        res = build_fn(ov.render(canvas[0].tolist()), ctx)
        tr.compiled = res.compile_ok
        tr.diagnostics = res.rendered_diagnostics() if not res.compile_ok else []
        if res.compile_ok:
            break
    return canvas, tr


def to_text(canvas: torch.Tensor, ov) -> str:
    return ov.render(canvas[0].tolist())
