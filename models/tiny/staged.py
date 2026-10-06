"""The staged decoder (.claude/plans/staged-decoder-experts.md).

Everything after the encoder. A program gets one trip:

  draft stage(s)  diffusion: all 64 slots predicted at once from a blank
                  canvas, each stage looped over the whole canvas before the
                  next one starts
  early exit      stop there when the draft's lowest slot confidence clears
                  `Config.exit_threshold`
  refiner         left to right, one slot at a time, reading the context and
                  the draft (as tokens, with the draft's confidence per slot)

Each stage has its own weights. On the NPU that is one weight load per stage
per program; loops are compute only.

The model is causal, so every left-to-right sampler (sample.ar_sample,
ar_backoff, play.py, the demo server) drives it unchanged: `encode_inputs`
runs the encoder and the draft once, and `decode` runs the refiner -- or hands
back the draft's own logits when the program exits early or there is no
refiner, which reads the draft out slot by slot under the same grammar rules.

Training (`staged_loss`, called from train.batch_loss) uses one trip too: the
draft always starts from a blank canvas, as it does at run time, and its loss
covers every slot (padding at `pad_weight`, as the diffusion arm). The refiner
trains left to right on the draft the draft stage just produced, with
`draft_noise` of its slots swapped for random ones so it keeps checking the
draft instead of copying it.

On its own training rows the draft is nearly always right, so the refiner
learned to copy it (R28). Cross-fitted drafts (plan step 2b) fix the input:
two draft-only models each train on half the rows (`fold_of`), each drafts
the other half (xdraft.py), and the refiner trains on those stored drafts
(`xdraft_tok`, `xdraft_conf` columns) instead of its own draft stage's.

Most of what is left is constants: the draft fills every slot at once, so its
slots can't agree on which of several same-kind constants each takes, and the
refiner keeps its pick (R28 point 11). `commit` (plan step 2c) makes every
draft loop a round: read out, fix the surest slots as tokens (`commit_rule`),
loop on, so the unsure slots are decided last, knowing the rest -- inside the
same loaded stage. Training runs the same rounds on the model's own picks.
"""
from __future__ import annotations

import os
import re
import zlib
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import TAG_CANVAS, DecoderLayer, StructuralModel, _init

PAD_ID, MASK_ID = 0, 1
KINDS = ("draft", "refine")

# Task types (the plan's Terms): what the request asks for, read off the
# world a row comes from. Held-out worlds are listed so exams route too.
# The page apps are named one by one: `app_store` is a data theme.
OBS_ACT = {"warehouse_robot", "elevator", "cards", "workshop", "rooms",
           "boatyard", "rooms_after", "house", "rpg"}
PAGES = {"app_checkout", "app_settings", "app_ticket", "app_coursebuilder"}
TYPES = ("obsact", "pages", "data")


def task_type(world: str) -> str:
    """obsact (family A and the borrowed games), pages (family C), or data
    (the CRUD themes, ask-or-act, recovery, and the service worlds)."""
    if world in OBS_ACT:
        return "obsact"
    if world in PAGES:
        return "pages"
    return "data"


def fold_of(task_id: str, n: int = 2) -> int:
    """Which of n folds a training row is in, by episode: every turn of one
    episode (`..._ep123_t4`) lands in the same fold, so a draft model never
    drafts a turn of an episode it trained on."""
    episode = re.sub(r"_t\d+$", "", task_id)
    return zlib.crc32(episode.encode("utf-8")) % n


def parse_stages(spec: str) -> list[tuple[str, int, int]]:
    """"draft:4x4,refine:2x1" -> [("draft", 4, 4), ("refine", 2, 1)]:
    kind, layers, loops. Draft stages come first."""
    out = []
    for part in spec.split(","):
        kind, _, shape = part.strip().partition(":")
        layers, _, loops = shape.partition("x")
        if kind not in KINDS or not layers.isdigit() or not loops.isdigit():
            raise ValueError(f"bad stage {part!r}: want kind:LAYERSxLOOPS, kind in {KINDS}")
        out.append((kind, int(layers), int(loops)))
    kinds = [k for k, _, _ in out]
    if not out or "refine" in kinds and "draft" in kinds[kinds.index("refine"):]:
        raise ValueError(f"stages {spec!r}: draft stages first, then refine stages")
    return out


class Stage(nn.Module):
    """A stack of decoder layers with its own weights, applied `loops` times.
    Each iteration adds its own learned vector, so the stack knows which pass
    it is on (R8's loop_emb)."""

    def __init__(self, c, kind: str, layers: int, loops: int):
        super().__init__()
        self.kind, self.loops = kind, loops
        self.layers = nn.ModuleList([DecoderLayer(c) for _ in range(layers)])
        self.loop_emb = nn.Embedding(loops, c.d) if loops > 1 else None

    def step(self, x, mem, pad, mask, i: int):
        """Loop i alone (a committing draft reads out between loops)."""
        if self.loop_emb is not None:
            x = x + self.loop_emb.weight[i]
        for layer in self.layers:
            x = layer(x, mem, pad, mask)
        return x

    def forward(self, x, mem, pad, mask):
        for i in range(self.loops):
            x = self.step(x, mem, pad, mask, i)
        return x


def commit_rule(conf: torch.Tensor, open_: torch.Tensor, rounds_left: int,
                thr: float) -> torch.Tensor:
    """(B, C) bool: the open slots a committing draft fixes this round. Every
    open slot at least `thr` confident, and never fewer than an even share of
    what is still open (open / rounds left), most confident first; the last
    round takes everything. Padding and easy keywords go early, the slots it
    is unsure of -- the constants that must agree with each other -- last."""
    if rounds_left <= 1:
        return open_.clone()
    cand = conf.masked_fill(~open_, -1.0)
    n_open = open_.sum(1)
    share = (n_open + rounds_left - 1) // rounds_left
    k = torch.maximum(share, ((cand >= thr) & open_).sum(1))
    rank = cand.argsort(1, descending=True).argsort(1)
    return open_ & (rank < k.unsqueeze(1))


def program_confidence(tok: torch.Tensor, conf: torch.Tensor) -> torch.Tensor:
    """(B,) the lowest slot confidence over the program: every slot up to and
    including its first PAD, since one wrong slot makes a wrong program and
    where it stops is part of it."""
    C = tok.size(1)
    is_pad = tok == PAD_ID
    first = torch.where(is_pad.any(1), is_pad.float().argmax(1),
                        torch.full_like(tok[:, 0], C - 1))
    live = torch.arange(C, device=tok.device)[None] <= first[:, None]
    return conf.masked_fill(~live, 1.0).min(1).values


def shift_right(tgt: torch.Tensor) -> torch.Tensor:
    """Teacher-forcing input, as train.shift_right with MASK as the start."""
    return torch.cat([torch.full_like(tgt[:, :1], MASK_ID), tgt[:, :-1]], dim=1)


def stop_mask(tgt: torch.Tensor) -> torch.Tensor:
    """The left-to-right loss slots, as train.batch_loss: the program and one
    PAD as the stop signal."""
    keep = tgt != PAD_ID
    keep[:, 1:] |= (tgt[:, :-1] != PAD_ID) & (tgt[:, 1:] == PAD_ID)
    return keep


class StagedModel(StructuralModel):
    """StructuralModel's encoder and output head, with the staged decoder in
    place of the plain stack. `self.dec` holds the stages, so quant's
    loop-body count covers all of them."""

    def __init__(self, c):
        super().__init__(c)
        if not c.causal:
            raise ValueError("a staged model is causal: its last stage reads left to right")
        spec = parse_stages(c.stages)
        self.dec = nn.ModuleList([Stage(c, *s) for s in spec])
        self.loop_emb = None
        self.n_draft = sum(k == "draft" for k, _, _ in spec)
        self.n_refine = len(spec) - self.n_draft
        if not self.n_draft:
            raise ValueError("a staged model needs a draft stage; a refiner alone is the plain ar arm")
        d = c.d
        self.draft_norm = nn.LayerNorm(d)          # dec_norm stays the refiner's
        self.draft_tag = nn.Parameter(torch.zeros(d))
        self.conf_proj = nn.Linear(1, d)
        self.dec.apply(_init)
        _init(self.conf_proj)
        nn.init.normal_(self.draft_tag, std=0.02)
        self.force_exit = False
        self.force_refine = False
        # sample.ar_backoff: an early-exited draft that fails the host's compile
        # check is refined instead of backed off on (results/R28.md point 15)
        self.refine_on_fail = True
        self.last = None

    # -- what an expert trains (freeze_shared) -----------------------------

    def trainable_parts(self) -> list[nn.Parameter]:
        """Experts keep the encoder and the output head (keyword rows,
        pointer projections, slot table) fixed. train_parts "decoder" trains
        every stage; "draft" only the draft stages, the refiner staying
        shared."""
        if self.c.train_parts == "draft":
            mods = [s for s in self.dec if s.kind == "draft"] + [self.draft_norm]
            return [p for m in mods for p in m.parameters()]
        mods = [self.dec, self.draft_norm, self.dec_norm, self.conf_proj]
        return [p for m in mods for p in m.parameters()] + [self.draft_tag]

    def train(self, mode: bool = True):
        super().train(mode)
        if self.c.freeze_shared:
            # fixed modules run as at inference: no dropout in the encoder
            for name, m in self.named_children():
                if name not in ("dec", "draft_norm", "dec_norm", "conf_proj"):
                    m.eval()
        return self

    def _encode(self, inputs: dict):
        if self.c.freeze_shared:
            with torch.no_grad():
                return StructuralModel.encode_inputs(self, inputs)
        return StructuralModel.encode_inputs(self, inputs)

    # -- the stages --------------------------------------------------------

    def _canvas_in(self, table, canvas):
        x = table.gather(1, canvas.long().unsqueeze(-1).expand(-1, -1, self.c.d))
        return self.drop(x + self.out_pos + self.tag_emb.weight[TAG_CANVAS])

    def draft(self, inputs: dict, mem, table, rounds: list | None = None):
        """One trip through the draft stages from a blank canvas -> logits.

        With `commit` (plan step 2c) every loop of every draft stage is a
        round: read every slot out, fix the ones `commit_rule` picks as tokens
        (their vector replaces the blank one), and loop on. Each slot's output
        is its read-out from the round it was committed in. `rounds`, when
        given, collects (logits, open slots) per round for the loss."""
        blank = torch.full((table.size(0), self.c.canvas), MASK_ID,
                           dtype=torch.long, device=table.device)
        x = self._canvas_in(table, blank)
        if not self.c.commit:
            for st in self.dec:
                if st.kind == "draft":
                    x = st(x, mem.mem, mem.pad, None)
            return self.head(self.draft_norm(x), table, mem, inputs)
        steps = [(st, i) for st in self.dec if st.kind == "draft" for i in range(st.loops)]
        open_ = torch.ones_like(blank, dtype=torch.bool)
        when = torch.full_like(blank, -1)
        blank_vec = table[:, MASK_ID].unsqueeze(1)                      # (B, 1, d)
        out = None
        for r, (st, i) in enumerate(steps):
            x = st.step(x, mem.mem, mem.pad, None, i)
            logits = self.head(self.draft_norm(x), table, mem, inputs)
            if rounds is not None:
                rounds.append((logits, open_))
            with torch.no_grad():
                conf, tok = logits.float().softmax(-1).max(-1)
                take = commit_rule(conf, open_, len(steps) - r, self.c.commit_conf)
            out = logits if out is None else torch.where(take.unsqueeze(-1), logits, out)
            when = torch.where(take, r, when)
            if r < len(steps) - 1:
                vec = table.gather(1, tok.unsqueeze(-1).expand(-1, -1, self.c.d))
                x = x + (vec - blank_vec) * take.unsqueeze(-1).to(x.dtype)
            open_ = open_ & ~take
        self.commit_round = when
        return out

    def draft_rows(self, table, tok, conf):
        """The draft as extra context rows for the refiner: each slot's token
        vector (the same table the canvas uses), its position, and the
        draft's confidence in it."""
        x = table.gather(1, tok.unsqueeze(-1).expand(-1, -1, self.c.d))
        return (x + self.out_pos + self.draft_tag
                + self.conf_proj(conf.unsqueeze(-1).to(x.dtype)))

    def refine(self, inputs: dict, canvas, mem, table):
        x = self._canvas_in(table, canvas)
        rows = mem.draft_rows
        ctx = torch.cat([mem.mem, rows], 1)
        pad = torch.cat([mem.pad, torch.zeros(rows.shape[:2], dtype=torch.bool,
                                              device=rows.device)], 1)
        for st in self.dec:
            if st.kind == "refine":
                x = st(x, ctx, pad, self.causal_mask)
        return self.head(self.dec_norm(x), table, mem, inputs)

    # -- the sampler protocol ------------------------------------------------

    def encode_inputs(self, inputs: dict):
        return self.after_encode(inputs, self._encode(inputs))

    def after_encode(self, inputs: dict, mem):
        """The draft trip and the exit decision on an encoded context (an
        expert bundle encodes once and hands the context to one expert).

        `cal` (calibrate.py; an expert's own, else the config's) holds the
        exit threshold, read on the raw program confidence, and the
        temperature that makes confidence comparable across experts."""
        table = self.slot_table(mem)
        dl = self.draft(inputs, mem, table)
        probs = dl.float().softmax(-1)
        conf, tok = probs.max(-1)
        mem.table, mem.draft_logits = table, dl
        if self.n_refine:
            mem.draft_rows = self.draft_rows(table, tok, conf)
        cal = getattr(self, "cal", None) or {}
        pc = program_confidence(tok, conf)
        temp = cal.get("temp", self.c.draft_temp)
        mem.draft_pc = program_confidence(
            tok, (dl.float() / temp).softmax(-1).max(-1).values)
        thr = _exit_override(cal.get("thr", self.c.exit_threshold))
        if self.force_exit or not self.n_refine or thr == "always":
            exit_ = torch.ones_like(pc, dtype=torch.bool)
        elif self.force_refine or thr is None or thr == "never":
            exit_ = torch.zeros_like(pc, dtype=torch.bool)
        else:
            exit_ = pc >= float(thr)
        mem.exit = exit_
        self.last = {"draft": tok.detach(), "conf": float(pc[0]), "exit": bool(exit_[0]),
                     "cal_conf": float(mem.draft_pc[0])}
        if self.c.commit:
            self.last["commit_round"] = self.commit_round[0].tolist()
        return mem

    def decode(self, inputs: dict, canvas, mem=None, loops=None):
        if mem is None:
            mem = self.encode_inputs(inputs)
        if not self.n_refine or bool(mem.exit.all()):
            return mem.draft_logits
        out = self.refine(inputs, canvas, mem, mem.table)
        if bool(mem.exit.any()):
            out = torch.where(mem.exit.view(-1, 1, 1), mem.draft_logits, out)
        return out

    @contextmanager
    def exiting(self):
        """Read the draft out as the program, whatever the threshold."""
        was = self.force_exit
        self.force_exit = True
        try:
            yield self
        finally:
            self.force_exit = was

    @contextmanager
    def refining(self):
        """Run the refiner, whatever the threshold (sample.ar_backoff, when an
        exited draft fails the host's compile check)."""
        was = self.force_refine
        self.force_refine = True
        try:
            yield self
        finally:
            self.force_refine = was

    # -- training ------------------------------------------------------------

    def _kind_of(self, ids: torch.Tensor) -> torch.Tensor:
        """Joint id -> 0 keyword, 1 tool, 2 field, 3 constant, 4 register."""
        c = self.c
        edges = torch.tensor([c.n_kw, c.n_kw + c.max_tool, c.n_kw + c.max_tool + c.max_field,
                              c.n_kw + c.max_tool + c.max_field + c.max_const], device=ids.device)
        return torch.bucketize(ids, edges, right=True)

    def staged_loss(self, inputs: dict, tgt, pad_weight: float = 1.0):
        """(loss, n_scored, logits, scored mask), train.batch_loss's shape.
        The logits and mask are the refiner's (the draft's when there is no
        refiner), so train.evaluate's per-kind accuracy reads the output."""
        mem = self._encode(inputs)
        table = self.slot_table(mem)
        rounds = [] if self.c.commit else None
        dl = self.draft(inputs, mem, table, rounds)
        y = tgt.flatten()
        w = torch.where(y == PAD_ID, pad_weight, 1.0)
        if rounds:
            # a committing draft: each round's loss covers the slots still open
            # as it began, so a slot is trained up to the round that fixes it
            l_draft = 0.0
            for logits, open_ in rounds:
                per = F.cross_entropy(logits.flatten(0, 1).float(), y, reduction="none")
                wr = w * open_.flatten()
                l_draft = l_draft + (per * wr).sum() / wr.sum().clamp(min=1e-6)
            l_draft = l_draft / len(rounds)
        else:
            per = F.cross_entropy(dl.flatten(0, 1).float(), y, reduction="none")
            l_draft = (per * w).sum() / w.sum().clamp(min=1e-6)
        if self.c.pick:
            # the constant picker's own loss (model.py, plan step 2d)
            l_draft = l_draft + self.pick_loss(mem, inputs, tgt)
        keep = stop_mask(tgt)
        if not self.n_refine:
            return l_draft, int(keep.sum()), dl, keep
        with torch.no_grad():
            probs = dl.float().softmax(-1)
            if self.training and "xdraft_tok" in inputs:
                # a draft from a model that never saw this row (xdraft.py)
                tok0 = inputs["xdraft_tok"].long()
                conf0 = inputs["xdraft_conf"].float()
            else:
                tok0, conf0 = probs.argmax(-1), None
            tok = tok0
            if self.training and (self.c.draft_noise > 0 or self.c.draft_swap > 0):
                present = self.present_mask(inputs, dl.device)            # (B, J)
                u = torch.rand(dl.shape, device=dl.device)
                if self.c.draft_noise > 0:
                    rnd = u.masked_fill(~present.unsqueeze(1), -1.0).argmax(-1)
                    swap = torch.rand(tok.shape, device=tok.device) < self.c.draft_noise
                    tok = torch.where(swap, rnd, tok)
                if self.c.draft_swap > 0:
                    # the draft's real mistakes look like this: another of this
                    # task's tools, fields, constants or registers in a pointer
                    # slot. On training rows the draft is nearly always right,
                    # so without them the refiner learns to copy it (R28).
                    kind = self._kind_of(tok)                             # (B, C)
                    same = self._kind_of(torch.arange(dl.size(-1), device=dl.device))
                    ok = present.unsqueeze(1) & (same[None, None] == kind.unsqueeze(-1))
                    alt = u.masked_fill(~ok, -1.0).argmax(-1)
                    swap = (torch.rand(tok.shape, device=tok.device) < self.c.draft_swap) & (kind > 0)
                    tok = torch.where(swap, alt, tok)
            conf = probs.gather(-1, tok.unsqueeze(-1)).squeeze(-1)
            if conf0 is not None:
                # the stored model's confidence where its token survived the noise
                conf = torch.where(tok == tok0, conf0, conf)
        mem.draft_rows = self.draft_rows(table, tok, conf)
        rl = self.refine(inputs, shift_right(tgt), mem, table)
        l_ref = F.cross_entropy(rl[keep].float(), tgt[keep])
        return l_draft + l_ref, int(keep.sum()), rl, keep


# -- experts (plan step 3) ------------------------------------------------------

def route_features(mem, inputs: dict, c) -> torch.Tensor:
    """(B, 3d): what the router reads from the encoder's output -- the mean
    request vector, the mean tool vector, and the mean register vector
    (where an earlier call's results, errors included, come back). Region A
    is laid out tools | fields | constants | registers | ... | request
    tokens (StructuralModel.encode_turn)."""
    x, pad = mem.mem, mem.pad
    live = (~pad).float().unsqueeze(-1)

    def mean(a, b):
        w = live[:, a:b]
        return (x[:, a:b] * w).sum(1) / w.sum(1).clamp(min=1.0)

    n_reg = inputs["reg_tok"].size(1) if inputs.get("reg_tok") is not None else 0
    n_req = inputs["req_tok"].size(1) if c.req_words else 1
    L = x.size(1)
    return torch.cat([mean(L - n_req, L), mean(0, c.max_tool),
                      mean(mem.n_ptr, mem.n_ptr + n_reg)], -1)


class Router(nn.Module):
    """Picks one expert per request from `route_features`."""

    def __init__(self, d: int, names: list[str], hidden: int = 128):
        super().__init__()
        self.names = list(names)
        self.net = nn.Sequential(nn.LayerNorm(3 * d), nn.Linear(3 * d, hidden),
                                 nn.GELU(), nn.Linear(hidden, len(names)))

    def forward(self, feats):
        return self.net(feats)


class ExpertBundle(nn.Module):
    """One expert per task type on one shared encoder, and the router.

    Every expert is a StagedModel trained from the same shared model with
    the encoder and output head fixed (train.py --freeze-shared), so their
    encoders are the same weights: the bundle encodes once with the first
    and hands the context to the expert the router picks. When the router's
    top choice is below `margin`, the top two experts both draft and the one
    with the higher calibrated draft confidence writes the program.

    `oracle` (set per task by evaluate.py --oracle-route) bypasses the
    router with the task's true type."""

    def __init__(self, experts: dict, router: Router | None, margin: float = 0.0):
        super().__init__()
        self.names = list(experts)
        self.experts = nn.ModuleList(experts.values())
        self.router = router
        self.margin = margin
        self.oracle = None
        self.c = self.experts[0].c
        for e in self.experts:
            e.c = self.c          # one config: play.py's widening reaches every expert
        self.last = None

    def encode_inputs(self, inputs: dict):
        mem = self.experts[0]._encode(inputs)
        picks, probs = self.route(inputs, mem)
        if len(picks) == 1:
            k = picks[0]
            mem = self.experts[k].after_encode(inputs, mem)
        else:
            # same encoder output, two drafts; keep the surer one
            mems = [self.experts[k].after_encode(inputs, _copy_mem(mem)) for k in picks]
            j = max(range(len(picks)), key=lambda i: float(mems[i].draft_pc[0]))
            k, mem = picks[j], mems[j]
        mem.expert = k
        self.last = {**self.experts[k].last, "expert": self.names[k],
                     "route_p": probs, "tried": [self.names[i] for i in picks]}
        return mem

    def route(self, inputs: dict, mem) -> tuple[list[int], list[float]]:
        if self.oracle is not None:
            return [self.names.index(self.oracle)], []
        if self.router is None:
            raise RuntimeError("an expert bundle without a router needs an oracle route")
        with torch.no_grad():
            logits = self.router(route_features(mem, inputs, self.c)).float()[0]
        # only the types this bundle has an expert for (a router can know
        # more types than the bundle holds, as in the bolt-on test)
        p = logits[[self.router.names.index(n) for n in self.names]].softmax(-1)
        order = p.argsort(descending=True).tolist()
        if float(p[order[0]]) < self.margin and len(order) > 1:
            return order[:2], p.tolist()
        return order[:1], p.tolist()

    def decode(self, inputs: dict, canvas, mem=None, loops=None):
        if mem is None:
            mem = self.encode_inputs(inputs)
        return self.experts[mem.expert].decode(inputs, canvas, mem)

    @contextmanager
    def exiting(self):
        was = [e.force_exit for e in self.experts]
        for e in self.experts:
            e.force_exit = True
        try:
            yield self
        finally:
            for e, w in zip(self.experts, was):
                e.force_exit = w

    def set_oracle(self, world: str | None):
        self.oracle = None if world is None else task_type(world)


def _copy_mem(mem):
    import copy
    return copy.copy(mem)


def save_bundle(path, experts: dict, router: Router | None, margin: float = 0.0) -> None:
    """experts: {name: (checkpoint path, cal dict or None)}. Checks that every
    expert shares the first one's encoder and output head."""
    import torch as _t
    out = {"bundle": True, "names": list(experts), "experts": {}, "margin": margin}
    first = None
    for name, (ckpt, cal) in experts.items():
        ck = _t.load(ckpt, map_location="cpu")
        m = StagedModel(_cfg(ck["cfg"]))
        m.load_state_dict(ck["model"])
        shared = {k: v for k, v in ck["model"].items() if not _expert_key(k)}
        if first is None:
            first = shared
        elif any(not _t.equal(first[k], shared[k]) for k in first):
            raise SystemExit(f"{name}: its encoder or head differs from {list(experts)[0]}'s")
        out["experts"][name] = {"cfg": ck["cfg"], "model": ck["model"], "cal": cal or {}}
    if router is not None:
        out["router"] = {"state": router.state_dict(), "names": router.names,
                         "d": router.net[1].in_features // 3,
                         "hidden": router.net[1].out_features}
    _t.save(out, path)


def _expert_key(k: str) -> bool:
    return k.startswith(("dec.", "draft_norm.", "dec_norm.", "conf_proj.")) or k == "draft_tag"


def _cfg(d: dict):
    from model import Config
    return Config(**{k: v for k, v in d.items() if k in Config.__dataclass_fields__})


def load_bundle(ck: dict, device) -> ExpertBundle:
    experts = {}
    for name in ck["names"]:
        e = ck["experts"][name]
        m = StagedModel(_cfg(e["cfg"]))
        m.load_state_dict(e["model"])
        m.cal = e.get("cal") or None
        experts[name] = m
    router = None
    if "router" in ck:
        r = ck["router"]
        router = Router(r["d"], r["names"], r["hidden"])
        router.load_state_dict(r["state"])
    b = ExpertBundle(experts, router, ck.get("margin", 0.0)).to(device)
    b.eval()
    return b


def _exit_override(thr):
    """STAGED_EXIT=never|always|<float> overrides the checkpoint's threshold,
    for drivers that load a model without a flag for it (play.py)."""
    env = os.environ.get("STAGED_EXIT")
    if env:
        return env if env in ("never", "always") else float(env)
    return thr
