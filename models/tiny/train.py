"""Train either arm. The arm is one flag; everything else is identical.

  python train.py --arm diffusion --cache data_cache_struct --epochs 5
  python train.py --arm ar        --cache data_cache_struct --epochs 5

The binding (flat or structural) comes from the cache's config.json, so the
R3 baseline and the structural model differ by the one flag given to prep.py.

Both read the same cache and write a checkpoint plus a JSONL log of every
evaluation, so the curves can be drawn afterwards without re-running anything.
Every evaluation records, beside the validation loss, the accuracy at each
kind of canvas slot (keyword, tool, field, constant, register), measured
directly: the diffusion arm is shown the reference canvas with every symbol
slot hidden and must fill them in one pass; the control is scored under
teacher forcing. Chance for tools and fields is about 6% and 5%. This is what
tells "undertrained" from "unable" on the curve (results/R3.md section 5).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import quant
from model import Config, build_model, mask_canvas
from prep import STRUCT_KEYS, cache_keys
from tok import OutVocab

MASK_ID, PAD_ID = 1, 0
KIND_NAMES = ("kw", "tool", "field", "const", "reg")


# -- data -----------------------------------------------------------------

def load_split(cache: Path, name: str, binding: str, limit: int | None = None,
               types: set | None = None, fold: tuple[int, int] | None = None,
               xdraft: Path | None = None) -> TensorDataset:
    """Memory-mapped, and sliced before anything is read into RAM.

    The training tensors are hundreds of MB. Holding them resident competes
    with the activations, and on a machine without much headroom the process
    starts swapping. That does not surface as an out-of-memory error -- it
    surfaces as training being mysteriously slow, which is much harder to
    recognise. mmap leaves the tensors on disk, and slicing first means a short
    run pays for only the rows it uses.
    """
    d = torch.load(cache / f"{name}.pt", mmap=True)
    if binding == "flat":
        # `sym` is absent from caches built before the pointer head existed; a run
        # without it simply trains the plain model.
        cols = (d["src"], d["pad"], d["tgt"],
                d.get("sym", torch.full_like(d["tgt"][:, :1], -1).expand(-1, 1)))
    else:
        # a cache built before step 4 has no register region; a
        # zero-width column makes the model skip it rather than fail
        n = d["tgt"].size(0)
        empty = {"reg_tok": torch.zeros((n, 0, d["tool_tok"].size(2)),
                                        dtype=d["tool_tok"].dtype),
                 "n_reg_bound": torch.zeros(n, dtype=d["n_tool"].dtype)}
        # a split cache (description-reading step 2) carries more columns
        keys = cache_keys(d)
        cols = tuple(d[k] if k in d else empty[k] for k in keys)
    if xdraft:
        # cross-fitted drafts, one per row of this split (xdraft.py), as two
        # more columns the staged refiner reads in training
        x = torch.load(xdraft)
        n = cols[0].size(0)
        if x["tok"].size(0) != n:
            raise ValueError(f"{xdraft}: {x['tok'].size(0)} drafts for {n} {name} rows")
        cols, keys = cols + (x["tok"], x["conf"]), tuple(keys) + ("xdraft_tok", "xdraft_conf")
    if types or fold:
        from staged import fold_of, task_type
        meta = json.loads((cache / f"{name}_meta.json").read_text(encoding="utf-8"))
        # one expert's rows (staged.task_type, from each row's world), and/or
        # one fold of them (staged.fold_of: a cross-fitting draft model's half)
        idx = torch.tensor([i for i, m in enumerate(meta)
                            if (not types or task_type(m["world"]) in types)
                            and (not fold or fold_of(m["task_id"], fold[1]) == fold[0])])
        cols = tuple(t[idx] for t in cols)
    if limit:
        cols = tuple(t[:limit].clone() for t in cols)
    ds = TensorDataset(*cols)
    ds.keys = STRUCT_KEYS if binding == "flat" else keys
    return ds


def unpack(batch, binding: str, out_vocab: int, keys=STRUCT_KEYS):
    """A batch from the loader -> (inputs dict, target canvas)."""
    if binding == "flat":
        src, pad, tgt, sym = batch
        sym = sym if sym.size(1) == out_vocab else None
        return {"src": src, "pad": pad, "sym": sym}, tgt
    inputs = dict(zip(keys, batch))
    tgt = inputs.pop("tgt").long()
    return inputs, tgt


def flat_kinds(ov: OutVocab) -> torch.Tensor:
    """Slot kind per flat-vocabulary id, so the flat baseline logs the same
    per-kind accuracies as the structural model."""
    kinds = []
    for t in ov.itos:
        if re.fullmatch(r"T\d+", t):
            kinds.append(1)
        elif re.fullmatch(r"F\d+", t):
            kinds.append(2)
        elif re.fullmatch(r"[CSNBDI]\d+", t):
            kinds.append(3)
        elif re.fullmatch(r"r\d+\.?", t):
            kinds.append(4)
        else:
            kinds.append(0)
    return torch.tensor(kinds)


# -- objective ------------------------------------------------------------

def shift_right(tgt: torch.Tensor, bos: int) -> torch.Tensor:
    """Teacher forcing input for the control arm."""
    return torch.cat([torch.full_like(tgt[:, :1], bos), tgt[:, :-1]], dim=1)


def called_tools(tgt: torch.Tensor, n_kw: int, max_tool: int) -> torch.Tensor:
    """(B, max_tool) bool: which of the row's tools its reference calls --
    the positives of the contrastive loss, straight from the canvas."""
    t = tgt.long() - n_kw
    ok = (t >= 0) & (t < max_tool)
    out = torch.zeros(tgt.size(0), max_tool, dtype=torch.bool, device=tgt.device)
    rows = torch.arange(tgt.size(0), device=tgt.device).unsqueeze(1).expand_as(t)
    out[rows[ok], t[ok]] = True
    return out


def stage_losses(model, mem, inputs, tgt, teacher=None) -> dict:
    """The two losses kept on from pretraining (plan §3), for the
    description and name stages: contrastive (the request toward the tools
    its reference calls, away from every other tool of the task, twins
    included) and relational distillation (the stage's cosines among the
    request and the task's tools matched to the teacher's)."""
    from stages import multi_positive_nce, relational_loss, twin_nce
    c = model.c
    st = mem.stage
    live = torch.arange(c.max_tool, device=tgt.device)[None] < inputs["n_tool"].long()[:, None]
    pos = called_tools(tgt, c.n_kw, c.max_tool)
    out = {}
    s = model.stage_scale.exp().clamp(max=100)
    for part in ("desc", "name"):
        q = F.normalize(st[f"q_{part}"], dim=-1)
        k = F.normalize(st[part], dim=-1)
        lg = s * torch.einsum("bw,bmw->bm", q, k)
        out[f"nce_{part}"] = multi_positive_nce(lg, pos, live)
        if "sig_group" in inputs:
            out[f"nce_twin_{part}"] = twin_nce(lg, pos, live, inputs["sig_group"].long())
        if teacher is not None and f"t_{part}" in inputs:
            idx = inputs[f"t_{part}"].long()
            ridx = inputs["t_req"].long()
            tv = teacher[idx.clamp(min=0)]
            tq = teacher[ridx.clamp(min=0)].unsqueeze(1)
            vec = torch.cat([st[f"q_{part}"].unsqueeze(1), st[part]], 1)
            tt = torch.cat([tq, tv], 1)
            lv = torch.cat([(ridx >= 0).unsqueeze(1), live & (idx >= 0)], 1)
            out[f"rel_{part}"] = relational_loss(vec, tt, lv)
    return out


def batch_loss(model, inputs, tgt, arm: str, generator=None, pad_weight: float = 1.0,
               loops: int | None = None, aux: dict | None = None):
    """Returns (loss, n_predicted, logits, scored_mask).

    `loops` overrides the configured loop count for this batch, which is how
    `--rand-loops` trains one checkpoint to serve every effort setting.
    `aux` ({"nce": w, "rel": w, "teacher": table}) adds the split encoder's
    stage losses to the program loss; the parts land in aux["last"].
    """
    if hasattr(model, "staged_loss"):
        # the staged decoder's draft + refiner objective (staged.py)
        return model.staged_loss(inputs, tgt, pad_weight)
    mem = None
    extra = 0.0
    if aux and getattr(model.c, "split", False) and (aux.get("nce") or aux.get("rel")):
        mem = model.encode_inputs(inputs)
        parts = stage_losses(model, mem, inputs, tgt, aux.get("teacher"))
        extra = sum(aux.get(k.split("_")[0], 0.0) * v for k, v in parts.items())
        aux["last"] = {k: float(v) for k, v in parts.items()}
    if arm == "diffusion":
        canvas, loss_mask, _ = mask_canvas(tgt, MASK_ID, generator)
        logits = model.decode(inputs, canvas, loops=loops, mem=mem)
        # Loss only on the slots that were hidden. The visible ones are free
        # and scoring them would let the model earn reward for copying.
        y = tgt[loss_mask]
        per = F.cross_entropy(logits[loss_mask], y, reduction="none")
        # Programs are about half the canvas, so most slots are padding and an
        # undertrained model collapses to predicting padding everywhere -- an
        # empty program, which is the single cheapest way to be wrong. Knowing
        # where a program ends is genuinely part of the task, so padding keeps a
        # share of the loss, but the share is a knob rather than an accident.
        if pad_weight != 1.0:
            w = torch.where(y == PAD_ID, pad_weight, 1.0)
            loss = (per * w).sum() / w.sum().clamp(min=1e-6)
        else:
            loss = per.mean()
        return loss + extra, int(loss_mask.sum()), logits, loss_mask
    logits = model.decode(inputs, shift_right(tgt, bos=MASK_ID), loops=loops, mem=mem)
    # Padding slots after the program end carry no information; scoring
    # them would reward predicting PAD forever. The diffusion arm keeps
    # them because knowing where a program stops is part of its job, so
    # the control gets one extra PAD as the stop signal and nothing more.
    keep = tgt != PAD_ID
    keep[:, 1:] |= (tgt[:, :-1] != PAD_ID) & (tgt[:, 1:] == PAD_ID)
    loss = F.cross_entropy(logits[keep], tgt[keep])
    return loss + extra, int(keep.sum()), logits, keep


# -- evaluation -----------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, arm, binding, kind_of, device, out_vocab, seed=1234,
             pad_weight=1.0, keys=STRUCT_KEYS) -> dict:
    """Validation loss plus per-slot-kind accuracy.

    `kind_of(ids)` maps a tensor of canvas ids to kind indices 0..4.

    Accuracy per kind is the grounding measurement. For the diffusion arm the
    reference canvas is shown with every pointer slot (tool, field, constant,
    register) hidden and the model fills them in one pass; keyword accuracy
    comes from the ordinary random-mask pass. For the control every slot is
    scored under teacher forcing. Chance for a tool slot is 1/n_tools.
    """
    model.eval()
    g = torch.Generator(device=device).manual_seed(seed)
    tot, n = 0.0, 0
    hit, cnt = defaultdict(int), defaultdict(int)
    chance = defaultdict(float)
    for batch in loader:
        batch = [t.to(device) for t in batch]
        inputs, tgt = unpack(batch, binding, out_vocab, keys)
        loss, k, logits, scored = batch_loss(model, inputs, tgt, arm, g, pad_weight)
        tot += loss.item() * k
        n += k
        kinds = kind_of(tgt)
        if arm == "diffusion":
            # Keyword accuracy from the random-mask pass ...
            pred = logits.argmax(-1)
            m = scored & (kinds == 0)
            hit["kw"] += int((pred[m] == tgt[m]).sum())
            cnt["kw"] += int(m.sum())
            # ... and symbol accuracy from the symbols-hidden pass.
            sym = kinds != 0
            canvas = torch.where(sym, torch.full_like(tgt, MASK_ID), tgt)
            pred = model.decode(inputs, canvas).argmax(-1)
            scored = sym
        else:
            pred = logits.argmax(-1)
        for ki, name in enumerate(KIND_NAMES):
            if arm == "diffusion" and ki == 0:
                continue
            m = scored & (kinds == ki)
            hit[name] += int((pred[m] == tgt[m]).sum())
            cnt[name] += int(m.sum())
        if binding == "structural":
            # Exact chance: a uniform pick over this task's declared symbols.
            for name, key in (("tool", "n_tool"), ("field", "n_field"), ("const", "n_const")):
                m = scored & (kinds == KIND_NAMES.index(name))
                per_row = m.sum(1).float() / inputs[key].float().clamp(min=1)
                chance[name] += float(per_row.sum())
    model.train()
    rec = {"val_loss": tot / max(n, 1)}
    for name in KIND_NAMES:
        rec[f"acc_{name}"] = hit[name] / cnt[name] if cnt[name] else None
        rec[f"n_{name}"] = cnt[name]
    for name in ("tool", "field", "const"):
        if cnt[name] and chance[name]:
            rec[f"chance_{name}"] = chance[name] / cnt[name]
    return rec


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["diffusion", "ar"], required=True)
    ap.add_argument("--cache", default="data_cache_struct")
    ap.add_argument("--binding", choices=["structural", "flat"], default=None,
                    help="must match the cache; taken from the cache when omitted")
    ap.add_argument("--out", default=None, help="run directory; default runs/<arm>_s<seed>")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--ff", type=int, default=None,
                    help="feed-forward width; 4x the model width by default, "
                         "which is what d=256 already used")
    ap.add_argument("--enc-layers", type=int, default=None,
                    help="flat: the encoder; structural: the turn encoder. "
                         "Default 3 flat, 2 structural")
    ap.add_argument("--line-layers", type=int, default=2, help="structural: per-line encoder")
    ap.add_argument("--graph-layers", type=int, default=1, help="structural: schema graph pass")
    ap.add_argument("--dec-layers", type=int, default=4)
    ap.add_argument("--dec-loops", type=int, default=1,
                    help="apply the decoder stack L times with the same weights: "
                         "step 2's compute knob, free on the NPU")
    ap.add_argument("--loop-emb", action="store_true",
                    help="give each loop iteration its own learned bias vector")
    ap.add_argument("--rand-loops", type=int, default=0,
                    help="sample the loop count from 1..N per batch, so one "
                         "checkpoint serves every effort setting (decision 9)")
    ap.add_argument("--weights", choices=list(quant.MODES), default="fp",
                    help="step 3: the format the transformer stacks train in. "
                         "int8/u4/tern are 4M/8M/16M resident in 4 MB of "
                         "memory tiles")
    ap.add_argument("--act-bits", type=int, default=8,
                    help="activation bits at every quantised matmul; 0 to leave "
                         "activations in floating point")
    ap.add_argument("--quant-ends", choices=["fp", "int8"], default="int8",
                    help="format of the embeddings and the output heads")
    ap.add_argument("--pointer", action="store_true",
                    help="flat only: decide symbol slots by pointing at the input")
    ap.add_argument("--dropout", type=float, default=0.1)
    # description-reading plan step 2 / step 4
    ap.add_argument("--pack", action="store_true",
                    help="run only occupied line slots, bucketed by length "
                         "(output-identical; saves the 88%% padding)")
    ap.add_argument("--split", action="store_true",
                    help="the split encoder: signature, description and name "
                         "stages, gated at the pointer. Needs a --split cache")
    ap.add_argument("--sig-w", type=int, default=128)
    ap.add_argument("--sig-layers", type=int, default=2)
    ap.add_argument("--desc-w", type=int, default=256)
    ap.add_argument("--desc-layers", type=int, default=4)
    ap.add_argument("--name-w", type=int, default=128)
    ap.add_argument("--name-layers", type=int, default=2)
    ap.add_argument("--stage-pool", choices=["cls", "mean"], default="cls")
    ap.add_argument("--stages-from", default=None, metavar="PT",
                    help="description and name stage weights from "
                         "stage_pretrain.py")
    ap.add_argument("--freeze-stages", action="store_true",
                    help="keep the loaded stages fixed (arm SPf)")
    ap.add_argument("--stage-lr-mult", type=float, default=1.0,
                    help="learning-rate multiplier on the description and "
                         "name stages (arm SPt uses < 1)")
    ap.add_argument("--lam-nce", type=float, default=0.0,
                    help="weight of the stages' contrastive loss")
    ap.add_argument("--lam-rel", type=float, default=0.0,
                    help="weight of the stages' relational distillation loss "
                         "(needs teacher.pt in the cache)")
    ap.add_argument("--reader", action="store_true",
                    help="a frozen external reader's vectors (the cache's "
                         "reader.pt, reader_table.py) for tool descriptions and "
                         "the request, and their cosine in the tool pointer "
                         "(npu-planner.md phase 2)")
    ap.add_argument("--reader-lines", action="store_true",
                    help="with --reader: the reader's vectors for every constant, "
                         "every field and the request in chunks, and a reading "
                         "term in the constant pointer (C0, tiny-general-agent-"
                         "menu.md; needs a cache built with prep.py --reader-lines)")
    ap.add_argument("--reader-file", default="reader.pt",
                    help="with --reader: which of the cache's reader tables "
                         "(reader_role.pt: the role-aware reader, role-aware-reader.md)")
    ap.add_argument("--no-req-words", action="store_true",
                    help="with --reader-lines: drop the request's own tokens, so "
                         "the request reaches the planner only as reader vectors")
    # the staged decoder (staged.py, .claude/plans/staged-decoder-experts.md)
    ap.add_argument("--stages", default="",
                    help="draft and refine stages, each kind:LAYERSxLOOPS, e.g. "
                         "draft:4x4,refine:2x1; replaces --dec-layers/--dec-loops. "
                         "Needs --arm ar")
    ap.add_argument("--draft-noise", type=float, default=0.1,
                    help="share of draft slots swapped for random ones before "
                         "the refiner reads them, in training")
    ap.add_argument("--draft-swap", type=float, default=0.0,
                    help="share of the draft's pointer slots swapped for another of "
                         "this task's tools/fields/constants/registers (same kind) "
                         "before the refiner reads them, in training")
    ap.add_argument("--init-from", default=None, metavar="PT",
                    help="start from this checkpoint's weights (an expert from "
                         "the shared model)")
    ap.add_argument("--freeze-shared", action="store_true",
                    help="experts: keep the encoder and the output head fixed; "
                         "train only the stages --train-parts names")
    ap.add_argument("--train-parts", choices=["decoder", "draft"], default="decoder")
    ap.add_argument("--types", default=None,
                    help="comma list of task types (staged.task_type): train and "
                         "validate on those rows only")
    ap.add_argument("--fold", default=None, metavar="K/N",
                    help="train on fold K of N only (staged.fold_of, by episode): "
                         "a cross-fitting draft model's half (plan step 2b)")
    ap.add_argument("--xdraft", default=None, metavar="PT",
                    help="cross-fitted drafts for the training rows (xdraft.py): "
                         "the staged refiner trains on them instead of its own "
                         "draft stage's")
    ap.add_argument("--min-steps", type=int, default=0,
                    help="raise --epochs until training takes at least this many "
                         "steps (small experts)")
    ap.add_argument("--pad-weight", type=float, default=1.0,
                    help="weight on padding slots in the diffusion loss")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="use only the first N training rows (smoke tests)")
    ap.add_argument("--limit-val", type=int, default=256,
                    help="validation rows per evaluation; the full split is "
                         "1415 rows and evaluating all of them costs more than "
                         "the training steps between evaluations")
    ap.add_argument("--eval-every", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    if args.ff is None:
        args.ff = 4 * args.d

    torch.manual_seed(args.seed)
    cache = Path(args.cache)
    meta = json.loads((cache / "config.json").read_text(encoding="utf-8"))
    binding = meta.get("binding", "flat")
    if args.binding and args.binding != binding:
        raise SystemExit(f"--binding {args.binding} but the cache was prepared with "
                         f"binding={binding}; prep.py decides the binding")
    args.binding = binding
    if args.enc_layers is None:
        args.enc_layers = 3 if binding == "flat" else 2

    run = Path(args.out or f"runs/{args.arm}_s{args.seed}")
    run.mkdir(parents=True, exist_ok=True)

    if binding == "flat":
        ov = OutVocab.load(cache / "out_vocab.json")
        out_vocab = len(ov)
        cfg = Config(
            in_vocab=meta["in_vocab"], out_vocab=out_vocab, d=args.d, ff=args.ff,
            enc_layers=args.enc_layers, dec_layers=args.dec_layers,
            dec_loops=args.dec_loops, dropout=args.dropout,
            loop_emb=args.loop_emb, rand_loops=args.rand_loops,
            weights=args.weights, act_bits=args.act_bits, quant_ends=args.quant_ends,
            max_in=meta["max_in"], canvas=meta["canvas"],
            causal=(args.arm == "ar"), pointer=args.pointer, binding="flat",
        )
        fk = flat_kinds(ov)
        kind_of = lambda ids: fk.to(ids.device)[ids]           # noqa: E731
    else:
        from canvas import Layout
        layout = Layout.from_dict(meta["layout"])
        out_vocab = layout.size
        if args.stages and args.arm != "ar":
            raise SystemExit("--stages needs --arm ar: the staged model's last stage reads left to right")
        cfg = Config(
            in_vocab=meta["in_vocab"], out_vocab=out_vocab, d=args.d, ff=args.ff,
            enc_layers=args.enc_layers, dec_layers=0 if args.stages else args.dec_layers,
            line_layers=args.line_layers, graph_layers=args.graph_layers,
            dec_loops=args.dec_loops, dropout=args.dropout, canvas=meta["canvas"],
            loop_emb=args.loop_emb, rand_loops=args.rand_loops,
            weights=args.weights, act_bits=args.act_bits, quant_ends=args.quant_ends,
            causal=(args.arm == "ar"), binding="structural",
            in_pad=meta["in_pad"], max_line=meta["max_line"], max_req=meta["max_req"],
            n_kw=layout.n_kw, max_tool=layout.max_tool, max_field=layout.max_field,
            max_const=layout.max_const, n_reg=layout.n_reg,
            pack=args.pack, split=args.split,
            sig_w=args.sig_w, sig_layers=args.sig_layers, desc_w=args.desc_w,
            desc_layers=args.desc_layers, name_w=args.name_w,
            name_layers=args.name_layers, stage_pool=args.stage_pool,
            max_sig=meta.get("max_sig", 48), max_desc=meta.get("max_desc", 64),
            max_name=meta.get("max_name", 16), reader=args.reader,
            reader_lines=args.reader_lines, max_chunk=meta.get("max_chunk", 0),
            req_words=not args.no_req_words, reader_file=args.reader_file,
            stages=args.stages, draft_noise=args.draft_noise, draft_swap=args.draft_swap,
            freeze_shared=args.freeze_shared, train_parts=args.train_parts,
        )
        if args.freeze_shared and not (args.stages and args.init_from):
            raise SystemExit("--freeze-shared trains an expert: it needs --stages and --init-from")
        if args.split and not meta.get("split"):
            raise SystemExit("--split needs a cache built with prep.py --split")
        if args.reader and not (cache / args.reader_file).exists():
            raise SystemExit(f"--reader needs the cache's {args.reader_file} (reader_table.py)")
        if args.reader_lines and not (args.reader and meta.get("max_chunk")):
            raise SystemExit("--reader-lines needs --reader and a cache built with "
                             "prep.py --reader-lines")
        if args.no_req_words and not args.reader_lines:
            raise SystemExit("--no-req-words needs --reader-lines")
        kind_of = layout.kind_tensor
        if args.pointer:
            raise SystemExit("--pointer is the flat binding's head; structural always points")
    device = torch.device(args.device)
    model = build_model(cfg)
    if binding == "flat" and args.pointer:
        model.set_symbol_ids([i for i, t in enumerate(ov.itos)
                              if re.fullmatch(r"[TFCSNBDI]\d+", t)])
    if args.split and args.stages_from:
        from stage_pretrain import load_stages
        load_stages(model, args.stages_from)
    if args.freeze_stages:
        for p_ in model.stage_params():
            p_.requires_grad_(False)
        model.frozen_stages = True
    if args.init_from:
        model.load_state_dict(torch.load(args.init_from, map_location="cpu")["model"])
    if args.freeze_shared:
        keep = {id(p_) for p_ in model.trainable_parts()}
        for p_ in model.parameters():
            p_.requires_grad_(id(p_) in keep)
    model = model.to(device)
    if args.reader:
        model.set_reader_table(torch.load(cache / args.reader_file)["table"].to(device))
    aux = None
    if args.split and (args.lam_nce or args.lam_rel):
        teacher = None
        if args.lam_rel:
            teacher = torch.load(cache / "teacher.pt")["table"].float().to(device)
        aux = {"nce": args.lam_nce, "rel": args.lam_rel, "teacher": teacher}

    types = set(args.types.split(",")) if args.types else None
    fold = tuple(int(v) for v in args.fold.split("/")) if args.fold else None
    if args.xdraft and not args.stages:
        ap.error("--xdraft needs --stages with a refine stage")
    train_ds = load_split(cache, "train", binding, args.limit_train, types, fold,
                          Path(args.xdraft) if args.xdraft else None)
    val_ds = load_split(cache, "val", binding, args.limit_val, types)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch)

    if args.min_steps and len(train_dl) * args.epochs < args.min_steps:
        args.epochs = math.ceil(args.min_steps / len(train_dl))
    steps = len(train_dl) * args.epochs
    if args.freeze_shared:
        groups = [p_ for p_ in model.parameters() if p_.requires_grad]
    elif args.split:
        stage_ids = {id(p_) for p_ in model.stage_params()}
        groups = [{"params": [p_ for p_ in model.parameters()
                              if id(p_) not in stage_ids and p_.requires_grad]},
                  {"params": [p_ for p_ in model.stage_params() if p_.requires_grad],
                   "lr": args.lr * args.stage_lr_mult}]
        groups = [g_ for g_ in groups if g_["params"]]
    else:
        groups = model.parameters()
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(steps - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    log = open(run / "log.jsonl", "a", encoding="utf-8")
    # The loop body is what has to fit in the NPU's 4 MB of memory tiles, and it
    # is not the model's parameter count: the encoders run once per world or once
    # per turn, the decoder stack runs L times per turn and is the resident part.
    body = quant.loop_body_params(model)
    budget = {"loop_body_params": body,
              "resident_bytes": quant.resident_bytes(model, args.weights),
              "resident_budget_bytes": 4 << 20,
              **getattr(model, "quant", {})}
    (run / "config.json").write_text(json.dumps(
        {**vars(args), "params": model.n_params(), **budget,
         "git_sha": git_sha(), **meta}, indent=1),
        encoding="utf-8")

    # flush: stdout is block-buffered when redirected to a file, so without this
    # a redirected run shows no sign of life until the first step print, and a
    # run that died at startup looks exactly like one that is working.
    print(f"arm={args.arm} binding={binding} params={model.n_params()/1e6:.2f}M "
          f"device={device} train={len(train_ds)} steps={steps}", flush=True)
    print(f"weights={args.weights} act_bits={args.act_bits if args.weights != 'fp' else 0} "
          f"loops={args.dec_loops} loop_body={body/1e6:.2f}M params "
          f"= {budget['resident_bytes']/(1<<20):.2f} MB resident "
          f"of 4.00 MB{'  OVER BUDGET' if budget['resident_bytes'] > (4 << 20) else ''}",
          flush=True)

    def eval_and_log(step, epoch, t0):
        rec = evaluate(model, val_dl, args.arm, binding, kind_of, device, out_vocab,
                       pad_weight=args.pad_weight, keys=val_ds.keys)
        if aux and aux.get("last"):
            rec.update({f"train_{k}": v for k, v in aux["last"].items()})
        rec = {"step": step, "epoch": epoch, **rec, "secs": time.time() - t0}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        accs = " ".join(f"{k[4:]} {v:.2f}" for k, v in rec.items()
                        if k.startswith("acc_") and v is not None)
        print(f"  == step {step} val_loss {rec['val_loss']:.4f}  acc: {accs}", flush=True)
        return rec["val_loss"]

    step, t0, run_loss, run_n = 0, time.time(), 0.0, 0
    best = float("inf")
    for epoch in range(args.epochs):
        for batch in train_dl:
            batch = [t.to(device) for t in batch]
            inputs, tgt = unpack(batch, binding, out_vocab, train_ds.keys)
            loops = (int(torch.randint(1, args.rand_loops + 1, (1,)).item())
                     if args.rand_loops else None)
            loss, k, _, _ = batch_loss(model, inputs, tgt, args.arm,
                                       pad_weight=args.pad_weight, loops=loops,
                                       aux=aux)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            run_loss += loss.item() * k
            run_n += k
            step += 1

            if step % 20 == 0:
                el = time.time() - t0
                print(f"  step {step:5d}/{steps} loss {run_loss/run_n:6.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} {el/step:5.2f}s/step",
                      flush=True)
                run_loss, run_n = 0.0, 0

            if step % args.eval_every == 0 or step == steps:
                vl = eval_and_log(step, epoch, t0)
                if vl < best:
                    best = vl
                    torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                                "step": step, "val_loss": vl}, run / "best.pt")

    torch.save({"cfg": cfg.__dict__, "model": model.state_dict(),
                "step": step, "val_loss": best}, run / "last.pt")
    log.close()
    print(f"done. best val_loss {best:.4f} -> {run}")


if __name__ == "__main__":
    main()
