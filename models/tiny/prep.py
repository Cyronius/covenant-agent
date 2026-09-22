"""Tokenize once, save tensors. The pod loads these and never touches the corpus.

Keeping preparation separate means the training machine needs neither the
covenant-agent checkout nor the corpus, only this cache plus the vocabularies.

Two bindings, one flag (`--binding`), written to the cache's config.json so
train.py and evaluate.py build the matching model:

  flat        the R3 model exactly: one token stream over the whole serialized
              context, an output vocabulary with a row per symbol.
  structural  decisions 6 and 7 of `.claude/plans/npu-native-planner.md`: the
              context is split into lines (one token tensor per tool, field
              and constant line, plus the request), a schema graph says which
              lines reference which, and the target is a canvas of keywords
              plus pointers into this task's symbols (`canvas.py`).
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import re
from pathlib import Path

import torch

from corpus import COVENANT, Example, load, program_tokens, split
from tok import OutVocab, train_input_tokenizer

CACHE = Path(__file__).parent / "data_cache"

# Column order of the structural tensors, as train.py's TensorDataset sees them.
STRUCT_KEYS = ("tool_tok", "field_tok", "const_tok", "reg_tok", "req_tok",
               "n_tool", "n_field", "n_const", "n_reg_bound", "adj", "tgt")


def compact_line(line: str, desc_chars: int = 60) -> str:
    """Trim a schema line's description to its first sentence, then to a cap."""
    if " :: " in line:
        head, desc = line.split(" :: ", 1)
        desc = desc.split(". ")[0]
        if len(desc) > desc_chars:
            desc = desc[:desc_chars].rsplit(" ", 1)[0]
        line = f"{head} :: {desc}"
    return line


def compact(src: str, desc_chars: int = 60) -> str:
    """Trim each schema line's description to its first sentence, then to a cap.

    The signature part of a tool line carries the types and symbols the program
    must get right; the description is what generalization to an unseen tool
    rests on. Trimming the tail of long descriptions buys sequence length
    without touching either.
    """
    return "\n".join(compact_line(l, desc_chars) for l in src.splitlines()) + "\n"


SYM_LINE = re.compile(r"^([TFCSNBDI]\d+)\b")
REG_LINE = re.compile(r"^(r\d{1,2})\b")


def symbol_positions(source: str, enc, ov: OutVocab, max_in: int) -> list[int]:
    """For each output-vocabulary id, where that symbol is declared in the input.

    Returns one input-token index per output vocabulary entry, or -1 when the
    symbol is not a per-request symbol or does not appear in this task.

    This exists because symbol identity is assigned per request. `T2` is a
    different tool in every task, so there is nothing stable for an embedding to
    learn -- the first run scored 8.6% compile with 100% parse for exactly this
    reason. What the task actually asks is "read the descriptions and name the
    matching one", which is a pointer, and a pointer needs to know where each
    candidate lives in the input.
    """
    pos = [-1] * len(ov)
    # Character offset of the start of each declaration line.
    off = 0
    starts: dict[str, int] = {}
    for line in source.splitlines(keepends=True):
        m = SYM_LINE.match(line)
        if m and m.group(1) in ov.stoi:
            starts.setdefault(m.group(1), off)
        off += len(line)
    if not starts:
        return pos
    # Map character offsets to token indices via the tokenizer's offsets.
    by_char: dict[int, int] = {}
    for ti, (a, _b) in enumerate(enc.offsets[:max_in]):
        by_char.setdefault(a, ti)
    for sym, cstart in starts.items():
        # The symbol may not start exactly at a token boundary; take the first
        # token whose span begins at or after the line start.
        ti = by_char.get(cstart)
        if ti is None:
            cands = [t for c, t in by_char.items() if c >= cstart]
            ti = min(cands) if cands else None
        if ti is not None and ti < max_in:
            pos[ov.stoi[sym]] = ti
    return pos


def encode_split(examples: list[Example], tk, ov: OutVocab, max_in: int, canvas: int):
    srcs = [compact(e.source) for e in examples]
    encs = tk.encode_batch(srcs)
    src = torch.tensor([e.ids[:max_in] + [tk.token_to_id("<pad>")] * max(0, max_in - len(e.ids))
                        for e in encs], dtype=torch.long)
    pad = src == tk.token_to_id("<pad>")
    tgt = torch.tensor([ov.encode(program_tokens(e.target), canvas) for e in examples],
                       dtype=torch.long)
    sym = torch.tensor([symbol_positions(s, e, ov, max_in) for s, e in zip(srcs, encs)],
                       dtype=torch.long)
    meta = [{"task_id": e.task_id, "level": e.level, "world": e.world} for e in examples]
    return {"src": src, "pad": pad, "tgt": tgt, "sym": sym, "meta": meta}


# -- structural binding ----------------------------------------------------

_FIELD_REF = re.compile(r"=(F\d+)\b")


class Lines:
    """One task's serialized context, split into the lines region A is made of."""

    def __init__(self, source: str):
        self.request = None
        self.tools: list[tuple[str, str]] = []      # (sym, line text)
        self.fields: list[tuple[str, str]] = []
        self.consts: list[tuple[str, str]] = []
        # plan step 4: the registers a continuation starts from — what is
        # bound, its type, and a bounded rendering of what it holds
        # (`harness/context.py`'s `render_register`). A region of its own
        # rather than more request tokens, so the request budget is untouched.
        self.regs: list[tuple[str, str]] = []
        section = None
        for line in source.splitlines():
            if line.startswith("REQUEST: "):
                self.request = line[len("REQUEST: "):]
                continue
            if line in ("TOOLS:", "FIELDS:", "CONSTANTS:", "REGISTERS:"):
                section = line[:-1]
                continue
            if section == "REGISTERS":
                m = REG_LINE.match(line)
                if not m:
                    raise SystemExit(f"unrecognized register line: {line!r}")
                self.regs.append((m.group(1), line))
                continue
            m = SYM_LINE.match(line)
            if not m:
                raise SystemExit(f"unrecognized context line: {line!r}")
            sym = m.group(1)
            dest = {"TOOLS": self.tools, "FIELDS": self.fields, "CONSTANTS": self.consts}[section]
            dest.append((sym, compact_line(line)))
        if self.request is None:
            raise SystemExit("context has no REQUEST line")

    def edges(self) -> list[tuple[int, int]]:
        """Undirected edges over [tools..., fields...]: a tool to every field
        its signature names, and fields of one entity to each other."""
        n_t = len(self.tools)
        fid = {s: i for i, (s, _) in enumerate(self.fields)}
        out = []
        for i, (_, text) in enumerate(self.tools):
            for f in _FIELD_REF.findall(text.split(" :: ", 1)[0]):
                if f not in fid:
                    raise SystemExit(f"tool line names undeclared field {f}: {text!r}")
                out.append((i, n_t + fid[f]))
        by_entity: dict[str, list[int]] = {}
        for j, (_, text) in enumerate(self.fields):
            ent = text.split(" ", 2)[1]
            if ent != "-":
                by_entity.setdefault(ent, []).append(n_t + j)
        for members in by_entity.values():
            for a in members:
                for b in members:
                    if a != b:
                        out.append((a, b))
        return out


def _pad_ids(ids: list[int], length: int, pad: int) -> list[int]:
    return ids + [pad] * (length - len(ids))


def encode_structural(examples: list[Example], tk, kws: list[str], layout, dims: dict,
                      canvas: int):
    """Per-line token tensors, graph edges, and canvas targets with pointer ids."""
    from canvas import TaskCodec, context_symbols

    pad_id = tk.token_to_id("<pad>")
    TL, RL = dims["max_line"], dims["max_req"]
    MT, MF, MC = layout.max_tool, layout.max_field, layout.max_const
    N = len(examples)

    MR = dims["max_reg"]
    tool_tok = torch.full((N, MT, TL), pad_id, dtype=torch.int16)
    field_tok = torch.full((N, MF, TL), pad_id, dtype=torch.int16)
    const_tok = torch.full((N, MC, TL), pad_id, dtype=torch.int16)
    reg_tok = torch.full((N, MR, TL), pad_id, dtype=torch.int16)
    req_tok = torch.full((N, RL), pad_id, dtype=torch.int16)
    n_tool = torch.zeros(N, dtype=torch.int16)
    n_field = torch.zeros(N, dtype=torch.int16)
    n_const = torch.zeros(N, dtype=torch.int16)
    n_reg_bound = torch.zeros(N, dtype=torch.int16)
    adj = torch.zeros((N, MT + MF, MT + MF), dtype=torch.bool)
    tgt = torch.zeros((N, canvas), dtype=torch.int16)
    meta = []
    max_len = 0
    max_line_seen = max_req_seen = 0

    # Tokenize every line of every task in one batch call.
    all_lines: list[str] = []
    per_task: list[Lines] = []
    for e in examples:
        ln = Lines(e.source)
        per_task.append(ln)
        all_lines.append(ln.request)
        all_lines.extend(t for _, t in ln.tools)
        all_lines.extend(t for _, t in ln.fields)
        all_lines.extend(t for _, t in ln.consts)
        all_lines.extend(t for _, t in ln.regs)
    encs = tk.encode_batch(all_lines)
    cursor = 0

    for n, (e, ln) in enumerate(zip(examples, per_task)):
        syms = context_symbols(e.row["context"])
        # The serializer prints symbols in exactly this order; the line lists
        # must agree with it or the pointer indices are wrong.
        if [s for s, _ in ln.tools] != syms["tools"] or [s for s, _ in ln.fields] != syms["fields"] \
                or [s for s, _ in ln.consts] != syms["consts"]:
            raise SystemExit(f"{e.task_id}: serialized symbol order disagrees with context")
        codec = TaskCodec(kws, layout, **syms)

        req = encs[cursor].ids
        cursor += 1
        groups = []
        for count in (len(ln.tools), len(ln.fields), len(ln.consts),
                      len(ln.regs)):
            groups.append([enc.ids for enc in encs[cursor:cursor + count]])
            cursor += count
        if len(ln.regs) > MR:
            raise SystemExit(f"{e.task_id}: {len(ln.regs)} bound registers, "
                             f"cap is {MR}; raise --max-reg")
        max_req_seen = max(max_req_seen, len(req))
        for g in groups:
            for ids in g:
                max_line_seen = max(max_line_seen, len(ids))
        if len(req) > RL:
            raise SystemExit(f"{e.task_id}: request is {len(req)} tokens, cap is {RL}; "
                             "raise --max-req (prep refuses to truncate)")
        for g in groups:
            for ids in g:
                if len(ids) > TL:
                    raise SystemExit(f"{e.task_id}: a schema line is {len(ids)} tokens, cap is "
                                     f"{TL}; raise --max-line (prep refuses to truncate)")

        req_tok[n, :len(req)] = torch.tensor(req, dtype=torch.int16)
        for dest, g in zip((tool_tok, field_tok, const_tok, reg_tok), groups):
            for i, ids in enumerate(g):
                dest[n, i, :len(ids)] = torch.tensor(ids, dtype=torch.int16)
        n_tool[n], n_field[n], n_const[n] = len(ln.tools), len(ln.fields), len(ln.consts)
        n_reg_bound[n] = len(ln.regs)

        L = len(ln.tools) + len(ln.fields)
        a = adj[n]
        a[torch.arange(MT + MF), torch.arange(MT + MF)] = True     # self, pad rows included
        for i, j in ln.edges():
            a[i, j] = True
            a[j, i] = True
        assert not a[:L, L:].any(), "a live line must not attend to a padded one"

        ids = codec.encode(e.target)          # verifies render(encode) == text
        max_len = max(max_len, len(ids))
        if len(ids) > canvas:
            # Excluded, not truncated, like MAX_PROGRAM_TOKENS. Reported by the caller.
            meta.append(None)
            continue
        tgt[n, :len(ids)] = torch.tensor(ids, dtype=torch.int16)
        meta.append({"task_id": e.task_id, "level": e.level, "world": e.world, "syms": syms})

    keep = [i for i, m in enumerate(meta) if m is not None]
    excluded = N - len(keep)
    if excluded:
        idx = torch.tensor(keep)
        tool_tok, field_tok, const_tok, reg_tok, req_tok = (
            t[idx] for t in (tool_tok, field_tok, const_tok, reg_tok, req_tok))
        n_tool, n_field, n_const, n_reg_bound, adj, tgt = (
            t[idx] for t in (n_tool, n_field, n_const, n_reg_bound, adj, tgt))
        meta = [m for m in meta if m is not None]
    d = {"tool_tok": tool_tok, "field_tok": field_tok, "const_tok": const_tok,
         "reg_tok": reg_tok, "req_tok": req_tok,
         "n_tool": n_tool, "n_field": n_field, "n_const": n_const,
         "n_reg_bound": n_reg_bound, "adj": adj, "tgt": tgt,
         "meta": meta,
         "stats": {"max_slots": max_len, "excluded": excluded, "kept": [examples[i] for i in keep],
                   "max_line_seen": max_line_seen, "max_req_seen": max_req_seen}}
    return d


def encode_one(source: str, syms: dict, tk, layout, dims: dict,
               device=None) -> dict:
    """Model inputs for one serialized context, encoded exactly as the cache
    encodes it (plan step 4).

    The cache is built once from a corpus; a second turn is not in it, because
    the registers a continuation starts from depend on what the model itself
    just ran. So the loop runner (`play.py`) re-serializes the context after
    every `PAUSE` and encodes it here — same tokenizer, same caps, same line
    order — rather than reaching into a pickled split.
    """
    pad_id = tk.token_to_id("<pad>")
    TL, RL, MR = dims["max_line"], dims["max_req"], dims.get("max_reg", 8)
    MT, MF, MC = layout.max_tool, layout.max_field, layout.max_const
    ln = Lines(source)
    if [s for s, _ in ln.tools] != syms["tools"] \
            or [s for s, _ in ln.fields] != syms["fields"] \
            or [s for s, _ in ln.consts] != syms["consts"]:
        raise SystemExit("serialized symbol order disagrees with the context")

    texts = [ln.request] + [t for _, t in ln.tools] + [t for _, t in ln.fields] \
        + [t for _, t in ln.consts] + [t for _, t in ln.regs]
    encs = tk.encode_batch(texts)
    req = encs[0].ids[:RL]
    cursor = 1
    groups = []
    for count in (len(ln.tools), len(ln.fields), len(ln.consts), len(ln.regs)):
        groups.append([e.ids[:TL] for e in encs[cursor:cursor + count]])
        cursor += count

    out = {}
    req_tok = torch.full((1, RL), pad_id, dtype=torch.int16)
    req_tok[0, :len(req)] = torch.tensor(req, dtype=torch.int16)
    out["req_tok"] = req_tok
    for key, cap, g in zip(("tool_tok", "field_tok", "const_tok", "reg_tok"),
                           (MT, MF, MC, MR), groups):
        t = torch.full((1, cap, TL), pad_id, dtype=torch.int16)
        for i, ids in enumerate(g[:cap]):
            t[0, i, :len(ids)] = torch.tensor(ids, dtype=torch.int16)
        out[key] = t
    for key, n in (("n_tool", len(ln.tools)), ("n_field", len(ln.fields)),
                   ("n_const", len(ln.consts)),
                   ("n_reg_bound", min(len(ln.regs), MR))):
        out[key] = torch.tensor([n], dtype=torch.int16)

    adj = torch.zeros((1, MT + MF, MT + MF), dtype=torch.bool)
    adj[0, torch.arange(MT + MF), torch.arange(MT + MF)] = True
    for i, j in ln.edges():
        adj[0, i, j] = True
        adj[0, j, i] = True
    out["adj"] = adj
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


def pick_holdout_worlds(counts: dict[str, int], n: int,
                        seed: int | None) -> list[str]:
    """Which worlds to hold out, drawn from the mid-sized band.

    The largest worlds are too much of the training set to give away and the
    smallest estimate nothing, so the draw is over the middle half by row
    count. `seed=None` and n=1 reproduce the historical pick -- the world at
    the median rank -- so a cache prepared before this flag existed still
    rebuilds byte for byte.
    """
    ranked = [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])]
    if not ranked:
        raise SystemExit("no worlds in the corpus")
    if seed is None and n == 1:
        return [ranked[len(ranked) // 2]]
    lo, hi = len(ranked) // 4, (3 * len(ranked)) // 4
    band = ranked[lo:hi] or ranked
    if n > len(band):
        raise SystemExit(
            f"--holdout-worlds {n} exceeds the {len(band)}-world mid band "
            f"of {len(ranked)} worlds; name them with --holdout-world")
    return sorted(random.Random(seed).sample(band, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=str(COVENANT / "data" / "s5_plain.jsonl"))
    ap.add_argument("--limit", type=int, default=20000)
    ap.add_argument("--binding", choices=["structural", "flat"], default="structural")
    ap.add_argument("--in-vocab", type=int, default=4096)
    ap.add_argument("--max-in", type=int, default=1280, help="flat: input token cap")
    ap.add_argument("--max-line", type=int, default=64, help="structural: tokens per schema line")
    ap.add_argument("--max-req", type=int, default=128, help="structural: request tokens")
    ap.add_argument("--max-reg", type=int, default=8,
                    help="structural: bound registers a continuation may "
                         "start from, one line each (plan step 4). A "
                         "segment after a PAUSE has as many as the paused "
                         "program left bound; 8 covers every reference in "
                         "the tree.")
    ap.add_argument("--canvas", type=int, default=64)
    ap.add_argument("--holdout-corpus", default=None, metavar="PATH",
                    help="take the holdout split from a SECOND corpus, "
                         "generated over the reserved eval worlds "
                         "(`data.gen --holdout`). Preferred over carving "
                         "worlds out of --corpus: it costs no training data "
                         "and gives 42 unseen worlds instead of one")
    ap.add_argument("--holdout-limit", type=int, default=None, metavar="N",
                    help="rows to load from --holdout-corpus (default: "
                         "--limit)")
    ap.add_argument("--holdout-world", default=None, metavar="NAME[,NAME...]",
                    help="world name(s) to carve out of --corpus; default "
                         "draws --holdout-worlds from the mid-sized band. "
                         "Ignored when --holdout-corpus is given")
    ap.add_argument("--holdout-worlds", type=int, default=1, metavar="N",
                    help="how many worlds to hold out when --holdout-world "
                         "does not name them (default 1)")
    ap.add_argument("--holdout-seed", type=int, default=None, metavar="S",
                    help="seed the world draw. Without it, N=1 keeps the "
                         "historical median-sized pick, so existing caches "
                         "reproduce; with it the drawn worlds vary, which is "
                         "what lets a seed sweep tell a seed-unstable encoder "
                         "from an unlucky world (R8 §5b)")
    ap.add_argument("--out", default=None, help="default data_cache (flat) or data_cache_struct")
    args = ap.parse_args()

    out = Path(args.out or (CACHE if args.binding == "flat" else CACHE.with_name("data_cache_struct")))
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.limit} from {Path(args.corpus).name} ...")
    # a paused task's registers come from running its reference, which
    # needs the generated theme worlds registered or every replay
    # raises and the continuation rows vanish silently
    from sandbox import register_themes
    register_themes()
    ex = load(Path(args.corpus), limit=args.limit)
    from corpus import DROPPED_REPLAY
    if DROPPED_REPLAY:
        print("  paused tasks dropped (no honest register state): "
              + ", ".join(f"{n}x {why}"
                          for why, n in DROPPED_REPLAY.most_common()))
    cont = sum(1 for e in ex if "#s" in e.task_id)
    print(f"  {len(ex)} examples, {len(set(e.world for e in ex))} worlds"
          + (f"; {cont} of them segments of a paused task, "
             f"{sum(1 for e in ex if e.task_id.endswith('#s0'))} first and the "
             f"rest continuations that start from bound registers"
             if cont else ""))

    counts = {}
    for e in ex:
        counts[e.world] = counts.get(e.world, 0) + 1

    config = {"corpus": Path(args.corpus).name, "limit": args.limit,
              "binding": args.binding, "canvas": args.canvas}

    if args.holdout_corpus:
        # The reserved eval worlds are already unseen by construction
        # (data/holdout/reserved_domains.json, reserved since S2, drawn by
        # `data.gen --holdout`), they carry the same ID:entity typing and
        # program structure as the training worlds, and there are 42 of
        # them. Measuring on those costs no training data. Carving worlds
        # out of --corpus costs 1% per world and gave R7 and R8 an n=1
        # estimate of a claim that is about variance.
        ho = load(Path(args.holdout_corpus),
                  limit=args.holdout_limit or args.limit)
        ho_worlds = sorted({e.world for e in ho})
        overlap = sorted({e.world for e in ex} & set(ho_worlds))
        if overlap:
            raise SystemExit(
                "--holdout-corpus shares worlds with --corpus, so the "
                f"holdout split is not unseen: {overlap[:5]}")
        print(f"  holdout corpus: {len(ho)} examples over "
              f"{len(ho_worlds)} unseen worlds, 0% of the training data")
        tr, va, te, _ = split(ex, seed=0)
        config.update({"holdout_corpus": Path(args.holdout_corpus).name,
                       "holdout_worlds": ho_worlds})
    else:
        if args.holdout_world:
            holdout_worlds = [w.strip() for w in args.holdout_world.split(",") if w.strip()]
            missing = [w for w in holdout_worlds if w not in counts]
            if missing:
                raise SystemExit(f"--holdout-world names worlds not in the corpus: {missing}")
        else:
            holdout_worlds = pick_holdout_worlds(counts, args.holdout_worlds,
                                                 args.holdout_seed)
        share = sum(counts[w] for w in holdout_worlds) / max(len(ex), 1)
        print(f"  carving out {len(holdout_worlds)} world(s): "
              f"{', '.join(holdout_worlds)}  ({share:.1%} of the training data)")
        tr, va, te, ho = split(ex, seed=0, holdout_worlds=set(holdout_worlds))
        config.update({"holdout_worlds": holdout_worlds,
                       "holdout_seed": args.holdout_seed})

    print(f"  train {len(tr)}  val {len(va)}  test {len(te)}  holdout {len(ho)}")
    # the layout, the keyword table and the flat symbol vocabulary have to
    # cover every split, or a holdout row overflows a tensor sized on train
    ex = ex + ho if args.holdout_corpus else ex

    if args.binding == "flat":
        ov = OutVocab.build(ex)
        ov.save(out / "out_vocab.json")
        print(f"  output vocab {len(ov)}")
        print("training input tokenizer ...")
        tk = train_input_tokenizer([compact(e.source) for e in tr[:8000]],
                                   args.in_vocab, max_length=args.max_in)
        tk.save(str(out / "in_tok.json"))
        config.update({"in_vocab": tk.get_vocab_size(), "out_vocab": len(ov), "max_in": args.max_in})
        parts = {}
        for name, part in (("train", tr), ("val", va), ("test", te), ("holdout", ho)):
            if not part:
                continue
            d = encode_split(part, tk, ov, args.max_in, args.canvas)
            torch.save({k: v for k, v in d.items() if k != "meta"}, out / f"{name}.pt")
            (out / f"{name}_meta.json").write_text(json.dumps(d["meta"]), encoding="utf-8")
            lens = (d["src"] != tk.token_to_id("<pad>")).sum(1)
            n_cut = int((lens >= args.max_in).sum())
            print(f"  {name:8s} {len(part):6d}  longest input {int(lens.max())}/{args.max_in}"
                  + (f"  TRUNCATED {n_cut}" if n_cut else ""))
            # A truncated input loses its CONSTANTS section, which the program
            # references by symbol. That is silent corruption, not a small loss of
            # context, so it fails here rather than showing up as a model error.
            if n_cut:
                raise SystemExit(
                    f"{n_cut} {name} inputs hit the {args.max_in}-token cap. "
                    f"Raise --max-in or drop the worlds with the longest schemas.")
            overflow = int((d["tgt"][:, -1] != ov.pad).sum())
            if overflow:
                raise SystemExit(f"{overflow} {name} programs fill the whole "
                                 f"{args.canvas}-slot canvas; raise --canvas.")
            parts[name] = part
    else:
        from canvas import Layout, build_keywords, save_keywords
        kws = build_keywords([e.target for e in ex])
        save_keywords(kws, out / "keywords.json")
        from canvas import base_keywords
        extra = kws[len(base_keywords()):]
        print(f"  {len(kws)} keyword rows" + (f"; corpus extras beyond the fixed table: {extra}"
                                             if extra else ""))
        max_tool = max(len(e.row["context"]["tools"]) for e in ex)
        max_field = max(len(e.row["context"]["fields"]) for e in ex)
        max_const = max(len(e.row["context"]["constants"]) for e in ex)
        layout = Layout(len(kws), max_tool, max_field, max_const)
        print(f"  layout: {layout.size} joint ids = {len(kws)} keywords + {max_tool} tools "
              f"+ {max_field} fields + {max_const} consts + {layout.n_reg} registers")

        print("training input tokenizer on schema lines and requests ...")
        texts = []
        for e in tr[:8000]:
            ln = Lines(e.source)
            texts.append(ln.request)
            texts.extend(t for _, t in ln.tools + ln.fields + ln.consts
                         + ln.regs)
        tk = train_input_tokenizer(texts, args.in_vocab, max_length=max(args.max_line, args.max_req))
        tk.no_truncation()       # lengths are checked below; nothing is cut silently
        tk.no_padding()
        tk.save(str(out / "in_tok.json"))
        dims = {"max_line": args.max_line, "max_req": args.max_req,
                "max_reg": args.max_reg}
        config.update({"in_vocab": tk.get_vocab_size(), "in_pad": tk.token_to_id("<pad>"),
                       "layout": layout.to_dict(), **dims})

        parts = {}
        max_slots = 0
        for name, part in (("train", tr), ("val", va), ("test", te), ("holdout", ho)):
            if not part:
                continue
            d = encode_structural(part, tk, kws, layout, dims, args.canvas)
            st = d.pop("stats")
            max_slots = max(max_slots, st["max_slots"])
            torch.save({k: v for k, v in d.items() if k != "meta"}, out / f"{name}.pt")
            (out / f"{name}_meta.json").write_text(json.dumps(d["meta"]), encoding="utf-8")
            print(f"  {name:8s} {len(st['kept']):6d}  longest line {st['max_line_seen']}/{dims['max_line']}"
                  f"  longest request {st['max_req_seen']}/{dims['max_req']}"
                  f"  longest canvas {st['max_slots']}/{args.canvas}"
                  + (f"  EXCLUDED {st['excluded']} (overflow the canvas after the compound split)"
                     if st["excluded"] else ""))
            parts[name] = st["kept"]
        config["max_slots"] = max_slots
        print(f"  longest program after splitting rN.Fk: {max_slots} of {args.canvas} slots")

    # The raw rows are what the scorer replays in the sandbox, kept separately
    # so the pod never needs them.
    with open(out / "rows.pkl", "wb") as fh:
        pickle.dump({n: [e.row for e in p]
                     for n, p in parts.items() if n != "train"}, fh)

    (out / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
