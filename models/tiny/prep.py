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
from collections import Counter
from pathlib import Path

import torch

from corpus import COVENANT, Example, load, program_tokens, split
from tok import OutVocab, train_input_tokenizer

CACHE = Path(__file__).parent / "data_cache"

# Column order of the structural tensors, as train.py's TensorDataset sees them.
STRUCT_KEYS = ("tool_tok", "field_tok", "const_tok", "reg_tok", "req_tok",
               "n_tool", "n_field", "n_const", "n_reg_bound", "adj", "tgt")
# The split encoder's extra columns (.claude/plans/description-reading.md
# step 2), present only in a cache built with --split: each tool line's
# signature, description and name as separate token rows, and indices into
# the teacher's embedding table (teacher.pt) for the two extra losses.
SPLIT_KEYS = ("tool_sig_tok", "tool_desc_tok", "tool_name_tok", "sig_group", "flip_tool",
              "t_desc", "t_name", "t_req")
# C0's columns (.claude/plans/tiny-general-agent-menu.md), present only in a
# cache built with --reader-lines: indices into reader.pt for each constant,
# each field and each chunk of the request.
READER_KEYS = ("t_const", "t_field", "t_chunk")


def cache_keys(d: dict) -> tuple:
    """The column order a cache split loads in: the structural columns, then
    whichever split and reader columns it carries."""
    return STRUCT_KEYS + tuple(k for k in SPLIT_KEYS + READER_KEYS if k in d)


def compact_desc(desc: str, desc_chars: int = 60) -> str:
    """A description's first sentence, then a cap at a word boundary."""
    desc = desc.split(". ")[0]
    if len(desc) > desc_chars:
        desc = desc[:desc_chars].rsplit(" ", 1)[0]
    return desc


def compact_line(line: str, desc_chars: int = 60) -> str:
    """Trim a schema line's description to its first sentence, then to a cap.

    60 characters is R3-R10's cap. It cuts a verbose theme's line to its
    boilerplate ("This tool returns every turnaround task currently") and
    the part that tells twins apart goes with it, so a cache built for the
    description-reading arms raises it (`--desc-chars`), for every arm."""
    if " :: " in line:
        head, desc = line.split(" :: ", 1)
        line = f"{head} :: {compact_desc(desc, desc_chars)}"
    return line


def split_tool_line(line: str, name_words: bool = False) -> tuple[str, str, str]:
    """(signature, name, description) of one compacted tool line. The
    signature keeps the symbol and drops the name, so it is the pre-0.8.0
    head exactly; the name is empty when the serializer omitted it.
    `name_words` hands the name stage `flag hoist plan review` rather than
    `flagHoistPlanReview`, so a small tokenizer spends its pieces on words,
    not on one identifier's capitals (vocab-push plan, part A)."""
    head, _, desc = line.partition(" :: ")
    sym, _, rest = head.partition(" ")
    name = ""
    if rest and not rest.startswith("("):
        name, _, rest = rest.partition(" ")
    if name_words and name:
        name = teacher_name(name)
    return f"{sym} {rest}", name, desc


def reader_text(text: str) -> str:
    """A field's or constant's value as the reader gets it (C0): the part
    after ` :: `, identifiers as words (`vendor_booking.stage "confirmed"` ->
    `vendor booking stage confirmed`). A dot between digits stays (`v4.2`)."""
    text = text.split(" :: ", 1)[-1]
    text = re.sub(r"(?<=[A-Za-z_])\.(?=[A-Za-z_])", " ", text.replace('"', ""))
    return " ".join(text.replace("_", " ").split())


def chunk_request(request: str, size: int = 4) -> list[str]:
    """The request as overlapping windows of `size` words, moving size//2
    words at a time, the last window ending on the last word (C0). A request
    of `size` words or fewer is one chunk."""
    w = request.split()
    if len(w) <= size:
        return [" ".join(w)]
    step = max(1, size // 2)
    starts = list(range(0, len(w) - size + 1, step))
    if starts[-1] + size < len(w):
        starts.append(len(w) - size)
    return [" ".join(w[s:s + size]) for s in starts]


def pad_token_id(tk) -> int:
    """Our tokenizers pad with `<pad>`; a BERT-family one (`--in-tok
    teacher:...`) with `[PAD]`."""
    for t in ("<pad>", "[PAD]"):
        i = tk.token_to_id(t)
        if i is not None:
            return i
    raise SystemExit("input tokenizer has no <pad> or [PAD] token")


def load_input_tokenizer(spec: str):
    """A ready-made input tokenizer: a tokenizer.json path, or
    `teacher:MODEL` for that model's own vocabulary. Special-token wrapping,
    truncation and padding are stripped: prep places every id itself and
    refuses to truncate."""
    from tokenizers import Tokenizer
    if spec.startswith("teacher:"):
        from huggingface_hub import hf_hub_download
        spec = hf_hub_download(spec.split(":", 1)[1], "tokenizer.json")
    raw = json.loads(Path(spec).read_text(encoding="utf-8"))
    raw.update({"post_processor": None, "truncation": None, "padding": None})
    return Tokenizer.from_str(json.dumps(raw))


def extra_tokenizer_texts(paths: list[str], name_words: bool) -> list[str]:
    """General-English text to learn the tokenizer's pieces from, beside the
    training themes (data/general, vocab-push plan part A). Rows are
    `{"text": ...}`; a names file goes through the same word split as tool
    names."""
    texts = []
    for p in paths:
        names = Path(p).stem.startswith("names")
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                t = json.loads(line)["text"]
                texts.append(teacher_name(t) if names and name_words else t)
    return texts


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

    def __init__(self, source: str, desc_chars: int = 60):
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
            dest.append((sym, compact_line(line, desc_chars)))
        if self.request is None:
            raise SystemExit("context has no REQUEST line")

    def edges(self) -> list[tuple[int, int]]:
        """Undirected edges over [tools..., fields...]: a tool to every field
        its signature names, and fields of one entity to each other. Indices
        are compact (field j is `len(tools) + j`); `layout_edges` places them
        in the padded tensor."""
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


def layout_edges(ln: "Lines", max_tool: int) -> list[tuple[int, int]]:
    """`ln.edges()` in the adjacency tensor's own layout, where the tool rows
    fill `max_tool` slots and field j sits at `max_tool + j`.

    Before 2026-09-23 the compact indices went into the tensor as they were,
    which is right only when a task has exactly `max_tool` tools. It did when
    every task had 18 (R3-R8). In R9's cache (14-18 tools, max_tool 50) and
    R10's (22 tools, max_tool 54) it held on no training row: a tool's edges to its fields
    landed on *padded tool rows*, and a field's edges to its entity on tool
    rows too: the schema graph pass of R9 and R10 linked no tool to any
    field. Found by the packed-line equivalence check, which is the first
    thing to notice a live row reading a padded one."""
    n_t = len(ln.tools)
    at = lambda k: k if k < n_t else max_tool + (k - n_t)   # noqa: E731
    return [(at(i), at(j)) for i, j in ln.edges()]


def _pad_ids(ids: list[int], length: int, pad: int) -> list[int]:
    return ids + [pad] * (length - len(ids))


def encode_structural(examples: list[Example], tk, kws: list[str], layout, dims: dict,
                      canvas: int):
    """Per-line token tensors, graph edges, and canvas targets with pointer ids."""
    from canvas import TaskCodec, context_symbols

    pad_id = pad_token_id(tk)
    TL, RL = dims["max_line"], dims["max_req"]
    MT, MF, MC = layout.max_tool, layout.max_field, layout.max_const
    N = len(examples)

    MR = dims["max_reg"]
    split = bool(dims.get("split"))
    name_words = bool(dims.get("name_words"))
    SL, DL, NL = dims.get("max_sig", TL), dims.get("max_desc", TL), dims.get("max_name", 16)
    if split:
        sig_tok = torch.full((N, MT, SL), pad_id, dtype=torch.int16)
        desc_tok = torch.full((N, MT, DL), pad_id, dtype=torch.int16)
        name_tok = torch.full((N, MT, NL), pad_id, dtype=torch.int16)
        # which tools of a row share a signature (the symbol aside): twins
        # carry the same id, -1 on padding. What the twin benchmark and the
        # contrastive loss's "twins first" read.
        sig_group = torch.full((N, MT), -1, dtype=torch.int16)
        # tools of a flip slot (provenance.flip_slot_tools): the twin
        # decisions whose answer is drawn uniformly (data.gen --twin-roles),
        # the population a grounding number is quoted on
        flip_tool = torch.zeros((N, MT), dtype=torch.bool)
    teacher_texts = []                    # per kept row: (request, descs, names)
    chunk_words = int(dims.get("chunk_words") or 0)
    reader_texts = []                     # C0, per kept row: (consts, fields, chunks)
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
    desc_chars = dims.get("desc_chars", 60)
    for e in examples:
        ln = Lines(e.source, desc_chars)
        per_task.append(ln)
        all_lines.append(ln.request)
        all_lines.extend(t for _, t in ln.tools)
        all_lines.extend(t for _, t in ln.fields)
        all_lines.extend(t for _, t in ln.consts)
        all_lines.extend(t for _, t in ln.regs)
        if split:
            for _, t in ln.tools:
                all_lines.extend(split_tool_line(t, name_words))
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
        parts = []
        if split:
            for _, t in ln.tools:
                parts.append(tuple(enc.ids for enc in encs[cursor:cursor + 3]))
                cursor += 3
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

        for sig_ids, name_ids, desc_ids in parts:
            for ids, cap, what in ((sig_ids, SL, "signature"), (desc_ids, DL, "description"),
                                   (name_ids, NL, "name")):
                if len(ids) > cap:
                    raise SystemExit(f"{e.task_id}: a tool {what} is {len(ids)} tokens, "
                                     f"cap is {cap}; raise --max-{what[:4]} (prep refuses "
                                     "to truncate)")
        for i, (sig_ids, name_ids, desc_ids) in enumerate(parts):
            sig_tok[n, i, :len(sig_ids)] = torch.tensor(sig_ids, dtype=torch.int16)
            desc_tok[n, i, :len(desc_ids)] = torch.tensor(desc_ids, dtype=torch.int16)
            name_tok[n, i, :len(name_ids)] = torch.tensor(name_ids, dtype=torch.int16)
        if split:
            tri = [split_tool_line(t, name_words) for _, t in ln.tools]
            gid: dict[str, int] = {}
            for i, (sg, _, _) in enumerate(tri):
                sig_group[n, i] = gid.setdefault(sg.split(" ", 1)[1], len(gid))
            fl = set((e.row.get("provenance") or {}).get("flip_slot_tools") or [])
            if fl:
                name_of = {t["sym"]: t["name"] for t in e.row["context"]["tools"]}
                for i, (sym, _) in enumerate(ln.tools):
                    flip_tool[n, i] = name_of.get(sym) in fl
            teacher_texts.append((ln.request, [d for _, _, d in tri],
                                  [nm for _, nm, _ in tri]))
        else:
            teacher_texts.append(None)
        reader_texts.append(([reader_text(t) for _, t in ln.consts],
                             [reader_text(t) for _, t in ln.fields],
                             chunk_request(ln.request, chunk_words))
                            if chunk_words else None)
        req_tok[n, :len(req)] = torch.tensor(req, dtype=torch.int16)
        for dest, g in zip((tool_tok, field_tok, const_tok, reg_tok), groups):
            for i, ids in enumerate(g):
                dest[n, i, :len(ids)] = torch.tensor(ids, dtype=torch.int16)
        n_tool[n], n_field[n], n_const[n] = len(ln.tools), len(ln.fields), len(ln.consts)
        n_reg_bound[n] = len(ln.regs)

        a = adj[n]
        a[torch.arange(MT + MF), torch.arange(MT + MF)] = True     # self, pad rows included
        for i, j in layout_edges(ln, MT):
            a[i, j] = True
            a[j, i] = True
        live = torch.zeros(MT + MF, dtype=torch.bool)
        live[:len(ln.tools)] = True
        live[MT:MT + len(ln.fields)] = True
        assert not a[live][:, ~live].any(), "a live line must not attend to a padded one"

        ids = codec.encode(e.target)          # verifies render(encode) == text
        max_len = max(max_len, len(ids))
        if len(ids) > canvas:
            # Excluded, not truncated, like MAX_PROGRAM_TOKENS. Reported by the caller.
            meta.append(None)
            teacher_texts[-1] = reader_texts[-1] = None
            continue
        tgt[n, :len(ids)] = torch.tensor(ids, dtype=torch.int16)
        meta.append({"task_id": e.task_id, "level": e.level, "world": e.world, "syms": syms,
                     **({"opaque": "opaque-names" in (e.row.get("tags") or []),
                         "swapped": bool((e.row.get("provenance") or {}).get("roles"))}
                        if split else {})})

    keep = [i for i, m in enumerate(meta) if m is not None]
    excluded = N - len(keep)
    if excluded:
        idx = torch.tensor(keep)
        tool_tok, field_tok, const_tok, reg_tok, req_tok = (
            t[idx] for t in (tool_tok, field_tok, const_tok, reg_tok, req_tok))
        n_tool, n_field, n_const, n_reg_bound, adj, tgt = (
            t[idx] for t in (n_tool, n_field, n_const, n_reg_bound, adj, tgt))
        if split:
            sig_tok, desc_tok, name_tok, sig_group, flip_tool = (
                t[idx] for t in (sig_tok, desc_tok, name_tok, sig_group, flip_tool))
        teacher_texts = [teacher_texts[i] for i in keep]
        reader_texts = [reader_texts[i] for i in keep]
        meta = [m for m in meta if m is not None]
    d = {"tool_tok": tool_tok, "field_tok": field_tok, "const_tok": const_tok,
         "reg_tok": reg_tok, "req_tok": req_tok,
         "n_tool": n_tool, "n_field": n_field, "n_const": n_const,
         "n_reg_bound": n_reg_bound, "adj": adj, "tgt": tgt,
         "meta": meta,
         "teacher_texts": teacher_texts,
         "reader_texts": reader_texts,
         "stats": {"max_slots": max_len, "excluded": excluded, "kept": [examples[i] for i in keep],
                   "max_line_seen": max_line_seen, "max_req_seen": max_req_seen}}
    if split:
        d.update({"tool_sig_tok": sig_tok, "tool_desc_tok": desc_tok,
                  "tool_name_tok": name_tok, "sig_group": sig_group,
                  "flip_tool": flip_tool})
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
    pad_id = pad_token_id(tk)
    TL, RL, MR = dims["max_line"], dims["max_req"], dims.get("max_reg", 8)
    MT, MF, MC = layout.max_tool, layout.max_field, layout.max_const
    ln = Lines(source, dims.get("desc_chars", 60))
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
    for i, j in layout_edges(ln, MT):
        adj[0, i, j] = True
        adj[0, j, i] = True
    out["adj"] = adj
    if dims.get("split"):
        caps = {"tool_sig_tok": dims["max_sig"], "tool_desc_tok": dims["max_desc"],
                "tool_name_tok": dims["max_name"]}
        tri = [split_tool_line(t, bool(dims.get("name_words"))) for _, t in ln.tools][:MT]
        for key, j in (("tool_sig_tok", 0), ("tool_name_tok", 1), ("tool_desc_tok", 2)):
            t = torch.full((1, MT, caps[key]), pad_id, dtype=torch.int16)
            for i, e in enumerate(tk.encode_batch([x[j] for x in tri])):
                ids = e.ids[:caps[key]]
                t[0, i, :len(ids)] = torch.tensor(ids, dtype=torch.int16)
            out[key] = t
    if device is not None:
        out = {k: v.to(device) for k, v in out.items()}
    return out


QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def teacher_name(name: str) -> str:
    """A tool name as words, for the teacher: `listSubmissions` and
    `list_submissions` both read `list submissions`."""
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", name)
    return " ".join(name.replace("_", " ").split()).lower()


def teacher_tables(encoded: dict, model: str, out: Path) -> None:
    """Embed every distinct description, name and request once with the
    teacher (bge's CLS pooling, unit length) and give each split index
    columns into the table: t_desc/t_name (N, max_tool), -1 on padding, and
    t_req (N,). The table ships as teacher.pt, float16. Requests get bge's
    query instruction, descriptions and names do not, the way the encoder
    was trained to be used."""
    from transformers import AutoModel, AutoTokenizer
    texts: dict[str, int] = {}

    def tid(t: str) -> int:
        if t not in texts:
            texts[t] = len(texts)
        return texts[t]

    for d in encoded.values():
        N, MT = d["tool_tok"].shape[:2]
        td = torch.full((N, MT), -1, dtype=torch.int32)
        tn = torch.full((N, MT), -1, dtype=torch.int32)
        tr = torch.full((N,), -1, dtype=torch.int32)
        for n, tt in enumerate(d["teacher_texts"]):
            if tt is None:
                continue
            req, descs, names = tt
            tr[n] = tid(QUERY_PREFIX + req)
            for i, (ds, nm) in enumerate(zip(descs, names)):
                td[n, i] = tid(ds)
                tn[n, i] = tid(teacher_name(nm) if nm else ds)
        d.update({"t_desc": td, "t_name": tn, "t_req": tr})
    order = sorted(texts, key=texts.get)
    print(f"  teacher: embedding {len(order)} distinct texts with {model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model)
    enc = AutoModel.from_pretrained(model).eval()
    rows = []
    with torch.no_grad():
        for i in range(0, len(order), 128):
            b = tok(order[i:i + 128], padding=True, truncation=True, max_length=128,
                    return_tensors="pt")
            rows.append(torch.nn.functional.normalize(
                enc(**b).last_hidden_state[:, 0], dim=-1).half())
    # the texts too, in table order, so another reader can embed exactly the
    # same rows against the same indices (reader_table.py)
    torch.save({"table": torch.cat(rows), "model": model, "texts": order}, out / "teacher.pt")
    return len(order)


def reader_line_tables(encoded: dict, base: int, out: Path) -> int:
    """C0's index columns: t_const (N, max_const), t_field (N, max_field) and
    t_chunk (N, max_chunk), -1 on padding, into the reader table's rows after
    the teacher's `base` texts. Only the reader embeds these (the teacher
    never needs them), so their texts go to reader_texts.jsonl in table order
    for reader_table.py to append. Returns max_chunk."""
    texts: dict[str, int] = {}

    def tid(t: str) -> int:
        if t not in texts:
            texts[t] = len(texts)
        return base + texts[t]

    MK = max(len(rt[2]) for d in encoded.values() for rt in d["reader_texts"] if rt)
    for d in encoded.values():
        N, MC = d["const_tok"].shape[:2]
        MF = d["field_tok"].shape[1]
        tc = torch.full((N, MC), -1, dtype=torch.int32)
        tf = torch.full((N, MF), -1, dtype=torch.int32)
        tk = torch.full((N, MK), -1, dtype=torch.int32)
        for n, rt in enumerate(d["reader_texts"]):
            consts, fields, chunks = rt
            for dest, items in ((tc, consts), (tf, fields), (tk, chunks)):
                for i, t in enumerate(items):
                    dest[n, i] = tid(t)
        d.update({"t_const": tc, "t_field": tf, "t_chunk": tk})
    with open(out / "reader_texts.jsonl", "w", encoding="utf-8", newline="\n") as fh:
        for t in sorted(texts, key=texts.get):
            fh.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"  reader lines: {len(texts)} distinct constant, field and chunk texts "
          f"after the teacher's {base}; up to {MK} chunks per request", flush=True)
    return MK


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
    ap.add_argument("--names", action="store_true",
                    help="render each tool's declared name on its line "
                         "(spec 0.8.0). Off reproduces every earlier cache")
    ap.add_argument("--desc-chars", type=int, default=60,
                    help="description cap after the first sentence; 60 is "
                         "R3-R10's, which cuts verbose themes to boilerplate")
    ap.add_argument("--split", action="store_true",
                    help="also store each tool line's signature, description "
                         "and name as separate token rows, for the split "
                         "encoder (description-reading plan step 2)")
    ap.add_argument("--max-sig", type=int, default=48)
    ap.add_argument("--max-desc", type=int, default=64)
    ap.add_argument("--max-name", type=int, default=16)
    ap.add_argument("--name-words", action="store_true",
                    help="with --split: the name stage reads a tool name as "
                         "words (`flag hoist plan review`), and the tokenizer "
                         "learns pieces for them")
    ap.add_argument("--in-tok", default=None, metavar="PATH|teacher:MODEL",
                    help="use this input tokenizer instead of training one "
                         "(e.g. teacher:unsloth/bge-small-en-v1.5 for the "
                         "teacher's 30,522 pieces)")
    ap.add_argument("--tok-extra", action="append", default=[], metavar="JSONL",
                    help="general-English rows ({\"text\": ...}) to learn the "
                         "tokenizer's pieces from beside the themes; repeatable "
                         "(data/general/texts.jsonl, names.jsonl)")
    ap.add_argument("--teacher", default=None, metavar="MODEL",
                    help="with --split: embed every distinct description, name "
                         "and request with this encoder (unsloth/bge-small-en-"
                         "v1.5) into teacher.pt, for the relational loss")
    ap.add_argument("--reader-lines", action="store_true",
                    help="with --split --teacher: also index every constant, "
                         "every field and the request in chunks for the reader "
                         "(C0, tiny-general-agent-menu.md); reader_table.py "
                         "embeds them from reader_texts.jsonl")
    ap.add_argument("--chunk-words", type=int, default=4,
                    help="with --reader-lines: words per request chunk; "
                         "chunks move half that at a time")
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
    if args.reader_lines and not (args.split and args.teacher):
        raise SystemExit("--reader-lines builds on the teacher's text table: "
                         "it needs --split and --teacher")

    out = Path(args.out or (CACHE if args.binding == "flat" else CACHE.with_name("data_cache_struct")))
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.limit} from {Path(args.corpus).name} ...")
    # a paused task's registers come from running its reference, which
    # needs the generated theme worlds registered or every replay
    # raises and the continuation rows vanish silently
    from sandbox import register_themes
    register_themes()
    ex = load(Path(args.corpus), limit=args.limit, names=args.names)
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
    if args.names or args.desc_chars != 60 or args.split:
        # recorded only when set, so a cache built without them writes the
        # config it always wrote
        config.update({"names": args.names, "desc_chars": args.desc_chars,
                       "split": args.split})

    if args.holdout_corpus:
        # The reserved eval worlds are already unseen by construction
        # (data/holdout/reserved_domains.json, reserved since S2, drawn by
        # `data.gen --holdout`), they carry the same ID:entity typing and
        # program structure as the training worlds, and there are 42 of
        # them. Measuring on those costs no training data. Carving worlds
        # out of --corpus costs 1% per world and gave R7 and R8 an n=1
        # estimate of a claim that is about variance.
        ho = load(Path(args.holdout_corpus),
                  limit=args.holdout_limit or args.limit, names=args.names)
        ho_worlds = sorted({e.world for e in ho})
        # A combined holdout is the plain and decoyed halves of the SAME
        # tasks, so the decoyed half has to be re-identified on the way in
        # (`+decoy`, results/R9.md §7). Assembling it with a plain `cat`
        # leaves every id twice, and nothing downstream notices: the cached
        # row and the row a scorer looks up by that id are then different
        # tasks with different tool counts, which reads as a broken encoder
        # rather than a broken corpus. Cost of finding this the slow way:
        # one cache rebuild.
        dupes = Counter(e.task_id for e in ho)
        repeated = [i for i, n in dupes.items() if n > 1]
        if repeated:
            raise SystemExit(
                f"--holdout-corpus has {len(repeated)} ids more than once, "
                f"e.g. {repeated[:3]}. A combined plain+decoyed holdout must "
                "suffix the decoyed half's ids (`+decoy`) -- see "
                "results/R9.md section 7 for the assembly.")
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

        if args.in_tok:
            print(f"input tokenizer: {args.in_tok}")
            tk = load_input_tokenizer(args.in_tok)
        else:
            print("training input tokenizer on schema lines and requests ...")
            texts = []
            for e in tr[:8000]:
                ln = Lines(e.source, args.desc_chars)
                texts.append(ln.request)
                texts.extend(t for _, t in ln.tools + ln.fields + ln.consts
                             + ln.regs)
                if args.name_words:
                    # the name stage reads names as words; learn pieces for those
                    texts.extend(split_tool_line(t, True)[1] for _, t in ln.tools)
            if args.tok_extra:
                extra = extra_tokenizer_texts(args.tok_extra, args.name_words)
                print(f"  plus {len(extra)} general-English texts from "
                      f"{', '.join(args.tok_extra)}")
                texts.extend(extra)
            tk = train_input_tokenizer(texts, args.in_vocab,
                                       max_length=max(args.max_line, args.max_req))
        tk.no_truncation()       # lengths are checked below; nothing is cut silently
        tk.no_padding()
        tk.save(str(out / "in_tok.json"))
        dims = {"max_line": args.max_line, "max_req": args.max_req,
                "max_reg": args.max_reg}
        if args.split or args.desc_chars != 60:
            dims.update({"desc_chars": args.desc_chars, "split": args.split})
        if args.split:
            dims.update({"max_sig": args.max_sig, "max_desc": args.max_desc,
                         "max_name": args.max_name})
        if args.name_words:
            dims["name_words"] = True
        if args.reader_lines:
            dims["chunk_words"] = args.chunk_words
        if args.in_tok or args.tok_extra:
            config["tokenizer"] = args.in_tok or {"trained_on": "themes+" + ",".join(args.tok_extra)}
        config.update({"in_vocab": tk.get_vocab_size(), "in_pad": pad_token_id(tk),
                       "layout": layout.to_dict(), **dims})

        parts = {}
        max_slots = 0
        encoded = {}
        for name, part in (("train", tr), ("val", va), ("test", te), ("holdout", ho)):
            if part:
                encoded[name] = encode_structural(part, tk, kws, layout, dims, args.canvas)
        if args.split and args.teacher:
            base = teacher_tables(encoded, args.teacher, out)
            if args.reader_lines:
                config["max_chunk"] = reader_line_tables(encoded, base, out)
        for name, d in encoded.items():
            d.pop("teacher_texts", None)
            d.pop("reader_texts", None)
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
