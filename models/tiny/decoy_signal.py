"""How much separates a called tool from its signature-identical decoys, in
the input the tiny model actually reads -- and did the model's choices track it?

The rendered line is `{sym} ({params}) -> {ret} [{effect}] :: {desc}`
(harness/context.py:382): no tool name. Inside a signature group the only
separators are the effect code and the description.
"""
import json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from harness.context import TaskContext, format_type, effect_code  # noqa
from tokenizers import Tokenizer

tk = Tokenizer.from_file(str(ROOT / "models/tiny/data_cache_ho42_reg/in_tok.json"))
tk.no_truncation(); tk.no_padding()

STOP = set("""a an the of to for and or in on at by with from into onto as is are be
it its this that these those any all each every one ones some no not than then
so up out off over under about your you their them they his her our we i me my
what which who whom whose when where how there here also just only still yet
has have had do does did was were been being will would can could should may
might must shall""".split())


def words(s):
    out = set()
    for w in re.findall(r"[a-z]+", s.lower()):
        if w in STOP or len(w) < 3:
            continue
        for suf in ("ing", "ed", "es", "s"):
            if w.endswith(suf) and len(w) - len(suf) >= 3:
                w = w[: -len(suf)]
                break
        out.add(w)
    return out


def sig(t):
    ps = " ".join(f"{p.sym}:{format_type(p.type)}{'' if p.required else '?'}"
                  for p in t.params)
    ret = format_type(t.returns) if t.returns else "-"
    return f"({ps}) -> {ret}"


rows = {}
for line in open(ROOT / "data/s5_holdout_both.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r["id"].endswith("+decoy"):
        rows[r["id"]] = r

gen = [json.loads(l) for l in open(
    Path(sys.argv[1]) if len(sys.argv) > 1 else
    ROOT / "models/tiny/out_reg/ho42_ar_s0_holdout_decoy.jsonl", encoding="utf-8")]

CALL = re.compile(r"CALL (T\d+)")
stats = Counter()
tok_diff, jacc = [], []
buckets = defaultdict(Counter)   # bucket -> {n, right}
lex = Counter()

for g in gen:
    base = g["task_id"].partition("#s")[0]
    row = rows.get(base)
    if row is None:
        stats["no_row"] += 1
        continue
    ctx = TaskContext.from_json(row["context"])
    decoy_names = set(row["provenance"].get("decoys") or [])
    by_sym = ctx.tools
    groups = defaultdict(list)
    for t in by_sym.values():
        groups[sig(t)].append(t)
    req = words(row["request"])
    ref_calls = CALL.findall(g["reference"])
    gen_calls = CALL.findall(g["program"])
    for i, rs in enumerate(ref_calls):
        t = by_sym.get(rs)
        if t is None:
            continue
        sibs = [s for s in groups[sig(t)] if s.sym != t.sym]
        if not sibs:
            stats["ref_call_unique"] += 1
            continue
        stats["ref_call_in_group"] += 1
        eff = effect_code(t.effects)
        same_eff = [s for s in sibs if effect_code(s.effects) == eff]
        eff_sep = len(same_eff) == 0
        stats["effect_alone_separates"] += eff_sep
        # description tokens, real vs every sibling sharing its effect code
        real_tok = tk.encode(t.desc).tokens
        rival = same_eff or sibs
        for s in rival:
            st = tk.encode(s.desc).tokens
            a, b = set(real_tok), set(st)
            jacc.append(len(a & b) / max(len(a | b), 1))
            tok_diff.append(len(a ^ b))
        # lexical: does the real description share more content words with
        # the request than every same-effect rival does?
        ov_real = len(words(t.desc) & req)
        ov_best_rival = max(len(words(s.desc) & req) for s in rival)
        if ov_real > ov_best_rival:
            lk = "real wins on overlap"
        elif ov_real == ov_best_rival:
            lk = "tie"
        else:
            lk = "a decoy wins on overlap"
        lex[lk] += 1
        # the model's choice at the same CALL position
        if i < len(gen_calls):
            chose = gen_calls[i]
            right = chose == rs
            in_grp = chose in {s.sym for s in groups[sig(t)]}
            for b in ("all", "effect separates" if eff_sep else "effect ties",
                      lk):
                buckets[b]["n"] += 1
                buckets[b]["right"] += right
                buckets[b]["in_group"] += in_grp

n = stats["ref_call_in_group"]
print(f"reference CALLs whose tool has a signature twin: {n} "
      f"(unique signature: {stats['ref_call_unique']})")
print(f"  effect tag alone separates the real tool: "
      f"{stats['effect_alone_separates']}/{n} "
      f"({stats['effect_alone_separates']/max(n,1):.1%})")
tok_diff.sort(); jacc.sort()
m = len(tok_diff)
print(f"\ndescription vs each same-effect rival, cache BPE ({m} pairs):")
print(f"  tokens not shared  p10 {tok_diff[m//10]}  median {tok_diff[m//2]}  "
      f"p90 {tok_diff[9*m//10]}")
print(f"  token Jaccard      p10 {jacc[m//10]:.2f}  median {jacc[m//2]:.2f}  "
      f"p90 {jacc[9*m//10]:.2f}")
tot = sum(lex.values())
print(f"\nbag-of-words reader (content words shared with the request), "
      f"{tot} calls:")
for k in ("real wins on overlap", "tie", "a decoy wins on overlap"):
    print(f"  {k:26s} {lex[k]:5d}  ({lex[k]/tot:.1%})")
print("\nthe model's choice at that CALL, by bucket:")
for b in ("all", "effect separates", "effect ties", "real wins on overlap",
          "tie", "a decoy wins on overlap"):
    c = buckets[b]
    if c["n"]:
        print(f"  {b:26s} n={c['n']:5d}  same_tool {c['right']/c['n']:6.1%}  "
              f"picked inside the group {c['in_group']/c['n']:6.1%}")


# --- a request-blind prior: does the model pick the description that looks
# most like the training corpus's, whatever the request says? -------------
def bigrams(s):
    w = re.findall(r"[a-z]+", s.lower())
    return set(zip(w, w[1:]))

train_bg = set()
seen = set()
for i, line in zip(range(30000), open(ROOT / "data/s5_plain.jsonl", encoding="utf-8")):
    r = json.loads(line)
    for t in r["context"]["tools"]:
        if t["desc"] not in seen:
            seen.add(t["desc"])
            train_bg |= bigrams(t["desc"])

def familiar(desc):
    b = bigrams(desc)
    return len(b & train_bg) / max(len(b), 1)

fam = defaultdict(Counter)
cross = defaultdict(Counter)
for g in gen:
    row = rows.get(g["task_id"].partition("#s")[0])
    if row is None:
        continue
    ctx = TaskContext.from_json(row["context"])
    groups = defaultdict(list)
    for t in ctx.tools.values():
        groups[sig(t)].append(t)
    req = words(row["request"])
    ref_calls, gen_calls = CALL.findall(g["reference"]), CALL.findall(g["program"])
    for i, rs in enumerate(ref_calls):
        t = ctx.tools.get(rs)
        if t is None or i >= len(gen_calls):
            continue
        eff = effect_code(t.effects)
        rival = [s for s in groups[sig(t)] if s.sym != t.sym
                 and effect_code(s.effects) == eff]
        if not rival:
            continue
        fr, fb = familiar(t.desc), max(familiar(s.desc) for s in rival)
        fk = ("real looks more familiar" if fr > fb else
              "tie" if fr == fb else "a decoy looks more familiar")
        ov_r = len(words(t.desc) & req)
        ov_b = max(len(words(s.desc) & req) for s in rival)
        lk = "request favours real" if ov_r > ov_b else (
             "request ties" if ov_r == ov_b else "request favours a decoy")
        right = gen_calls[i] == rs
        fam[fk]["n"] += 1; fam[fk]["right"] += right
        cross[(fk, lk)]["n"] += 1; cross[(fk, lk)]["right"] += right

print("\nfamiliarity = share of a description's word bigrams seen in training "
      f"tool descriptions ({len(seen)} distinct training descriptions)")
for k in ("real looks more familiar", "tie", "a decoy looks more familiar"):
    c = fam[k]
    if c["n"]:
        print(f"  {k:28s} n={c['n']:5d}  same_tool {c['right']/c['n']:6.1%}")
print("\nboth cues at once (same_tool, n):")
fks = ("real looks more familiar", "tie", "a decoy looks more familiar")
lks = ("request favours real", "request ties", "request favours a decoy")
print(f"  {'':28s}" + "".join(f"{l:>26s}" for l in lks))
for fk in fks:
    cells = []
    for lk in lks:
        c = cross[(fk, lk)]
        cells.append(f"{c['right']/c['n']:6.1%} (n={c['n']:4d})" if c["n"] else "-")
    print(f"  {fk:28s}" + "".join(f"{x:>26s}" for x in cells))
