"""R18: at lookalike pairs (the needed tool R and its same-shape sibling S of
another entity), when the pick is one of the two: does it follow list
position, and does it lean toward the parent entity (the one the other's
fields point at: spot.advertiser makes advertiser the parent of spot)?
Teacher-forced, the pick restricted to the task's tools.

  cd models/tiny && python ../../results/logs/pair_lean_r18.py
"""
import pickle, re, sys, importlib.util
from pathlib import Path
import torch
sys.path.insert(0, '.'); sys.argv = sys.argv[:1]
from evaluate import Split, load_model
spec = importlib.util.spec_from_file_location('c', str(Path(__file__).parent / 'cue_conflict_r18.py')); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
torch.set_grad_enabled(False)
want = ('A0   0% S6, seed 0', 'A0   25% S6', 'A0   100% S6 (R13)', 'SPt  25% S6')
for label, ckpt, cache in [m for m in c.MODELS if m[0] in want]:
    split = Split(Path(cache), 'holdout', torch.device('cpu'))
    rows = {r['id']: r for r in pickle.load(open(Path(cache) / 'rows.pkl', 'rb'))['holdout']}
    model = load_model(Path(ckpt), torch.device('cpu'))
    n = right = earlier = parent = parent_right = parent_n = 0
    done = 0
    for i, m in enumerate(split.meta):
        tid = m['task_id']
        if '+' in tid or '#' in tid: continue
        row = rows.get(tid)
        if row is None or len(row['reference']['segments']) != 1: continue
        dec = dict(c.decisions(row))
        if not dec: continue
        ov = split.codec(i); tgt = split.target(i)
        canvas = torch.cat([torch.full_like(tgt[:, :1], 1), tgt[:, :-1]], 1)
        logits = model.decode(split.inputs(i, model), canvas)[0]
        toks = ov.decode(tgt[0].tolist())
        tools = {t['sym']: t for t in row['context']['tools']}
        order = [t['sym'] for t in row['context']['tools']]
        # parent: an entity that some field of the other entity points at (spot.advertiser: ID:advertiser)
        refs = {(f.get('entity'), (f.get('type') or '')[3:]) for f in row['context']['fields'] if (f.get('type') or '').startswith('ID:')}
        seen = set()
        for p in range(1, len(toks)):
            r = toks[p]
            if toks[p - 1] != 'CALL' or r not in dec or r in seen: continue
            seen.add(r); s = dec[r]
            syms = list(tools); jid = [split.layout.tool0 + ov.tool_id[x] for x in syms]
            pick = syms[int(logits[p, jid].argmax())]
            if pick not in (r, s): continue
            n += 1; right += pick == r
            earlier += order.index(pick) < order.index(r if pick == s else s)
            er, es = c.entity(tools[r]), c.entity(tools[s])
            par = er if (es, er) in refs else es if (er, es) in refs else None
            if par:
                parent_n += 1; parent += c.entity(tools[pick]) == par
                parent_right += er == par
        done += 1
        if done >= 800: break
    print(f"{label:22s} pair decisions {n}: right {right/n:5.1%}   picked the earlier-listed tool {earlier/n:5.1%}   "
          f"picked the parent entity {parent/max(parent_n,1):5.1%} (parent is right {parent_right/max(parent_n,1):5.1%}, n={parent_n})", flush=True)
