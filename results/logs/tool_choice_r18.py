"""R18: teacher-forced tool choice on the S6 plain exam, restricted to the
task's tools as decoding's CALL mask is, for every model in
cue_conflict_r18.MODELS: accuracy where the needed tool has a same-shape
sibling of another entity (a lookalike) against where its shape is unique,
and where the wrong picks at lookalike pairs go.

  cd models/tiny && python ../../results/logs/tool_choice_r18.py
"""
import pickle, sys, importlib.util
from collections import Counter
from pathlib import Path
import torch
sys.path.insert(0, '.'); sys.argv = sys.argv[:1]
from evaluate import Split, load_model
spec = importlib.util.spec_from_file_location('c', str(Path(__file__).parent / 'cue_conflict_r18.py')); c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
torch.set_grad_enabled(False)
for label, ckpt, cache in c.MODELS:
    split = Split(Path(cache), 'holdout', torch.device('cpu'))
    rows = {r['id']: r for r in pickle.load(open(Path(cache) / 'rows.pkl', 'rb'))['holdout']}
    model = load_model(Path(ckpt), torch.device('cpu'))
    t0 = split.layout.tool0
    acc = {'sibling': [0, 0], 'unique': [0, 0]}; where = Counter(); done = 0
    for i, m in enumerate(split.meta):
        tid = m['task_id']
        if '#' in tid or ('+' in tid and not tid.endswith('+s6')): continue
        row = rows.get(tid)
        if row is None or len(row['reference']['segments']) != 1: continue
        ov = split.codec(i); tgt = split.target(i)
        canvas = torch.cat([torch.full_like(tgt[:, :1], 1), tgt[:, :-1]], 1)
        logits = model.decode(split.inputs(i, model), canvas)[0]
        toks = ov.decode(tgt[0].tolist())
        tools = {t['sym']: t for t in row['context']['tools']}
        syms = list(tools)
        jid = [t0 + ov.tool_id[s] for s in syms]
        dec = dict(c.decisions(row))
        for p in range(1, len(toks)):
            r = toks[p]
            if toks[p - 1] != 'CALL' or r not in tools: continue
            pick = syms[int(logits[p, jid].argmax())]
            sib = any(s != r and c.shape(t) == c.shape(tools[r]) and c.entity(t) and c.entity(t) != c.entity(tools[r]) for s, t in tools.items())
            k = 'sibling' if sib else 'unique'
            acc[k][0] += pick == r; acc[k][1] += 1
            if r in dec and pick != r:
                where['its lookalike' if pick == dec[r] else ('another tool, same entity' if c.entity(tools[pick]) == c.entity(tools[r])
                      else 'a generic tool (no entity)' if c.entity(tools[pick]) is None else 'another tool, other entity')] += 1
        done += 1
        if done >= 800: break
    w = sum(where.values())
    print(f"{label:22s} with a lookalike {acc['sibling'][0]/acc['sibling'][1]:6.1%}  shape-unique {acc['unique'][0]/acc['unique'][1]:6.1%}   "
          f"wrong at lookalike pairs: " + ", ".join(f"{k} {v/w:.0%}" for k, v in where.most_common()), flush=True)
