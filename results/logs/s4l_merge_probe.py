"""Can the layers we want to tie be MERGED rather than picked?

Tie plan: L4-L19 collapse into one 4-layer group looped 4x. Position p of
that group must stand in for layers {4+p, 8+p, 12+p, 16+p}. This asks two
questions about each such merge set:
  1. does averaging survive?  (norm of the mean vs mean of the norms)
  2. if base+LoRA, what rank does the residual actually need?
"""
import glob, os, re, torch
from safetensors.torch import safe_open

D = 'baselines/qwen/models/merged_s1_pruned_v2_hf'
W = {}
with safe_open(sorted(glob.glob(os.path.join(D, '*.safetensors')))[0], framework='pt') as g:
    for k in g.keys():
        m = re.match(r'model\.language_model\.layers\.(\d+)\.(.+)', k)
        if m:
            W[(int(m.group(1)), m.group(2))] = g.get_tensor(k).float()

TENSORS = ['mlp.down_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight']

print("MERGE SETS: position p of the shared group stands in for L{4+p,8+p,12+p,16+p}\n")
for p in range(3):                      # positions 0-2 are linear-attention layers
    layers = [4 + p, 8 + p, 12 + p, 16 + p]
    print(f"=== group position {p}  <- layers {layers}")
    for t in TENSORS:
        if any((l, t) not in W for l in layers):
            continue
        S = torch.stack([W[(l, t)] for l in layers])
        mean = S.mean(0)
        nm, nrm = mean.norm().item(), S.norm(dim=(1, 2)).mean().item()
        cs = [torch.nn.functional.cosine_similarity(mean.flatten(), W[(l, t)].flatten(), dim=0).item()
              for l in layers]
        print(f"  {t:22s} ||mean||/mean||W|| = {nm/nrm:.3f}   cos(mean, each) = "
              + ' '.join(f'{c:+.3f}' for c in cs))
        # how much of each layer does a rank-r correction on top of the mean recover?
        line = []
        for l in layers[:1]:
            R = W[(l, t)] - mean
            sv = torch.linalg.svdvals(R)
            tot = (sv ** 2).sum()
            for r in (8, 32, 64, 128, 256):
                line.append(f"r{r}={(sv[:r]**2).sum()/tot*100:4.1f}%")
        print(f"  {'':22s} rank-r energy of (L{layers[0]} - mean): " + '  '.join(line))
        # baseline: same question but base = a picked layer (stepwise init)
        R = W[(layers[0], t)] - W[(layers[2], t)]
        sv = torch.linalg.svdvals(R); tot = (sv ** 2).sum()
        print(f"  {'':22s} rank-r energy of (L{layers[0]} - L{layers[2]}) : "
              + '  '.join(f'r{r}={(sv[:r]**2).sum()/tot*100:4.1f}%' for r in (8, 32, 64, 128, 256)))
    print()

print("full rank of these matrices is 1024 -- a rank-32 LoRA can carry at most 32/1024 of the")
print("directions, so 'energy at r=32' is the ceiling on what relaxed tying can restore at that rank.")
