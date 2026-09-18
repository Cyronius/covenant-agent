# Running this on a pod

This machine cannot be trusted for timings. Two measurements of the same code
disagreed by a factor of three in the wrong direction, and the cause was never
isolated on a box where process enumeration hangs under load. So the division of
labour is not about convenience:

- **laptop** — prepare data, check correctness, score results. All reliable here.
- **pod** — every number. Training, generation, and anything with a clock on it.

## One command, start to finish

On the laptop:

```bash
python prep.py --limit 30000          # once; writes data_cache/, ~376 MB
bash selftest.sh                      # correctness only, no timing
python pack.py --out pod_bundle.tar.gz
```

Upload the bundle, then on the pod:

```bash
tar xzf pod_bundle.tar.gz && cd tiny
pip install torch tokenizers
bash run_phase1.sh
```

Bring back `out/*.jsonl` and `runs/*/log.jsonl`, then on the laptop:

```bash
for f in out/*.jsonl; do python evaluate.py --score --gen-out "$f" --split test; done
python probe.py --gen out/diff_s0_k8.jsonl --compiled-only
```

## What the bundle contains

Everything needed, nothing else. About 380 MB.

| | why |
|---|---|
| `tiny/*.py` | the experiment |
| `tiny/data_cache/` | pre-tokenized tensors, so the pod never parses the 500 MB corpus nor trains a tokenizer |
| `core/`, `harness/context.py` | covenant-agent's parser, typechecker and compiler: pure stdlib, 214 KB |

Shipping the compiler means the repair loop runs on the pod rather than being
the one arm that waits for the laptop. The Node sandbox stays behind, because
executing a program against world state is scoring, and scoring happens after
the pod is shut down.

## What `run_phase1.sh` does

1. **A sanity run first.** 256 rows, a few steps. It proves the model runs on
   this device and puts a real step time in the log before anything long starts.
2. **Both arms, three seeds.** Skips any arm already trained, so an interrupted
   session resumes instead of restarting.
3. **The control**, generated once per seed on test and on the held-out world.
4. **The diffusion arm across step counts** 1, 2, 4, 8, 16, 32. This sweep is
   the experiment. The control has no equivalent knob, which is why results get
   reported against forward passes rather than steps.
5. **The compiler in the loop** at 8 steps, to compare against plain 8 steps.
   The quality difference is what the critic bought; the pass count is what it
   cost.
6. **A capacity check** at `d=512` and 6 layers each side. If the two arms swap
   places between the small and large models, the small result was a capacity
   artifact and the large one is the finding.

Environment overrides: `SEEDS=0` for a first look, `SKIP_BIG=1` to drop the
capacity check, `EPOCHS`, `BATCH`, `STEPS_SWEEP`.

## Sizing and cost

25,463 training rows, 398 steps per epoch at batch 64. The encoder over ~1100
input tokens is the dominant cost. A 3090 or A10 class card should do a step in
well under a second, putting one arm-seed at a few minutes and the whole script
including the capacity check inside one to two GPU-hours.

Start with `SEEDS=0 SKIP_BIG=1` to confirm the numbers look sane, then run the
full thing. That first pass costs maybe fifteen minutes and protects the rest.

Batch 64 at 24 GB is comfortable. The memory note in the README explains why
batch size matters more than it looks: attention over the input holds an
`S x S` matrix per head for the backward pass.
