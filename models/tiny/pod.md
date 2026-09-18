# Running this on a pod

This machine cannot be trusted for timings. Two measurements of the same code
disagreed by a factor of three in the wrong direction, and the cause was never
isolated on a box where process enumeration hangs under load. So the division of
labour is not about convenience:

- **laptop** — prepare data, check correctness, score results. All reliable here.
- **pod** — every number. Training, generation, and anything with a clock on it.

## Step 1: structural binding on the R3 model

`.claude/plans/npu-native-planner.md`, sequence step 1. Same data as R3
(`s5_plain`, all 28,568 usable single-segment tasks: 25,463 train, 1,415 val,
1,415 test, the `bookstore` world's 275 tasks held out), same scorer, both arms,
three seeds, structural binding. Gate: `acc_tool` on the curve leaves chance
(about 6%), and compile rate on the laptop leaves single digits.

On the laptop, once:

```bash
python prep.py --limit 30000                     # data_cache_struct/, ~306 MB
bash selftest.sh data_cache_struct               # correctness only; must end ALL PASS
python pack.py --cache data_cache_struct --out pod_bundle.tar.gz    # ~26 MB compressed
```

Upload the bundle to a staging path and rename it into place (never upload over
a file a job may be reading). Then on the pod:

```bash
tar xzf pod_bundle.tar.gz && cd tiny
pip install torch tokenizers
SEEDS=0 CACHE=data_cache_struct bash run_step1.sh > step1_s0.log 2>&1   # first look
CACHE=data_cache_struct bash run_step1.sh > step1.log 2>&1              # all three seeds
```

Watch `runs/s1_*/log.jsonl`, not the console: the trainer flushes a JSON line at
every evaluation with `val_loss`, `acc_kw acc_tool acc_field acc_const acc_reg`
and the chance rates. `acc_tool` is the number to read first; it is available
while the pod is running and it is the gate. Never download a file a job is
still writing.

Bring back `out/` (every generation, plus `out/curves/<run>/log.jsonl` and
`config.json`, which the script copies at the end so nothing is left behind),
then on the laptop:

```bash
for f in out/s1_*_test.jsonl out/s1_*_k*.jsonl; do
  python evaluate.py --score --cache data_cache_struct --gen-out "$f" --split test
done
for f in out/s1_*_holdout.jsonl; do
  python evaluate.py --score --cache data_cache_struct --gen-out "$f" --split holdout
done
python probe.py --gen out/s1_diff_s0_k8.jsonl --compiled-only      # fill order, decision 11
python diagnose.py --gen out/s1_diff_s0_k8.jsonl --cache data_cache_struct
cp -r out/curves/* ../../results/logs/r3/curves/                   # and read them
```

Each `.score.json` carries, beside parse/compile/goal, `slot_acc` per kind and
`call_agreement` (same tool / same effect against chance), so the R3 section-3
table can be reproduced from tracked files.

**The R3 baseline through the same script.** Prepare a flat cache and point the
script at it; the binding is read from the cache and nothing else changes:

```bash
python prep.py --limit 30000 --binding flat --out data_cache_flat
python pack.py --cache data_cache_flat --out pod_bundle_flat.tar.gz
# pod
CACHE=data_cache_flat TAG=flat bash run_step1.sh
```

Note this flat cache uses the split vocabulary (401 entries, `r0.` `F6` as two
slots), which is the one representational change both bindings share. The
run 1 and run 2 caches (`data_cache`, `data_cache_ptr`) are untouched and still
evaluate their checkpoints.

Environment overrides for `run_step1.sh`: `SEEDS`, `EPOCHS` (12), `BATCH` (64),
`STEPS_SWEEP` (1 2 4 8 16 32), `DEC_LOOPS` (1; step 2 sweeps it), `TAG` (s1).

## What the bundle contains

Everything needed, nothing else.

| | why |
|---|---|
| `tiny/*.py`, the scripts | the experiment |
| `tiny/data_cache_struct/` | per-line token tensors, graph edges, canvas targets with pointer ids, `keywords.json`, the input tokenizer, the raw rows for scoring |
| `core/`, `harness/context.py` | covenant-agent's parser, typechecker and compiler: pure stdlib, 214 KB |

Shipping the compiler means the repair loop runs on the pod rather than being
the one arm that waits for the laptop. The Node sandbox stays behind, because
executing a program against world state is scoring, and scoring happens after
the pod is shut down.

## What `run_step1.sh` does

1. **A sanity run first.** 256 rows, a few steps. It proves the model runs on
   this device, checks the log carries per-slot accuracy, and puts a real step
   time in the log before anything long starts.
2. **Both arms, three seeds, 12 epochs.** Twice run 1's length: its control
   converged under 6 epochs while still at chance on symbols, and the diffusion
   arm was still improving. Skips any run already trained.
3. **The control**, generated once per seed on test and on the held-out world.
4. **The diffusion arm across step counts** 1, 2, 4, 8, 16, 32 on test, and 8 on
   the held-out world.
5. **The compiler in the loop** at 8 steps, 2 rounds, to compare against plain
   8 steps.
6. **The curves copied into `out/curves/`.** Runs 1 and 2 brought theirs back
   and nobody read them; this time they travel with the generations.

## Sizing and cost

25,463 training rows, 398 steps per epoch at batch 64, 12 epochs, about 4,800
steps per arm-seed. The structural encoder is lighter per example than the flat
one (attention over 64-token lines rather than an 1,100-token stream), so a
3090 or 4090 should do a step in well under R3's 0.19 s; expect each arm-seed
inside fifteen minutes and the whole script, three seeds, inside about two
GPU-hours. Measure the sanity run's step time before trusting that.

Start with `SEEDS=0` to confirm the numbers look sane, then run the full thing.
