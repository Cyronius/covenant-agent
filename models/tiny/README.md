# Canvas: the R3 tiny planner

Does a masked-diffusion generator write better programs than a left-to-right one,
at the same size and the same compute?

The task is covenant-agent's: read an English request plus a list of tools, write
a short program that carries it out. Programs are 30 tokens at the median over a
vocabulary of a few hundred symbols, so the whole program fits on one 64-slot
canvas and can be denoised in a single pass. There is no sliding window here and
there does not need to be.

The reason to ask the question on programs rather than prose is that the answer
is checkable. Covenant-agent ships a parser, a typechecker and a sandbox, so a
generated program is right or wrong against world state, not against a reference
string. That also makes the compiler usable *inside* the sampler, which is the
part of this worth being curious about.

## What is being compared

Two arms, the same 7.9M-parameter encoder-decoder, differing in one line:

| arm | decoder self-attention | input | output |
|---|---|---|---|
| `diffusion` | unmasked | canvas with slots hidden | a prediction for every slot |
| `ar` | causal | target shifted right | the next token |

The control is not a strawman. It shares the encoder, the widths, the parameter
count and the data. Results are reported against forward passes, not steps,
because the control spends one pass per token while the diffusion arm spends
however many it is given.

## The planted dependency

Reference programs have no `EFFECTS` header. This pipeline derives one and puts
it on the first line. The header is the union of the effects of every tool the
program calls, so the top of the program is determined by the bottom. A
left-to-right generator has to predict it before writing the calls. A diffusion
generator can leave it until last. Header accuracy is reported on its own.

## Structural binding (step 1 of the NPU-native planner)

`results/R3.md` section 3 diagnosed why both 7.9M arms sat at chance on tool
selection: the input BPE tokenizer splits `T23` into `T` `2` `3`, nothing ties
the input `T8` to the output token `T8`, and the output vocabulary holds `r0.F6`
as one token, so a third of the field references are not even symbol slots.
`.claude/plans/npu-native-planner.md` (decisions 4, 6, 7, 8, 11) replaces the
representation rather than the tokenizer. That is what `--binding structural`
builds; `--binding flat` is the R3 model, unchanged.

**What changed.** One flag to `prep.py`, recorded in the cache's `config.json`;
`train.py` and `evaluate.py` build whichever model the cache declares.

| | flat (R3) | structural |
|---|---|---|
| context | one token stream, about 1,100 tokens | one token tensor per tool line, field line and constant line, plus the request tokens |
| encoder | 3 layers over the stream | a line encoder (one vector per line, pooled through a CLS token), a sparse graph pass over the line vectors, a turn encoder over tools + fields + constants + request |
| region A | the encoder states | a tagged sequence: TOOL, FIELD, CONST, REQUEST vectors, then the canvas tagged CANVAS |
| output | a row per vocabulary entry (401) | a fixed keyword head (62 rows) plus pointer scores: dot products against this task's tool, field and constant vectors and 32 register embeddings |
| undeclared symbol | masked, if `--pointer` | has no row; cannot be produced |
| canvas input | the token's embedding | a keyword's embedding, or the region-A vector the slot points at |

The graph pass is sparse by construction: a tool attends to the fields its
signature names (`=F2`), a field attends to the tools that name it and to the
other fields of its entity, and every line attends to itself. The edges come
from the serialized signature text and are stored in the cache (`adj`). The
line encoder and graph pass depend only on the world's schema and are separate
functions (`encode_world`), so they are cacheable per world as decision 6
says; the turn encoder (`encode_turn`) is the per-turn part. Caching itself is
not implemented.

**The canvas** (`canvas.py`). Every slot holds one joint id over

```
[ keywords (62) | tools (<=18) | fields (<=23) | constants (<=10) | registers (32) ]
```

PAD is 0 and MASK is 1 as before. The pointer ranges beyond what a task declares
are masked to `-inf` in the forward pass, so the grammar's symbol half is
automatic; the keyword half is a per-task `kw_allowed` mask (everything but
MASK, and an effect name only if some declared tool carries it), applied in the
same place. Two register forms: `r0`..`r15` (an operand) and `r0.`..`r15.` (the
receiver of a field access). `r0.F6` is two slots, `r0.` then `F6`, so every
field reference is a field slot the pointer head can reach.

**The split is shared by both bindings**, because it lives in
`corpus.program_tokens` / `detokenize` (plan:
`.claude/plans/canvas-field-access-split.md`). Forward: a whitespace token
matching `^r(\d{1,2})\.(F\d+)$` becomes `r<n>.` then `F<k>`. Inverse: when a
line is joined, a part ending in `.` absorbs the next token. Both are total
functions of the token sequence with no grammar knowledge; a dangling `r0.`
renders verbatim and is a `PARSE_ERROR`, never repaired. The flat vocabulary
goes from 500 entries (115 compounds) to 401 (0 compounds); the longest program
in `s5_plain` goes from 56 to 57 of 64 slots, and nothing is excluded.
`MAX_PROGRAM_TOKENS` is now checked on the split count. The old caches
(`data_cache`, `data_cache_ptr`, 500-entry vocabulary) still load and their
checkpoints still generate byte-identical programs; a fresh flat cache is the
same architecture on the split vocabulary.

The samplers also apply one local rule from the canvas itself
(`sample.local_mask`): after a receiver slot the next slot must be a field,
and a receiver cannot sit before a filled non-field. Everything entity-aware
(a field of the entity in `r0`, a register bound before use) stays with the
host typechecker in the repair loop, as decision 8 says.

**Instrumentation.** Every evaluation in `runs/<name>/log.jsonl` records, beside
`val_loss`, `acc_kw acc_tool acc_field acc_const acc_reg` with their counts and
the exact chance rates (`chance_tool` is the mean of 1/n_tools). For the
diffusion arm the symbol accuracies come from one pass over the reference
canvas with every pointer slot hidden; for the control they are teacher-forced.
`evaluate.py --score` adds per-slot-kind accuracy against the reference canvas
(same slot index) and the line-aligned CALL agreement from `diagnose.py` (same
tool, same effect class, each against chance). `probe.py` reads the surface
tokens `evaluate.py` writes per slot, so it works under either binding.

**Model size.** Structural at the R3 widths (d=256, line 2, graph 1, turn 2,
decoder 4) is 9.4M parameters against the flat 7.9M; the difference is the
extra encoder layers. `--dec-loops` applies the decoder stack repeatedly under
both bindings (step 2 of the sequence).

**Step 1's result is in `results/R7.md`.** Tool slots 6% to 99.5%, compile 8.6%
to 99.9%, and 96.4% goal success on a world the model never trained on. Read it
before running anything below, because it moves both of the next steps: the
model is saturated, so step 2 sweeps a smaller block, and the control arm beats
the diffusion decode by 10 points, which is what step 2 has to close.

## The loop and the weight format (steps 2 and 3)

The NPU holds one stage's weights in 4 MB of memory tiles and pays nothing to
reapply them, so the two knobs the design cares about are how many times the
block runs and how many bits a weight costs. Both are flags here, and both are
swept by a script.

**The loop.** `--dec-loops L` applies the decoder stack L times with the same
weights, so the parameter count does not move (`test_quant.py` asserts that,
because a sweep whose parameters drift is measuring the wrong thing). Two
optional knobs come with it, both cheap on the NPU:

| flag | what | why it is a knob and not a default |
|---|---|---|
| `--loop-emb` | one learned bias vector per iteration, added to the residual at the start of each | 8k parameters and one add per loop on chip, but it is a change to the recorded design, so it is measured beside plain looping rather than assumed |
| `--rand-loops N` | sample the loop count from 1..N per batch | decision 9 calls L "the effort dial", which only holds if one checkpoint serves several L; a model trained at a fixed L has never seen another |

`evaluate.py --dec-loops N` turns the dial at inference, overriding whatever the
checkpoint trained at. The generation file records the loop count and block
depth per row, so `report.py` can draw goal success against block applications
rather than against denoising steps.

**The weight format.** `--weights {fp,int8,u4,tern}` trains the transformer
stacks in the format they will be deployed in, from step zero, with int8
activations per token (`--act-bits`) and the embeddings and heads held at int8
(`--quant-ends`). `quant.py` has the details and the reasoning; the short version
is that it is fake quantisation through a straight-through estimator, applied as
a `torch.nn.utils.parametrize` parametrization so it also covers
`nn.MultiheadAttention`'s packed `in_proj_weight`, and so the checkpoint keeps
the master float weights. Every run prints and records what its loop body costs
on chip:

```
weights=tern act_bits=8 loops=8 loop_body=4.21M params = 1.00 MB resident of 4.00 MB
```

Step 1's model is that line: 4.21M parameters in the loop body, which is 4.02 MB
at int8, 2.01 MB at 4-bit and 1.00 MB at ternary. All three fit.

**What the two sweeps are for.** `run_step2.sh` asks whether goal success rises
with L at fixed parameters, and draws the unlooped depth ladder beside it so a
win from looping is never confused with a win from more distinct weights.
`run_step3.sh` compares the four formats at *matched resident bytes* — int8 at
one layer, 4-bit at two, ternary at four — because the question is not whether
rounding hurts (it does) but which format writes better programs in the same
4 MB. Both scripts skip work that already exists, so an interrupted sweep
resumes, and both take `EXTRA`/`GEN_EXTRA` so the whole thing can be smoke-tested
locally on a tiny model before it costs pod time.

One kernel measurement changed step 3 before it ran.
`models/npu/kernels/tern_mk/README.md` found that this part has a native
`mmul<4,16,16,int8,uint4>` whose unpack is folded into the MAC, while ternary has
no 2-bit equivalent and pays 0.0156 cycles per weight in software. So decision 3's
"ternary strictly dominates 4-bit" does not hold here, the fallback from ternary
is 4-bit at 8M rather than int8 at 4M, and the format question is capacity
against a 12.5% tax rather than a free win.

## Running it

```bash
python prep.py --limit 30000                        # structural cache: data_cache_struct/
python prep.py --limit 30000 --binding flat --out data_cache_flat   # the R3 baseline
bash selftest.sh data_cache_struct                  # round trip must be 40/40 and 20/20
python train.py --arm diffusion --cache data_cache_struct --epochs 12 --batch 64 --pad-weight 0.5
python train.py --arm ar        --cache data_cache_struct --epochs 12 --batch 64
python evaluate.py --ckpt runs/diffusion_s0/best.pt --cache data_cache_struct --split test \
                   --steps 8 --repair-rounds 2 --gen-out runs/diff_test.jsonl
python probe.py --gen runs/diff_test.jsonl --compiled-only
python diagnose.py --gen runs/diff_test.jsonl --cache data_cache_struct
```

The sweeps, rather than single runs:

```bash
bash run_step2.sh                 # the loop sweep, the depth ladder, the dial
bash run_step3.sh                 # four weight formats at matched resident bytes
python report.py --dir out        # the tables, and step 2's gate verdict
```

`test_pipeline.py` is the one to run first and after any change to the data path.
It takes reference programs out of the tensor cache, decodes them back to text,
compiles them and executes them. If that is not 100%, no model number below it
means anything.

## Splitting the work across machines

The sandbox needs Node; a GPU pod does not. So:

```bash
# laptop
python prep.py --limit 30000
python pack.py --cache data_cache_struct --out pod_bundle.tar.gz
# pod: needs torch and tokenizers, the cache directory, and pure-Python imports only
CACHE=data_cache_struct bash run_step1.sh
# laptop
python evaluate.py --score --cache data_cache_struct --gen-out out/s1_diff_s0_k8.jsonl --split test
```

Smoke-test locally with `--limit-train 128 --d 128 --enc-layers 2 --dec-layers 2`
(batch 4 to 8 on a CPU), then run seeds and sweeps on a pod. On the structural
smoke cache that configuration is 3.0M parameters and about 2 s/step here,
which per the memory section below is not a number to size anything by.

Watch a long run through `runs/<name>/log.jsonl`, which the trainer writes and
flushes at every evaluation. Do not watch it by piping the console output through
`tail` or `head`: those buffer until the process exits, so the log looks empty
for hours and a run that has died looks identical to one that is working.
Redirect to a file instead, or read the JSONL.

## Files

| file | what |
|---|---|
| `corpus.py` | load tasks, derive the header, program text to tokens and back; splits `r0.F6` into `r0.` `F6` and folds it back |
| `tok.py` | byte-pair tokenizer for the input, exact symbol table for the flat output |
| `canvas.py` | structural binding: the keyword table, the joint-id layout, the per-task codec (text to pointer ids and back), the keyword mask |
| `prep.py` | tokenize once, write tensors, refuse to truncate silently; `--binding structural|flat` |
| `model.py` | `CanvasModel` (flat) and `StructuralModel` (line encoder, graph pass, turn encoder, keyword + pointer head); the masked-diffusion objective |
| `train.py` | both arms, one flag; per-slot-kind accuracy at every evaluation |
| `sample.py` | cosine-schedule unmasking, the receiver rule, and the compiler repair loop |
| `evaluate.py` | generate programs, then score them in the sandbox; slot accuracy and CALL agreement |
| `diagnose.py` | signature uniqueness, and the line-aligned CALL agreement (also used by `evaluate.py --score`) |
| `probe.py` | when the sampler decides each kind of token (decision 11's evidence) |
| `ablate_desc.py` | permute descriptions or requests at inference (flat binding only) |
| `report.py` | the accuracy-against-passes table over a directory of scored runs, plus the loop sweep and its gate verdict |
| `quant.py` | the weight formats (int8, 4-bit, ternary) as quantisation-aware training, and the resident-byte arithmetic |
| `sandbox.py` | register the 104 generated theme worlds so the sandbox can run them |
| `test_pipeline.py` | the reference round trip, the check everything else rests on |
| `test_samplers.py` | both samplers reproduce an oracle; the receiver rule never blocks a reference |
| `test_quant.py` | the quantised model is the deployed model: level counts, per-row scales, the estimator, the checkpoint round trip |
| `selftest.sh` | the correctness checks, on either binding |
| `run_step1.sh` | the pod script for step 1: both arms, three seeds, the step sweep, curves copied out |
| `run_step2.sh` | step 2: the loop-count sweep at fixed parameters, the unlooped depth ladder beside it, and the effort dial |
| `run_step3.sh` | step 3: four weight formats at matched resident bytes |
| `run_phase1.sh` | the R3 phase-1 recipe (6 epochs, flat) |
| `pack.py` | the pod bundle: code, cache, compiler |

Not named `data.py` on purpose: that shadows covenant-agent's `data` package and
breaks every import of the theme generator.

## Memory, and why a too-large batch looks like a slow model

Under the flat binding the encoder reads about 1100 input tokens, and
self-attention over them holds an `S x S` score matrix per head that autograd
keeps for the backward pass (the structural binding's largest attention is the
line encoder's, 64 x 64 over about 50 lines per task, so it is far lighter per
example; the guidance below is for flat):

```
batch x heads x 1280 x 1280 x 4 bytes   ~= 105 MB per head-batch unit
batch 16, 4 heads                       ~= 420 MB per attention layer
x2 for the softmax intermediate, x5 layers ~= 4 GB
```

Add a 300 MB dataset and a laptop without much free RAM starts swapping. That
does **not** raise an out-of-memory error. Training simply gets slower, by a
factor of five or more, with no message saying why -- and on a badly swapping
machine even `python -c "print(1)"` stops returning, which makes it look like
something far stranger than a batch size mistake.

Guidance:

| device | batch | note |
|---|---|---|
| CPU, 16 GB | 4 to 8 | measure the first step time before walking away |
| CPU, 32 GB+ | 8 to 16 | |
| 24 GB GPU | 32 to 64 | the intended setting |

The tensors load memory-mapped and `--limit-train` slices before reading, so a
short run does not pull the whole 300 MB in. Validation defaults to 256 rows
rather than all 1415, because evaluating the full split costs more than the
training between evaluations.

Measured step times on the development CPU, which delivers about 22 GFLOP/s:

| config | batch | s/step |
|---|---|---|
| d=256, enc 3, dec 4 | 16 | 8.7 |
| d=192, enc 2, dec 3 | 4 | 29.7 |

Those two are the wrong way round -- the second is a smaller model with a
quarter of the batch and it measured three times slower. Arithmetic on the layer
shapes predicts about 3 s/step for it, so the second number is roughly ten times
worse than it should be and the first is the only one that ever matched
expectation. The cause was not isolated. Both were taken with nothing else
knowingly running, on a machine where process enumeration hangs under load, so
"nothing else running" could not actually be confirmed.

Read this as: **do not trust CPU step times here, and do not size a run by
them.** Measure the first step of a pod run instead, where the numbers behave.
If a CPU step costs far more than the shape arithmetic predicts, suspect memory
or a competing process before suspecting the model.

## Things to watch

**Padding collapse.** Programs fill about half the canvas, so most slots are
padding and an undertrained model predicts padding everywhere, which is an empty
program. Knowing where a program ends is genuinely part of the job, so padding
keeps a share of the loss, but `--pad-weight` makes the share adjustable rather
than accidental.

**The schedule, which turned out to matter more than the step count.** The cosine
schedule commits a fixed number of slots per pass whether or not the model is sure
of them, so every program costs exactly `--steps` passes. `--threshold 0.99`
commits instead every slot the model is at least that confident of, and at least
one so it cannot stall, with `--steps` as a cap. Measured on step 1's diffusion
checkpoint, full test split:

| schedule | mean passes | compile | goal |
|---|---|---|---|
| cosine, 8 steps | 8.0 | 90.7% | 86.1% |
| cosine, 8 steps + compiler repair | 8.6 | 94.1% | 88.2% |
| cosine, 32 steps | 32.0 | 94.8% | 88.9% |
| **threshold 0.99** | **2.7** | **97.1%** | **89.4%** |
| threshold 0.99 + compiler repair | 2.9 | 97.2% | 89.5% |

Better than 32 cosine steps at a twelfth of the passes, and better than the
compiler in the loop without needing the compiler. It wins at every level, most of
all where the cosine schedule was weakest: level 3 (a compound filter predicate)
goes 1% -> 16% -> 18% across the three rows above, level 5 (IF/ELSE) 48% -> 67% ->
71%. What a pass buys is not refinement, it is a smaller commit granularity, and
paying 32 passes on every program to get it is waste: most programs have nothing
hard in them and finish in one or two passes, while a compound predicate takes
about fifteen. That is compute proportional to difficulty, which is what the NPU
wants from a decode.

**The argmax.** The sampler takes the most likely token, which is the opposite of
what the prose decoder in the parent design does. That is deliberate. Here the
target is near-deterministic given the input and a compiler decides correctness,
so the mode is the right thing to want. Sampling instead is a knob to try later,
and it should hurt.

**One world has a long schema.** Input length is capped at 1280 tokens, which
covers every world in the corpus. `prep.py` fails rather than truncating, because
a truncated input silently loses its constants section, which programs reference
by symbol.

## Where this sits

This is covenant-agent's R3, the tiny-planner track in `PLAN.md` section 5,
started here with one 7.9M model rather than the full size curve. Results and
the pass bar are in `results/R3.md`. It began life as the phase 1 experiment of
`abc-diffusion-window` (the prose decoder design), and moved here on 2026-09-17
once it became clear that what it was measuring, whether a small model can bind
per-request symbols to their declaration lines, is this repo's question.

**One thing to know before reading any number from it.** In `s5_plain`, the
typed signature on a tool line (everything before `::`) identifies the called
tool uniquely in every reference program. The description is never needed to
choose a tool. So this experiment measures whether a model can read the request
and bind a symbol to its signature line, not whether it can match English to
English. `diagnose.py` reports that, and the effect-class agreement between a
generated program and its reference, for any generated file.
