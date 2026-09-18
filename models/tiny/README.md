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

## Running it

```bash
python prep.py --limit 20000                    # tokenize once, write tensors
python test_pipeline.py --cache data_cache      # references must score 100%
python train.py --arm diffusion --epochs 5
python train.py --arm ar        --epochs 5
python evaluate.py --ckpt runs/diffusion_s0/best.pt --split test \
                   --steps 8 --repair-rounds 2 --gen-out runs/diff_test.jsonl
python probe.py --gen runs/diff_test.jsonl --compiled-only
```

`test_pipeline.py` is the one to run first and after any change to the data path.
It takes reference programs out of the tensor cache, decodes them back to text,
compiles them and executes them. If that is not 100%, no model number below it
means anything.

## Splitting the work across machines

The sandbox needs Node; a GPU pod does not. So:

```bash
# laptop
python prep.py --limit 20000
# pod: needs torch, the data_cache directory, and pure-Python imports only
python train.py --arm diffusion --epochs 5
python evaluate.py --ckpt runs/diffusion_s0/best.pt --generate --gen-out g.jsonl
# laptop
python evaluate.py --score --gen-out g.jsonl --split test
```

On this CPU a full-config step is about 8.7 seconds at batch 16. Smoke-test
locally with `--limit-train 128 --d 128 --enc-layers 2 --dec-layers 2`, then run
seeds and sweeps on a pod.

Watch a long run through `runs/<name>/log.jsonl`, which the trainer writes and
flushes at every evaluation. Do not watch it by piping the console output through
`tail` or `head`: those buffer until the process exits, so the log looks empty
for hours and a run that has died looks identical to one that is working.
Redirect to a file instead, or read the JSONL.

## Files

| file | what |
|---|---|
| `corpus.py` | load tasks, derive the header, program text to tokens and back |
| `tok.py` | byte-pair tokenizer for the input, exact symbol table for the output |
| `prep.py` | tokenize once, write tensors, refuse to truncate silently |
| `model.py` | the shared encoder-decoder and the masked-diffusion training objective |
| `train.py` | both arms, one flag |
| `sample.py` | cosine-schedule unmasking, and the compiler repair loop |
| `evaluate.py` | generate programs, then score them in the sandbox |
| `probe.py` | when the sampler decides each kind of token |
| `sandbox.py` | register the 104 generated theme worlds so the sandbox can run them |
| `test_pipeline.py` | the reference round trip, the check everything else rests on |

Not named `data.py` on purpose: that shadows covenant-agent's `data` package and
breaks every import of the theme generator.

## Memory, and why a too-large batch looks like a slow model

The encoder reads about 1100 input tokens, and self-attention over them holds an
`S x S` score matrix per head that autograd keeps for the backward pass:

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
