# tern_mk: what a weight format costs on this NPU, in cycles

Measured 2026-09-18 at `a09fc2a`, Peano/mlir-aie 1.4.2, `--target=aie2p`.

The first piece of step 5 of [`npu-native-planner.md`](../../../../.claude/plans/npu-native-planner.md),
cut down to the question decision 3 rests on: what does unpacking a packed
weight cost, and how does it compare to the multiplies it feeds? Decision 3
answers it by argument; this answers it with the compiler.

These are compile-time schedules, not hardware timings. `kernel_remarks.py`
compiles one translation unit and reports what Peano's post-pipeliner made of
each loop: bundles per loop body, which is cycles per iteration, and the II
floor any schedule has to respect. ResMII is the part of that floor set by
whichever VLIW slot the body uses most; RecMII is the part set by the longest
dependence chain carried around the loop. No xclbin is built and nothing runs
on the NPU, so every number here is what the compiler believes, and hardware
can only be slower. It is enough to compare formats.

## The answer

Per weight, at M canvas slots filled in parallel:

| format | software unpack | MAC | total at M=8 | at M=16 | at M=64 |
|---|---|---|---|---|---|
| int8, 4M resident | none | 0.00195·M | 0.016 | 0.031 | 0.125 |
| 4-bit, 8M resident | none, the hardware does it | 0.00195·M | 0.016 | 0.031 | 0.125 |
| ternary to int8 scratch, 16M resident | 0.031 | 0.00195·M | 0.047 | 0.062 | 0.156 |
| ternary to nibble scratch, 16M resident | 0.125 | 0.00195·M | 0.141 | 0.156 | 0.250 |

Cycles per weight. The MAC term is per weight *per slot*: a weight applied to
64 slots does 64 MACs.

Three things follow, and two of them contradict decision 3.

### The MAC costs the same whatever the format, and it is already at peak

Both `mmul<8,8,8,int8,int8>` and `mmul<4,16,16,int8,uint4>` schedule at 0.00195
cycles per weight per slot, which is 512 MACs per cycle, the part's int8 rate.
The int4 mode is not a wider mode: `mmul<4,16,16>` issues two `mac_4x8_8x16`
intrinsics of 512 MACs each. So the format neither buys nor costs multiply
throughput. It only decides what has to happen before the multiply.

### Ternary and 4-bit pay the same software unpack, and 4-bit does not have to pay it

Both unpack at 0.031 cycles per weight, exactly as decision 3 argues. But aie2p
instantiates `mmul<M,K,N,int8,uint4>` (`aie_api/detail/aie2p/mmul_8_4.hpp`,
shape `<4,16,16>`), whose B operand is packed nibbles straight from memory with
the unpack folded into the MAC (`mac_4x8_8x16_conf(..., b.unpack_sign(b_sign),
...)`). There is no 2-bit vector type and no 2-bit mmul shape, so ternary has
no such path.

Decision 3's "ternary strictly dominates 4-bit" does not hold on this silicon.
What ternary actually offers is a trade: twice the resident parameters for a
flat 0.031 cycles per weight, which is 25% overhead at the canvas's 64 slots,
100% at 16 slots, and 200% at 8.

### int8 at 4M is dominated, not a fallback

It costs the same cycles per weight per slot as 4-bit and holds half the
resident parameters, and its one claimed advantage, escaping the unpack, is
something 4-bit also has. Decision 3 keeps int8 at 4M as the fallback if
ternary fails. The fallback should be 4-bit at 8M.

### Break-even is 16 slots, not 8

Decision 3 says "at M = 8 the unpack falls to the same order as the
multiplies." At M=8 the unpack is 2x the multiplies, and 0.031/0.00195 puts
parity at M=16. The design survives this, because its canvas is 64 slots where
the unpack is a 25% tax, but it survives for a different reason than the one
recorded, and an M=8 design would have been unpack-bound.

## Why the unpack costs what it does

It is store-bound rather than mask-bound. Both unpack loops sit at a ResMII
equal to their store count, and both write one byte per weight through a
32-byte store port:

| loop | bundles/iter | weights/iter | stores/iter | cycles/weight |
|---|---|---|---|---|
| `tmk_u_t2_it` (2-bit to int8) | 4 | 128 | 4 | 0.031 |
| `tmk_u_q4_it` (4-bit to int8) | 2 | 64 | 2 | 0.031 |
| `tmk_u_t2_u4_it` (2-bit to nibble) | 16 | 128 | 1 | 0.125 |

32 bytes of scratch per cycle accounts for the first two rows, and it is why
the two formats tie exactly: the mask count differs, the byte count does not.
The masking hides under the stores.

The nibble target writes half the bytes and should therefore cost half as much,
0.0156. It measures 8x worse because 2-bit codes at bit positions 4 and 6
cannot stay in place: a nibble cannot hold their factors of 16 and 64, so they
need real shifts, and the loop turns op-bound (ResMII 16 against 1 store). The
mask-only trick that makes the int8 target free, leaving each field where it
lies and folding its power-of-two factor into the scale the way
`gemm_q4_dequant.h` does for the high nibble, runs out at the nibble target. A
better nibble expand would halve ternary's overhead, to 12% at 64 slots, and it
is worth attempting before ternary is judged: the format still has a 2x store
advantage that this implementation gives back.

## Two hardware facts found on the way

Dense int8 tops out at 512 MACs per issue. `mmul_8_8.hpp` declares `<8,16,8>`,
which looks like a 1024-MAC shape, but its B operand is
`sparse_vector<int8,128>`: it is the structured-sparsity mode, not a dense one.
Every dense int8 path on this part is 512 MACs per cycle. Structured sparsity
is a separate lever this project has not costed.

Register blocking caps at about 4 accumulators. Bundles per MAC issue against
live 8x8 int32 accumulators, indexed addressing:

| accumulators (MB×NB) | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| cycles per MAC issue | 2.0 | 1.5 | 1.75 | 2.75 to 2.9 | spills (ResMII 56) |

Four is the best of these, eight already costs 60%, and sixteen spills. Any
block shape chosen later (decision 14 leaves it open) has to tile within four.

## The addressing trap, worth 4x and silent

Every loop here was first written with an index multiply in the body
(`load_v(p + i * 32)`). All of them reported a RecMII far above their ResMII
and got no software pipeline: the compiler reads the recomputed address as a
loop-carried dependence. Rewriting with one post-increment per stream, using
plain pointers or `aie::begin_vector` where the type allows it, removes it:

| loop | indexed | incrementing |
|---|---|---|
| 2-bit unpack | 17 bundles (RecMII 14) | 4 (RecMII 1, at its MII floor) |
| 4-bit unpack | 12 | 2 |
| int8 MAC, 2×2 | 7 | 4 |
| int4 MAC, 2×2 | 14 | 9 |

Nothing warns about this. The kernel is correct either way and 4x slower. It
belongs with the trap catalogue in openflowlm-next's `open_kernels/README.md`.
`aie::vector_iterator` has no default constructor, so an array of them cannot
be declared and filled; use plain pointers when the count is a template
parameter.

## Files

| | |
|---|---|
| `tmk.h` | shapes, and the mask-only unpack argument in full |
| `tmk_u_t2.cc`, `tmk_u_t2_it.cc` | 2-bit to int8 scratch, indexed and incrementing |
| `tmk_u_q4.cc`, `tmk_u_q4_it.cc` | 4-bit to int8 scratch, both forms |
| `tmk_u_t2_u4.cc`, `tmk_u_t2_u4_it.cc` | 2-bit to nibble scratch, both forms |
| `tmk_mac_i8.cc`, `tmk_mac_i8_it.cc` | `mmul<8,8,8,int8,int8>`, MB×NB blocked |
| `tmk_mac_u4k.cc`, `tmk_mac_u4k_it.cc` | `mmul<4,16,16,int8,uint4>`, packed B |
| `tmk_mac_i8w.cc` | `mmul<8,16,8>`, the sparse shape; does not compile dense |
| `tmk_mac_u4_41616.cc` | the one-issue probe that found the native int4 shape |
| `remarks*.sh` | the sweeps |

## Running it

WSL, the ironenv venv, and openflowlm-next's `kernel_remarks.py` imported
read-only, so nothing is written into that repo:

```
wsl -d Ubuntu-24.04 -- bash /mnt/c/code/covenant-agent/models/npu/kernels/tern_mk/remarks.sh
```

`remarks.sh` sweeps the unpacks and the int8 blocking, `remarks2.sh` the wide
shapes, `remarks3.sh` the MAC loops under incrementing addressing. Each prints
per-loop bundle counts and II bounds; divide by the weights per iteration in
the tables above to get cycles per weight.

## What this does not settle

The capacity question is untouched. Ternary buys 2x the resident parameters for
25% more time per weight at 64 slots; whether 16M ternary weights beat 8M
4-bit weights at the task is bet 1, a training result rather than a kernel
result. What these numbers do say is that the speed argument decision 3 makes
for ternary is not available. The choice is capacity against a 25% tax, and it
should be made on the step-2 and step-3 runs.
