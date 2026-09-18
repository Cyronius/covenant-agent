# tern_mk: what a weight format costs on this NPU, in cycles

Measured 2026-09-18 at `a09fc2a`, Peano/mlir-aie 1.4.2, `--target=aie2p`.
Amended the same day at `b98a522`: a better 2-bit to nibble expand
(`tmk_u_t2_u4p_it.cc`) is 8x cheaper than the one first measured, which halves
ternary's overhead and restores decision 3's break-even at 8 slots. The two
sections it contradicts are marked and corrected in place rather than deleted,
because what the first version got wrong is the more useful half of the lesson:
the mask-only trick had not run out, it had been applied to the wrong axis.

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
| **ternary to paired nibbles, 16M resident** | **0.0156** | 0.00195·M | **0.031** | **0.047** | **0.141** |
| ternary to int8 scratch, 16M resident | 0.031 | 0.00195·M | 0.047 | 0.062 | 0.156 |
| ternary to nibble scratch, first attempt | 0.125 | 0.00195·M | 0.141 | 0.156 | 0.250 |

Cycles per weight. The MAC term is per weight *per slot*: a weight applied to
64 slots does 64 MACs.

The paired row is the one to design against. It is the cheapest ternary path at
every M, it is the only one that reaches the store floor its byte count implies,
and it feeds the native int4 MAC rather than an int8 scratch buffer. Ternary's
cost against 4-bit is then 12.5% at the canvas's 64 slots and exactly 2x at 8.

Three things follow, and two of them contradict decision 3.

### The MAC costs the same whatever the format, and it is already at peak

Both `mmul<8,8,8,int8,int8>` and `mmul<4,16,16,int8,uint4>` schedule at 0.00195
cycles per weight per slot, which is 512 MACs per cycle, the part's int8 rate.
The int4 mode is not a wider mode: `mmul<4,16,16>` issues two `mac_4x8_8x16`
intrinsics of 512 MACs each. So the format neither buys nor costs multiply
throughput. It only decides what has to happen before the multiply.

### 4-bit pays no software unpack, and ternary pays half of what the arithmetic assumed

aie2p instantiates `mmul<M,K,N,int8,uint4>` (`aie_api/detail/aie2p/mmul_8_4.hpp`,
shape `<4,16,16>`), whose B operand is packed nibbles straight from memory with
the unpack folded into the MAC (`mac_4x8_8x16_conf(..., b.unpack_sign(b_sign),
...)`). There is no 2-bit vector type and no 2-bit mmul shape, so ternary has no
such path and must produce nibbles itself.

**Corrected 2026-09-18.** This section first read "both unpack at 0.031 cycles
per weight, exactly as decision 3 argues." Ternary does not: the paired expand
gets 2-bit codes into nibbles for 0.0156, half the cost of getting them into
int8 scratch, because it writes half the bytes and nothing else changes. So
ternary's cost per weight is *lower* than 4-bit's would be if 4-bit needed
software at all.

Decision 3's "ternary strictly dominates 4-bit" still does not hold, because
4-bit's unpack is free in hardware and ternary's is not zero. What ternary
offers is a trade: twice the resident parameters for a flat 0.0156 cycles per
weight, which is 12.5% overhead at the canvas's 64 slots, 50% at 16 slots and
100% at 8.

### int8 at 4M is dominated, not a fallback

It costs the same cycles per weight per slot as 4-bit and holds half the
resident parameters, and its one claimed advantage, escaping the unpack, is
something 4-bit also has. Decision 3 keeps int8 at 4M as the fallback if
ternary fails. The fallback should be 4-bit at 8M.

### Break-even is 8 slots after all

Decision 3 says "at M = 8 the unpack falls to the same order as the multiplies."
The first measurement put parity at M=16 and this section said so. With the
paired expand, 0.0156/0.00195 puts parity at exactly M=8, which is decision 3's
number. The design's 64-slot canvas was never in danger either way; what changed
is that an M=8 design is now viable rather than unpack-bound, and the M=8 figure
in the plan's cost model needs no footnote.

## Why the unpack costs what it does

It is store-bound rather than mask-bound. Every unpack loop that pipelines sits
at a ResMII equal to its store count, through a 32-byte store port:

| loop | bundles/iter | weights/iter | stores/iter | cycles/weight |
|---|---|---|---|---|
| **`tmk_u_t2_u4p_it`** (2-bit to paired nibbles) | **2** | 128 | 2 | **0.0156** |
| `tmk_u_t2_it` (2-bit to int8) | 4 | 128 | 4 | 0.031 |
| `tmk_u_q4_it` (4-bit to int8) | 2 | 64 | 2 | 0.031 |
| `tmk_u_t2_u4_it` (2-bit to nibble, first attempt) | 16 | 128 | 1 | 0.125 |

32 bytes of scratch per cycle accounts for every row but the last, and it is why
2-bit-to-int8 and 4-bit-to-int8 tie exactly: the mask count differs, the byte
count does not. The masking hides under the stores. The paired expand is the same
story at half the bytes, so it lands at half the cost, and its ResMII of 2 with
RecMII 1 says it is at its own store floor with nothing left to win.

**What the first attempt got wrong, which is the part worth keeping.** It writes
half the bytes and should have cost 0.0156; it measured 8x worse, ResMII 16
against 1 store, because it insisted that a nibble hold the same weight the
source byte held at bit positions 4 and 6 — factors of 16 and 64, which a nibble
cannot hold, so they needed real shifts and the loop went op-bound. This README
concluded from that that the mask-only trick "runs out at the nibble target."

It does not. It had been applied to the wrong axis. Masking a packed byte in
place with `0x33` already yields a byte whose two nibbles are two valid weights,
codes `c0` and `c2` verbatim; `0xCC` yields one holding `c1` and `c3` each
multiplied by four. The factor of 16 on the high nibble never appears in the
arithmetic, because the MAC reads each nibble as its own operand: 16 is a byte
offset, not a value. One mask per output byte, no shift, no interleave, no
select.

The factor of four costs nothing either. `mmul`'s B operand is 16k by 16n with
consecutive elements at consecutive n, so the two nibbles of a byte are two
adjacent output rows at one k, and decision 3 already carries one scale per
output row: the rows fed from `0xCC` bytes take that scale divided by four. If
the element order turns out to be the other way — the evidence is the
`extract<128>` k-half split in `mmul_8_4.hpp`, which is an inference and not a
documented layout — the correction becomes one extra accumulator out of the four
the part keeps without spilling, and still no operation per weight. `tmk_pack.py`
is the reference packer and checks both orders against numpy on real ternary
matrices, including the zero point, because a wrong layout would compile,
schedule at the same two bundles, and quietly compute something that is not the
matrix product.

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
| `tmk_u_t2_u4.cc`, `tmk_u_t2_u4_it.cc` | 2-bit to nibble scratch, the first attempt, both forms |
| `tmk_u_t2_u4p_it.cc` | 2-bit to paired nibbles: two masks, no shifts, the store floor |
| `tmk_pack.py` | the packer that layout implies, checked against numpy under both possible B element orders |
| `tmk_mac_i8.cc`, `tmk_mac_i8_it.cc` | `mmul<8,8,8,int8,int8>`, MB×NB blocked |
| `tmk_mac_u4k.cc`, `tmk_mac_u4k_it.cc` | `mmul<4,16,16,int8,uint4>`, packed B |
| `tmk_mac_i8w.cc` | `mmul<8,16,8>`, the sparse shape; does not compile dense |
| `tmk_mac_u4_41616.cc` | the one-issue probe that found the native int4 shape |
| `remarks*.sh` | the sweeps; `remarks4.sh` is the paired expand against the three it replaces |

## Running it

WSL, the ironenv venv, and openflowlm-next's `kernel_remarks.py` imported
read-only, so nothing is written into that repo:

```
wsl -d Ubuntu-24.04 -- bash /mnt/c/code/covenant-agent/models/npu/kernels/tern_mk/remarks.sh
```

`remarks.sh` sweeps the unpacks and the int8 blocking, `remarks2.sh` the wide
shapes, `remarks3.sh` the MAC loops under incrementing addressing, `remarks4.sh`
the paired nibble expand against the three unpacks it replaces. The layout check
needs no toolchain: `python tmk_pack.py` on any machine with numpy. Each prints
per-loop bundle counts and II bounds; divide by the weights per iteration in
the tables above to get cycles per weight.

## What this does not settle

The capacity question is untouched. Ternary buys 2x the resident parameters for
12.5% more time per weight at 64 slots; whether 16M ternary weights beat 8M
4-bit weights at the task is bet 1, a training result rather than a kernel
result. `models/tiny/run_step3.sh` is that measurement, and it compares the
formats at matched resident bytes for this reason: the kernel side has now said
everything it can say, and it says the tax is small enough that capacity decides.

Two things here are compile-time only and stay that way until an xclbin runs.
The B element order is inferred from a header rather than observed, and the
paired expand's correctness is checked in Python against a model of the MAC
rather than against the MAC. Both are cheap to settle at step 5 and neither
changes a cycle count.
