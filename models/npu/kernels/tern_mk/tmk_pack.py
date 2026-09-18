"""The packer the paired nibble expand implies, and proof that it round-trips.

`tmk_u_t2_u4p_it.cc` gets 2-bit weights into the native int4 MAC path for two
mask operations and no shifts, which is eight times cheaper than the expand it
replaces. The kernel is eight lines; the layout is the real work, and a wrong
layout would compile, schedule at the same two bundles, and quietly compute
something that is not the matrix product. So the layout lives here, in a
reference packer, with the arithmetic checked end to end against numpy on real
ternary matrices.

WHAT THE KERNEL DOES, as algebra. A packed byte holds four 2-bit codes. Masking
it in place with 0x33 leaves a byte whose two nibbles are codes c0 and c2
verbatim; masking with 0xCC leaves one whose two nibbles are c1 and c3 each
multiplied by four, because bits [3:2] and [7:6] sit two places above their
nibble's base. The MAC reads each nibble as its own uint4 operand, so the high
nibble's factor of 16 never enters the arithmetic -- it is a byte offset, not a
value.

WHAT THAT COSTS, and why it is free. mmul's B operand is 16k by 16n with
consecutive elements at consecutive n, so the two nibbles of a byte are two
adjacent output rows at one k. The factor of four is therefore per output row,
and decision 3 already carries one scale per output row: rows fed from 0xCC
bytes take that scale divided by four. So the packer's only job is to put the
right four weights in each source byte:

    b[1:0] = w[n0    ][k]      b[5:4] = w[n0 + 1][k]      scale s
    b[3:2] = w[n1    ][k]      b[7:6] = w[n1 + 1][k]      scale s / 4

with (n0, n0+1) an even-indexed row pair and (n1, n1+1) the pair 32 rows up, so
that a 64-row block splits into two halves of 32 and each half is contiguous in
the scratch the kernel writes.

TERNARY IS UNSIGNED HERE. The codes are {0, 1, 2} for weights {-1, 0, +1}, with
a zero point of 1, because the factor-of-four rows would read as negative under
a signed nibble (12 is -4 in int4) and the fold would be wrong. The zero point
costs one term per output row against the activation sum, which the asymmetric
path pays anyway.

    python tmk_pack.py
"""
from __future__ import annotations

import numpy as np

N, K = 64, 64              # one weight block, as tmk.h defines it
HALF = N // 2              # rows 0..31 at factor 1, rows 32..63 at factor 4
ZERO_POINT = 1             # code 1 means weight 0


def masks_are_what_they_claim() -> None:
    """The bit claim, over every byte value there is rather than an example."""
    b = np.arange(256, dtype=np.uint8)
    c0, c1, c2, c3 = (b & 0x03), (b >> 2) & 0x03, (b >> 4) & 0x03, (b >> 6) & 0x03
    lo_byte = b & 0x33
    hi_byte = b & 0xCC
    assert np.array_equal(lo_byte & 0x0F, c0)
    assert np.array_equal(lo_byte >> 4, c2)
    assert np.array_equal(hi_byte & 0x0F, c1 * 4)
    assert np.array_equal(hi_byte >> 4, c3 * 4)
    print("1. masks: 0x33 gives (c0, c2) at factor 1, 0xCC gives (c1, c3) at "
          "factor 4, for all 256 bytes")


def pack(codes: np.ndarray) -> np.ndarray:
    """(N, K) uint8 codes in {0,1,2} -> (N*K/4,) packed bytes.

    Row pairs (2m, 2m+1) of the low half share a byte's bits [1:0] and [5:4];
    the matching pair of the high half takes bits [3:2] and [7:6]. So one byte
    carries four weights at one k: two rows that will be scaled by s and two
    that will be scaled by s/4.
    """
    assert codes.shape == (N, K) and codes.max() <= 3
    out = np.zeros((HALF // 2, K), dtype=np.uint8)
    for m in range(HALF // 2):                       # 16 row-pairs per half
        out[m] = (codes[2 * m] & 3)
        out[m] |= (codes[HALF + 2 * m] & 3) << 2
        out[m] |= (codes[2 * m + 1] & 3) << 4
        out[m] |= (codes[HALF + 2 * m + 1] & 3) << 6
    return out.reshape(-1)


def expand(packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """What the kernel writes: two runs of bytes, each byte two uint4 weights.

    This is `*lo++ = 0x33 & q` and `*hi++ = 0xCC & q`, nothing else, so the two
    lines here and the two lines there have to stay the same two lines.
    """
    return packed & 0x33, packed & 0xCC


def nibbles(stream: np.ndarray) -> np.ndarray:
    """How the MAC reads a byte run: element 2i is the low nibble of byte i."""
    out = np.empty(stream.size * 2, dtype=np.int32)
    out[0::2] = stream & 0x0F
    out[1::2] = stream >> 4
    return out


def dot_from_packed(packed: np.ndarray, act: np.ndarray) -> np.ndarray:
    """The whole path: packed bytes plus int8 activations -> (N,) int32 products.

    Every step here is something the kernel or its scale table does. The int32
    accumulate is the MAC; the divide by four is the per-row scale on the high
    half; the activation sum is the zero point.
    """
    lo, hi = expand(packed)
    acc = np.zeros(N, dtype=np.int64)
    for stream, rows, factor in ((lo, range(0, HALF), 1), (hi, range(HALF, N), 4)):
        vals = nibbles(stream).reshape(HALF // 2, K, 2)   # (row-pair, k, which row)
        for m in range(HALF // 2):
            for which in (0, 1):
                row = rows[2 * m + which]
                # The MAC's int32 accumulate over k, then the row's scale.
                raw = int(np.dot(vals[m, :, which], act.astype(np.int64)))
                assert raw % factor == 0, "the factor-4 fold is not exact"
                acc[row] = raw // factor - ZERO_POINT * int(act.sum())
    return acc


def round_trip(seed: int) -> None:
    rng = np.random.default_rng(seed)
    w = rng.integers(-1, 2, size=(N, K)).astype(np.int64)          # ternary
    act = rng.integers(-128, 128, size=K).astype(np.int64)         # int8
    codes = (w + ZERO_POINT).astype(np.uint8)
    got = dot_from_packed(pack(codes), act)
    want = w @ act
    assert np.array_equal(got, want), (got[:4], want[:4])


def pack_k_pairs(codes: np.ndarray) -> np.ndarray:
    """The same packing under the other possible B element order.

    The per-row scale fold rests on consecutive B elements being consecutive n.
    The evidence for that is `mmul_8_4.hpp`, whose aie2p path splits B into
    `extract<128>(0)` and `extract<128>(1)` and feeds them as two 8k x 16n
    halves, which is k-outer ordering. It is an inference from the split rather
    than a documented layout, so the other order is modelled here too: if
    consecutive elements turn out to be consecutive k, a byte's two nibbles are
    two k positions of ONE output row, the factor of four no longer folds into
    that row's scale, and it needs its own accumulator instead.

    That is the whole difference. One accumulator out of the four the part keeps
    without spilling, no extra operation per weight, and the expand still costs
    two masks and two stores. Both orders are exact, so the cycle result does
    not depend on which one the hardware has.
    """
    assert codes.shape == (N, K) and codes.max() <= 3
    out = np.zeros((N, K // 4), dtype=np.uint8)
    for j in range(K // 4):
        out[:, j] = (codes[:, 4 * j] & 3)                  # k = 4j,   factor 1
        out[:, j] |= (codes[:, 4 * j + 1] & 3) << 2        # k = 4j+1, factor 4
        out[:, j] |= (codes[:, 4 * j + 2] & 3) << 4        # k = 4j+2, factor 1
        out[:, j] |= (codes[:, 4 * j + 3] & 3) << 6        # k = 4j+3, factor 4
    return out


def dot_two_acc(packed: np.ndarray, act: np.ndarray) -> np.ndarray:
    """Two accumulators, one per factor, combined once per output row."""
    lo, hi = expand(packed)
    acc = np.zeros(N, dtype=np.int64)
    for row in range(N):
        a1 = nibbles(lo[row])            # k = 4j and 4j+2, literal codes
        a4 = nibbles(hi[row])            # k = 4j+1 and 4j+3, codes x 4
        k1 = np.concatenate([act[0::4, None], act[2::4, None]], axis=1).reshape(-1)
        k4 = np.concatenate([act[1::4, None], act[3::4, None]], axis=1).reshape(-1)
        raw4 = int(np.dot(a4, k4))
        assert raw4 % 4 == 0, "the factor-4 fold is not exact"
        acc[row] = int(np.dot(a1, k1)) + raw4 // 4 - ZERO_POINT * int(act.sum())
    return acc


def round_trip_k_order(seed: int) -> None:
    rng = np.random.default_rng(seed)
    w = rng.integers(-1, 2, size=(N, K)).astype(np.int64)
    act = rng.integers(-128, 128, size=K).astype(np.int64)
    codes = (w + ZERO_POINT).astype(np.uint8)
    got = dot_two_acc(pack_k_pairs(codes), act)
    assert np.array_equal(got, w @ act), (got[:4], (w @ act)[:4])


def old_layout_still_holds() -> None:
    """The four-stream int8 expand, for the record: the same in-place masks with
    factors 1, 4, 16, 64, each folded into its stream's scale. That layout is
    what `tmk_u_t2_it.cc` writes, it costs twice as much per weight, and it is
    the fallback if a future MAC shape wants int8 weights after all."""
    b = np.arange(256, dtype=np.uint8)
    for shift, factor in ((0, 1), (2, 4), (4, 16), (6, 64)):
        stream = b & (0x03 << shift)
        assert np.array_equal(stream, ((b >> shift) & 0x03) * factor)
    print("6. the int8-target layout is unchanged: four in-place masks, factors "
          "1, 4, 16, 64")


if __name__ == "__main__":
    masks_are_what_they_claim()
    for seed in range(8):
        round_trip(seed)
    print("2. pack -> mask -> nibble -> int32 MAC -> per-row scale reproduces "
          "W @ a exactly, 8 random ternary blocks")
    # The corners the random matrices are unlikely to hit.
    rng = np.random.default_rng(0)
    act = rng.integers(-128, 128, size=K).astype(np.int64)
    for name, w in (("all -1", -np.ones((N, K), dtype=np.int64)),
                    ("all +1", np.ones((N, K), dtype=np.int64)),
                    ("all 0", np.zeros((N, K), dtype=np.int64))):
        got = dot_from_packed(pack((w + ZERO_POINT).astype(np.uint8)), act)
        assert np.array_equal(got, w @ act), name
    print("3. and at the corners: all -1, all +1, all 0")
    # The store count the cycle number rests on.
    codes = rng.integers(0, 3, size=(N, K)).astype(np.uint8)
    packed = pack(codes)
    lo, hi = expand(packed)
    assert packed.size == N * K // 4 == 1024
    assert lo.size + hi.size == N * K // 2 == 2048
    print(f"4. {packed.size} packed bytes -> {lo.size + hi.size} scratch bytes "
          f"for {N * K} weights: half of what the int8 target stores")
    for seed in range(4):
        round_trip_k_order(seed)
    print("5. and under the other B element order, with two accumulators "
          "instead of a per-row scale: also exact")
    old_layout_still_holds()
    print("ALL PASS")
