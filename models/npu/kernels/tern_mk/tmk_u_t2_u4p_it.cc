//===- tmk_u_t2_u4p_it.cc — 2-bit to nibble expand, PAIRED, no shifts ---===//
//
// tmk_u_t2_u4_it.cc costs 0.125 cycles per weight, 8x its own store floor,
// because it insists that a nibble hold the same weight the source byte held at
// bit positions 4 and 6: those carry factors of 16 and 64, a nibble cannot hold
// either, so they need real shifts and the loop turns op-bound (ResMII 16
// against 1 store). tern_mk's README concluded that the mask-only trick "runs
// out at the nibble target."
//
// It does not. The trick was applied to the wrong axis. Masking IN PLACE with
// 0x33 already produces a byte whose two nibbles are two valid weights:
//
//   b       = c3 c2 c1 c0        four 2-bit codes, c0 at bits [1:0]
//   b&0x33  = low nibble b[1:0] = c0,  high nibble b[5:4] = c2     factor 1
//   b&0xCC  = low nibble b[3:2] = c1 << 2, high nibble b[7:6] = c3 << 2
//
// The MAC reads each nibble as its own uint4 operand, so the high nibble's
// factor of 16 never appears -- it is a byte offset, not a value. Both nibbles
// of the 0x33 byte are literal codes; both nibbles of the 0xCC byte carry the
// same factor of 4. One mask per output byte, no shift, no interleave, no
// select: one load, two masks, two stores per 32 packed bytes.
//
// WHERE THE FACTOR OF 4 GOES, and why it is free. mmul's B operand is 16k x
// 16n with consecutive elements at consecutive n, so the two nibbles of a byte
// are two adjacent OUTPUT ROWS at one k. The factor of 4 therefore applies per
// output row, and decision 3 already carries one scale per output row: the rows
// fed from 0xCC bytes get that scale divided by four. No extra accumulator, no
// extra op, no correction term. The packer chooses which four weights share a
// source byte, which is what makes this a layout decision rather than a trick:
//
//   b[1:0] = w[n0    ][k]      b[5:4] = w[n0 + 1][k]     scale s
//   b[3:2] = w[n1    ][k]      b[7:6] = w[n1 + 1][k]     scale s / 4
//
// tmk_pack.py is the reference packer and checks that round trip on real
// ternary matrices. The claim this file makes is only about cycles.

#include "tmk.h"

extern "C" void tmk_u_t2_u4p_it(const uint8_t *__restrict p,
                                uint8_t *__restrict scr) {
  event0();
  constexpr unsigned kIters = TMK_BYTES_T2 / TMK_VB;   // 32
  constexpr unsigned kHalf = TMK_W / 4;                // 1024 weights per half

  auto in = aie::begin_vector<TMK_VB>(p);
  // Two output runs, one per factor. Each holds 32 bytes per iteration and each
  // byte is two weights, so the pair covers all 128 weights of the input.
  auto lo = aie::begin_vector<TMK_VB>(scr + 0);
  auto hi = aie::begin_vector<TMK_VB>(scr + kHalf / 2);

#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kIters; ++i) {
    const aie::vector<uint8_t, TMK_VB> q = *in++;
    *lo++ = aie::bit_and((uint8_t)0x33, q);   // codes at bits [1:0] and [5:4]
    *hi++ = aie::bit_and((uint8_t)0xCC, q);   // codes at bits [3:2] and [7:6], x4
  }
  event1();
}
