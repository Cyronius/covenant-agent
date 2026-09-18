//===- tmk_u_q4.cc — unpack 4-bit weights to int8 scratch, masks only ---===//
//
// One 64x64 weight block: 2048 packed bytes -> 4096 int8.
// 64 iterations, each: one 32-byte load, two masks, two 32-byte stores.
// The high-nibble stream carries a factor of 16, folded into its scale —
// gemm_q4_dequant.h's own trick, kept so the two formats are compared on the
// same terms.

#include "tmk.h"

extern "C" void tmk_u_q4(const uint8_t *__restrict p, int8_t *__restrict scr) {
  event0();
  constexpr unsigned kIters = TMK_BYTES_Q4 / TMK_VB;          // 64
  constexpr unsigned kStream = TMK_W / 2;                     // 2048 weights per stream

#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kIters; ++i) {
    const aie::vector<uint8_t, TMK_VB> q = aie::load_v<TMK_VB>(p + i * TMK_VB);
    aie::store_v((uint8_t *)scr + 0 * kStream + i * TMK_VB, aie::bit_and((uint8_t)0x0F, q));
    aie::store_v((uint8_t *)scr + 1 * kStream + i * TMK_VB, aie::bit_and((uint8_t)0xF0, q));
  }
  event1();
}
