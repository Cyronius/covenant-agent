//===- tmk_u_t2.cc — unpack 2-bit weights to int8 scratch, masks only ---===//
//
// One 64x64 weight block: 1024 packed bytes -> 4096 int8.
// 32 iterations, each: one 32-byte load, four masks, four 32-byte stores.
// The four streams carry factors 1, 4, 16, 64; each folds into its stream's
// scale at the accumulator, so nothing is shifted here (see tmk.h).

#include "tmk.h"

extern "C" void tmk_u_t2(const uint8_t *__restrict p, int8_t *__restrict scr) {
  event0();
  constexpr unsigned kIters = TMK_BYTES_T2 / TMK_VB;          // 32
  constexpr unsigned kStream = TMK_W / 4;                     // 1024 weights per stream

#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kIters; ++i) {
    const aie::vector<uint8_t, TMK_VB> q = aie::load_v<TMK_VB>(p + i * TMK_VB);
    aie::store_v((uint8_t *)scr + 0 * kStream + i * TMK_VB, aie::bit_and((uint8_t)0x03, q));
    aie::store_v((uint8_t *)scr + 1 * kStream + i * TMK_VB, aie::bit_and((uint8_t)0x0C, q));
    aie::store_v((uint8_t *)scr + 2 * kStream + i * TMK_VB, aie::bit_and((uint8_t)0x30, q));
    aie::store_v((uint8_t *)scr + 3 * kStream + i * TMK_VB, aie::bit_and((uint8_t)0xC0, q));
  }
  event1();
}
