//===- tmk_u_q4_it.cc — 4-bit unpack to int8, iterator form ---===//
//
// tmk_u_q4.cc's counterpart under the addressing that removes the false
// loop-carried dependence (see tmk_u_t2_it.cc). Here so the two formats are
// compared at each one's best known schedule rather than at whichever one the
// compiler happened to handle better.

#include "tmk.h"

extern "C" void tmk_u_q4_it(const uint8_t *__restrict p, int8_t *__restrict scr) {
  event0();
  constexpr unsigned kIters = TMK_BYTES_Q4 / TMK_VB;   // 64
  constexpr unsigned kStream = TMK_W / 2;              // 2048 weights per stream

  auto in = aie::begin_vector<TMK_VB>(p);
  auto s0 = aie::begin_vector<TMK_VB>((uint8_t *)scr + 0 * kStream);
  auto s1 = aie::begin_vector<TMK_VB>((uint8_t *)scr + 1 * kStream);

#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kIters; ++i) {
    const aie::vector<uint8_t, TMK_VB> q = *in++;
    *s0++ = aie::bit_and((uint8_t)0x0F, q);
    *s1++ = aie::bit_and((uint8_t)0xF0, q);
  }
  event1();
}
