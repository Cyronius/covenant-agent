//===- tmk_u_t2_it.cc — the same 2-bit unpack, written with iterators ---===//
//
// tmk_u_t2.cc schedules at 17 bundles against a ResMII of 6 and gets no
// software pipeline: RecMII is 14, so the compiler believes each iteration
// depends on the last. There is no such dependence — p and scr are __restrict
// and disjoint — so the suspect is the addressing, where a load and four
// stores off indexed pointers are re-derived every iteration and the pointer
// arithmetic itself carries the chain.
//
// This is the same loop through aie::vector iterators, which is how the
// kernels in openflowlm-next that do pipeline are written: one post-increment
// per stream, no index multiply, and the increment is the only loop-carried
// value. If RecMII drops to ResMII the unpack cost falls by about 3x and every
// break-even against the multiplies moves with it.

#include "tmk.h"

extern "C" void tmk_u_t2_it(const uint8_t *__restrict p, int8_t *__restrict scr) {
  event0();
  constexpr unsigned kIters = TMK_BYTES_T2 / TMK_VB;   // 32
  constexpr unsigned kStream = TMK_W / 4;              // 1024 weights per stream

  auto in = aie::begin_vector<TMK_VB>(p);
  auto s0 = aie::begin_vector<TMK_VB>((uint8_t *)scr + 0 * kStream);
  auto s1 = aie::begin_vector<TMK_VB>((uint8_t *)scr + 1 * kStream);
  auto s2 = aie::begin_vector<TMK_VB>((uint8_t *)scr + 2 * kStream);
  auto s3 = aie::begin_vector<TMK_VB>((uint8_t *)scr + 3 * kStream);

#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kIters; ++i) {
    const aie::vector<uint8_t, TMK_VB> q = *in++;
    *s0++ = aie::bit_and((uint8_t)0x03, q);
    *s1++ = aie::bit_and((uint8_t)0x0C, q);
    *s2++ = aie::bit_and((uint8_t)0x30, q);
    *s3++ = aie::bit_and((uint8_t)0xC0, q);
  }
  event1();
}
