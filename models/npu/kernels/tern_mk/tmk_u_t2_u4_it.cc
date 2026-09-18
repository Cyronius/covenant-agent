//===- tmk_u_t2_u4_it.cc — 2-bit to NIBBLE expand, iterator form ---===//
//
// tmk_u_t2_u4.cc under the addressing of tmk_u_t2_it.cc. This is the loop that
// decides whether ternary can ride the int4 MAC path: the 4-bit format feeds
// mmul<4,16,16,int8,uint4> from memory with no software unpack at all, so if
// ternary wants the same 1024-MAC issue it has to arrive as nibbles, and this
// is what that costs.

#include "tmk.h"

extern "C" void tmk_u_t2_u4_it(const uint8_t *__restrict p, uint8_t *__restrict scr) {
  event0();
  constexpr unsigned kIters = TMK_BYTES_T2 / TMK_VB;   // 32
  const aie::mask<64> odd = aie::mask<64>::from_uint32(0xAAAAAAAA, 0xAAAAAAAA);

  auto in = aie::begin_vector<TMK_VB>(p);
  auto out = aie::begin_vector<64>(scr);

#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kIters; ++i) {
    const aie::vector<uint8_t, TMK_VB> q = *in++;
    const auto [d0, d1] = aie::interleave_zip(q, q, 1);
    const aie::vector<uint8_t, 64> d = aie::concat(d0, d1);

    const aie::vector<uint8_t, 64> e =
        aie::bit_or(aie::bit_and((uint8_t)0x03, d),
                    aie::upshift(aie::bit_and((uint8_t)0x0C, d), 2));
    const aie::vector<uint8_t, 64> o =
        aie::bit_or(aie::downshift(aie::bit_and((uint8_t)0x30, d), 4),
                    aie::downshift(aie::bit_and((uint8_t)0xC0, d), 2));

    *out++ = aie::select(e, o, odd);
  }
  event1();
}
