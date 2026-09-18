//===- tmk_u_t2_u4.cc — expand 2-bit weights to NIBBLES, not bytes ---===//
//
// The int4 B operand changes what ternary has to unpack to. aie2p feeds
// mmul<4,16,16,int8,uint4> straight from packed nibbles, so a 2-bit weight's
// job is only to become a nibble — half the scratch and half the store traffic
// of the int8 target in tmk_u_t2.cc, and it lands on the same MAC path the
// 4-bit format uses, so the two formats differ in exactly one term: this loop.
//
// 1024 packed bytes -> 2048 bytes of nibbles. Per iteration: one 32-byte load
// (128 codes), duplicated so each source byte appears twice, then the even
// lane keeps codes 0,1 and the odd lane codes 2,3, each pair packed into one
// byte's two nibbles. Unlike the byte target this cannot be mask-only: codes 2
// and 3 sit above bit 4 and a nibble cannot hold their in-place factors of 16
// and 64, so they are shifted down. That asymmetry is the price of the nibble
// target and it is what this loop is here to price.

#include "tmk.h"

extern "C" void tmk_u_t2_u4(const uint8_t *__restrict p, uint8_t *__restrict scr) {
  event0();
  constexpr unsigned kIters = TMK_BYTES_T2 / TMK_VB;   // 32

#pragma clang loop unroll(disable)
  for (unsigned i = 0; i < kIters; ++i) {
    const aie::vector<uint8_t, TMK_VB> q = aie::load_v<TMK_VB>(p + i * TMK_VB);
    // each source byte twice: [b0,b0,b1,b1,...]
    const auto [d0, d1] = aie::interleave_zip(q, q, 1);
    const aie::vector<uint8_t, 64> d = aie::concat(d0, d1);

    // even lane: codes 0,1 -> nibbles 0,1     (b & 0x03) | ((b & 0x0C) << 2)
    const aie::vector<uint8_t, 64> e =
        aie::bit_or(aie::bit_and((uint8_t)0x03, d),
                    aie::upshift(aie::bit_and((uint8_t)0x0C, d), 2));
    // odd lane: codes 2,3 -> nibbles 0,1      ((b & 0x30) >> 4) | ((b & 0xC0) >> 2)
    const aie::vector<uint8_t, 64> o =
        aie::bit_or(aie::downshift(aie::bit_and((uint8_t)0x30, d), 4),
                    aie::downshift(aie::bit_and((uint8_t)0xC0, d), 2));

    const aie::mask<64> odd = aie::mask<64>::from_uint32(0xAAAAAAAA, 0xAAAAAAAA);
    aie::store_v(scr + i * 64, aie::select(e, o, odd));
  }
  event1();
}
