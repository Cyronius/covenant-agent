//===- tmk_mac_u4k_it.cc — packed-4-bit MACs under incrementing addressing ---===//
//
// tmk_mac_u4k.cc's counterpart to tmk_mac_i8_it.cc. Slot blocks are 4 wide and
// output blocks 16, because mmul<4,16,16,int8,uint4> is the one sub-byte shape
// aie2p instantiates. The B operand stays packed: 128 bytes per tile, and the
// unpack is the hardware's, inside the MAC.
//
//   -DTMK_MB=<slot blocks of 4>  -DTMK_NB=<output blocks of 16>

#include "tmk.h"

#ifndef TMK_MB
#define TMK_MB 1
#endif
#ifndef TMK_NB
#define TMK_NB 1
#endif
static constexpr unsigned kKT4 = TMK_K / 16;

extern "C" void tmk_mac_u4k_it(const int8_t *__restrict a, const uint8_t *__restrict b,
                               int32_t *__restrict y) {
  event0();
  aie::mmul<4, 16, 16, int8_t, uint4> C[TMK_MB][TMK_NB];

  const int8_t *ap[TMK_MB];
  const uint8_t *bp[TMK_NB];
#pragma clang loop unroll(full)
  for (unsigned mb = 0; mb < TMK_MB; ++mb)
    ap[mb] = a + mb * kKT4 * 64;
#pragma clang loop unroll(full)
  for (unsigned nb = 0; nb < TMK_NB; ++nb)
    bp[nb] = b + nb * kKT4 * 128;

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKT4; ++kb) {
    aie::vector<uint4, 256> B[TMK_NB];
#pragma clang loop unroll(full)
    for (unsigned nb = 0; nb < TMK_NB; ++nb) {
      B[nb] = aie::load_v<256>((const uint4 *)bp[nb]);
      bp[nb] += 128;
    }

#pragma clang loop unroll(full)
    for (unsigned mb = 0; mb < TMK_MB; ++mb) {
      const aie::vector<int8_t, 64> A = aie::load_v<64>(ap[mb]);
      ap[mb] += 64;
#pragma clang loop unroll(full)
      for (unsigned nb = 0; nb < TMK_NB; ++nb)
        C[mb][nb].mac(A, B[nb]);
    }
  }

#pragma clang loop unroll(full)
  for (unsigned mb = 0; mb < TMK_MB; ++mb)
#pragma clang loop unroll(full)
    for (unsigned nb = 0; nb < TMK_NB; ++nb)
      aie::store_v(y + (mb * TMK_NB + nb) * 64, C[mb][nb].template to_vector<int32_t>());
  event1();
}
