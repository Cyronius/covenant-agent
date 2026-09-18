//===- tmk_mac_u4k.cc — MACs with a PACKED 4-bit B operand, k-loop ---===//
//
// Y[M,16] += A[M,K] @ B[K,16] with B left packed: mmul<4,16,16,int8,uint4>,
// the one sub-byte shape aie2p instantiates. One issue eats 4x16x16 = 1024
// MACs and 256 weights, against mmul<8,8,8,int8,int8>'s 512 and 64, and the
// hardware unpack sits inside the MAC (mac_4x8_8x16_conf(..., b.unpack_sign(),
// ...)), so no software unpack runs at all.
//
// A slot block here is 4 slots, not 8. Per body: NB B loads (128 B each),
// MB A loads, MB*NB issues, NB*256 weights consumed.
//
//   -DTMK_MB=<slot blocks of 4>  -DTMK_NB=<output blocks of 16>

#include "tmk.h"

#ifndef TMK_MB
#define TMK_MB 1
#endif
#ifndef TMK_NB
#define TMK_NB 1
#endif
static constexpr unsigned kKT4 = TMK_K / 16;   // 16-wide k-tiles

extern "C" void tmk_mac_u4k(const int8_t *__restrict a, const uint8_t *__restrict b,
                            int32_t *__restrict y) {
  event0();
  aie::mmul<4, 16, 16, int8_t, uint4> C[TMK_MB][TMK_NB];

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKT4; ++kb) {
    aie::vector<uint4, 256> B[TMK_NB];
#pragma clang loop unroll(full)
    for (unsigned nb = 0; nb < TMK_NB; ++nb)
      B[nb] = aie::load_v<256>((const uint4 *)(b + (nb * kKT4 + kb) * 128));

#pragma clang loop unroll(full)
    for (unsigned mb = 0; mb < TMK_MB; ++mb) {
      const aie::vector<int8_t, 64> A = aie::load_v<64>(a + (mb * kKT4 + kb) * 64);
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
