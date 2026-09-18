//===- tmk_mac_i8w.cc — the WIDE int8 shape, mmul<8,16,8> ---===//
//
// aie2p's mmul_8_8.hpp declares <2,8,8>, <4,8,8>, <4,16,8>, <8,16,8>, <8,8,8>.
// <8,16,8> is 1024 MACs per issue against <8,8,8>'s 512 and eats 128 weights
// per B tile instead of 64. If it schedules as well per issue, it halves the
// int8 MAC cost per weight and moves every break-even against the unpack.
// Same body shape as tmk_mac_i8.cc so the two are read off the same table.

#include "tmk.h"

#ifndef TMK_MB
#define TMK_MB 1
#endif
#ifndef TMK_NB
#define TMK_NB 1
#endif
static constexpr unsigned kKTW = TMK_K / 16;

extern "C" void tmk_mac_i8w(const int8_t *__restrict a, const int8_t *__restrict b,
                            int32_t *__restrict y) {
  event0();
  aie::mmul<8, 16, 8, int8_t, int8_t> C[TMK_MB][TMK_NB];

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKTW; ++kb) {
    aie::vector<int8_t, 128> B[TMK_NB];
#pragma clang loop unroll(full)
    for (unsigned nb = 0; nb < TMK_NB; ++nb)
      B[nb] = aie::load_v<128>(b + (nb * kKTW + kb) * 128);

#pragma clang loop unroll(full)
    for (unsigned mb = 0; mb < TMK_MB; ++mb) {
      const aie::vector<int8_t, 128> A = aie::load_v<128>(a + (mb * kKTW + kb) * 128);
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
