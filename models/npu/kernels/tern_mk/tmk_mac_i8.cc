//===- tmk_mac_i8.cc — the multiplies the unpack feeds, int8 B operand ---===//
//
// Y[M,N] += A[M,K] @ B[K,N] over one k-tile loop, register-blocked in BOTH
// dimensions: TMK_MB slot blocks of 8 by TMK_NB output blocks of 8.
//
// Both knobs matter and a kernel with only one of them lies. Per body there
// are TMK_NB B loads, TMK_MB A loads and TMK_MB*TMK_NB MAC issues, so the A
// loads amortise over NB and the B loads over MB. Measured at NB=1 the loop
// looks A-load bound at large M and the MAC cost per weight comes out several
// times too high — which is an artefact of the shape, not the machine.
//
// Live accumulators = MB*NB, each 8x8 int32. Where that exceeds the register
// file the schedule spills, and the bundle count says so.
//
//   -DTMK_MB=<slot blocks>  -DTMK_NB=<output blocks>

#include "tmk.h"

#ifndef TMK_MB
#define TMK_MB 1
#endif
#ifndef TMK_NB
#define TMK_NB 1
#endif
static constexpr unsigned kKT = TMK_K / TMK_MM_K;   // 8 k-tiles

extern "C" void tmk_mac_i8(const int8_t *__restrict a, const int8_t *__restrict b,
                           int32_t *__restrict y) {
  event0();
  aie::mmul<TMK_MM_M, TMK_MM_K, TMK_MM_N, int8_t, int8_t> C[TMK_MB][TMK_NB];

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKT; ++kb) {
    aie::vector<int8_t, 64> B[TMK_NB];
#pragma clang loop unroll(full)
    for (unsigned nb = 0; nb < TMK_NB; ++nb)
      B[nb] = aie::load_v<64>(b + (nb * kKT + kb) * 64);

#pragma clang loop unroll(full)
    for (unsigned mb = 0; mb < TMK_MB; ++mb) {
      const aie::vector<int8_t, 64> A = aie::load_v<64>(a + (mb * kKT + kb) * 64);
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
