//===- tmk_mac_i8_it.cc — int8 MACs under incrementing addressing ---===//
//
// tmk_mac_i8.cc with the addressing that fixed the unpacks: no index multiply
// in the loop body, one post-increment per stream. It matters which side of
// the comparison gets the better schedule — if the MACs speed up and the
// unpack does not, every format that has to unpack looks worse — so both sides
// are measured at their best known form before any ratio is quoted.
//
// aie::vector_iterator has no default constructor, so an array of them cannot
// be declared and filled; plain pointers carry the same induction.
//
//   -DTMK_MB=<slot blocks of 8>  -DTMK_NB=<output blocks of 8>

#include "tmk.h"

#ifndef TMK_MB
#define TMK_MB 1
#endif
#ifndef TMK_NB
#define TMK_NB 1
#endif
static constexpr unsigned kKT = TMK_K / TMK_MM_K;   // 8 k-tiles

extern "C" void tmk_mac_i8_it(const int8_t *__restrict a, const int8_t *__restrict b,
                              int32_t *__restrict y) {
  event0();
  aie::mmul<TMK_MM_M, TMK_MM_K, TMK_MM_N, int8_t, int8_t> C[TMK_MB][TMK_NB];

  const int8_t *ap[TMK_MB];
  const int8_t *bp[TMK_NB];
#pragma clang loop unroll(full)
  for (unsigned mb = 0; mb < TMK_MB; ++mb)
    ap[mb] = a + mb * kKT * 64;
#pragma clang loop unroll(full)
  for (unsigned nb = 0; nb < TMK_NB; ++nb)
    bp[nb] = b + nb * kKT * 64;

#pragma clang loop unroll(disable)
  for (unsigned kb = 0; kb < kKT; ++kb) {
    aie::vector<int8_t, 64> B[TMK_NB];
#pragma clang loop unroll(full)
    for (unsigned nb = 0; nb < TMK_NB; ++nb) {
      B[nb] = aie::load_v<64>(bp[nb]);
      bp[nb] += 64;
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
