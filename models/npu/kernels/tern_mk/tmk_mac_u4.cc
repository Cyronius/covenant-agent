//===- tmk_mac_u4.cc — the same multiplies with a PACKED 4-bit B operand ---===//
//
// aie2p's aie_api declares mmul<M,K,N,int8,uint4> (detail/aie2p/mmul_8_4.hpp,
// which lowers to mac_8x8_8x8 with b.unpack_sign() in front of it). If that
// instantiates here, 4-bit weights feed the matrix unit with no software
// unpack at all — one hardware unpack inside the MAC — and decision 3's
// premise that 2-bit and 4-bit pay the same unpack is wrong in 4-bit's favour.
// That is exactly what this TU is here to find out; it compiles or it does not.

#include "tmk.h"

#ifndef TMK_M
#define TMK_M 8
#endif
static constexpr unsigned kMB = TMK_M / TMK_MM_M;

extern "C" void tmk_mac_u4(const int8_t *__restrict a, const uint8_t *__restrict b,
                           int32_t *__restrict y) {
  event0();
  aie::mmul<TMK_MM_M, TMK_MM_K, TMK_MM_N, int8_t, uint4> C[kMB];

#pragma clang loop unroll(full)
  for (unsigned mb = 0; mb < kMB; ++mb)
    C[mb].mul(aie::load_v<64>(a + mb * TMK_K * TMK_MM_M),
              aie::load_v<64>((const uint4 *)b));

#pragma clang loop unroll(disable)
  for (unsigned kb = 1; kb < TMK_K / TMK_MM_K; ++kb) {
    const aie::vector<uint4, 64> B = aie::load_v<64>((const uint4 *)(b + kb * 32));
#pragma clang loop unroll(full)
    for (unsigned mb = 0; mb < kMB; ++mb)
      C[mb].mac(aie::load_v<64>(a + (mb * (TMK_K / TMK_MM_K) + kb) * 64), B);
  }

#pragma clang loop unroll(full)
  for (unsigned mb = 0; mb < kMB; ++mb)
    aie::store_v(y + mb * 64, C[mb].template to_vector<int32_t>());
  event1();
}
