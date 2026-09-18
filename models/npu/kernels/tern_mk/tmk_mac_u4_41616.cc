//===- tmk_mac_u4_41616.cc — the other int4 shape aie_api declares ---===//
//
// mmul<8,8,8,int8,uint4> does not instantiate for aie2p. mmul_8_4.hpp declares
// one more shape, <4,16,16>, behind its own guard. If that one is live, 4-bit
// weights still reach the matrix unit with no software unpack and the format
// comparison changes; if it is not, aie2p has no native sub-byte B operand at
// all and every packed format pays a software unpack, which is the premise
// decision 3 argues from.

#include "tmk.h"

extern "C" void tmk_mac_u4_41616(const int8_t *__restrict a, const uint8_t *__restrict b,
                                 int32_t *__restrict y) {
  event0();
  aie::mmul<4, 16, 16, int8_t, uint4> C;
  C.mul(aie::load_v<64>(a), aie::load_v<256>((const uint4 *)b));
  aie::store_v(y, C.template to_vector<int32_t>());
  event1();
}
