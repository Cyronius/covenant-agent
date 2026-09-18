#pragma once
//===- tmk.h — ternary microkernel: shared shapes and unpack helpers ---*- C++ -*-===//
//
// Step 5 of .claude/plans/npu-native-planner.md, cut down to the one question
// decision 3 rests on: what does the unpack cost per weight, and how does it
// compare to the multiplies it feeds, for 2-bit, 4-bit and int8 weights.
//
// Nothing here is a design. Each .cc is one loop so kernel_remarks.py can put a
// cycle count and an II bound against it; no xclbin is built and no hardware is
// touched. Numbers come out of Peano, which is how a format choice gets judged
// in seconds instead of a build-and-run cycle.
//
// LAYOUT. A weight block is N=64 output rows by K=64, consumed as 8x8 B tiles
// (k-major, the shape aie::mmul<8,8,8> wants: B = vector<T,64> = 8 k by 8 n).
// The packed streams are laid out so that ONE masked vector op produces one
// contiguous run of unpacked weights — no shift, no shuffle, no scalar gather.
//
// THE MASK-ONLY TRICK, and why 2-bit and 4-bit cost the same per weight.
// gemm_q4_dequant.h never shifts a nibble down; it masks the high nibble in
// place (0xF0) and folds the resulting factor of 16 into the scale. The same
// applies one level further: the four 2-bit codes in a byte are masked in
// place (0x03, 0x0C, 0x30, 0xC0), carrying factors 1, 4, 16, 64, and each
// factor folds into its stream's scale. A stream is a set of k positions, so
// the four streams partition K rather than multiplying the MAC count.
//
//   32 bytes of q4  -> 2 masks -> 64 weights   = 1 op per 32 weights
//   32 bytes of t2  -> 4 masks -> 128 weights  = 1 op per 32 weights
//
// That is decision 3's claim as arithmetic. What it does not count is the
// stores into scratch and whatever the scheduler makes of the dependence
// chain, which is the part that has to be measured rather than argued.
//
// PACKED STREAM ORDER. For t2, stream s (code position s in each byte) holds
// the weights for k in [16s, 16s+16) of each 64-k group; for q4, stream s
// holds k in [32s, 32s+32). So the unpack writes each stream to a contiguous
// scratch run and the MAC reads 8x8 tiles out of it with plain loads.

#include <aie_api/aie.hpp>
#include <stdint.h>

// One weight block: 64 output rows x 64 K = 4096 weights.
static constexpr unsigned TMK_N = 64;
static constexpr unsigned TMK_K = 64;
static constexpr unsigned TMK_W = TMK_N * TMK_K;   // 4096 weights

// Bytes of packed weight per block, per format.
static constexpr unsigned TMK_BYTES_I8 = TMK_W;        // 4096
static constexpr unsigned TMK_BYTES_Q4 = TMK_W / 2;    // 2048
static constexpr unsigned TMK_BYTES_T2 = TMK_W / 4;    // 1024

// The vector width every loop below works in: 32 bytes in, because that is the
// load that keeps a 2-bit block's four streams inside one 128-lane store.
static constexpr unsigned TMK_VB = 32;

// mmul<8,8,8>: A is 8 slots x 8 k int8, B is 8 k x 8 n, C is 8x8 int32.
static constexpr unsigned TMK_MM_M = 8;
static constexpr unsigned TMK_MM_K = 8;
static constexpr unsigned TMK_MM_N = 8;
