// Q8_0 Direct Path Accelerator — Type and Constant Definitions
// Zynq 7010 target, LUT-based scale multipliers, 8×8 systolic array
//
// DSP usage: 64 (8×8 systolic array only — scale multipliers in LUT)

#ifndef MATMUL_Q8_H
#define MATMUL_Q8_H

#include <ap_int.h>

constexpr int N_Q8 = 64;
constexpr int BLOCK_Q8 = 8;
constexpr int NUM_BLOCKS_Q8 = N_Q8 / BLOCK_Q8;
constexpr int Q8_BLOCK_SIZE = 32;
constexpr int Q8_BLOCK_BYTES = 34;
constexpr int N_COMBINED_SCALES = 128;  // 64 rows × 2 Q8 blocks per row

typedef ap_int<16> in_t_q8;
typedef ap_int<32> prod_t_q8;
typedef ap_int<64> acc_t_q8;

#endif
