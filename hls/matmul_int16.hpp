// INT16 Matrix Multiply Accelerator — Type and Constant Definitions
// Zynq 7010 target, 8×8 systolic array, N=64

#ifndef MATMUL_INT16_H
#define MATMUL_INT16_H

#include <ap_int.h>

constexpr int N16 = 64;
constexpr int BLOCK16 = 8;
constexpr int NUM_BLOCKS16 = N16 / BLOCK16;

typedef ap_int<16> in_t16;
typedef ap_int<32> prod_t16;
typedef ap_int<64> acc_t16;

#endif
