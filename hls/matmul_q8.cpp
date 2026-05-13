#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>
#include "matmul_q8.hpp"

constexpr int N = 64;
constexpr int BLOCK = 8;
constexpr int NUM_BLOCKS = N / BLOCK;
constexpr int Q8_BLOCK_SIZE = 32;
constexpr int Q8_BLOCK_BYTES = 34;
constexpr int ROWS_PER_SCALE = 1;

constexpr int STATUS_IDLE = 0;
constexpr int STATUS_RUNNING = 1;
constexpr int STATUS_DONE = 2;
constexpr int STATUS_ERROR = 3;

typedef ap_int<16> in_t;
typedef ap_int<32> prod_t;
typedef ap_int<64> acc_t;

// Combined scale (UQ8.8 fixed-point): block_scale / row_scale
// 8 integer bits, 8 fractional bits. Range [0, 256), precision 1/256.
typedef ap_uint<16> combined_scale_t;
constexpr int SCALE_FRAC_BITS = 8;

// Convert FP16 bits to UQ8.8 fixed-point (LUT-only: bitfield extract)
inline combined_scale_t f16_bits_to_uq8_8(ap_uint<16> f16_bits) {
#pragma HLS INLINE
    ap_uint<1> sign = f16_bits[15];
    ap_uint<5> exp = f16_bits.range(14, 10);
    ap_uint<10> mant = f16_bits.range(9, 0);

    ap_uint<16> result = 0;
    if (sign == 0) {
        if (exp >= 15) {
            // exp >= 15: normalized, value >= 1
            ap_uint<8> int_part = (ap_uint<8>)(1 << (exp - 15));
            ap_uint<8> frac_part = (exp >= 23) ? 0 :
                (ap_uint<8>)(mant.range(9 - (exp - 15), 10 - (exp - 15) + 7)) >> (23 - exp);
            result.range(15, 8) = int_part;
            result.range(7, 0) = frac_part;
        } else if (exp >= 7) {
            // exp in [7, 14]: subnormal range, value in (0, 1)
            result.range(15, 8) = 0;
            ap_uint<8> frac = (ap_uint<1>(1) << (exp - 7)) |
                              (mant >> (10 - (exp - 7)));
            result.range(7, 0) = frac;
        }
    }
    return result;
}

// Scale multiply: INT8 × UQ8.8 → result clamped to INT16
// Single integer multiply — HLS implements in LUTs with config_bind -mul_style luts
inline in_t q8_dequant_lut(ap_int<8> q8_val, combined_scale_t scale) {
#pragma HLS INLINE
    ap_int<24> product = (ap_int<24>)q8_val * (ap_int<24>)scale;
    ap_int<16> val = (ap_int<16>)(product >> SCALE_FRAC_BITS);
    if (val > 32767) val = 32767;
    if (val < -32768) val = -32768;
    return val;
}

// 8×8 systolic array: INT16×INT16→INT64
void systolic_array_8x8(in_t A[BLOCK][BLOCK], in_t B[BLOCK][BLOCK],
                         acc_t C[BLOCK][BLOCK]) {
    #pragma HLS ARRAY_PARTITION variable=A complete dim=1
    #pragma HLS ARRAY_PARTITION variable=B complete dim=2
    #pragma HLS ARRAY_PARTITION variable=C complete

    in_t a_reg[BLOCK][BLOCK];
    in_t b_reg[BLOCK][BLOCK];
    #pragma HLS ARRAY_PARTITION variable=a_reg complete
    #pragma HLS ARRAY_PARTITION variable=b_reg complete

    acc_t c_reg[BLOCK][BLOCK];
    #pragma HLS ARRAY_PARTITION variable=c_reg complete

    for (int i = 0; i < BLOCK; i++) {
        for (int j = 0; j < BLOCK; j++) {
            #pragma HLS UNROLL
            c_reg[i][j] = 0;
        }
    }

    for (int k = 0; k < BLOCK; k++) {
        for (int i = 0; i < BLOCK; i++) {
            #pragma HLS UNROLL
            a_reg[i][k] = A[i][k];
        }
        for (int j = 0; j < BLOCK; j++) {
            #pragma HLS UNROLL
            b_reg[k][j] = B[k][j];
        }
    }

    for (int t = 0; t < BLOCK; t++) {
        for (int i = 0; i < BLOCK; i++) {
            for (int j = 0; j < BLOCK; j++) {
                #pragma HLS UNROLL
                prod_t a_val = (prod_t)a_reg[i][t];
                prod_t b_val = (prod_t)b_reg[t][j];
                c_reg[i][j] += (acc_t)(a_val * b_val);
            }
        }
    }

    for (int i = 0; i < BLOCK; i++) {
        for (int j = 0; j < BLOCK; j++) {
            #pragma HLS UNROLL
            C[i][j] = c_reg[i][j];
        }
    }
}

void matmul_64x64(in_t A[N][N], in_t B[N][N], acc_t C[N][N]) {
    #pragma HLS DATAFLOW

    in_t A_buf[NUM_BLOCKS][BLOCK][N];
    in_t B_buf[NUM_BLOCKS][BLOCK][N];
    #pragma HLS ARRAY_PARTITION variable=A_buf cyclic factor=2 dim=2
    #pragma HLS ARRAY_PARTITION variable=B_buf cyclic factor=2 dim=2

    for (int p = 0; p < NUM_BLOCKS; p++) {
        for (int i = 0; i < BLOCK; i++) {
            for (int k = 0; k < N; k++) {
                #pragma HLS PIPELINE II=1
                A_buf[p][i][k] = A[p * BLOCK + i][k];
            }
        }
    }

    for (int q = 0; q < NUM_BLOCKS; q++) {
        for (int k = 0; k < BLOCK; k++) {
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                B_buf[q][k][j] = B[q * BLOCK + k][j];
            }
        }
    }

    acc_t P[NUM_BLOCKS][NUM_BLOCKS][BLOCK][BLOCK];
    #pragma HLS ARRAY_PARTITION variable=P complete dim=3
    #pragma HLS ARRAY_PARTITION variable=P complete dim=4

    for (int p = 0; p < NUM_BLOCKS; p++) {
        for (int q = 0; q < NUM_BLOCKS; q++) {
            in_t A_block[BLOCK][BLOCK];
            in_t B_block[BLOCK][BLOCK];
            acc_t C_block[BLOCK][BLOCK];

            for (int i = 0; i < BLOCK; i++) {
                for (int k = 0; k < BLOCK; k++) {
                    #pragma HLS PIPELINE II=1
                    A_block[i][k] = A_buf[p][i][q * BLOCK + k];
                }
            }

            for (int k = 0; k < BLOCK; k++) {
                for (int j = 0; j < BLOCK; j++) {
                    #pragma HLS PIPELINE II=1
                    B_block[k][j] = B_buf[q][k][j];
                }
            }

            systolic_array_8x8(A_block, B_block, C_block);

            for (int i = 0; i < BLOCK; i++) {
                for (int j = 0; j < BLOCK; j++) {
                    #pragma HLS PIPELINE II=1
                    P[p][q][i][j] = C_block[i][j];
                }
            }
        }
    }

    for (int p = 0; p < NUM_BLOCKS; p++) {
        for (int i = 0; i < BLOCK; i++) {
            for (int q = 0; q < NUM_BLOCKS; q++) {
                for (int j = 0; j < BLOCK; j++) {
                    #pragma HLS PIPELINE II=1
                    C[p * BLOCK + i][q * BLOCK + j] = P[p][q][i][j];
                }
            }
        }
    }
}

// 1×64 vector-matrix multiply (Q8→INT16 via LUT-based scale multipliers)
// combined_scales: 128 × UQ8.8 fixed-point (2 per row × 64 rows)
//   scale[row][0] for elements [0..31], scale[row][1] for elements [32..63]
void vecmul_1x64_q8(ap_int<8> B_q8[N][N],
                    combined_scale_t combined_scales[N][2],
                    in_t A[N], acc_t result[N]) {
    #pragma HLS ARRAY_PARTITION variable=A complete
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=B_q8 cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=combined_scales complete dim=1

    acc_t row_acc[N];
    #pragma HLS ARRAY_PARTITION variable=row_acc cyclic factor=8

    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=8
        row_acc[i] = 0;
    }

    for (int k = 0; k < N; k++) {
        #pragma HLS PIPELINE II=1
        in_t a_val = A[k];

        for (int i = 0; i < N; i++) {
            #pragma HLS UNROLL factor=8

            int block_idx = k / Q8_BLOCK_SIZE;
            ap_int<8> q8_val = B_q8[i][k];

            combined_scale_t cs = combined_scales[i][block_idx];

            in_t b_val = q8_dequant_lut(q8_val, cs);

            prod_t prod = (prod_t)a_val * (prod_t)b_val;
            row_acc[i] += (acc_t)prod;
        }
    }

    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=8
        result[i] = row_acc[i];
    }
}

// Top-level kernel for Q8_0 direct path with LUT-based scale multipliers
// DSP usage: 64 (8×8 systolic array), 0 for scale multipliers (LUT)
void matmul_q8(
    ap_int<8> A[N * N],                           // Q8_0 weight bytes (4096 B)
    combined_scale_t combined_scales[N * 2],      // 128 × UQ8.8 fixed-point (256 B)
    in_t X[N],                                     // Activation vector (INT16, 128 B)
    acc_t Y[N],                                    // Output vector (INT64, 512 B)
    volatile ap_uint<32> *control,
    volatile ap_uint<32> *status,
    ap_uint<1> &interrupt
) {
    #pragma HLS INTERFACE m_axi port=A bundle=aximm0 offset=slave depth=4096
    #pragma HLS INTERFACE m_axi port=combined_scales bundle=aximm1 offset=slave depth=128
    #pragma HLS INTERFACE m_axi port=X bundle=aximm2 offset=slave depth=64
    #pragma HLS INTERFACE m_axi port=Y bundle=aximm3 offset=slave depth=64
    #pragma HLS INTERFACE s_axilite port=control bundle=control
    #pragma HLS INTERFACE s_axilite port=status bundle=control
    #pragma HLS INTERFACE ap_vld port=interrupt bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS RESOURCE variable=A core=RAM_2P_BRAM

    *status = STATUS_IDLE;

    bool start = (*control) & 0x1;
    if (!start) {
        return;
    }

    *status = STATUS_RUNNING;

    ap_uint<1> int_enable = ((*control) >> 1) & 0x1;
    ap_uint<1> op_vecmul = ((*control) >> 3) & 0x1;

    bool vec_mode = (op_vecmul == 1);

    if (vec_mode) {
        ap_int<8> B_mat[N][N];
        #pragma HLS ARRAY_PARTITION variable=B_mat complete

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                B_mat[i][j] = A[i * N + j];
            }
        }

        combined_scale_t cs_mat[N][2];
        #pragma HLS ARRAY_PARTITION variable=cs_mat complete dim=1
        for (int i = 0; i < N; i++) {
            for (int b = 0; b < 2; b++) {
                #pragma HLS PIPELINE II=1
                cs_mat[i][b] = combined_scales[i * 2 + b];
            }
        }

        acc_t result[N];
        vecmul_1x64_q8(B_mat, cs_mat, X, result);

        for (int i = 0; i < N; i++) {
            #pragma HLS PIPELINE II=1
            Y[i] = result[i];
        }
    } else {
        in_t B_int16[N][N];
        #pragma HLS ARRAY_PARTITION variable=B_int16 complete

        combined_scale_t cs_mat[N][2];
        #pragma HLS ARRAY_PARTITION variable=cs_mat complete dim=1
        for (int i = 0; i < N; i++) {
            for (int b = 0; b < 2; b++) {
                #pragma HLS PIPELINE II=1
                cs_mat[i][b] = combined_scales[i * 2 + b];
            }
        }

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                int block_idx = j / Q8_BLOCK_SIZE;
                ap_int<8> q8_val = A[i * N + j];
                B_int16[i][j] = q8_dequant_lut(q8_val, cs_mat[i][block_idx]);
            }
        }

        in_t X_mat[N][N];
        #pragma HLS ARRAY_PARTITION variable=X_mat complete
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                #pragma HLS PIPELINE II=1
                X_mat[i][j] = (i == 0) ? X[j] : 0;
            }
        }

        acc_t Y_mat[N][N];
        matmul_64x64(X_mat, B_int16, Y_mat);

        for (int i = 0; i < N; i++) {
            #pragma HLS PIPELINE II=1
            Y[i] = Y_mat[0][i];
        }
    }

    *status = STATUS_DONE;
    if (int_enable) {
        interrupt = 1;
    }
}
