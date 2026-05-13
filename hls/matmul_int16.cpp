// INT16 Matrix Multiply Accelerator for Zynq 7010
// Based on matmul_int8.cpp, adapted for INT16×INT16→INT64 compute.
//
// Resource estimates (vs INT8 baseline):
//   DSP:    64 (same — INT16×INT16 fits in one DSP48E1 per MAC)
//   BRAM:  ~48 KB (2×: A_buf 8K, B_buf 8K, P 32K = 36% of 135 KB)
//   LUT:   ~16K (2× wider datapaths, ~57% of 28K)
//   FF:    ~16K (wider pipeline registers, ~45% of 35.2K)
//
// Performance (same as INT8, same architecture):
//   VecMul: N=64 cycles per 64×64 tile
//   MatMul: N³/BLOCK = 32,768 cycles per 64×64 matrix
//   Clock:  150 MHz

#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>

constexpr int N = 64;
constexpr int BLOCK = 8;
constexpr int NUM_BLOCKS = N / BLOCK;

constexpr int STATUS_IDLE = 0;
constexpr int STATUS_RUNNING = 1;
constexpr int STATUS_DONE = 2;
constexpr int STATUS_ERROR = 3;

typedef ap_int<16> in_t;
typedef ap_int<32> prod_t;
typedef ap_int<64> acc_t;

// 8×8 systolic array: INT16×INT16→INT64
// Each DSP48E1: one INT16×INT16 multiply per cycle, 48-bit accumulator
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

// 64×64 blocked matrix multiply
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

// 1×64 vector-matrix multiply
void vecmul_1x64(in_t vec[N], in_t M[N][N], acc_t result[N]) {
    #pragma HLS ARRAY_PARTITION variable=vec complete
    #pragma HLS ARRAY_PARTITION variable=result cyclic factor=8
    #pragma HLS ARRAY_PARTITION variable=M cyclic factor=8 dim=2

    acc_t row_acc[N];
    #pragma HLS ARRAY_PARTITION variable=row_acc cyclic factor=8

    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=8
        row_acc[i] = 0;
    }

    for (int k = 0; k < N; k++) {
        #pragma HLS PIPELINE II=1
        in_t vk = vec[k];
        for (int i = 0; i < N; i++) {
            #pragma HLS UNROLL factor=8
            row_acc[i] += (acc_t)vk * (acc_t)M[k][i];
        }
    }

    for (int i = 0; i < N; i++) {
        #pragma HLS UNROLL factor=8
        result[i] = row_acc[i];
    }
}

// Top-level kernel
void matmul_int16(
    in_t A[N * N],
    in_t B[N * N],
    acc_t C[N * N],
    volatile ap_uint<32> *control,
    volatile ap_uint<32> *status,
    ap_uint<1> &interrupt
) {
    #pragma HLS INTERFACE m_axi port=A bundle=aximm0 offset=slave depth=4096
    #pragma HLS INTERFACE m_axi port=B bundle=aximm1 offset=slave depth=4096
    #pragma HLS INTERFACE m_axi port=C bundle=aximm2 offset=slave depth=4096
    #pragma HLS INTERFACE s_axilite port=control bundle=control
    #pragma HLS INTERFACE s_axilite port=status bundle=control
    #pragma HLS INTERFACE ap_vld port=interrupt bundle=control
    #pragma HLS INTERFACE s_axilite port=return bundle=control

    #pragma HLS RESOURCE variable=B core=RAM_2P_URAM

    *status = STATUS_IDLE;

    bool start = (*control) & 0x1;
    if (!start) {
        return;
    }

    *status = STATUS_RUNNING;

    ap_uint<1> op_vecmul = ((*control) >> 3) & 0x1;
    ap_uint<1> int_enable = ((*control) >> 1) & 0x1;

    bool vec_mode = (op_vecmul == 1);

    in_t A_mat[N][N];
    in_t B_mat[N][N];
    acc_t C_mat[N][N];
    #pragma HLS ARRAY_PARTITION variable=A_mat complete
    #pragma HLS ARRAY_PARTITION variable=B_mat cyclic factor=8 dim=2
    #pragma HLS ARRAY_PARTITION variable=C_mat complete

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            A_mat[i][j] = A[i * N + j];
        }
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            B_mat[i][j] = B[i * N + j];
        }
    }

    if (vec_mode) {
        acc_t result[N];
        #pragma HLS ARRAY_PARTITION variable=result cyclic factor=8
        vecmul_1x64(A_mat[0], B_mat, result);
        for (int i = 0; i < N; i++) {
            #pragma HLS PIPELINE II=1
            C_mat[0][i] = result[i];
        }
    } else {
        matmul_64x64(A_mat, B_mat, C_mat);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            #pragma HLS PIPELINE II=1
            C[i * N + j] = C_mat[i][j];
        }
    }

    *status = STATUS_DONE;
    if (int_enable) {
        interrupt = 1;
    }
}
