// Q8_0 Direct Path — C++ Simulation of FPGA Q8→INT16 Matmul
//
// matmul_q8_full computes: y = x @ W  where W is Q8_0 quantized
// GGUF storage: W[output][input] at flat_idx = input + output * input_dim
//            = col + row * cols  (column-major, cols = ne[0] = input_dim)

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include "fpga_sim.hpp"

struct Tensor {
    char name[128];
    uint64_t rows, cols;
    uint32_t type;
    uint64_t n_bytes;
    uint8_t* data;
};

namespace q8_path {

using in_t = fpga_sim::in16_t;
using acc_t = fpga_sim::acc16_t;
constexpr int N = fpga_sim::N;
constexpr int Q8_BLOCK_SIZE = 32;
constexpr int Q8_BLOCK_BYTES = 34;

inline float read_f16(const uint8_t* data) {
    uint16_t raw = (uint16_t)data[0] | ((uint16_t)data[1] << 8);
    uint32_t sign = (raw >> 15) & 0x1;
    uint32_t exp = (raw >> 10) & 0x1F;
    uint32_t mant = raw & 0x3FF;
    if (exp == 0) {
        return (mant == 0) ? 0.0f : (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
    }
    if (exp == 31) return 0.0f;
    return (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, (int)exp - 15);
}

void matmul_q8_full(const Tensor* A,
                    const float* x,
                    float* y,
                    int rows, int cols,
                    const float* row_max_abs = nullptr) {
    if (rows == 0 || cols == 0) return;
    memset(y, 0, (size_t)rows * sizeof(float));

    float x_scale = 0.0f;
    for (int j = 0; j < cols; j++) x_scale = fmaxf(x_scale, fabsf(x[j]));
    x_scale = (x_scale < 1e-10f) ? 1.0f : x_scale / 32767.0f;

    int16_t x_q[896];
    for (int j = 0; j < cols; j++)
        x_q[j] = (int16_t)roundf(x[j] / x_scale);

    int64_t tile_count = 0;
    int64_t mac_ops = 0;

    for (int r = 0; r < rows; r += N) {
        int r_size = std::min(N, rows - r);

        float row_scale[N];
        if (row_max_abs) {
            for (int i = 0; i < r_size; i++) {
                int row = r + i;
                float max_abs = row_max_abs[row];
                row_scale[i] = (max_abs < 1e-10f) ? 1.0f : max_abs / 32767.0f;
            }
        } else {
            for (int i = 0; i < r_size; i++) {
                int row = r + i;
                float max_abs = 0.0f;
                for (int j = 0; j < cols; j++) {
                    uint64_t flat_idx = (uint64_t)j + (uint64_t)row * A->rows;
                    uint64_t block = flat_idx / Q8_BLOCK_SIZE;
                    uint64_t offset = flat_idx % Q8_BLOCK_SIZE;
                    uint64_t block_off = block * Q8_BLOCK_BYTES + 2 + offset;
                    int8_t q8_val = (int8_t)A->data[block_off];
                    uint64_t scale_off = block * Q8_BLOCK_BYTES;
                    float block_scale = read_f16(A->data + scale_off);
                    float dequant_val = q8_val * block_scale;
                    float a = fabsf(dequant_val);
                    if (a > max_abs) max_abs = a;
                }
                row_scale[i] = (max_abs < 1e-10f) ? 1.0f : max_abs / 32767.0f;
            }
        }
        for (int i = r_size; i < N; i++) row_scale[i] = 1.0f;

        for (int c = 0; c < cols; c += N) {
            int c_size = std::min(N, cols - c);

            uint8_t q8_bytes[N][N];
            for (int i = 0; i < r_size; i++) {
                int row = r + i;
                for (int k = 0; k < c_size; k++) {
                    int col = c + k;
                    uint64_t flat_idx = (uint64_t)col + (uint64_t)row * A->rows;
                    uint64_t block = flat_idx / Q8_BLOCK_SIZE;
                    uint64_t offset = flat_idx % Q8_BLOCK_SIZE;
                    uint64_t block_off = block * Q8_BLOCK_BYTES + 2 + offset;
                    q8_bytes[i][k] = A->data[block_off];
                }
                for (int k = c_size; k < N; k++) q8_bytes[i][k] = 0;
            }
            for (int i = r_size; i < N; i++)
                for (int k = 0; k < N; k++)
                    q8_bytes[i][k] = 0;

            int16_t W_int16[N][N];
            for (int i = 0; i < r_size; i++) {
                float row_s = row_scale[i];
                for (int k = 0; k < c_size; k++) {
                    int col = c + k;
                    int row = r + i;
                    uint64_t flat_idx = (uint64_t)col + (uint64_t)row * A->rows;
                    uint64_t block = flat_idx / Q8_BLOCK_SIZE;
                    uint64_t scale_off = block * Q8_BLOCK_BYTES;
                    float block_scale = read_f16(A->data + scale_off);

                    int8_t q8_val = (int8_t)q8_bytes[i][k];
                    float dequant_val = q8_val * block_scale;
                    float val_norm = dequant_val / row_s;

                    if (val_norm > 32767.0f) val_norm = 32767.0f;
                    if (val_norm < -32768.0f) val_norm = -32768.0f;
                    W_int16[k][i] = (int16_t)(val_norm >= 0 ? (val_norm + 0.5f) : (val_norm - 0.5f));
                }
                for (int k = c_size; k < N; k++) W_int16[k][i] = 0;
            }
            for (int i = r_size; i < N; i++)
                for (int k = 0; k < N; k++)
                    W_int16[k][i] = 0;

            int16_t vec[N] = {0};
            for (int k = 0; k < c_size; k++)
                vec[k] = x_q[c + k];
            for (int k = c_size; k < N; k++) vec[k] = 0;

            int64_t result[N] = {0};
            for (int k = 0; k < N; k++) {
                int16_t vk = vec[k];
                for (int i = 0; i < r_size; i++) {
                    result[i] += (int64_t)vk * (int64_t)W_int16[k][i];
                }
            }

            for (int i = 0; i < r_size; i++)
                y[r + i] += (double)result[i] * x_scale * row_scale[i];

            tile_count++;
            mac_ops += r_size * c_size;

#ifdef Q8_DEBUG
            if (tile_count <= 3) {
                fprintf(stderr, "[Q8] tile %lld: r=%d c=%d result[0]=%lld y[%d]=%.6e (x_scale=%.6e row_scale=%.6e)\n",
                        (long long)tile_count, r, c, (long long)result[0], r,
                        (double)y[r], (double)x_scale, (double)row_scale[0]);
            }
#endif
        }
    }

#ifdef Q8_DEBUG
    fprintf(stderr, "[Q8] DONE: tiles=%lld mac_ops=%lld y[0]=%.6e y[1]=%.6e y[2]=%.6e\n",
            (long long)tile_count, (long long)mac_ops, (double)y[0], (double)y[1], (double)y[2]);
#endif
    fpga_sim::g_timing.total_tiles += tile_count;
    fpga_sim::g_timing.total_mac_ops += mac_ops;
    fpga_sim::g_timing.total_fpga_cycles += tile_count * fpga_sim::TileCycleBudget::Q8_TILE_CYCLES;
}

} // namespace q8_path

// External tensor lookup (implemented in tmac_gguf.cpp)
extern Tensor* get_tensor_info(const char* name);

void q8_logits_matmul_with_tensor(const Tensor* emb_t, const float* x, float* y, int rows, int cols) {
    if (emb_t) {
        char row_max_name[256];
        snprintf(row_max_name, sizeof(row_max_name), "%s_row_max_abs", emb_t->name);
        const float* row_max_abs = nullptr;
        Tensor* row_max_t = get_tensor_info(row_max_name);
        if (row_max_t && row_max_t->type == 0 && row_max_t->cols == 1) {
            row_max_abs = (const float*)row_max_t->data;
        }

        extern void matmul_fpga_q8(const Tensor* A, const float* x, float* y, int rows, int cols, const float* row_max_abs = nullptr);
        matmul_fpga_q8(emb_t, x, y, rows, cols, row_max_abs);
    }
}