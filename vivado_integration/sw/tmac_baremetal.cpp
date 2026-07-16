/*
 * T-MAC Bare-Metal Inference Engine for Zynq 7010
 * FPGA accelerated: Q8_0 and Q5_0 via HP descriptor-chain
 * CPU: RMSNorm, RoPE, SiLU, SoftMax, Attention, Q4_K/Q6_K/F32 matmul fallback
 *
 * Model loaded at 0x00200000 by bootloader/script.
 * Results written to 0x1F000000 for host readback.
 * Status via UART0 (115200 baud).
 */

#include "tmac_baremetal.h"

// ===== Global State =====
static InferenceState g_state;
static int g_debug = 0;

// ===== Model Loading =====
static int load_tmac(uint32_t ddr_base) {
    uint8_t* base = (uint8_t*)(uintptr_t)ddr_base;
    uint8_t* p = base;

    if (p[0] != 0x54 || p[1] != 0x4D || p[2] != 0x41 || p[3] != 0x43) {
        uart_puts("ERROR: Bad magic\n"); return -1;
    }
    p += 4;

    uint64_t n_tensors;
    memcpy(&n_tensors, p, 8); p += 8;
    if (n_tensors > 200) { uart_puts("ERROR: Too many tensors\n"); return -1; }

    g_state.tensors = (Tensor*)(uintptr_t)0x17E10000UL; // tensor table in scratch area
    g_state.ntensors = (int)n_tensors;

    uart_puts("Loading "); uart_putdec((int)n_tensors); uart_puts(" tensors\n");

    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t name_len;
        memcpy(&name_len, p, 8); p += 8;
        if (name_len >= 128) name_len = 127;
        memcpy(g_state.tensors[i].name, p, name_len);
        g_state.tensors[i].name[name_len] = 0;
        p += name_len;

        uint64_t rows, cols, n_bytes;
        uint32_t type;
        memcpy(&rows, p, 8); p += 8;
        memcpy(&cols, p, 8); p += 8;
        memcpy(&type, p, 4); p += 4;
        memcpy(&n_bytes, p, 8); p += 8;

        g_state.tensors[i].rows = rows;
        g_state.tensors[i].cols = cols;
        g_state.tensors[i].type = type;
        g_state.tensors[i].n_bytes = n_bytes;
        g_state.tensors[i].data = p;

        if (g_debug && i < 5) {
            uart_puts("  ["); uart_putdec((int)i); uart_puts("] ");
            uart_puts(g_state.tensors[i].name);
            uart_puts(" rows="); uart_putdec((int)rows);
            uart_puts(" cols="); uart_putdec((int)cols);
            uart_puts(" type="); uart_putdec((int)type);
            uart_puts(" bytes="); uart_putdec((int)n_bytes);
            uart_puts("\n");
        }
        p += n_bytes;
    }

    uint64_t total_bytes = (uint64_t)(p - base);
    uart_puts("OK Loaded "); uart_putdec((int)n_tensors);
    uart_puts(" tensors, "); uart_putdec((int)(total_bytes / 1048576));
    uart_puts(" MB\n");
    return 0;
}

// ===== Tensor Access =====
static Tensor* get_tensor(const char* name) {
    for (int i = 0; i < g_state.ntensors; i++) {
        const char* tn = g_state.tensors[i].name;
        int j = 0;
        while (name[j] && tn[j] && name[j] == tn[j]) j++;
        if (name[j] == 0 && tn[j] == 0) return &g_state.tensors[i];
    }
    return 0;
}

// Dequantize one element from a tensor
static float dequant(const Tensor* t, uint64_t idx) {
    uint8_t* d = t->data;
    if (t->type == TENSOR_F32) return ((float*)d)[idx];
    if (t->type == TENSOR_F16) return f16_to_f32(((uint16_t*)d)[idx]);

    if (t->type == TENSOR_Q8_0) {
        uint64_t block = idx / 32, off = idx % 32;
        uint64_t bo = block * 34;
        uint16_t sr = (uint16_t)d[bo] | ((uint16_t)d[bo+1] << 8);
        float scale = f16_to_f32(sr);
        return (float)(int8_t)d[bo + 2 + off] * scale;
    }
    if (t->type == TENSOR_Q5_0) {
        uint64_t block = idx / 32, off = idx % 32;
        uint64_t bo = block * 22;
        float d_val = f16_to_f32((uint16_t)d[bo] | ((uint16_t)d[bo+1] << 8));
        uint32_t qh = (uint32_t)d[bo+2] | ((uint32_t)d[bo+3]<<8) |
                      ((uint32_t)d[bo+4]<<16) | ((uint32_t)d[bo+5]<<24);
        uint64_t j = off < 16 ? off : off - 16;
        uint8_t qs = d[bo + 6 + j];
        uint8_t ql = (off < 16) ? (qs & 0xF) : (qs >> 4);
        uint8_t qh_bit = (qh >> off) & 1;
        int q = ((qh_bit << 4) | ql) - 16;
        return d_val * (float)q;
    }
    if (t->type == TENSOR_Q6_K) {
        uint64_t bo = (idx / 256) * 210;
        float super = f16_to_f32((uint16_t)d[bo+208] | ((uint16_t)d[bo+209]<<8));
        uint64_t wi = idx % 256;
        int half = wi / 128, pos = wi % 128, l = pos % 32, sub = pos / 32;
        int lo = half * 64 + l + (sub % 2) * 32;
        uint8_t ql_n = (sub < 2) ? (d[lo] & 0xF) : (d[lo] >> 4);
        int ho = 128 + half * 32 + l;
        uint8_t qh_b = (d[ho] >> (sub * 2)) & 0x3;
        int q6 = ((qh_b << 4) | ql_n) - 32;
        int sco = 192 + half * 8 + (l / 16) + sub * 2;
        float sf = (float)(int8_t)d[sco];
        return super * sf * (float)q6;
    }
    if (t->type == TENSOR_Q4_K) {
        uint64_t bo = (idx / 256) * 144;
        float d_val = f16_to_f32((uint16_t)d[bo] | ((uint16_t)d[bo+1]<<8));
        float dmin = f16_to_f32((uint16_t)d[bo+2] | ((uint16_t)d[bo+3]<<8));
        uint64_t wi = idx % 256;
        int sub = wi / 32, j = wi % 32;
        uint8_t* sc = d + bo + 4;
        int sc_val, m_val;
        if (sub < 4) { sc_val = sc[sub] & 63; m_val = sc[sub + 4] & 63; }
        else { sc_val = (sc[sub+4] & 0xF) | ((sc[sub-4] >> 6) << 4);
               m_val = (sc[sub+4] >> 4) | ((sc[sub] >> 6) << 4); }
        uint8_t qs = d[bo + 16 + (sub/2)*32 + j];
        uint8_t q4 = (sub % 2 == 0) ? (qs & 0xF) : (qs >> 4);
        return d_val * (float)sc_val * (float)q4 - dmin * (float)m_val;
    }
    return 0.0f;
}

// ===== RMS Norm =====
static void rms_norm(float* o, const float* x, int n, const Tensor* t) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum / n + 1e-6f);
    float* w = (float*)t->data;
    for (int i = 0; i < n; i++) o[i] = x[i] * w[i] * scale;
}

// ===== SiLU =====
static void silu_forward(float* y, const float* x, int n) {
    for (int i = 0; i < n; i++) y[i] = x[i] / (1.0f + expf(-x[i]));
}

// ===== RoPE =====
static void apply_rope(float* q, float* k, int pos) {
    const float rope_base = 1000000.0f;
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d += 2) {
            float theta = 1.0f / powf(rope_base, (float)d / HEAD_DIM);
            float freq = pos * theta;
            float c = cosf(freq), s = sinf(freq);
            int idx = h * HEAD_DIM + d;
            float q0 = q[idx], q1 = q[idx+1];
            q[idx]   = q0 * c - q1 * s;
            q[idx+1] = q0 * s + q1 * c;
        }
    }
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d += 2) {
            float theta = 1.0f / powf(rope_base, (float)d / HEAD_DIM);
            float freq = pos * theta;
            float c = cosf(freq), s = sinf(freq);
            int idx = h * HEAD_DIM + d;
            float k0 = k[idx], k1 = k[idx+1];
            k[idx]   = k0 * c - k1 * s;
            k[idx+1] = k0 * s + k1 * c;
        }
    }
}

// ===== FP16 read from data pointer =====
static inline float read_f16_p(const uint8_t* p) {
    return f16_to_f32((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

// ====================================================================
// MATMUL DISPATCH
// ====================================================================

// ---- CPU matmul: dequant + float accumulate ----
static void cpu_matmul(const Tensor* A, const float* x, float* y, int rows, int cols) {
    memset(y, 0, rows * sizeof(float));
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += dequant(A, (uint64_t)i * cols + j) * x[j];
        }
        y[i] = sum;
    }
}

// ---- Q8_0 FPGA Path via HP descriptor chain ----
// Pre-process: extract INT8 values from Q8_0 blocks, pack row-major
// into FPGA weight format (8 INT8 per 64-bit word, row-major).
// Then build descriptor chain and let HP FSM compute.
//
// Tile: 64×896 = 14 column groups × 64 cols
// Weight data per tile: 14 × 4096 = 57344 bytes
// Weights are read from DDR directly by HP read master.
//
// For the first implementation, process one 64-row tile at a time
// (build one descriptor, run, read results, dequantize, repeat).
// Pre-process one Q8_0 tile: extract INT8 from Q8 blocks to FPGA format.
// fpga_wt: output buffer (14336 bytes for 14 groups × 1024 bytes)
// Actually Q8_GROUP_BYTES=4096 (512 × 64-bit words), so 14×4096=57344.
static void q8_preprocess_tile(const Tensor* A, int row0, uint8_t* fpga_wt) {
    const int stride_blocks = Q8_TILE_COLS / 32;  // 28 blocks per row
    const int cols = (int)A->cols;

    for (int g = 0; g < Q8_NUM_GROUPS; g++) {
        int col0 = g * Q8_GROUP_COLS;
        uint8_t* group_out = fpga_wt + g * Q8_GROUP_BYTES;

        // Column-major format: group_out[col * 64 + row] = W[row][col]
        // This matches the RTL wmem layout where address = g*64 + k
        // and the Q8 core reads column-major data.
        for (int r = 0; r < Q8_TILE_ROWS; r++) {
            int row = row0 + r;
            if (row >= (int)A->rows) break;
            for (int c = 0; c < Q8_GROUP_COLS; c++) {
                int blk32 = (row * cols + col0 + c) / 32;
                int blk_off = (blk32 / stride_blocks) * stride_blocks + (blk32 % stride_blocks);
                uint64_t block_idx = (uint64_t)blk_off;
                uint64_t block_off = block_idx * 34;
                int blk_elem = (row * cols + col0 + c) % 32;
                // Transpose: col-major output
                group_out[c * Q8_TILE_ROWS + r] = A->data[block_off + 2 + blk_elem];
            }
        }
        // Zero-fill rows beyond actual rows in the incomplete tile
        for (int r = 0; r < Q8_TILE_ROWS; r++) {
            int row = row0 + r;
            if (row >= (int)A->rows) {
                for (int c = 0; c < Q8_GROUP_COLS; c++)
                    group_out[c * Q8_TILE_ROWS + r] = 0;
            }
        }
    }

    // Write per-column scales (all 1.0 = 0x0100 UQ8.8) at offset Q8_TILE_WEIGHT_BYTES.
    // The C post-processing in fpga_q8_tile handles the real dequantization via
    // row_inv and x_scale; smem scales in the FPGA are no-ops for now.
    uint8_t* scale_out = fpga_wt + Q8_TILE_WEIGHT_BYTES;
    for (int g = 0; g < Q8_NUM_GROUPS; g++) {
        for (int c = 0; c < Q8_GROUP_COLS; c++) {
            scale_out[g * Q8_GROUP_SCALE_BYTES + c * 2 + 0] = 0x00;
            scale_out[g * Q8_GROUP_SCALE_BYTES + c * 2 + 1] = 0x01; // 0x0100 LE = 1.0
        }
    }
}

// Row_inv computation for Q8_0 tile: 32767/max_abs for each of 64 rows
static void q8_compute_row_inv(const Tensor* A, int row0, float* row_inv) {
    const int cols = (int)A->cols;
    for (int r = 0; r < Q8_TILE_ROWS && (row0 + r) < (int)A->rows; r++) {
        int row = row0 + r;
        float max_abs = 0.0f;
        for (int j = 0; j < cols; j++) {
            float v = dequant(A, (uint64_t)row * cols + j);
            float a = fabsf(v);
            if (a > max_abs) max_abs = a;
        }
        if (max_abs < 1e-10f) max_abs = 1.0f;
        row_inv[r] = 32767.0f / max_abs;
    }
}

// Run Q8_0 matmul via FPGA HP descriptor for one 64-row tile
static void fpga_q8_tile(const uint8_t* fpga_wt, const int16_t* x_q,
                         float* y, int row0, float x_scale,
                         const float* row_inv, int nrows)
{
    // Format: one descriptor per tile
    // weight_addr points to pre-processed weight data + scales (Q8_TILE_STRIDE bytes)
    // act_addr points to INT16 activations (1792 bytes = 896×2)
    // result_addr points to 64×8 = 512 bytes output

    uint32_t desc_addr = DESC_CHAIN_BASE;
    uint32_t wt_addr = FPGA_WEIGHT_REFMT;
    uint32_t act_addr = FPGA_ACT_BUF;
    uint32_t res_addr = FPGA_RES_BUF;

    // 1. Write weight data + scales to scratch
    memcpy((void*)(uintptr_t)wt_addr, fpga_wt, Q8_TILE_STRIDE);

    // 2. Write activations
    memcpy((void*)(uintptr_t)act_addr, x_q, Q8_TILE_COLS * 2);

    // 3. Build descriptor (32 bytes, little-endian)
    Descriptor* d = (Descriptor*)(uintptr_t)desc_addr;
    d->next_addr = 0;
    d->weight_addr = wt_addr;
    d->act_addr = act_addr;
    d->result_addr = res_addr;
    d->tensor_type = DESC_Q8;
    d->num_groups = Q8_NUM_GROUPS;
    d->act_total_bytes[0] = (Q8_TILE_COLS * 2) & 0xFF;
    d->act_total_bytes[1] = ((Q8_TILE_COLS * 2) >> 8) & 0xFF;
    d->act_total_bytes[2] = ((Q8_TILE_COLS * 2) >> 16) & 0xFF;

    // 4. Set up and start descriptor chain
    reg_write32(REG_DESC_BASE, desc_addr);
    __asm__ volatile("dsb" ::: "memory");
    reg_write32(REG_START, 1);

    // 5. Poll for completion (reg_desc_head increments, or wait for busy=0)
    uint32_t timeout = 10000000; // ~100ms at 100MHz
    while (timeout--) {
        uint32_t status = reg_read32(REG_STATUS);
        if (!(status & 0x8000)) break; // busy bit [15] = 0
    }
    if (timeout == 0) uart_puts("WARN: Q8 FPGA timeout\n");

    // Read 48-bit signed accumulator from two 32-bit DDR words (zero-extended to 64b)
    // 6. Read results from result buffer (48-bit signed in 64-bit words, zero-extended)
    uint32_t* res32 = (uint32_t*)(uintptr_t)res_addr;
    for (int i = 0; i < nrows; i++) {
        uint32_t lo = res32[i * 2];
        uint32_t hi = res32[i * 2 + 1];
        uint64_t raw = (uint64_t)lo | ((uint64_t)hi << 32);
        if (raw & ((uint64_t)1 << 47)) raw |= 0xFFFF000000000000ULL; // sign-extend 48→64
        int32_t acc = (int32_t)(int64_t)raw;
        float inv = row_inv ? row_inv[i] : 1.0f;
        if (inv < 1e-10f) inv = 1.0f;
        y[row0 + i] += (float)acc * x_scale / inv;
    }
}

// ---- Q5_0 FPGA Path via HP descriptor chain ----
// Each tile: 4 rows × 896 cols = 56 FPGA blocks (48 bytes each, interleaved)
// FPGA block format (48 bytes):
//   [0:1]   core0_d[15:0]  (GGUF f16 scale for core0's current row+block)
//   [2:5]   core0_qh[31:0]
//   [6:21]  core0_qs[127:0]
//   [22:23] core1_d[15:0]  (GGUF f16 scale for core1's current row+block)
//   [24:27] core1_qh[31:0]
//   [28:43] core1_qs[127:0]
//   [44:47] padding
// Blocks 0-27 → row0/core0, row2/core1. Blocks 28-55 → row1/core0, row3/core1.
// Norm (4 × UQ8.8 LE uint16) at tile offset 2688 (8 bytes).
// Result: 4 rows × 8 bytes each = 32 bytes

static void fpga_q5_0_tile(const Tensor* A, int row0, const int16_t* x_q,
                           float* y, float x_scale)
{
    uint32_t desc_addr = DESC_CHAIN_BASE;
    uint32_t wt_addr = FPGA_WEIGHT_REFMT;
    uint32_t act_addr = FPGA_ACT_BUF;
    uint32_t res_addr = FPGA_RES_BUF;

    int cols = (int)A->cols;
    int stride_blocks = cols / 32;  // 28 for 896
    int nrows = Q5_TILE_ROWS;
    if (row0 + nrows > (int)A->rows) nrows = (int)A->rows - row0;

    // 1. Generate 56 FPGA blocks (48 bytes each) from GGUF 22-byte blocks
    // Interleave: blocks 0-27 use tile rows (0,2); blocks 28-55 use tile rows (1,3)
    for (int bi = 0; bi < Q5_TILE_BLOCKS; bi++) {
        int group = bi / 28;                    // 0 or 1 (within-tile row group)
        int blk_in_row = bi % 28;               // block index within row (0..27)
        int core0_model_row = row0 + group;      // core0 row for this FPGA block
        int core1_model_row = row0 + 2 + group;  // core1 row for this FPGA block

        uint8_t* fpga_blk = (uint8_t*)(uintptr_t)(wt_addr + bi * Q5_BLOCK_SIZE);

        // Core0's GGUF block
        if (core0_model_row < (int)A->rows) {
            uint64_t src_off = ((uint64_t)core0_model_row * stride_blocks + blk_in_row) * 22;
            memcpy(fpga_blk, A->data + src_off, 22);
        } else {
            memset(fpga_blk, 0, 22);
        }

        // Core1's GGUF block
        if (core1_model_row < (int)A->rows) {
            uint64_t src_off = ((uint64_t)core1_model_row * stride_blocks + blk_in_row) * 22;
            memcpy(fpga_blk + 22, A->data + src_off, 22);
        } else {
            memset(fpga_blk + 22, 0, 22);
        }

        // Zero padding
        memset(fpga_blk + 44, 0, 4);
    }

    // Compute row_norm (UQ8.8) for each of the 4 rows
    // norm_val = 32767 / max_abs, stored as uint16 LE (UQ8.8 format)
    float ri[4];
    for (int r = 0; r < nrows; r++) {
        float max_abs = 0.0f;
        int row = row0 + r;
        for (int bi = 0; bi < stride_blocks; bi++) {
            uint64_t bo = ((uint64_t)row * stride_blocks + bi) * 22;
            uint32_t qh = (uint32_t)A->data[bo+2] | ((uint32_t)A->data[bo+3]<<8) |
                          ((uint32_t)A->data[bo+4]<<16) | ((uint32_t)A->data[bo+5]<<24);
            float d = read_f16_p(A->data + bo);
            for (int wi = 0; wi < 32; wi++) {
                uint64_t j = wi < 16 ? wi : wi - 16;
                uint8_t ql = (wi < 16) ? (A->data[bo+6+j] & 0xF) : (A->data[bo+6+j] >> 4);
                uint8_t qh_bit = (qh >> wi) & 1;
                int q5 = ((qh_bit << 4) | ql) - 16;
                float ab = fabsf(d * (float)q5);
                if (ab > max_abs) max_abs = ab;
            }
        }
        // For out-of-range rows, use 1.0 norm (weights are zeroed, contributes nothing)
        max_abs = (max_abs < 1e-10f) ? 1.0f : max_abs;
        ri[r] = 32767.0f / max_abs;
    }

    // Write norm values at tile offset 2688 (4 × UQ8.8 LE uint16)
    for (int r = 0; r < 4; r++) {
        uint16_t uq = (uint16_t)(ri[r < nrows ? r : 0] * 256.0f + 0.5f);
        uint8_t* dst = (uint8_t*)(uintptr_t)(wt_addr + Q5_TILE_NORM_OFFSET + r * 2);
        dst[0] = uq & 0xFF;
        dst[1] = (uq >> 8) & 0xFF;
    }

    // 2. Write activations
    memcpy((void*)(uintptr_t)act_addr, x_q, cols * 2);

    // 3. Build descriptor
    Descriptor* d = (Descriptor*)(uintptr_t)desc_addr;
    d->next_addr = 0;
    d->weight_addr = wt_addr;
    d->act_addr = act_addr;
    d->result_addr = res_addr;
    d->tensor_type = DESC_Q5_0;
    d->num_groups = 0;
    int act_bytes = cols * 2;
    d->act_total_bytes[0] = act_bytes & 0xFF;
    d->act_total_bytes[1] = (act_bytes >> 8) & 0xFF;
    d->act_total_bytes[2] = (act_bytes >> 16) & 0xFF;

    // 4. Start
    reg_write32(REG_DESC_BASE, desc_addr);
    __asm__ volatile("dsb" ::: "memory");
    reg_write32(REG_START, 1);

    // 5. Poll
    uint32_t timeout = 10000000;
    while (timeout--) {
        uint32_t status = reg_read32(REG_STATUS);
        if (!(status & 0x8000)) break;
    }
    if (timeout == 0) uart_puts("WARN: Q5_0 FPGA timeout\n");

    // 6. Read results (48-bit signed in 64-bit words, zero-extended)
    uint32_t* res32 = (uint32_t*)(uintptr_t)res_addr;
    for (int i = 0; i < nrows; i++) {
        uint32_t lo = res32[i * 2];
        uint32_t hi = res32[i * 2 + 1];
        uint64_t raw = (uint64_t)lo | ((uint64_t)hi << 32);
        if (raw & ((uint64_t)1 << 47)) raw |= 0xFFFF000000000000ULL;
        int32_t acc = (int32_t)(int64_t)raw;
        float inv = ri[i];
        if (inv < 1e-10f) inv = 1.0f;
        y[row0 + i] += (float)acc * x_scale / inv;
    }
}

// ---- Top-level matmul dispatch ----
// For Q8_0 and Q5_0: use FPGA via HP descriptor chain
// For Q4_K, Q6_K, F32, F16: CPU fallback
static void matmul(const Tensor* A, const float* x, float* y, int rows, int cols) {
    // Quantize activations to INT16
    float x_scale = 0.0f;
    for (int j = 0; j < cols; j++) {
        float a = fabsf(x[j]);
        if (a > x_scale) x_scale = a;
    }
    x_scale = (x_scale < 1e-10f) ? 1.0f : x_scale / 32767.0f;

    // Allocate x_q on stack (896 × 2 = 1792 bytes)
    int16_t x_q[896];

    for (int j = 0; j < cols; j++) {
        float v = x[j] / x_scale;
        if (v >= 32767.0f) x_q[j] = 32767;
        else if (v <= -32768.0f) x_q[j] = -32768;
        else x_q[j] = (int16_t)(v + (v >= 0 ? 0.5f : -0.5f));
    }

    memset(y, 0, rows * sizeof(float));

    if (A->type == TENSOR_Q8_0) {
        // FPGA Q8_0 path: 64-row tiles
        int tile_rows = Q8_TILE_ROWS;
        float row_inv_buf[64];

        for (int r0 = 0; r0 < rows; r0 += tile_rows) {
            int nr = min_int(tile_rows, rows - r0);

            // Pre-process tile weights
            q8_preprocess_tile(A, r0, (uint8_t*)(uintptr_t)FPGA_WEIGHT_REFMT);

            // Compute row_inv for this tile
            for (int i = 0; i < nr; i++) {
                int row = r0 + i;
                float max_abs = 0.0f;
                for (int j = 0; j < cols; j++) {
                    float v = fabsf(dequant(A, (uint64_t)row * cols + j));
                    if (v > max_abs) max_abs = v;
                }
                max_abs = (max_abs < 1e-10f) ? 1.0f : max_abs;
                row_inv_buf[i] = 32767.0f / max_abs;
            }
            for (int i = nr; i < tile_rows; i++) row_inv_buf[i] = 1.0f;

            // Run FPGA tile
            fpga_q8_tile((uint8_t*)(uintptr_t)FPGA_WEIGHT_REFMT,
                        x_q, y, r0, x_scale, row_inv_buf, nr);
        }
    }
    else if (A->type == TENSOR_Q5_0) {
        // FPGA Q5_0 path: 4-row tiles
        for (int r0 = 0; r0 < rows; r0 += Q5_TILE_ROWS) {
            fpga_q5_0_tile(A, r0, x_q, y, x_scale);
        }
    }
    else {
        // CPU fallback for Q4_K, Q6_K, F32, F16
        cpu_matmul(A, x, y, rows, cols);
    }
}

// ====================================================================
// ATTENTION with KV Cache
// ====================================================================
static void attention_forward(float* context, float* q_vec, int layer, int pos) {
    memset(context, 0, HIDDEN_DIM * sizeof(float));
    int q_per_kv = NUM_HEADS / NUM_KV_HEADS;

    for (int qh = 0; qh < NUM_HEADS; qh++) {
        int kv = qh / q_per_kv;
        float* qh_data = q_vec + qh * HEAD_DIM;
        float* ctx_h = context + qh * HEAD_DIM;

        // Compute scores
        float scores[MAX_SEQ_LEN];
        for (int p = 0; p <= pos; p++) {
            float* k_cached = g_state.k_cache[layer][p] + kv * HEAD_DIM;
            float s = 0.0f;
            for (int d = 0; d < HEAD_DIM; d++)
                s += qh_data[d] * k_cached[d];
            scores[p] = s / sqrtf((float)HEAD_DIM);
        }

        // Softmax
        float max_s = scores[0];
        for (int p = 1; p <= pos; p++)
            if (scores[p] > max_s) max_s = scores[p];
        float sum_exp = 0.0f;
        for (int p = 0; p <= pos; p++)
            sum_exp += expf(scores[p] - max_s);
        float log_sum = logf(sum_exp) + max_s;

        // Weighted sum of values
        for (int p = 0; p <= pos; p++) {
            float* v_cached = g_state.v_cache[layer][p] + kv * HEAD_DIM;
            float w = expf(scores[p] - log_sum);
            for (int d = 0; d < HEAD_DIM; d++)
                ctx_h[d] += w * v_cached[d];
        }
    }
}

// ===== Minimal name formatter (no snprintf) =====
static int fmt_name(char* buf, int maxlen, int layer, const char* suffix) {
    const char* prefix = "blk.";
    int pos = 0;
    while (*prefix && pos < maxlen - 1) buf[pos++] = *prefix++;
    int l = layer;
    char digits[4]; int nd = 0;
    if (l == 0) digits[nd++] = '0';
    else { while (l && nd < 4) { digits[nd++] = '0' + (l % 10); l /= 10; } }
    for (int i = nd - 1; i >= 0 && pos < maxlen - 1; i--) buf[pos++] = digits[i];
    if (pos < maxlen - 1) buf[pos++] = '.';
    while (*suffix && pos < maxlen - 1) buf[pos++] = *suffix++;
    buf[pos] = 0;
    return pos;
}

// ====================================================================
// FORWARD LAYER
// ====================================================================
static void forward_layer(float* hidden, int layer, int pos) {
    char name[128];
    Tensor* t;
    float original_hidden[HIDDEN_DIM];
    memcpy(original_hidden, hidden, HIDDEN_DIM * sizeof(float));

    float attn_norm_out[HIDDEN_DIM];
    float q_vec[HIDDEN_DIM];
    float k_new[K_DIM];
    float v_new[V_DIM];
    float context[HIDDEN_DIM];
    float attn_out[HIDDEN_DIM];
    float ffn_norm_out[HIDDEN_DIM];
    float gate[INTER_DIM];
    float up[INTER_DIM];
    float ffn_out[HIDDEN_DIM];

    // Attention norm
    {
        fmt_name(name, 128, layer, "attn_norm.weight");
        t = get_tensor(name);
        if (t) rms_norm(attn_norm_out, original_hidden, HIDDEN_DIM, t);
        else { memcpy(attn_norm_out, original_hidden, HIDDEN_DIM * sizeof(float)); }
    }

    // Q
    {
        fmt_name(name, 128, layer, "attn_q.weight");
        t = get_tensor(name);
        if (t) matmul(t, attn_norm_out, q_vec, HIDDEN_DIM, HIDDEN_DIM);
        fmt_name(name, 128, layer, "attn_q.bias");
        t = get_tensor(name);
        if (t) { float* b = (float*)t->data; for (int i = 0; i < HIDDEN_DIM; i++) q_vec[i] += b[i]; }
    }

    // K
    {
        fmt_name(name, 128, layer, "attn_k.weight");
        t = get_tensor(name);
        if (t) matmul(t, attn_norm_out, k_new, K_DIM, HIDDEN_DIM);
        fmt_name(name, 128, layer, "attn_k.bias");
        t = get_tensor(name);
        if (t) { float* b = (float*)t->data; for (int i = 0; i < K_DIM; i++) k_new[i] += b[i]; }
    }

    // V
    {
        fmt_name(name, 128, layer, "attn_v.weight");
        t = get_tensor(name);
        if (t) matmul(t, attn_norm_out, v_new, V_DIM, HIDDEN_DIM);
        fmt_name(name, 128, layer, "attn_v.bias");
        t = get_tensor(name);
        if (t) { float* b = (float*)t->data; for (int i = 0; i < V_DIM; i++) v_new[i] += b[i]; }
    }

    // RoPE
    apply_rope(q_vec, k_new, pos);

    // Store to KV cache
    for (int i = 0; i < K_DIM; i++) g_state.k_cache[layer][pos][i] = k_new[i];
    for (int i = 0; i < V_DIM; i++) g_state.v_cache[layer][pos][i] = v_new[i];

    // Attention
    attention_forward(context, q_vec, layer, pos);

    // Attention output projection
    {
        fmt_name(name, 128, layer, "attn_output.weight");
        t = get_tensor(name);
        if (t) matmul(t, context, attn_out, HIDDEN_DIM, HIDDEN_DIM);
    }

    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] = original_hidden[i] + attn_out[i];

    // FFN norm
    {
        fmt_name(name, 128, layer, "ffn_norm.weight");
        t = get_tensor(name);
        if (t) rms_norm(ffn_norm_out, hidden, HIDDEN_DIM, t);
        else { memcpy(ffn_norm_out, hidden, HIDDEN_DIM * sizeof(float)); }
    }

    // FFN gate
    {
        fmt_name(name, 128, layer, "ffn_gate.weight");
        t = get_tensor(name);
        if (t) matmul(t, ffn_norm_out, gate, INTER_DIM, HIDDEN_DIM);
    }

    // FFN up
    {
        fmt_name(name, 128, layer, "ffn_up.weight");
        t = get_tensor(name);
        if (t) matmul(t, ffn_norm_out, up, INTER_DIM, HIDDEN_DIM);
    }

    // SwiGLU
    silu_forward(gate, gate, INTER_DIM);
    for (int i = 0; i < INTER_DIM; i++) gate[i] *= up[i];

    // FFN down
    {
        fmt_name(name, 128, layer, "ffn_down.weight");
        t = get_tensor(name);
        if (t) matmul(t, gate, ffn_out, HIDDEN_DIM, INTER_DIM);
    }

    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] += ffn_out[i];
}

// ====================================================================
// LOGITS
// ====================================================================
static void get_logits(float* logits, const float* hidden) {
    Tensor* emb_t = get_tensor("token_embd.weight");
    if (!emb_t) { uart_puts("ERROR: No token_embd.weight\n"); return; }

    Tensor* norm_t = get_tensor("output_norm.weight");
    float norm_hidden[HIDDEN_DIM];
    if (norm_t) {
        rms_norm(norm_hidden, hidden, HIDDEN_DIM, norm_t);
        matmul(emb_t, norm_hidden, logits, VOCAB_SIZE, HIDDEN_DIM);
    } else {
        matmul(emb_t, hidden, logits, VOCAB_SIZE, HIDDEN_DIM);
    }
}

// ====================================================================
// SAMPLING
// ====================================================================
static int sample_token(float* logits, int top_k) {
    // Simple argmax for now (deterministic)
    int best = 0;
    float best_val = logits[0];
    for (int i = 1; i < VOCAB_SIZE; i++) {
        if (logits[i] > best_val) { best_val = logits[i]; best = i; }
    }
    return best;
}

// ====================================================================
// EMBEDDING
// ====================================================================
static void process_embedding(float* hidden, int token_id) {
    Tensor* emb_t = get_tensor("token_embd.weight");
    if (!emb_t) { memset(hidden, 0, HIDDEN_DIM * sizeof(float)); return; }
    for (int i = 0; i < HIDDEN_DIM; i++)
        hidden[i] = dequant(emb_t, (uint64_t)token_id * HIDDEN_DIM + i);
}

// ====================================================================
// INFERENCE LOOP
// ====================================================================
static void reset_kv_cache() {
    g_state.seq_len = 0;
    memset(g_state.k_cache, 0, sizeof(g_state.k_cache));
    memset(g_state.v_cache, 0, sizeof(g_state.v_cache));
}

static int run_inference(const int* prompt_tokens, int n_prompt) {
    reset_kv_cache();

    // Process prompt tokens (prefill phase)
    uart_puts("Processing prompt...\n");
    for (int t = 0; t < n_prompt; t++) {
        process_embedding(g_state.hidden, prompt_tokens[t]);
        for (int layer = 0; layer < NUM_LAYERS; layer++)
            forward_layer(g_state.hidden, layer, t);
    }
    g_state.seq_len = n_prompt;

    // Generate tokens
    uart_puts("Generating...\n");
    g_state.n_output_tokens = 0;

    // First token: get logits from last hidden state
    get_logits(g_state.logits, g_state.hidden);
    int token = sample_token(g_state.logits, 40);
    g_state.output_tokens[g_state.n_output_tokens++] = token;

    // Generate more tokens
    for (int gen = 0; gen < 10; gen++) {
        int pos = n_prompt + gen;
        if (pos >= MAX_SEQ_LEN) break;

        process_embedding(g_state.hidden, token);
        for (int layer = 0; layer < NUM_LAYERS; layer++)
            forward_layer(g_state.hidden, layer, pos);

        get_logits(g_state.logits, g_state.hidden);
        token = sample_token(g_state.logits, 40);
        g_state.output_tokens[g_state.n_output_tokens++] = token;

        if (token == 151643) break; // EOS token
    }

    return g_state.n_output_tokens;
}

// ====================================================================
// MAIN
// ====================================================================
// Prompt tokens come from a fixed DDR buffer (written by boot script)

extern "C" int main(void) {
    // Enable FPU (VFPv3 on Cortex-A9)
    __asm__ volatile(
        "mrc p15, 0, r0, c1, c0, 2\n"
        "orr r0, r0, #(0xF << 20)\n"
        "mcr p15, 0, r0, c1, c0, 2\n"
        "dsb\nisb\n"
        "mov r3, #0x40000000\n"
        "vmsr fpexc, r3\n"
        ::: "r0", "r3", "memory");
    uart_init();
    uart_puts("\n=== T-MAC Bare-Metal Inference Engine ===\n");
    uart_puts("Zynq 7010 | FPGA Q8_0 + Q5_0 | CPU fallback\n\n");

    // Initialize state
    memset(&g_state, 0, sizeof(g_state));
    reg_write32(REG_Q8_NUM_GROUPS, 14);
    uart_puts("FPGA Q8_NUM_GROUPS=14\n");

    // Load model from DDR
    if (load_tmac(MODEL_BASE) != 0) {
        uart_puts("FATAL: Model load failed\n");
        return -1;
    }

    // Check prompt buffer (at 0x1F001000 = DESC_CHAIN_BASE, first word = count)
    uint32_t* prompt_buf = (uint32_t*)(uintptr_t)0x1F001000UL;
    int n_prompt = (int)prompt_buf[0];
    uint32_t* prompt_tokens_buf = &prompt_buf[1];

    // For now, use a fixed test token or read from DDR
    if (n_prompt > 0 && n_prompt < 1000) {
        // Read prompt from DDR buffer
        int prompt_tokens[256];
        int np = n_prompt > 256 ? 256 : n_prompt;
        for (int i = 0; i < np; i++) prompt_tokens[i] = (int)prompt_tokens_buf[i];
        uart_puts("Prompt: ");
        for (int i = 0; i < np && i < 10; i++) { uart_putdec(prompt_tokens[i]); uart_putc(' '); }
        if (np > 10) uart_puts("...");
        uart_puts("\n");

        run_inference(prompt_tokens, np);
    } else {
        // Default: use token 0 as single-token prompt for testing
        uart_puts("No prompt buffer found, using token 0\n");
        int test_prompt[] = {0};  // BOS-like single token
        run_inference(test_prompt, 1);
    }

    // Write output tokens to DDR
    uint32_t* output = (uint32_t*)(uintptr_t)OUTPUT_BUF;
    output[0] = (uint32_t)g_state.n_output_tokens;
    for (int i = 0; i < g_state.n_output_tokens; i++)
        output[1 + i] = (uint32_t)g_state.output_tokens[i];

    uart_puts("\n=== Inference Complete ===\n");
    uart_puts("Generated "); uart_putdec(g_state.n_output_tokens); uart_puts(" tokens\n");
    uart_puts("Tokens: ");
    for (int i = 0; i < g_state.n_output_tokens; i++) {
        uart_putdec(g_state.output_tokens[i]);
        uart_putc(' ');
    }
    uart_puts("\nOutput written to 0x1F000000\n");

    return 0;
}
