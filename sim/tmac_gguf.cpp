// T-MAC Inference for Zynq 7010 - KV Cache Version

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <chrono>
#include "fpga_sim.hpp"
#include "Transaction Tracer/fpga_profiler.hpp"

// ===========================================================================
// CHROME TRACE EVENT PROFILER
// ===========================================================================
struct TraceEvent {
    const char* name;
    int64_t ts_us;
    char ph; // 'B' = begin, 'E' = end
};

static std::vector<TraceEvent> g_trace_events;
static bool g_perf_enabled = false;

static int64_t now_us() {
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(
        now.time_since_epoch()).count();
}

static void trace_begin(const char* name) {
    if (!g_perf_enabled) return;
    g_trace_events.push_back({name, now_us(), 'B'});
}

static void trace_end(const char* name) {
    if (!g_perf_enabled) return;
    g_trace_events.push_back({name, now_us(), 'E'});
}

struct TraceScope {
    const char* name_;
    TraceScope(const char* name) : name_(name) { trace_begin(name_); }
    ~TraceScope() { trace_end(name_); }
};

#define PROFILE_SCOPE(name) TraceScope CONCAT(_trace_, __LINE__)(name)
#define CONCAT_(a, b) a ## b
#define CONCAT(a, b) CONCAT_(a, b)

static void dump_chrome_trace(const char* path) {
    if (!g_perf_enabled || g_trace_events.empty()) return;
    FILE* f = fopen(path, "w");
    if (!f) return;
    fprintf(f, "{\"traceEvents\":[\n");
    for (size_t i = 0; i < g_trace_events.size(); i++) {
        auto& e = g_trace_events[i];
        fprintf(f, "{\"ph\":\"%c\",\"name\":\"%s\",\"cat\":\"pipeline\","
                "\"ts\":%lld,\"pid\":1,\"tid\":1}",
                e.ph, e.name, (long long)e.ts_us);
        if (i + 1 < g_trace_events.size()) fprintf(f, ",");
        fprintf(f, "\n");
    }
    fprintf(f, "]}\n");
    fclose(f);
    printf("[TRACE] Wrote Chrome trace: %s\n", path);
}

struct TraceAgg {
    const char* name;
    int64_t total_us;
    int64_t count;
};

static void print_trace_summary() {
    if (g_trace_events.empty()) return;

    // Build name → total duration map from begin/end pairs
    std::vector<TraceAgg> agg;
    auto find_agg = [&](const char* name) -> TraceAgg* {
        for (auto& a : agg) if (strcmp(a.name, name) == 0) return &a;
        return nullptr;
    };

    // Stack-based pairing
    struct Frame { const char* name; int64_t start_us; };
    Frame stack[64];
    int depth = 0;

    for (auto& e : g_trace_events) {
        if (e.ph == 'B') {
            if (depth < 64) stack[depth++] = {e.name, e.ts_us};
        } else if (e.ph == 'E' && depth > 0) {
            depth--;
            auto* a = find_agg(stack[depth].name);
            int64_t dur = e.ts_us - stack[depth].start_us;
            if (a) { a->total_us += dur; a->count++; }
            else agg.push_back({stack[depth].name, dur, 1});
        }
    }

    if (agg.empty()) return;

    // Sort by total time descending
    std::sort(agg.begin(), agg.end(),
              [](const TraceAgg& a, const TraceAgg& b) { return a.total_us > b.total_us; });

    int64_t total_us = 0;
    for (auto& a : agg) total_us += a.total_us;

    printf("\n[PERF — PIPELINE TIMING BREAKDOWN]\n");
    printf("  %-20s %12s %8s %10s %8s\n", "Operation", "Time (ms)", "Calls", "Avg (ms)", "Share");
    printf("  %s\n", std::string(65, '-').c_str());
    for (auto& a : agg) {
        double ms = a.total_us / 1000.0;
        double avg = ms / a.count;
        double pct = a.total_us * 100.0 / total_us;
        printf("  %-20s %10.2f %8lld %8.4f %6.1f%%\n",
               a.name, ms, (long long)a.count, avg, pct);
    }
    printf("  %s\n", std::string(65, '-').c_str());
    printf("  %-20s %10.2f\n", "TOTAL", total_us / 1000.0);

    if (!agg.empty()) {
        printf("\n  >> Hottest: %s (%.1f ms, %.1f%%)\n",
               agg[0].name, agg[0].total_us / 1000.0,
               agg[0].total_us * 100.0 / total_us);
    }
    printf("\n");
}

// ===========================================================================
// CONFIG - Qwen2-0.5B
// ===========================================================================
constexpr int HIDDEN_DIM = 896;
constexpr int INTER_DIM = 4864;
constexpr int VOCAB_SIZE = 151936;
constexpr int NUM_LAYERS = 24;
constexpr int NUM_HEADS = 14;
constexpr int HEAD_DIM = 64;
constexpr int NUM_KV_HEADS = 2;
constexpr int K_DIM = NUM_KV_HEADS * HEAD_DIM;  // 128
constexpr int V_DIM = K_DIM;
constexpr int MAX_SEQ_LEN = 256;

constexpr size_t DDR_SIZE = 512 * 1024 * 1024;

// Tensor types
constexpr int TENSOR_F32 = 0;
constexpr int TENSOR_F16 = 1;
constexpr int TENSOR_Q8_0 = 8;
constexpr int TENSOR_Q6_K = 14;
constexpr int TENSOR_Q5_0 = 6;
constexpr int TENSOR_Q4_K = 12;

// ===========================================================================
// TENSOR STORAGE
// ===========================================================================
struct Tensor {
    char name[128];
    uint64_t rows, cols;
    uint32_t type;
    uint64_t n_bytes;
    uint8_t* data;
};

static Tensor* g_tensors = nullptr;
static int g_ntensors = 0;
static uint8_t* g_ddr = nullptr;

// KV Cache
static float g_k_cache[NUM_LAYERS][MAX_SEQ_LEN][K_DIM];
static float g_v_cache[NUM_LAYERS][MAX_SEQ_LEN][V_DIM];
static int g_seq_len = 0;
static bool g_use_fpga = false;
static bool g_fpga_int16 = false;
static bool g_fpga_int16_q8path = false;
static bool g_fpga_q8 = false;

// ===========================================================================
// F16 TO F32
// ===========================================================================
float f16_to_f32(uint16_t f16) {
    uint32_t sign = (f16 >> 15) & 0x1;
    uint32_t exp = (f16 >> 10) & 0x1F;
    uint32_t mant = f16 & 0x3FF;
    if (exp == 0) {
        if (mant == 0) return 0.0f;
        return (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
    } else if (exp == 31) {
        return mant ? NAN : (sign ? -INFINITY : INFINITY);
    } else {
        return (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, (int)exp - 15);
    }
}

// ===========================================================================
// DEQUANTIZATION
// ===========================================================================

// Q8_0 block structure: [scale:FP16 (2B)][val:INT8 (32B)] per 32 elements
// Returns float value at element index idx
float dequant_q8_0(const uint8_t* data, uint64_t idx) {
    uint64_t block = idx / 32;
    uint64_t offset = idx % 32;
    uint64_t block_offset = block * 34;
    uint16_t scale_raw = data[block_offset] | (data[block_offset + 1] << 8);
    float scale = f16_to_f32(scale_raw);
    int8_t val = (int8_t)data[block_offset + 2 + offset];
    return val * scale;
}

// FPGA Q8_0→INT16 dequantization: converts raw Q8_0 bytes to INT16 directly
// This simulates what the FPGA would do: per-32-element block, compute
// scale = max_abs / 32767, then val_int16 = round(val_q8 * 127 / max_abs)
// The FPGA would stream Q8_0 bytes and output INT16 without ever going to float.
// Returns the INT16 dequantized value.
inline fpga_sim::in16_t dequant_q8_to_int16(const uint8_t* data, uint64_t idx) {
    uint64_t block = idx / 32;
    uint64_t offset = idx % 32;
    uint64_t block_offset = block * 34;
    // Read scale (FP16) — same as Q8_0 decoder
    uint16_t scale_raw = data[block_offset] | (data[block_offset + 1] << 8);
    (void)scale_raw;  // not used in FPGA path (FPGA recomputes max_abs per row)
    int8_t val_q8 = (int8_t)data[block_offset + 2 + offset];
    // Convert: Q8_0 is already scaled by scale, we want INT16 with new scale
    // New INT16 scale = max_abs / 32767 where max_abs comes from max Q8_0 value
    // Since Q8_0 stores val * scale, and we need val * new_scale where
    // new_scale = max_abs(Q8_0 values in block) / 32767
    // We approximate: scale_factor = 1.0 / scale * (max_abs / 32767)
    // For simplicity, FPGA would compute: val_int16 = round(val_q8 * (127/127))
    // which is just val_q8 directly if the scale is absorbed into the matmul scale.
    // Simpler still: the FPGA computes scale_inv = 32767.0 / computed_max
    // and multiplies each Q8_0 value directly.
    // Here we do the direct conversion: val * scale / new_scale, clamped to INT16 range.
    float max_abs = fabsf((float)val_q8);
    for (uint64_t i = 0; i < 32; i++) {
        int8_t v = (int8_t)data[block_offset + 2 + i];
        float a = fabsf((float)v);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1.0f;
    float scale_factor = max_abs / 32767.0f;
    float val_f = (float)val_q8;  // Q8_0 val already in -127..127 range
    int16_t val_i16 = (int16_t)roundf(val_f / scale_factor);
    return val_i16;
}

// Per-block Q8_0→INT16 scale factors (for batch dequantization)
// Returns: scale_int16 = max_abs(Q8 block) / 32767 encoded as fp16-like value
// and the Q8_0 byte offset for the block
struct Q8BlockInfo {
    float scale_factor;  // max_abs / 32767 for this block
    uint64_t offset;     // byte offset to block data
};

// Get Q8_0 block info for a given element index
inline Q8BlockInfo get_q8_block_info(const uint8_t* data, uint64_t idx) {
    uint64_t block = idx / 32;
    uint64_t block_offset = block * 34;
    uint16_t scale_raw = data[block_offset] | (data[block_offset + 1] << 8);
    float scale = f16_to_f32(scale_raw);
    float max_abs = 0;
    for (uint64_t i = 0; i < 32; i++) {
        int8_t v = (int8_t)data[block_offset + 2 + i];
        float a = fabsf((float)v);
        if (a > max_abs) max_abs = a;
    }
    if (max_abs < 1e-10f) max_abs = 1.0f;
    float scale_factor = max_abs / 32767.0f;
    return {scale_factor, block_offset + 2};  // offset points to first INT8 value
}

float dequant_q5_0(const uint8_t* data, uint64_t idx) {
    uint64_t block = idx / 32;
    uint64_t offset = idx % 32;
    uint64_t base = block * 22;
    uint16_t d_raw = data[base] | (data[base + 1] << 8);
    float d = f16_to_f32(d_raw);
    uint32_t qh = data[base + 2] | (data[base + 3] << 8) |
                  (data[base + 4] << 16) | (data[base + 5] << 24);
    // ggml Q5_0: pairs elements (j, j+16) per qs byte
    // qs[j] lower nibble = element j, upper nibble = element j+16
    // qh bit j = high bit for element j, qh bit j+12 = high bit for element j+16
    uint64_t j = offset < 16 ? offset : offset - 16;
    uint8_t qs_byte = data[base + 6 + j];
    uint8_t ql = (offset < 16) ? (qs_byte & 0xF) : (qs_byte >> 4);
    uint8_t qh_bit = (qh >> offset) & 1;
    int8_t q = ((qh_bit << 4) | ql) - 16;
    return d * (float)q;
}

float dequant_q6_k(const uint8_t* data, uint64_t idx) {
    const uint64_t QK_K = 256;
    const uint64_t block = idx / QK_K;
    const uint64_t offset = idx % QK_K;
    const uint64_t base = block * 210;
    uint16_t d_raw = data[base + 208] | (data[base + 209] << 8);
    float d = f16_to_f32(d_raw);
    uint64_t half = offset / 128;
    uint64_t pos = offset % 128;
    uint64_t l = pos % 32;
    uint64_t sub = pos / 32;
    uint64_t ql_base = base + half * 64;
    uint64_t qh_base = base + 128 + half * 32;
    uint64_t sc_base = base + 192 + half * 8;
    uint8_t ql_byte;
    uint8_t qh_shift;
    if (sub == 0) {
        ql_byte = data[ql_base + l];
        qh_shift = 0;
    } else if (sub == 1) {
        ql_byte = data[ql_base + l + 32];
        qh_shift = 2;
    } else if (sub == 2) {
        ql_byte = data[ql_base + l];
        qh_shift = 4;
    } else {
        ql_byte = data[ql_base + l + 32];
        qh_shift = 6;
    }
    uint8_t ql_nibble = (sub == 0 || sub == 1) ? (ql_byte & 0xF) : (ql_byte >> 4);
    uint8_t qh_bits = (data[qh_base + l] >> qh_shift) & 0x3;
    int8_t q = (int8_t)((qh_bits << 4) | ql_nibble) - 32;
    uint64_t is = l / 16;
    uint64_t scale_idx = is + sub * 2;
    int8_t scale = (int8_t)data[sc_base + scale_idx];
    return d * scale * q;
}

float dequant_q4_k(const uint8_t* data, uint64_t idx) {
    const uint64_t QK_K = 256;
    const uint64_t block = idx / QK_K;
    const uint64_t offset = idx % QK_K;
    const uint64_t base = block * 144;

    uint16_t d_raw = data[base] | (data[base + 1] << 8);
    float d = f16_to_f32(d_raw);
    uint16_t dmin_raw = data[base + 2] | (data[base + 3] << 8);
    float dmin = f16_to_f32(dmin_raw);

    uint64_t sub_block = offset / 32;
    uint64_t q_pos = offset % 32;
    uint64_t qs_byte_idx = (sub_block / 2) * 32 + q_pos;

    uint8_t qs_byte = data[base + 16 + qs_byte_idx];
    uint8_t q4 = (sub_block % 2 == 0) ? (qs_byte & 0xF) : (qs_byte >> 4);

    const uint8_t* scales = data + base + 4;
    uint8_t sc, m;
    if (sub_block < 4) {
        sc = scales[sub_block] & 63;
        m = scales[sub_block + 4] & 63;
    } else {
        sc = (scales[sub_block + 4] & 0xF) | ((scales[sub_block - 4] >> 6) << 4);
        m = (scales[sub_block + 4] >> 4) | ((scales[sub_block] >> 6) << 4);
    }

    return d * (float)sc * (float)q4 - dmin * (float)m;
}

// ===========================================================================
// TENSOR ACCESS
// ===========================================================================
float read_tensor(const Tensor* t, uint64_t row, uint64_t col) {
    // GGUF/GGML stores 2D weights column-major: ne[0]=input_dim, ne[1]=output_dim
    // Element at logical W[output][input] is at flat index: input + output * input_dim
    // t->cols = ne[0] = input_dim, t->rows = ne[1] = output_dim
    // For 1D tensors (cols==1): flat index = row
    uint64_t idx = (t->cols == 1) ? row : (col + row * t->cols);
    if (t->type == TENSOR_F32) return ((float*)t->data)[idx];
    if (t->type == TENSOR_F16) return f16_to_f32(((uint16_t*)t->data)[idx]);
    if (t->type == TENSOR_Q8_0) return dequant_q8_0(t->data, idx);
    if (t->type == TENSOR_Q5_0) return dequant_q5_0(t->data, idx);
    if (t->type == TENSOR_Q6_K) return dequant_q6_k(t->data, idx);
    if (t->type == TENSOR_Q4_K) return dequant_q4_k(t->data, idx);
    return 0.0f;
}

float read_embedding(const Tensor* t, uint64_t token_id, uint64_t feature) {
    // GGUF token_embd.weight: shape [vocab_size, hidden_dim] = [151936, 896]
    // GGML stores ne[0]=896 (hidden), ne[1]=151936 (vocab)
    // ggml_get_rows interprets ne[1] as rows — row token_id starts at token_id*896
    // Element (token_id, feature) = data[token_id * hidden_dim + feature]
    // t->cols = 151936 (vocab_size), but we use HIDDEN_DIM=896 for stride
    uint64_t idx = token_id * HIDDEN_DIM + feature;
    if (t->type == TENSOR_Q8_0) return dequant_q8_0(t->data, idx);
    if (t->type == TENSOR_F32) return ((float*)t->data)[idx];
    if (t->type == TENSOR_F16) return f16_to_f32(((uint16_t*)t->data)[idx]);
    if (t->type == TENSOR_Q5_0) return dequant_q5_0(t->data, idx);
    if (t->type == TENSOR_Q6_K) return dequant_q6_k(t->data, idx);
    if (t->type == TENSOR_Q4_K) return dequant_q4_k(t->data, idx);
    return 0.0f;
}

Tensor* get_tensor_info(const char* name) {
    for (int i = 0; i < g_ntensors; i++) {
        if (strcmp(g_tensors[i].name, name) == 0) return &g_tensors[i];
    }
    return nullptr;
}

extern void q8_logits_matmul_with_tensor(const Tensor* emb_t, const float* x, float* y, int rows, int cols);

// ===========================================================================
// RMS NORM
// ===========================================================================
float rms(const float* x, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    return sqrtf(sum / n);
}

void rms_norm(float* o, const float* x, int n, const Tensor* t) {
    float sum = 0;
    for (int i = 0; i < n; i++) sum += x[i] * x[i];
    float mean = sum / n;
    float scale = 1.0f / sqrtf(mean + 1e-6f);
    float* w = (float*)t->data;
    for (int i = 0; i < n; i++) o[i] = x[i] * w[i] * scale;
}

// ===========================================================================
// SILU
// ===========================================================================
void silu(float* y, const float* x, int n) {
    for (int i = 0; i < n; i++) y[i] = x[i] / (1.0f + expf(-x[i]));
}

// ===========================================================================
// MATMUL
// ===========================================================================
void matmul_fpga(const Tensor* A, const float* x, float* y, int rows, int cols);
void matmul_fpga_int16(const Tensor* A, const float* x, float* y, int rows, int cols);
void matmul_fpga_q8(const Tensor* A, const float* x, float* y, int rows, int cols, const float* row_max_abs = nullptr);
void get_logits_q8(float* logits, const float* hidden);

void matmul(const Tensor* A, const float* x, float* y, int rows, int cols) {
    if (g_use_fpga) {
        if (g_fpga_q8 && A->type == TENSOR_Q8_0) {
            matmul_fpga_q8(A, x, y, rows, cols);
        } else if (g_fpga_int16) {
            matmul_fpga_int16(A, x, y, rows, cols);
        } else {
            matmul_fpga(A, x, y, rows, cols);
        }
        return;
    }
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        for (int j = 0; j < cols; j++) sum += read_tensor(A, i, j) * x[j];
        y[i] = sum;
    }
}

void matmul_transposed(const Tensor* A, const float* x, float* y, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float sum = 0;
        for (int j = 0; j < cols; j++) sum += read_tensor(A, j, i) * x[j];
        y[i] = sum;
    }
}

// ===========================================================================
// FPGA MATMUL (SIMULATED INT8 SYSTOLIC ACCELERATOR)
// ===========================================================================
void matmul_fpga(const Tensor* A, const float* x, float* y, int rows, int cols) {
    auto t0 = std::chrono::high_resolution_clock::now();
    memset(y, 0, rows * sizeof(float));

    float x_scale = 0;
    for (int j = 0; j < cols; j++) x_scale = fmaxf(x_scale, fabsf(x[j]));
    x_scale = (x_scale < 1e-10f) ? 1.0f : x_scale / 127.0f;

    std::vector<fpga_sim::in_t> x_q(cols);
    for (int j = 0; j < cols; j++)
        x_q[j] = (fpga_sim::in_t)roundf(x[j] / x_scale);

    fpga_sim::g_timing.total_mac_ops += (int64_t)rows * cols;

    for (int r = 0; r < rows; r += fpga_sim::N) {
        int r_size = std::min(fpga_sim::N, rows - r);

        std::vector<float> W_buf(r_size * cols);
        for (int i = 0; i < r_size; i++)
            for (int j = 0; j < cols; j++)
                W_buf[i * cols + j] = read_tensor(A, r + i, j);

        float row_scale[fpga_sim::N];
        for (int i = 0; i < r_size; i++) {
            float m = 0;
            for (int j = 0; j < cols; j++) m = fmaxf(m, fabsf(W_buf[i * cols + j]));
            row_scale[i] = (m < 1e-10f) ? 1.0f : m / 127.0f;
        }

        for (int c = 0; c < cols; c += fpga_sim::N) {
            int c_size = std::min(fpga_sim::N, cols - c);

            fpga_sim::in_t W_q[fpga_sim::N][fpga_sim::N] = {{0}};
            for (int i = 0; i < r_size; i++)
                for (int k = 0; k < c_size; k++)
                    W_q[k][i] = (fpga_sim::in_t)roundf(
                        W_buf[i * cols + c + k] / row_scale[i]);

            fpga_sim::in_t vec[fpga_sim::N] = {0};
            for (int k = 0; k < c_size; k++) vec[k] = x_q[c + k];

            fpga_sim::acc_t result[fpga_sim::N] = {0};
            fpga_sim::axi_vecmul_tile_int8(vec, W_q, result);

            for (int i = 0; i < r_size; i++)
                y[r + i] += (float)result[i] * x_scale * row_scale[i];

            fpga_sim::g_timing.total_tiles++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fpga_sim::g_timing.cpu_ms += ms;
}

// ===========================================================================
// FPGA MATMUL INT16 (HIGHER PRECISION SIMULATION)
// ===========================================================================
void matmul_fpga_int16(const Tensor* A, const float* x, float* y, int rows, int cols) {
    auto t0 = std::chrono::high_resolution_clock::now();
    memset(y, 0, rows * sizeof(float));

    float x_scale = 0;
    for (int j = 0; j < cols; j++) x_scale = fmaxf(x_scale, fabsf(x[j]));
    x_scale = (x_scale < 1e-10f) ? 1.0f : x_scale / 32767.0f;

    std::vector<fpga_sim::in16_t> x_q(cols);
    for (int j = 0; j < cols; j++)
        x_q[j] = (fpga_sim::in16_t)roundf(x[j] / x_scale);

    fpga_sim::g_timing.total_mac_ops += (int64_t)rows * cols;

    for (int r = 0; r < rows; r += fpga_sim::N) {
        int r_size = std::min(fpga_sim::N, rows - r);

        std::vector<float> W_buf(r_size * cols);
        for (int i = 0; i < r_size; i++)
            for (int j = 0; j < cols; j++)
                W_buf[i * cols + j] = read_tensor(A, r + i, j);

        float row_scale[fpga_sim::N];
        for (int i = 0; i < r_size; i++) {
            float m = 0;
            for (int j = 0; j < cols; j++) m = fmaxf(m, fabsf(W_buf[i * cols + j]));
            row_scale[i] = (m < 1e-10f) ? 1.0f : m / 32767.0f;
        }

        for (int c = 0; c < cols; c += fpga_sim::N) {
            int c_size = std::min(fpga_sim::N, cols - c);

            fpga_sim::in16_t W_q[fpga_sim::N][fpga_sim::N] = {{0}};
            for (int i = 0; i < r_size; i++)
                for (int k = 0; k < c_size; k++)
                    W_q[k][i] = (fpga_sim::in16_t)roundf(
                        W_buf[i * cols + c + k] / row_scale[i]);

            fpga_sim::in16_t vec[fpga_sim::N] = {0};
            for (int k = 0; k < c_size; k++) vec[k] = x_q[c + k];

            fpga_sim::acc16_t result[fpga_sim::N] = {0};
            fpga_sim::axi_vecmul_tile_int16(vec, W_q, result);

            for (int i = 0; i < r_size; i++)
                y[r + i] += (double)result[i] * x_scale * row_scale[i];

            fpga_sim::g_timing.total_tiles++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fpga_sim::g_timing.cpu_ms += ms;
}

// ===========================================================================
// FPGA MATMUL Q8_0 DIRECT PATH
// FPGA receives Q8_0 bytes + row scales, does Q8→INT16 internally.
// This moves the dequantization workload from CPU to FPGA.
// ===========================================================================
void matmul_fpga_q8(const Tensor* A, const float* x, float* y, int rows, int cols, const float* row_max_abs) {
    auto t0 = std::chrono::high_resolution_clock::now();
    memset(y, 0, rows * sizeof(float));

    constexpr int Q8_BLOCK_BYTES = 34;
    constexpr int Q8_BLOCK_SIZE = 32;

    float x_scale = 0;
    for (int j = 0; j < cols; j++) x_scale = fmaxf(x_scale, fabsf(x[j]));
    x_scale = (x_scale < 1e-10f) ? 1.0f : x_scale / 32767.0f;

    std::vector<fpga_sim::in16_t> x_q(cols);
    for (int j = 0; j < cols; j++)
        x_q[j] = (fpga_sim::in16_t)roundf(x[j] / x_scale);

    fpga_sim::g_timing.total_mac_ops += (int64_t)rows * cols;

    for (int r = 0; r < rows; r += fpga_sim::N) {
        int r_size = std::min(fpga_sim::N, rows - r);

        // Read pure INT8 weight values from Q8_0 tensor (stripping FP16 headers)
        std::vector<uint8_t> W_q8(r_size * cols);
        for (int i = 0; i < r_size; i++)
            for (int j = 0; j < cols; j++) {
                uint64_t idx = (uint64_t)(r + i) * cols + j;
                uint64_t block = idx / Q8_BLOCK_SIZE;
                uint64_t off = idx % Q8_BLOCK_SIZE;
                uint64_t block_off = block * Q8_BLOCK_BYTES + 2 + off;
                W_q8[i * cols + j] = A->data[block_off];
            }

        // Compute per-row scales
        float row_scale[fpga_sim::N];
        if (row_max_abs) {
            for (int i = 0; i < r_size; i++) {
                float max_abs = row_max_abs[r + i];
                row_scale[i] = (max_abs < 1e-10f) ? 1.0f : max_abs / 32767.0f;
            }
        } else {
            for (int i = 0; i < r_size; i++) {
                float max_abs = 0.0f;
                for (int j = 0; j < cols; j++) {
                    uint64_t idx = (uint64_t)(r + i) * cols + j;
                    uint64_t block = idx / Q8_BLOCK_SIZE;
                    uint64_t off = idx % Q8_BLOCK_SIZE;
                    uint64_t block_off = block * Q8_BLOCK_BYTES + 2 + off;
                    int8_t val_q8 = (int8_t)A->data[block_off];
                    uint64_t block_base = block * Q8_BLOCK_BYTES;
                    uint16_t scale_raw = (uint16_t)A->data[block_base] | ((uint16_t)A->data[block_base + 1] << 8);
                    float dequant_val = val_q8 * f16_to_f32(scale_raw);
                    float a = fabsf(dequant_val);
                    if (a > max_abs) max_abs = a;
                }
                row_scale[i] = (max_abs < 1e-10f) ? 1.0f : max_abs / 32767.0f;
            }
        }
        for (int i = r_size; i < fpga_sim::N; i++) row_scale[i] = 1.0f;

        for (int c = 0; c < cols; c += fpga_sim::N) {
            int c_size = std::min(fpga_sim::N, cols - c);

            // Extract pure INT8 tile (64×64) from W_q8
            uint8_t q8_tile[fpga_sim::N][fpga_sim::N];
            for (int i = 0; i < r_size; i++)
                for (int k = 0; k < c_size; k++)
                    q8_tile[k][i] = W_q8[i * cols + c + k];
            for (int i = r_size; i < fpga_sim::N; i++)
                for (int k = 0; k < fpga_sim::N; k++)
                    q8_tile[k][i] = 0;
            for (int i = 0; i < r_size; i++)
                for (int k = c_size; k < fpga_sim::N; k++)
                    q8_tile[k][i] = 0;

            // Precompute combined UQ8.8 scales from original tensor FP16 headers
            uint16_t combined_scales[fpga_sim::N * 2] = {0};
            for (int i = 0; i < r_size; i++) {
                for (int blk = 0; blk < 2; blk++) {
                    uint64_t first_col = c + blk * Q8_BLOCK_SIZE;
                    if (first_col >= (uint64_t)cols) { combined_scales[i * 2 + blk] = 0; continue; }
                    uint64_t flat_idx = first_col + (uint64_t)(r + i) * cols;
                    uint64_t block_num = flat_idx / Q8_BLOCK_SIZE;
                    uint64_t block_base = block_num * Q8_BLOCK_BYTES;
                    uint16_t f16_raw = (uint16_t)A->data[block_base] | ((uint16_t)A->data[block_base + 1] << 8);
                    float block_scale = f16_to_f32(f16_raw);
                    float row_s = row_scale[i];
                    float row_inv = (row_s < 1e-10f) ? 1.0f : (1.0f / row_s);
                    float combined = block_scale * row_inv;
                    if (combined >= 256.0f) combined = 255.996f;
                    if (combined < 0.0f) combined = 0.0f;
                    combined_scales[i * 2 + blk] = (uint16_t)(combined * 256.0f + 0.5f);
                }
            }

            fpga_sim::in16_t vec[fpga_sim::N] = {0};
            for (int k = 0; k < c_size; k++) vec[k] = x_q[c + k];

            fpga_sim::acc16_t result[fpga_sim::N] = {0};
            fpga_sim::axi_vecmul_tile_q8(
                (const uint8_t*)q8_tile,
                combined_scales,
                vec,
                result
            );

            for (int i = 0; i < r_size; i++)
                y[r + i] += (double)result[i] * x_scale * row_scale[i];

            fpga_sim::g_timing.total_tiles++;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    fpga_sim::g_timing.cpu_ms += ms;
}

// ===========================================================================
// ROTARY EMBEDDING
// ===========================================================================
void apply_rope(float* q, float* k, int pos) {
    const float rope_base = 1000000.0f;
    for (int h = 0; h < NUM_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d += 2) {
            float theta = 1.0f / powf(rope_base, (float)d / HEAD_DIM);
            float freq = pos * theta;
            float cos_val = cosf(freq);
            float sin_val = sinf(freq);
            int idx = h * HEAD_DIM + d;
            float q0 = q[idx], q1 = q[idx + 1];
            q[idx] = q0 * cos_val - q1 * sin_val;
            q[idx + 1] = q0 * sin_val + q1 * cos_val;
        }
    }
    for (int h = 0; h < NUM_KV_HEADS; h++) {
        for (int d = 0; d < HEAD_DIM; d += 2) {
            float theta = 1.0f / powf(rope_base, (float)d / HEAD_DIM);
            float freq = pos * theta;
            float cos_val = cosf(freq);
            float sin_val = sinf(freq);
            int idx = h * HEAD_DIM + d;
            float k0 = k[idx], k1 = k[idx + 1];
            k[idx] = k0 * cos_val - k1 * sin_val;
            k[idx + 1] = k0 * sin_val + k1 * cos_val;
        }
    }
}

// ===========================================================================
// FORWARD LAYER with KV Cache
// ===========================================================================
void forward_layer_with_cache(float* hidden, int layer, int pos) {
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
        PROFILE_SCOPE("attn_norm");
        snprintf(name, 128, "blk.%d.attn_norm.weight", layer);
        t = get_tensor_info(name);
        rms_norm(attn_norm_out, original_hidden, HIDDEN_DIM, t);
    }

    // Q
    {
        PROFILE_SCOPE("attn_q");
        snprintf(name, 128, "blk.%d.attn_q.weight", layer);
        t = get_tensor_info(name);
        matmul(t, attn_norm_out, q_vec, HIDDEN_DIM, HIDDEN_DIM);
        snprintf(name, 128, "blk.%d.attn_q.bias", layer);
        t = get_tensor_info(name);
        if (t) { float* b = (float*)t->data; for (int i = 0; i < HIDDEN_DIM; i++) q_vec[i] += b[i]; }
    }

    // K
    {
        PROFILE_SCOPE("attn_k");
        snprintf(name, 128, "blk.%d.attn_k.weight", layer);
        t = get_tensor_info(name);
        matmul(t, attn_norm_out, k_new, K_DIM, HIDDEN_DIM);
        snprintf(name, 128, "blk.%d.attn_k.bias", layer);
        t = get_tensor_info(name);
        if (t) { float* b = (float*)t->data; for (int i = 0; i < K_DIM; i++) k_new[i] += b[i]; }
    }

    // V
    {
        PROFILE_SCOPE("attn_v");
        snprintf(name, 128, "blk.%d.attn_v.weight", layer);
        t = get_tensor_info(name);
        matmul(t, attn_norm_out, v_new, V_DIM, HIDDEN_DIM);
        snprintf(name, 128, "blk.%d.attn_v.bias", layer);
        t = get_tensor_info(name);
        if (t) { float* b = (float*)t->data; for (int i = 0; i < V_DIM; i++) v_new[i] += b[i]; }
    }

    {
        PROFILE_SCOPE("rope");
        apply_rope(q_vec, k_new, pos);
    }

    for (int i = 0; i < K_DIM; i++) g_k_cache[layer][pos][i] = k_new[i];
    for (int i = 0; i < V_DIM; i++) g_v_cache[layer][pos][i] = v_new[i];

    {
        PROFILE_SCOPE("attention");
        memset(context, 0, HIDDEN_DIM * sizeof(float));
        int q_per_kv = NUM_HEADS / NUM_KV_HEADS;

        for (int qh = 0; qh < NUM_HEADS; qh++) {
            int kv = qh / q_per_kv;
            float* qh_data = q_vec + qh * HEAD_DIM;
            float* ctx_h = context + qh * HEAD_DIM;

            float scores[MAX_SEQ_LEN];
            for (int p = 0; p <= pos; p++) {
                float* k_cached = g_k_cache[layer][p] + kv * HEAD_DIM;
                scores[p] = 0;
                for (int d = 0; d < HEAD_DIM; d++) {
                    scores[p] += qh_data[d] * k_cached[d];
                }
                scores[p] /= sqrtf((float)HEAD_DIM);
            }

            float max_score = scores[0];
            for (int p = 1; p <= pos; p++) if (scores[p] > max_score) max_score = scores[p];
            float exp_sum = 0;
            for (int p = 0; p <= pos; p++) exp_sum += expf(scores[p] - max_score);
            float log_sum = logf(exp_sum) + max_score;

            for (int d = 0; d < HEAD_DIM; d++) {
                float weighted_sum = 0;
                for (int p = 0; p <= pos; p++) {
                    float* v_cached = g_v_cache[layer][p] + kv * HEAD_DIM;
                    float exp_score = expf(scores[p] - log_sum);
                    weighted_sum += exp_score * v_cached[d];
                }
                ctx_h[d] = weighted_sum;
            }
        }
    }

    // Attention output
    {
        PROFILE_SCOPE("attn_output");
        snprintf(name, 128, "blk.%d.attn_output.weight", layer);
        t = get_tensor_info(name);
        matmul(t, context, attn_out, HIDDEN_DIM, HIDDEN_DIM);
    }

    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] = original_hidden[i] + attn_out[i];

    // FFN norm
    {
        PROFILE_SCOPE("ffn_norm");
        snprintf(name, 128, "blk.%d.ffn_norm.weight", layer);
        t = get_tensor_info(name);
        rms_norm(ffn_norm_out, hidden, HIDDEN_DIM, t);
    }

    // FFN gate
    {
        PROFILE_SCOPE("ffn_gate");
        snprintf(name, 128, "blk.%d.ffn_gate.weight", layer);
        t = get_tensor_info(name);
        matmul(t, ffn_norm_out, gate, INTER_DIM, HIDDEN_DIM);
    }

    // FFN up
    {
        PROFILE_SCOPE("ffn_up");
        snprintf(name, 128, "blk.%d.ffn_up.weight", layer);
        t = get_tensor_info(name);
        matmul(t, ffn_norm_out, up, INTER_DIM, HIDDEN_DIM);
    }

    {
        PROFILE_SCOPE("silu_x_up");
        silu(gate, gate, INTER_DIM);
        for (int i = 0; i < INTER_DIM; i++) gate[i] *= up[i];
    }

    // FFN down
    {
        PROFILE_SCOPE("ffn_down");
        snprintf(name, 128, "blk.%d.ffn_down.weight", layer);
        t = get_tensor_info(name);
        matmul(t, gate, ffn_out, HIDDEN_DIM, INTER_DIM);
    }

    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] += ffn_out[i];
}

// ===========================================================================
// FORWARD (no cache - single token)
// ===========================================================================
void forward_layer(float* hidden, int layer, int pos) {
    char name[128];
    Tensor* t;
    float original_hidden[HIDDEN_DIM];
    memcpy(original_hidden, hidden, HIDDEN_DIM * sizeof(float));

    // Attention norm
    snprintf(name, 128, "blk.%d.attn_norm.weight", layer);
    t = get_tensor_info(name);
    float attn_norm_out[HIDDEN_DIM];
    rms_norm(attn_norm_out, original_hidden, HIDDEN_DIM, t);

    // Q, K, V
    snprintf(name, 128, "blk.%d.attn_q.weight", layer);
    t = get_tensor_info(name);
    float q[HIDDEN_DIM];
    matmul(t, attn_norm_out, q, HIDDEN_DIM, HIDDEN_DIM);
    snprintf(name, 128, "blk.%d.attn_q.bias", layer);
    t = get_tensor_info(name);
    if (t) { float* b = (float*)t->data; for (int i = 0; i < HIDDEN_DIM; i++) q[i] += b[i]; }

    snprintf(name, 128, "blk.%d.attn_k.weight", layer);
    t = get_tensor_info(name);
    float k[K_DIM];
    matmul(t, attn_norm_out, k, K_DIM, HIDDEN_DIM);
    snprintf(name, 128, "blk.%d.attn_k.bias", layer);
    t = get_tensor_info(name);
    if (t) { float* b = (float*)t->data; for (int i = 0; i < K_DIM; i++) k[i] += b[i]; }

    snprintf(name, 128, "blk.%d.attn_v.weight", layer);
    t = get_tensor_info(name);
    float v[V_DIM];
    matmul(t, attn_norm_out, v, V_DIM, HIDDEN_DIM);
    snprintf(name, 128, "blk.%d.attn_v.bias", layer);
    t = get_tensor_info(name);
    if (t) { float* b = (float*)t->data; for (int i = 0; i < V_DIM; i++) v[i] += b[i]; }

    apply_rope(q, k, pos);

    // GQA attention (single token)
    float context[HIDDEN_DIM] = {0};
    int q_per_kv = NUM_HEADS / NUM_KV_HEADS;
    for (int qh = 0; qh < NUM_HEADS; qh++) {
        int kv = qh / q_per_kv;
        float* qh_data = q + qh * HEAD_DIM;
        float* kh_data = k + kv * HEAD_DIM;
        float* vh_data = v + kv * HEAD_DIM;
        float* ctx_h = context + qh * HEAD_DIM;

        float score = 0;
        for (int d = 0; d < HEAD_DIM; d++) score += qh_data[d] * kh_data[d];
        score /= sqrtf((float)HEAD_DIM);
        float softmax_w = 1.0f;

        for (int d = 0; d < HEAD_DIM; d++) ctx_h[d] = softmax_w * vh_data[d];
    }

    // Attention output projection
    snprintf(name, 128, "blk.%d.attn_output.weight", layer);
    t = get_tensor_info(name);
    float attn_out[HIDDEN_DIM];
    matmul(t, context, attn_out, HIDDEN_DIM, HIDDEN_DIM);

    // Residual
    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] = original_hidden[i] + attn_out[i];

    // FFN norm
    snprintf(name, 128, "blk.%d.ffn_norm.weight", layer);
    t = get_tensor_info(name);
    float ffn_norm_out[HIDDEN_DIM];
    rms_norm(ffn_norm_out, hidden, HIDDEN_DIM, t);

    // FFN gate
    snprintf(name, 128, "blk.%d.ffn_gate.weight", layer);
    t = get_tensor_info(name);
    float gate[INTER_DIM];
    matmul(t, ffn_norm_out, gate, INTER_DIM, HIDDEN_DIM);

    // FFN up
    snprintf(name, 128, "blk.%d.ffn_up.weight", layer);
    t = get_tensor_info(name);
    float up[INTER_DIM];
    matmul(t, ffn_norm_out, up, INTER_DIM, HIDDEN_DIM);

    // SwiGLU
    silu(gate, gate, INTER_DIM);
    for (int i = 0; i < INTER_DIM; i++) gate[i] *= up[i];

    // FFN down
    snprintf(name, 128, "blk.%d.ffn_down.weight", layer);
    t = get_tensor_info(name);
    float ffn_out[HIDDEN_DIM];
    matmul(t, gate, ffn_out, HIDDEN_DIM, INTER_DIM);

    // Final residual
    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] += ffn_out[i];
}

// ===========================================================================
// GET LOGITS — Q8_0 PATH
// FPGA receives hidden (1×896 FP32) once, streams Q8_0 embeddings,
// does Q8→INT16 dequant internally, returns raw logits (151936 floats).
// This bypasses CPU dequantization for the token embedding lookup.
// ===========================================================================
void get_logits_q8(float* logits, const float* hidden) {
#ifdef Q8_DEBUG
    printf("[LOGITS_Q8] get_logits_q8 called\n"); fflush(stdout);
#endif
    Tensor* emb_t = get_tensor_info("token_embd.weight");
#ifdef Q8_DEBUG
    printf("[LOGITS_Q8] emb_t=%p\n", (void*)emb_t); fflush(stdout);
#endif
    if (!emb_t) { printf("[ERROR] No token_embd.weight\n"); return; }

    Tensor* norm_t = get_tensor_info("output_norm.weight");
    float norm_hidden[HIDDEN_DIM];
    {
        PROFILE_SCOPE("output_norm");
        if (norm_t) {
            rms_norm(norm_hidden, hidden, HIDDEN_DIM, norm_t);
            hidden = norm_hidden;
        }
    }

    {
        PROFILE_SCOPE("logits");
        extern void q8_logits_matmul_with_tensor(const Tensor* t, const float* x, float* y, int rows, int cols);
        q8_logits_matmul_with_tensor(emb_t, hidden, logits, VOCAB_SIZE, HIDDEN_DIM);
    }
}

// ===========================================================================
// GET LOGITS — DEFAULT PATH
// ===========================================================================
void get_logits(float* logits, const float* hidden) {
    Tensor* emb_t = get_tensor_info("token_embd.weight");
    if (!emb_t) { printf("[ERROR] No token_embd.weight\n"); return; }

    Tensor* norm_t = get_tensor_info("output_norm.weight");
    float norm_hidden[HIDDEN_DIM];
    {
        PROFILE_SCOPE("output_norm");
        if (norm_t) {
            rms_norm(norm_hidden, hidden, HIDDEN_DIM, norm_t);
            hidden = norm_hidden;
        }
    }

    {
        PROFILE_SCOPE("logits");
        matmul(emb_t, hidden, logits, VOCAB_SIZE, HIDDEN_DIM);
    }
}

// ===========================================================================
// TOP-K SAMPLING
// ===========================================================================
int sample_top_k(float* logits, int k) {
    std::vector<std::pair<float, int>> topk;
    for (int i = 0; i < VOCAB_SIZE; i++) topk.push_back({logits[i], i});
    std::partial_sort(topk.begin(), topk.begin() + k, topk.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    float sum = 0;
    for (int i = 0; i < k; i++) topk[i].first = expf(topk[i].first);
    for (int i = 0; i < k; i++) sum += topk[i].first;
    for (int i = 0; i < k; i++) topk[i].first /= sum;
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0;
    for (int i = 0; i < k; i++) {
        cumsum += topk[i].first;
        if (r < cumsum) return topk[i].second;
    }
    return topk[k-1].second;
}

// ===========================================================================
// LOAD TMAC
// ===========================================================================
int load_tmac(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { printf("[ERROR] Cannot open %s\n", path); return -1; }
    uint8_t magic[4];
    if (fread(magic, 1, 4, f) != 4) return -1;
    uint64_t n_tensors;
    if (fread(&n_tensors, 8, 1, f) != 1) return -1;
    g_tensors = (Tensor*)malloc(n_tensors * sizeof(Tensor));
    g_ntensors = n_tensors;
    uint64_t offset = 0;
    for (uint64_t i = 0; i < n_tensors; i++) {
        uint64_t name_len, rows, cols, n_bytes;
        uint32_t type;
        if (fread(&name_len, 8, 1, f) != 1) { printf("[ERROR] fread name_len failed at tensor %lu\n", (unsigned long)i); return -1; }
        if (name_len >= 128) {
            printf("[WARN] name_len=%lu >= 128 for tensor %lu, truncating to 127\n", (unsigned long)name_len, (unsigned long)i);
            if (fread(g_tensors[i].name, 1, 127, f) != 127) { printf("[ERROR] fread name failed at tensor %lu\n", (unsigned long)i); return -1; }
            g_tensors[i].name[127] = 0;
            fseek(f, name_len - 127, SEEK_CUR);
        } else {
            if (fread(g_tensors[i].name, 1, name_len, f) != name_len) { printf("[ERROR] fread name failed at tensor %lu\n", (unsigned long)i); return -1; }
            g_tensors[i].name[name_len] = 0;
        }
        g_tensors[i].name[name_len] = 0;
        if (fread(&rows, 8, 1, f) != 1) { printf("[ERROR] fread rows failed at tensor %lu\n", (unsigned long)i); return -1; }
        if (fread(&cols, 8, 1, f) != 1) { printf("[ERROR] fread cols failed at tensor %lu\n", (unsigned long)i); return -1; }
        if (fread(&type, 4, 1, f) != 1) { printf("[ERROR] fread type failed at tensor %lu\n", (unsigned long)i); return -1; }
        if (fread(&n_bytes, 8, 1, f) != 1) { printf("[ERROR] fread n_bytes failed at tensor %lu\n", (unsigned long)i); return -1; }
        if (offset + n_bytes > DDR_SIZE) { printf("[ERROR] overflow at tensor %lu: offset=%lu n_bytes=%lu DDR_SIZE=%lu\n", (unsigned long)i, (unsigned long)offset, (unsigned long)n_bytes, (unsigned long)(uint64_t)DDR_SIZE); return -1; }
        g_tensors[i].rows = rows;
        g_tensors[i].cols = cols;
        g_tensors[i].type = type;
        g_tensors[i].n_bytes = n_bytes;
        g_tensors[i].data = g_ddr + offset;
        if (fread(g_tensors[i].data, 1, n_bytes, f) != n_bytes) { printf("[ERROR] fread data failed at tensor %lu: name=%s rows=%lu cols=%lu n_bytes=%lu\n", (unsigned long)i, g_tensors[i].name, (unsigned long)rows, (unsigned long)cols, (unsigned long)n_bytes); return -1; }
        offset += n_bytes;
        if (i < 5 || i >= n_tensors - 3) fprintf(stderr, "[LOAD] tensor[%lu] %s rows=%lu cols=%lu type=%u n_bytes=%lu offset_after=%lu\n", (unsigned long)i, g_tensors[i].name, (unsigned long)rows, (unsigned long)cols, type, (unsigned long)n_bytes, (unsigned long)offset);
    }
    fclose(f);
    fprintf(stderr, "[LOAD] done loading, about to print OK\n"); fflush(stderr);
    printf("[OK] Loaded %lu tensors, DDR: %.1f MB / %d MB\n",
           (unsigned long)n_tensors, offset / (1024.0 * 1024.0), (int)(DDR_SIZE / (1024 * 1024))); fflush(stdout);
    fprintf(stderr, "[LOAD] about to return\n"); fflush(stderr);
    return 0;
}

// ===========================================================================
// INFERENCE with CACHE
// ===========================================================================
void reset_cache() {
    g_seq_len = 0;
    memset(g_k_cache, 0, sizeof(g_k_cache));
    memset(g_v_cache, 0, sizeof(g_v_cache));
}

void process_embedding(float* hidden, int token_id) {
    PROFILE_SCOPE("embedding");
    Tensor* emb_t = get_tensor_info("token_embd.weight");
    if (!emb_t) return;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = read_embedding(emb_t, token_id, i);
    }
}

void forward_all_layers(float* hidden, int pos) {
    PROFILE_SCOPE("forward_all_layers");
    for (int layer = 0; layer < NUM_LAYERS; layer++) {
        forward_layer_with_cache(hidden, layer, pos);
    }
}

int sample_token(float* logits, int top_k) {
    std::vector<std::pair<float, int>> candidates;
    for (int i = 0; i < VOCAB_SIZE; i++) candidates.push_back({logits[i], i});
    std::partial_sort(candidates.begin(), candidates.begin() + top_k, candidates.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    float sum = 0;
    for (int i = 0; i < top_k; i++) candidates[i].first = expf(candidates[i].first);
    for (int i = 0; i < top_k; i++) sum += candidates[i].first;
    for (int i = 0; i < top_k; i++) candidates[i].first /= sum;
    float r = (float)rand() / (float)RAND_MAX;
    float cumsum = 0;
    for (int i = 0; i < top_k; i++) {
        cumsum += candidates[i].first;
        if (r < cumsum) return candidates[i].second;
    }
    return candidates[0].second;
}

int process_prompt(float* hidden, const std::vector<int>& tokens, bool dump_layers) {
    reset_cache();
    Tensor* emb_t = get_tensor_info("token_embd.weight");
    if (!emb_t) { printf("[ERROR] No embedding\n"); return -1; }

    for (size_t t = 0; t < tokens.size(); t++) {
        int token_id = tokens[t];
        process_embedding(hidden, token_id);

        if (dump_layers) {
            char filename[64];
            snprintf(filename, sizeof(filename), "/tmp/cpp_layer_0.bin");
            FILE* f = fopen(filename, "wb");
            fwrite(hidden, sizeof(float), HIDDEN_DIM, f);
            fclose(f);
        }

        for (int layer = 0; layer < NUM_LAYERS; layer++) {
            forward_layer_with_cache(hidden, layer, (int)t);

            if (dump_layers && t == tokens.size() - 1) {
                char filename[64];
                snprintf(filename, sizeof(filename), "/tmp/cpp_layer_%d.bin", layer + 1);
                FILE* f = fopen(filename, "wb");
                fwrite(hidden, sizeof(float), HIDDEN_DIM, f);
                fclose(f);
            }
        }
    }
    g_seq_len = (int)tokens.size();
    return 0;
}

void generate(float* hidden, float* logits, int prompt_len, int n_tokens, int top_k) {
    Tensor* emb_t = get_tensor_info("token_embd.weight");
    if (!emb_t) return;

    for (int gen = 0; gen < n_tokens; gen++) {
        int pos = prompt_len + gen;
        if (pos >= MAX_SEQ_LEN) break;

        if (g_fpga_q8) {
            get_logits_q8(logits, hidden);
        } else {
            get_logits(logits, hidden);
        }
        int next_token = sample_token(logits, top_k);
        printf("%d\n", next_token);
        fflush(stdout);

        process_embedding(hidden, next_token);
        forward_all_layers(hidden, pos);
    }
}

// ===========================================================================
// MAIN
// ===========================================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model.tmac> [--generate N] [--dump-layers] [--fpga] [--fpga-int16] [--fpga-q8] [--perf]\n", argv[0]);
        printf("  Prompt tokens read from stdin, generated tokens printed to stdout\n");
        printf("  --fpga:       Use INT8 FPGA simulation path (lower accuracy)\n");
        printf("  --fpga-int16: Use INT16 FPGA simulation path (higher accuracy, recommended)\n");
        printf("  --fpga-q8:    Use Q8_0 direct path — FPGA handles Q8→INT16 dequant (experimental)\n");
        printf("  --perf:       Enable pipeline profiling (Chrome trace JSON + bottleneck analysis)\n");
        return 1;
    }

    g_ddr = (uint8_t*)malloc(DDR_SIZE);
    if (!g_ddr) { printf("[ERROR] Cannot allocate DDR\n"); return 1; }
    printf("[OK] Allocated %d MB DDR\n", (int)(DDR_SIZE / (1024 * 1024))); fflush(stdout);
    fpga_sim::g_timing.reset();
    if (load_tmac(argv[1]) != 0) { printf("[ERROR] Failed to load TMAC\n"); return 1; }

    float* hidden = new float[HIDDEN_DIM];
    float* logits = new float[VOCAB_SIZE];
    int generate_n = 0;
    bool dump_layers = false;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            generate_n = atoi(argv[++i]);
        }
        if (strcmp(argv[i], "--dump-layers") == 0) dump_layers = true;
        if (strcmp(argv[i], "--fpga") == 0) g_use_fpga = true;
        if (strcmp(argv[i], "--fpga-int16") == 0) { g_use_fpga = true; g_fpga_int16 = true; }
        if (strcmp(argv[i], "--fpga-q8") == 0) { g_use_fpga = true; g_fpga_q8 = true; g_fpga_int16 = true; }
        if (strcmp(argv[i], "--perf") == 0) g_perf_enabled = true;
    }

    std::vector<int> tokens;
    int token;
    while (scanf("%d", &token) == 1) tokens.push_back(token);
    if (tokens.empty()) { printf("[ERROR] No prompt tokens\n"); return 1; }

    {
        PROFILE_SCOPE("process_prompt");
        if (process_prompt(hidden, tokens, dump_layers) != 0) return 1;
    }

    if (generate_n > 0) {
        generate(hidden, logits, (int)tokens.size(), generate_n, 40);
    } else {
        if (g_fpga_q8) {
            get_logits_q8(logits, hidden);
        } else {
            get_logits(logits, hidden);
        }
        printf("[");
        std::vector<std::pair<float, int>> top;
        for (int i = 0; i < VOCAB_SIZE; i++) top.push_back({logits[i], i});
        std::partial_sort(top.begin(), top.begin() + 10, top.end(),
                         [](const auto& a, const auto& b) { return a.first > b.first; });
        for (int i = 0; i < 10; i++) {
            printf("{\"id\":%d,\"logit\":%.3f}", top[i].second, top[i].first);
            if (i < 9) printf(",");
        }
        printf("]\n");
    }

    if (g_use_fpga) {
        fpga_sim::g_timing.total_fpga_cycles += fpga_sim::accel().total_cycles();
        fpga_sim::g_timing.report();
    }
    if (g_perf_enabled) {
        print_trace_summary();
        dump_chrome_trace("/tmp/pipeline_trace.json");
    }
    free(g_ddr);
    free(g_tensors);
    delete[] hidden;
    delete[] logits;
    return 0;
}