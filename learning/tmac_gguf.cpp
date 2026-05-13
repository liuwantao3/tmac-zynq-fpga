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
float dequant_q8_0(const uint8_t* data, uint64_t idx) {
    uint64_t block = idx / 32;
    uint64_t offset = idx % 32;
    uint64_t block_offset = block * 34;
    uint16_t scale_raw = data[block_offset] | (data[block_offset + 1] << 8);
    float scale = f16_to_f32(scale_raw);
    int8_t val = (int8_t)data[block_offset + 2 + offset];
    return val * scale;
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
    // For 1D tensors (cols==1): flat index = row
    uint64_t idx = (t->cols == 1) ? row : (col + row * t->rows);
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

static Tensor* get_tensor_info(const char* name) {
    for (int i = 0; i < g_ntensors; i++) {
        if (strcmp(g_tensors[i].name, name) == 0) return &g_tensors[i];
    }
    return nullptr;
}

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
void matmul(const Tensor* A, const float* x, float* y, int rows, int cols) {
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

    // Attention norm
    snprintf(name, 128, "blk.%d.attn_norm.weight", layer);
    t = get_tensor_info(name);
    float attn_norm_out[HIDDEN_DIM];
    rms_norm(attn_norm_out, original_hidden, HIDDEN_DIM, t);
    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/attn_norm_out.bin", "wb"); fwrite(attn_norm_out, sizeof(float), HIDDEN_DIM, f); fclose(f); }

    // Q, K, V
    snprintf(name, 128, "blk.%d.attn_q.weight", layer);
    t = get_tensor_info(name);
    float q[HIDDEN_DIM];
    matmul(t, attn_norm_out, q, HIDDEN_DIM, HIDDEN_DIM);
    snprintf(name, 128, "blk.%d.attn_q.bias", layer);
    t = get_tensor_info(name);
    if (t) { float* b = (float*)t->data; for (int i = 0; i < HIDDEN_DIM; i++) q[i] += b[i]; }
    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/q_after_bias.bin", "wb"); fwrite(q, sizeof(float), HIDDEN_DIM, f); fclose(f); }

    snprintf(name, 128, "blk.%d.attn_k.weight", layer);
    t = get_tensor_info(name);
    float k_new[K_DIM];
    matmul(t, attn_norm_out, k_new, K_DIM, HIDDEN_DIM);
    snprintf(name, 128, "blk.%d.attn_k.bias", layer);
    t = get_tensor_info(name);
    if (t) { float* b = (float*)t->data; for (int i = 0; i < K_DIM; i++) k_new[i] += b[i]; }
    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/k_after_bias.bin", "wb"); fwrite(k_new, sizeof(float), K_DIM, f); fclose(f); }

    snprintf(name, 128, "blk.%d.attn_v.weight", layer);
    t = get_tensor_info(name);
    float v_new[V_DIM];
    matmul(t, attn_norm_out, v_new, V_DIM, HIDDEN_DIM);
    snprintf(name, 128, "blk.%d.attn_v.bias", layer);
    t = get_tensor_info(name);
    if (t) { float* b = (float*)t->data; for (int i = 0; i < V_DIM; i++) v_new[i] += b[i]; }
    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/v_after_bias.bin", "wb"); fwrite(v_new, sizeof(float), V_DIM, f); fclose(f); }

    apply_rope(q, k_new, pos);

    for (int i = 0; i < K_DIM; i++) g_k_cache[layer][pos][i] = k_new[i];
    for (int i = 0; i < V_DIM; i++) g_v_cache[layer][pos][i] = v_new[i];

    float context[HIDDEN_DIM] = {0};
    int q_per_kv = NUM_HEADS / NUM_KV_HEADS;

    for (int qh = 0; qh < NUM_HEADS; qh++) {
        int kv = qh / q_per_kv;
        float* qh_data = q + qh * HEAD_DIM;
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

    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/context.bin", "wb"); fwrite(context, sizeof(float), HIDDEN_DIM, f); fclose(f); }

    snprintf(name, 128, "blk.%d.attn_output.weight", layer);
    t = get_tensor_info(name);
    float attn_out[HIDDEN_DIM];
    matmul(t, context, attn_out, HIDDEN_DIM, HIDDEN_DIM);
    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/attn_out.bin", "wb"); fwrite(attn_out, sizeof(float), HIDDEN_DIM, f); fclose(f); }

    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] = original_hidden[i] + attn_out[i];
    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/after_attn_res.bin", "wb"); fwrite(hidden, sizeof(float), HIDDEN_DIM, f); fclose(f); }

    snprintf(name, 128, "blk.%d.ffn_norm.weight", layer);
    t = get_tensor_info(name);
    float ffn_norm_out[HIDDEN_DIM];
    rms_norm(ffn_norm_out, hidden, HIDDEN_DIM, t);
    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/ffn_norm_out.bin", "wb"); fwrite(ffn_norm_out, sizeof(float), HIDDEN_DIM, f); fclose(f); }

    snprintf(name, 128, "blk.%d.ffn_gate.weight", layer);
    t = get_tensor_info(name);
    float gate[INTER_DIM];
    matmul(t, ffn_norm_out, gate, INTER_DIM, HIDDEN_DIM);
    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/gate.bin", "wb"); fwrite(gate, sizeof(float), INTER_DIM, f); fclose(f); }

    snprintf(name, 128, "blk.%d.ffn_up.weight", layer);
    t = get_tensor_info(name);
    float up[INTER_DIM];
    matmul(t, ffn_norm_out, up, INTER_DIM, HIDDEN_DIM);
    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/up.bin", "wb"); fwrite(up, sizeof(float), INTER_DIM, f); fclose(f); }

    silu(gate, gate, INTER_DIM);
    for (int i = 0; i < INTER_DIM; i++) gate[i] *= up[i];
    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/gate_x_up.bin", "wb"); fwrite(gate, sizeof(float), INTER_DIM, f); fclose(f); }

    snprintf(name, 128, "blk.%d.ffn_down.weight", layer);
    t = get_tensor_info(name);
    float ffn_out[HIDDEN_DIM];
    matmul(t, gate, ffn_out, HIDDEN_DIM, INTER_DIM);
    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/ffn_out.bin", "wb"); fwrite(ffn_out, sizeof(float), HIDDEN_DIM, f); fclose(f); }

    for (int i = 0; i < HIDDEN_DIM; i++) hidden[i] += ffn_out[i];
    if (layer == 0) { FILE* f = fopen("/tmp/cpp_inter/final.bin", "wb"); fwrite(hidden, sizeof(float), HIDDEN_DIM, f); fclose(f); }
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
// GET LOGITS
// ===========================================================================
void get_logits(float* logits, const float* hidden) {
    Tensor* emb_t = get_tensor_info("token_embd.weight");
    if (!emb_t) { printf("[ERROR] No token_embd.weight\n"); return; }

    // Apply output norm before lm_head
    Tensor* norm_t = get_tensor_info("output_norm.weight");
    float norm_hidden[HIDDEN_DIM];
    if (norm_t) {
        rms_norm(norm_hidden, hidden, HIDDEN_DIM, norm_t);
        hidden = norm_hidden;
    }

    for (int v = 0; v < VOCAB_SIZE; v++) {
        float sum = 0;
        for (int h = 0; h < HIDDEN_DIM; h++) {
            sum += hidden[h] * read_embedding(emb_t, v, h);
        }
        logits[v] = sum;
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
    float r = (float)rand() / RAND_MAX;
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
        fread(&name_len, 8, 1, f);
        fread(g_tensors[i].name, 1, name_len, f);
        g_tensors[i].name[name_len] = 0;
        fread(&rows, 8, 1, f);
        fread(&cols, 8, 1, f);
        fread(&type, 4, 1, f);
        fread(&n_bytes, 8, 1, f);
        g_tensors[i].rows = rows;
        g_tensors[i].cols = cols;
        g_tensors[i].type = type;
        g_tensors[i].n_bytes = n_bytes;
        g_tensors[i].data = g_ddr + offset;
        fread(g_tensors[i].data, 1, n_bytes, f);
        offset += n_bytes;
    }
    fclose(f);
    printf("[OK] Loaded %d tensors, DDR: %.1f MB / %d MB\n",
           n_tensors, offset / (1024.0 * 1024.0), (int)(DDR_SIZE / (1024 * 1024)));
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
    Tensor* emb_t = get_tensor_info("token_embd.weight");
    if (!emb_t) return;
    for (int i = 0; i < HIDDEN_DIM; i++) {
        hidden[i] = read_embedding(emb_t, token_id, i);
    }
}

void forward_all_layers(float* hidden, int pos) {
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
    float r = (float)rand() / RAND_MAX;
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

        get_logits(logits, hidden);
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
        printf("Usage: %s <model.tmac> [--generate N] [--dump-layers]\n", argv[0]);
        printf("  Prompt tokens read from stdin, generated tokens printed to stdout\n");
        return 1;
    }

    g_ddr = (uint8_t*)malloc(DDR_SIZE);
    if (!g_ddr) { printf("[ERROR] Cannot allocate DDR\n"); return 1; }
    printf("[OK] Allocated %d MB DDR\n", (int)(DDR_SIZE / (1024 * 1024)));

    if (load_tmac(argv[1]) != 0) return 1;

    float hidden[HIDDEN_DIM];
    float logits[VOCAB_SIZE];

    int generate_n = 0;
    bool dump_layers = false;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--generate") == 0 && i + 1 < argc) {
            generate_n = atoi(argv[++i]);
        }
        if (strcmp(argv[i], "--dump-layers") == 0) dump_layers = true;
    }

    std::vector<int> tokens;
    int token;
    while (scanf("%d", &token) == 1) tokens.push_back(token);
    if (tokens.empty()) { printf("[ERROR] No prompt tokens\n"); return 1; }

    if (process_prompt(hidden, tokens, dump_layers) != 0) return 1;

    if (generate_n > 0) {
        generate(hidden, logits, (int)tokens.size(), generate_n, 40);
    } else {
        get_logits(logits, hidden);
        { FILE* lf = fopen("/tmp/cpp_logits.bin", "wb"); fwrite(logits, sizeof(float), VOCAB_SIZE, lf); fclose(lf); }
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

    free(g_ddr);
    free(g_tensors);
    return 0;
}