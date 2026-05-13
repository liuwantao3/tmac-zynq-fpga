/*
 * T-MAC Runtime Implementation
 * Zynq 7010 ARM-side inference engine
 */

#include "tmac_runtime.hpp"
#include "tmac_fpga.hpp"
#include <cmath>
#include <algorithm>
#include <stdio.h>

namespace tmac {

TMacRuntime::TMacRuntime()
    : initialized_(false)
    , fpga_(*(new FPGA()))  // Simplified, actual would be proper member
{
}

TMacRuntime::~TMacRuntime() {
    if (act_buf_a_) delete[] act_buf_a_;
    if (act_buf_b_) delete[] act_buf_b_;
    if (act_buf_c_) delete[] act_buf_c_;
    if (kv_cache_.key_cache) delete[] kv_cache_.key_cache;
    if (kv_cache_.value_cache) delete[] kv_cache_.value_cache;
}

int TMacRuntime::init(const ModelConfig& config) {
    config_ = config;
    initialized_ = false;

    // Allocate scratch buffers in DDR (cached for FPGA access)
    act_buf_a_ = new int8_t[32 * 1024 * 1024];  // 32MB
    act_buf_b_ = new int8_t[16 * 1024 * 1024];  // 16MB
    act_buf_c_ = new int32_t[16 * 1024 * 1024 / 4];  // 16MB

    // Allocate KV cache
    int kv_size = config_.num_layers * config_.max_seq_len *
                  config_.num_heads * config_.head_dim * sizeof(int32_t);
    kv_cache_.key_cache = new int32_t[kv_size / sizeof(int32_t)];
    kv_cache_.value_cache = new int32_t[kv_size / sizeof(int32_t)];
    kv_cache_.max_seq = config_.max_seq_len;

    // Initialize FPGA
    // fpga_.init();  // Would call TMacFPGA init

    initialized_ = true;
    printf("[T-MAC] Runtime initialized\n");
    return 0;
}

int TMacRuntime::load_weights(const char* weight_path) {
    // Load quantized weights from file
    // In practice, would parse GGUF/GPTQ format
    printf("[T-MAC] Loading weights from %s\n", weight_path);
    return 0;
}

void TMacRuntime::layer_norm(const float* input, float* output,
                             const LayerNorm& ln, int size) {
    LayerNormImpl::forward(output, input, ln.gamma, ln.beta, size);
}

void TMacRuntime::softmax(float* input, int size) {
    SoftmaxImpl::forward(input, size);
}

void TMacRuntime::silu(float* input, int size) {
    ActivationImpl::silu(input, size);
}

void TMacRuntime::gelu(float* input, int size) {
    ActivationImpl::gelu(input, size);
}

int TMacRuntime::sample_temperature(const float* logits, int vocab_size, float temperature) {
    // Apply temperature
    float sum = 0;
    float scaled[vocab_size];

    for (int i = 0; i < vocab_size; i++) {
        scaled[i] = logits[i] / temperature;
        sum += std::exp(scaled[i]);
    }

    // Sample
    float r = (float)rand() / RAND_MAX * sum;
    float cumsum = 0;
    for (int i = 0; i < vocab_size; i++) {
        cumsum += std::exp(scaled[i]);
        if (r <= cumsum) return i;
    }
    return vocab_size - 1;
}

void LayerNormImpl::forward(float* output, const float* input,
                            const float* gamma, const float* beta, int size) {
    // Compute mean
    float mean = 0;
    for (int i = 0; i < size; i++) mean += input[i];
    mean /= size;

    // Compute variance
    float var = 0;
    for (int i = 0; i < size; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    var /= size;

    // Normalize
    float inv_std = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < size; i++) {
        float x = (input[i] - mean) * inv_std;
        output[i] = x * gamma[i] + beta[i];
    }
}

void SoftmaxImpl::forward(float* input, int size) {
    // Find max for numerical stability
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }

    // Compute exp and sum
    float sum = 0;
    for (int i = 0; i < size; i++) {
        input[i] = std::exp(input[i] - max_val);
        sum += input[i];
    }

    // Normalize
    for (int i = 0; i < size; i++) {
        input[i] /= sum;
    }
}

void ActivationImpl::silu(float* data, int size) {
    // SiLU(x) = x * sigmoid(x)
    for (int i = 0; i < size; i++) {
        float x = data[i];
        float sigmoid = 1.0f / (1.0f + std::exp(-x));
        data[i] = x * sigmoid;
    }
}

void ActivationImpl::gelu(float* data, int size) {
    // GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const float sqrt_2_over_pi = 0.7978845608f;
    const float coeff = 0.044715f;

    for (int i = 0; i < size; i++) {
        float x = data[i];
        float x_cubed = x * x * x;
        float tanh_arg = sqrt_2_over_pi * (x + coeff * x_cubed);
        float tanh_val = std::tanh(tanh_arg);
        data[i] = 0.5f * x * (1.0f + tanh_val);
    }
}

} // namespace tmac