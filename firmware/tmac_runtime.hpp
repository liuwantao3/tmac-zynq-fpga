/*
 * T-MAC Runtime for Zynq ARM
 * Lightweight inference runtime with FPGA acceleration
 */

#ifndef T_MAC_RUNTIME_HPP
#define T_MAC_RUNTIME_HPP

#include <vector>
#include <cstring>
#include <functional>

namespace tmac {

constexpr int MAX_SEQ_LEN = 2048;
constexpr int MAX_LAYERS = 64;
constexpr int HIDDEN_DIM = 896;  // Qwen 0.5B
constexpr int INTERMEDIATE_DIM = 4864;

struct ModelConfig {
    int vocab_size;
    int hidden_dim;
    int intermediate_dim;
    int num_layers;
    int max_seq_len;
    int num_heads;
    int head_dim;
};

struct LayerNorm {
    float gamma[896];
    float beta[896];
};

struct QuantWeight {
    int8_t* data;
    int8_t* zeros;
    int8_t* scales;
    int32_t* lut;  // Lookup table for T-MAC
    int rows, cols;
};

class TMacRuntime {
public:
    TMacRuntime();
    ~TMacRuntime();

    int init(const ModelConfig& config);
    int load_weights(const char* weight_path);

    // Token processing
    int forward(const std::vector<int>& input_ids, std::vector<int>& output_ids);

    // Layer operations
    void layer_norm(const float* input, float* output, const LayerNorm& ln, int size);
    void softmax(float* input, int size);
    void silu(float* input, int size);
    void gelu(float* input, int size);

    // FPGA accelerated operations
    int gemm_fpga(int8_t* A, int8_t* B, int32_t* C, int M, int N, int K);
    int gemv_fpga(int8_t* vec, int8_t* M, int32_t* result, int N);

    // Sampling
    int sample_argmax(const float* logits, int vocab_size);
    int sample_temperature(const float* logits, int vocab_size, float temperature);

private:
    ModelConfig config_;
    bool initialized_;

    // Weights storage
    std::vector<QuantWeight> q_weights_;
    std::vector<LayerNorm> layer_norms_;

    // KV Cache (managed by ARM)
    struct KVCache {
        int32_t* key_cache;   // [layers, seq, heads, head_dim]
        int32_t* value_cache;
        int max_seq;
    };
    KVCache kv_cache_;

    // Scratch buffers (DDR)
    int8_t* act_buf_a_;   // 32MB activation buffer
    int8_t* act_buf_b_;   // 16MB activation buffer
    int32_t* act_buf_c_;  // 16MB output buffer

    // FPGA interface
    class FPGA& fpga_;
};

class LayerNormImpl {
public:
    static void forward(float* output, const float* input,
                       const float* gamma, const float* beta, int size);
};

class SoftmaxImpl {
public:
    static void forward(float* input, int size);
};

class ActivationImpl {
public:
    static void silu(float* data, int size);
    static void gelu(float* data, int size);
};

} // namespace tmac

#endif // T_MAC_RUNTIME_HPP