// Q8_0 Logits Path — C++ Simulation of FPGA Q8→INT16 Matmul
// Used for token embeddings (logits layer).
// Calls matmul_fpga_q8 (defined in tmac_gguf.cpp) which simulates FPGA
// Q8_0 dequant + INT16×INT16 matmul pipeline.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include "fpga_sim.hpp"

struct Tensor {
    char name[128];
    uint64_t rows, cols;
    uint32_t type;
    uint64_t n_bytes;
    uint8_t* data;
};

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