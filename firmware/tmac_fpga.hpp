/*
 * T-MAC FPGA Interface Layer
 * Zynq 7010 ARM-side driver for matmul_q8 IP
 *
 * FPGA Design: Q8_0 direct path — receives Q8_0 bytes + UQ8.8 combined scales,
 * does Q8→INT16 dequant (LUT-based), then INT16×INT16→INT64 systolic matmul.
 *
 * ARM handles: FP32→INT16 quantization of activations, combined scale precompute.
 * FPGA handles: Q8→INT16 dequant, INT16 systolic array, output accumulation.
 *
 * Usage:
 *   TMacFPGA fpga;
 *   fpga.init();
 *   fpga.gemm(q8_weights, combined_scales, activation, result, config);
 *   fpga.gemv(activation, q8_weights, result, N);
 */

#ifndef T_MAC_FPGA_HPP
#define T_MAC_FPGA_HPP

#include <cstdint>
#include <cstring>
#include <atomic>

namespace tmac {

// matmul_q8 Vitis HLS auto-generated AXI-Lite register map
// 0x00: AP_CTRL     [0]=ap_start, [1]=ap_done, [2]=ap_idle, [3]=ap_ready
// 0x04: GIE         [0]=Global Interrupt Enable
// 0x08: IER         [0]=IP Interrupt Enable
// 0x0C: ISR         [0]=ap_done interrupt status (W1C)
// 0x10: A_ADDR_LO   Q8 weight bytes base addr (aximm0)
// 0x14: A_ADDR_HI
// 0x1C: combined_scales_ADDR_LO (aximm1)
// 0x20: combined_scales_ADDR_HI
// 0x28: X_ADDR_LO   Activation vector base addr (aximm2)
// 0x2C: X_ADDR_HI
// 0x34: Y_ADDR_LO   Output vector base addr (aximm3)
// 0x38: Y_ADDR_HI
constexpr uint32_t CTRL_REG    = 0x40000000;

// Control register custom bit positions
constexpr uint32_t CTRL_INT_ENABLE  = 1u << 1;
constexpr uint32_t CTRL_OP_VECMUL   = 1u << 3;

constexpr uint32_t STATUS_IDLE    = 0;
constexpr uint32_t STATUS_RUNNING = 1;
constexpr uint32_t STATUS_DONE    = 2;
constexpr uint32_t STATUS_ERROR   = 3;

// Constants for matmul_q8 kernel (N=64 fixed)
constexpr int MATMUL_N = 64;
constexpr int Q8_BLOCK_SIZE = 32;
constexpr int Q8_BLOCK_BYTES = 34;

// DDR memory map (512 MB total)
constexpr uint32_t DDR_WEIGHT_BASE = 0x08000000;  // Model weights
constexpr uint32_t DDR_SCRATCH_BASE = 0x01000000; // Activation buffers

struct MatmulConfig {
    bool is_vecmul;     // false=MatMul, true=VecMul
    bool int_enable;    // Interrupt enable
};

class TMacFPGA {
public:
    TMacFPGA();
    ~TMacFPGA();

    int init();
    int gemm_q8(uint8_t* q8_weights, uint16_t* combined_scales,
                int16_t* activation, int64_t* result, int M, int N, int K,
                const MatmulConfig& config);
    int gemv_q8(int16_t* vec, uint8_t* q8_weights, uint16_t* scales,
                int64_t* result, int N);
    void wait_done();
    void reset();

private:
    void write_reg(uint32_t offset, uint32_t value);
    uint32_t read_reg(uint32_t offset);
    void start_compute();
    bool poll_status(uint32_t expected, int timeout_ms = 1000);
};

class ScopedTimer {
public:
    ScopedTimer(const char* name) : name_(name), start_(clock()) {}
    ~ScopedTimer() {
        clock_t end = clock();
        double ms = 1000.0 * (end - start_) / CLOCKS_PER_SEC;
        printf("[TIMER] %s: %.2f ms\n", name_, ms);
    }
private:
    const char* name_;
    clock_t start_;
};

} // namespace tmac

#endif // T_MAC_FPGA_HPP