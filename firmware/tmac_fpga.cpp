/*
 * T-MAC FPGA Interface Implementation
 * Zynq 7010 ARM-side driver for matmul_q8 IP
 *
 * ARM prepares: Q8_0 weight bytes + UQ8.8 combined scales + INT16 activations
 * FPGA does: Q8→INT16 dequant (LUT) + INT16×INT16 systolic matmul (DSP)
 */

#include "tmac_fpga.hpp"
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <pthread.h>

namespace tmac {

static volatile uint32_t* s_ctrl_base = nullptr;
static int s_fd = -1;

TMacFPGA::TMacFPGA() {
    // mmap is deferred to init()
}

TMacFPGA::~TMacFPGA() {
    if (s_fd >= 0) {
        close(s_fd);
        s_fd = -1;
        s_ctrl_base = nullptr;
    }
}

int TMacFPGA::init() {
    if (s_ctrl_base != nullptr) {
        return 0; // already initialized
    }

    // Open /dev/mem for direct register access
    s_fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (s_fd < 0) {
        perror("open /dev/mem");
        return -1;
    }

    // Map control registers (AXI GP0)
    s_ctrl_base = (volatile uint32_t*)mmap(
        nullptr, 4096,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        s_fd,
        CTRL_REG & ~0xFFF // page-aligned
    );

    if (s_ctrl_base == MAP_FAILED) {
        perror("mmap control registers");
        close(s_fd);
        s_fd = -1;
        return -1;
    }

    // Offset to our registers within the page
    // Already mapped at CTRL_REG page boundary

    printf("[T-MAC FPGA] Initialized\n");
    printf("[T-MAC FPGA] Control reg base: %p\n", s_ctrl_base);

    return 0;
}

void TMacFPGA::write_reg(uint32_t offset, uint32_t value) {
    if (s_ctrl_base == nullptr) {
        fprintf(stderr, "[T-MAC FPGA] Not initialized!\n");
        return;
    }
    s_ctrl_base[offset / sizeof(uint32_t)] = value;
}

uint32_t TMacFPGA::read_reg(uint32_t offset) {
    if (s_ctrl_base == nullptr) {
        return 0xFFFFFFFF;
    }
    return s_ctrl_base[offset / sizeof(uint32_t)];
}

void TMacFPGA::start_compute() {
    uint32_t ctrl = read_reg(0);
    ctrl |= 0x1; // set start bit
    write_reg(0, ctrl);
}

bool TMacFPGA::poll_status(uint32_t expected, int timeout_ms) {
    int waited = 0;
    while (waited < timeout_ms) {
        uint32_t status = read_reg(sizeof(uint32_t)); // status at offset 4
        if (status == expected) {
            return true;
        }
        usleep(100); // 100us
        waited += 100;
    }
    return false;
}

void TMacFPGA::wait_done() {
    poll_status(STATUS_DONE, 5000); // 5s timeout
}

void TMacFPGA::reset() {
    uint32_t ctrl = read_reg(0);
    ctrl &= ~0x1; // clear start bit
    ctrl |= 0x10; // software reset
    write_reg(0, ctrl);
    usleep(1000);
    ctrl &= ~0x10;
    write_reg(0, ctrl);
}

int TMacFPGA::gemm_q8(uint8_t* q8_weights, uint16_t* combined_scales,
                      int16_t* activation, int64_t* result, int M, int N, int K,
                      const MatmulConfig& config) {
    // matmul_q8 kernel processes 64×64 tiles
    // Each tile: Q8 weight bytes (4 KB) + combined scales (256 B)
    //          + INT16 activation (128 B) → INT64 result (512 B)
    //
    // Data flow:
    //   1. Write tile data to AXI master buffers in DDR
    //   2. Set AXI master base addresses via auto-generated address regs
    //   3. Set control register (int_enable, op_vecmul)
    //   4. Write AP_START
    //   5. Poll STATUS_DONE or wait for interrupt
    //   6. Read result from Y buffer
    //
    // ARM-side scale precompute (before calling this):
    //   combined = block_scale / row_scale  (as UQ8.8 fixed-point)
    //
    // (Implementation placeholder — actual register offsets and DDR layout
    //  depend on Vitis HLS IP integration in the Vivado block design.)

    uint32_t ctrl = 0;
    ctrl |= (config.is_vecmul ? CTRL_OP_VECMUL : 0);
    ctrl |= (config.int_enable ? CTRL_INT_ENABLE : 0);
    write_reg(0, ctrl);

    start_compute();
    wait_done();
    return 0;
}

int TMacFPGA::gemv_q8(int16_t* vec, uint8_t* q8_weights, uint16_t* scales,
                       int64_t* result, int N) {
    MatmulConfig config;
    config.is_vecmul = true;
    config.int_enable = true;
    // VecMul: M=1, N=N, K=N (vec @ q8_weight_matrix)
    return gemm_q8(q8_weights, scales, vec, result, 1, N, N, config);
}

} // namespace tmac