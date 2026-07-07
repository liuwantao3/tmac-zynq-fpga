#pragma once

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <thread>
#include <atomic>

struct Tensor;

namespace fpga_sim {

constexpr int N = 64;
constexpr int BLOCK = 8;
constexpr int NUM_BLOCKS = N / BLOCK;
constexpr int Q8_BLOCK_SIZE = 32;

// INT8 types (matches HLS ap_int<8>)
using in_t = int8_t;
using prod_t = int16_t;
using acc_t = int32_t;

// INT16 types (matches HLS ap_int<16>, ap_int<64>)
using in16_t = int16_t;
using acc16_t = int64_t;

// ===========================================================================
// Bit-accurate compute primitives (match HLS kernel logic)
// ===========================================================================

inline void vecmul_1x64(in_t vec[N], in_t M[N][N], acc_t result[N]) {
    acc_t row_acc[N] = {0};
    for (int k = 0; k < N; k++) {
        in_t vk = vec[k];
        for (int i = 0; i < N; i++)
            row_acc[i] += (acc_t)((prod_t)vk * (prod_t)M[k][i]);
    }
    for (int i = 0; i < N; i++)
        result[i] = row_acc[i];
}

inline void vecmul_1x64_int16(in16_t vec[N], in16_t M[N][N], acc16_t result[N]) {
    acc16_t row_acc[N] = {0};
    for (int k = 0; k < N; k++) {
        in16_t vk = vec[k];
        for (int i = 0; i < N; i++)
            row_acc[i] += (acc16_t)vk * (acc16_t)M[k][i];
    }
    for (int i = 0; i < N; i++)
        result[i] = row_acc[i];
}

// ===========================================================================
// AXI-Lite Register Map (matches Vitis HLS s_axilite for matmul_int8/16)
// ===========================================================================
// Offset  Name       Access  Description
// 0x00    AP_CTRL    R/W     [0]=ap_start(W), [1]=ap_done(R), [2]=ap_idle(R),
//                            [3]=ap_ready(R)
// 0x04    GIE        R/W     [0]=Global Interrupt Enable
// 0x08    IER        R/W     [0]=IP Interrupt Enable (ap_done)
// 0x0C    ISR        R/W     [0]=ap_done intr status (W1C: write 1 to clear)
// 0x10    CTRL_USER  R/W     Custom: [1]=int_enable, [3]=op_vecmul, [4]=mode_int16
// 0x14    STATUS     R       Custom: 0=IDLE, 1=RUNNING, 2=DONE, 3=ERROR
// 0x18    A_ADDR_LO  R/W     AXI MM base (low 32b) — unused in simulation
// 0x1C    A_ADDR_HI  R/W
// 0x20    B_ADDR_LO  R/W
// 0x24    B_ADDR_HI  R/W
// 0x28    C_ADDR_LO  R/W
// 0x2C    C_ADDR_HI  R/W
// 0x30    SIZE_REG   R/W     [23:16]=M, [15:8]=N, [7:0]=K

enum AxiReg : uint32_t {
    REG_AP_CTRL   = 0x00,
    REG_GIE       = 0x04,
    REG_IER       = 0x08,
    REG_ISR       = 0x0C,
    REG_CTRL_USER = 0x10,
    REG_STATUS    = 0x14,

    REG_SIZE      = 0x30,
};

constexpr uint32_t AP_START = 1u << 0;
constexpr uint32_t AP_DONE  = 1u << 1;
constexpr uint32_t AP_IDLE  = 1u << 2;
constexpr uint32_t AP_READY = 1u << 3;

constexpr uint32_t STATUS_IDLE    = 0;
constexpr uint32_t STATUS_RUNNING = 1;
constexpr uint32_t STATUS_DONE    = 2;
constexpr uint32_t STATUS_ERROR   = 3;

// CTRL_USER bit positions

constexpr uint32_t CTRL_OP_VECMUL   = 1u << 3;
constexpr uint32_t CTRL_MODE_INT16  = 1u << 4;

// Q8_0 direct path: FPGA receives Q8_0 bytes + scales, does Q8→INT16 internally
constexpr uint32_t CTRL_MODE_Q8PATH = 1u << 5;

// Q4_K direct path: FPGA receives Q4_K blocks, does Q4_K→INT16 internally
constexpr uint32_t CTRL_MODE_Q4K = 1u << 6;



// Matches actual Verilog timing: 1 IDLE exit + 512 COMPUTE + 1 DRAIN + 1 DRAIN2
constexpr uint32_t CYCLES_PER_TILE = 515;

// FP32↔FP16 conversion (shared across simulation files)
inline float read_f16(const uint8_t* data) {
    uint16_t raw = (uint16_t)data[0] | ((uint16_t)data[1] << 8);
    uint32_t sign = (raw >> 15) & 0x1;
    uint32_t exp = (raw >> 10) & 0x1F;
    uint32_t mant = raw & 0x3FF;
    if (exp == 0) {
        return (mant == 0) ? 0.0f : (sign ? -1.0f : 1.0f) * (mant / 1024.0f) * powf(2.0f, -14.0f);
    }
    if (exp == 31) return (mant == 0) ? (sign ? -INFINITY : INFINITY) : NAN;
    return (sign ? -1.0f : 1.0f) * (1.0f + mant / 1024.0f) * powf(2.0f, (int)exp - 15);
}

// FP32→FP16 (standard round-to-nearest-even, truncates mantissa)
inline uint16_t write_f16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 31) & 0x1;
    int32_t exp = ((int32_t)((u >> 23) & 0xFF)) - 127;
    uint32_t mant = u & 0x7FFFFF;
    if (exp >= 16) return (sign << 15) | (0x1F << 10);
    if (exp < -14) return sign << 15;
    uint32_t f16_exp = exp + 15;
    uint32_t f16_mant = mant >> 13;
    uint32_t rbit = (mant >> 12) & 0x1;
    if (rbit && (f16_mant & 0x1) == 0) f16_mant++; // round to nearest even
    if (f16_mant >= 0x400) { f16_mant = 0; f16_exp++; }
    return (sign << 15) | ((f16_exp & 0x1F) << 10) | (f16_mant & 0x3FF);
}




// ===========================================================================
// Timing accumulator (unchanged)
// ===========================================================================
struct TimingStats {
    int64_t total_tiles = 0;
    int64_t total_mac_ops = 0;
    double cpu_ms = 0;
    uint64_t total_fpga_cycles = 0;

    void reset() { total_tiles = 0; total_mac_ops = 0; cpu_ms = 0; total_fpga_cycles = 0; }

    void report() const {
        if (total_tiles == 0) return;
        double naive_gops = (cpu_ms > 0)
            ? total_mac_ops / (cpu_ms / 1000.0) / 1e9
            : 0;
        double fpga_cycles = total_fpga_cycles > 0 ? (double)total_fpga_cycles : (double)total_tiles * N;
        double fpga_time_us = fpga_cycles / 150.0;
        double speedup = (cpu_ms > 0)
            ? (cpu_ms * 1000.0) / fpga_time_us
            : 0;
        printf("\n[FPGA TIMING SUMMARY]\n");
        printf("  Tiles: %-8lld  MACs: %-10lld  CPU: %.2f ms  NAIVE: %.2f Gop/s\n",
               (long long)total_tiles, (long long)total_mac_ops, cpu_ms, naive_gops);
        printf("  FPGA: %.0f cycles @ 150 MHz = %.2f us  Speedup: %.1fx\n",
               fpga_cycles, fpga_time_us, speedup);
    }
};

inline TimingStats g_timing;

// ===========================================================================
// MatmulAccel — simulates the matmul_int8 / matmul_int16 HLS IP with AXI-Lite
// register interface, background compute thread, and interrupt generation.
//
// Usage (ARM-side):
//   MatmulAccel accel;
//   accel.write_reg(REG_CTRL_USER, CTRL_OP_VECMUL | CTRL_MODE_INT16);
//   accel.write_reg(REG_SIZE, (M<<16)|(N<<8)|K);
//   memcpy(accel.ddr(), ...);           // write A, B data to DDR
//   accel.write_reg(REG_GIE, 1);        // enable global interrupt
//   accel.write_reg(REG_IER, 1);        // enable IP interrupt
//   accel.write_reg(REG_AP_CTRL, AP_START);   // start
//   accel.wait_done();                  // poll or ISR
//   memcpy(out, accel.ddr() + 8192, ...);     // read C from DDR
// ===========================================================================
class MatmulAccel {
public:
    static constexpr uint32_t DDR_BYTES = 64 * 1024;

    MatmulAccel() {
        memset(regs_, 0, sizeof(regs_));
        regs_[REG_AP_CTRL / 4] = AP_IDLE | AP_READY;
        regs_[REG_STATUS / 4]  = STATUS_IDLE;
        fsm_thread_ = std::thread(&MatmulAccel::fsm_loop, this);
    }

    ~MatmulAccel() {
        stop_.store(true);
        if (fsm_thread_.joinable()) fsm_thread_.join();
    }

    // AXI-Lite write (ARM → FPGA)
    void write_reg(uint32_t byte_off, uint32_t val) {
        if (byte_off >= sizeof(regs_)) return;
        switch (byte_off) {
        case REG_AP_CTRL:
            if (val & AP_START) { regs_[byte_off / 4] |= AP_START; start_flag_.store(true); }
            else                 regs_[byte_off / 4] &= ~AP_START;
            break;
        case REG_STATUS:  break; // read-only
        case REG_ISR:     // W1C
            regs_[byte_off / 4] &= ~val;
            if (!(regs_[byte_off / 4] & 1)) intr_flag_.store(false);
            break;
        default:
            regs_[byte_off / 4] = val;
            break;
        }
    }

    uint32_t read_reg(uint32_t byte_off) const {
        if (byte_off >= sizeof(regs_)) return 0xFFFFFFFF;
        return regs_[byte_off / 4];
    }

    // Fast wait: spins on the done flag with yield
    bool wait_done(int timeout_ms = 5000) {
        auto deadline = std::chrono::steady_clock::now()
                      + std::chrono::milliseconds(timeout_ms);
        while (!done_flag_.load()) {
            if (std::chrono::steady_clock::now() >= deadline) return false;
            std::this_thread::yield();
        }
        done_flag_.store(false);
        return true;
    }



    // Total FPGA cycles consumed (for performance accounting)
    uint64_t total_cycles() const { return total_cycles_; }


    uint8_t* ddr() { return ddr_; }
    const uint8_t* ddr() const { return ddr_; }



    // Q8_0 direct path support
    void set_q8_path(bool enable) { q8_path_ = enable; }


private:
    mutable uint32_t regs_[64];
    uint8_t ddr_[DDR_BYTES];
    std::atomic<bool> intr_flag_{false};
    std::atomic<bool> start_flag_{false};
    std::atomic<bool> done_flag_{false};
    std::atomic<bool> stop_{false};
    std::thread fsm_thread_;

    uint64_t total_cycles_ = 0;
    bool q8_path_ = false;

    // Background thread: FPGA state machine
    void fsm_loop() {
        while (!stop_.load()) {
            if (start_flag_.load()) {
                start_flag_.store(false);

                regs_[REG_AP_CTRL / 4] &= ~(AP_IDLE | AP_DONE);
                regs_[REG_STATUS / 4]   = STATUS_RUNNING;

                run_compute();

                regs_[REG_AP_CTRL / 4] &= ~AP_START;
                regs_[REG_AP_CTRL / 4] |= AP_DONE | AP_READY;
                regs_[REG_STATUS / 4]   = STATUS_DONE;
                done_flag_.store(true);

                if ((regs_[REG_GIE / 4] & 1) && (regs_[REG_IER / 4] & 1)) {
                    regs_[REG_ISR / 4] |= 1;
                    intr_flag_.store(true);
                }
            }
            std::this_thread::yield();
        }
    }

    void run_compute() {
        uint32_t ctrl = regs_[REG_CTRL_USER / 4];
        uint32_t sz   = regs_[REG_SIZE / 4];
        uint32_t dimM = (sz >> 16) & 0xFF;
        uint32_t dimN = (sz >> 8)  & 0xFF;
        uint32_t dimK = sz & 0xFF;
        if (dimM == 0) dimM = 64; if (dimN == 0) dimN = 64; if (dimK == 0) dimK = 64;

        bool use_q8 = (ctrl & CTRL_MODE_Q8PATH) != 0 && q8_path_;

        if (ctrl & CTRL_MODE_INT16) {
            compute_int16(dimM, dimN, dimK, (ctrl & CTRL_OP_VECMUL) != 0, use_q8);
        } else {
            compute_int8(dimM, dimN, dimK, (ctrl & CTRL_OP_VECMUL) != 0);
        }

        // Account for tile compute cycles in performance model
        uint64_t num_tiles = ((dimM + N - 1) / N) * ((dimN + N - 1) / N);
        total_cycles_ += num_tiles * CYCLES_PER_TILE;
    }

    void compute_int16(uint32_t dimM, uint32_t dimN, uint32_t dimK, bool vecmul, bool q8_path) {
        in16_t* mA = (in16_t*)ddr_;
        in16_t* mB = (in16_t*)(ddr_ + 8192);
        acc16_t* mC = (acc16_t*)(ddr_ + 16384);
        memset(mC, 0, N * N * sizeof(acc16_t));

        // Q8 path: dequant INT8 weights + UQ8.8 scales → INT16 in-place
        if (q8_path) {
            // DDR layout:
            //   offset 8192:  q8_tile[64][64] raw INT8 bytes (col-major: [k][n])
            //   offset 12288: combined_scales[64][2] uint16_t UQ8.8
            // CRITICAL: q8_bytes and mB share offset 8192. mB[k][n] (int16_t, 2 bytes)
            //   overlaps with q8_bytes[k][n+1]. Writing mB[0][0] corrupts q8_bytes[0][1].
            //   Pre-load ALL q8 data into local array + pre-load combined scales.
            const uint8_t* q8_src = ddr_ + 8192;
            uint8_t q8_copy[N * N];
            for (int i = 0; i < N * N; i++) q8_copy[i] = q8_src[i];
            const uint16_t* combined_src = (const uint16_t*)(ddr_ + 8192 + 64 * 64);
            uint16_t combined[N * 2];
            for (int i = 0; i < N * 2; i++) combined[i] = combined_src[i];
            uint32_t max_k = (dimK > N) ? N : dimK;
            uint32_t max_n = (dimN > N) ? N : dimN;
            for (uint32_t k = 0; k < max_k; k++) {
                for (uint32_t n = 0; n < max_n; n++) {
                    int8_t q8_val = (int8_t)q8_copy[k * N + n];
                    uint16_t cs = combined[n * 2 + k / Q8_BLOCK_SIZE];
                    int32_t prod = (int32_t)q8_val * (int32_t)cs;
                    int16_t deq = (int16_t)(prod >> 8);
                    if (deq > 32767) deq = 32767;
                    if (deq < -32768) deq = -32768;
                    mB[k * dimN + n] = deq;
                }
            }
        }

        if (vecmul && dimM == 1) {
            in16_t vec[N] = {0};
            in16_t mat[N][N] = {{0}};
            for (uint32_t k = 0; k < dimK && k < N; k++) vec[k] = mA[k];
            for (uint32_t k = 0; k < dimK && k < N; k++)
                for (uint32_t n = 0; n < dimN && n < N; n++)
                    mat[k][n] = mB[k * dimN + n];
            acc16_t result[N] = {0};
            vecmul_1x64_int16(vec, mat, result);
            for (uint32_t n = 0; n < dimN && n < N; n++) mC[n] = result[n];
        } else {
            for (uint32_t p = 0; p < NUM_BLOCKS; p++)
                for (uint32_t q = 0; q < NUM_BLOCKS; q++)
                    for (uint32_t r = 0; r < NUM_BLOCKS; r++) {
                        in16_t aB[BLOCK][BLOCK] = {{0}}, bB[BLOCK][BLOCK] = {{0}};
                        acc16_t cB[BLOCK][BLOCK] = {{0}};
                        for (int i = 0; i < BLOCK; i++)
                            for (int k = 0; k < BLOCK; k++)
                                aB[i][k] = mA[(p * BLOCK + i) * dimN + (r * BLOCK + k)];
                        for (int k = 0; k < BLOCK; k++)
                            for (int j = 0; j < BLOCK; j++)
                                bB[k][j] = mB[(r * BLOCK + k) * dimN + (q * BLOCK + j)];
                        for (int t = 0; t < BLOCK; t++)
                            for (int i = 0; i < BLOCK; i++)
                                for (int j = 0; j < BLOCK; j++)
                                    cB[i][j] += (acc16_t)aB[i][t] * bB[t][j];
                        for (int i = 0; i < BLOCK; i++)
                            for (int j = 0; j < BLOCK; j++)
                                mC[(p * BLOCK + i) * dimN + (q * BLOCK + j)] += cB[i][j];
                    }
        }
    }

    void compute_int8(uint32_t dimM, uint32_t dimN, uint32_t dimK, bool vecmul) {
        in_t* mA = (in_t*)ddr_;
        in_t* mB = (in_t*)(ddr_ + 4096);
        acc_t* mC = (acc_t*)(ddr_ + 8192);
        memset(mC, 0, N * N * sizeof(acc_t));

        if (vecmul && dimM == 1) {
            in_t vec[N] = {0};
            in_t mat[N][N] = {{0}};
            for (uint32_t k = 0; k < dimK && k < N; k++) vec[k] = mA[k];
            for (uint32_t k = 0; k < dimK && k < N; k++)
                for (uint32_t n = 0; n < dimN && n < N; n++)
                    mat[k][n] = mB[k * dimN + n];
            acc_t result[N] = {0};
            vecmul_1x64(vec, mat, result);
            for (uint32_t n = 0; n < dimN && n < N; n++) mC[n] = result[n];
        } else {
            for (uint32_t p = 0; p < NUM_BLOCKS; p++)
                for (uint32_t q = 0; q < NUM_BLOCKS; q++)
                    for (uint32_t r = 0; r < NUM_BLOCKS; r++) {
                        in_t aB[BLOCK][BLOCK] = {{0}}, bB[BLOCK][BLOCK] = {{0}};
                        acc_t cB[BLOCK][BLOCK] = {{0}};
                        for (int i = 0; i < BLOCK; i++)
                            for (int k = 0; k < BLOCK; k++)
                                aB[i][k] = mA[(p * BLOCK + i) * dimN + (r * BLOCK + k)];
                        for (int k = 0; k < BLOCK; k++)
                            for (int j = 0; j < BLOCK; j++)
                                bB[k][j] = mB[(r * BLOCK + k) * dimN + (q * BLOCK + j)];
                        for (int t = 0; t < BLOCK; t++)
                            for (int i = 0; i < BLOCK; i++)
                                for (int j = 0; j < BLOCK; j++)
                                    cB[i][j] += (acc_t)aB[i][t] * bB[t][j];
                        for (int i = 0; i < BLOCK; i++)
                            for (int j = 0; j < BLOCK; j++)
                                mC[(p * BLOCK + i) * dimN + (q * BLOCK + j)] += cB[i][j];
                    }
        }
    }
};

// ===========================================================================
// High-level tile wrappers — used by tmac_gguf.cpp inference engine.
// Each call goes through the full AXI protocol: DDR copy → register write
// → START → wait DONE → result read. The FSM thread runs the systolic array.
// ===========================================================================

// Global singleton accelerator (shared by all inference calls)
inline MatmulAccel& accel() {
    static MatmulAccel instance;
    return instance;
}

// INT16 vecmul tile: vec[64] × W[64][64] → result[64] through AXI interface
inline void axi_vecmul_tile_int16(const in16_t vec[N], const in16_t W[N][N], acc16_t result[N]) {
    auto& a = accel();
    memcpy(a.ddr(), vec, N * sizeof(in16_t));
    memcpy(a.ddr() + 8192, W, N * N * sizeof(in16_t));
    a.write_reg(REG_CTRL_USER, CTRL_OP_VECMUL | CTRL_MODE_INT16);
    a.write_reg(REG_SIZE, (1 << 16) | (N << 8) | N);
    a.write_reg(REG_AP_CTRL, AP_START);
    a.wait_done();
    memcpy(result, a.ddr() + 16384, N * sizeof(acc16_t));
}

// ===========================================================================
// Q8_0 Direct Path — FPGA receives Q8_0 bytes + row scales, does Q8→INT16
// internally. This moves the dequantization workload from CPU to FPGA.
//
// ARM side (CPU):   sends raw Q8_0 bytes (4 KB/tile) + combined scales (256 B)
// FPGA side:        Q8_0→INT16 dequant + INT16×INT16 systolic matmul (DSP, 64×64 tile)
// Tile transfer:    ~6.9 µs vs INT16 ~13.5 µs (2× faster)
//
// Usage:
//   const uint8_t* q8_bytes = ...;  // 64×64 Q8_0 compressed bytes
//   float row_scales[14];           // per-row normalization scales
//   axi_vecmul_tile_q8(q8_bytes, row_scales, vec, result);
// ===========================================================================

// Q8_0 direct tile: FPGA receives pure INT8 weight bytes + combined UQ8.8 scales.
// ARM precomputes combined = block_scale / row_scale as UQ8.8 fixed-point.
// FPGA does: val = (q8_val * combined) >> 8 → single LUT-based integer multiply.
// combined_scales[128]: row-major, [row*2 + blk] for row in [0,N), blk in [0,2)
inline void axi_vecmul_tile_q8(const uint8_t* q8_W,                // 64×64 pure INT8
                                const uint16_t combined_scales[128], // precomputed UQ8.8
                                const in16_t vec[N],                // activation vector
                                acc16_t result[N]) {
    auto& a = accel();
    a.set_q8_path(true);

    // Write activation vector A to DDR offset 0
    memcpy(a.ddr(), vec, N * sizeof(in16_t));

    // Write pure INT8 weight bytes to B offset
    memcpy(a.ddr() + 8192, q8_W, N * N);

    // Write combined UQ8.8 fixed-point scales (128 × 2 bytes = 256 B)
    uint16_t* scale_base = (uint16_t*)(a.ddr() + 8192 + N * N);
    memcpy(scale_base, combined_scales, 128 * sizeof(uint16_t));

    // Set Q8 path mode
    a.write_reg(REG_CTRL_USER, CTRL_OP_VECMUL | CTRL_MODE_INT16 | CTRL_MODE_Q8PATH);
    a.write_reg(REG_SIZE, (1 << 16) | (N << 8) | N);
    a.write_reg(REG_AP_CTRL, AP_START);
    a.wait_done();
    memcpy(result, a.ddr() + 16384, N * sizeof(acc16_t));

    a.set_q8_path(false);
}



// ===========================================================================
// Q4_K Direct Path — FPGA receives Q4_K blocks, does Q4_K→INT16 internally.
// Tile: 64×64 weights = 16 Q4_K blocks × 144 bytes = 2304 bytes.
// ARM side (CPU): sends raw Q4_K blocks (2304 B/tile) + activation vector.
// FPGA side: Q4_K→INT16 dequant + INT16 matmul.
// ===========================================================================

// Q4_K block constants
constexpr int Q4K_BLOCK_SIZE = 256;           // 256 weights per block
constexpr int Q4K_BLOCK_BYTES = 144;          // 144 bytes per block
constexpr int Q4K_TILE_BLOCKS_OLD = 16;       // old 64×64 tile: 16 blocks
// New tile: 896 rows × 64 cols = 224 blocks (224 × 144 = 32256 bytes)
// 224 = 896 / 4 rows per block, 1 column group of 64 cols
constexpr int Q4K_TILE_BLOCKS = 224;          // 224 blocks per 896×64 tile
constexpr int Q4K_TILE_BYTES = Q4K_TILE_BLOCKS * Q4K_BLOCK_BYTES;  // 32256

// Q5_0 block constants (32 values/block, 22 bytes)
constexpr int Q5_0_BLOCK_SIZE = 32;
constexpr int Q5_0_BLOCK_BYTES = 22;
constexpr int Q5_0_224BLOCK_BYTES = 224 * Q5_0_BLOCK_BYTES;  // 4928

// HP FSM Q5_0 2-core 56-block interleaved format (48 bytes/block)
constexpr int Q5_0_2CORE_BLOCK_BYTES = 48;         // 22+22+4 padding
constexpr int Q5_0_56BLOCK_BYTES = 56 * Q5_0_2CORE_BLOCK_BYTES;  // 2688
constexpr int Q5_0_TILE_NORM_BYTES = 4 * 2;        // 8 (4 × UQ8.8)
constexpr int Q5_0_TILE_BYTES = Q5_0_56BLOCK_BYTES + Q5_0_TILE_NORM_BYTES;  // 2696

// Q6_K block constants (256 values/block, 210 bytes)
constexpr int Q6_K_BLOCK_SIZE = 256;
constexpr int Q6_K_BLOCK_BYTES = 210;
constexpr int Q6_K_32BLOCK_BYTES = 32 * Q6_K_BLOCK_BYTES;    // 6720

// Dequantize a single Q4_K block (144 bytes) → INT16[256]
// Block layout: 4 rows × 64 cols in column-major order within the tile
//   sub_block: 0-7, each covering 32 consecutive tensor indices
//   For a 64×64 tile: sub_block = row_in_block*2 + (col>=32 ? 1 : 0)
inline void dequant_q4k_block_to_int16(const uint8_t block[Q4K_BLOCK_BYTES],
                                        int16_t out[Q4K_BLOCK_SIZE]) {
    float d = read_f16(block);
    float dmin = read_f16(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;

    for (int sub = 0; sub < 8; sub++) {
        int sc, m;
        if (sub < 4) {
            sc = scales[sub] & 63;
            m = scales[sub + 4] & 63;
        } else {
            sc = (scales[sub + 4] & 0xF) | ((scales[sub - 4] >> 6) << 4);
            m = (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4);
        }
        sc = std::min(sc, 63);
        m = std::min(m, 63);

        for (int j = 0; j < 32; j++) {
            uint64_t qs_byte_idx = (sub / 2) * 32 + j;
            uint8_t q4 = (sub % 2 == 0) ? (qs[qs_byte_idx] & 0xF) : (qs[qs_byte_idx] >> 4);
            float val = d * (float)sc * (float)q4 - dmin * (float)m;
            int idx = sub * 32 + j;
            if (val >= 32767.0f) out[idx] = 32767;
            else if (val <= -32768.0f) out[idx] = -32768;
            else out[idx] = (int16_t)roundf(val);
        }
    }
}

// Dequantize a full 64×64 tile from 16 Q4_K blocks → INT16[64][64]
// Each block covers 4 rows × 64 cols in the tile (rows 4*BI..4*BI+3)
inline void dequant_q4k_tile(const uint8_t q4k_blocks[Q4K_TILE_BYTES],
                              int16_t tile[fpga_sim::N][fpga_sim::N]) {
    memset(tile, 0, fpga_sim::N * fpga_sim::N * sizeof(int16_t));
    int16_t block_out[Q4K_BLOCK_SIZE];
    for (int bi = 0; bi < Q4K_TILE_BLOCKS; bi++) {
        dequant_q4k_block_to_int16(q4k_blocks + bi * Q4K_BLOCK_BYTES, block_out);
        int row_base = bi * 4;  // 4 rows per block
        for (int idx = 0; idx < Q4K_BLOCK_SIZE; idx++) {
            int r = row_base + idx / 64;
            int c = idx % 64;
            if (r < fpga_sim::N)
                tile[c][r] = block_out[idx];  // tile[col][row] = mat[col][row]
        }
    }
}

// ===========================================================================
// AXI-Lite Buffer Simulation — matches matmul_top.v AXI-Lite buffer I/O.
//
// The buffer-based accelerator uses AXI-Lite writes to populate internal
// buffers at specific address ranges, then sets registers to trigger compute.
// Results are read back from result buffer addresses.
//
// AXI-Lite address map (matches verilog/matmul_top.v):
//   0x0000-0x00FF:  Control registers (AP_CTRL, GIE, IER, ISR, CTRL_USER, STATUS)
//   0x1000-0x1FFF:  Weight buffer lower 4KB (Q8: all, Q4K: first half)
//   0x2000-0x20FF:  Scale buffer (Q8 mode only)
//   0x2100-0x217F:  Activation buffer (64 × int16_t)
//   0x2200-0x2FFF:  Weight buffer upper (Q4K: second half)
//   0x4000-0x41FF:  Result buffer (64 × int48_t, lower 32b @ 0x4000, upper 16b @ 0x4200)
//
// Usage:
//   axilite_write_buf(AXI_WEIGHT_ADDR(offset), data, strb);  // write to weight_buf
//   axilite_write_buf(AXI_ACT_ADDR(i), act_buf[i], 0xF);     // write to act_buf
//   axilite_write_reg(REG_AP_CTRL, AP_START);                 // trigger compute
//   axilite_wait_done();
//   val = axilite_read_buf(AXI_RES_ADDR(i));                  // read result
// ===========================================================================

// AXI-Lite buffer address constants (matches Verilog localparams)
constexpr uint32_t AXI_WEIGHT_BASE  = 0x1000;  // weight_buf start
constexpr uint32_t AXI_WEIGHT_Q4K_MAX = 0x9000; // weight_buf end (0x1000 + 32768)
constexpr uint32_t AXI_SCALE_BASE   = 0x2000;  // scale_buf start
constexpr uint32_t AXI_ACT_BASE     = 0x2100;  // act_buf start
constexpr uint32_t AXI_RES_BASE     = 0x4000;  // result_buf lower 32b
constexpr uint32_t AXI_RES_HI_BASE  = 0x4200;  // result_buf upper 16b

// AXI address for result entry i, lower 32 bits (0-63)
inline uint32_t AXI_RES_ADDR(int i) {
    return AXI_RES_BASE + i * 4;
}

// AXI address for result entry i, upper 16 bits (0-63)
inline uint32_t AXI_RES_HI_ADDR(int i) {
    return AXI_RES_HI_BASE + i * 4;
}

// Simulated AXI-Lite buffer accelerator state
struct AxiliteAccelState {
    // Internal buffers (match Verilog exactly)
    // Q4K mode needs 32256 bytes (224 blocks × 144 bytes), 32768 provides margin
    uint8_t  weight_buf[32768] = {0}; // 0x1000-0x9FFF (Q4K: 32256 bytes used)
    uint16_t scale_buf[896]   = {0};  // Q8: 128 entries, Q4K: 896 entries for row_scale
    uint16_t act_buf[64]      = {0};  // 0x2100-0x217F
    uint64_t result_buf[64]   = {0};  // 0x4000-0x41FF (48-bit results)

    // Control registers
    uint32_t reg_ap_ctrl   = AP_IDLE | AP_READY;
    uint32_t reg_ctrl_user = 0;
    uint32_t reg_status    = STATUS_IDLE;

    // Simulated done flag
    bool done_flag = false;
};

// Write to AXI-Lite buffer: simulates write_buffer task in Verilog
// Writes 32-bit data to the correct buffer based on address, with byte strobes.
// In Q4K mode, weight_buf accepts writes across the full 0x1000-0x2FFF range
// (scale_buf/act_buf are unused in Q4K mode, so no aliasing issue).
inline void axilite_write_buf(AxiliteAccelState& s, uint32_t addr, uint32_t data,
                               uint32_t strb, bool mode_q4k = false) {
    // Scale buffer writes (Q8 mode only — Q4K uses this address range for weight data)
    if (!mode_q4k && addr >= AXI_SCALE_BASE && addr < AXI_SCALE_BASE + 256) {
        int off = addr - AXI_SCALE_BASE;
        int word_off = off & ~3;
        int idx = word_off / 2;
        if (idx < 127) {
            s.scale_buf[idx]     = (uint16_t)(data & 0xFFFF);
            s.scale_buf[idx + 1] = (uint16_t)((data >> 16) & 0xFFFF);
        }
    }
    // Weight buffer writes — exclusion guard matches Verilog
    if (addr >= AXI_WEIGHT_BASE && addr < AXI_WEIGHT_Q4K_MAX) {
        int off = addr - AXI_WEIGHT_BASE;
        bool is_scale_region = (addr >= AXI_SCALE_BASE && addr < AXI_SCALE_BASE + 256);
        bool is_act_region   = (addr >= AXI_ACT_BASE   && addr < AXI_ACT_BASE + 128);
        bool skip = false;
        if (!mode_q4k && (is_scale_region || is_act_region)) skip = true;
        if (!skip) {
            int wb_max = mode_q4k ? 32768 : 8192;
            for (int b = 0; b < 4 && (off + b) < wb_max; b++)
                if (strb & (1u << b)) s.weight_buf[off + b] = (data >> (b * 8)) & 0xFF;
        }
    }
}

// Read from AXI-Lite buffer
inline uint32_t axilite_read_buf(const AxiliteAccelState& s, uint32_t addr) {
    if (addr >= AXI_RES_BASE && addr < AXI_RES_BASE + 256) {
        int idx = (addr - AXI_RES_BASE) / 4;
        if (idx < 64) return (uint32_t)(s.result_buf[idx] & 0xFFFFFFFF);
    }
    if (addr >= AXI_RES_HI_BASE && addr < AXI_RES_HI_BASE + 256) {
        int idx = (addr - AXI_RES_HI_BASE) / 4;
        if (idx < 64) return (uint32_t)(s.result_buf[idx] >> 32);
    }
    return 0;
}

// Phase 1 INT16 run: reassemble pre-dequant INT16 from byte-serialized weight_buf,
// then matmul. This matches the Phase 1 Verilog loading FSM (LP_WEIGHT).
inline void axilite_int16_run(AxiliteAccelState& s) {
    int16_t wmem[512][8] = {{0}};
    for (int load_addr = 0; load_addr < 8192; load_addr++) {
        int entry = load_addr / 16;
        int byte_lane = load_addr % 16;
        int half = (byte_lane >= 8) ? 1 : 0;
        int byte_in_half = byte_lane % 8;
        int int16_idx = byte_in_half / 2;
        int byte_in_int16 = byte_in_half % 2;
        uint8_t val = s.weight_buf[load_addr];
        int16_t entry_arr[8];
        memcpy(entry_arr, &wmem[entry], 16);
        int idx_in_entry = half * 4 + int16_idx;
        if (byte_in_int16 == 0)
            entry_arr[idx_in_entry] = (entry_arr[idx_in_entry] & 0xFF00) | val;
        else
            entry_arr[idx_in_entry] = (entry_arr[idx_in_entry] & 0x00FF) | ((int16_t)val << 8);
        memcpy(&wmem[entry], entry_arr, 16);
    }

    int16_t mat[N][N] = {{0}};
    for (int c = 0; c < N; c++)
        for (int r = 0; r < N; r++) {
            int idx = c * N + r;
            mat[c][r] = wmem[idx / 8][idx % 8];
        }

    int16_t vec[N] = {0};
    for (int i = 0; i < N; i++) vec[i] = s.act_buf[i];

    acc16_t result[N] = {0};
    vecmul_1x64_int16(vec, mat, result);

    for (int i = 0; i < N; i++)
        s.result_buf[i] = (uint64_t)(int64_t)result[i];

    s.done_flag = true;
    s.reg_status = STATUS_DONE;
}

// Phase 2 raw block run: decode Q4_K blocks from weight_buf, apply row normalization
// via row_scale, then matmul.
// Each block covers 256 flat values = 1 row x 256 cols (for cols >= 256).
// act = activation vector (length ncols), ncols = number of columns to process per block.
//   ncols=64 for 64-col tiles (wi_offset selects which 64), ncols=256 for full-block tiles.
inline void axilite_q4k_run(AxiliteAccelState& s, const float* row_inv,
                            const int16_t* act, int ncols,
                            int nblocks = 56,
                            int64_t* tile_result = nullptr, int row_base = 0,
                            int wi_offset = 0) {
    int64_t tile_result_local[896] = {0};
    int64_t* result_out = tile_result ? tile_result : tile_result_local;

    for (int bi = 0; bi < nblocks; bi++) {
        const uint8_t* block_start = s.weight_buf + bi * Q4K_BLOCK_BYTES;

        float d = read_f16(block_start);
        float dmin = read_f16(block_start + 2);
        const uint8_t* scales = block_start + 4;
        const uint8_t* qs = block_start + 16;

        int out_row = row_base + bi;
        float inv = row_inv[out_row];

        int64_t acc = 0;
        for (int wio = 0; wio < ncols; wio++) {
            int wi = wi_offset + wio;
            if (wi >= 256) break;
            int sub = wi / 32;
            int j = wi % 32;

            int sc, m;
            if (sub < 4) {
                sc = scales[sub] & 63;
                m = scales[sub + 4] & 63;
            } else {
                sc = (scales[sub + 4] & 0xF) | ((scales[sub - 4] >> 6) << 4);
                m = (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4);
            }
            sc = std::min(sc, 63);
            m = std::min(m, 63);

            uint64_t qs_byte_idx = (sub / 2) * 32 + j;
            uint8_t q4 = (sub % 2 == 0) ? (qs[qs_byte_idx] & 0xF) : (qs[qs_byte_idx] >> 4);

            float val = (d * (float)sc * (float)q4 - dmin * (float)m) * inv;
            int16_t val_i16;
            if (val > 32767.0f) val_i16 = 32767;
            else if (val < -32768.0f) val_i16 = -32768;
            else val_i16 = (int16_t)roundf(val);

            acc += (int64_t)act[wio] * (int64_t)val_i16;
        }

        if (out_row < 896)
            result_out[out_row] += acc;
    }

    for (int i = 0; i < 64 && (row_base + i) < 896; i++)
        s.result_buf[i] = (uint64_t)(int64_t)result_out[row_base + i];

    s.done_flag = true;
    s.reg_status = STATUS_DONE;
}

// Phase 1 INT16 tile path: sends pre-dequant INT16 weights through AXI-Lite buffer.
// Used for backward compatibility testing.
inline void axi_vecmul_tile_int16_axilite(const int16_t tile_int16[N][N],
                                           const in16_t vec[N],
                                           acc16_t result[N]) {
    AxiliteAccelState s;
    for (int i = 0; i < N; i += 2) {
        s.act_buf[i]     = vec[i];
        s.act_buf[i + 1] = vec[i + 1];
    }
    for (int c = 0; c < N; c++) {
        for (int r = 0; r < N; r += 2) {
            int byte_off = c * N * 2 + r * 2;
            uint32_t w0 = (uint16_t)(tile_int16[c][r]);
            uint32_t w1 = (uint16_t)(tile_int16[c][r + 1]);
            uint32_t data_word = w0 | (w1 << 16);
            axilite_write_buf(s, AXI_WEIGHT_BASE + byte_off, data_word, 0xF, true);
        }
    }
    axilite_int16_run(s);
    for (int i = 0; i < N; i++) {
        uint32_t lo = axilite_read_buf(s, AXI_RES_ADDR(i));
        uint32_t hi = axilite_read_buf(s, AXI_RES_HI_ADDR(i));
        result[i] = (acc16_t)((uint64_t)hi << 32) | lo;
    }
}

// Phase 2 raw Q4_K block path: sends Q4_K blocks via simulated buffer writes,
// then runs FPGA internal block decode + row normalization + matmul.
// act = activation vector (length ncols), ncols = number of columns per tile (64 or 256).
// Produces nblocks output values per invocation.
inline void axi_vecmul_tile_q4k_axilite(const uint8_t* q4k_blocks,
                                         int nblocks,
                                         const in16_t* act, int ncols,
                                         const float* row_inv,
                                         int64_t* result,
                                         int row_base = 0) {
    AxiliteAccelState s;

    int tile_bytes = nblocks * Q4K_BLOCK_BYTES;
    for (int off = 0; off < tile_bytes; off += 4) {
        uint32_t data_word = (uint32_t)q4k_blocks[off]
                           | ((uint32_t)q4k_blocks[off + 1] << 8)
                           | ((uint32_t)q4k_blocks[off + 2] << 16)
                           | ((uint32_t)q4k_blocks[off + 3] << 24);
        axilite_write_buf(s, AXI_WEIGHT_BASE + off, data_word, 0xF, true);
    }

    axilite_q4k_run(s, row_inv, act, ncols, nblocks, result, row_base, 0);
}

// Overload: 16-block tile variant, 64 cols (backward compat for tests)
inline void axi_vecmul_tile_q4k_axilite(const uint8_t q4k_blocks[2304],
                                         const in16_t vec[64],
                                         acc16_t result[64]) {
    float row_inv[16];
    for (int i = 0; i < 16; i++) row_inv[i] = 1.0f;

    int64_t result_full[896] = {0};
    axi_vecmul_tile_q4k_axilite(q4k_blocks, 16, vec, 64, row_inv, result_full);

    for (int i = 0; i < 16; i++) result[i] = (int16_t)result_full[i];
}

// Overload: 16-block tile variant with row_scale, 64 cols
inline void axi_vecmul_tile_q4k_axilite(const uint8_t q4k_blocks[2304],
                                         const in16_t vec[64],
                                         const uint16_t row_scale_uq[64],
                                         acc16_t result[64]) {
    float row_inv[16];
    for (int i = 0; i < 16; i++) row_inv[i] = (float)row_scale_uq[i] / 256.0f;

    int64_t result_full[896] = {0};
    axi_vecmul_tile_q4k_axilite(q4k_blocks, 16, vec, 64, row_inv, result_full);

    for (int i = 0; i < 16; i++) result[i] = (int16_t)result_full[i];
}



// ===========================================================================
// Q4_K ROW2 Path — Direct 2-row Q4_K block decode + matmul with row_scale.
//
// Q4_K block-dequant values are very small (order 0.001-0.1), so direct
// INT16 rounding loses all precision (every value rounds to 0).
// Solution: 2-pass row_scale normalization.
//
// Pass 1: decode each block to float, find max_abs per row → row_scale
// Pass 2: decode each block, normalize by row_scale→INT16, MAC with xq
//
// Row normalization: y[i] = sum_k round(W_ik * 32767/row_max) * x_q_k
//                       ≈ (sum_k W_ik * x_k) * x_scale / row_scale  (after dequant)
// (The old pre-tiling path used same row_scale technique but with 64×64 tiles)
// ===========================================================================

// 2x896 tile: 7 Q4_K blocks → 1792 elements → 2 accumulator rows
// Matches matmul_q4k_2x896_core.v exactly.
constexpr int Q4K_7BLOCK_BYTES = 7 * Q4K_BLOCK_BYTES;  // 1008
constexpr int Q4K_28BLOCK_BYTES = 28 * Q4K_BLOCK_BYTES; // 4032

inline void axi_vecmul_tile_q4k_2x896_axilite(
    const uint8_t blocks_7[Q4K_7BLOCK_BYTES],
    const in16_t act_reg[896],
    const uint16_t row_scale[2],
    acc16_t result[2]) {
    constexpr int TILE_COLS = 896;
    int64_t acc[2] = {0};

    for (int bi = 0; bi < 7; bi++) {
        const uint8_t* blk = blocks_7 + bi * Q4K_BLOCK_BYTES;

        // S24.8 fixed-point decode
        int d_int = (int)roundf(read_f16(blk) * 256.0f);
        int dmin_int = (int)roundf(read_f16(blk + 2) * 256.0f);

        const uint8_t* scales = blk + 4;
        const uint8_t* qs = blk + 16;

        for (int wi = 0; wi < 256; wi++) {
            int sub = wi / 32;
            int j = wi % 32;

            int sc_used, m_used;
            if (sub < 4) {
                sc_used = scales[sub] & 63;
                m_used  = scales[sub + 4] & 63;
            } else {
                sc_used = (scales[sub - 4] >> 6) | ((scales[sub + 4] & 0xF) << 2);
                m_used  = (scales[sub] >> 6) | ((scales[sub + 4] >> 4) << 2);
            }

            int qs_idx = (sub / 2) * 32 + j;
            int q4 = (sub % 2 == 0) ? (qs[qs_idx] & 0xF) : (qs[qs_idx] >> 4);

            int a = d_int * sc_used;
            int b = dmin_int * m_used;
            int val = (a * q4 - b) >> 8;

            int flat = bi * 256 + wi;
            int row = flat / TILE_COLS;
            int col = flat % TILE_COLS;

            int val_norm = val * (int)row_scale[row] >> 8;
            if (val_norm > 32767) val_norm = 32767;
            else if (val_norm < -32768) val_norm = -32768;

            acc[row] += (int64_t)val_norm * (int64_t)act_reg[col];
        }
    }

    result[0] = acc[0];
    result[1] = acc[1];
}

// 8×896 variant: 28 blocks, 8-row accumulator. Same block decode logic.
inline void axi_vecmul_tile_q4k_8x896_axilite(
    const uint8_t blocks_28[Q4K_28BLOCK_BYTES],
    const in16_t act_reg[896],
    const uint16_t row_scale[8],
    acc16_t result[8]) {
    constexpr int TILE_COLS = 896;
    constexpr int NBLOCKS = 28;
    constexpr int NROWS = 8;
    int64_t acc[NROWS] = {0};

    for (int bi = 0; bi < NBLOCKS; bi++) {
        const uint8_t* blk = blocks_28 + bi * Q4K_BLOCK_BYTES;

        int d_int = (int)roundf(read_f16(blk) * 256.0f);
        int dmin_int = (int)roundf(read_f16(blk + 2) * 256.0f);

        const uint8_t* scales = blk + 4;
        const uint8_t* qs = blk + 16;

        for (int wi = 0; wi < 256; wi++) {
            int sub = wi / 32;
            int j = wi % 32;

            int sc_used, m_used;
            if (sub < 4) {
                sc_used = scales[sub] & 63;
                m_used  = scales[sub + 4] & 63;
            } else {
                sc_used = (scales[sub - 4] >> 6) | ((scales[sub + 4] & 0xF) << 2);
                m_used  = (scales[sub] >> 6) | ((scales[sub + 4] >> 4) << 2);
            }

            int qs_idx = (sub / 2) * 32 + j;
            int q4 = (sub % 2 == 0) ? (qs[qs_idx] & 0xF) : (qs[qs_idx] >> 4);

            int a = d_int * sc_used;
            int b = dmin_int * m_used;
            int val = (a * q4 - b) >> 8;

            int flat = bi * 256 + wi;
            int row = flat / TILE_COLS;
            int col = flat % TILE_COLS;

            int val_norm = val * (int)row_scale[row] >> 8;
            if (val_norm > 32767) val_norm = 32767;
            else if (val_norm < -32768) val_norm = -32768;

            acc[row] += (int64_t)val_norm * (int64_t)act_reg[col];
        }
    }

    for (int i = 0; i < NROWS; i++) result[i] = acc[i];
}

// =======================================================================
// Q5_0 8×896 core: 224 blocks (8 rows × 896 cols), 5-bit block decode.
// Each Q5_0 block = 32 vals, 22 bytes: FP16 d + 4B high bits + 16B nibbles.
// row_inv = FP32, 32767/max_abs per row.
// =======================================================================
inline void axi_vecmul_tile_q5_0_8x896_axilite(
    const uint8_t blocks_224[Q5_0_224BLOCK_BYTES],
    const in16_t act_reg[896],
    const float row_inv[8],
    int64_t* result,
    int row_base) {
    constexpr int TILE_COLS = 896;
    constexpr int NBLOCKS = 224;
    constexpr int NROWS = 8;
    int64_t acc[NROWS] = {0};

    // F16 dequantization: same as Q6_K/Q4_K (FP32, no fixed-point approximation)
    for (int bi = 0; bi < NBLOCKS; bi++) {
        const uint8_t* blk = blocks_224 + bi * Q5_0_BLOCK_BYTES;

        float d = read_f16(blk);

        // Load qh (4 bytes, little-endian)
        uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8) |
                      ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);

        for (int wi = 0; wi < 32; wi++) {
            int flat = bi * 32 + wi;
            int row = flat / TILE_COLS;
            int col = flat % TILE_COLS;

            int64_t j = wi < 16 ? wi : wi - 16;
            uint8_t qs_byte = blk[6 + j];
            uint8_t ql = (wi < 16) ? (qs_byte & 0xF) : (qs_byte >> 4);
            uint8_t qh_bit = (qh >> wi) & 1;
            int q5 = ((qh_bit << 4) | ql) - 16;

            float val_f = d * (float)q5;
            float norm = val_f * row_inv[row];

            int16_t val_i16;
            if (norm > 32767.0f) val_i16 = 32767;
            else if (norm < -32768.0f) val_i16 = -32768;
            else val_i16 = (int16_t)roundf(norm);

            acc[row] += (int64_t)val_i16 * (int64_t)act_reg[col];
        }
    }

    for (int i = 0; i < NROWS; i++) {
        int r = row_base + i;
        result[r] += acc[i];
    }
}

// =======================================================================
// Q6_K axilite path: decode Q6_K blocks, normalize by FP32 row_inv, MAC.
// Blocks cover 256 values each, block stride = cols/256.
// =======================================================================
inline void axi_vecmul_tile_q6_k_axilite(const uint8_t* blocks,
                                          int nblocks,
                                          const in16_t* act, int ncols,
                                          const float* row_inv,
                                          int64_t* result,
                                          int row_base) {
    constexpr int BLOCK_VALS = 256;

    for (int bi = 0; bi < nblocks; bi++) {
        const uint8_t* blk = blocks + bi * Q6_K_BLOCK_BYTES;

        float super_scale = read_f16(blk + 208);

        int out_row = row_base + bi;
        float inv = row_inv[out_row];
        int64_t acc = 0;

        for (int wi = 0; wi < BLOCK_VALS; wi++) {
            if (wi >= ncols) break;
            int half = wi / 128;
            int pos = wi % 128;
            int l = pos % 32;
            int sub = pos / 32;

            // L: bytes [0-127] — low 4 bits of each value
            //   half 0: blk[0..63], half 1: blk[64..127]
            //   sub 0,2: l, sub 1,3: l+32 within the half
            int L_off = half * 64 + l + (sub % 2) * 32;
            uint8_t ql_byte = blk[L_off];
            uint8_t ql_nibble = (sub < 2) ? (ql_byte & 0xF) : (ql_byte >> 4);

            // H: bytes [128-191] — high 2 bits, 4 sub-blocks × 2 bits per byte
            //   half 0: blk[128..159], half 1: blk[160..191]
            int H_off = 128 + half * 32 + l;
            int qh_shift = sub * 2;
            uint8_t qh_bits = (blk[H_off] >> qh_shift) & 0x3;

            int q6 = ((qh_bits << 4) | ql_nibble) - 32;

            // Scales: bytes [192-207] — int8, 16 per block (2 per sub-block)
            //   half 0: blk[192..199], half 1: blk[200..207]
            //   sub 0,1: (l/16), sub 2,3: (l/16) + offset by sub*2
            int sc_off = 192 + half * 8 + (l / 16) + sub * 2;
            float scale_f = (float)(int8_t)blk[sc_off];

            float val = super_scale * scale_f * (float)q6 * inv;
            int16_t val_i16;
            if (val > 32767.0f) val_i16 = 32767;
            else if (val < -32768.0f) val_i16 = -32768;
            else val_i16 = (int16_t)roundf(val);

            acc += (int64_t)act[wi] * (int64_t)val_i16;
        }

        if (out_row < 896)
            result[out_row] += acc;
    }
}

// Decode a Q4_K block to float array (no rounding, unlike block_to_int16)
inline void dequant_q4k_block_to_float(const uint8_t block[Q4K_BLOCK_BYTES],
                                        float out[Q4K_BLOCK_SIZE]) {
    float d = read_f16(block);
    float dmin = read_f16(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;
    for (int sub = 0; sub < 8; sub++) {
        int sc, m;
        if (sub < 4) {
            sc = scales[sub] & 63;
            m = scales[sub + 4] & 63;
        } else {
            sc = (scales[sub + 4] & 0xF) | ((scales[sub - 4] >> 6) << 4);
            m = (scales[sub + 4] >> 4) | ((scales[sub] >> 6) << 4);
        }
        sc = std::min(sc, 63);
        m = std::min(m, 63);
        for (int j = 0; j < 32; j++) {
            uint64_t qs_byte_idx = (sub / 2) * 32 + j;
            uint8_t q4 = (sub % 2 == 0) ? (qs[qs_byte_idx] & 0xF) : (qs[qs_byte_idx] >> 4);
            int idx = sub * 32 + j;
            out[idx] = d * (float)sc * (float)q4 - dmin * (float)m;
        }
    }
}

// Simulate FPGA Q4_K → row-normalized INT16 decode + matmul for a 2-row chunk.
// Uses row_scale normalization: each weight is quantized as round(val * 32767/row_scale[row].
// Single-pass decode+normalize+MAC (row_scale precomputed externally in matmul_fpga_q4k).
// tensor_data: pointer to tensor's raw Q4_K block data
// cols: number of columns per row (hidden dim or intermediate dim)
// row_start: starting row index (must be even)
// xq: quantized activation vector (length cols)
// result: output [result[row_start], result[row_start+1]]
// row_scale: max_abs per row from external pass-1 scan
inline void axi_vecmul_q4k_row2(const uint8_t* tensor_data, int cols,
                                 int row_start,
                                 const int16_t* xq,
                                 acc16_t result[2],
                                 const float row_scale[2]) {
    uint64_t nblocks = ((uint64_t)cols * 2 + Q4K_BLOCK_SIZE - 1) / Q4K_BLOCK_SIZE;
    uint64_t base_block = (uint64_t)row_start * cols / Q4K_BLOCK_SIZE;
    uint64_t row_start_flat = (uint64_t)row_start * cols;

    float row_inv[2];
    for (int i = 0; i < 2; i++) {
        float rs = (row_scale[i] < 1e-10f) ? 1.0f : row_scale[i] / 32767.0f;
        row_inv[i] = (rs < 1e-10f) ? 0.0f : 1.0f / rs;
    }

    int64_t acc[2] = {0};
    float block_f32[Q4K_BLOCK_SIZE];

    for (uint64_t b = 0; b < nblocks; b++) {
        dequant_q4k_block_to_float(tensor_data + (base_block + b) * Q4K_BLOCK_BYTES, block_f32);
        for (int i = 0; i < Q4K_BLOCK_SIZE; i++) {
            uint64_t flat_idx = (base_block + b) * Q4K_BLOCK_SIZE + i;
            uint64_t rel_idx = flat_idx - row_start_flat;
            if (rel_idx >= (uint64_t)cols * 2) break;
            int row = (int)(rel_idx / cols);
            int col = (int)(rel_idx % cols);
            int16_t w_i16 = (int16_t)roundf(block_f32[i] * row_inv[row]);
            acc[row] += (int64_t)w_i16 * (int64_t)xq[col];
        }
    }

    result[0] = acc[0];
    result[1] = acc[1];
}

// =======================================================================
// HP FSM Q5_0 4×896 model: 56 blocks (2 cores × 2 rows), interleaved
// 48-byte per-block format: core0_d+qh+qs + core1_d+qh+qs + padding.
// row_norm = UQ8.8 per row (4 values).
// Matches verilog/matmul_q5_0_core.v (2-core, hp_fsm_top.v dispatch).
// =======================================================================
inline void axi_vecmul_tile_q5_0_4x896_axilite(
    const uint8_t blocks_56[Q5_0_56BLOCK_BYTES],
    const in16_t act_reg[896],
    const float row_norm[4],
    int64_t* result,
    int row_base) {
    constexpr int NROWS = 4;
    constexpr int NBLOCKS = 56;
    constexpr int BLOCK_COLS = 32;
    constexpr int COLS = 896;
    constexpr int ROW_BLOCKS = COLS / BLOCK_COLS;  // 28
    int64_t acc[NROWS] = {0};

    for (int bi = 0; bi < NBLOCKS; bi++) {
        const uint8_t* blk = blocks_56 + bi * Q5_0_2CORE_BLOCK_BYTES;

        float d0 = read_f16(blk);
        uint32_t qh0;
        memcpy(&qh0, blk + 2, 4);

        float d1 = read_f16(blk + 22);
        uint32_t qh1;
        memcpy(&qh1, blk + 24, 4);

        int row0 = (bi < ROW_BLOCKS) ? 0 : 1;  // core0 row within tile
        int row1 = (bi < ROW_BLOCKS) ? 2 : 3;  // core1 row within tile
        int col_base = (bi % ROW_BLOCKS) * BLOCK_COLS;

        for (int wi = 0; wi < BLOCK_COLS; wi++) {
            int col = col_base + wi;

            int j = wi < 16 ? wi : wi - 16;
            uint8_t qs_byte0 = blk[6 + j];
            uint8_t ql0 = (wi < 16) ? (qs_byte0 & 0xF) : (qs_byte0 >> 4);
            uint8_t qh_bit0 = (qh0 >> wi) & 1;
            int q5_0 = ((qh_bit0 << 4) | ql0) - 16;

            uint8_t qs_byte1 = blk[28 + j];
            uint8_t ql1 = (wi < 16) ? (qs_byte1 & 0xF) : (qs_byte1 >> 4);
            uint8_t qh_bit1 = (qh1 >> wi) & 1;
            int q5_1 = ((qh_bit1 << 4) | ql1) - 16;

            float norm0 = d0 * (float)q5_0 * row_norm[row0];
            float norm1 = d1 * (float)q5_1 * row_norm[row1];

            int16_t v0 = (norm0 > 32767.0f) ? 32767 : (norm0 < -32768.0f ? -32768 : (int16_t)roundf(norm0));
            int16_t v1 = (norm1 > 32767.0f) ? 32767 : (norm1 < -32768.0f ? -32768 : (int16_t)roundf(norm1));

            acc[row0] += (int64_t)v0 * (int64_t)act_reg[col];
            acc[row1] += (int64_t)v1 * (int64_t)act_reg[col];
        }
    }

    for (int i = 0; i < NROWS; i++)
        result[row_base + i] += acc[i];
}

// ===========================================================================
// Phase B Descriptor Chain — matches verilog/matmul_top.v descriptor format
// ===========================================================================
#pragma pack(push, 1)
struct PhaseBDescriptor {
    uint32_t next_desc_addr;   // 0x00
    uint32_t weight_addr;      // 0x04
    uint32_t act_addr;         // 0x08
    uint32_t result_addr;      // 0x0C
    uint16_t num_tiles;        // 0x10
    uint16_t tile_bytes;       // 0x12
    uint8_t  tensor_type;      // 0x14
    uint8_t  tile_res_rows;    // 0x15
    uint8_t  flags;            // 0x16  [0]=interrupt
    uint16_t act_total_bytes;  // 0x17-0x18
    uint8_t  num_col_groups;   // 0x19
    uint8_t  reserved[6];      // 0x1A-0x1F
};
#pragma pack(pop)
static_assert(sizeof(PhaseBDescriptor) == 32, "PhaseBDescriptor must be 32 bytes");

// Tensor type constants for descriptor (matches TENSOR_Q*)
constexpr uint8_t DESC_TYPE_INT16 = 0;
constexpr uint8_t DESC_TYPE_Q5_0  = 6;
constexpr uint8_t DESC_TYPE_Q8_0  = 8;
constexpr uint8_t DESC_TYPE_Q4_K  = 12;
constexpr uint8_t DESC_TYPE_Q6_K  = 14;
constexpr uint8_t DESC_TYPE_CPU_OP = 15; // CPU-only op: FPGA signals CPU and pauses
constexpr uint8_t DESC_FLAG_INTERRUPT = 0x01;

// Phase B tile constants (block data + scale data for HP burst)
constexpr int PHASEB_Q5_0_ROWS        = 8;
constexpr int PHASEB_Q5_0_COLS        = 896;
constexpr int PHASEB_Q5_0_BLK_BYTES   = 224 * 22;  // 4928
constexpr int PHASEB_Q5_0_SCL_BYTES   = 8 * 2;     // 16  (row_inv UQ16.8)
constexpr int PHASEB_Q5_0_TILE_BYTES  = PHASEB_Q5_0_BLK_BYTES + PHASEB_Q5_0_SCL_BYTES; // 4944

constexpr int PHASEB_Q6_K_ROWS        = 32;
constexpr int PHASEB_Q6_K_COLS        = 256;
constexpr int PHASEB_Q6_K_BLK_BYTES   = 32 * 210;  // 6720
constexpr int PHASEB_Q6_K_SCL_BYTES   = 32 * 4;     // 128 (row_inv UQ24.8)
constexpr int PHASEB_Q6_K_TILE_BYTES  = 6848;

constexpr int PHASEB_Q4_K_ROWS        = 56;
constexpr int PHASEB_Q4_K_COLS        = 256;
constexpr int PHASEB_Q4_K_BLK_BYTES   = 56 * 144;  // 8064
constexpr int PHASEB_Q4_K_SCL_BYTES   = 56 * 4;     // 224 (row_inv UQ24.8)
constexpr int PHASEB_Q4_K_TILE_BYTES  = 8288;

constexpr int PHASEB_Q8_0_ROWS        = 64;
constexpr int PHASEB_Q8_0_COLS        = 64;
constexpr int PHASEB_Q8_0_BLK_BYTES   = 4096;
constexpr int PHASEB_Q8_0_SCL_BYTES   = 128 * 2;   // combined scales UQ8.8
constexpr int PHASEB_Q8_0_TILE_BYTES  = 4352;

constexpr int PHASEB_INT16_ROWS       = 64;
constexpr int PHASEB_INT16_COLS       = 64;
constexpr int PHASEB_INT16_BLK_BYTES  = 8192;
constexpr int PHASEB_INT16_SCL_BYTES  = 0;
constexpr int PHASEB_INT16_TILE_BYTES = 8192;

} // namespace fpga_sim
