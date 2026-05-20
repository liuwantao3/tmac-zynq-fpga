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



struct TileCycleBudget {
    static constexpr uint32_t DSP_FILL_CYCLES    = 4;
    static constexpr uint32_t DSP_DRAIN_CYCLES    = 4;
    static constexpr uint32_t MODE_SWITCH_CYCLES = 2;
    static constexpr uint32_t COMPUTE_CYCLES        = 64;  // N×N systolic at 1 op/cycle

    // Total cycles for one tile pass through all 3 phases
    static constexpr uint32_t INT16_TILE_CYCLES =
        DSP_FILL_CYCLES + MODE_SWITCH_CYCLES + COMPUTE_CYCLES + DSP_DRAIN_CYCLES +  // quant
        DSP_FILL_CYCLES + MODE_SWITCH_CYCLES + COMPUTE_CYCLES + DSP_DRAIN_CYCLES +  // compute
        DSP_FILL_CYCLES + MODE_SWITCH_CYCLES + COMPUTE_CYCLES + DSP_DRAIN_CYCLES;   // dequant
    // = 10 + 10 + 10 = 30 cycles overhead + 192 compute = 202 cycles

    // Q8_0 path adds 4 extra cycles per phase (slower scale search on FPGA)
    static constexpr uint32_t Q8_TILE_CYCLES = INT16_TILE_CYCLES + 12;
    // = 214 cycles

    // For pure compute-only mode (if Q/D already done on ARM)
    static constexpr uint32_t COMPUTE_ONLY_CYCLES =
        DSP_FILL_CYCLES + MODE_SWITCH_CYCLES + COMPUTE_CYCLES + DSP_DRAIN_CYCLES;
    // = 10 cycles per tile

    static constexpr double US_PER_CYCLE = 6.67 / 1000.0;  // 150 MHz

    static double tile_us(int mode_q8) {
        return (mode_q8 ? Q8_TILE_CYCLES : INT16_TILE_CYCLES) * US_PER_CYCLE;
    }
};

// FP16→FP32 conversion (shared across simulation files)
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
        uint32_t cycles_per_tile = use_q8 ? TileCycleBudget::Q8_TILE_CYCLES
                                          : TileCycleBudget::COMPUTE_ONLY_CYCLES;
        total_cycles_ += num_tiles * cycles_per_tile;
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
// ARM side (CPU):   sends raw Q8_0 bytes (4 KB/tile) + row scales (56 B)
// FPGA side:        Q8_0→INT16 dequant + INT16 systolic matmul + dequant
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
constexpr int Q4K_TILE_BLOCKS = 16;           // 16 blocks per 64×64 tile
constexpr int Q4K_TILE_BYTES = Q4K_TILE_BLOCKS * Q4K_BLOCK_BYTES;  // 2304

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
constexpr uint32_t AXI_WEIGHT_Q4K_MAX = 0x3000; // weight_buf end (exclusive)
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
    uint8_t  weight_buf[8192] = {0};  // 0x1000-0x2FFF
    uint16_t scale_buf[128]   = {0};  // 0x2000-0x20FF (Q8 only)
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
            for (int b = 0; b < 4 && (off + b) < 8192; b++)
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

// Simulate the loading FSM + Q4K core compute in one step.
// This mirrors the Verilog: loading FSM → core start → compute → drain.
inline void axilite_q4k_run(AxiliteAccelState& s) {
    // Phase 1: Loading FSM — copy weight_buf → wmem, act_buf → cmem
    // Simulate Q4K core's wmem as int16_t[512][8] = 4096 INT16 values
    int16_t wmem[512][8] = {{0}};  // 512 entries × 8 INT16s
    for (int load_addr = 0; load_addr < 8192; load_addr++) {
        // Address format matches Verilog loading FSM:
        //   entry = load_addr / 16  (0-511)
        //   byte_lane = load_addr % 16  (0-15)
        //   byte_lane 0-7 → wmem_lo, 8-15 → wmem_hi
        //   Within each 64-bit half: byte_lane[2:0] selects byte position
        int entry = load_addr / 16;
        int byte_lane = load_addr % 16;
        int half = (byte_lane >= 8) ? 1 : 0;    // lo/hi 64-bit half
        int byte_in_half = byte_lane % 8;         // byte within 64-bit half
        int int16_idx = byte_in_half / 2;         // which INT16 in the half
        int byte_in_int16 = byte_in_half % 2;     // lo/hi byte of INT16

        uint8_t val = s.weight_buf[load_addr];
        int16_t entry_arr[8];  // 8 INT16s per wmem entry
        memcpy(entry_arr, &wmem[entry], 16);

        // Build the INT16 at position int16_idx within the half
        int idx_in_entry = half * 4 + int16_idx;  // 0-7
        if (byte_in_int16 == 0)
            entry_arr[idx_in_entry] = (entry_arr[idx_in_entry] & 0xFF00) | val;
        else
            entry_arr[idx_in_entry] = (entry_arr[idx_in_entry] & 0x00FF) | ((int16_t)val << 8);

        memcpy(&wmem[entry], entry_arr, 16);
    }

    // Phase 2: Compute — vecmul_1x64_int16 with wmem rearranged
    // wmem[entry][8] holds 8 INT16s per entry.
    // The tile is 64×64, col-major: tile[col][row] = wmem[(col*64+row)/8][(col*64+row)%8]
    int16_t mat[N][N] = {{0}};
    for (int c = 0; c < N; c++) {
        for (int r = 0; r < N; r++) {
            int idx = c * N + r;
            int entry = idx / 8;
            int pos = idx % 8;
            mat[c][r] = wmem[entry][pos];
        }
    }

    int16_t vec[N] = {0};
    for (int i = 0; i < N; i++) vec[i] = s.act_buf[i];

    acc16_t result[N] = {0};
    vecmul_1x64_int16(vec, mat, result);

    // Phase 3: Drain — write results to result_buf
    for (int i = 0; i < N; i++)
        s.result_buf[i] = (uint64_t)(int64_t)result[i];

    s.done_flag = true;
    s.reg_status = STATUS_DONE;
}

// High-level Q4K tile function using AXI-Lite buffer interface.
// Writes INT16 pre-dequant tile data via simulated buffer writes,
// triggers compute, reads results.
// This exactly mirrors the Verilog data flow in matmul_top.v + matmul_q4k_core.v.
//
// tile_int16: 64×64 INT16 matrix, col-major: tile_int16[col][row]
// vec:        64 INT16 activation vector
// result:     64 INT64 output
inline void axi_vecmul_tile_q4k_axilite(const int16_t tile_int16[N][N],
                                         const in16_t vec[N],
                                         acc16_t result[N]) {
    AxiliteAccelState s;

    // Write activation vector directly to act_buf (bypass axilite_write_buf
    // to avoid overwrite by weight writes at the same 0x2100-0x217F address range)
    for (int i = 0; i < N; i += 2) {
        s.act_buf[i]     = vec[i];
        s.act_buf[i + 1] = vec[i + 1];
    }

    // Write weight data: 64×64 INT16 = 4096 values × 2 bytes = 8192 bytes
    // AXI addresses: 0x1000-0x2FFF (contiguous, passes through scale/act hole in Q4K mode)
    // This overwrites weight_buf at 0x2100-0x217F (col 34) but act_buf retains activations
    // Write 2 INT16s per 32-bit word for efficiency:
    //   addr = 0x1000 + col*128 + row*2  (row even → word-aligned)
    //   data = tile_int16[col][row] (16b) | tile_int16[col][row+1] << 16 (16b)
    //   strb = 0xF  (write all 4 bytes)
    for (int c = 0; c < N; c++) {
        for (int r = 0; r < N; r += 2) {
            int byte_off = c * N * 2 + r * 2;  // 2 bytes per INT16, 2 INT16s per word
            uint32_t w0 = (uint16_t)(tile_int16[c][r]);
            uint32_t w1 = (uint16_t)(tile_int16[c][r + 1]);
            uint32_t data_word = w0 | (w1 << 16);
            axilite_write_buf(s, AXI_WEIGHT_BASE + byte_off, data_word, 0xF, true);
        }
    }

    // Run compute (simulates loading FSM + core)
    axilite_q4k_run(s);

    // Read results from result buffer
    for (int i = 0; i < N; i++) {
        uint32_t lo = axilite_read_buf(s, AXI_RES_ADDR(i));
        uint32_t hi = axilite_read_buf(s, AXI_RES_HI_ADDR(i));
        result[i] = (acc16_t)((uint64_t)hi << 32) | lo;
    }
}



} // namespace fpga_sim
