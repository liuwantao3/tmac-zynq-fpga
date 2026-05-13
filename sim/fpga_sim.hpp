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
    REG_A_ADDR_LO = 0x18,
    REG_B_ADDR_LO = 0x20,
    REG_C_ADDR_LO = 0x28,
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
constexpr uint32_t CTRL_INT_ENABLE  = 1u << 1;
constexpr uint32_t CTRL_OP_VECMUL   = 1u << 3;
constexpr uint32_t CTRL_MODE_INT16  = 1u << 4;

// Q8_0 direct path: FPGA receives Q8_0 bytes + scales, does Q8→INT16 internally
constexpr uint32_t CTRL_MODE_Q8PATH = 1u << 5;

// ===========================================================================
// Phase-Accurate DSP Timing Model
//
// Models the time-multiplexed DSP pipeline within a single 64-PE systolic array.
// Each tile goes through 3 phases: QUANT → COMPUTE → DEQUANT
// The same 64 DSPs are reused across phases with pipeline flush overhead.
//
// Phase timing at 150 MHz (6.67 ns/cycle):
//   DSP_FILL_CYCLES:     4 cycles  (~27 ns) — pipeline fills
//   DSP_DRAIN_CYCLES:    4 cycles  (~27 ns) — pipeline drains
//   MODE_SWITCH_CYCLES:  2 cycles  (~13 ns) — LUT muxes select new mode
//   COMPUTE_CYCLES:     64 cycles (~427 ns) — systolic array active
// ===========================================================================

enum class DspPhase : uint8_t {
    QUANT_PHASE   = 0,
    COMPUTE_PHASE = 1,
    DEQUANT_PHASE = 2
};

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

// Pipeline simulation statistics
struct PipelineStats {
    double cpu_prep_ms = 0;      // total CPU prep wall time
    double fpga_sleep_ms = 0;    // total FPGA sleep time
    double wall_ms = 0;          // total end-to-end wall time
    int64_t cpu_tiles = 0;       // tiles pushed by CPU
    int64_t fpga_tiles = 0;      // tiles processed by FPGA

    void report(const char* label) const {
        printf("\n[PIPELINE %s]\n", label);
        printf("  CPU prep:  %.2f ms (%lld tiles)\n", cpu_prep_ms, (long long)cpu_tiles);
        printf("  FPGA:      %.2f ms (%lld tiles)\n", fpga_sleep_ms, (long long)fpga_tiles);
        printf("  Wall:      %.2f ms\n", wall_ms);
        if (cpu_tiles > 0) {
            double cpu_us_per_tile = cpu_prep_ms * 1000.0 / cpu_tiles;
            double fpga_us_per_tile = fpga_sleep_ms * 1000.0 / fpga_tiles;
            printf("  CPU/tile:  %.3f us  FPGA/tile: %.3f us\n", cpu_us_per_tile, fpga_us_per_tile);
            printf("  Pipeline bound: %s\n",
                   cpu_us_per_tile > fpga_us_per_tile ? "CPU" : "FPGA");
        }
    }
};

// ===========================================================================
// AXI Transfer Timing Model
//
// Models AXI-Lite GP0 on Zynq 7010 at 150 MHz, 32-bit bus.
// Each register write = 5 cycles (AW+W+B handshake).
// Tile data transfer bandwidth depends on tile size and bus width.
// ===========================================================================

struct AxiTiming {
    static constexpr double BUS_MHZ = 150.0;
    static constexpr double BUS_MBPS = BUS_MHZ * 4.0 * 0.8;  // 32-bit @ 80% efficiency ≈ 480 MB/s

    static constexpr uint32_t REG_WRITE_CYCLES = 5;
    static constexpr uint32_t REG_WRITE_US      = 5 * 1000 / 150;  // ~33 ns = 0.033 µs

    // Per-tile AXI transfer time based on data size
    static double tile_transfer_us(size_t bytes) {
        return bytes / BUS_MBPS * 1000.0;  // ms→µs conversion
    }

    // Tile sizes for different paths
    static constexpr size_t INT8_TILE_BYTES   = 64 * 64 * 1;   // 4,096 B
    static constexpr size_t INT16_TILE_BYTES  = 64 * 64 * 2;   // 8,192 B
    static constexpr size_t Q8_0_TILE_BYTES  = 64 * 64 * 1;   // 4,096 B (Q8_0 raw bytes)
    static constexpr size_t Q8_SCALE_BYTES   = 128 * 2;         // 128 combined UQ8.8 scales (256 B)

    // Precomputed transfer times (inlined since all values are constexpr)
    static constexpr double INT8_TILE_US   = (double)(INT8_TILE_BYTES) / BUS_MBPS * 1000.0;
    static constexpr double INT16_TILE_US  = (double)(INT16_TILE_BYTES) / BUS_MBPS * 1000.0;
    static constexpr double Q8_TILE_US     = (double)(Q8_0_TILE_BYTES + Q8_SCALE_BYTES) / BUS_MBPS * 1000.0;
};

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

// Forward declare tracer callback type (defined externally by axitrace.hpp)
using AxiTraceFn = void (*)(void* ctx, uint32_t cycle, int dir,
                            uint32_t addr, uint32_t data);

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
//
// Optional tracing: attach a callback to observe every register access.
//   accel.set_tracer(my_fn, my_ctx);
//   // my_fn(ctx, cycle, dir, addr, data) is called on every write/read
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
        trace_cycle_++;
        if (trace_fn_) trace_fn_(trace_ctx_, trace_cycle_, 0, byte_off, val);
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
        trace_cycle_++;
        if (trace_fn_) trace_fn_(trace_ctx_, trace_cycle_, 1, byte_off, regs_[byte_off / 4]);
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

    // Attach an AXI transaction tracer callback
    void set_tracer(AxiTraceFn fn, void* ctx = nullptr) {
        trace_fn_ = fn; trace_ctx_ = ctx;
    }

    uint64_t trace_cycle() const { return trace_cycle_; }

    // Total FPGA cycles consumed (for performance accounting)
    uint64_t total_cycles() const { return total_cycles_; }
    void add_cycles(uint64_t n) { total_cycles_ += n; }

    uint8_t* ddr() { return ddr_; }
    const uint8_t* ddr() const { return ddr_; }

    bool interrupt() const { return intr_flag_.load(); }
    void set_latency_ms(uint32_t ms) { latency_ms_ = ms; }

    // Q8_0 direct path support
    void set_q8_path(bool enable) { q8_path_ = enable; }
    bool q8_path() const { return q8_path_; }

private:
    mutable uint32_t regs_[64];
    uint8_t ddr_[DDR_BYTES];
    std::atomic<bool> intr_flag_{false};
    std::atomic<bool> start_flag_{false};
    std::atomic<bool> done_flag_{false};
    std::atomic<bool> stop_{false};
    std::thread fsm_thread_;
    uint32_t latency_ms_ = 0;  // default: no artificial latency
    AxiTraceFn trace_fn_ = nullptr;
    void* trace_ctx_ = nullptr;
    mutable uint64_t trace_cycle_ = 0;
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

        if (latency_ms_ > 0)
            std::this_thread::sleep_for(std::chrono::milliseconds(latency_ms_));

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

// INT8 vecmul tile: vec[64] × W[64][64] → result[64] through AXI interface
inline void axi_vecmul_tile_int8(const in_t vec[N], const in_t W[N][N], acc_t result[N]) {
    auto& a = accel();
    memcpy(a.ddr(), vec, N * sizeof(in_t));
    memcpy(a.ddr() + 4096, W, N * N * sizeof(in_t));
    a.write_reg(REG_CTRL_USER, CTRL_OP_VECMUL);
    a.write_reg(REG_SIZE, (1 << 16) | (N << 8) | N);
    a.write_reg(REG_AP_CTRL, AP_START);
    a.wait_done();
    memcpy(result, a.ddr() + 8192, N * sizeof(acc_t));
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

// Q8_0 block: 32 elements, each = val * scale, packed as [scale:FP16][val:INT8×32]
inline uint16_t read_q8_scale(const uint8_t* data, uint64_t block_idx) {
    uint64_t off = block_idx * 34;
    return (uint16_t)data[off] | ((uint16_t)data[off + 1] << 8);
}

// Compute per-row scales from Q8_0 data (same as ARM-side precompute)
inline void compute_q8_row_scales(const uint8_t* q8_data, int row, int cols,
                                   float* scales_out, int max_rows) {
    for (int r = 0; r < max_rows; r++) {
        float max_abs = 0.0f;
        for (int c = 0; c < cols; c++) {
            uint64_t idx = (uint64_t)r * cols + c;
            uint64_t block = idx / 32;
            uint64_t off_in_block = idx % 32;
            uint64_t block_off = block * 34 + 2 + off_in_block;
            int8_t val = (int8_t)q8_data[block_off];
            float a = fabsf((float)val);
            if (a > max_abs) max_abs = a;
        }
        scales_out[r] = (max_abs < 1e-10f) ? 1.0f : max_abs / 32767.0f;
    }
    (void)row; (void)cols;
}

// Precompute combined UQ8.8 fixed-point scales for a 64×64 tile's rows.
// Each row needs block_scale[row][blk] (from the Q8_0 tensor) and row_scale[row].
// Combined = block_scale / row_scale → UQ8.8 fixed-point.
// FPGA receives these and does LUT-based integer multiply: q8_val * combined >> 8.
inline void compute_combined_scales(const float block_scales[64][2],
                                     const float row_scales[64],
                                     uint16_t combined_scales[128]) {
    for (int r = 0; r < 64; r++) {
        float row_inv = (row_scales[r] < 1e-10f) ? 1.0f : (1.0f / row_scales[r]);
        for (int blk = 0; blk < 2; blk++) {
            float combined = block_scales[r][blk] * row_inv;
            if (combined >= 256.0f) combined = 255.996f;
            if (combined < 0.0f) combined = 0.0f;
            combined_scales[r * 2 + blk] = (uint16_t)(combined * 256.0f + 0.5f);
        }
    }
}

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
// Q8_0 Logits Path — FPGA handles full (1×896) @ (896×151936) matmul
//
// Flow:
//   1. CPU sends hidden vector (1×896 FP32) ONCE to FPGA BRAM
//   2. FPGA streams Q8_0 token_embd.weight blocks from DDR
//   3. FPGA does Q8→INT16 dequantization internally
//   4. FPGA computes hidden × Q8_embeddings
//   5. CPU receives raw_logits (151936 floats) for top-k comparison
//
// DDR layout for logits Q8 path (fits within 512 MB):
//   Hidden vector (1×896 FP32):        offset 0,       ~3.5 KB
//   Q8_0 embeddings (896×151936):      offset 65536,   ~139 MB
//   Raw logits output (151936 FP32):   offset 150000000, ~607 KB
// ===========================================================================

inline void fpga_logits_q8(const Tensor* emb_t,
                           const float* hidden,
                           float* raw_logits,
                           int hidden_dim,
                           int vocab_size) {
    auto t0 = std::chrono::high_resolution_clock::now();

    memset(raw_logits, 0, vocab_size * sizeof(float));
    extern void q8_logits_matmul_with_tensor(const Tensor* t, const float* x, float* y, int rows, int cols);
    q8_logits_matmul_with_tensor(emb_t, hidden, raw_logits, vocab_size, hidden_dim);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    g_timing.cpu_ms += ms;
}

} // namespace fpga_sim
