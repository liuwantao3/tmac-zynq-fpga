// AXI-Lite Transaction Tracer + Pipeline Profiler Demo
//
// Shows how to use AxiTrace and Profiler alongside MatmulAccel to:
//   1. Trace every AXI-Lite register transaction with cycle timing
//   2. Generate VCD waveform viewable in GTKWave
//   3. Identify the bottleneck stage in the inference pipeline
//
// Build: g++ -std=c++17 -pthread -O2 -o demo demo.cpp
// Run:   ./demo

#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <chrono>
#include <cmath>

// Adjust path for header locations
#include "../fpga_sim.hpp"
#include "axitrace.hpp"
#include "fpga_profiler.hpp"

using namespace fpga_sim;

static Profiler g_prof;  // profile at global scope

// ===========================================================================
// Instrumented tile wrappers
// ===========================================================================
static void traced_vecmul_int8(const in_t vec[N], const in_t W[N][N],
                                acc_t result[N], AxiTrace*) {
    auto& a = accel();
    g_prof.begin(Profiler::STAGE_DDR_COPY);
    memcpy(a.ddr(), vec, N * sizeof(in_t));
    memcpy(a.ddr() + 4096, W, N * N * sizeof(in_t));
    g_prof.end();

    g_prof.begin(Profiler::STAGE_AXI_SETUP);
    a.write_reg(REG_CTRL_USER, CTRL_OP_VECMUL);
    a.write_reg(REG_SIZE, (1 << 16) | (N << 8) | N);
    a.write_reg(REG_AP_CTRL, AP_START);
    g_prof.end();

    g_prof.begin(Profiler::STAGE_COMPUTE);
    a.wait_done();
    g_prof.end();

    g_prof.begin(Profiler::STAGE_DDR_COPY);
    memcpy(result, a.ddr() + 8192, N * sizeof(acc_t));
    g_prof.end();
}

static void traced_vecmul_int16(const in16_t vec[N], const in16_t W[N][N],
                                 acc16_t result[N], AxiTrace* trace) {
    auto& a = accel();

    if (trace) {
        // Trace logs the AXI protocol; accel.write_reg triggers the FSM
        trace->log_write(REG_CTRL_USER, CTRL_OP_VECMUL | CTRL_MODE_INT16);
        trace->log_write(REG_SIZE, (1 << 16) | (N << 8) | N);
        trace->log_state(1);
        trace->log_compute_start(1, N, N, true);
    }

    g_prof.begin(Profiler::STAGE_DDR_COPY);
    memcpy(a.ddr(), vec, N * sizeof(in16_t));
    memcpy(a.ddr() + 8192, W, N * N * sizeof(in16_t));
    g_prof.end();

    g_prof.begin(Profiler::STAGE_AXI_SETUP);
    a.write_reg(REG_CTRL_USER, CTRL_OP_VECMUL | CTRL_MODE_INT16);
    a.write_reg(REG_SIZE, (1 << 16) | (N << 8) | N);

    if (trace) trace->log_write(REG_AP_CTRL, AP_START);
    a.write_reg(REG_AP_CTRL, AP_START);
    g_prof.end();

    g_prof.begin(Profiler::STAGE_COMPUTE);
    a.wait_done();
    if (trace) {
        trace->log_compute_end();
        trace->log_state(2);
        trace->log_interrupt(true);
    }
    g_prof.end();

    g_prof.begin(Profiler::STAGE_DDR_COPY);
    memcpy(result, a.ddr() + 16384, N * sizeof(acc16_t));
    g_prof.end();

    if (trace) trace->log_interrupt(false);
}

// ===========================================================================
// Simulated inference with profiling
// ===========================================================================
static float run_traced_inference(AxiTrace* trace, bool use_int16) {
    auto t0 = std::chrono::high_resolution_clock::now();

    g_prof.begin_layer(0);

    // Simulate a single matmul: rows=128, cols=896 (like Q projection)
    int rows = 128, cols = 896;

    // Input vector
    std::vector<float> x(cols);
    for (int j = 0; j < cols; j++) x[j] = (float)(j % 100) / 100.0f;

    // Output
    std::vector<float> y(rows, 0.0f);

    // Quantize input
    g_prof.begin(Profiler::STAGE_QUANTIZE);
    float x_scale = 0;
    for (int j = 0; j < cols; j++) x_scale = fmaxf(x_scale, fabsf(x[j]));
    x_scale = (x_scale < 1e-10f) ? 1.0f
              : (use_int16 ? x_scale / 32767.0f : x_scale / 127.0f);
    g_prof.end();

    for (int r = 0; r < rows; r += N) {
        int r_size = std::min(N, rows - r);
        g_prof.begin_tile(r, 0, r_size, cols);

        // Generate fake weights
        if (use_int16) {
            in16_t W_q[N][N] = {{0}};
            in16_t vec[N] = {0};
            for (int k = 0; k < cols && k < N; k++)
                vec[k] = (in16_t)roundf(x[k] / x_scale);
            for (int i = 0; i < r_size; i++)
                for (int k = 0; k < N && k < cols; k++)
                    W_q[k][i] = (in16_t)((i * cols + k) % 100);

            acc16_t result[N] = {0};
            traced_vecmul_int16(vec, W_q, result, trace);

            g_prof.begin(Profiler::STAGE_DEQUANTIZE);
            for (int i = 0; i < r_size; i++)
                y[r + i] += (double)result[i] * x_scale * x_scale;
            g_prof.end();
        } else {
            in_t W_q[N][N] = {{0}};
            in_t vec[N] = {0};
            for (int k = 0; k < cols && k < N; k++)
                vec[k] = (in_t)roundf(x[k] / x_scale);
            for (int i = 0; i < r_size; i++)
                for (int k = 0; k < N && k < cols; k++)
                    W_q[k][i] = (in_t)((i * cols + k) % 127);

            acc_t result[N] = {0};
            traced_vecmul_int8(vec, W_q, result, trace);

            g_prof.begin(Profiler::STAGE_DEQUANTIZE);
            for (int i = 0; i < r_size; i++)
                y[r + i] += (float)result[i] * x_scale * x_scale;
            g_prof.end();
        }
        g_prof.end_tile();
    }

    g_prof.end_layer();

    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<float, std::milli>(t1 - t0).count();
}

// ===========================================================================
// Main
// ===========================================================================
int main() {
    printf("AXI-Lite Transaction Tracer + Pipeline Profiler Demo\n");
    printf("====================================================\n\n");

    // ---- Part 1: AXI Transaction Trace ------------------------------------
    printf("[Part 1] AXI-Lite Transaction Trace (single INT16 VecMul tile)\n");
    {
        AxiTrace trace;
        trace.set_compute_cycles(128);  // slow compute for visible VCD

        in16_t vec[N] = {0};
        in16_t W[N][N] = {{0}};
        acc16_t result[N] = {0};
        for (int i = 0; i < N; i++) vec[i] = (in16_t)i;
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                W[j][i] = (in16_t)(i * N + j);

        traced_vecmul_int16(vec, W, result, &trace);

        trace.print_summary();
        trace.dump_vcd("/tmp/axi_trace.vcd");
        trace.dump_csv("/tmp/axi_trace.csv");
    }

    // ---- Part 2: Pipeline Profiler ----------------------------------------
    printf("\n[Part 2] Pipeline Profiler — 128×896 matmul INT16\n");
    {
        g_prof.begin(Profiler::STAGE_OVERHEAD, "full inference");
        float ms = run_traced_inference(nullptr, true);
        g_prof.end();
        printf("  Wall time: %.2f ms\n", ms);
        g_prof.report();
        g_prof.dump_csv("/tmp/pipeline_profile.csv");
    }

    // ---- Part 3: Compare INT8 vs INT16 ------------------------------------
    printf("\n[Part 3] INT8 vs INT16 Profiling Comparison\n");
    {
        Profiler p8, p16;

        // INT8 run
        {
            using namespace fpga_sim;
            Profiler::Scope s8(p8, Profiler::STAGE_OVERHEAD, "int8 matmul");
            p8.begin_layer(0);
            int rows = 64, cols = 896;
            std::vector<float> x(cols);
            for (int j = 0; j < cols; j++) x[j] = 1.0f;

            p8.begin(Profiler::STAGE_QUANTIZE);
            p8.end();

            for (int r = 0; r < rows; r += N) {
                p8.begin_tile(r, 0, std::min(N, rows - r), cols);
                in_t vec[N] = {0};
                in_t W[N][N] = {{0}};
                for (int k = 0; k < N; k++) vec[k] = (in_t)(k % 127);
                for (int i = 0; i < N; i++)
                    for (int k = 0; k < N; k++)
                        W[k][i] = (in_t)((i + k) % 127);
                acc_t result[N] = {0};
                traced_vecmul_int8(vec, W, result, nullptr);
                p8.begin(Profiler::STAGE_DEQUANTIZE);
                p8.end();
                p8.end_tile();
            }
            p8.end_layer();
        }

        // INT16 run
        {
            Profiler::Scope s16(p16, Profiler::STAGE_OVERHEAD, "int16 matmul");
            p16.begin_layer(0);
            int rows = 64, cols = 896;
            std::vector<float> x(cols);
            for (int j = 0; j < cols; j++) x[j] = 1.0f;

            p16.begin(Profiler::STAGE_QUANTIZE);
            p16.end();

            for (int r = 0; r < rows; r += N) {
                p16.begin_tile(r, 0, std::min(N, rows - r), cols);
                in16_t vec[N] = {0};
                in16_t W[N][N] = {{0}};
                for (int k = 0; k < N; k++) vec[k] = (in16_t)(k % 1000);
                for (int i = 0; i < N; i++)
                    for (int k = 0; k < N; k++)
                        W[k][i] = (in16_t)((i + k) % 1000);
                acc16_t result[N] = {0};
                traced_vecmul_int16(vec, W, result, nullptr);
                p16.begin(Profiler::STAGE_DEQUANTIZE);
                p16.end();
                p16.end_tile();
            }
            p16.end_layer();
        }

        printf("  INT8  profile:\n");  p8.report();
        printf("  INT16 profile:\n"); p16.report();
    }

    printf("\nDone. View VCD with: gtkwave /tmp/axi_trace.vcd\n");
    printf("      CSV analysis:  cat /tmp/axi_trace.csv\n");
    printf("      Pipeline CSV:  cat /tmp/pipeline_profile.csv\n");
    return 0;
}
