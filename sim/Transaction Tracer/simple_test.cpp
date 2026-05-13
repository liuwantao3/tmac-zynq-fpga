#include <cstdio>
#include <cstring>
#include "../fpga_sim.hpp"
#include "axitrace.hpp"
using namespace fpga_sim;

int main() {
    printf("Creating trace + accel...\n"); fflush(stdout);
    AxiTrace trace;
    trace.set_compute_cycles(64);
    auto& a = accel();

    // Regular compute without trace (just to verify accel works)
    in16_t vec[64] = {0};
    in16_t W[64][64] = {{0}};
    acc16_t result[64] = {0};
    for (int i = 0; i < 64; i++) vec[i] = (in16_t)i;
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++)
            W[j][i] = (in16_t)(i * 64 + j);
    memcpy(a.ddr(), vec, 64 * sizeof(in16_t));
    memcpy(a.ddr() + 8192, W, 64 * 64 * sizeof(in16_t));
    a.write_reg(REG_CTRL_USER, CTRL_OP_VECMUL | CTRL_MODE_INT16);
    a.write_reg(REG_SIZE, (1 << 16) | (64 << 8) | 64);
    printf("wait_done...\n"); fflush(stdout);
    a.write_reg(REG_AP_CTRL, AP_START);
    a.wait_done();
    memcpy(result, a.ddr() + 16384, 64 * sizeof(acc16_t));
    printf("result[0] = %lld\n", (long long)result[0]);

    // Now test trace (standalone, no accel interaction)
    printf("\n-- Trace logging --\n");
    trace.log_write(REG_CTRL_USER, CTRL_OP_VECMUL | CTRL_MODE_INT16);
    trace.log_write(REG_SIZE, (1 << 16) | (64 << 8) | 64);
    trace.log_state(1);
    trace.log_compute_start(1, 64, 64, true);
    trace.log_write(REG_AP_CTRL, AP_START);
    trace.log_compute_end();
    trace.log_state(2);
    trace.log_interrupt(true);
    trace.log_interrupt(false);

    trace.print_summary();
    trace.dump_vcd("/tmp/test_trace.vcd");
    trace.dump_csv("/tmp/test_trace.csv");
    printf("All done!\n");
    return 0;
}
