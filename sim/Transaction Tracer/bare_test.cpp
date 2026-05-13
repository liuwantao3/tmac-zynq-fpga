#include <cstdio>
#include <cstring>
#include <cstdint>
#include "../fpga_sim.hpp"
using namespace fpga_sim;

int main() {
    printf("Starting...\n"); fflush(stdout);
    auto& a = accel();

    int16_t vec[64] = {0};
    int16_t W[64][64] = {{0}};
    int64_t result[64] = {0};
    for (int i = 0; i < 64; i++) vec[i] = (int16_t)i;
    for (int i = 0; i < 64; i++)
        for (int j = 0; j < 64; j++)
            W[j][i] = (int16_t)(i * 64 + j);

    printf("DDR...\n"); fflush(stdout);
    memcpy(a.ddr(), vec, 64 * sizeof(int16_t));
    memcpy(a.ddr() + 8192, W, 64 * 64 * sizeof(int16_t));

    printf("Regs...\n"); fflush(stdout);
    a.write_reg(REG_CTRL_USER, CTRL_OP_VECMUL | CTRL_MODE_INT16);
    a.write_reg(REG_SIZE, (1 << 16) | (64 << 8) | 64);
    printf("Start...\n"); fflush(stdout);
    a.write_reg(REG_AP_CTRL, AP_START);
    printf("Wait...\n"); fflush(stdout);
    bool ok = a.wait_done(5000);
    printf("Done: %d\n", ok ? 1 : 0);
    memcpy(result, a.ddr() + 16384, 64 * sizeof(int64_t));
    printf("result[0] = %lld\n", (long long)result[0]);
    return 0;
}
