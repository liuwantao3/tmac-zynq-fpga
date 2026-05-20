// System integration test — exercises INT16 and Q4_K FPGA simulation paths
// 
// Architecture:
//   - INT16: general matmul (attention QKV, etc.) — CPU dequants weights to INT16
//   - Q4_K:  FFN layers (gate/up/down) — pre-dequantized INT16 through AXI-Lite buffers
//   - Q8_0:  logits only (token embeddings) — tested via matmul_q8.cpp + tmac_gguf --fpga-q8
//
// Compile: g++ -std=c++14 -O2 -I. -o test_integration test_integration.cpp -lpthread
// Run:     ./test_integration

#include "fpga_sim.hpp"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <cassert>

using namespace fpga_sim;

// Test config (N = fpga_sim::N = 64)
constexpr int REPEAT_TILES = 5;

// ===========================================================================
// Helper: generate random INT16 tile
// ===========================================================================
void gen_rand_tile(int16_t tile[N][N], int seed) {
    srand(seed);
    for (int c = 0; c < N; c++)
        for (int r = 0; r < N; r++)
            tile[c][r] = (int16_t)(rand() % 200 - 100);  // [-100, 100)
}

// Generate activations
void gen_rand_vec(int16_t vec[N], int seed) {
    srand(seed + 1000);
    for (int i = 0; i < N; i++)
        vec[i] = (int16_t)(rand() % 200 - 100);
}

// ===========================================================================
// Test: compare INT16 and Q4K-AXILITE paths against CPU reference
// ===========================================================================
int test_roundtrip() {
    int errors = 0;
    int total = 0;

    for (int seed = 0; seed < REPEAT_TILES; seed++) {
        int16_t tile[N][N];
        int16_t vec[N];
        gen_rand_tile(tile, seed);
        gen_rand_vec(vec, seed + 2000);

        // ---- CPU reference ----
        acc16_t ref[N];
        vecmul_1x64_int16(vec, tile, ref);

        // ---- INT16 FPGA path (DDR-based, general matmul) ----
        acc16_t fpga_int16[N];
        axi_vecmul_tile_int16(vec, tile, fpga_int16);

        // ---- Q4_K AXI-Lite buffer path (FFN layers, pre-dequant INT16) ----
        acc16_t fpga_q4k_axilite[N];
        axi_vecmul_tile_q4k_axilite(tile, vec, fpga_q4k_axilite);

        // ---- Compare ----
        for (int i = 0; i < N; i++) {
            total++;
            if (fpga_int16[i] != ref[i]) {
                printf("  FAIL: seed=%d row=%d INT16: got %ld, expected %ld\n",
                       seed, i, (long)fpga_int16[i], (long)ref[i]);
                errors++;
                if (errors > 10) break;
            }
            if (fpga_q4k_axilite[i] != ref[i]) {
                printf("  FAIL: seed=%d row=%d Q4K-AXILITE: got %ld, expected %ld\n",
                       seed, i, (long)fpga_q4k_axilite[i], (long)ref[i]);
                errors++;
                if (errors > 10) break;
            }
        }
        if (errors > 10) break;
    }
    return errors;
}

// ===========================================================================
// Test: AXI-Lite buffer write address mapping
// ===========================================================================
int test_axilite_addr_map() {
    int errors = 0;

    AxiliteAccelState s;

    // Write fixed pattern through AXI-Lite buffer interface
    for (int off = 0; off < 8192; off += 4) {
        uint32_t data = 0x03020100u;  // byte3=3, byte2=2, byte1=1, byte0=0
        axilite_write_buf(s, AXI_WEIGHT_BASE + off, data, 0xF, true);
    }

    // Verify: weight_buf[i] = i % 4 (byte lane within word)
    for (int i = 0; i < 8192; i++) {
        uint8_t expected = (i % 4);
        if (s.weight_buf[i] != expected) {
            printf("  FAIL: weight_buf[%d] = %d, expected %d\n",
                   i, s.weight_buf[i], (uint8_t)(i & 0xFF));
            errors++;
            if (errors > 5) break;
        }
    }
    if (errors == 0)
        printf("  PASS: weight_buf 8192 bytes verified\n");

    return errors;
}

// ===========================================================================
// Test: AXI-Lite Q4K compute (pre-dequant INT16 through buffer)
// ===========================================================================
int test_axilite_compute() {
    int errors = 0;

    int16_t tile[N][N];
    int16_t vec[N];
    gen_rand_tile(tile, 42);
    gen_rand_vec(vec, 42);

    // Reference
    acc16_t ref[N];
    vecmul_1x64_int16(vec, tile, ref);

    // AXI-Lite buffer path
    acc16_t result[N];
    axi_vecmul_tile_q4k_axilite(tile, vec, result);

    for (int i = 0; i < N; i++) {
        if (result[i] != ref[i]) {
            printf("  FAIL: row %d: got %ld, expected %ld\n",
                   i, (long)result[i], (long)ref[i]);
            errors++;
            if (errors > 5) break;
        }
    }
    if (errors == 0)
        printf("  PASS: all 64 rows match reference\n");
    return errors;
}

int main() {
    int total_errors = 0;
    int total_tests = 0;

    printf("========================================\n");
    printf("FPGA Integration Test Suite\n");
    printf("========================================\n\n");

    // Test 1: AXI-Lite address mapping
    printf("Test 1: AXI-Lite buffer write address map\n");
    total_errors += test_axilite_addr_map();
    total_tests++;
    printf("\n");

    // Test 2: AXI-Lite Q4K compute vs reference
    printf("Test 2: AXI-Lite Q4K compute vs vecmul_1x64_int16\n");
    total_errors += test_axilite_compute();
    total_tests++;
    printf("\n");

    // Test 3: Roundtrip — INT16 + Q4K-AXILITE vs CPU reference
    printf("Test 3: Roundtrip (INT16+Q4K-AXILITE) vs CPU reference (%d random tiles)\n", REPEAT_TILES);
    total_errors += test_roundtrip();
    total_tests++;
    printf("\n");

    // Summary
    printf("========================================\n");
    if (total_errors == 0)
        printf("ALL %d TESTS PASSED\n", total_tests);
    else
        printf("%d / %d TESTS FAILED (%d errors)\n",
               total_errors, total_tests, total_errors);

    // Cleanup
    auto& a = accel();
    (void)a;

    return total_errors > 0 ? 1 : 0;
}
