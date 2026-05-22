// System integration test — exercises INT16 and Q4_K FPGA simulation paths
// 
// Architecture:
//   - INT16: general matmul (attention QKV, etc.) — CPU dequants weights to INT16
//   - Q4_K:  FFN layers (gate/up/down) — raw Q4_K blocks through AXI-Lite buffers
//   - Q8_0:  logits only (token embeddings) — tested via matmul_q8.cpp + tmac_gguf --fpga-q8
//
// Phase 2: Q4_K block decode is on FPGA. Tests verify raw block path matches CPU.
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

constexpr int REPEAT_TILES = 5;

void gen_rand_tile(int16_t tile[N][N], int seed) {
    srand(seed);
    for (int c = 0; c < N; c++)
        for (int r = 0; r < N; r++)
            tile[c][r] = (int16_t)(rand() % 200 - 100);
}

void gen_rand_vec(int16_t vec[N], int seed) {
    srand(seed + 1000);
    for (int i = 0; i < N; i++)
        vec[i] = (int16_t)(rand() % 200 - 100);
}

// ===========================================================================
// Helper: encode f16 into two bytes (little-endian)
// ===========================================================================
void f16_encode(float val, uint8_t* out) {
    uint32_t sign = (val < 0) ? 1 : 0; if (val < 0) val = -val;
    uint32_t exp = 0;
    uint32_t mant = 0;
    if (val == 0.0f) { exp = 0; mant = 0; }
    else {
        int e;
        float frac = frexpf(val, &e);
        exp = e + 14;
        mant = (uint32_t)((frac - 0.5f) * 2048.0f * 2.0f);
        if (exp >= 31) { exp = 31; mant = 0; }
        else if (exp == 0) exp = 1;
    }
    uint16_t raw = (sign << 15) | (exp << 10) | (mant & 0x3FF);
    out[0] = raw & 0xFF;
    out[1] = (raw >> 8) & 0xFF;
}

// ===========================================================================
// Generate a Q4_K block with constant weight value
//   d = 1.0, dmin = 0.0, sc = 1, m = 0, all quants = q4_const
//   Result: each weight = d * sc * q4 = 1.0 * 1 * q4_const = q4_const
// ===========================================================================
void gen_q4k_block_constant(uint8_t block[Q4K_BLOCK_BYTES], uint8_t q4_const) {
    memset(block, 0, Q4K_BLOCK_BYTES);

    // d = 1.0, dmin = 0.0
    f16_encode(1.0f, block);
    f16_encode(0.0f, block + 2);

    // Scales: sc[0..7] = 1, m[0..7] = 0
    // bytes 4-7: sc[0..3] (low 6 bits = 1)
    block[4] = 0x01; block[5] = 0x01; block[6] = 0x01; block[7] = 0x01;
    // bytes 8-11: m[0..3] (low 6 bits = 0)
    // bytes 12-15: sc[4..7] low nibble = 1, m[4..7] high nibble = 0
    block[8]  = 0x01; block[9]  = 0x01; block[10] = 0x01; block[11] = 0x01;

    // Quants: all = q4_const (nibble), each byte = q4_const | (q4_const << 4)
    uint8_t qs_byte = q4_const | (q4_const << 4);
    for (int i = 16; i < Q4K_BLOCK_BYTES; i++)
        block[i] = qs_byte;
}

// ===========================================================================
// Fill a tile buffer with Q4_K blocks.
void fill_tile_blocks(uint8_t* tile_blocks, int nblocks,
                      const uint8_t block[Q4K_BLOCK_BYTES]) {
    for (int bi = 0; bi < nblocks; bi++)
        memcpy(tile_blocks + bi * Q4K_BLOCK_BYTES, block, Q4K_BLOCK_BYTES);
}

// ===========================================================================
// Test: AXI-Lite Q4K raw block path vs CPU reference
// Uses 16 blocks (2304 bytes) = 16 rows × 256 cols.
// For a 64-col tile: each block contributes wi=0..63 to its row.
// With constant data and identity row_scale, all rows = same value.
// ===========================================================================
int test_q4k_blocks() {
    int errors = 0;
    const int NBLOCKS = 16;

    // Generate constant Q4_K blocks: all weights = 8
    uint8_t block[Q4K_BLOCK_BYTES];
    gen_q4k_block_constant(block, 8);
    uint8_t tile_blocks[NBLOCKS * Q4K_BLOCK_BYTES];
    fill_tile_blocks(tile_blocks, NBLOCKS, block);

    // Decode one block for reference
    int16_t block_decoded[Q4K_BLOCK_SIZE];
    dequant_q4k_block_to_int16(block, block_decoded);

    // Build reference tile [64][NBLOCKS] (col-major)
    // Each block covers 1 row × 256 cols. For a 64-col tile (wi=0..63):
    // row b = block b's values at wi=0..63
    int16_t ref_tile[64][NBLOCKS] = {{0}};
    for (int bi = 0; bi < NBLOCKS; bi++) {
        for (int wio = 0; wio < 64; wio++) {
            ref_tile[wio][bi] = block_decoded[wio];
        }
    }

    // Activations: all ones
    int16_t vec[64];
    for (int i = 0; i < 64; i++) vec[i] = 1;

    // CPU reference: vec[64] × ref_tile[64][NBLOCKS] → result[NBLOCKS]
    int16_t ref_result[NBLOCKS] = {0};
    for (int row = 0; row < NBLOCKS; row++) {
        int64_t acc = 0;
        for (int k = 0; k < 64; k++)
            acc += (int64_t)vec[k] * (int64_t)ref_tile[k][row];
        ref_result[row] = (int16_t)acc;
    }

    // Identity row_scale: max_abs = 32767, row_inv = 32767/32767 = 1.0
    float row_inv[NBLOCKS];
    for (int i = 0; i < NBLOCKS; i++) row_inv[i] = 1.0f;

    // Raw block AXI-Lite path (64-col tile)
    int64_t fpga_result[896] = {0};
    axi_vecmul_tile_q4k_axilite(tile_blocks, NBLOCKS, vec, 64, row_inv, fpga_result);

    // Compare all rows
    for (int i = 0; i < NBLOCKS; i++) {
        if ((int16_t)fpga_result[i] != ref_result[i]) {
            printf("  FAIL: row %d: got %d, expected %d\n",
                   i, fpga_result[i], ref_result[i]);
            errors++;
            if (errors > 5) break;
        }
    }
    if (errors == 0)
        printf("  PASS: all %d rows match reference (val=%d)\n", NBLOCKS, ref_result[0]);
    return errors;
}

// ===========================================================================
// Test: AXI-Lite buffer write address mapping
// Verifies first 8192 bytes (Verilog weight_buf size) with Q4K_TILE_BYTES writes.
// ===========================================================================
int test_axilite_addr_map() {
    int errors = 0;
    AxiliteAccelState s;

    for (int off = 0; off < Q4K_TILE_BYTES; off += 4) {
        uint32_t data = 0x03020100u;
        axilite_write_buf(s, AXI_WEIGHT_BASE + off, data, 0xF, true);
    }

    for (int i = 0; i < 8192; i++) {  // Only verify Verilog buffer size
        uint8_t expected = (i % 4);
        if (s.weight_buf[i] != expected) {
            printf("  FAIL: weight_buf[%d] = %d, expected %d\n",
                   i, s.weight_buf[i], (uint8_t)(i & 0xFF));
            errors++;
            if (errors > 5) break;
        }
    }
    if (errors == 0)
        printf("  PASS: weight_buf %d bytes verified\n", 8192);
    return errors;
}

// ===========================================================================
// Test: INT16 AXI-Lite buffer path (Phase 1, backward compat)
// ===========================================================================
int test_int16_axilite() {
    int errors = 0;

    int16_t tile[N][N];
    int16_t vec[N];
    gen_rand_tile(tile, 42);
    gen_rand_vec(vec, 42);

    acc16_t ref[N];
    vecmul_1x64_int16(vec, tile, ref);

    acc16_t result[N];
    axi_vecmul_tile_int16_axilite(tile, vec, result);

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
    printf("FPGA Integration Test Suite (Phase 2)\n");
    printf("========================================\n\n");

    // Test 1: AXI-Lite address mapping
    printf("Test 1: AXI-Lite buffer write address map\n");
    total_errors += test_axilite_addr_map();
    total_tests++;
    printf("\n");

    // Test 2: INT16 AXI-Lite buffer path (backward compat)
    printf("Test 2: INT16 AXI-Lite buffer path vs vecmul_1x64_int16\n");
    total_errors += test_int16_axilite();
    total_tests++;
    printf("\n");

    // Test 3: Q4_K raw block path (Phase 2)
    printf("Test 3: Q4_K raw block AXI-Lite path vs reference\n");
    total_errors += test_q4k_blocks();
    total_tests++;
    printf("\n");

    // Summary
    printf("========================================\n");
    if (total_errors == 0)
        printf("ALL %d TESTS PASSED\n", total_tests);
    else
        printf("%d / %d TESTS FAILED (%d errors)\n",
               total_errors, total_tests, total_errors);

    auto& a = accel();
    (void)a;

    return total_errors > 0 ? 1 : 0;
}
