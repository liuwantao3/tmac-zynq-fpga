// Self-test for the bit-exact golden models (sim/golden_model.hpp).
// Uses the worked examples from docs/maths.md.
#include "golden_model.hpp"
#include <cstdio>

using namespace golden;

static int g_fail = 0;
#define CHECK(cond) do { if (!(cond)) { \
    printf("FAIL: %s (line %d)\n", #cond, __LINE__); g_fail++; } \
} while (0)

int main() {
    // ---- f16_decode_s2416: 1.0 -> 65536, 0.5 -> 32768, 0.25 -> 16384 ----
    CHECK(f16_decode_s2416(0x3C00) == 65536);  // 1.0
    CHECK(f16_decode_s2416(0x3800) == 32768);  // 0.5
    CHECK(f16_decode_s2416(0x3400) == 16384);  // 0.25
    CHECK(f16_decode_s2416(0x0000) == 0);      // subnormal -> 0
    CHECK(f16_decode_s2416(0x7C00) == 0);      // inf -> 0

    // ---- Q8 dequant worked example: q8=50, sc=512 (2.0) -> 100 ----
    CHECK(q8_dequant(50, 512) == 100);
    CHECK(q8_dequant(1, 256) == 1);          // 1 x 1.0 = 1
    CHECK(q8_dequant(-1, 256) == -1);        // -1 x 1.0 = -1
    CHECK(q8_dequant(2, 512) == 4);          // 2 x 2.0 = 4

    // ---- Q8 tile: all weights=1, sc=1.0, act=1 -> each row = 64 ----
    {
        int8_t W[64*64]; uint16_t sc[64*2]; int16_t act[64]; int64_t out[64];
        for (int i = 0; i < 64*64; i++) W[i] = 1;
        for (int i = 0; i < 64*2; i++) sc[i] = 256;
        for (int i = 0; i < 64; i++) act[i] = 1;
        q8_tile_golden(W, sc, act, out);
        for (int r = 0; r < 64; r++) CHECK(out[r] == 64);
    }

    // ---- Q8 tile: col-pattern W[r][c]=c+1, sc=1.0, act=1 -> 2080 ----
    {
        int8_t W[64*64]; uint16_t sc[64*2]; int16_t act[64]; int64_t out[64];
        for (int r = 0; r < 64; r++) for (int c = 0; c < 64; c++) W[r*64+c] = (int8_t)(c+1);
        for (int i = 0; i < 64*2; i++) sc[i] = 256;
        for (int i = 0; i < 64; i++) act[i] = 1;
        q8_tile_golden(W, sc, act, out);
        for (int r = 0; r < 64; r++) CHECK(out[r] == 2080);
    }

    // ---- Q5 d_pre worked example: d=0.25, norm=1.0 -> 16384 ----
    CHECK(q5_d_pre(0x3400, 256) == 16384);    // 0.25 x 1.0 = 0.25 -> 16384

    // ---- Q5 decode: build a block where all q5 = +1 (encoded value 17) ----
    {
        uint8_t blk[22];
        blk[0] = 0x00; blk[1] = 0x34;         // d = 0.25 (0x3400)
        // q5 encoded value 17 = 0b10001 -> qh bit=1, ql=1
        uint32_t qh = 0xFFFFFFFFu;             // all high bits set
        blk[2] = qh & 0xFF; blk[3] = (qh>>8) & 0xFF;
        blk[4] = (qh>>16) & 0xFF; blk[5] = (qh>>24) & 0xFF;
        for (int i = 0; i < 16; i++) blk[6+i] = 0x11;  // low & high nibbles = 1
        for (int wi = 0; wi < 32; wi++) CHECK(q5_decode(blk, wi) == 1);
    }

    // ---- Q5 tile: all q5=+1, d=0.25, norm=1.0, act=1 -> 896 x 16384 = 14680064 ----
    {
        uint8_t blocks[4*28*22]; uint16_t norm[4]; int16_t act[896]; int64_t out[4];
        for (int r = 0; r < 4; r++) for (int b = 0; b < 28; b++) {
            uint8_t* blk = blocks + (r*28+b)*22;
            blk[0] = 0x00; blk[1] = 0x34;         // d = 0.25
            uint32_t qh = 0xFFFFFFFFu;
            blk[2] = qh & 0xFF; blk[3] = (qh>>8) & 0xFF;
            blk[4] = (qh>>16) & 0xFF; blk[5] = (qh>>24) & 0xFF;
            for (int i = 0; i < 16; i++) blk[6+i] = 0x11;
        }
        for (int i = 0; i < 4; i++) norm[i] = 256;      // 1.0
        for (int i = 0; i < 896; i++) act[i] = 1;
        q5_tile_golden(blocks, norm, act, out);
        // d_pre = 16384, dq = 16384*1 = 16384, sum over 896 = 14680064
        for (int r = 0; r < 4; r++) CHECK(out[r] == 14680064);
    }

    if (g_fail == 0) printf("ALL GOLDEN MODEL TESTS PASSED\n");
    else             printf("%d CHECKS FAILED\n", g_fail);
    return g_fail;
}
