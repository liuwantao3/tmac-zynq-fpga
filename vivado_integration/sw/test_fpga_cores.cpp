// Standalone FPGA Core Test Wrapper — results via DDR buffer, no UART
#include "tmac_baremetal.h"

#define MAX_Q8_ROWS 64
#define MAX_Q5_ROWS 4

// Output buffer at fixed address 0x1F000000 (OUTPUT_BUF):
//   [0] = magic: 0xBAD0=started, 0xBAD1=all done
//   [1..8] = progress markers
//   [9] = g_ntests
//  [10] = g_npassed
//  [11] = g_nfailed
//  [12..] = per-test details

#define OUT(i, v) do { volatile uint32_t* _o = (volatile uint32_t*)(uintptr_t)OUTPUT_BUF; _o[i] = (v); } while(0)

static int g_ntests = 0, g_npassed = 0, g_nfailed = 0;
static void set_marker(int idx, uint32_t val) { OUT(idx, val); }
static inline int fneq(float a, float b) { float d = a - b; return (d < -0.5f) || (d > 0.5f); }

// ===== Q8_0 Column-major weight gen =====
typedef float (*q8_pattern_fn)(int row, int col);

static void gen_q8_weights(uint8_t* fpga_wt, int nrows, q8_pattern_fn fn) {
    for (int r = 0; r < nrows; r++)
        for (int c = 0; c < Q8_GROUP_COLS; c++) {
            int8_t i8 = (int8_t)fn(r, c);
            fpga_wt[c * nrows + r] = (uint8_t)i8;
        }
}

// ===== Q8 reference matmul (single group, reused for all 14 groups) =====
static void q8_ref_matmul(const uint8_t* fpga_wt, const int16_t* x_q,
                          float* y, int nrows, float x_scale)
{
    memset(y, 0, nrows * sizeof(float));
    for (int r = 0; r < nrows; r++) {
        int32_t acc = 0;
        for (int c = 0; c < 896; c++) {
            int ci = c % 64;
            int8_t w = (int8_t)fpga_wt[ci * nrows + r];
            acc += (int32_t)w * (int32_t)x_q[c];
        }
        y[r] = (float)acc * x_scale;
    }
}

// ===== Q8_0 FPGA test =====
// Read 48-bit signed accumulator from two 32-bit DDR words (zero-extended to 64b)
static inline int32_t read48(const uint32_t* base, int i) {
    uint32_t lo = base[i * 2];
    uint32_t hi = base[i * 2 + 1];
    uint64_t raw = (uint64_t)lo | ((uint64_t)hi << 32);
    // Sign-extend from bit 47
    if (raw & ((uint64_t)1 << 47)) raw |= 0xFFFF000000000000ULL;
    return (int32_t)(int64_t)raw;  // S24.8 fits in int32_t
}

static void test_q8(const char* name, int idx, q8_pattern_fn w_fn,
                    const float* act)
{
    set_marker(idx, 0x80000000 | idx);
    OUT(1, 0x0000u);  // clear progress
    OUT(2, 0x1001u);
    uint32_t wt_addr = FPGA_WEIGHT_REFMT;
    uint32_t act_addr = FPGA_ACT_BUF;
    uint32_t res_addr = FPGA_RES_BUF;
    uint32_t desc_addr = DESC_CHAIN_BASE;

    // 1. Generate ONE group's weights
    OUT(2, 0x1010u);
    gen_q8_weights((uint8_t*)(uintptr_t)wt_addr, 64, w_fn);
    OUT(2, 0x1020u);

    // 2. Scales: 14 groups x 256 bytes at weight_addr + 4096 + g*256
    OUT(2, 0x1021u);
    for (int g = 0; g < 14; g++) {
        uint16_t* sc = (uint16_t*)(uintptr_t)(wt_addr + 4096 + g * 256);
        for (int i = 0; i < 128; i++) sc[i] = 0x0100;
    }

    // 3. Activations per group at act_addr + g*128
    float x_scale = 1.0f;
    for (int j = 0; j < 896; j++) { float a = act[j] < 0 ? -act[j] : act[j]; if (a > x_scale) x_scale = a; }
    x_scale = (x_scale < 1e-10f) ? 1.0f : x_scale / 32767.0f;
    for (int g = 0; g < 14; g++) {
        int16_t* xq = (int16_t*)(uintptr_t)(act_addr + g * 128);
        for (int j = 0; j < 64; j++) {
            float v = act[g * 64 + j] / x_scale;
            xq[j] = (int16_t)(v + (v >= 0 ? 0.5f : -0.5f));
        }
    }

    // 4. Descriptor
    Descriptor* d = (Descriptor*)(uintptr_t)desc_addr;
    d->next_addr = 0; d->weight_addr = wt_addr; d->act_addr = act_addr;
    d->result_addr = res_addr; d->tensor_type = DESC_Q8;
    d->num_groups = 14; d->act_total_bytes[0] = 128;
    d->act_total_bytes[1] = 0; d->act_total_bytes[2] = 0;

    // 5. Run
    reg_write32(REG_Q8_NUM_GROUPS, 14);
    reg_write32(REG_DESC_BASE, desc_addr);
    __asm__ volatile("dsb" ::: "memory");
    reg_write32(REG_START, 1);
    uint32_t tout = 50000;
    while (tout--) { if (!(reg_read32(REG_STATUS) & 0x8000)) break; }
    set_marker(idx, 0x80000002 | idx);

    // 6. Read results (48-bit signed, zero-extended to 64b in DDR)
    uint32_t* res32 = (uint32_t*)(uintptr_t)res_addr;
    float y_fpga[64];
    for (int i = 0; i < 64; i++) {
        int32_t acc = read48(res32, i);
        y_fpga[i] = (float)acc * x_scale;
    }

    // 7. Reference
    int16_t x_q[896];
    for (int j = 0; j < 896; j++) {
        float v = act[j] / x_scale;
        x_q[j] = (int16_t)(v + (v >= 0 ? 0.5f : -0.5f));
    }
    float y_ref[64];
    q8_ref_matmul((uint8_t*)(uintptr_t)wt_addr, x_q, y_ref, 64, x_scale);

    // 8. Compare
    int ok = 1;
    for (int i = 0; i < 64 && ok; i++) {
        if (fneq(y_fpga[i], y_ref[i])) ok = 0;
    }
    OUT(12 + idx, ok ? 1u : 0u);
    if (ok) g_npassed++; else g_nfailed++;
    g_ntests++;
}

// ===== Q5_0 Block Generator =====
static void gen_q5_blocks(uint8_t* buf, int nrows, int ncols, int8_t q5_val) {
    int stride = ncols / 32;
    for (int r = 0; r < nrows; r++)
        for (int bi = 0; bi < stride; bi++) {
            uint8_t* blk = buf + (r * stride + bi) * 22;
            uint16_t d_f16;
            float d_f32 = (float)q5_val;
            uint32_t fu; memcpy(&fu, &d_f32, 4);
            uint32_t sign = (fu >> 31) & 1;
            int32_t exp = ((int32_t)((fu >> 23) & 0xFF)) - 127;
            uint32_t mant = fu & 0x7FFFFF;
            if (exp >= 16) d_f16 = (sign << 15) | (0x1F << 10);
            else if (exp < -14) d_f16 = sign << 15;
            else {
                uint32_t f16_exp = exp + 15;
                uint32_t f16_mant = mant >> 13;
                d_f16 = (sign << 15) | ((f16_exp & 0x1F) << 10) | (f16_mant & 0x3FF);
            }
            blk[0] = d_f16 & 0xFF; blk[1] = d_f16 >> 8;
            blk[2] = blk[3] = blk[4] = blk[5] = 0;
            uint8_t nib = (uint8_t)(q5_val + 16) & 0xF;
            for (int wi = 0; wi < 16; wi++) blk[6 + wi] = nib | (nib << 4);
        }
}

// ===== Q5_0 FPGA test =====
static void test_q5(const char* name, int idx, int8_t q5_val) {
    set_marker(idx, 0x90000000 | idx);
    uint32_t wt_addr = FPGA_WEIGHT_REFMT;
    uint32_t act_addr = FPGA_ACT_BUF;
    uint32_t res_addr = FPGA_RES_BUF;
    uint32_t desc_addr = DESC_CHAIN_BASE;

    gen_q5_blocks((uint8_t*)(uintptr_t)wt_addr, 4, 896, q5_val);
    for (int r = 0; r < 4; r++) {
        uint16_t uq = (uint16_t)(1.0f * 256.0f + 0.5f);
        memcpy((void*)(uintptr_t)(wt_addr + 4928 + r * 2), &uq, 2);
    }

    int16_t x_q[896];
    for (int j = 0; j < 896; j++) x_q[j] = 1;
    memcpy((void*)(uintptr_t)act_addr, x_q, 896 * 2);

    Descriptor* d = (Descriptor*)(uintptr_t)desc_addr;
    d->next_addr = 0; d->weight_addr = wt_addr; d->act_addr = act_addr;
    d->result_addr = res_addr; d->tensor_type = DESC_Q5_0;
    d->num_groups = 0;
    int act_bytes = 896 * 2;
    d->act_total_bytes[0] = act_bytes & 0xFF;
    d->act_total_bytes[1] = (act_bytes >> 8) & 0xFF;
    d->act_total_bytes[2] = (act_bytes >> 16) & 0xFF;

    reg_write32(REG_DESC_BASE, desc_addr);
    __asm__ volatile("dsb" ::: "memory");
    reg_write32(REG_START, 1);
    uint32_t tout = 50000;
    while (tout--) { if (!(reg_read32(REG_STATUS) & 0x8000)) break; }
    set_marker(idx, 0x90000002 | idx);

    uint32_t* res32 = (uint32_t*)(uintptr_t)res_addr;
    float expected = (float)(q5_val * q5_val) * 896.0f;
    int ok = 1;
    for (int i = 0; i < 4 && ok; i++) {
        int32_t acc = read48(res32, i);
        float y = (float)acc;
        if (fneq(y, expected)) ok = 0;
    }
    OUT(12 + idx, ok ? 1u : 0u);
    if (ok) g_npassed++; else g_nfailed++;
    g_ntests++;
}

// ===== Patterns =====
static float p_all1(int r, int c) { (void)r; (void)c; return 1.0f; }
static float p_allm1(int r, int c) { (void)r; (void)c; return -1.0f; }

static float act_all1[896];
static float act_ramp[896];

extern "C" int main(void) {
    // Enable FPU (VFPv3 on Cortex-A9)
    __asm__ volatile(
        "mrc p15, 0, r0, c1, c0, 2\n"
        "orr r0, r0, #(0xF << 20)\n"
        "mcr p15, 0, r0, c1, c0, 2\n"
        "dsb\nisb\n"
        "mov r3, #0x40000000\n"
        "vmsr fpexc, r3\n"
        ::: "r0", "r3", "memory");
    OUT(0, 0xBAD0u);  // started

    // Init activations
    OUT(1, 0x0001u);
    for (int i = 0; i < 896; i++) { act_all1[i] = 1.0f; act_ramp[i] = (float)(i % 127 - 63); }
    OUT(1, 0x0002u);

    // Run Q8 tests
    OUT(1, 0x0101u);
    test_q8("Q8 all-1s",  1, p_all1,  act_all1);
    OUT(1, 0x0102u);
    test_q8("Q8 all(-1)s", 2, p_allm1, act_all1);
    OUT(1, 0x0103u);

    // Run Q5 tests
    OUT(1, 0x0501u);
    test_q5("Q5 val=1",  5, 1);
    OUT(1, 0x0502u);
    test_q5("Q5 val=0",  6, 0);
    OUT(1, 0x0503u);
    test_q5("Q5 val=-1", 7, -1);
    OUT(1, 0x0504u);

    // Results
    OUT(0, 0xBAD1u);  // done
    OUT(9, (uint32_t)g_ntests);
    OUT(10, (uint32_t)g_npassed);
    OUT(11, (uint32_t)g_nfailed);
    return g_nfailed;
}
