// FPGA Core Test Wrapper - Q8_0 and Q5_0 cores via HP descriptor chain
// Block formats match API.md exactly.
#include "tmac_baremetal.h"

static int g_ntests = 0, g_npassed = 0, g_nfailed = 0;

#define OUT(i,v) do { volatile uint32_t*_o=(volatile uint32_t*)(uintptr_t)0x1F000000UL; _o[i]=(v); }while(0)
static inline int fneq(float a, float b) { float d = a - b; return (d < -0.5f) || (d > 0.5f); }

// Read 48-bit signed accumulator zero-extended to 64b in DDR
static inline int32_t read48(const uint32_t* base, int i) {
    uint64_t raw = (uint64_t)base[i*2] | ((uint64_t)base[i*2+1] << 32);
    if (raw & ((uint64_t)1 << 47)) raw |= 0xFFFF000000000000ULL;
    return (int32_t)(int64_t)raw;
}

// ===== Q8_0: 64×64 tile, single group (num_groups=1) =====
// Weight data (4096 B): column-major, 8 byte-lanes × 512 entries
// Scale data (256 B): at weight_addr + 4096, 8 banks × 16 entries × 2 bytes

static void gen_q8_weights(uint8_t* buf, int nrows, float (*fn)(int,int)) {
    for (int r = 0; r < nrows; r++)
        for (int c = 0; c < 64; c++)
            buf[c * nrows + r] = (uint8_t)(int8_t)fn(r, c);
}

static void test_q8(const char* name, int idx, float (*w_fn)(int,int),
                    const float* act) {
    OUT(2, 0x80000000 | idx);
    uint32_t wt = 0x1F004000, act_a = 0x1F002000, res = 0x1F003000, desc = 0x1F001000;
    // Generate ONE group's weights (4096 bytes)
    gen_q8_weights((uint8_t*)(uintptr_t)wt, 64, w_fn);
    // Scales at wt + 4096: 8 banks × 16 × 2 bytes = 256 bytes, all = 1.0 (UQ8.8 0x0100)
    uint16_t* sc = (uint16_t*)(uintptr_t)(wt + 4096);
    for (int i = 0; i < 128; i++) sc[i] = 0x0100;
    // Activations: 64 int16 at act_addr
    int16_t* aq = (int16_t*)(uintptr_t)act_a;
    for (int i = 0; i < 64; i++) aq[i] = (int16_t)(act[i] + (act[i]>=0?0.5f:-0.5f));
    // Descriptor: Q8 (type 0), num_groups=1, act_bytes=128
    uint32_t* d = (uint32_t*)(uintptr_t)desc;
    d[0]=0; d[1]=wt; d[2]=act_a; d[3]=res; d[4]=0; d[5]=0x00000100; d[6]=128; d[7]=0;
    // Start
    reg_write32(0x10, 1);      // REG_Q8_NUM_GROUPS = 1
    reg_write32(0x18, desc);   // REG_DESC_BASE
    __asm__ volatile("dsb" ::: "memory");
    reg_write32(0x00, 1);      // REG_START
    uint32_t tout = 50000;
    while (tout--) { if (!(reg_read32(0x14) & 0x8000)) break; }
    // Read results
    uint32_t* r32 = (uint32_t*)(uintptr_t)res;
    float y_fpga[64];
    for (int i = 0; i < 64; i++) y_fpga[i] = (float)read48(r32, i);
    // Reference: W×vec, 64 cols
    float y_ref[64];
    memset(y_ref, 0, sizeof(y_ref));
    for (int r = 0; r < 64; r++)
        for (int c = 0; c < 64; c++)
            y_ref[r] += (float)(int8_t)act[c] * (float)(int8_t)w_fn(r, c);
    // Compare
    int ok = 1;
    for (int i = 0; i < 64 && ok; i++) if (fneq(y_fpga[i], y_ref[i])) ok = 0;
    OUT(12+idx, ok?1u:0u);
    if (ok) g_npassed++; else g_nfailed++;
    g_ntests++;
}

// ===== Q5_0: 4-row tile, 48-byte blocks =====
// Block (48 B): core0_d(2)+qh(4)+qs(16) + core1_d(2)+qh(4)+qs(16) + pad(4)
// 56 blocks/tile, then 8 bytes norm (4×UQ8.8) = 2696 B/tile
// Activations: 896 × int16 at act_addr

static uint16_t f32_to_f16(float f) {
    uint32_t u; memcpy(&u, &f, 4);
    uint32_t s=(u>>31)&1, e=((u>>23)&0xFF)-127, m=u&0x7FFFFF;
    if (e>=16) return (s<<15)|(0x1F<<10);
    if (e<-14) return s<<15;
    uint32_t f16_e=e+15, f16_m=m>>13;
    if ((m>>12)&1 && (f16_m&1)) f16_m++;
    if (f16_m>=0x400) { f16_m=0; f16_e++; }
    return (s<<15)|((f16_e&0x1F)<<10)|(f16_m&0x3FF);
}

// f16_decode matching Verilog matmul_q5_0_core.v (sign-agnostic, S24.8 output: 1.0→256)
static int32_t f16_decode_c(uint16_t f16) {
    int32_t exp = (f16 >> 10) & 0x1F;
    int32_t mant = f16 & 0x3FF;
    if (exp == 0 || exp == 31) return 0;
    if (exp >= 17) return (1024 + mant) << (exp - 17);
    return ((1024 + mant + (1 << (16 - exp))) >> (17 - exp));
}

static void gen_q5_tile(uint8_t* buf, int8_t q5_val) {
    uint16_t d_f16 = f32_to_f16(1.0f);      // GGUF block scale: always non-negative
    uint8_t enc = (uint8_t)(q5_val + 16);  // Q5 encoded value: q5_val + 16 (0..31)
    uint8_t ql = enc & 0xF;               // low 4 bits
    uint32_t qh = (enc & 0x10) ? 0xFFFFFFFFu : 0u;  // bit 4 for ALL 32 elements
    for (int bi = 0; bi < 56; bi++) {
        uint8_t* blk = buf + bi * 48;
        // core0: d (2), qh (4), qs (16) = 22 bytes
        blk[0] = d_f16 & 0xFF; blk[1] = d_f16 >> 8;
        blk[2] = qh & 0xFF; blk[3] = (qh>>8) & 0xFF;
        blk[4] = (qh>>16) & 0xFF; blk[5] = (qh>>24) & 0xFF;
        for (int wi = 0; wi < 16; wi++) blk[6+wi] = ql | (ql << 4);
        // core1: same values
        blk[22] = blk[0]; blk[23] = blk[1];
        blk[24] = blk[2]; blk[25] = blk[3]; blk[26] = blk[4]; blk[27] = blk[5];
        for (int wi = 0; wi < 16; wi++) blk[28+wi] = ql | (ql << 4);
        blk[44]=blk[45]=blk[46]=blk[47]=0;
    }
    // Norm values at tile + 2688: all 1.0 = UQ8.8 0x0100
    for (int r = 0; r < 4; r++) {
        buf[2688 + r*2] = 0x00; buf[2688 + r*2+1] = 0x01;
    }
}

static void test_q5(const char* name, int idx, int8_t q5_val) {
    OUT(2, 0x90000000 | idx);
    uint32_t wt = 0x1F004000, act_a = 0x1F002000, res = 0x1F003000, desc = 0x1F001000;
    // Generate one Q5_0 tile (2696 bytes)
    gen_q5_tile((uint8_t*)(uintptr_t)wt, q5_val);
    // Activations: all 1
    int16_t* aq = (int16_t*)(uintptr_t)act_a;
    for (int i = 0; i < 896; i++) aq[i] = 1;
    // Descriptor: Q5 (type 1), num_tiles=1, act_bytes=1792
    uint32_t* d = (uint32_t*)(uintptr_t)desc;
    d[0]=0; d[1]=wt; d[2]=act_a; d[3]=res; d[4]=0x00000001; d[5]=0x00000100; d[6]=1792; d[7]=0;
    // Start
    reg_write32(0x18, desc);
    __asm__ volatile("dsb" ::: "memory");
    reg_write32(0x00, 1);
    uint32_t tout = 50000;
    while (tout--) { if (!(reg_read32(0x14) & 0x8000)) break; }
    // Read results (4 rows)
    uint32_t* r32 = (uint32_t*)(uintptr_t)res;
    // FPGA: d_pre = f16_decode(d) × norm >> 8, acc = Σ d_pre × q5 × act
    // d = 1.0 → f16_decode(0x3C00)=256, norm=256 → d_pre=256
    // q5_decoded = q5_val, act=1 → row = d_pre × q5_val × 896
    int32_t d_fp   = f16_decode_c(f32_to_f16(1.0f));   // 256
    int32_t d_pre  = (d_fp * 256) >> 8;                 // norm=256 UQ8.8 → 256
    float expected = (float)(d_pre * q5_val) * 896.0f;
    int ok = 1;
    for (int i = 0; i < 4 && ok; i++)
        if (fneq((float)read48(r32, i), expected)) ok = 0;
    OUT(12+idx, ok?1u:0u);
    if (ok) g_npassed++; else g_nfailed++;
    g_ntests++;
}

static float p_all1(int r, int c) { (void)r; (void)c; return 1.0f; }
static float p_allm1(int r, int c) { (void)r; (void)c; return -1.0f; }
static float act1[64];

extern "C" int main(void) {
    __asm__ volatile(
        "mrc p15,0,r0,c1,c0,2\n orr r0,r0,#(0xF<<20)\n mcr p15,0,r0,c1,c0,2\n dsb\nisb\n"
        "mov r3,#0x40000000\n vmsr fpexc,r3\n" ::: "r0","r3","memory");
    OUT(0, 0xBAD0u);
    for (int i = 0; i < 64; i++) act1[i] = 1.0f;
    test_q8("Q8 all-1s",  1, p_all1,  act1);
    test_q8("Q8 all(-1)s", 2, p_allm1, act1);
    test_q5("Q5 val=1",  5, 1);
    test_q5("Q5 val=0",  6, 0);
    test_q5("Q5 val=-1", 7, -1);
    OUT(0, 0xBAD1u);
    OUT(9, (uint32_t)g_ntests);
    OUT(10, (uint32_t)g_npassed);
    OUT(11, (uint32_t)g_nfailed);
    return g_nfailed;
}
