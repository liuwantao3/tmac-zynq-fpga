#ifndef TMAC_BAREMETAL_H
#define TMAC_BAREMETAL_H

#include <stdint.h>

// ===== Model Constants (Qwen2-0.5B) =====
#define HIDDEN_DIM     896
#define INTER_DIM      4864
#define VOCAB_SIZE     151936
#define NUM_LAYERS     24
#define NUM_HEADS      14
#define HEAD_DIM       64
#define NUM_KV_HEADS   2
#define K_DIM          128   // NUM_KV_HEADS * HEAD_DIM
#define V_DIM          128
#define MAX_SEQ_LEN    256

// Tensor types
#define TENSOR_F32     0
#define TENSOR_F16     1
#define TENSOR_Q8_0    8
#define TENSOR_Q6_K    14
#define TENSOR_Q5_0    6
#define TENSOR_Q4_K    12

// ===== DDR Memory Map =====
// Model loaded at MODEL_BASE by bootloader/script
// Program lives at 0x00100000 (from link.ld)
#define MODEL_BASE           0x00200000UL
#define OUTPUT_BUF           0x1F000000UL   // Output tokens
#define DESC_CHAIN_BASE      0x1F001000UL   // Descriptor chain (512 bytes = 16×32)
#define FPGA_ACT_BUF         0x1F002000UL   // Activation scratch (8KB)
#define FPGA_RES_BUF         0x1F003000UL   // Result scratch (8KB)
#define FPGA_WEIGHT_REFMT    0x1F004000UL   // Weight reformat (128KB)
#define SCRATCH_F32          0x1F010000UL   // Float scratch for CPU ops (64KB)

// Chain intermediate buffers (INT16 activations) within FPGA_ACT_BUF
#define BUF_ACT              FPGA_ACT_BUF           // Input activation for current op
#define BUF_NORM_OUT         (FPGA_ACT_BUF + 0x800) // Norm output (2KB)
#define BUF_Q                (FPGA_ACT_BUF + 0x1000) // Q (2KB)
#define BUF_K                (FPGA_ACT_BUF + 0x1800) // K (2KB, but K_DIM=128)
#define BUF_V                (FPGA_ACT_BUF + 0x2000) // V (2KB)
#define BUF_CONTEXT          (FPGA_ACT_BUF + 0x2800) // Context (2KB)
#define BUF_ATTN_OUT         (FPGA_ACT_BUF + 0x3000) // Attn output (2KB)
#define BUF_FFN_NORM_OUT     (FPGA_ACT_BUF + 0x3800) // FFN norm output (2KB)
#define BUF_GATE             (FPGA_ACT_BUF + 0x4000) // Gate (4KB)
#define BUF_UP               (FPGA_ACT_BUF + 0x5000) // Up (4KB)
#define BUF_SWIGLU_OUT       (FPGA_ACT_BUF + 0x6000) // SwiGLU output (4KB)
#define BUF_FFN_OUT          (FPGA_ACT_BUF + 0x7000) // FFN output (2KB)

// CPU_OP descriptor indices within the layer chain
#define CHAIN_IDX_ATTN_NORM       0
#define CHAIN_IDX_Q_MATMUL        1
#define CHAIN_IDX_K_MATMUL        2
#define CHAIN_IDX_V_MATMUL        3
#define CHAIN_IDX_BIAS_ROPE_ATTN  4
#define CHAIN_IDX_ATTN_OUT_MATMUL 5
#define CHAIN_IDX_RESIDUAL_NORM   6
#define CHAIN_IDX_GATE_MATMUL     7
#define CHAIN_IDX_UP_MATMUL       8
#define CHAIN_IDX_SWIGLU          9
#define CHAIN_IDX_RESIDUAL        10
#define CHAIN_MAX_DESC            11

// Chain timeout (in status polls)
#define CHAIN_TIMEOUT      10000000
#define CHAIN_POLL_US      10

// Register map (matches hp_fsm_top.v)
#define IP_BASE            0x43C00000UL
#define REG_START          0x00
#define REG_CHAIN_CTRL     0x04
#define REG_GIE            0x08
#define REG_ISR            0x0C
#define REG_Q8_NUM_GROUPS  0x10
#define REG_STATUS         0x14
#define REG_DESC_BASE      0x18
#define REG_DESC_TAIL      0x1C
#define REG_DESC_HEAD      0x20
#define REG_DEBUG          0x28
#define REG_CLK_CNT        0x2C
#define REG_CLK_CNT_SLOW   0x30
#define REG_Q8_DEBUG       0x3C
#define REG_Q5_DEBUG       0x40
#define REG_Q5_DBG_CAP0    0x44

// CHAIN_CTRL bits
#define CHAIN_CTRL_RESUME        (1<<0)
#define CHAIN_CTRL_CPU_OP_PENDING (1<<2)
#define CHAIN_CTRL_INTR_ENABLE   (1<<3)

// Descriptor format (32 bytes, packed)
typedef struct __attribute__((packed)) {
    uint32_t next_addr;          // 0x00
    uint32_t weight_addr;        // 0x04
    uint32_t act_addr;           // 0x08
    uint32_t result_addr;        // 0x0C
    uint16_t tensor_type;        // 0x10: 0=Q8, 1=Q5_0, 15=CPU_OP
    uint16_t reserved0;          // 0x12
    uint8_t  num_groups;         // 0x14: [3:0]=Q8 column groups
    uint8_t  reserved1;          // 0x15
    uint16_t num_tiles;          // 0x16-0x17 (LE)
    uint8_t  act_total_bytes[3]; // 0x18-0x1A (24-bit LE)
    uint8_t  reserved3[5];       // 0x1B-0x1F
} Descriptor;

// Descriptor constants
#define DESC_Q8     0
#define DESC_Q5_0   1
#define DESC_CPU_OP 15

// Helper: write descriptor fields
static inline void desc_write(Descriptor* d, uint32_t next, uint32_t weight,
    uint32_t act, uint32_t result, uint16_t type, uint8_t groups,
    uint16_t tiles, uint32_t act_bytes)
{
    d->next_addr = next;
    d->weight_addr = weight;
    d->act_addr = act;
    d->result_addr = result;
    d->tensor_type = type;
    d->num_groups = groups;
    d->num_tiles = tiles;
    d->act_total_bytes[0] = act_bytes & 0xFF;
    d->act_total_bytes[1] = (act_bytes >> 8) & 0xFF;
    d->act_total_bytes[2] = (act_bytes >> 16) & 0xFF;
}

// Q8_0 tile constants
#define Q8_TILE_ROWS    64
#define Q8_TILE_COLS    896
#define Q8_GROUP_COLS   64
#define Q8_NUM_GROUPS   14
#define Q8_GROUP_BYTES  4096
#define Q8_TILE_WEIGHT_BYTES (Q8_NUM_GROUPS * Q8_GROUP_BYTES)
#define Q8_GROUP_SCALE_BYTES 256   // 64 × UQ8.8 per group
#define Q8_TILE_SCALE_BYTES  (Q8_NUM_GROUPS * Q8_GROUP_SCALE_BYTES)
#define Q8_TILE_STRIDE       (Q8_TILE_WEIGHT_BYTES + Q8_TILE_SCALE_BYTES) // 14 × 4352 = 60928

// Q5_0 tile constants
#define Q5_TILE_ROWS        4
#define Q5_TILE_BLOCKS      56   // FPGA blocks per tile (28/row × 2 rows per core)
#define Q5_BLOCK_SIZE       48   // bytes per FPGA block (2 GGUF blocks × 22 + 4 pad)
#define Q5_TILE_BYTES       2688 // 56 × 48
#define Q5_TILE_NORM_OFFSET 2688 // norm follows block data
#define Q5_TILE_TOTAL       2696 // 2688 + 8 bytes norm

// ===== Tensor Table Entry =====
typedef struct {
    char     name[128];
    uint64_t rows;
    uint64_t cols;
    uint32_t type;
    uint64_t n_bytes;
    uint8_t* data;
} Tensor;

// ===== Inference State =====
typedef struct {
    Tensor* tensors;
    int     ntensors;
    float   k_cache[NUM_LAYERS][MAX_SEQ_LEN][K_DIM];
    float   v_cache[NUM_LAYERS][MAX_SEQ_LEN][V_DIM];
    int     seq_len;
    float   hidden[HIDDEN_DIM];
    float   logits[VOCAB_SIZE];
    int     output_tokens[MAX_SEQ_LEN * 2];
    int     n_output_tokens;
} InferenceState;

// ===== Register Access =====
static inline void reg_write32(uint32_t off, uint32_t val) {
    volatile uint32_t* p = (volatile uint32_t*)(uintptr_t)(IP_BASE + off);
    __asm__ volatile("dsb" ::: "memory");
    *p = val;
    __asm__ volatile("dsb" ::: "memory");
}

static inline uint32_t reg_read32(uint32_t off) {
    volatile uint32_t* p = (volatile uint32_t*)(uintptr_t)(IP_BASE + off);
    __asm__ volatile("dsb" ::: "memory");
    uint32_t v = *p;
    __asm__ volatile("dsb" ::: "memory");
    return v;
}

// ===== Math Helpers (no standard lib) =====
static inline int min_int(int a, int b) { return a < b ? a : b; }
static inline int max_int(int a, int b) { return a > b ? a : b; }
static inline int clamp_int(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// FP16 ↔ FP32
static inline float f16_to_f32(uint16_t f16) {
    uint32_t sign = (f16 >> 15) & 0x1;
    uint32_t exp  = (f16 >> 10) & 0x1F;
    uint32_t mant = f16 & 0x3FF;
    if (exp == 0) {
        return (mant == 0) ? 0.0f
            : ((sign ? -1.0f : 1.0f) * (mant / 1024.0f) * 0.00006103515625f);
    }
    if (exp == 31) return mant ? 0.0f : (sign ? -1.0f / 0.0f : 1.0f / 0.0f);
    float val = 1.0f + mant / 1024.0f;
    int e = (int)exp - 15;
    if (e >= 0) { while (e--) val *= 2.0f; }
    else { while (e++) val *= 0.5f; }
    return sign ? -val : val;
}

// Math functions for bare-metal ARM (no libm available).
// Uses VFPv3 hardware for sqrt/round, polynomial approximations for others.

static inline float sqrtf(float x) { float r; __asm__("vsqrt.f32 %0, %1" : "=t"(r) : "t"(x)); return r; }
static inline float roundf(float x) { return (float)(int)(x + (x >= 0 ? 0.5f : -0.5f)); }
static inline float fabsf(float x) { return x < 0 ? -x : x; }

// expf(x) — range-reduced polynomial, accurate to ~1e-5
static inline float expf(float x) {
    if (x < -80.0f) return 0.0f;
    if (x > 80.0f) return 3.40282347e+38f;
    float n = roundf(x * 1.442695040f); // x / ln(2)
    float f = x - n * 0.693147180f;     // remainder
    float f2 = f * f, f3 = f2 * f, f4 = f2 * f2;
    float p = 1.0f + f + f2 * 0.5f + f3 * 0.1666667f + f4 * 0.0416667f + f3 * f2 * 0.00833333f;
    union { float f; uint32_t u; } fu = {p};
    fu.u += (uint32_t)((int)n) << 23;
    return fu.f;
}

// logf(x) — for x > 0, accurate to ~1e-5
static inline float logf(float x) {
    union { float f; uint32_t u; } fu = {x};
    int e = ((fu.u >> 23) & 0xFF) - 127;
    fu.u = (fu.u & 0x7FFFFF) | (127 << 23); // normalize to [1,2)
    float t = fu.f - 1.0f;
    float t2 = t * t, t3 = t2 * t, t4 = t2 * t2;
    float p = t - t2 * 0.5f + t3 * 0.3333333f - t4 * 0.25f + t3 * t2 * 0.2f;
    return (float)e * 0.693147180f + p;
}

// sinf(x) — range reduce to [-pi/4, pi/4], polynomial
static inline float sinf(float x) {
    // Reduce to [0, 2pi)
    float p = x * 0.159154943f; // 1/(2pi)
    float n = roundf(p - 0.5f); // nearest half-integer
    float r = x - n * 6.283185307f; // remainder in [-pi, pi]
    // Further reduce to [-pi/2, pi/2]
    if (r > 1.570796327f) r = 3.141592654f - r;
    else if (r < -1.570796327f) r = -3.141592654f - r;
    // sin(r) ≈ r - r^3/6 + r^5/120 - r^7/5040
    float r2 = r * r;
    float r3 = r2 * r;
    float r5 = r3 * r2;
    float r7 = r5 * r2;
    float s = r - r3 * 0.1666667f + r5 * 0.00833333f - r7 * 0.000198413f;
    // Adjust sign by n (n is half-integer, sin(x + n*pi) = (-1)^n * sin(x))
    uint32_t ni = (uint32_t)(n + 0.5f);
    return (ni & 1) ? -s : s;
}

// cosf(x) — cos(x) = sin(x + pi/2)
static inline float cosf(float x) {
    return sinf(x + 1.570796327f);
}

// powf(x, y) — via exp(y * log(x))
static inline float powf(float x, float y) {
    if (x <= 0.0f) return 0.0f;
    return expf(y * logf(x));
}

// memcpy/memset for bare-metal
static inline void* memcpy(void* d, const void* s, unsigned int n) {
    unsigned char* dp = (unsigned char*)d;
    const unsigned char* sp = (const unsigned char*)s;
    while (n--) *dp++ = *sp++;
    return d;
}
static inline void* memset(void* d, int c, unsigned int n) {
    unsigned char* dp = (unsigned char*)d;
    while (n--) *dp++ = (unsigned char)c;
    return d;
}

// UART0 initialization (115200 baud, 8N1)
static inline void uart_init(void) {
    volatile uint32_t* uart = (volatile uint32_t*)0xE0000000UL;
    // Disable TX/RX, reset
    uart[0] = 0x00000000;
    uart[0] = 0x00000001;  // reset
    uart[0] = 0x00000000;  // clear reset
    // Mode: normal mode, 8-bit, 1 stop, no parity, clock / 8 = 115200
    // MR = (CHMODE=01) | (CHRL=00 for 8bit) | (NBSTOP=0 for 1stop) | (PAR=000) | (CLKS=100 for /8)
    uart[1] = 0x00000020;  // MR: bits[6:4]=100 (div8), bits[1:0]=01 (normal)
    // Baud: 50 MHz / (8 * 115200) = 54.2 => 54
    uart[8] = 54;          // BRGR at offset 0x20
    uart[9] = 0x00000000;  // BDIV at offset 0x24 (no divisor)
    // Enable TX
    uart[0] = 0x00000020;  // CR: TXEN=1 (bit 5), RXEN=0
}

// Simple putchar via UART0 (PS7 UART at 0xE0000000)
static inline void uart_putc(char c) {
    volatile uint32_t* uart = (volatile uint32_t*)0xE0000000UL;
    if (c == '\n') uart_putc('\r');
    // Skip wait if TXFIFO full (don't hang)
    if (uart[5] & (1u << 4)) return;
    uart[7] = c;
}

static inline void uart_puts(const char* s) {
    while (*s) uart_putc(*s++);
}

static inline void uart_puthex(uint32_t val) {
    static const char hex[] = "0123456789ABCDEF";
    uart_putc('0'); uart_putc('x');
    for (int i = 28; i >= 0; i -= 4) uart_putc(hex[(val >> i) & 0xF]);
}

static inline void uart_putdec(int val) {
    if (val < 0) { uart_putc('-'); val = -val; }
    char buf[12]; int i = 0;
    if (val == 0) { uart_putc('0'); return; }
    while (val) { buf[i++] = '0' + (val % 10); val /= 10; }
    while (i) uart_putc(buf[--i]);
}

#endif
