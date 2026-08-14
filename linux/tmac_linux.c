/*
 * T-MAC Linux Userspace Inference Engine
 * FPGA accelerated: Q8_0 and Q5_0 via HP descriptor-chain (/dev/mem)
 * CPU: RMSNorm, RoPE, SiLU, SoftMax, Attention, Q4_K/Q6_K/F32 fallback
 *
 * Build: arm-linux-gnueabihf-gcc -O2 -o tmac tmac_linux.c -lm
 * Run:   ./tmac /path/to/model.tmac [token_id]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <math.h>
#include <unistd.h>
#include <time.h>

// ===== DDR Memory Map (same as bare-metal) =====
#define MODEL_BASE           0x00200000UL
#define OUTPUT_BUF           0x1F000000UL
#define DESC_CHAIN_BASE      0x1F001000UL
#define FPGA_WEIGHT_REFMT    0x1F004000UL
#define SCRATCH_F32          0x1F010000UL

// ===== Register Map =====
#define IP_BASE              0x43C00000UL
#define REG_START            0x00
#define REG_CHAIN_CTRL       0x04
#define REG_GIE             0x08
#define REG_ISR             0x0C
#define REG_Q8_NUM_GROUPS   0x10
#define REG_STATUS          0x14
#define REG_DESC_BASE        0x18
#define REG_DESC_TAIL        0x1C
#define REG_DESC_HEAD        0x20
#define REG_DEBUG           0x28
#define REG_CLK_CNT         0x2C
#define REG_Q8_DEBUG        0x3C
#define REG_Q5_DEBUG        0x40
#define CHAIN_CTRL_INTR_ENABLE (1<<3)

// ===== Model Constants =====
#define HIDDEN_DIM     896
#define INTER_DIM      4864
#define VOCAB_SIZE     151936
#define NUM_LAYERS     24
#define NUM_HEADS      14
#define HEAD_DIM       64
#define NUM_KV_HEADS   2
#define K_DIM          128
#define V_DIM          128
#define MAX_SEQ_LEN    256

// ===== Tensor Types =====
#define TENSOR_F32     0
#define TENSOR_F16     1
#define TENSOR_Q8_0    8
#define TENSOR_Q6_K    14
#define TENSOR_Q5_0    6
#define TENSOR_Q4_K    12

// ===== FPGA Constants =====
#define Q8_TILE_ROWS      64
#define Q8_TILE_COLS      896
#define Q8_GROUP_COLS     64
#define Q8_NUM_GROUPS     14
#define Q8_GROUP_BYTES    4096
#define Q8_GROUP_SCALE_BYTES 256
#define Q8_TILE_WEIGHT_BYTES (Q8_NUM_GROUPS * Q8_GROUP_BYTES)
#define Q8_TILE_SCALE_BYTES  (Q8_NUM_GROUPS * Q8_GROUP_SCALE_BYTES)
#define Q8_TILE_STRIDE       (Q8_TILE_WEIGHT_BYTES + Q8_TILE_SCALE_BYTES)

#define Q5_TILE_ROWS        4
#define Q5_TILE_BLOCKS      56
#define Q5_BLOCK_SIZE       48
#define Q5_TILE_BYTES       2688
#define Q5_TILE_NORM_OFFSET 2688
#define Q5_TILE_TOTAL       2696

#define DESC_Q8     0
#define DESC_Q5_0   1
#define DESC_CPU_OP 15
#define CHAIN_TIMEOUT 10000000

// ===== Global State =====
typedef struct {
    char     name[128];
    uint64_t rows, cols;
    uint32_t type;
    uint64_t n_bytes;
    uint8_t* data;
} Tensor;

static Tensor* g_tensors = NULL;
static int g_ntensors = 0;

static Tensor* get_tensor(const char* name); /* fwd decl (used by load_model) */

/* --cpu flag: force the pure-CPU matmul path for A/B comparison */
static int g_use_cpu = 0;
/* --compare flag: run BOTH CPU + FPGA paths per matmul and print the diff */
static int g_compare = 0;

/* F32 scratch for forward_layer (avoids stack overflow) */
static float* g_scratch = NULL;

/* /dev/mem mappings */
static volatile uint32_t* g_gp0 = NULL;  /* 0x43C00000 */
static float g_kcache[NUM_LAYERS][MAX_SEQ_LEN][K_DIM];
static float g_vcache[NUM_LAYERS][MAX_SEQ_LEN][V_DIM];
static float g_kcache_cpu[NUM_LAYERS][MAX_SEQ_LEN][K_DIM];
static float g_vcache_cpu[NUM_LAYERS][MAX_SEQ_LEN][V_DIM];
/* active KV cache (path selectable for --trace dual-path runs) */
static float (*g_akc)[MAX_SEQ_LEN][K_DIM] = g_kcache;
static float (*g_avc)[MAX_SEQ_LEN][V_DIM] = g_vcache;

// ===== /dev/mem Access =====
#include <sys/syscall.h>   /* cacheflush syscall (ARM) */

/* Flush/invalidate a DDR range so the PL (FPGA) sees CPU writes and the CPU
 * sees PL writes. /dev/mem on ARM maps DDR cacheable; without this the CPU
 * writes sit in L1/L2 (PL reads stale DDR -> garbage/zeros) and result reads
 * return stale cache. ARM cacheflush syscall: cacheflush(addr, end, flags)
 * CACHEFLUSH_DL1=1 flush (clean+invalidate) dcache, CACHEFLUSH_DL2=2,
 * CACHEFLUSH_DI=4 flush icache. NOTE flags=0 does NOTHING on ARM. */
#define CACHEFLUSH_DL1 1
#if defined(__arm__)
static inline void dcache_range(void* addr, size_t size) {
    syscall(__ARM_NR_cacheflush, (long)addr, (long)addr + size, CACHEFLUSH_DL1);
}
#else
static inline void dcache_range(void* addr, size_t size) { (void)addr; (void)size; }
#endif
static inline void dcache_flush(void* addr, size_t size) { dcache_range(addr, size); }
static inline void dcache_inval(void* addr, size_t size) { dcache_range(addr, size); }

static volatile uint32_t* map_mem(uint32_t base, size_t size) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem"); return NULL; }
    volatile uint32_t* p = mmap(NULL, size, PROT_READ|PROT_WRITE,
                                MAP_SHARED, fd, base);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap"); return NULL; }
    return p;
}

/* DDR window used by the FPGA chain (weights/acts/descriptors/results).
 * Under Linux the MMU is ON, so physical DDR addresses MUST be mapped via
 * /dev/mem before dereferencing them — the original code dereferenced the
 * raw physical constants as virtual pointers, which segfaulted immediately. */
#define DDR_MAP_BASE 0x1F000000UL
#define DDR_MAP_SIZE 0x40000UL   /* 256 KB covers OUTPUT_BUF..FPGA_WEIGHT_REFMT+stride */
static volatile uint32_t* g_ddr = NULL;   /* mapped virtual base */

static int map_ddr(void) {
    g_ddr = map_mem(DDR_MAP_BASE, DDR_MAP_SIZE);
    if (!g_ddr) { fprintf(stderr, "Failed to map DDR window 0x%08lx\n", (unsigned long)DDR_MAP_BASE); return -1; }
    printf("DDR window mapped: 0x%08lx +%lu -> %p\n", (unsigned long)DDR_MAP_BASE,
           (unsigned long)DDR_MAP_SIZE, (void*)g_ddr);
    return 0;
}

/* translate a physical DDR address to its mapped virtual address */
static inline void* ddr(uint32_t phys) {
    return (void*)((uintptr_t)g_ddr + (phys - DDR_MAP_BASE));
}

static inline uint32_t reg_read(int off) { return g_gp0[off/4]; }
static inline void reg_write(int off, uint32_t v) { g_gp0[off/4] = v; }

// ===== Model Loading =====
static int load_model(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) { perror("fopen"); return -1; }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "TMAC", 4) != 0) {
        fclose(f); fprintf(stderr, "Bad magic\n"); return -1;
    }

    uint64_t ntensors;
    if (fread(&ntensors, 8, 1, f) != 1) { fclose(f); return -1; }
    if (ntensors > 500) { fclose(f); fprintf(stderr, "Too many tensors\n"); return -1; }

    g_ntensors = (int)ntensors;
    g_tensors = calloc(g_ntensors, sizeof(Tensor));
    if (!g_tensors) { fclose(f); return -1; }

    for (int i = 0; i < g_ntensors; i++) {
        uint64_t name_len;
        fread(&name_len, 8, 1, f);
        if (name_len >= 128) name_len = 127;
        fread(g_tensors[i].name, 1, name_len, f);
        g_tensors[i].name[name_len] = 0;
        fread(&g_tensors[i].rows, 8, 1, f);
        fread(&g_tensors[i].cols, 8, 1, f);
        fread(&g_tensors[i].type, 4, 1, f);
        fread(&g_tensors[i].n_bytes, 8, 1, f);
        g_tensors[i].data = malloc(g_tensors[i].n_bytes);
        fread(g_tensors[i].data, 1, g_tensors[i].n_bytes, f);
    }
    fclose(f);
    printf("Loaded %d tensors\n", g_ntensors);
    return 0;
}

static Tensor* get_tensor(const char* name) {
    for (int i = 0; i < g_ntensors; i++)
        if (strcmp(g_tensors[i].name, name) == 0)
            return &g_tensors[i];
    return NULL;
}

static inline float f16_to_f32(uint16_t v) {
    uint32_t sign = (v >> 15) & 1, exp = (v >> 10) & 0x1F, mant = v & 0x3FF;
    /* Use float ternary for sign so -1.0f*mant does NOT wrap unsigned:
     * (sign?-1:1)*mant (int * uint32) overflows for negative subnormals,
     * decoding e.g. 0x80ac as +256.0 instead of -1.03e-5. Matches sim. */
    if (exp == 0) return (mant==0)?0.0f:((sign ? -1.0f : 1.0f) * (mant / 1024.0f) * 0.00006103515625f);
    if (exp==31) return mant?0.0f:(sign?-INFINITY:INFINITY);
    return (sign?-1.0f:1.0f)*ldexpf(1.0f+mant/1024.0f, (int)exp-15);
}

static float dequant(const Tensor* t, uint64_t idx) {
    uint8_t* d = t->data;
    if (t->type == TENSOR_F32) return ((float*)d)[idx];
    if (t->type == TENSOR_F16) return f16_to_f32(((uint16_t*)d)[idx]);
    if (t->type == TENSOR_Q8_0) {
        uint64_t bo = (idx/32) * 34;
        float scale = f16_to_f32((uint16_t)d[bo]|((uint16_t)d[bo+1]<<8));
        return (float)(int8_t)d[bo+2+(idx%32)] * scale;
    }
    if (t->type == TENSOR_Q5_0) {
        uint64_t bo = (idx/32) * 22;
        float d_val = f16_to_f32((uint16_t)d[bo]|((uint16_t)d[bo+1]<<8));
        uint32_t qh = *(uint32_t*)(d+bo+2);
        uint64_t j = (idx%32) < 16 ? (idx%32) : (idx%32)-16;
        uint8_t ql = ((idx%32) < 16) ? (d[bo+6+j]&0xF) : (d[bo+6+j]>>4);
        /* NOTE: `- 16` applies to the FULL (qh_bit<<4)|ql value. The original
         * `...|ql - 16` parsed as `...|(ql-16)` (precedence bug), corrupting
         * every Q5_0 value with the qh bit set. Matches sim/tmac_gguf.cpp:322. */
        int q = (int)((((qh>>(idx%32))&1)<<4) | ql) - 16;
        return d_val * (float)q;
    }
    if (t->type == TENSOR_Q6_K) {
        uint64_t bo = (idx/256)*210;
        float super = f16_to_f32(*(uint16_t*)(d+bo+208));
        uint64_t wi = idx%256;
        int half=wi/128, pos=wi%128, l=pos%32, sub=pos/32;
        /* ql/qh/scale reads MUST add bo (block base): only the super scale
         * did before, so blocks >= 1 read block 0's ql/qh/scales -> garbage. */
        int lo = bo + half*64 + l + (sub%2)*32;
        int ql = (d[lo]>>((sub<2)?0:4)) & 0xF;
        int qh = (d[bo+128+half*32+l]>>(sub*2)) & 3;
        /* `-32` applies to the FULL (qh<<4)|ql value (fixes | vs - precedence) */
        int q = (int)((qh<<4)|ql) - 32;
        return super * (float)(int8_t)d[bo+192+half*8+(l/16)+sub*2] * (float)q;
    }
    if (t->type == TENSOR_Q4_K) {
        uint64_t bo = (idx/256)*144;
        float d_val=f16_to_f32(*(uint16_t*)(d+bo)), dmin=f16_to_f32(*(uint16_t*)(d+bo+2));
        uint64_t wi=idx%256;
        int sub=wi/32, j=wi%32;
        uint8_t* sc=d+bo+4;
        int scv=(sub<4)?(sc[sub]&63):((sc[sub+4]&0xF)|((sc[sub-4]>>6)<<4));
        int mv=(sub<4)?(sc[sub+4]&63):((sc[sub+4]>>4)|((sc[sub]>>6)<<4));
        int q4 = (d[bo+16+(sub/2)*32+j]>>((sub%2)?4:0)) & 0xF;
        return d_val*scv*q4 - dmin*mv;
    }
    return 0.0f;
}

// ===== FPGA Chain Runner =====
typedef struct __attribute__((packed)) {
    uint32_t next_addr, weight_addr, act_addr, result_addr;
    uint16_t tensor_type, reserved0;
    uint8_t  num_groups, reserved1;
    uint16_t num_tiles;
    uint8_t  act_total_bytes[3], reserved2[5];
} Descriptor;

static void desc_write(Descriptor* d, uint32_t next, uint32_t wt, uint32_t act,
    uint32_t res, uint16_t type, uint8_t groups, uint16_t tiles, uint32_t ab) {
    d->next_addr=next; d->weight_addr=wt; d->act_addr=act; d->result_addr=res;
    d->tensor_type=type; d->num_groups=groups; d->num_tiles=tiles;
    d->act_total_bytes[0]=ab&0xFF; d->act_total_bytes[1]=(ab>>8)&0xFF;
    d->act_total_bytes[2]=(ab>>16)&0xFF;
}

static int chain_run(uint32_t base, int ndesc) {
    dcache_flush(ddr(base), 32);
    reg_write(REG_DESC_BASE, base);
    reg_write(REG_DESC_TAIL, 1);
    __sync_synchronize();
    reg_write(REG_START, 1);
    uint32_t timeout = CHAIN_TIMEOUT;
    while (timeout--) {
        if (!(reg_read(REG_STATUS) & 0x8000)) return 0;
        for (volatile int i = 0; i < 10; i++);
    }
    fprintf(stderr, "TIMEOUT: STATUS=0x%08x DEBUG=0x%08x\n",
            reg_read(REG_STATUS), reg_read(REG_DEBUG));
    return -1;
}

// ===== Q8 Preprocessing =====
static void q8_preprocess_tile(const Tensor* A, int row0, uint8_t* fpga_wt, float* row_scale) {
    int cols = (int)A->cols;
    // Compute per-row normalization scales (match C++ sim: max_abs/32767)
    for (int r = 0; r < Q8_TILE_ROWS; r++) {
        int row = row0 + r; float max_abs = 0.0f;
        if (row >= (int)A->rows) { row_scale[r] = 1.0f; continue; }
        for (int j = 0; j < cols; j++) {
            uint64_t flat = (uint64_t)row * cols + j;
            uint64_t bo = (flat / 32) * 34;
            int8_t v = (int8_t)A->data[bo + 2 + (flat % 32)];
            float d = f16_to_f32(*(uint16_t*)(A->data + bo));
            float a = (float)v * d; if (a < 0) a = -a;
            if (a > max_abs) max_abs = a;
        }
        row_scale[r] = (max_abs < 1e-10f) ? 1.0f : max_abs / 32767.0f;
    }
    for (int g = 0; g < Q8_NUM_GROUPS; g++) {
        int col0 = g * Q8_GROUP_COLS;
        uint8_t* go = fpga_wt + g * Q8_GROUP_BYTES;
        for (int r = 0; r < Q8_TILE_ROWS; r++) {
            int row = row0 + r;
            if (row >= (int)A->rows) break;
            for (int c = 0; c < Q8_GROUP_COLS; c++) {
                uint64_t flat = (uint64_t)row * cols + col0 + c;
                uint64_t bo = (flat / 32) * 34;
                go[(r >> 3) * 512 + c * 8 + (r & 7)] = A->data[bo + 2 + (flat % 32)];
            }
        }
    }
    // Scales with real Q8_0 block d values, normalized by row_scale
    uint8_t* scale_out = fpga_wt + Q8_TILE_WEIGHT_BYTES;
    for (int g = 0; g < Q8_NUM_GROUPS; g++) {
        int col0 = g * Q8_GROUP_COLS;
        uint8_t* gs = scale_out + g * Q8_GROUP_SCALE_BYTES;
        memset(gs, 0, Q8_GROUP_SCALE_BYTES);
        for (int r = 0; r < Q8_TILE_ROWS; r++) {
            int row = row0 + r;
            if (row >= (int)A->rows) continue;
            for (int h = 0; h < 2; h++) {
                uint64_t flat = (uint64_t)row * cols + col0 + h * 32;
                uint64_t bo = (flat / 32) * 34;
                float d_float = f16_to_f32(*(uint16_t*)(A->data + bo));
                float row_s = row_scale[r];
                float row_inv = (row_s < 1e-10f) ? 1.0f : (1.0f / row_s);
                float combined = d_float * row_inv;
                uint32_t uq = (uint32_t)(combined * 256.0f + 0.5f);
                if (uq > 65535) uq = 65535;
                int sc_addr = ((r>>3)<<4) | ((r&7)<<1) | h;
                *(uint16_t*)(gs + sc_addr*2) = (uint16_t)uq;
            }
        }
    }
}

// ===== Q5 Preprocessing =====
static int q5_preprocess_tile(const Tensor* A, int row0, uint32_t wt_addr, float* ri) {
    int cols = (int)A->cols, stride = cols / 32;
    int nrows = Q5_TILE_ROWS;
    if (row0 + nrows > (int)A->rows) nrows = (int)A->rows - row0;

    for (int bi = 0; bi < Q5_TILE_BLOCKS; bi++) {
        int group = bi / 28, blk_in_row = bi % 28;
        uint8_t* blk = (uint8_t*)ddr(wt_addr + bi * Q5_BLOCK_SIZE);
        for (int c = 0; c < 2; c++) {
            int mr = row0 + group + c * 2;
            if (mr < (int)A->rows) {
                uint64_t off = ((uint64_t)mr * stride + blk_in_row) * 22;
                memcpy(blk + c * 22, A->data + off, 22);
            } else memset(blk + c * 22, 0, 22);
        }
        memset(blk + 44, 0, 4);
    }

    /* Row normalization set to 1.0 (UQ8.8 = 0x0100) — NOT 32767/max_abs.
     * The Q5 core computes d_pre = f16_decode(d)·norm>>8 with d_pre S16
     * (±32767). With real model data, max_abs is small so ri = 32767/max_abs
     * is huge (50k-2M) → d_pre = f16_decode(d)·ri saturates at ±32767,
     * losing all weight variation (verified: bare-metal test uses ri=1.0
     * and passes; Linux with large ri gives near-zero results).
     * With ri=1.0: d_pre = f16_decode(d) = 256·d (no saturation for d<128),
     * raw = Σ 256·d·q5·act, and the 48-bit S24.8 accumulator handles the
     * full 896-element dot product. Correct scaling: y = raw·x_scale/256. */
    for (int r = 0; r < 4; r++) {
        ri[r] = 1.0f;
        *(uint16_t*)ddr(wt_addr + Q5_TILE_NORM_OFFSET + r*2) = 0x0100; /* UQ8.8 1.0 */
    }
    return nrows;
}

// ===== Quantize Float -> INT16 =====
static float quantize(const float* x, int16_t* xq, int n) {
    float m = 0; for (int j=0; j<n; j++) { float a=fabsf(x[j]); if (a>m) m=a; }
    float s = (m < 1e-10f) ? 1.0f : m/32767.0f;
    for (int j=0; j<n; j++) {
        float v = x[j]/s;
        xq[j] = (int16_t)(v + (v>=0?0.5f:-0.5f));
        if (v >= 32767.0f) xq[j] = 32767;
        else if (v <= -32768.0f) xq[j] = -32768;
    }
    return s;
}

// ===== Bit-exact golden models (see docs/maths.md, sim/golden_model.hpp) =====
// These replicate the exact integer/fixed-point arithmetic of the Q8/Q5 cores,
// so FPGA raw accumulators can be compared against a float-free reference.

// f16 -> S24.16 (1.0 -> 65536), bit-exact port of the Verilog f16_decode.
// SIGN-HANDLING (2026-08-14): llama.cpp quantizes Q5_0 with the negative-d
// trick (d = max/-16); ~50% of real blocks have negative d, and dequant
// (q-16)·d requires the sign. Bit 15 negates the magnitude (matches the fixed
// matmul_q5_0_core.v f16_decode; was previously sign-agnostic and flipped the
// sign of every weight in negative-d blocks).
static inline int32_t f16_decode_s2416(uint16_t f16) {
    int32_t exp  = (f16 >> 10) & 0x1F;
    int32_t mant = f16 & 0x3FF;
    int32_t mag;
    if (exp == 0 || exp == 31) mag = 0;
    else if (exp >= 9) mag = (1024 + mant) << (exp - 9);
    else mag = ((1024 + mant) + (1 << (8 - exp))) >> (9 - exp);
    return (f16 & 0x8000) ? -mag : mag;
}

// Q8 combined scale for weight (row, col): UQ8.8 = round(block_scale/row_scale*256).
// Must match q8_preprocess_tile's computation bit-for-bit.
static inline uint16_t q8_combined_scale(const Tensor* A, int row, int col,
                                         float row_scale) {
    uint64_t bo = ((uint64_t)row * A->cols + col) / 32 * 34;
    float d_float = f16_to_f32((uint16_t)A->data[bo] | ((uint16_t)A->data[bo+1] << 8));
    float row_inv = (row_scale < 1e-10f) ? 1.0f : (1.0f / row_scale);
    float combined = d_float * row_inv;
    uint32_t uq = (uint32_t)(combined * 256.0f + 0.5f);
    if (uq > 65535) uq = 65535;
    return (uint16_t)uq;
}

// Q8 golden raw accumulator for one tile: deq = (q8*sc)>>8, acc = SUM(deq*act).
static void q8_golden_raw(const Tensor* A, int row0, int nrows, int cols,
                          const int16_t* xq, const float* row_scale, int64_t* gold) {
    for (int i = 0; i < nrows; i++) {
        int row = row0 + i;
        int64_t acc = 0;
        for (int c = 0; c < cols; c++) {
            uint64_t flat = (uint64_t)row * cols + c;
            int8_t q8 = (int8_t)A->data[(flat / 32) * 34 + 2 + (flat % 32)];
            uint16_t sc = q8_combined_scale(A, row, c, row_scale[i]);
            int16_t deq = (int16_t)(((int32_t)q8 * (int32_t)sc) >> 8);
            acc += (int64_t)deq * (int64_t)xq[c];
        }
        gold[i] = acc;
    }
}

// Q5 d_pre = clamp((f16_decode(d) * norm) >> 8, S16).
static inline int16_t q5_d_pre_c(uint16_t d_f16, uint16_t norm) {
    int64_t shr = ((int64_t)f16_decode_s2416(d_f16) * (int64_t)norm) >> 8;
    if (shr > 32767) return 32767;
    if (shr < -32768) return -32768;
    return (int16_t)shr;
}

// Q5 golden raw accumulator for one tile (norm = 256 = 1.0, as q5_preprocess_tile sets).
static void q5_golden_raw(const Tensor* A, int row0, int nrows, int cols,
                          const int16_t* xq, int64_t* gold) {
    int stride = cols / 32;
    for (int i = 0; i < nrows; i++) {
        int row = row0 + i;
        int64_t acc = 0;
        for (int blk = 0; blk < stride; blk++) {
            uint8_t* b = A->data + ((uint64_t)row * stride + blk) * 22;
            uint16_t d_f16 = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
            int16_t d_pre = q5_d_pre_c(d_f16, 256);
            uint32_t qh = (uint32_t)b[2] | ((uint32_t)b[3] << 8) |
                          ((uint32_t)b[4] << 16) | ((uint32_t)b[5] << 24);
            for (int wi = 0; wi < 32; wi++) {
                int j = (wi < 16) ? wi : wi - 16;
                uint8_t ql = (wi < 16) ? (b[6+j] & 0xF) : (b[6+j] >> 4);
                int q5 = (((qh >> wi) & 1) << 4 | ql) - 16;
                int32_t dq = (int32_t)d_pre * (int32_t)q5;
                acc += (int64_t)dq * (int64_t)xq[blk * 32 + wi];
            }
        }
        gold[i] = acc;
    }
}

// ===== FPGA Matmuls =====
static int fpga_q8_tile(const Tensor* A, const uint8_t* wt, const int16_t* xq,
    float* y, int row0, float x_scale, int nrows, const float* row_scale)
{
    memcpy(ddr(FPGA_WEIGHT_REFMT), wt, Q8_TILE_STRIDE);
    memcpy(ddr(FPGA_WEIGHT_REFMT + Q8_TILE_STRIDE), xq, Q8_TILE_COLS*2);
    dcache_flush(ddr(FPGA_WEIGHT_REFMT), Q8_TILE_STRIDE + Q8_TILE_COLS*2);

    Descriptor* d = (Descriptor*)ddr(DESC_CHAIN_BASE);
    // act_bytes = ONE group's activations (64 int16 = 128 bytes), NOT the whole
    // 896-column tile. The FSM's LOAD_ACT runs per column-group at
    // act_addr + col_group*128, so it reads 128 bytes/group.
    desc_write(d, 0, FPGA_WEIGHT_REFMT, FPGA_WEIGHT_REFMT+Q8_TILE_STRIDE,
               FPGA_WEIGHT_REFMT+Q8_TILE_STRIDE+0x10000, DESC_Q8,
               Q8_NUM_GROUPS, 1, Q8_GROUP_COLS*2);

    if (chain_run(DESC_CHAIN_BASE, 1) < 0) return -1;

    // Golden raw reference (bit-exact fixed point, no float mismatch)
    int64_t gold[Q8_TILE_ROWS];
    q8_golden_raw(A, row0, nrows, (int)A->cols, xq, row_scale, gold);

    uint32_t* r = (uint32_t*)ddr(FPGA_WEIGHT_REFMT+Q8_TILE_STRIDE+0x10000);
    dcache_inval(r, nrows*8);

    /* ---- Q8 first-run warm-up (2026-08-14) ----
     * Root cause (hardware-verified): a Q8 descriptor processed immediately
     * after a Q5_0 descriptor produces corrupted results on rows 16-63 (acc
     * bank groups g=2..7). Re-running the identical descriptor produces
     * bit-exact correct results (verified via rerun + per-group isolation:
     * grp14 fpga == golden exactly). The corruption is stale FSM/core state
     * left by the Q5->Q8 transition that is cleared by the first Q8 compute.
     *
     * Mitigation: run every Q8 descriptor twice (warm-up + real). The first
     * run clears the stale state and its result is discarded; the second run
     * is bit-exact correct. This adds one extra chain_run per Q8 tile (2x
     * cost on Q8 matmuls: attn_v + logits), which is small vs the Q5/Q6/Q4
     * CPU-side work. */
    {
        /* The descriptor and DDR data are already set up above (weights/scales/
         * acts copied + flushed, descriptor written). Run once as warm-up to
         * clear any stale FSM state left by a preceding Q5_0 descriptor. */
        if (chain_run(DESC_CHAIN_BASE, 1) < 0) return -1;
    }

    if (chain_run(DESC_CHAIN_BASE, 1) < 0) return -1;
    dcache_inval(r, nrows*8);

    int64_t max_raw_diff = 0; int max_raw_row = -1;
    for (int i=0; i<nrows; i++) {
        uint64_t raw = (uint64_t)r[i*2] | ((uint64_t)r[i*2+1]<<32);
        if (raw & (1ULL<<47)) raw |= 0xFFFF000000000000ULL;
        int64_t raw_s = (int64_t)raw;
        int64_t diff = raw_s - gold[i]; if (diff < 0) diff = -diff;
        if (diff > max_raw_diff) { max_raw_diff = diff; max_raw_row = row0 + i; }
        if (g_compare && i == 0 && row0 == 0)
            printf("    [q8 r0=%d] acc=%lld gold=%lld  xs=%.6f  rs=%.6f\n",
                   row0, raw_s, (long long)gold[i], (double)x_scale,
                   (double)row_scale[i]);
        /* raw = dequant*act sum. scales include row_inv normalization.
         * dequant = (q8 * sc) >> 8 already removes the UQ8.8 factor,
         * so no /256 needed (unlike Q5 which accumulates at 256x).
         * raw_s is S48; cast to float directly (not int32) so large row sums
         * (|raw| > 2^31) do not wrap. */
        y[row0+i] += (float)raw_s * x_scale * row_scale[i];
    }
    if (g_compare && max_raw_diff > 0)
        printf("    [q8 RAWDIFF] max=%lld at row=%d\n",
               (long long)max_raw_diff, max_raw_row);
    return 0;
}

static int fpga_q5_tile(const Tensor* A, int row0, const int16_t* xq,
    float* y, float x_scale)
{
    float ri[4];
    uint32_t wt = FPGA_WEIGHT_REFMT;
    uint32_t res = 0x1F003000;  /* known-working address (proven by CPU_OP selftest) */
    int nrows = q5_preprocess_tile(A, row0, wt, ri);
    memcpy(ddr(wt+Q5_TILE_TOTAL), xq, (int)A->cols*2);
    dcache_flush(ddr(wt), Q5_TILE_TOTAL + (int)A->cols*2);

    Descriptor* d = (Descriptor*)ddr(DESC_CHAIN_BASE);
    desc_write(d, 0, wt, wt+Q5_TILE_TOTAL, res, DESC_Q5_0, 0, 1, (int)A->cols*2);

    if (chain_run(DESC_CHAIN_BASE, 1) < 0) return -1;

    // Golden raw reference (bit-exact fixed point, no float mismatch)
    int64_t gold[Q5_TILE_ROWS];
    q5_golden_raw(A, row0, nrows, (int)A->cols, xq, gold);

    uint32_t* r = (uint32_t*)ddr(res);
    dcache_inval(r, nrows*8);
    int64_t max_raw_diff = 0; int max_raw_row = -1;
    for (int i=0; i<nrows; i++) {
        uint64_t raw = (uint64_t)r[i*2] | ((uint64_t)r[i*2+1]<<32);
        if (raw & (1ULL<<47)) raw |= 0xFFFF000000000000ULL;
        int64_t raw_s = (int64_t)raw;
        int64_t diff = raw_s - gold[i]; if (diff < 0) diff = -diff;
        if (diff > max_raw_diff) { max_raw_diff = diff; max_raw_row = row0 + i; }
        if (g_compare && i == 0 && row0 == 0)
            printf("    [q5 r0=%d] acc=%lld gold=%lld  xs=%.6f\n",
                   row0, raw_s, (long long)gold[i], (double)x_scale);
        /* d_pre = f16_decode(d) = 65536·d (S24.16, ri=1.0). raw = Σ 65536·d·q5·act.
         * y = raw·x_scale/65536 = Σ d·q5·x.
         * raw_s is S48; cast to float directly (not int32) so large row sums
         * (|raw| > 2^31) do not wrap. */
        y[row0+i] += (float)raw_s * x_scale / 65536.0f;
    }
    if (g_compare && max_raw_diff > 0)
        printf("    [q5 RAWDIFF] max=%lld at row=%d\n",
               (long long)max_raw_diff, max_raw_row);
    return 0;
}

// ===== CPU Matmul =====
static void cpu_matmul(const Tensor* A, const float* x, float* y, int rows, int cols) {
    memset(y, 0, rows*4);
    for (int i=0; i<rows; i++) {
        float s=0;
        for (int j=0; j<cols; j++) s += dequant(A, (uint64_t)i*cols+j) * x[j];
        y[i] = s;
    }
}

/* one matmul via the FPGA core (or CPU for Q6_K/Q4_K/cols>2048) */
static void matmul_impl(const Tensor* A, const float* x, float* y, int rows, int cols) {
    int16_t xq[2048];
    if (cols > 2048) { cpu_matmul(A,x,y,rows,cols); return; }
    float xs = quantize(x, xq, cols);
    memset(y, 0, rows*4);

    if (A->type == TENSOR_Q8_0) {
        for (int r0=0; r0<rows; r0+=Q8_TILE_ROWS) {
            int nr = (rows-r0 < Q8_TILE_ROWS) ? rows-r0 : Q8_TILE_ROWS;
            float row_scale[Q8_TILE_ROWS];
            q8_preprocess_tile(A, r0, (uint8_t*)ddr(FPGA_WEIGHT_REFMT), row_scale);
            fpga_q8_tile(A, (uint8_t*)ddr(FPGA_WEIGHT_REFMT), xq, y, r0, xs, nr, row_scale);
        }
    } else if (A->type == TENSOR_Q5_0) {
        for (int r0=0; r0<rows; r0+=Q5_TILE_ROWS) fpga_q5_tile(A, r0, xq, y, xs);
    } else {
        cpu_matmul(A, x, y, rows, cols);
    }
}

static void matmul(const Tensor* A, const float* x, float* y, int rows, int cols) {
    if (g_compare) {
        /* run both paths into separate buffers and report the difference */
        static float y_cpu[16384];  /* enough for INTER_DIM 4864 */
        static float y_fpga[16384];
        matmul_impl(A, x, y_fpga, rows, cols);   /* FPGA path */
        cpu_matmul(A, x, y_cpu, rows, cols);     /* CPU reference */
        float md = 0;
        for (int i = 0; i < rows; i++) {
            float d = y_fpga[i] - y_cpu[i]; if (d < 0) d = -d;
            if (d > md) md = d;
        }
        printf("  cmp %-28s %5dx%-5d %-6s cpu[0]=%12.4f fpga[0]=%12.4f maxdiff=%12.5f\n",
               A->name, rows, cols,
               (cols > 2048 || (A->type!=TENSOR_Q8_0 && A->type!=TENSOR_Q5_0)) ? "cpu" : "fpga",
               y_cpu[0], y_fpga[0], md);
        memcpy(y, y_fpga, rows*4);   /* keep FPGA result for downstream */
        return;
    }
    if (g_use_cpu) {
        /* --cpu: pure-CPU fallback for A/B comparison */
        cpu_matmul(A, x, y, rows, cols);
        return;
    }
    matmul_impl(A, x, y, rows, cols);
}

// ===== CPU Ops =====
static void rms_norm(float* o, const float* x, int n, const Tensor* t) {
    float ss=0; for (int i=0;i<n;i++) ss+=x[i]*x[i];
    float s = 1.0f/sqrtf(ss/n + 1e-6f);
    float* w = (float*)t->data;
    for (int i=0;i<n;i++) o[i]=x[i]*w[i]*s;
}

static void silu(float* y, const float* x, int n) {
    for (int i=0;i<n;i++) y[i]=x[i]/(1.0f+expf(-x[i]));
}

static void rope(float* q, float* k, int pos) {
    float base = 1000000.0f;
    for (int h=0;h<NUM_HEADS;h++) for (int d=0;d<HEAD_DIM;d+=2) {
        float th = 1.0f/powf(base, (float)d/HEAD_DIM);
        float f = pos*th, c=cosf(f), s=sinf(f);
        int idx = h*HEAD_DIM+d;
        float q0=q[idx], q1=q[idx+1]; q[idx]=q0*c-q1*s; q[idx+1]=q0*s+q1*c;
    }
    for (int h=0;h<NUM_KV_HEADS;h++) for (int d=0;d<HEAD_DIM;d+=2) {
        float th = 1.0f/powf(base, (float)d/HEAD_DIM);
        float f = pos*th, c=cosf(f), s=sinf(f);
        int idx = h*HEAD_DIM+d;
        float k0=k[idx], k1=k[idx+1]; k[idx]=k0*c-k1*s; k[idx+1]=k0*s+k1*c;
    }
}

static void attention(float* ctx, float* qv, int layer, int pos, int seqlen) {
    int qpk = NUM_HEADS / NUM_KV_HEADS;
    memset(ctx, 0, HIDDEN_DIM*4);

    for (int qh=0; qh<NUM_HEADS; qh++) {
        int kv = qh/qpk;
        float* qd = qv + qh*HEAD_DIM;
        float* ch = ctx + qh*HEAD_DIM;
        float scores[MAX_SEQ_LEN];
        float ms = -1e10f;
        for (int p=0; p<=pos; p++) {
            float* kc = g_akc[layer][p] + kv*HEAD_DIM;
            float s = 0;
            for (int d=0; d<HEAD_DIM; d++) s += qd[d] * kc[d];
            scores[p] = s / sqrtf(HEAD_DIM);
            if (scores[p] > ms) ms = scores[p];
        }
        float se = 0;
        for (int p=0; p<=pos; p++) se += expf(scores[p] - ms);
        float ls = logf(se) + ms;
        for (int p=0; p<=pos; p++) {
            float* vc = g_avc[layer][p] + kv*HEAD_DIM;
            float w = expf(scores[p] - ls);
            for (int d=0; d<HEAD_DIM; d++) ch[d] += w * vc[d];
        }
    }
}

// ===== Name Formatting =====
static int fmt_name(char* buf, int layer, const char* suffix) {
    return snprintf(buf, 128, "blk.%d.%s", layer, suffix);
}

// ===== Forward Layer =====
static void forward_layer(float* hidden, int layer, int pos) {
    char name[128];
    Tensor* t;
    float* scratch = g_scratch;
    float* safe = hidden; /* used for original_hidden (small, OK on stack) */
    float orig_hid[HIDDEN_DIM];
    memcpy(orig_hid, hidden, HIDDEN_DIM*4);
    /* Distinct scratch regions (g_scratch is 65536 floats = 256 KB).
     * WARNING: the original code aliased norm_out/qv/ctx/attn_out/fnorm/gate
     * all at scratch[0], so each stage overwrote the previous input/output —
     * that collapsed the hidden state and every logit became equal (token 0). */
    float* norm_out = scratch;              /* 0        .. 896   */
    float* qv       = scratch + 1024;       /* 1024     .. 1920  */
    float* kv       = scratch + 2048;       /* 2048     .. 2176  */
    float* vv       = scratch + 2304;       /* 2304     .. 2432  */
    float* ctx      = scratch + 2560;       /* 2560     .. 3456  */
    float* attn_out = scratch + 3584;       /* 3584     .. 4480  */
    float* fnorm    = scratch + 4608;       /* 4608     .. 5504  */
    float* gate     = scratch + 5632;       /* 5632     .. 10496 */
    float* up       = scratch + 10752;      /* 10752    .. 15616 */
    float* fout     = scratch + 15872;      /* 15872    .. 16768 */

    /* Attn norm */
    if ((t = get_tensor((fmt_name(name,layer,"attn_norm.weight"),name))))
        rms_norm(norm_out, orig_hid, HIDDEN_DIM, t);
    else memcpy(norm_out, orig_hid, HIDDEN_DIM*4);

    /* Q */
    if ((t = get_tensor((fmt_name(name,layer,"attn_q.weight"),name))))
        matmul(t, norm_out, qv, HIDDEN_DIM, HIDDEN_DIM);
    if ((t = get_tensor((fmt_name(name,layer,"attn_q.bias"),name)))) {
        float* b = (float*)t->data; for (int i=0;i<HIDDEN_DIM;i++) qv[i] += b[i];
    }
    /* K */
    if ((t = get_tensor((fmt_name(name,layer,"attn_k.weight"),name))))
        matmul(t, norm_out, kv, K_DIM, HIDDEN_DIM);
    if ((t = get_tensor((fmt_name(name,layer,"attn_k.bias"),name)))) {
        float* b = (float*)t->data; for (int i=0;i<K_DIM;i++) kv[i] += b[i];
    }
    /* V */
    if ((t = get_tensor((fmt_name(name,layer,"attn_v.weight"),name))))
        matmul(t, norm_out, vv, V_DIM, HIDDEN_DIM);
    if ((t = get_tensor((fmt_name(name,layer,"attn_v.bias"),name)))) {
        float* b = (float*)t->data; for (int i=0;i<V_DIM;i++) vv[i] += b[i];
    }

    rope(qv, kv, pos);
    memcpy(g_akc[layer][pos], kv, K_DIM*4);
    memcpy(g_avc[layer][pos], vv, V_DIM*4);
    attention(ctx, qv, layer, pos, pos+1);

    if ((t = get_tensor((fmt_name(name,layer,"attn_output.weight"),name))))
        matmul(t, ctx, attn_out, HIDDEN_DIM, HIDDEN_DIM);
    for (int i=0;i<HIDDEN_DIM;i++) hidden[i] = orig_hid[i] + attn_out[i];

    /* FFN */
    if ((t = get_tensor((fmt_name(name,layer,"ffn_norm.weight"),name))))
        rms_norm(fnorm, hidden, HIDDEN_DIM, t);
    else memcpy(fnorm, hidden, HIDDEN_DIM*4);

    if ((t = get_tensor((fmt_name(name,layer,"ffn_gate.weight"),name))))
        matmul(t, fnorm, gate, INTER_DIM, HIDDEN_DIM);
    if ((t = get_tensor((fmt_name(name,layer,"ffn_up.weight"),name))))
        matmul(t, fnorm, up, INTER_DIM, HIDDEN_DIM);

    silu(gate, gate, INTER_DIM);
    for (int i=0;i<INTER_DIM;i++) gate[i] *= up[i];

    if ((t = get_tensor((fmt_name(name,layer,"ffn_down.weight"),name))))
        matmul(t, gate, fout, HIDDEN_DIM, INTER_DIM);
    for (int i=0;i<HIDDEN_DIM;i++) hidden[i] += fout[i];
}

// ===== Inference Loop =====
/* --trace: run FPGA + CPU paths per layer with separate KV caches, print
 * a one-line hidden-state summary per layer so we can see where they diverge. */
static int run_trace(const int* prompt, int np) {
    memset(g_kcache, 0, sizeof(g_kcache));
    memset(g_vcache, 0, sizeof(g_vcache));
    memset(g_kcache_cpu, 0, sizeof(g_kcache_cpu));
    memset(g_vcache_cpu, 0, sizeof(g_vcache_cpu));

    Tensor* emb = get_tensor("token_embd.weight");
    if (!emb) { fprintf(stderr,"No token_embd.weight\n"); return -1; }
    float hidden[HIDDEN_DIM];      /* FPGA path running state */
    float h_cpu[HIDDEN_DIM];       /* CPU  path running state */
    for (int i=0;i<HIDDEN_DIM;i++)
        hidden[i] = h_cpu[i] = dequant(emb, (uint64_t)prompt[0]*HIDDEN_DIM+i);

    printf("Trace: per-layer hidden state, FPGA vs CPU (token %d)\n", prompt[0]);
    printf("      %-4s %-22s %-22s %s\n", "L", "fpga", "cpu", "maxdiff");
    for (int l=0; l<NUM_LAYERS; l++) {
        g_use_cpu = 0; g_akc = g_kcache;      g_avc = g_vcache;
        forward_layer(hidden, l, 0);
        g_use_cpu = 1; g_akc = g_kcache_cpu;  g_avc = g_vcache_cpu;
        forward_layer(h_cpu, l, 0);

        double nf=0, nc=0; float md=0;
        for (int i=0;i<HIDDEN_DIM;i++) {
            nf += (double)hidden[i]*hidden[i];
            nc += (double)h_cpu[i]*h_cpu[i];
            float d = hidden[i]-h_cpu[i]; if (d<0) d=-d; if (d>md) md=d;
        }
        printf("  %-4d n=%9.2f h0=%+9.5f n=%9.2f h0=%+9.5f %12.5f\n",
               l, sqrt(nf), hidden[0], sqrt(nc), h_cpu[0], md);
        fflush(stdout);
    }
    return 0;
}

static int run_inference(const int* prompt, int np) {
    memset(g_kcache, 0, sizeof(g_kcache));
    memset(g_vcache, 0, sizeof(g_vcache));

    float hidden[HIDDEN_DIM];
    float logits[VOCAB_SIZE];

    printf("Processing %d prompt tokens...\n", np);
    for (int t=0; t<np; t++) {
        Tensor* emb = get_tensor("token_embd.weight");
        if (!emb) { memset(hidden,0,sizeof(hidden)); }
        else for (int i=0;i<HIDDEN_DIM;i++)
            hidden[i] = dequant(emb, (uint64_t)prompt[t]*HIDDEN_DIM+i);
        for (int l=0; l<NUM_LAYERS; l++) forward_layer(hidden, l, t);
    }

    int ntokens = 0;
    int output[MAX_SEQ_LEN*2];

    for (int gen=0; gen<10; gen++) {
        int pos = np + gen;
        if (pos >= MAX_SEQ_LEN) break;

        /* Logits from current hidden state (prompt's last token on gen 0) */
        Tensor* norm = get_tensor("output_norm.weight");
        float norm_hid[HIDDEN_DIM];
        if (norm) rms_norm(norm_hid, hidden, HIDDEN_DIM, norm);
        else memcpy(norm_hid, hidden, sizeof(norm_hid));

        Tensor* emb_w = get_tensor("token_embd.weight");
        if (!emb_w) { fprintf(stderr,"No token_embd.weight\n"); break; }

        int best=0; float best_v=-1e10f;
        for (int i=0; i<VOCAB_SIZE; i++) {
            float s=0;
            for (int j=0; j<HIDDEN_DIM; j++)
                s += dequant(emb_w, (uint64_t)i*HIDDEN_DIM+j) * norm_hid[j];
            logits[i] = s;
            if (s > best_v) { best_v=s; best=i; }
        }

        output[ntokens++] = best;
        printf("  token %d: %d\n", gen, best);
        if (best == 151643) break; /* EOS */

        /* Embed the sampled token and forward at pos for the next iteration */
        Tensor* emb = get_tensor("token_embd.weight");
        if (emb) for (int i=0;i<HIDDEN_DIM;i++)
            hidden[i] = dequant(emb, (uint64_t)best*HIDDEN_DIM+i);
        else memset(hidden, 0, sizeof(hidden));

        for (int l=0; l<NUM_LAYERS; l++) forward_layer(hidden, l, pos);
    }
    return ntokens;
}

// ===== Main =====
int main(int argc, char** argv) {
    printf("T-MAC Linux Inference\n");

    /* --cpu flag: run every matmul on the CPU (no FPGA) for A/B comparison */
    /* --compare flag: run both CPU+FPGA per matmul and print the diff */
    /* --trace flag: per-layer hidden-state comparison (FPGA vs CPU) */
    /* --selftest: minimal CPU_OP DDR copy (isolates PL DDR path from compute) */
    const char* model_path = NULL;
    int prompt[256];
    int np = 0;
    int do_selftest = 0;
    int do_trace = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--cpu") == 0) {
            g_use_cpu = 1;
        } else if (strcmp(argv[i], "--compare") == 0) {
            g_compare = 1;
        } else if (strcmp(argv[i], "--trace") == 0) {
            do_trace = 1;
        } else if (strcmp(argv[i], "--selftest") == 0) {
            do_selftest = 1;
        } else if (model_path == NULL) {
            model_path = argv[i];          /* first non-flag arg = model */
        } else if (np < 256) {
            prompt[np++] = atoi(argv[i]);  /* optional prompt token ids */
        }
    }
    if (np == 0) prompt[np++] = 151646;    /* default: bare prompt token */
    if (g_use_cpu) printf("CPU-only mode (--cpu): FPGA matmuls forced to CPU\n");
    if (g_compare) printf("Compare mode (--compare): CPU vs FPGA per matmul\n");
    if (do_trace) printf("Trace mode (--trace): per-layer hidden-state FPGA vs CPU\n");
    printf("Prompt tokens: %d (", np);
    for (int i = 0; i < np; i++) printf("%s%d", i ? " " : "", prompt[i]);
    printf(")\n");

    g_scratch = (float*)malloc(0x40000); /* 256KB scratch */
    if (!g_scratch) { fprintf(stderr,"No memory\n"); return 1; }

    if (g_use_cpu) {
        printf("CPU-only mode: skipping FPGA init\n");
    } else {
        g_gp0 = map_mem(IP_BASE, 0x10000);
        if (!g_gp0) return 1;
        if (map_ddr() < 0) return 1;

        /* Init FPGA CPU_OP registers */
        reg_write(REG_Q8_NUM_GROUPS, 14);
        reg_write(REG_CHAIN_CTRL, CHAIN_CTRL_INTR_ENABLE);
        reg_write(REG_GIE, 1);
        reg_write(REG_ISR, 1);
        printf("FPGA initialized: CHAIN_CTRL=0x%08x\n", reg_read(REG_CHAIN_CTRL));

        /* Enable AFI0 HP0 slave port (both read and write channels). SD boot
         * does not configure AFI — only bare-metal JTAG scripts do. Without
         * this the PL's HP0 port cannot reach DDR at all.
         * NOTE: AFI0 registers are at SLCR offset 0x8000-0x8008 — the SLCR
         * map must cover 0x10000 (64KB) to reach them (not just 0x1000!). */
        {
            volatile uint32_t* slcr = map_mem(0xF8000000, 0x10000);
            if (!slcr) { fprintf(stderr,"Cannot map SLCR\n"); return 1; }
            slcr[0x0008/4] = 0x0000DF0D;      /* unlock SLCR (SLCR_UNLOCK) */
            slcr[0x8000/4] = 0x00000005;      /* AFI0_CTRL: enable + SLVERR */
            slcr[0x8008/4] = 0x00000001;      /* AFI0_WRCHAN: write enable */
            slcr[0x0004/4] = 0x0000767B;      /* lock SLCR (SLCR_LOCK) */
            printf("AFI0 enabled (CTRL=0x%08lx WRCHAN=0x%08lx)\n",
                   (unsigned long)slcr[0x8000/4], (unsigned long)slcr[0x8008/4]);
        }

        if (do_selftest) {
            /* passthrough mode: clear intr-enable so CPU_OP does plain copy */
            reg_write(REG_CHAIN_CTRL, 0);
            uint32_t act  = 0x1F002000;
            uint32_t res  = 0x1F003000;
            /* write 8 known words to act */
            for (int i = 0; i < 8; i++) *(uint32_t*)ddr(act + i*4) = 0x11110000 + i;
            /* sentinel in result */
            for (int i = 0; i < 8; i++) *(uint32_t*)ddr(res + i*4) = 0xDEADBEEF;
            dcache_flush(ddr(act), 32);
            dcache_flush(ddr(res), 32);

            Descriptor* d = (Descriptor*)ddr(DESC_CHAIN_BASE);
            desc_write(d, 0, 0, act, res, DESC_CPU_OP, 0, 1, 32);
            if (chain_run(DESC_CHAIN_BASE, 1) < 0) return 1;
            dcache_inval(ddr(res), 32);
            int ok = 1;
            printf("CPU_OP selftest: act[0..7] -> res[0..7]\n");
            for (int i = 0; i < 8; i++) {
                uint32_t a = *(uint32_t*)ddr(act + i*4);
                uint32_t r = *(uint32_t*)ddr(res + i*4);
                printf("  [%d] act=0x%08x res=0x%08x %s\n", i, a, r,
                       (a == r) ? "OK" : "** MISMATCH **");
                if (a != r) ok = 0;
            }
            printf("CPU_OP selftest: %s\n", ok ? "PASS" : "FAIL");
            return ok ? 0 : 1;
        }
    }

    if (!model_path) { fprintf(stderr,"Usage: %s [--cpu|--compare|--trace|--selftest] model.tmac [token ...]\n", argv[0]); return 1; }
    if (load_model(model_path) < 0) return 1;

    if (do_trace) return run_trace(prompt, np);

    int tokens = run_inference(prompt, np);
    printf("\nGenerated %d tokens\n", tokens);
    return 0;
}
