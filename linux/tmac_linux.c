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

/* F32 scratch for forward_layer (avoids stack overflow) */
static float* g_scratch = NULL;

/* /dev/mem mappings */
static volatile uint32_t* g_gp0 = NULL;  /* 0x43C00000 */
static float g_kcache[NUM_LAYERS][MAX_SEQ_LEN][K_DIM];
static float g_vcache[NUM_LAYERS][MAX_SEQ_LEN][V_DIM];

// ===== /dev/mem Access =====
static volatile uint32_t* map_mem(uint32_t base, size_t size) {
    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) { perror("/dev/mem"); return NULL; }
    volatile uint32_t* p = mmap(NULL, size, PROT_READ|PROT_WRITE,
                                MAP_SHARED, fd, base);
    close(fd);
    if (p == MAP_FAILED) { perror("mmap"); return NULL; }
    return p;
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
    if (exp == 0) return (mant==0)?0.0f:((sign?-1:1)*mant/1024.0f*0.00006103515625f);
    if (exp==31) return mant?0.0f:(sign?-INFINITY:INFINITY);
    return (sign?-1:1)*ldexpf(1.0f+mant/1024.0f, (int)exp-15);
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
        return d_val * (float)((((qh>>(idx%32))&1)<<4)|ql - 16);
    }
    if (t->type == TENSOR_Q6_K) {
        uint64_t bo = (idx/256)*210;
        float super = f16_to_f32(*(uint16_t*)(d+bo+208));
        uint64_t wi = idx%256;
        int half=wi/128, pos=wi%128, l=pos%32, sub=pos/32;
        int lo = half*64 + l + (sub%2)*32;
        int ql = (d[lo]>>((sub<2)?0:4)) & 0xF;
        int qh = (d[128+half*32+l]>>(sub*2)) & 3;
        return super * (float)(int8_t)d[192+half*8+(l/16)+sub*2] * (float)((qh<<4)|ql-32);
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
    reg_write(REG_DESC_BASE, base);
    reg_write(REG_DESC_TAIL, 1);
    __sync_synchronize();
    reg_write(REG_START, 1);
    uint32_t timeout = CHAIN_TIMEOUT;
    while (timeout--) {
        if (!(reg_read(REG_STATUS) & 0x8000)) return 0;
        for (volatile int i = 0; i < 10; i++);
    }
    fprintf(stderr, "TIMEOUT: DEBUG=0x%08x Q8=0x%08x\n",
            reg_read(REG_DEBUG), reg_read(REG_Q8_DEBUG));
    return -1;
}

// ===== Q8 Preprocessing =====
static void q8_preprocess_tile(const Tensor* A, int row0, uint8_t* fpga_wt) {
    int cols = (int)A->cols;
    for (int g = 0; g < Q8_NUM_GROUPS; g++) {
        int col0 = g * Q8_GROUP_COLS;
        uint8_t* go = fpga_wt + g * Q8_GROUP_BYTES;
        for (int r = 0; r < Q8_TILE_ROWS; r++) {
            int row = row0 + r;
            if (row >= (int)A->rows) break;
            for (int c = 0; c < Q8_GROUP_COLS; c++) {
                uint64_t flat = (uint64_t)row * cols + col0 + c;
                uint64_t bo = (flat / 32) * 34;
                go[c * Q8_TILE_ROWS + r] = A->data[bo + 2 + (flat % 32)];
            }
        }
    }
    // Scales with real Q8_0 block d values
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
                uint32_t uq = (uint32_t)(d_float * 256.0f + 0.5f);
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
        uint8_t* blk = (uint8_t*)(uintptr_t)(wt_addr + bi * Q5_BLOCK_SIZE);
        for (int c = 0; c < 2; c++) {
            int mr = row0 + group + c * 2;
            if (mr < (int)A->rows) {
                uint64_t off = ((uint64_t)mr * stride + blk_in_row) * 22;
                memcpy(blk + c * 22, A->data + off, 22);
            } else memset(blk + c * 22, 0, 22);
        }
        memset(blk + 44, 0, 4);
    }

    for (int r = 0; r < nrows; r++) {
        float m = 0; int row = row0 + r;
        for (int bi = 0; bi < stride; bi++) {
            uint64_t bo = ((uint64_t)row * stride + bi) * 22;
            float d = f16_to_f32(*(uint16_t*)(A->data + bo));
            uint32_t qh = *(uint32_t*)(A->data + bo + 2);
            for (int wi = 0; wi < 32; wi++) {
                uint64_t j = wi < 16 ? wi : wi - 16;
                int q5 = (((qh>>wi)&1)<<4) | ((A->data[bo+6+j]>>((wi<16)?0:4))&0xF);
                q5 -= 16;
                float a = fabsf(d * q5);
                if (a > m) m = a;
            }
        }
        ri[r] = (m < 1e-10f) ? 1.0f : 32767.0f / m;
    }
    for (int r = 0; r < 4; r++) {
        *(uint16_t*)(uintptr_t)(wt_addr + Q5_TILE_NORM_OFFSET + r*2) =
            (uint16_t)(ri[r < nrows ? r : 0] * 256.0f + 0.5f);
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

// ===== FPGA Matmuls =====
static int fpga_q8_tile(const uint8_t* wt, const int16_t* xq, float* y,
    int row0, float x_scale, int nrows)
{
    memcpy((void*)(uintptr_t)FPGA_WEIGHT_REFMT, wt, Q8_TILE_STRIDE);
    memcpy((void*)(uintptr_t)FPGA_WEIGHT_REFMT, xq, Q8_TILE_COLS*2);

    Descriptor* d = (Descriptor*)(uintptr_t)DESC_CHAIN_BASE;
    desc_write(d, 0, FPGA_WEIGHT_REFMT, FPGA_WEIGHT_REFMT+Q8_TILE_STRIDE,
               FPGA_WEIGHT_REFMT+Q8_TILE_STRIDE+0x10000, DESC_Q8,
               Q8_NUM_GROUPS, 1, Q8_TILE_COLS*2);

    if (chain_run(DESC_CHAIN_BASE, 1) < 0) return -1;

    uint32_t* r = (uint32_t*)(uintptr_t)(FPGA_WEIGHT_REFMT+Q8_TILE_STRIDE+0x10000);
    for (int i=0; i<nrows; i++) {
        uint64_t raw = (uint64_t)r[i*2] | ((uint64_t)r[i*2+1]<<32);
        if (raw & (1ULL<<47)) raw |= 0xFFFF000000000000ULL;
        y[row0+i] += (float)(int32_t)(int64_t)raw * x_scale;
    }
    return 0;
}

static int fpga_q5_tile(const Tensor* A, int row0, const int16_t* xq,
    float* y, float x_scale)
{
    float ri[4];
    uint32_t wt = FPGA_WEIGHT_REFMT;
    int nrows = q5_preprocess_tile(A, row0, wt, ri);
    memcpy((void*)(uintptr_t)(wt+Q5_TILE_TOTAL), xq, (int)A->cols*2);

    Descriptor* d = (Descriptor*)(uintptr_t)DESC_CHAIN_BASE;
    desc_write(d, 0, wt, wt+Q5_TILE_TOTAL, wt+Q5_TILE_TOTAL+0x800,
               DESC_Q5_0, 0, 1, (int)A->cols*2);

    if (chain_run(DESC_CHAIN_BASE, 1) < 0) return -1;

    uint32_t* r = (uint32_t*)(uintptr_t)(wt+Q5_TILE_TOTAL+0x800);
    for (int i=0; i<nrows; i++) {
        uint64_t raw = (uint64_t)r[i*2] | ((uint64_t)r[i*2+1]<<32);
        if (raw & (1ULL<<47)) raw |= 0xFFFF000000000000ULL;
        y[row0+i] += (float)(int32_t)(int64_t)raw * x_scale / ri[i];
    }
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

static void matmul(const Tensor* A, const float* x, float* y, int rows, int cols) {
    int16_t xq[2048];
    if (cols > 2048) { cpu_matmul(A,x,y,rows,cols); return; }
    float xs = quantize(x, xq, cols);
    memset(y, 0, rows*4);

    if (A->type == TENSOR_Q8_0) {
        for (int r0=0; r0<rows; r0+=Q8_TILE_ROWS) {
            int nr = (rows-r0 < Q8_TILE_ROWS) ? rows-r0 : Q8_TILE_ROWS;
            q8_preprocess_tile(A, r0, (uint8_t*)(uintptr_t)FPGA_WEIGHT_REFMT);
            fpga_q8_tile((uint8_t*)(uintptr_t)FPGA_WEIGHT_REFMT, xq, y, r0, xs, nr);
        }
    } else if (A->type == TENSOR_Q5_0) {
        for (int r0=0; r0<rows; r0+=Q5_TILE_ROWS) fpga_q5_tile(A, r0, xq, y, xs);
    } else {
        cpu_matmul(A, x, y, rows, cols);
    }
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
            float* kc = g_kcache[layer][p] + kv*HEAD_DIM;
            float s = 0;
            for (int d=0; d<HEAD_DIM; d++) s += qd[d] * kc[d];
            scores[p] = s / sqrtf(HEAD_DIM);
            if (scores[p] > ms) ms = scores[p];
        }
        float se = 0;
        for (int p=0; p<=pos; p++) se += expf(scores[p] - ms);
        float ls = logf(se) + ms;
        for (int p=0; p<=pos; p++) {
            float* vc = g_vcache[layer][p] + kv*HEAD_DIM;
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

    float* norm_out = scratch;
    float* qv = scratch;
    float* kv = scratch + 896;
    float* vv = scratch + 1024;
    float* ctx = scratch;
    float* attn_out = scratch;
    float* fnorm = scratch;
    float* gate = scratch;
    float* up = scratch + 5000;
    float* fout = scratch;
    (void)fout;

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
    memcpy(g_kcache[layer][pos], kv, K_DIM*4);
    memcpy(g_vcache[layer][pos], vv, V_DIM*4);
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
        matmul(t, gate, scratch, HIDDEN_DIM, INTER_DIM); // reuse scratch as fout
    for (int i=0;i<HIDDEN_DIM;i++) hidden[i] += scratch[i];
}

// ===== Inference Loop =====
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

    int seqlen = np;
    int ntokens = 0;
    int output[MAX_SEQ_LEN*2];
    int token = 151646; /* BOS */

    for (int gen=0; gen<10; gen++) {
        int pos = np + gen;
        if (pos >= MAX_SEQ_LEN) break;

        Tensor* emb = get_tensor("token_embd.weight");
        if (emb) for (int i=0;i<HIDDEN_DIM;i++)
            hidden[i] = dequant(emb, (uint64_t)token*HIDDEN_DIM+i);
        else memset(hidden, 0, sizeof(hidden));

        for (int l=0; l<NUM_LAYERS; l++) forward_layer(hidden, l, pos);

        /* Logits */
        Tensor* norm = get_tensor("output_norm.weight");
        float norm_hid[HIDDEN_DIM];
        if (norm) rms_norm(norm_hid, hidden, HIDDEN_DIM, norm);
        else memcpy(norm_hid, hidden, sizeof(norm_hid));

        Tensor* emb_w = get_tensor("token_embd.weight");
        if (!emb_w) { fprintf(stderr,"No token_embd.weight\n"); break; }

        /* Compute logits for first few tokens only (sampling) */
        int best=0; float best_v=-1e10f;
        for (int i=0; i<VOCAB_SIZE; i++) {
            float s=0;
            for (int j=0; j<HIDDEN_DIM; j++)
                s += dequant(emb_w, (uint64_t)i*HIDDEN_DIM+j) * norm_hid[j];
            logits[i] = s;
            if (s > best_v) { best_v=s; best=i; }
        }
        token = best;
        output[ntokens++] = token;
        printf("  token %d: %d\n", gen, token);
        if (token == 151643) break; /* EOS */
    }
    return ntokens;
}

// ===== Main =====
int main(int argc, char** argv) {
    printf("T-MAC Linux Inference\n");

    g_gp0 = map_mem(IP_BASE, 0x10000);
    if (!g_gp0) return 1;

    g_scratch = (float*)malloc(0x40000); /* 256KB scratch */
    if (!g_scratch) { fprintf(stderr,"No memory\n"); return 1; }

    /* Init FPGA CPU_OP registers */
    reg_write(REG_Q8_NUM_GROUPS, 14);
    reg_write(REG_CHAIN_CTRL, CHAIN_CTRL_INTR_ENABLE);
    reg_write(REG_GIE, 1);
    reg_write(REG_ISR, 1);
    printf("FPGA initialized: CHAIN_CTRL=0x%08x\n", reg_read(REG_CHAIN_CTRL));

    if (argc < 2) { fprintf(stderr,"Usage: %s model.tmac [token]\n", argv[0]); return 1; }
    if (load_model(argv[1]) < 0) return 1;

    int prompt_token = (argc > 2) ? atoi(argv[2]) : 151646;
    int tokens = run_inference(&prompt_token, 1);
    printf("\nGenerated %d tokens\n", tokens);
    return 0;
}
