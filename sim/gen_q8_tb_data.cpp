// Generate $readmemh files for the tb_hw_fsm_comprehensive.v real-data Q8 test.
// Extracts blk.0.attn_v tile 0 (rows 0-63), lays out weights/scales/acts in the
// EXACT DDR layout q8_preprocess_tile produces (row-group-major weights, permuted
// scales), computes the golden raw accumulators via golden::q8_tile_golden.
//
// Build: g++ -O2 -std=c++17 -I . -I sim -I gguf gen_q8_tb_data.cpp -o /tmp/gen_q8tb
// Run:   /tmp/gen_q8tb models/model.tmac blk.0.attn_v.weight 0
// Emits: verilog/q8_real.mem    (DDR words, contiguous from word 0x60000 = addr 0x00300000)
//        verilog/q8_golden.mem  (64 x int64 hex)
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>

#include "sim/golden_model.hpp"

struct Tensor { std::string name; uint64_t rows, cols; uint32_t type, n_bytes; std::vector<uint8_t> data; };
static uint64_t rq(FILE* f){uint64_t v;if(fread(&v,8,1,f)!=1){fprintf(stderr,"EOF\n");exit(1);}return v;}
static uint32_t ri(FILE* f){uint32_t v;if(fread(&v,4,1,f)!=1){fprintf(stderr,"EOF\n");exit(1);}return v;}
static std::vector<Tensor> load(const char* path){
    FILE* f=fopen(path,"rb"); if(!f){perror(path);exit(1);}
    char m[4]; fread(m,1,4,f);
    if(memcmp(m,"TMAC",4)){fprintf(stderr,"bad magic\n");exit(1);}
    uint64_t n=rq(f); std::vector<Tensor> ts;
    for(uint64_t i=0;i<n;i++){Tensor t; uint64_t nl=rq(f); t.name.resize(nl); fread(&t.name[0],1,nl,f);
        t.rows=rq(f); t.cols=rq(f); t.type=ri(f); t.n_bytes=rq(f);
        t.data.resize(t.n_bytes); fread(t.data.data(),1,t.n_bytes,f); ts.push_back(t);}
    fclose(f); return ts;
}
static inline float f16_to_f32(uint16_t v){
    uint32_t s=(v>>15)&1,e=(v>>10)&0x1F,m=v&0x3FF;
    if(e==0)return (m==0)?0.0f:(s?-1.0f:1.0f)*((float)m/1024.0f)*0.00006103515625f;
    if(e==31)return NAN;
    return (s?-1.0f:1.0f)*(1.0f+(float)m/1024.0f)*powf(2.0f,(float)((int)e-15));
}

#define Q8_TILE_ROWS 64
#define Q8_GROUP_COLS 64
#define Q8_NUM_GROUPS 14
#define Q8_GROUP_BYTES 4096
#define Q8_GROUP_SCALE_BYTES 256
#define Q8_TILE_WEIGHT_BYTES (Q8_NUM_GROUPS*Q8_GROUP_BYTES)
#define Q8_TILE_SCALE_BYTES  (Q8_NUM_GROUPS*Q8_GROUP_SCALE_BYTES)

int main(int argc,char**argv){
    if(argc<4){fprintf(stderr,"usage: %s model.tmac tensor row0\n",argv[0]);return 1;}
    auto ts=load(argv[1]);
    const Tensor* A=nullptr; for(auto&t:ts)if(t.name==argv[2]){A=&t;break;}
    if(!A){fprintf(stderr,"not found\n");return 1;}
    int row0=atoi(argv[3]);
    uint64_t cols=A->cols;
    printf("tensor %s %llux%llu type=%u row0=%d\n",A->name.c_str(),(unsigned long long)A->rows,(unsigned long long)cols,A->type,row0);

    // ---- row_scale (match q8_preprocess_tile) ----
    float row_scale[Q8_TILE_ROWS];
    for(int r=0;r<Q8_TILE_ROWS;r++){
        int row=row0+r; float max_abs=0;
        if(row>=(int)A->rows){row_scale[r]=1.0f;continue;}
        for(uint64_t j=0;j<cols;j++){
            uint64_t flat=(uint64_t)row*cols+j; uint64_t bo=(flat/32)*34;
            int8_t v=(int8_t)A->data[bo+2+(flat%32)];
            float d=f16_to_f32((uint16_t)A->data[bo]|((uint16_t)A->data[bo+1]<<8));
            float a=fabsf((float)v*d); if(a>max_abs)max_abs=a;
        }
        row_scale[r]=(max_abs<1e-10f)?1.0f:max_abs/32767.0f;
    }

    // ---- weights (row-group-major per group) + scales (permuted per group) ----
    std::vector<uint8_t> wt(Q8_TILE_WEIGHT_BYTES + Q8_TILE_SCALE_BYTES, 0);
    for(int g=0;g<Q8_NUM_GROUPS;g++){
        int col0=g*Q8_GROUP_COLS;
        uint8_t* go=wt.data()+g*Q8_GROUP_BYTES;
        for(int r=0;r<Q8_TILE_ROWS;r++){
            int row=row0+r; if(row>=(int)A->rows)break;
            for(int c=0;c<Q8_GROUP_COLS;c++){
                uint64_t flat=(uint64_t)row*cols+col0+c;
                uint64_t bo=(flat/32)*34;
                go[(r>>3)*512+c*8+(r&7)]=A->data[bo+2+(flat%32)];
            }
        }
        uint8_t* gs=wt.data()+Q8_TILE_WEIGHT_BYTES+g*Q8_GROUP_SCALE_BYTES;
        for(int r=0;r<Q8_TILE_ROWS;r++){
            int row=row0+r; if(row>=(int)A->rows)continue;
            for(int h=0;h<2;h++){
                uint64_t flat=(uint64_t)row*cols+col0+h*32;
                uint64_t bo=(flat/32)*34;
                float d_float=f16_to_f32((uint16_t)A->data[bo]|((uint16_t)A->data[bo+1]<<8));
                float row_s=row_scale[r];
                float row_inv=(row_s<1e-10f)?1.0f:(1.0f/row_s);
                float combined=d_float*row_inv;
                uint32_t uq=(uint32_t)(combined*256.0f+0.5f);
                if(uq>65535)uq=65535;
                int sc_addr=((r>>3)<<4)|((r&7)<<1)|h;
                *(uint16_t*)(gs+sc_addr*2)=(uint16_t)uq;
            }
        }
    }

    // ---- activations: deterministic pseudo-real pattern (any int16; golden matches) ----
    std::vector<int16_t> xq(cols);
    for(uint64_t c=0;c<cols;c++) xq[c]=(int16_t)((int)(c*7919 % 2000) - 1000);

    // ---- golden via q8_tile_golden, per 64-col group ----
    int64_t gold[Q8_TILE_ROWS]={0};
    for(int r=0;r<Q8_TILE_ROWS;r++){
        int row=row0+r;
        if(row>=(int)A->rows)continue;
        int64_t acc=0;
        for(int g=0;g<Q8_NUM_GROUPS;g++){
            int8_t W[64*64]={0}; uint16_t sc[64*2]={0}; int16_t act[64]={0};
            for(int c=0;c<Q8_GROUP_COLS;c++){
                uint64_t flat=(uint64_t)row*cols+g*Q8_GROUP_COLS+c;
                uint64_t bo=(flat/32)*34;
                W[c]= (int8_t)A->data[bo+2+(flat%32)];   // this row only -> W[r*64+c] with r=0
                act[c]=xq[g*Q8_GROUP_COLS+c];
            }
            // scale for this row, both blocks (from the permuted table we built)
            uint8_t* gs=wt.data()+Q8_TILE_WEIGHT_BYTES+g*Q8_GROUP_SCALE_BYTES;
            for(int h=0;h<2;h++){
                int sc_addr=((r>>3)<<4)|((r&7)<<1)|h;
                sc[h]=*(uint16_t*)(gs+sc_addr*2);
            }
            int8_t Wr[64*64]={0};
            for(int c=0;c<Q8_GROUP_COLS;c++) Wr[c]=W[c];   // row-major: Wr[r=0][c]
            int64_t out[64]={0};
            golden::q8_tile_golden(Wr, sc, act, out);
            acc += out[0];
        }
        gold[r]=acc;
    }

    // ---- emit q8_real.mem: DDR words from addr 0x00300000 ----
    // wt base = 0x00300000; region = weights(57344) + scales(3584) + acts(1792) = 62720 bytes = 7840 words
    uint32_t base=0x00300000;
    std::vector<uint8_t> ddr(Q8_TILE_WEIGHT_BYTES+Q8_TILE_SCALE_BYTES+cols*2,0);
    memcpy(ddr.data(), wt.data(), Q8_TILE_WEIGHT_BYTES+Q8_TILE_SCALE_BYTES);
    memcpy(ddr.data()+Q8_TILE_WEIGHT_BYTES+Q8_TILE_SCALE_BYTES, xq.data(), cols*2);
    size_t nbytes=ddr.size();
    size_t nwords=nbytes/8; // 62720/8 = 7840
    FILE* f=fopen("verilog/q8_real.mem","w");
    for(size_t w=0;w<nwords;w++){
        uint64_t v=0;
        for(int b=0;b<8;b++) v |= (uint64_t)ddr[w*8+b] << (8*b);
        fprintf(f,"%016llx\n",(unsigned long long)v);
    }
    fclose(f);
    FILE* g=fopen("verilog/q8_golden.mem","w");
    for(int r=0;r<Q8_TILE_ROWS;r++) fprintf(g,"%016llx\n",(unsigned long long)(uint64_t)gold[r]);
    fclose(g);
    printf("wrote verilog/q8_real.mem (%zu words, addr 0x00300000..0x%08X) + verilog/q8_golden.mem\n",
           nwords, base+(uint32_t)nbytes);
    return 0;
}
