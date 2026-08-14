// ============================================================================
// Bit-exact golden models of the Q8_0 and Q5_0 FPGA compute cores.
//
// These replicate the exact integer/fixed-point arithmetic of
//   verilog/matmul_q8_core.v  and  verilog/matmul_q5_0_core.v
// so that FPGA results can be compared against a reference that has NO
// floating-point mismatch. See docs/maths.md for the notation and derivations.
//
// All functions are header-only and dependency-free (C++11).
// ============================================================================

#ifndef GOLDEN_MODEL_HPP
#define GOLDEN_MODEL_HPP

#include <stdint.h>
#include <stddef.h>

namespace golden {

// ===========================================================================
// Common fixed-point helpers
// ===========================================================================

// Sign-extend a 48-bit signed value to int64. The cores store accumulators as
// S48 (`reg signed [47:0]`); real model data never overflows, but for bit-exact
// comparison we normalize the accumulator to its S48 value.
static inline int64_t s48(int64_t v) {
    if (v & (int64_t(1) << 47)) v |= ~((int64_t(1) << 48) - 1);
    else                        v &= (int64_t(1) << 48) - 1;
    return v;
}

// Clamp an int64 to S16 range [-32768, 32767] (matches the Verilog d_pre clamp).
static inline int16_t clamp_s16(int64_t v) {
    if (v > 32767) return 32767;
    if (v < -32768) return -32768;
    return (int16_t)v;
}

// ===========================================================================
// f16_decode: half -> S24.16 (1.0 -> 65536). Bit-exact port of the Verilog
// function in matmul_q5_0_core.v. 16 fractional bits (was S24.8/1.0->256) so
// small block scales (d<0.004) no longer round to zero.
//
// SIGN-HANDLING (2026-08-14): llama.cpp quantizes Q5_0 blocks with the
// negative-d trick (d = max/-16); ~50% of real blocks have a negative d, and
// dequant (q-16)·d requires the sign. Bit 15 is applied to negate the
// magnitude, matching the fixed matmul_q5_0_core.v f16_decode.
// ===========================================================================
static inline int32_t f16_decode_s2416(uint16_t f16) {
    int32_t exp  = (f16 >> 10) & 0x1F;
    int32_t mant = f16 & 0x3FF;
    int32_t mag;
    if (exp == 0 || exp == 31) mag = 0;                       // subnormal/inf/NaN
    else if (exp >= 9) mag = (1024 + mant) << (exp - 9);      // exact
    else mag = ((1024 + mant) + (1 << (8 - exp))) >> (9 - exp); // round-half-up
    return (f16 & 0x8000) ? -mag : mag;                       // apply sign bit
}

// ===========================================================================
// Q8_0 core
//
//   deq = (q8 * sc) >>> 8      -> S16 (truncation, no saturation needed)
//   acc = SUM(deq * act)       -> S48
//
// Inputs (logical form, already preprocessed by the host):
//   W    : 64x64 INT8 weights, row-major (W[r][c] = weight for output row r,
//          input column c).
//   sc   : 64x2 UQ8.8 scales (sc[r][blk], blk = c/32).
//   act  : 64 x S16 activations (quantized input vector).
// Output:
//   out[64] : S48 accumulator per output row.
// ===========================================================================
static inline int16_t q8_dequant(int8_t q8, uint16_t sc) {
    // 8x16 signed product; >>> 8 arithmetic shift; truncate to S16.
    int32_t prod = (int32_t)q8 * (int32_t)sc;
    return (int16_t)(prod >> 8);
}

static inline void q8_tile_golden(const int8_t* W,    // 64*64 row-major
                                  const uint16_t* sc, // 64*2 row-major
                                  const int16_t* act, // 64
                                  int64_t* out)       // 64 (S48)
{
    for (int r = 0; r < 64; r++) {
        int64_t acc = 0;
        for (int c = 0; c < 64; c++) {
            int16_t deq = q8_dequant(W[r * 64 + c], sc[r * 2 + (c >> 5)]);
            acc += (int64_t)deq * (int64_t)act[c];
        }
        out[r] = s48(acc);
    }
}

// ===========================================================================
// Q5_0 core
//
//   d_pre = (f16_decode(d) * norm) >>> 8   -> clamp S16
//   dq    = d_pre * q5                     -> 21-bit signed
//   acc   = SUM(dq * act)                  -> S48
//
// Inputs:
//   blocks : 4 rows x 28 blocks of raw GGUF Q5_0 data, 22 bytes/block:
//              [0:1] d (f16 LE), [2:5] qh (u32 LE), [6:21] qs (16 bytes).
//            Row-major: blocks[row][blk][22].
//   norm   : 4 x UQ8.8 per-row normalization (1.0 -> 256).
//   act    : 896 x S16 activations.
// Output:
//   out[4] : S48 accumulator per row.
// ===========================================================================
static inline int16_t q5_d_pre(uint16_t d_f16, uint16_t norm) {
    int64_t d_fp   = f16_decode_s2416(d_f16);        // S24.16 (may be negative)
    int64_t shr    = (d_fp * (int64_t)norm) >> 8;    // >>> 8 (arithmetic shift)
    return clamp_s16(shr);
}

// Decode the 5-bit signed weight for element `wi` (0..31) of a block.
static inline int8_t q5_decode(const uint8_t* blk, int wi) {
    uint32_t qh = (uint32_t)blk[2] | ((uint32_t)blk[3] << 8) |
                  ((uint32_t)blk[4] << 16) | ((uint32_t)blk[5] << 24);
    int j        = (wi < 16) ? wi : wi - 16;
    uint8_t qs_b = blk[6 + j];
    uint8_t ql   = (wi < 16) ? (qs_b & 0xF) : (qs_b >> 4);
    uint8_t qh_b = (qh >> wi) & 1;
    int q5 = ((qh_b << 4) | ql) - 16;
    return (int8_t)q5;
}

static inline void q5_tile_golden(const uint8_t* blocks, // 4*28*22 row-major
                                  const uint16_t* norm,  // 4 (UQ8.8)
                                  const int16_t* act,    // 896 (S16)
                                  int64_t* out)          // 4 (S48)
{
    for (int r = 0; r < 4; r++) {
        int64_t acc = 0;
        for (int blk = 0; blk < 28; blk++) {
            const uint8_t* b = blocks + (r * 28 + blk) * 22;
            int16_t d_pre = q5_d_pre((uint16_t)(b[0] | (b[1] << 8)), norm[r]);
            for (int wi = 0; wi < 32; wi++) {
                int8_t q5 = q5_decode(b, wi);
                int32_t dq = (int32_t)d_pre * (int32_t)q5;   // 21-bit, exact
                acc += (int64_t)dq * (int64_t)act[blk * 32 + wi];
            }
        }
        out[r] = s48(acc);
    }
}

} // namespace golden

#endif // GOLDEN_MODEL_HPP
