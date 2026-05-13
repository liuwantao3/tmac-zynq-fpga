# HLS FPGA Kernels

High-level synthesis sources for the Q8_0 direct-path matrix multiplication accelerator.

## Status
**Design phase** — Q8_0 LUT-based approach selected. FPGA receives Q8_0 weight bytes
+ combined fixed-point scales (UQ8.8), does Q8→INT16 dequant via LUT multipliers,
then INT16×INT16 systolic matmul. All 64 DSPs preserved for the 8×8 systolic array;
all scale multiplication absorbed by LUT fabric (~14K LUTs).

## Files
- **`matmul_q8.cpp`** — **Primary kernel**: Q8_0 direct path with LUT scale multipliers
- `matmul_q8.hpp` — Q8 kernel type defines
- `matmul_int16.cpp` — INT16 kernel (legacy fallback, same 8×8 systolic)
- `matmul_int16.hpp` — INT16 type defines
- `matmul_int8.cpp` — INT8 kernel (deprecated: SwiGLU needs >8-bit)
- `script_q8.tcl` — Q8 kernel synthesis script (recommended)
- `script_int16.tcl` — INT16 kernel synthesis script (alternative)
- `script.tcl` — INT8 kernel synthesis script (deprecated)
- `hls_config.tcl` — Shared HLS directives (config_bind -mul_style luts)

---

## Kernel: matmul_q8 (Primary)

Q8_0 direct path with LUT-based scale multipliers. INT16×INT16→INT64 systolic.

```cpp
void matmul_q8(
    ap_int<8> A[N * N],                   // Q8_0 weight bytes (4096 B)
    combined_scale_t combined_scales[N*2], // 128 × UQ8.8 (256 B)
    in_t X[N],                            // Activation vector (INT16, 128 B)
    acc_t Y[N],                           // Output vector (INT64, 512 B)
    volatile ap_uint<32> *control,
    volatile ap_uint<32> *status,
    ap_uint<1> &interrupt
);
```

### Dequantization (LUT-based, zero DSP)
```text
val = (q8_val * combined_scale) >> 8
```
Where `combined_scale = (block_scale / row_scale)` precomputed on ARM as UQ8.8.
Single INT8×UQ8.8 integer multiply → HLS maps to LUTs via `config_bind -mul_style luts`.

### Config Register
| Bit | Field | Description |
|-----|-------|-------------|
| [0] | start | Start operation |
| [1] | int_enable | Enable interrupt |
| [3] | op_vecmul | 0=MatMul, 1=VecMul |

### Resource Usage (Zynq 7010)
| Resource | Available | Used | Percent |
|----------|-----------|------|---------|
| DSP | 80 | 64 | 80% (8×8 systolic only) |
| LUT | 17,600 | ~14K | ~80% (scale multipliers + control) |
| BRAM | 135 KB | ~36 KB | ~27% |
| FF | 35,200 | ~16K | ~45% |

---

## Kernel: matmul_int16 (Alternative)

INT16×INT16→INT64 (higher precision, same 8×8 systolic). Requires ARM-side
Q8→INT16 dequantization — uses DSPs more efficiently but burns CPU cycles.

```cpp
void matmul_int16(
    in_t A[N * N],       // ap_int<16>
    in_t B[N * N],       // ap_int<16>
    acc_t C[N * N],      // ap_int<64>
    volatile ap_uint<32> *control,
    volatile ap_ap_uint<32> *status,
    ap_uint<1> &interrupt
);
```

### Resource Usage (Zynq 7010)
| Resource | Available | Used | Percent |
|----------|-----------|------|---------|
| DSP | 80 | 64 | 80% |
| LUT | 17,600 | ~16K | ~90% |
| BRAM | 135 KB | ~48 KB | ~36% |
| FF | 35,200 | ~16K | ~45% |

### Performance
| Mode | Cycles/tile | Time @ 150 MHz |
|------|------------|----------------|
| VecMul (1×64) | 64 | 0.43 µs |
| MatMul (64×64) | 32,768 | 218 µs |

---

## Target Specs (Zynq 7010)
| Resource | Available |
|----------|-----------|
| DSP | 80 |
| LUT | 17,600 |
| BRAM | 135 KB (60 × BRAM18K) |
| FF | 35,200 |
| DDR | 512 MB |

## Build (in Docker)
```bash
cd /workspace/fpga/hls

# Q8_0 kernel (recommended — LUT scales, DSP systolic)
vitis_hls -s script_q8.tcl

# INT16 kernel (alternative)
vitis_hls -s script_int16.tcl

# INT8 kernel (deprecated)
vitis_hls -s script.tcl
```

### Why INT16?
INT8 quantization is insufficient for Qwen2-0.5B due to SwiGLU error amplification
(max logit diff 35.6, top-1 wrong). INT16 achieves near-identical results
(max logit diff 0.24, top-5 match 5/5). See `sim/tmac_gguf.cpp --fpga-int16`.
