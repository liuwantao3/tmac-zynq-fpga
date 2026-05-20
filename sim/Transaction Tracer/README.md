# Pipeline Profiler

## What It Is

`fpga_profiler.hpp` defines `fpga_sim::Profiler`, an instrumentation-based profiler that categorizes CPU time into 6 FPGA-simulation pipeline stages:

| Stage | What it measures |
|-------|-----------------|
| `STAGE_QUANTIZE` | float → int8/int16 conversion |
| `STAGE_DDR_COPY` | memcpy to/from simulated DDR |
| `STAGE_AXI_SETUP` | AXI-Lite register writes |
| `STAGE_COMPUTE` | FPGA compute (simulated systolic array) |
| `STAGE_DEQUANTIZE` | int32/int64 → float + scale multiply |
| `STAGE_OVERHEAD` | loop control, branches, etc. |

## How It Differs from the Chrome Trace Profiler

The Chrome Trace Profiler (`PROFILE_SCOPE` in `tmac_gguf.cpp`, activated by `--perf`) is **function-oriented** — it records raw function names (`matmul_fpga_q4k`, `rms_norm`, `attention`) and outputs Chrome Trace Events to `/tmp/pipeline_trace.json`. It tells you *which function* is slow.

This Profiler is **pipeline-stage-oriented** — it breaks each function's time into the 6 stages above, tells you *why that function is slow* (e.g. "quantize takes 60%"), identifies the bottleneck, and emits optimization recommendations. It also tracks per-tile and per-layer timing, and exports CSV for further analysis.

They are complementary: Chrome Trace narrows to a function, this Profiler digs into the root cause within it.

## Intended Usage

Wrap code sections with stage-appropriate timers, then call `report()` at the end:

```cpp
#include "Transaction Tracer/fpga_profiler.hpp"

void matmul_fpga_q4k(Tensor* out, const Tensor* a, const Tensor* b, ...) {
    auto& prof = fpga_sim::prof();

    for each tile:
        prof.begin_tile(row, col, dimM, dimN);

        {   auto s = fpga_sim::Profiler::Scope(prof, Profiler::STAGE_QUANTIZE);
            // float → int16 conversion
        }
        {   auto s = fpga_sim::Profiler::Scope(prof, Profiler::STAGE_DDR_COPY);
            // memcpy to simulated DDR
        }
        {   auto s = fpga_sim::Profiler::Scope(prof, Profiler::STAGE_COMPUTE);
            // vecmul_1x64_int16
        }
        {   auto s = fpga_sim::Profiler::Scope(prof, Profiler::STAGE_DEQUANTIZE);
            // int32 → float + scale
        }

        prof.end_tile();

    // at end of inference:
    prof.report();       // stage breakdown + bottleneck + recommendations
    prof.dump_csv("/tmp/profiler.csv");  // per-tile CSV
}
```

## Current Status

This file is **not actively used** — it is included (or was, before cleanup) in `tmac_gguf.cpp` but no `prof().begin()`/`prof().end()` calls have been wired in. The code is functionally sound and ready for integration if pipeline-stage-level profiling is needed beyond the existing Chrome Trace profiler.

## Minor Notes

- No `reset()` method — `tile_times_` accumulates across runs (fine for single-run profiling)
- `report()` total call count sums 3 of 6 stages (cosmetic only)
- Layer `-1` sentinel avoids wraparound in `begin_layer()`
