# HLS Shared Configuration
# N=64 fixed, supports Q8_0 direct path (primary) and INT16 fallback
# Zynq 7010: 80 DSP, 17,600 LUT, 135 KB BRAM (NO URAM)

# Pipeline inner loops of matmul blocks
set_directive_pipeline "matmul_64x64" -II 1
set_directive_pipeline "vecmul_1x64" -II 1
set_directive_pipeline "vecmul_1x64_q8" -II 1

# Cyclic array partitioning for parallel systolic access
set_directive_array_partition "matmul_64x64" -factor 8 -dim 2 -type cyclic
set_directive_array_partition "vecmul_1x64" -factor 8 -dim 2 -type cyclic
set_directive_array_partition "vecmul_1x64_q8" -factor 8 -dim 2 -type cyclic

# BRAM for array storage (Zynq 7010: BRAM only, no URAM)
set_directive_resource "matmul_64x64" -resource "B_buf" -core RAM_2P_BRAM
set_directive_resource "vecmul_1x64" -resource "M" -core RAM_2P_BRAM
set_directive_resource "vecmul_1x64_q8" -resource "B_q8" -core RAM_2P_BRAM

# Force all scale multipliers to LUT fabric (zero DSPs for dequant)
config_bind -mul_style luts

config_compile -name_latency 0 -prefer_std_logic