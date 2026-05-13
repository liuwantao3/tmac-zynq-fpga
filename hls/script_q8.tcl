# HLS Synthesis Script for matmul_q8
# Q8_0 direct path: FPGA receives Q8_0 bytes + combined fixed-point scales,
# does Q8→INT16 dequant via LUT-based multipliers, then INT16×INT16 systolic.
#
# Resource usage (Zynq 7010, 80 DSPs):
#   DSP:  64 (8×8 systolic array, 80% of 80)
#   LUT:  ~14K (scale multipliers + control, ~80% of 17,600)
#   BRAM: ~36 KB (27% of 135 KB)
#   FF:   ~16K (45% of 35,200)

set project_name "matmul_q8"
set top_func "matmul_q8"

open_project -reset $project_name

add_files matmul_q8.cpp
add_files matmul_q8.hpp

set_property top $top_func [get_filesets sources_1]

open_solution -reset "solution1"

config_compile -name_latency 0 -prefer_std_logic

# Force all non-systolic multipliers to LUT (preserve all 64 DSPs for array)
config_bind -mul_style luts

set_clock_uncertainty 12.5%

set_solution_property STALL_MODE builtin

source hls_config.tcl

csynth_design

export_design -format ip_catalog -version 1.0

exit
