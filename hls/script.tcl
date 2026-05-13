# HLS Synthesis Script for matmul_int8
# Supports: 64/128 configurable N, INT8, Matrix/Vector multiply

set project_name "matmul_int8"
set top_func "matmul_int8"

open_project -reset $project_name

add_files matmul_int8.cpp

set_property top $top_func [get_filesets sources_1]

open_solution -reset "solution1"

config_compile -name_latency 0 -prefer_std_logic

set_clock_uncertainty 12.5%

set_solution_property STALL_MODE builtin

source hls_config.tcl

csynth_design

export_design -format ip_catalog -version 1.0

exit