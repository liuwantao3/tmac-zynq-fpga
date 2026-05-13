# Vivado Block Design Script
# Usage: vivado -mode batch -source block_design.tcl
# Note: HLS IP must be exported first via vitis_hls -s script_q8.tcl
# Then generate hls_ip.tcl from the exported IP before running this.

set project_name "matmul_q8"

create_project $project_name ./vivado_project -part xc7z010clg400-1

set_property target_language VHDL [current_project]
set_property simulator_language Mixed [current_project]

create_bd_design -autoize "system"

# TODO: Generate hls_ip.tcl from exported HLS IP:
#   vitis_hls -s ../hls/script_q8.tcl
#   Then source the generated IPI tcl script here.
# source ./hls_ip.tcl

validate_bd_design
save_bd_design

generate_target all [get_files ./vivado_project/${project_name}.bd]

make_wrapper -files [get_files ./vivado_project/${project_name}.bd] -top

launch_runs impl_1 -to_step write_bitstream -Jobs 4

wait_on_run impl_1

exit