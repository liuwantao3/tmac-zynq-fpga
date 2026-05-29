# Vivado block design script: INT16 matmul on Zynq-7010
# Usage: vivado -mode batch -source build_bd.tcl

set origin "D:/Users/u/tmac-zynq-fpga"
set proj_dir "$origin/vivado_integration/proj_bd"
set rtl_dir  "$origin/vivado_integration/rtl"
set ver_dir  "$origin/verilog"

file mkdir $proj_dir

create_project -force matmul_bd $proj_dir -part xc7z010clg400-1

# Add RTL sources
add_files -fileset sources_1 [list \
    [file normalize "$rtl_dir/axi_wrap_int16.v"] \
    [file normalize "$ver_dir/matmul_int16_core.v"] \
]
update_compile_order -fileset sources_1

# Create block design
create_bd_design "system"

# Zynq PS7 - configure clock BEFORE automation
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
set_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 100.0 [get_bd_cells ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR"} [get_bd_cells ps7]

# RTL module as block
create_bd_cell -type module -reference axi_wrap_int16 axi_wrap

# AXI Interconnect (1 master, 1 slave)
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_intercon
set_property CONFIG.NUM_MI 1 [get_bd_cells axi_intercon]
set_property CONFIG.NUM_SI 1 [get_bd_cells axi_intercon]

# Connect AXI bus: PS7 -> Interconnect -> Wrapper
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] \
    [get_bd_intf_pins axi_intercon/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_intercon/M00_AXI] \
    [get_bd_intf_pins axi_wrap/s_axil]

# Connect clocks
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ps7/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_intercon/ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_intercon/S00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_intercon/M00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_wrap/clk]

# Connect resets
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_intercon/ARESETN]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_intercon/S00_ARESETN]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_intercon/M00_ARESETN]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_wrap/rst_n]

# Validate
validate_bd_design
save_bd_design

# Clock constraint for top-level implementation
set xdc_file [open "$proj_dir/system.xdc" w]
puts $xdc_file "# Clock constraint for PS7 FCLK_CLK0"
puts $xdc_file "create_clock -period 10.000 -name clk_100m \[get_pins -hier *FCLK_CLK0\]"
close $xdc_file
add_files -fileset constrs_1 "$proj_dir/system.xdc"

# Create HDL wrapper
make_wrapper -files [get_files [file normalize "$proj_dir/matmul_bd.srcs/sources_1/bd/system/system.bd"]] -top
add_files -norecurse [file normalize "$proj_dir/matmul_bd.srcs/sources_1/bd/system/hdl/system_wrapper.v"]
update_compile_order -fileset sources_1

# Launch synthesis
launch_runs synth_1 -jobs 2
wait_on_run synth_1

# Open synth design and write checkpoint
open_run synth_1
write_checkpoint -force "$proj_dir/post_synth.dcp"

# Launch implementation and bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1

# Open impl design and write checkpoint
open_run impl_1
write_checkpoint -force "$proj_dir/post_impl.dcp"

puts "=== DONE ==="
