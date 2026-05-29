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

# Zynq PS7 - configure clock AND DDR BEFORE automation
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
set_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 100.0 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_PARTNO "MT41J256M16 RE-125" [get_bd_cells ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR"} [get_bd_cells ps7]

# Force DDR geometry for MicroPhase Z7-Lite (MT41J256M16, 512MB)
set_property CONFIG.PCW_UIPARAM_DDR_PARTNO "MT41J256M16 RE-125" [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_DEVICE_CAPACITY "4096 MBits" [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_BUS_WIDTH "16 Bit" [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_DRAM_WIDTH "16 Bits" [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_ROW_ADDR_COUNT 15 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_COL_ADDR_COUNT 10 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_BANK_ADDR_COUNT 3 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_SPEED_BIN "DDR3_1066F" [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_FREQ_MHZ 533.333333 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_CL 7 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_CWL 6 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_AL 0 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY0 0.250 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY1 0.250 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY2 0.250 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_BOARD_DELAY3 0.250 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_TRAIN_DATA_EYE 1 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_TRAIN_READ_GATE 1 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_TRAIN_WRITE_LEVEL 1 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_ECC "Disabled" [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_MEMORY_TYPE "DDR 3" [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_HIGH_TEMP "Normal (0-85)" [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_USE_INTERNAL_VREF 0 [get_bd_cells ps7]

# Use UART0 (MIO 14/15) for serial output on MicroPhase Z7-Lite
set_property CONFIG.PCW_UART0_PERIPHERAL_ENABLE 1 [get_bd_cells ps7]
set_property CONFIG.PCW_UART0_UART0_IO "MIO 14 .. 15" [get_bd_cells ps7]
set_property CONFIG.PCW_UART1_PERIPHERAL_ENABLE 0 [get_bd_cells ps7]

# Ensure FCLK_CLK0 is enabled for PL fabric
set_property CONFIG.PCW_EN_CLK0_PORT 1 [get_bd_cells ps7]
set_property CONFIG.PCW_FCLK_CLK0_BUF TRUE [get_bd_cells ps7]

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

# Create HDL wrapper and set as top
make_wrapper -files [get_files [file normalize "$proj_dir/matmul_bd.srcs/sources_1/bd/system/system.bd"]] -top
add_files -norecurse [file normalize "$proj_dir/matmul_bd.srcs/sources_1/bd/system/hdl/system_wrapper.v"]
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1

# Launch synthesis
launch_runs synth_1 -jobs 2
wait_on_run synth_1

# Open synth design and write checkpoint
open_run synth_1
write_checkpoint -force "$proj_dir/post_synth.dcp"

# Diagnostics: check clock constraints
puts "=== CLOCK REPORT ==="
report_clocks -file "$proj_dir/clocks.rpt"
report_timing_summary -file "$proj_dir/timing.rpt"
report_io -file "$proj_dir/io.rpt"

# Check if *FCLK_CLK0 pattern matches any pins
set fclk_pins [get_pins -hier -quiet *FCLK_CLK0]
puts "FCLK_CLK0 pins found: [llength $fclk_pins]"
if {[llength $fclk_pins] > 0} {
    puts "First pin: [lindex $fclk_pins 0]"
}

# Launch implementation and bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1

# Open impl design and write checkpoint
open_run impl_1
write_checkpoint -force "$proj_dir/post_impl.dcp"

# Export hardware platform (XSA) for Vitis
write_hw_platform -fixed -include_bit -force "$proj_dir/matmul_bd.xsa"

puts "=== DONE ==="
