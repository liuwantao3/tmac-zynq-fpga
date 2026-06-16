# Vivado block design: matmul_top with AXI4-Lite (GP0) + AXI HP (HP0, async clock)
# Usage: vivado -mode batch -source build_bd.tcl

set origin "D:/Users/u/tmac-zynq-fpga"
set proj_dir "$origin/vivado_integration/proj_bd"
set rtl_dir  "$origin/vivado_integration/rtl"
set ver_dir  "$origin/verilog"

file mkdir $proj_dir
create_project -force matmul_bd $proj_dir -part xc7z010clg400-1

# Add RTL sources (AXI HP + INT16 only)
add_files -fileset sources_1 [list \
    [file normalize "$rtl_dir/axi_hp_int16_top.v"] \
    [file normalize "$rtl_dir/axi_wrap_int16.v"] \
    [file normalize "$ver_dir/matmul_int16_core.v"] \
    [file normalize "$ver_dir/axihp_read_master.v"] \
    [file normalize "$ver_dir/axihp_write_master.v"] \
]
update_compile_order -fileset sources_1

# Create block design
create_bd_design "system"

# ======== PS7 ========
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
set_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 100.0 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_PARTNO "MT41J256M16 RE-125" [get_bd_cells ps7]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR"} [get_bd_cells ps7]

# DDR config for MicroPhase Z7-Lite
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

# UART0 on MIO 14/15
set_property CONFIG.PCW_UART0_PERIPHERAL_ENABLE 1 [get_bd_cells ps7]
set_property CONFIG.PCW_UART0_UART0_IO "MIO 14 .. 15" [get_bd_cells ps7]
set_property CONFIG.PCW_UART1_PERIPHERAL_ENABLE 0 [get_bd_cells ps7]

# HP0: primary PL↔DDR port. Uses FCLK_CLK1 (async from PS internal clock)
# to ensure boot ROM allocates write FIFO blocks (synchronous clock disables FIFO).
set_property CONFIG.PCW_USE_S_AXI_HP0 1 [get_bd_cells ps7]
set_property CONFIG.PCW_S_AXI_HP0_DATA_WIDTH 64 [get_bd_cells ps7]
set_property CONFIG.PCW_S_AXI_HP0_ID_WIDTH 6 [get_bd_cells ps7]
set_property CONFIG.PCW_S_AXI_HP0_BASEADDR 0x00000000 [get_bd_cells ps7]
set_property CONFIG.PCW_S_AXI_HP0_HIGHADDR 0x3FFFFFFF [get_bd_cells ps7]

# PL fabric clocks: FCLK_CLK0 = 100 MHz (fabric logic),
# FCLK_CLK1 = 100 MHz (async domain for HP0, avoids FIFO bypass)
set_property CONFIG.PCW_EN_CLK0_PORT 1 [get_bd_cells ps7]
set_property CONFIG.PCW_FCLK_CLK0_BUF TRUE [get_bd_cells ps7]
set_property CONFIG.PCW_EN_CLK1_PORT 1 [get_bd_cells ps7]
set_property CONFIG.PCW_FCLK_CLK1_BUF TRUE [get_bd_cells ps7]
set_property CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ 100.0 [get_bd_cells ps7]

# ======== axi_hp_int16_top RTL module ========
create_bd_cell -type module -reference axi_hp_int16_top axi_hp_top

# ======== AXI4-Lite interconnect (GP0 → control) ========
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_lite
set_property CONFIG.NUM_MI 1 [get_bd_cells axi_lite]
set_property CONFIG.NUM_SI 1 [get_bd_cells axi_lite]

connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins axi_lite/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_lite/M00_AXI] [get_bd_intf_pins axi_hp_top/S_AXI]

# ======== AXI HP interconnect (top → PS7 HP0 = DDR, read + write path) ========
# HP0 clock domain uses FCLK_CLK1 (async) so boot ROM allocates write FIFO blocks.
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_hp
set_property CONFIG.NUM_MI 1 [get_bd_cells axi_hp]
set_property CONFIG.NUM_SI 1 [get_bd_cells axi_hp]
set_property CONFIG.PROTOCOL "AXI3" [get_bd_cells axi_hp]
set_property CONFIG.DATA_WIDTH 64 [get_bd_cells axi_hp]
set_property CONFIG.ADDR_WIDTH 32 [get_bd_cells axi_hp]

connect_bd_intf_net [get_bd_intf_pins axi_hp_top/M_AXI_HP] [get_bd_intf_pins axi_hp/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_hp/M00_AXI] [get_bd_intf_pins ps7/S_AXI_HP0]

# ======== Clocks ========
# Fabric clock domain (FCLK_CLK0): GP0, AXI Lite, module logic, read master
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ps7/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_lite/ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_lite/S00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_lite/M00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_hp_top/clk]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_hp/S00_ACLK]

# HP0 clock domain (FCLK_CLK1, async): forces boot ROM to allocate write FIFO blocks
connect_bd_net [get_bd_pins ps7/FCLK_CLK1] [get_bd_pins axi_hp/ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK1] [get_bd_pins axi_hp/M00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK1] [get_bd_pins ps7/S_AXI_HP0_ACLK]

# ======== Resets ========
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_lite/ARESETN]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_lite/S00_ARESETN]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_lite/M00_ARESETN]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_hp_top/rst_n]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_hp/ARESETN]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_hp/S00_ARESETN]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_hp/M00_ARESETN]


# Validate
validate_bd_design

# Address mapping
# GP0 → axi_hp_top control at 0x43C00000
set ctrl_seg [get_bd_addr_segs -quiet axi_hp_top/S_AXI/reg0]
if {$ctrl_seg eq ""} {
    assign_bd_address
} else {
    create_bd_addr_seg -range 0x00010000 -offset 0x43C00000 \
        [get_bd_addr_spaces ps7/Data] $ctrl_seg SEG_hp_top_ctrl
}
# HP0: axi_hp_top → PS7 S_AXI_HP0 → DDR 0x00000000-0x1FFFFFFF
set hp_seg [get_bd_addr_segs -quiet ps7/S_AXI_HP0/HP0_DDR_LOW]
if {$hp_seg eq ""} { set hp_seg [get_bd_addr_segs -quiet ps7/S_AXI_HP0/Reg] }
if {$hp_seg ne ""} {
    create_bd_addr_seg -range 0x20000000 -offset 0x00000000 \
        [get_bd_addr_spaces axi_hp_top/M_AXI_HP] $hp_seg SEG_hp_top_ddr
    puts "HP0 DDR mapped OK"
} else {
    puts "ERROR: Cannot find PS7 HP0 address segment!"
    puts "Available HP0 segments: [get_bd_addr_segs -quiet ps7/S_AXI_HP0/*]"
    assign_bd_address
}

save_bd_design

# Clock constraint
set xdc_file [open "$proj_dir/system.xdc" w]
puts $xdc_file "create_clock -period 10.000 -name clk_100m \[get_pins -hier *FCLK_CLK0\]"
close $xdc_file
add_files -fileset constrs_1 "$proj_dir/system.xdc"

# Wrapper
make_wrapper -files [get_files [file normalize "$proj_dir/matmul_bd.srcs/sources_1/bd/system/system.bd"]] -top
add_files -norecurse [file normalize "$proj_dir/matmul_bd.srcs/sources_1/bd/system/hdl/system_wrapper.v"]
set_property top system_wrapper [current_fileset]
update_compile_order -fileset sources_1

# Synthesis
launch_runs synth_1 -jobs 8
wait_on_run synth_1

# Checkpoint
open_run synth_1
write_checkpoint -force "$proj_dir/post_synth.dcp"

# Implementation + bitstream
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1

open_run impl_1
write_checkpoint -force "$proj_dir/post_impl.dcp"

# XSA
write_hw_platform -fixed -include_bit -force "$proj_dir/matmul_bd.xsa"

puts "=== DONE ==="
