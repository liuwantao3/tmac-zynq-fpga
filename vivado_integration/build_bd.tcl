# Vivado block design: HP loopback test — read from DDR-A, write to DDR-B
# Usage: vivado -mode batch -source build_bd.tcl

set origin "D:/Users/u/tmac-zynq-fpga"
set proj_dir "$origin/vivado_integration/proj_bd"
set rtl_dir  "$origin/vivado_integration/rtl"
set ver_dir  "$origin/verilog"

file mkdir $proj_dir
create_project -force matmul_bd $proj_dir -part xc7z010clg400-1

# Add RTL sources
add_files -fileset sources_1 [list \
    [file normalize "$ver_dir/axihp_read_master.v"] \
    [file normalize "$ver_dir/axihp_write_master.v"] \
    [file normalize "$ver_dir/matmul_q8_core.v"] \
    [file normalize "$ver_dir/matmul_q5_0_core.v"] \
    [file normalize "$rtl_dir/hp_fsm_top.v"] \
]
update_compile_order -fileset sources_1

# Create block design
create_bd_design "system"

# ======== PS7 ========
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
set_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ 100.0 [get_bd_cells ps7]
set_property CONFIG.PCW_UIPARAM_DDR_PARTNO "MT41J256M16 RE-125" [get_bd_cells ps7]

# HP0 config MUST be set BEFORE apply_bd_automation — automation parses HP0
# parameters to configure the internal AXI data path muxing. Setting them
# after automation leaves the internal mux at default 32-bit even if the
# XML parameter reports 64.
set_property CONFIG.PCW_USE_S_AXI_HP0 1 [get_bd_cells ps7]
set_property CONFIG.PCW_S_AXI_HP0_DATA_WIDTH 64 [get_bd_cells ps7]
set_property CONFIG.PCW_S_AXI_HP0_ID_WIDTH 6 [get_bd_cells ps7]
set_property CONFIG.PCW_S_AXI_HP0_BASEADDR 0x00000000 [get_bd_cells ps7]
set_property CONFIG.PCW_S_AXI_HP0_HIGHADDR 0x3FFFFFFF [get_bd_cells ps7]

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

# PL fabric clocks: FCLK_CLK0 = 100 MHz (fabric logic),
# FCLK_CLK1 = 100 MHz (async domain for HP0, avoids FIFO bypass)
set_property CONFIG.PCW_EN_CLK0_PORT 1 [get_bd_cells ps7]
set_property CONFIG.PCW_FCLK_CLK0_BUF TRUE [get_bd_cells ps7]
set_property CONFIG.PCW_EN_CLK1_PORT 1 [get_bd_cells ps7]
set_property CONFIG.PCW_FCLK_CLK1_BUF TRUE [get_bd_cells ps7]
set_property CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ 100.0 [get_bd_cells ps7]

# ======== hp_fsm_top (descriptor FSM, no compute cores) ========
create_bd_cell -type module -reference hp_fsm_top axi_hp_top

# ======== AXI4-Lite interconnect (GP0 → control) ========
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_lite
set_property CONFIG.NUM_MI 1 [get_bd_cells axi_lite]
set_property CONFIG.NUM_SI 1 [get_bd_cells axi_lite]

connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] [get_bd_intf_pins axi_lite/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_lite/M00_AXI] [get_bd_intf_pins axi_hp_top/S_AXI]

# ======== PS7 HP0 direct connection (no interconnect) ========
# Both sides are AXI3 on the same FCLK_CLK0 domain, so no CDC or protocol adaptation needed.
# Using an AXI interconnect previously caused S00_DATA_WIDTH auto-detection issues (default 32
# when side is 64), inserting a 64→32 data width converter that dropped the upper 32 bits.
# CRITICAL: Force M_AXI_HP data width to 64. Vivado BD does NOT parse DATA_WIDTH from
# X_INTERFACE_PARAMETER in the Verilog RTL — it defaults to 32 unless explicitly set here.
set_property CONFIG.DATA_WIDTH 64 [get_bd_intf_pins axi_hp_top/M_AXI_HP]
connect_bd_intf_net [get_bd_intf_pins axi_hp_top/M_AXI_HP] [get_bd_intf_pins ps7/S_AXI_HP0]

# ======== Clocks ========
# Fabric clock domain (FCLK_CLK0): GP0, AXI Lite, module logic, read master
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ps7/M_AXI_GP0_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_lite/ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_lite/S00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_lite/M00_ACLK]
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins axi_hp_top/clk]

# HP0 clock: PS7 HP0 receives FCLK_CLK0 (same as PL logic)
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] [get_bd_pins ps7/S_AXI_HP0_ACLK]

# ======== Resets ========
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_lite/ARESETN]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_lite/S00_ARESETN]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_lite/M00_ARESETN]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] [get_bd_pins axi_hp_top/rst_n]


# Validate
validate_bd_design

# Address mapping — assign both GP0 and HP0 segments manually
# Get the slave segment objects
set gp0_seg [get_bd_addr_segs -of_objects [get_bd_intf_pins axi_hp_top/S_AXI]]
set hp0_seg [get_bd_addr_segs -of_objects [get_bd_intf_pins ps7/S_AXI_HP0]]

puts "GP0 slave seg: $gp0_seg"
puts "HP0 slave seg: $hp0_seg"

# Check if gp0_seg is valid; if not, get the segment by path
if {$gp0_seg eq ""} {
    set gp0_seg [get_bd_addr_segs -quiet {axi_hp_top/S_AXI/reg0}]
    puts "GP0 seg from path: $gp0_seg"
}

# GP0: CPU→PL control registers at 0x43C00000 (64K range)
if {$gp0_seg ne ""} {
    assign_bd_address -offset 0x43C00000 -range 64K $gp0_seg
} else {
    puts "ERROR: cannot find GP0 address segment"
}

# HP0: PL→DDR at 0x00000000 (256 MB)
create_bd_addr_seg -range 0x20000000 -offset 0x00000000 \
    [get_bd_addr_spaces axi_hp_top/M_AXI_HP] \
    $hp0_seg \
    SEG_axi_hp_top_HP0_DDR

puts "GP0 control: [get_bd_addr_segs -quiet -of_objects [get_bd_addr_spaces ps7/Data]]"
puts "HP0 DDR:     [get_bd_addr_segs -quiet -of_objects [get_bd_addr_spaces axi_hp_top/M_AXI_HP]]"

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
set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY full [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FSM_EXTRACTION one_hot [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]
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

# Patch generated ps7_init.tcl to enable FCLK_CLK0 and FCLK_CLK1
# (Vivado-generated init code excludes clock enable bits from the mask)
# MUST be at the very end — any subsequent Vivado step regenerates the file.
set ps7_init_file [file normalize "$proj_dir/matmul_bd.gen/sources_1/bd/system/ip/system_ps7_0/ps7_init.tcl"]
if {[file exists $ps7_init_file]} {
    set fd [open $ps7_init_file r]
    set content [read $fd]
    close $fd
    # Replace mask + value for FPGA_CLK_CTRL (0xF8000170): add CLK0_EN + CLK1_EN
    regsub -all {mask_write 0XF8000170 0x03F03F30 0x00400400} $content {mask_write 0XF8000170 0x03F83FB0 0x00480480} content
    # Same for FPGA_CLK_CTRL2 (0xF8000180, CLK3)
    regsub -all {mask_write 0XF8000180 0x03F03F30 0x00400400} $content {mask_write 0XF8000180 0x03F83FB0 0x00480480} content
    set fd [open $ps7_init_file w]
    puts -nonewline $fd $content
    close $fd
    puts "ps7_init.tcl patched: CLK0_EN + CLK1_EN added"
} else {
    puts "WARNING: ps7_init.tcl not found at $ps7_init_file"
}

puts "=== DONE ==="
