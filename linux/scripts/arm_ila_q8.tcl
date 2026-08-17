# arm_ila_q8.tcl
#
# Phase A: arm the ILA for the Q8 first-run corruption capture, then disconnect.
# The ILA stays armed in the FPGA until dbg_bus[0] (dbg_first_q8) fires.
#
# Run:  vivado.bat -mode tcl -source linux/scripts/arm_ila_q8.tcl
# Then on the board:  ./tmac model.tmac --nowarmup
# Then read with:     linux/scripts/read_ila_q8.tcl
#
# dbg_first_q8 = 1-cycle pulse on the FIRST Q8 COMPUTE after a Q5_0 descriptor.
# Trigger position 150 => ~150 samples pre-trigger (Q5_READ_RES / LOAD tail),
# ~874 post-trigger (CLEAR_ACC + 512-cycle COMPUTE + READ_RES).

open_hw_manager
connect_hw_server
open_hw_target

set dev [current_hw_device]
set_property PROBES.FILE {D:/Users/u/tmac-zynq-fpga/linux/boot/system_wrapper_ila.ltx} $dev
refresh_hw_device $dev

set ila [get_hw_ilas -of_objects $dev]
set pr [get_hw_probes -of_objects $ila -filter {NAME == {system_i/axi_hp_top_dbg_bus}}]

# TRIGGER_MODE / TRIGGER_CONDITION / CAPTURE_MODE are read-only at runtime and
# already at their defaults from the BD config (BASIC_ONLY / AND / ALWAYS).
set_property CONTROL.TRIGGER_POSITION 150 $ila
set_property TRIGGER_COMPARE_VALUE "eq142'h[string repeat X 35]1" $pr

puts "Trigger compare = [get_property TRIGGER_COMPARE_VALUE $pr]"
puts "TRIGGER_POSITION = [get_property CONTROL.TRIGGER_POSITION $ila]"
puts "ARMING ... run tmac on the board, then read_ila_q8.tcl"

run_hw_ila $ila
puts "ILA armed (run_hw_ila returned; trigger armed in FPGA)"

close_hw_target
disconnect_hw_server
close_hw_manager
exit