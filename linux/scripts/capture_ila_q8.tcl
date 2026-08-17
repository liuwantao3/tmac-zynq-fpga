# capture_ila_q8.tcl
#
# ILA capture for the Q8 first-run corruption diagnosis (2026-08-16).
# Run with Vivado (NOT xsdb -- Hardware Manager commands):
#   vivado.bat -mode tcl -source linux/scripts/capture_ila_q8.tcl
#
# Flow (board already booted to Linux shell on USB-UART):
#   1. Host: start this script (arms the ILA, then blocks waiting for trigger)
#   2. Board: ./tmac model.tmac --nowarmup
#   3. Script returns when the ILA triggered + captured; writes the trace
#      to linux/boot/ila_q8_trace.csv for programmatic analysis.
#
# The ILA is armed with trigger = dbg_bus[0] (dbg_first_q8), a 1-cycle pulse
# on the FIRST Q8 COMPUTE after a Q5_0 descriptor. Trigger position 85 => ~870
# samples post-trigger (covers CLEAR_ACC + full 512-cycle COMPUTE + READ_RES)
# and ~154 pre-trigger (Q5_READ_RES / LOAD_* tail).

open_hw_manager
connect_hw_server
open_hw_target

set dev [current_hw_device]
refresh_hw_device $dev

set_property PROBES.FILE {D:/Users/u/tmac-zynq-fpga/linux/boot/system_wrapper_ila.ltx} $dev

# Arm the ILA: trigger on dbg_first_q8 (probe0 bit 0 == 1), window mostly post-trigger
set ila [get_hw_ilas -of_objects $dev]
set_property CONTROL.TRIGGER_POSITION 85 $ila
set_property CONTROL.TRIGGER_CONDITION {{probe0[0] == 1'b1}} $ila

puts "ILA armed. On the board run:  ./tmac model.tmac --nowarmup"
puts "Waiting for trigger (dbg_first_q8) + capture..."

run_hw_ila $ila

set t0 [clock milliseconds]
while {[clock milliseconds] - $t0 < 300000} {
    set st [get_property STATUS.CAPTURE_STATUS $ila]
    if {$st == "Idle"} { break }
    after 500
}
puts "Capture status: [get_property STATUS.CAPTURE_STATUS $ila]"

write_hw_ila_data -force -csv_file D:/Users/u/tmac-zynq-fpga/linux/boot/ila_q8_trace.csv -data_depth 1024 $ila
puts "Trace written to linux/boot/ila_q8_trace.csv"
puts "bits: [0]=dbg_first_q8 [5:1]=state [12:9]=col_group [18:13]=res_addr"
puts "      [67:63]=acc_clr_cnt [68]=p2_valid [74:69]=p2_row_base"
puts "      [106:75]=acc_rw_rd[31:0] [138:107]=p2_partial0[31:0] [141:139]=pre_read_g"

close_hw_target
disconnect_hw_server
close_hw_manager
exit