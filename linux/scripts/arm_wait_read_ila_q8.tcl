# arm_wait_read_ila_q8.tcl
#
# Single-session ILA capture for the Q8 first-run corruption diagnosis.
# Keeps the JTAG connection open across arm -> trigger -> read (closing the
# target after arming disarms the ILA, so it must stay connected).
#
# Usage:
#   1. Host (this script, keeps running): vivado.bat -mode tcl -source linux/scripts/arm_wait_read_ila_q8.tcl
#   2. Board: ./tmac model.tmac --nowarmup
#   3. Script polls until the trigger fires, then writes linux/boot/ila_q8_trace.csv
#
# dbg_first_q8 = 1-cycle pulse on the FIRST Q8 COMPUTE after a Q5_0 descriptor.
# Trigger position 150 => ~150 pre (Q5 tail), ~874 post (CLEAR_ACC + COMPUTE + READ_RES).

open_hw_manager
connect_hw_server
open_hw_target

set dev [current_hw_device]
set_property PROBES.FILE {D:/Users/u/tmac-zynq-fpga/linux/boot/system_wrapper_ila.ltx} $dev
refresh_hw_device $dev

# IMPORTANT: dbg_q8_trig_done is a once-per-boot latch. Reprogram the PL so
# the trigger logic starts fresh (any prior tmac run already latched it).
set_property PROGRAM.FILE {D:/Users/u/tmac-zynq-fpga/linux/boot/system_wrapper.bit} $dev
program_hw_devices $dev
refresh_hw_device $dev

set ila [get_hw_ilas -of_objects $dev]
set pr [get_hw_probes -of_objects $ila -filter {NAME == {system_i/axi_hp_top_dbg_bus}}]

# Trigger on dbg_first_q8 (bit0=1), which is (state==COMPUTE) && q5_seen
# && !dbg_q8_trig_done -- i.e. the FIRST Q8 COMPUTE after a Q5_0 descriptor,
# exactly the corrupted run. dbg_first_q8=1 already guarantees state==COMPUTE
# (5'd13, state[2:0]=101), so matching the low nibble 1011 (char35='B')
# uniquely selects it. All other bits are don't-care.
set_property CONTROL.TRIGGER_POSITION 150 $ila
set_property TRIGGER_COMPARE_VALUE "eq142'h[string repeat X 35]B" $pr

puts "ARMED - run on the board now:  ./tmac model.tmac --nowarmup"
run_hw_ila $ila

set t0 [clock milliseconds]
set done 0
while {[clock milliseconds] - $t0 < 300000 && !$done} {
    set n [get_property STATUS.SAMPLE_COUNT $ila]
    set cs [get_property STATUS.CORE_STATUS $ila]
    if {$n == 1024 && $cs eq "IDLE"} { set done 1 }
    after 500
}
puts "CORE_STATUS = [get_property STATUS.CORE_STATUS $ila]  SAMPLE_COUNT = [get_property STATUS.SAMPLE_COUNT $ila]"

if {[get_property STATUS.SAMPLE_COUNT $ila] == 1024} {
    upload_hw_ila_data $ila
    set data [get_hw_ila_data -of_objects $ila]
    write_hw_ila_data -force -csv_file D:/Users/u/tmac-zynq-fpga/linux/boot/ila_q8_trace.csv $data
    puts "Trace written to linux/boot/ila_q8_trace.csv"
} else {
    puts "NO CAPTURE - trigger never fired (or tmac never ran while armed)"
}

close_hw_target
disconnect_hw_server
close_hw_manager
exit