# ILA capture test: program debug_test, arm ILA, save data
open_hw_manager
connect_hw_server
open_hw_target
current_hw_device [lindex [get_hw_devices] end]
puts "Device: [get_property NAME [current_hw_device]]"

set proj D:/Users/u/tmac-zynq-fpga/vivado_integration/debug_proj
set_property PROGRAM.FILE $proj/debug_test.runs/impl_1/debug_test.bit [current_hw_device]
set_property PROBES.FILE $proj/debug_test.runs/impl_1/debug_test.ltx [current_hw_device]

# Program
program_hw_devices [current_hw_device]

# Load probes
set_property PROBES.FILE $proj/debug_test.runs/impl_1/debug_test.ltx [current_hw_device]

# Set scan chain mask to enable ILA detection
set_property BSCAN_SWITCH_USER_MASK 0x02 [current_hw_device]
refresh_hw_device [current_hw_device]

set ila [lindex [get_hw_ilas -of_objects [current_hw_device]] 0]
puts "ILA: [get_property NAME $ila]"

# Arm and capture
set_property CONTROL.DATA_DEPTH 1024 $ila
run_hw_ila $ila
after 2000

# Read and save
puts "Saving..."
file mkdir D:/Users/u/tmac-zynq-fpga/vivado_integration/debug

if {[catch {
    set data [read_hw_ila_data $ila]
    write_hw_ila_data -force -csv_file D:/Users/u/tmac-zynq-fpga/vivado_integration/debug/ila_test.csv $data
    puts "Method 1 (read_hw_ila_data) OK"
}]} {
    puts "Method 1 failed, trying alternative..."
    if {[catch {
        write_hw_ila_data -force -csv_file D:/Users/u/tmac-zynq-fpga/vivado_integration/debug/ila_test.csv $ila
    }]} {
        puts "All methods failed"
    }
}
puts "=== Done ==="
