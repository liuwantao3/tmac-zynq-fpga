# Minimal: program, arm ILA, wait, save
open_hw_manager; connect_hw_server; open_hw_target
set fpga [lindex [get_hw_devices] end]; current_hw_device $fpga
set_property PROGRAM.FILE D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.bit [current_hw_device]
program_hw_devices [current_hw_device]
set_property PROBES.FILE D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.ltx [current_hw_device]
refresh_hw_device [current_hw_device]
set ila [lindex [get_hw_ilas -of_objects [current_hw_device]] 0]
run_hw_ila $ila
puts "=== ILA armed. Now run hp_test in another terminal ==="
after 30000
file mkdir D:/Users/u/tmac-zynq-fpga/vivado_integration/debug
write_hw_ila_data -force -csv_file D:/Users/u/tmac-zynq-fpga/vivado_integration/debug/ila_hp.csv $ila
puts "=== Data saved to debug/ila_hp.csv ==="
