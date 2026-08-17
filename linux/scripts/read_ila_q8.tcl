# read_ila_q8.tcl - read back an already-captured ILA (data persists in FPGA
# until reprogrammed). Usage: vivado.bat -mode tcl -source linux/scripts/read_ila_q8.tcl
open_hw_manager
connect_hw_server
open_hw_target

set dev [current_hw_device]
set_property PROBES.FILE {D:/Users/u/tmac-zynq-fpga/linux/boot/system_wrapper_ila.ltx} $dev
refresh_hw_device $dev

set ila [get_hw_ilas -of_objects $dev]
puts "CORE_STATUS = [get_property STATUS.CORE_STATUS $ila]  SAMPLE_COUNT = [get_property STATUS.SAMPLE_COUNT $ila]"

upload_hw_ila_data $ila
set data [get_hw_ila_data -of_objects $ila]
write_hw_ila_data -force -csv_file D:/Users/u/tmac-zynq-fpga/linux/boot/ila_q8_trace.csv $data
puts "Trace written"

close_hw_target
disconnect_hw_server
close_hw_manager
exit