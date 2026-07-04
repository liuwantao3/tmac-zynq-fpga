set proj_dir D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd
open_checkpoint $proj_dir.runs/impl_1/system_wrapper_physopt.dcp
route_design
write_checkpoint -force $proj_dir/post_route.dcp
report_timing_summary -file $proj_dir.runs/impl_1/system_wrapper_timing_summary_routed.rpt
report_utilization -file $proj_dir.runs/impl_1/system_wrapper_utilization_routed.rpt
write_bitstream -force $proj_dir.runs/impl_1/system_wrapper.bit
exit
