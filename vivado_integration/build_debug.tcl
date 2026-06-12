# Build debug_test: synthesize + ILA + bitstream
set origin "D:/Users/u/tmac-zynq-fpga"
set proj_dir "$origin/vivado_integration/debug_proj"
set rtl_dir  "$origin/vivado_integration/rtl"

file mkdir $proj_dir
create_project -force debug_test $proj_dir -part xc7z010clg400-1

# Add source
add_files -fileset sources_1 [file normalize "$rtl_dir/debug_test.v"]
set_property top debug_test [current_fileset]
update_compile_order -fileset sources_1

# Synthesize
launch_runs synth_1 -jobs 8
wait_on_run synth_1

# Open synthesized design and create ILA
open_run synth_1

create_debug_core u_ila_0 ila
set_property C_DATA_DEPTH 1024 [get_debug_cores u_ila_0]
set_property C_TRIGIN_EN false [get_debug_cores u_ila_0]
set_property C_TRIGOUT_EN false [get_debug_cores u_ila_0]

# Connect clock to BUFG output (routable)
connect_debug_port u_ila_0/clk [get_nets {clk_IBUF_BUFG}]

# Probes
connect_debug_port u_ila_0/probe0 [get_nets {dbg_tick}]
puts "ILA created with 1 probe"
write_checkpoint -force "$proj_dir/post_synth.dcp"

# Implement
# Implement (with relaxed DRC for unconstrained test design)
set_property STEPS.write_bitstream.TCL.PRE {D:/Users/u/tmac-zynq-fpga/vivado_integration/write_bitstream_hook.tcl} [get_runs impl_1]
launch_runs impl_1 -to_step write_bitstream -jobs 8
wait_on_run impl_1
puts "=== BUILD DONE ==="
puts "Bitstream: $proj_dir/debug_test.runs/impl_1/debug_test.bit"
