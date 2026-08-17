# Improve timing: reopen post-synth, place+phys_opt+route with aggressive directives
# Run: vivado.bat -mode batch -source improve_timing.tcl  (from vivado_integration/)
set origin "D:/Users/u/tmac-zynq-fpga"
set proj_dir "$origin/vivado_integration/proj_bd"
open_project "$proj_dir/matmul_bd.xpr"

set_property STEPS.SYNTH_DESIGN.ARGS.FLATTEN_HIERARCHY full [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.FSM_EXTRACTION one_hot [get_runs synth_1]
set_property STEPS.SYNTH_DESIGN.ARGS.RETIMING true [get_runs synth_1]

# Use a more aggressive placement/routing strategy
set_property strategy Performance_Explore [get_runs impl_1]
set_property STEPS.PLACE_DESIGN.ARGS.DIRECTIVE ExtraTimingOpt [get_runs impl_1]
set_property STEPS.ROUTE_DESIGN.ARGS.DIRECTIVE AggressiveExplore [get_runs impl_1]
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]

reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs 12
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    puts "ERROR: impl_1 did not complete"
    exit 1
}

open_run impl_1
write_checkpoint -force "$proj_dir/post_impl.dcp"
write_hw_platform -fixed -include_bit -force "$proj_dir/matmul_bd.xsa"

# Re-patch ps7_init.tcl (same as build_bd.tcl)
set ps7_init_file [file normalize "$proj_dir/matmul_bd.gen/sources_1/bd/system/ip/system_ps7_0/ps7_init.tcl"]
if {[file exists $ps7_init_file]} {
    set fd [open $ps7_init_file r]
    set content [read $fd]
    close $fd
    regsub -all {mask_write 0XF8000170 0x03F03F30 0x00400400} $content {mask_write 0XF8000170 0x03F83FB0 0x00480480} content
    regsub -all {mask_write 0XF8000180 0x03F03F30 0x00400400} $content {mask_write 0XF8000180 0x03F83FB0 0x00480480} content
    set fd [open $ps7_init_file w]
    puts -nonewline $fd $content
    close $fd
    puts "ps7_init.tcl patched: CLK0_EN + CLK1_EN added"
} else {
    puts "WARNING: ps7_init.tcl not found at $ps7_init_file"
}
puts "=== DONE (improve_timing) ==="