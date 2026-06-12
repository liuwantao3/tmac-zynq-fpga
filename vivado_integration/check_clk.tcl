open_project D:/Users/u/tmac-zynq-fpga/vivado_integration/debug_proj/debug_test.xpr
open_run synth_1
set nets [get_nets -hierarchical -filter {NAME =~ "*clk*"}]
puts "Clock-related nets: [llength $nets]"
foreach n $nets {
    puts "  [get_property NAME $n] type=[get_property TYPE $n]"
}
