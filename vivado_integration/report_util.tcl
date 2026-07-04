open_checkpoint D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper_placed.dcp
# Find cells with wmem in name
set wmem_cells [get_cells -hierarchical -regexp {.*wmem.*}]
puts "wmem cells: [llength $wmem_cells]"
foreach c $wmem_cells {
    puts "  [get_property NAME $c] : [get_property REF_NAME $c]"
}
# Also check what the u_q8 instances are
set uq8_cells [get_cells -hierarchical -regexp {.*u_q8.*}]
puts "u_q8 sub-cells: [llength $uq8_cells]"
# Look for RAMB in any property
set all_refs [get_cells -hierarchical]
foreach c $all_refs {
    set ref [get_property REF_NAME $c]
    if {[string match "*RAMB*" $ref] || [string match "*ramb*" $ref]} {
        puts "Found: [get_property NAME $c] -> $ref"
    }
}
exit
