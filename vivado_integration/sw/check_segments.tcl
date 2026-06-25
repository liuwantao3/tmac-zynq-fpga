open_project D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.xpr
current_project [get_projects matmul_bd]
set bd_obj [get_bd_designs system]
puts "BD object: $bd_obj"
if {$bd_obj != ""} {
    current_bd_design $bd_obj
    set gp0_seg [get_bd_addr_segs -of_objects [get_bd_intf_pins axi_hp_top/S_AXI]]
    puts "GP0 slave seg: $gp0_seg"
    if {$gp0_seg != ""} {
        set gp0_segs [get_bd_addr_segs -quiet -of_objects [get_bd_addr_spaces ps7/Data]]
        puts "GP0 control: $gp0_segs"
    } else {
        puts "GP0 slave seg is EMPTY"
    }
    set hp0_seg [get_bd_addr_segs -of_objects [get_bd_intf_pins ps7/S_AXI_HP0]]
    puts "HP0 slave seg: $hp0_seg"
    if {$hp0_seg != ""} {
        set hp0_segs [get_bd_addr_segs -quiet -of_objects [get_bd_addr_spaces axi_hp_top/M_AXI_HP]]
        puts "HP0 DDR: $hp0_segs"
    } else {
        puts "HP0 slave seg is EMPTY"
    }
} else {
    puts "ERROR: No BD found"
}
close_project
exit
