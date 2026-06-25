open_checkpoint D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper_placed.dcp

set inst {system_i/axi_hp}
set props [list_property [get_cells $inst]]
foreach p $props {
    set v [get_property $p [get_cells $inst]]
    if {[string match -nocase "*DATA*WIDTH*" $p] || [string match -nocase "*XBAR*" $p]} {
        puts "$p = $v"
    }
}

set cc {system_i/axi_hp/s00_couplers/auto_cc/inst}
puts "auto_cc ref: [get_property REF_NAME [get_cells $cc]]"
set pins [get_pins -of_objects [get_cells $cc] -filter {NAME =~ "*RDATA*" || NAME =~ "*rdata*"}]
foreach p $pins {
    if {[get_property DIRECTION $p] eq "IN"} {
        set left [get_property LEFT $p]
        set right [get_property RIGHT $p]
        puts "INPUT: $p => ${left}:${right}"
    }
}

set xbar [get_cells {system_i/axi_hp/gen_xbar.xbar}]
if {[llength $xbar] > 0} {
    set spins [get_pins -of_objects $xbar -filter {NAME =~ "*RDATA*"}]
    foreach p $spins {
        set d [get_property DIRECTION $p]
        set left [get_property LEFT $p]
        set right [get_property RIGHT $p]
        puts "XBAR ${d}: $p => ${left}:${right}"
    }
} else {
    puts "No xbar cell found"
    # Try to find crossbar cell
    set all [get_cells -regexp {.*xbar.*} -hierarchical]
    puts "Matching cells: $all"
}

exit
