open_checkpoint D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper_placed.dcp

# Check the FIFO generator top for read data channel
set rdch_top [get_cells system_i/axi_hp/s00_couplers/auto_cc/inst/gen_clock_conv.gen_async_conv.asyncfifo_axi/inst_fifo_gen/gaxi_full_lite.gread_ch.grdch2.axi_rdch]
puts "RDCH top: $rdch_top Ref: [get_property REF_NAME $rdch_top]"

set rdata_pins [get_pins -of_objects $rdch_top -filter {NAME =~ "*rdata*" || NAME =~ "*din*" || NAME =~ "*dout*"}]
foreach p $rdata_pins {
    set left [get_property LEFT $p]
    set right [get_property RIGHT $p]
    set dir [get_property DIRECTION $p]
    puts "  $dir: $p => $left:$right"
}

# Also check the inst level
set inst [get_cells system_i/axi_hp/s00_couplers/auto_cc/inst]
set all_pins [get_pins -of_objects $inst]
foreach p $all_pins {
    set left [get_property LEFT $p]
    set right [get_property RIGHT $p]
    if {$left != "" && $right >= 32} {
        set dir [get_property DIRECTION $p]
        puts "  $dir: $p => $left:$right"
    }
}

exit
