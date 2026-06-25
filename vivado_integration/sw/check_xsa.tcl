puts "XSCT check"
set xsa "D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.xsa"
if {[file exists $xsa]} {
    set fsize [file size $xsa]
    puts "XSA exists: $fsize bytes"
} else {
    puts "XSA not found"
}
puts "Done"
exit
