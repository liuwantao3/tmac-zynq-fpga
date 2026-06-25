puts "=== Generate ps7_init from XSA ==="
set xsa "D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.xsa"
set outdir "D:/Users/u/tmac-zynq-fpga/vivado_integration/sw/hsi_output"

file mkdir $outdir

hsi open_hw_design $xsa

# Check PS7 properties
set ps7 [hsi get_cells -hier ps7]
puts "PS7 cell: $ps7"
puts "PCW_USE_S_AXI_HP0 = [hsi get_property CONFIG.PCW_USE_S_AXI_HP0 $ps7]"
puts "PCW_S_AXI_HP0_DATA_WIDTH = [hsi get_property CONFIG.PCW_S_AXI_HP0_DATA_WIDTH $ps7]"

# Generate PS7 configuration files
set old_cwd [pwd]
cd $outdir
hsi write_ps_configuration -name ps7_init
cd $old_cwd

puts "Generated files in $outdir:"
foreach f [glob -nocomplain -directory $outdir *] {
    puts "  $f"
}

puts "=== Done ==="
exit
