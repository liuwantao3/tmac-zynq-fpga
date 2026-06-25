set xsa "D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.xsa"
set outdir "D:/Users/u/tmac-zynq-fpga/vivado_integration/sw/hsi_output"
file mkdir $outdir

hsi open_hw_design $xsa

# Check write_ps_configuration usage
puts "=== write_ps_configuration help ==="
set help [hsi write_ps_configuration -help]
puts $help

# Try without options
puts "=== Generating with default options ==="
set old_cwd [pwd]
cd $outdir
if {[catch {
    hsi write_ps_configuration
    puts "Success"
    puts "Files: [glob -directory $outdir *]"
} err]} {
    puts "Error: $err"
}
cd $old_cwd

hsi close_hw_design
exit
