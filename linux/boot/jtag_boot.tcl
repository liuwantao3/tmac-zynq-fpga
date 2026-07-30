# Minimal JTAG Linux boot — proves kernel starts (watch UART at 115200 baud)
# C:\Xilinx\Vivado\2023.1\bin\xsdb.bat linux/boot/jtag_boot.tcl
set BIT  {D:/Users/u/tmac-zynq-fpga/linux/boot/system_wrapper.bit}
set PS7  {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.gen/sources_1/bd/system/ip/system_ps7_0/ps7_init.tcl}
set ZIMG {D:/Users/u/tmac-zynq-fpga/linux/boot/zImage}
set DTB  {D:/Users/u/tmac-zynq-fpga/linux/boot/devicetree.dtb}
proc r32 {a} { set r [mrd $a 1]; if {[regexp {:\s+([0-9A-Fa-f]+)} $r d]} { return [expr "0x$d"] }; return -1 }
proc w32 {a v} { mwr -force $a $v }

puts "=== JTAG Linux Boot ==="

configparams force-mem-accesses 1
connect; after 5000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200

# Bitstream
puts "1. Bitstream..."
fpga -file $BIT; after 2000
configparams force-mem-accesses 1
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200

# PS7 init
puts "2. PS7 init..."
source $PS7
ps7_mio_init_data_3_0; after 20
ps7_pll_init_data_3_0; after 20
ps7_clock_init_data_3_0; after 20
ps7_ddr_init_data_3_0; after 200
ps7_peripherals_init_data_3_0; after 20
ps7_post_config_3_0; after 200

# AFI
puts "3. AFI..."
w32 0xF8000008 0x0000DF0D; after 10
w32 0xF8000910 0x0000000F; after 10
w32 0xF8008000 0x00000005; after 10
w32 0xF8008004 0x00000044; after 10
w32 0xF8008008 0x00000001; after 10
w32 0xF8000004 0x0000767B; after 20

# Load kernel, DTB, initrd
puts "4. Loading kernel to 0x00108000..."
dow -data $ZIMG 0x00108000; after 200
puts "5. Loading DTB to 0x02000000..."
dow -data $DTB 0x02000000; after 200
# Boot
puts "7. Starting kernel (r0=0 r1=0xFFFFFFFF r2=0x02000000 PC=0x00108000)..."
puts "   Watch UART at 115200 baud!"
reg R0 0
reg R1 0xFFFFFFFF
reg R2 0x02000000
reg PC 0x00108000
reg CPSR 0x00000013
con
puts "   Kernel running! XSDB will halt CPU in 30s."
after 30000
catch {stop}
puts "=== Done ==="
exit
