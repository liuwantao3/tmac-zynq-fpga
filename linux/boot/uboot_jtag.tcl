# Load U-Boot via JTAG and start it
# C:\Xilinx\Vivado\2023.1\bin\xsdb.bat linux/boot/uboot_jtag.tcl

set BIT {D:/Users/u/tmac-zynq-fpga/linux/boot/system_wrapper.bit}
set PS7 {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.gen/sources_1/bd/system/ip/system_ps7_0/ps7_init.tcl}
set UBOOT {D:/Users/u/tmac-zynq-fpga/linux/boot/u-boot.elf}
set ZIMG {D:/Users/u/tmac-zynq-fpga/linux/boot/uImage}
set DTB  {D:/Users/u/tmac-zynq-fpga/linux/boot/devicetree.dtb}
set RAMFS {D:/Users/u/tmac-zynq-fpga/linux/boot/uramdisk.image.gz}
set TMAC {D:/Users/u/tmac-zynq-fpga/linux/boot/tmac}
set MODEL {D:/Users/u/tmac-zynq-fpga/models/model.tmac}

proc r32 {a} { set r [mrd $a 1]; regexp {:\s+([0-9A-Fa-f]+)} $r d; return [expr "0x$d"] }
proc w32 {a v} { mwr -force $a $v }

puts "=== JTAG U-Boot Load ==="
configparams force-mem-accesses 1
connect; after 5000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200

puts "1. Bitstream..."
fpga -file $BIT; after 2000
configparams force-mem-accesses 1
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200

puts "2. PS7 init..."
source $PS7
ps7_mio_init_data_3_0; after 20
ps7_pll_init_data_3_0; after 20
ps7_clock_init_data_3_0; after 20
ps7_ddr_init_data_3_0; after 200
ps7_peripherals_init_data_3_0; after 20
ps7_post_config_3_0; after 200

puts "3. AFI..."
w32 0xF8000008 0x0000DF0D; after 10
w32 0xF8000910 0x0000000F; after 10
w32 0xF8008000 0x00000005; after 10
w32 0xF8008004 0x00000044; after 10
w32 0xF8008008 0x00000001; after 10
w32 0xF8000004 0x0000767B; after 20

puts "4. Load kernel+DTB+initrd to DDR for U-Boot..."
dow -data $ZIMG 0x00108000; after 200
dow -data $DTB 0x02000000; after 200
dow -data $RAMFS 0x03000000; after 200
dow -data $TMAC 0x01000000; after 200
dow -data $MODEL 0x00200000; after 500

puts "5. Load and start U-Boot..."
dow $UBOOT; after 200

puts ""
puts "=== U-Boot starting ==="
puts "Watch UART console at 115200 baud!"
puts "When you see U-Boot prompt, you can type:"
puts "  mmc rescan"
puts "  fatls mmc 0"
puts "  bootm 0x00108000 0x03000000 0x02000000"
puts "  (kernel_im_addr initrd_addr dtb_addr)"
puts "Starting U-Boot in 3 seconds..."
after 3000
con
puts "U-Boot is running. XSDB will halt in 60s."
after 60000
catch {stop}
puts "=== Done ==="
exit
