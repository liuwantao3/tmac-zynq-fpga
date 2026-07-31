# Boot Linux on MicroPhase Z7-Lite via JTAG, for use from the Vitis GUI XSCT console.
#
# Run in Vitis GUI:  Xilinx -> XSCT Console  ->  source {D:/Users/u/tmac-zynq-fpga/vitis_linux/scripts/boot_linux_jtag.tcl}
# Or standalone:     C:\Xilinx\Vivado\2023.1\bin\xsdb.bat D:/Users/u/tmac-zynq-fpga/vitis_linux/scripts/boot_linux_jtag.tcl
#
# Flow: bitstream -> ps7_init -> AFI -> load zImage/dtb/initramfs -> boot kernel (r0=0 r1=~0 r2=dtb pc=zImage)
# Kernel console goes to UART0 (broken CH340 on this board) AND/OR DCC.
# DCC capture via readjtaguart is capped (~544 B/session), so it only proves the kernel starts.
# Verified signals of a live kernel: PC advances out of zImage, CLK_CNT keeps incrementing.
#
# Requires a board power-cycle before ps7_init (PLL re-lock hang).

set BIT  {D:/Users/u/tmac-zynq-fpga/vitis_linux/workspace/z7_linux/hw/matmul_bd.bit}
set PS7  {D:/Users/u/tmac-zynq-fpga/vitis_linux/workspace/z7_linux/hw/ps7_init.tcl}
set ZIMG {D:/Users/u/tmac-zynq-fpga/vitis_linux/prebuilt/zImage}
set DTB  {D:/Users/u/tmac-zynq-fpga/vitis_linux/prebuilt/devicetree.dtb}
set RAMFS {D:/Users/u/tmac-zynq-fpga/vitis_linux/prebuilt/uramdisk.image.gz}

set KERNEL_LOAD 0x00108000
set DTB_LOAD    0x02000000
set RAMFS_LOAD  0x03000000

proc r32 {a} { set r [mrd $a 1]; if {[regexp {:\s+([0-9A-Fa-f]+)} $r d]} { return [expr "0x$d"] }; return -1 }
proc w32 {a v} { mwr -force $a $v }

puts "=== Vitis Linux project: JTAG boot (z7_linux platform) ==="

configparams force-mem-accesses 1
connect; after 5000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200

puts "1. Bitstream (from platform hw/)..."
fpga -file $BIT; after 2000
configparams force-mem-accesses 1
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200

puts "2. PS7 init (must be after power-cycle!)..."
source $PS7
ps7_mio_init_data_3_0; after 20
ps7_pll_init_data_3_0; after 20
ps7_clock_init_data_3_0; after 20
ps7_ddr_init_data_3_0; after 200
ps7_peripherals_init_data_3_0; after 20
ps7_post_config_3_0; after 200
puts "   PLL_STATUS=[format 0x%08x [r32 0xF800010C]]"

puts "3. AFI (HP0 for FPGA cores)..."
w32 0xF8000008 0x0000DF0D; after 10
w32 0xF8000910 0x0000000F; after 10
w32 0xF8008000 0x00000005; after 10
w32 0xF8008004 0x00000044; after 10
w32 0xF8008008 0x00000001; after 10
w32 0xF8000004 0x0000767B; after 20

puts "4. Loading kernel zImage to 0x[format %08x $KERNEL_LOAD]..."
dow -data $ZIMG $KERNEL_LOAD; after 300

puts "5. Loading DTB to 0x[format %08x $DTB_LOAD]..."
dow -data $DTB $DTB_LOAD; after 200

puts "6. Loading initramfs to 0x[format %08x $RAMFS_LOAD]..."
dow -data $RAMFS $RAMFS_LOAD; after 300

puts "7. DCC capture + boot..."
set dcc_fp [open "dcc_boot_output.txt" w]
readjtaguart -start -handle $dcc_fp
catch {stop}; after 200
targets -set -filter {name =~ "*Cortex-A9*#0*"}; after 200
rwr r0 0
rwr r1 0xFFFFFFFF
rwr r2 $DTB_LOAD
rwr pc $KERNEL_LOAD
rwr cpsr 0x00000013
after 100
con

puts "   Kernel booting. Waiting 30s..."
after 30000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 100
catch {stop}; after 200
catch {rdreg pc} msg; puts "   pc = $msg"
readjtaguart -stop
close $dcc_fp

puts "\n=== DCC Console Output (first 544 B) ==="
set fp [open "dcc_boot_output.txt" r]
puts [read $fp]
close $fp
puts "\n=== Done ==="
