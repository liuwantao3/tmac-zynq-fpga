# Boot Linux on MicroPhase Z7-Lite via JTAG, for use from the Vitis GUI XSCT console.
#
# Run in Vitis GUI:  Xilinx -> XSCT Console  ->  source {D:/Users/u/tmac-zynq-fpga/vitis_linux/scripts/boot_linux_jtag.tcl}
# Or standalone:     C:\Xilinx\Vivado\2023.1\bin\xsdb.bat D:/Users/u/tmac-zynq-fpga/vitis_linux/scripts/boot_linux_jtag.tcl
#
# Flow: connect -> bitstream -> ps7_init (always, after power-cycle) -> AFI ->
#       load zImage/dtb/initramfs -> boot kernel (r0=0 r1=~0 r2=dtb pc=zImage)
#
# NOTES:
# - A physical power-cycle is REQUIRED before ps7_init (PLL re-lock hang), and
#   also clears a wedged CPU / DAP AP-error state from a prior crashed run.
# - ps7_init runs UNCONDITIONALLY: after a true power-cycle the PLL_STATUS can
#   transiently read locked before ps7_init programs it, so skipping it leaves
#   the SLCR/AFI uninitialized and AFI writes fault (AP transaction timeout,
#   DAP status 0x30000021). Running it from a cold state is the documented flow.
# - After a cold boot + ps7_init the MMU is OFF, so dow/mrd to DDR work directly.
#   Do NOT add an MMU teardown here: on a warm (non-power-cycled) re-run the CPU
#   may sit in a corrupt Abort state (dow -> "MMU 1st level translation table
#   walk external abort") — the fix for that is another power-cycle, not a stub.
#
# This is a U-Boot-less hand boot: the kernel finds the initramfs (raw gzipped
# cpio, initramfs.cpio.gz) via the DTB /chosen/linux,initrd-start/end properties
# baked into devicetree-jtag.dtb by linux/patch_dtb_initrd.py (see linux/README.md).
# Kernel console is on the USB-UART0 (CH340): the DTB /chosen/bootargs is empty,
# so the kernel uses the baked-in CONFIG_CMDLINE "earlycon console=ttyPS0,115200 ...".
# Open a 115200 8N1 terminal (PuTTY) on the CH340 COM port to see kernel/initramfs
# boot output.

set BIT  {D:/Users/u/tmac-zynq-fpga/vitis_linux/workspace/z7_linux/hw/matmul_bd.bit}
set PS7  {D:/Users/u/tmac-zynq-fpga/vitis_linux/workspace/z7_linux/hw/ps7_init.tcl}
set ZIMG {D:/Users/u/tmac-zynq-fpga/vitis_linux/prebuilt/zImage}
set DTB  {D:/Users/u/tmac-zynq-fpga/vitis_linux/prebuilt/devicetree-jtag.dtb}
set RAMFS {D:/Users/u/tmac-zynq-fpga/vitis_linux/prebuilt/initramfs.cpio.gz}

set KERNEL_LOAD 0x00108000
set DTB_LOAD    0x02000000
set RAMFS_LOAD  0x03000000

proc r32 {a} { set r [mrd $a 1]; if {[regexp {:\s+([0-9A-Fa-f]+)} $r -> d]} { return [expr "0x$d"] }; return -1 }
proc w32 {a v} { mwr -force $a $v }

puts "=== Vitis Linux project: JTAG boot (z7_linux platform) ==="

configparams force-mem-accesses 1
connect; after 5000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200
# Halt both cores so neither a running core nor a wedged core interferes with
# core-mediated dow/mrd accesses (a leftover-MMU core causes "MMU 1st level
# translation table walk external abort" on dow).
catch {stop}; after 100
catch {targets -set -filter {name =~ "*Cortex-A9*#1*"}}; after 100
catch {stop}; after 100
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 100

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

puts "6. Loading DTB to 0x[format %08x $DTB_LOAD]..."
dow -data $DTB $DTB_LOAD; after 200

puts "7. Loading initramfs (raw gzip cpio) to 0x[format %08x $RAMFS_LOAD]..."
puts "   Kernel locates it via DTB /chosen/linux,initrd-start/end..."
dow -data $RAMFS $RAMFS_LOAD; after 300

puts "8. Booting kernel (console on USB-UART0, 115200 8N1)..."
puts "   Attach PuTTY to the CH340 COM port to watch kernel/initramfs output."
catch {stop}; after 200
targets -set -filter {name =~ "*Cortex-A9*#0*"}; after 200
catch {rwr cpsr 0x000000D3}; after 100    ;# SVC, IRQ+FIQ disabled
catch {rwr r0 0}; after 50
catch {rwr r1 0xFFFFFFFF}; after 50
catch {rwr r2 $DTB_LOAD}; after 50
catch {rwr r3 0}; after 50
catch {rwr r4 0}; after 50
catch {rwr r5 0}; after 50
catch {rwr r6 0}; after 50
catch {rwr r7 0}; after 50
catch {rwr r8 0}; after 50
catch {rwr r9 0}; after 50
catch {rwr r10 0}; after 50
catch {rwr r11 0}; after 50
catch {rwr r12 0}; after 50
catch {rwr sp 0xFFFFFFFF}; after 50
catch {rwr lr 0}; after 50
catch {rwr pc $KERNEL_LOAD}; after 50
after 100
con

puts "   Kernel booting. Waiting 30s..."
after 30000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 100
catch {stop}; after 200
catch {rrd pc} msg; puts "   pc = $msg"

puts "\n=== Done. Kernel console log was on USB-UART0 (see PuTTY). ==="
