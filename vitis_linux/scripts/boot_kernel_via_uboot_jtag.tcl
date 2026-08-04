# Boot Linux on MicroPhase Z7-Lite via JTAG **through U-Boot**.
#
# Instead of the fragile manual kernel hand-boot (r0/r1/r2/pc/cpsr), load the
# proven-good U-Boot over JTAG and let U-Boot's distro boot auto-run boot.scr
# from the SD card (FAT32). U-Boot self-initializes the CPU (immune to a wedged
# core left by a crashed boot), then bootm does the MMU/cache teardown, DTB
# fixup and initrd setup that the kernel requires.
#
# SD card must be inserted (F: SD_BOOT, FAT32, 5 files: BOOT.BIN uImage
# devicetree.dtb uramdisk.image.gz boot.scr).
#
# Run:  C:\Xilinx\Vivado\2023.1\bin\xsdb.bat D:/Users/u/tmac-zynq-fpga/vitis_linux/scripts/boot_kernel_via_uboot_jtag.tcl

set ELF  {D:/Users/u/tmac-zynq-fpga/linux/boot/u-boot.elf}
set PS7  {D:/Users/u/tmac-zynq-fpga/vitis_linux/workspace/z7_linux/hw/ps7_init.tcl}

proc r32 {a} {
    set r [mrd $a 1]
    set r [string trim $r]
    if {[regexp {([0-9A-Fa-f]+)$} $r v]} { return [expr "0x$v"] }
    return -1
}

puts "=== Boot Linux via JTAG-loaded U-Boot (SD auto-boot) ==="
configparams force-mem-accesses 1
connect; after 5000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200

set pll [r32 0xF800010C]
puts "1. PLL_STATUS=[format 0x%08x $pll]"
if {($pll & 0x7) == 0x7} {
    puts "   PLLs locked -> skipping ps7_init"
} else {
    puts "2. PS7 init..."
    source $PS7
    ps7_mio_init_data_3_0; after 20
    ps7_pll_init_data_3_0; after 20
    ps7_clock_init_data_3_0; after 20
    ps7_ddr_init_data_3_0; after 200
    ps7_peripherals_init_data_3_0; after 20
    ps7_post_config_3_0; after 200
    puts "   PLL_STATUS=[format 0x%08x [r32 0xF800010C]]"
}

puts "3. Load u-boot.elf over JTAG..."
catch {stop}; after 200
targets -set -filter {name =~ "*Cortex-A9*#0*"}; after 200
dow $ELF; after 500

puts "4. con -> U-Boot runs, distro boot auto-runs boot.scr from SD..."
puts "   Watch the CH340 terminal: U-Boot banner, MMC detect, boot.scr"
puts "   fatload uImage/dtb/uramdisk, bootm -> Linux kernel. Wait 60 s..."
con
after 60000

puts "5. State after 60 s:"
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200
catch {stop}; after 300
foreach reg {pc sp lr cpsr} {
    catch {rrd $reg} msg; puts "   $msg"
}
puts "=== Done. Kernel console log was on USB-UART0 (see PuTTY). ==="
