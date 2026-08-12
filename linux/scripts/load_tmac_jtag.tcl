# Fast-iteration JTAG loader: write the recompiled tmac-debug binary into
# a DDR scratch area so the board-side shell can pick it up without an SD
# card swap.
#
# Workflow (board stays booted, SD card stays in):
#   1. PC: arm-linux-gnueabihf-gcc -static -O2 -o /tmp/tmac-debug tmac_linux.c -lm
#   2. PC: xsdb.bat D:/Users/u/tmac-zynq-fpga/linux/scripts/load_tmac_jtag.tcl
#   3. Board shell:
#        dd if=/dev/mem of=/root/tmac-debug bs=4096 skip=519680 count=115
#        chmod +x /root/tmac-debug
#        /root/tmac-debug --selftest
#
# Binary is written to physical DDR 0x1F100000 (above the FPGA window at
# 0x1F000000), 115 blocks x 4096 = 471040 bytes covers the ~468 KB binary.
# Adjust count= if the binary size changes significantly.

set BIN  {D:/Users/u/tmac-zynq-fpga/linux/boot/tmac-debug}
set STAGING 0x1F100000

proc r32 {a} { set r [mrd $a 1]; if {[regexp {:\s+([0-9A-Fa-f]+)} $r -> d]} { return [expr "0x$d"] }; return -1 }

puts "=== JTAG load: write tmac-debug to DDR 0x[format %08x $STAGING] ==="

configparams force-mem-accesses 1
connect; after 2000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200

set file_size [file size $BIN]
set blocks [expr ($file_size + 4095) / 4096]
puts "  binary: $BIN  size=$file_size  blocks=$blocks"

# Halt CPU briefly to write memory (Linux resumes fine)
catch {stop}; after 200
targets -set -filter {name =~ "*Cortex-A9*#0*"}; after 200

puts "  dow -data $BIN -> 0x[format %08x $STAGING]..."
dow -data $BIN $STAGING; after 500

puts "  verifying first 8 bytes..."
set v0 [r32 $STAGING]
set v1 [r32 [expr $STAGING + 4]]
puts "  mem[0x[format %08x $STAGING]] = 0x[format %08x $v0] 0x[format %08x $v1]"
puts "  (should be ELF magic 0x464C457F = 0x7F454C46)"

puts "  con -> resuming Linux..."
con; after 500
set pc [r32 0]; puts "  pc = 0x[format %08x $pc]"

puts ""
puts "=== Binary staged at 0x[format %08X $STAGING]  ($blocks blocks) ==="
puts "Run on the board UART:"
puts "  dd if=/dev/mem of=/root/tmac-debug bs=4096 skip=[expr $STAGING / 4096] count=$blocks"
puts "  chmod +x /root/tmac-debug"
puts "  /root/tmac-debug --selftest   (or --compare, --cpu, etc.)"
