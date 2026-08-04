# Debug the FSBL SD path via JTAG on MicroPhase Z7-Lite.
#
# Reproduces the SD boot path under JTAG control:
#   - BOOT_MODE_REG (0xF800025C) is read-only and reads 0x05 (SD) on this
#     board, so FSBL naturally takes the SD path (InitSD + read BOOT.BIN
#     partitions + PCAP bitstream + load U-Boot),
#   - installs hardware breakpoints at FsblFallback and FsblHookFallback,
#   - runs and, on halt/timeout, dumps PC + regs + REBOOT_STATUS + DDR.
#
# If the partition-load failure reproduces, PC should stop at 0x5BC
# (FsblHookFallback @0x5AC + its while(1)). Clean UART0 banner while running
# isolates whether the garbling is SD-boot-specific vs the FSBL binary.
#
# Requires: fresh power-cycle (PLL re-lock rule) and SD card with BOOT.BIN
# inserted (FSBL in SD mode reads the card itself). No XSDB ps7_init needed —
# FSBL performs its own PS init.

set ELF  {D:/Users/u/tmac-zynq-fpga/linux/boot/fsbl.elf}
set ENTRY 0x0               ;# _vector_table (reset vec b _boot@0x12c b _start@0x48bc)
set SP 0xFFFF6000
set BOOTMODE 0xF800025C

set aHookFallback  0x5AC    ;# FsblHookFallback (print + while(1))
set aFsblFallback  0x16A4   ;# FsblFallback (hooks + FailStatus + while(1))
set aLoadBootImage 0x0D98   ;# LoadBootImage entry
set aPartitionMove 0x0A58   ;# PartitionMove entry

puts "=== FSBL SD-mode JTAG debug (Z7-Lite) ==="

configparams force-mem-accesses 1
connect; after 5000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200

puts "1. Reading BOOT_MODE_REG (0xF800025C)..."
puts "   BootModeReg = [mrd 0xF800025C 1]   (5=SD, read-only)"
after 100

puts "2. Downloading FSBL ELF to OCM..."
dow $ELF; after 500

puts "3. Setting BootROM handoff registers..."
catch {stop}; after 200
rwr r0 0            ;# BootMode arg (FSBL re-reads BOOT_MODE_REG anyway)
rwr r1 0
rwr r2 0
rwr sp 0xFFFF6000
rwr lr 0x0000013C   ;# EndlessLoop0
rwr pc $ENTRY
rwr cpsr 0x00000013
after 100

puts "4. Installing hardware breakpoints..."
bpadd -addr $aHookFallback -type hw
bpadd -addr $aFsblFallback -type hw
bpadd -addr $aLoadBootImage -type hw
bpadd -addr $aPartitionMove -type hw
puts "   at FsblHookFallback=0x5AC FsblFallback=0x16A4 LoadBootImage=0xD98 PartitionMove=0xA58"
after 200

puts "5. Running FSBL in SD mode (watch UART0 / CH340)..."
con

puts "   Running. Waiting 30s for breakpoint halt / completion..."
after 30000
catch {stop}; after 300

puts "\n6. State after run:"
catch {rrd pc} msg; puts "   pc     = $msg"
catch {rrd lr} msg; puts "   lr     = $msg"
catch {rrd r0} msg; puts "   r0     = $msg"
catch {rrd r3} msg; puts "   r3     = $msg"
catch {rrd sp} msg; puts "   sp     = $msg"
set rb [mrd 0xF8000258 1]
puts "   REBOOT_STATUS (0xF8000258): $rb"
set pll [mrd 0xF800010C 1]
puts "   PLL_STATUS (0xF800010C):    $pll"
set devcfg [mrd 0xF8007000 1]
puts "   DEVCFG_CTRL (0xF8007000):   $devcfg"

puts "\n7. Reading back what FSBL wrote to DDR temp (0x04000000)..."
for {set j 0} {$j < 4} {incr j} {
    set a [format 0x%08X [expr {0x04000000 + $j*4}]]
    puts "   $a = [mrd $a 1]"
}

puts "\n8. Clearing breakpoints..."
bpremove -all
after 100

puts "\n=== Done. If pc==0x5BC the SD partition-load failure is reproduced. ==="
puts "=== FSBL banner should have appeared on CH340 (clean under JTAG?). ==="
