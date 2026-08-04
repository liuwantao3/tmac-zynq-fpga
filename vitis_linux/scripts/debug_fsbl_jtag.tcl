# Debug the FSBL via JTAG on MicroPhase Z7-Lite.
#
# Loads the actual FSBL ELF (linux/boot/fsbl.elf) into OCM (linked at 0x0,
# OCM alias; .stack at 0xFFFF0000, initial SP 0xFFFF6000), sets registers as
# BootROM would, and runs it. FSBL does its own ps7_init (PLL/DDR/MIO/UART0)
# and prints its banner on UART0 (CH340, 115200 8N1) — no XSDB ps7_init needed.
#
# Note: FSBL in JTAG boot mode (BOOT_MODE=0) initializes the PS and then
# hands back / idles (no bitstream, no U-Boot load, since BootROM wouldn't
# provide those in JTAG mode). What we verify here: FSBL binary runs, PLL/DDR
# init completes, UART0 banner prints — i.e. FSBL code itself is healthy.
#
# Requires a fresh power-cycle before running (PLL re-lock hang rule).
# Board should be in any boot mode; we drive everything over JTAG.

set ELF  {D:/Users/u/tmac-zynq-fpga/linux/boot/fsbl.elf}
set ENTRY 0x0               ;# _vector_table (reset vec b _boot@0x12c b _start@0x48bc)
set SP 0xFFFF6000           ;# initial stack pointer (from _start)

puts "=== FSBL JTAG debug (Z7-Lite) ==="

configparams force-mem-accesses 1
connect; after 5000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200

puts "1. Downloading FSBL ELF to OCM..."
dow $ELF; after 500

puts "2. Setting BootROM handoff registers (BootMode=JTAG=0)..."
catch {stop}; after 200
rwr r0 0            ;# BootMode = JTAG
rwr r1 0            ;# reserved
rwr r2 0            ;# reserved
rwr sp 0xFFFF6000
rwr lr 0x0000013C   ;# EndlessLoop0 (safe return if FSBL returns)
rwr pc $ENTRY
rwr cpsr 0x00000013
after 100
puts "   BootModeReg (0xF800025C) = [mrd 0xF800025C 1]"

puts "3. Running FSBL (watch UART0 / CH340 for the banner)..."
con

puts "   FSBL running. Waiting 15s..."
after 15000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 100
catch {stop}; after 200

puts "4. State after run:"
catch {rrd pc} msg; puts "   pc = $msg"
catch {rrd lr} msg; puts "   lr = $msg"
catch {rrd cpsr} msg; puts "   cpsr = $msg"
set pll [mrd 0xF800010C 1]
puts "   PLL_STATUS (0xF800010C): $pll"
set ddr [mrd 0xF8000B74 1]
puts "   DDR (0xF8000B74): $ddr"

puts "\n=== Done. FSBL banner should have appeared on the CH340 UART0. ==="
