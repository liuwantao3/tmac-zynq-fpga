# run_serial.tcl — headless serial-console test for MicroPhase Z7-Lite.
# Programs the FPGA, runs ps7_init, loads the serial test ELF and runs it.
# Console output appears on the USB-UART terminal (115200 8N1) — open a
# terminal (e.g. PuTTY/RealTerm) on the CH340 COM port before running.
#
# Power-cycle the board before running (ps7_init PLL re-lock hang)!
#
# C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vitis_bm\scripts\run_serial.tcl

# Bitstream + ps7_init come from the generated platform (build.tcl must have
# run once). Equivalent to the Vivado build outputs under vivado_integration/proj_bd/.
set BITSTREAM {D:/Users/u/tmac-zynq-fpga/vitis_bm/z7_bm/hw/matmul_bd.bit}
set PS7_INIT   {D:/Users/u/tmac-zynq-fpga/vitis_bm/z7_bm/hw/ps7_init.tcl}
set ELF_PATH   {D:/Users/u/tmac-zynq-fpga/vivado_integration/sw/uart_test.elf}

set GP0_BASE 0x43C00000

proc read32 {addr} {
    set r [mrd $addr 1]
    if {[regexp {:\s+([0-9A-Fa-f]+)} $r full data]} { return [expr "0x$data"] }
    return -1
}
proc gp0_read {reg} {
    global GP0_BASE
    return [read32 [format 0x%08x [expr $GP0_BASE + $reg]]]
}

puts "=============================================="
puts "  UART Serial Console Test (uart_test.elf)"
puts "=============================================="

configparams force-mem-accesses 1
connect
after 15000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}
after 200
puts "Target: [targets]"

# ===== Load bitstream =====
puts "Loading bitstream..."
fpga -file $BITSTREAM
after 2000
configparams force-mem-accesses 1
targets -set -filter {name =~ "*Cortex-A9*#0*"}
after 200

# ===== PS7 init (individual steps) =====
puts "PS7 init..."
source $PS7_INIT
after 200
ps7_mio_init_data_3_0
after 50
ps7_pll_init_data_3_0
after 50
ps7_clock_init_data_3_0
after 50
ps7_ddr_init_data_3_0
after 200
ps7_peripherals_init_data_3_0
after 50
ps7_post_config_3_0
after 200

set pll_status [read32 0xF800010C]
puts "  PLL_STATUS=[format 0x%08x $pll_status]"
if {[expr ($pll_status & 7)] != 7} { puts "  ERROR: PLLs not locked!"; exit 1 }

# ===== AFI config =====
puts "Configuring AFI..."
mwr -force 0xF8000008 0x0000DF0D   ;# unlock SLCR
after 20
mwr -force 0xF8000910 0x0000000F   ;# LVL_SHFTR_EN
after 20
mwr -force 0xF8008000 0x00000005   ;# AFI0_CTRL
after 20
mwr -force 0xF8008004 0x00000044   ;# AFI0_PART
after 20
mwr -force 0xF8008008 0x00000001   ;# AFI0_WRCHAN
after 20
mwr -force 0xF8000004 0x0000767B   ;# lock SLCR
after 50

# ===== Verify PL clock =====
set clk_cnt [gp0_read 0x2C]
puts "  CLK_CNT = [format 0x%08X $clk_cnt] (should be non-zero)"

# ===== Load + run =====
puts "Loading ELF..."
dow $ELF_PATH
after 200
puts "Running (serial output on USB-UART at 115200 8N1)..."
con
after 3000

# Confirm the app is alive: CLK_CNT must advance
set c1 [gp0_read 0x2C]
after 500
set c2 [gp0_read 0x2C]
puts "  CLK_CNT advancing: 0x[format %08X $c1] -> 0x[format %08X $c2]"
if {$c1 == $c2} {
    puts "  WARNING: CLK_CNT not advancing — app/PL not running?"
} else {
    puts "  App is running. Check the USB-UART terminal for banner + ticks."
}

stop
puts "Done."
exit
