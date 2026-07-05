# FPGA Core Test Wrapper Runner
# Loads test_fpga_cores.elf onto Zynq 7010, runs it, reports results.
# Power-cycle the board before running!
# C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_test_fpga_cores.tcl

set BITSTREAM  {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.bit}
set PS7_INIT   {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.gen/sources_1/bd/system/ip/system_ps7_0/ps7_init.tcl}
set ELF_PATH   {D:/Users/u/tmac-zynq-fpga/vivado_integration/sw/test_fpga_cores.elf}

set OUTPUT_BUF 0x1F000000
set GP0_BASE   0x43C00000

proc read32 {addr} {
    set r [mrd $addr 1]
    if {[regexp {:\s+([0-9A-Fa-f]+)} $r full data]} { return [expr "0x$data"] }
    return -1
}

proc gp0_read {reg} {
    global GP0_BASE
    return [read32 [format 0x%08x [expr $GP0_BASE + $reg]]]
}

# ===== Connect =====
puts "=============================================="
puts "  FPGA Core Test Wrapper"
puts "=============================================="
puts ""

configparams force-mem-accesses 1
connect
after 15000

# Select ARM core for memory access (same pattern as comprehensive test)
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}
after 200
puts "Target selected: [targets]"

# If DAP target, try clearing errors via RUT (restore-unlock-test)
set cur [targets]
if {[string first "DAP" $cur] >= 0} {
    puts "DAP error detected. Issuing RUT sequence..."
    # Halt the DAP, resync, retry
    rst -system
    after 500
    configparams force-mem-accesses 1
    connect
    after 10000
    catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}
    after 200
    puts "After RUT: [targets]"
}

# ===== Load bitstream =====
puts "Loading bitstream..."
fpga -file $BITSTREAM
after 2000
configparams force-mem-accesses 1
puts "Bitstream loaded"

# Re-select ARM core (fpga command may change current target)
configparams force-mem-accesses 1
after 500
puts "  Targets after bitstream:"
targets
set num_sel2 [targets -set -filter {name =~ "*Cortex-A9*#0*"}]
after 200
puts "  Re-selected $num_sel2 target(s): [targets]"

# ===== PS7 Init (individual steps) =====
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

# Verify PLL lock
set pll_status [read32 0xF800010C]
puts "  PLL_STATUS=[format 0x%08x $pll_status]"
if {[expr ($pll_status & 7)] != 7} { puts "  ERROR: PLLs not locked!"; exit 1 }
puts "  PLLs locked OK"

# ===== AFI config (matches comprehensive test) =====
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
if {$clk_cnt == 0} { puts "  WARNING: CLK_CNT=0, PL may not be running" }

# ===== Load ELF (same approach as working test_minimal) =====
puts "Loading ELF..."
dow $ELF_PATH
after 200
puts "  ELF loaded"

# Write markers to output buffer
puts "Writing test markers..."
mwr -force $OUTPUT_BUF 0xDEADBEEF
mwr -force [expr $OUTPUT_BUF + 4] 0xCAFEBABE
mwr -force [expr $OUTPUT_BUF + 8] 0x12345678
puts "  OUTPUT_BUF[0]=[read32 $OUTPUT_BUF] [read32 [expr $OUTPUT_BUF+4]] [read32 [expr $OUTPUT_BUF+8]]"

# ===== Run =====
puts "Running test..."
after 100
con
after 5000

# Halt
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}
after 100
stop
after 200

# Check FPGA state
set status [gp0_read 0x14]
set debug  [gp0_read 0x28]
set q8dbg  [gp0_read 0x3C]
set clk    [gp0_read 0x2C]
puts "  STATUS=0x[format %04X $status] DEBUG=0x[format %08X $debug] Q8DBG=0x[format %08X $q8dbg] CLK=[format %08X $clk]"

# ===== Read results (at words [9], [10], [11]) =====
set ntotal  [read32 [expr $OUTPUT_BUF + 36]]
set npassed [read32 [expr $OUTPUT_BUF + 40]]
set nfailed [read32 [expr $OUTPUT_BUF + 44]]
# Also dump raw buffer
puts "Raw: [mrd $OUTPUT_BUF 20]"

puts ""
puts "=============================================="
puts "  RESULTS"
puts "=============================================="
puts "  Total tests: $ntotal"
puts "  Passed:      $npassed"
puts "  Failed:      $nfailed"
puts "=============================================="

if {$nfailed == 0 && $ntotal > 0} {
    puts "  ALL TESTS PASSED!"
} elseif {$ntotal > 0} {
    puts "  $nfailed TESTS FAILED!"
} else {
    puts "  Program did not write results (markers still present)"
}

puts ""
puts "ARM halted."
exit
