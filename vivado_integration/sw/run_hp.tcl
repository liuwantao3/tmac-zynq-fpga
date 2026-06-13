# XSDB: full flow — program FPGA, init PS7, load ELF, run with live capture
connect
after 500

# Program FPGA (works once after power cycle)
fpga -file {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.bit}
puts "=== FPGA programmed ==="

source {D:/Users/u/tmac-zynq-fpga/vivado_integration/ps7_init.tcl}
targets -set 1
puts "Running ps7_init (AFI0+AFI1 configured)..."
if {[catch {ps7_init; ps7_post_config} err]} {
    puts "ps7_init failed: $err"
    exit 1
}
puts "PS7 initialized"

targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}
puts "ARM core selected"
catch {rst -processor}
dow {D:/Users/u/workspace/tmac/Debug/tmac.elf}
con

# Switch to DAP/APU target for all memory reads
catch {targets -set 1}
after 200

# Spin 1: AFI experiment results
set timeout 15000
set elapsed 0
while {$elapsed < $timeout} {
    after 100
    set val [mrd 0x00203100 1]
    if {[string first "00000001" $val] >= 0} {
        puts "=== SPIN 1: AFI experiment ==="
        puts "AFI experiment (dbg[0..15]):"
        puts [mrd 0x002032D0 16]
        puts "AFI1_WR_CHANNEL reads back from ARM: [mrd 0x002032F0 1]"
        mwr 0x00203100 0
        break
    }
    set elapsed [expr {$elapsed + 100}]
}
if {$elapsed >= $timeout} {
    puts "TIMEOUT spin 1"
    exit 1
}

# Spin 2: Write test results
set elapsed 0
while {$elapsed < $timeout} {
    after 100
    set val [mrd 0x00203104 1]
    if {[string first "00000001" $val] >= 0} {
        puts "=== SPIN 2: Write test results ==="
        puts "WR_TEST scratch dump (ARM read after L2 inv):"
        puts [mrd 0x00203240 10]
        puts "DDR direct at WR_TEST_ADDR (DAP bypasses cache):"
        puts [mrd 0x00201000 8]
        puts "PL reg_debug at spin2: [mrd 0x00203264 1]"
        mwr 0x00203104 0
        break
    }
    set elapsed [expr {$elapsed + 100}]
}
if {$elapsed >= $timeout} {
    puts "TIMEOUT spin 2"
    exit 1
}

# Wait for compute test to complete
after 15000

puts "=== reg_debug capture ==="
puts "dbg0 (before start):"; puts [mrd 0x00203000 1]
puts "dbg_at_start:"; puts [mrd 0x00203004 1]
puts "completion marker + dbg_done + status + iter:"; puts [mrd 0x00203008 4]
puts "dbg_running (first poll):"; puts [mrd 0x00203018 1]
puts "timeout marker:"; puts [mrd 0x0020301C 2]

puts "=== Compute results (first 16 lower 32 bits) ==="
puts [mrd 0x00203030 16]
puts "=== DDR direct at RES_ADDR (DAP bypasses cache) ==="
puts [mrd 0x00202080 32]

puts "=== PS7 status + SLCR unlock value ==="
puts [mrd 0x00203270 3]
puts "=== PS7 AFI0 registers ==="
puts [mrd 0x00203278 7]
puts "=== PS7 AFI1 registers ==="
puts [mrd 0x002032B4 5]
puts "=== DDR status ==="
puts [mrd 0x00203290 1]

puts "=== OCM marker (cache-free) ==="
puts [mrd 0x00010000 4]

puts "=== DONE marker ==="
puts [mrd 0x00203200 1]
puts [mrd 0x00203204 1]
puts "=== DONE ==="
