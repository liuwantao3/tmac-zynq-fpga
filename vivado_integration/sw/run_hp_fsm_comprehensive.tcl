# HP FSM Comprehensive Hardware Test (7 baseline + 1 Q8 compute test)
# Power-cycle the board before running!
# C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_hp_fsm_comprehensive.tcl

set DDR_BASE  0x00100000
set GP0_BASE  0x43C00000

proc read32 {addr} {
    set r [mrd $addr 1]
    if {[regexp {:\s+([0-9A-Fa-f]+)} $r full data]} { return [expr "0x$data"] }
    return -1
}

proc write32 {addr val} {
    mwr -force $addr $val
}

# GP0 register offsets
set REG_START     0x00
set REG_STATUS    0x14
set REG_DESC_BASE 0x18
set REG_DESC_TAIL 0x1C
set REG_DESC_HEAD 0x20
set REG_DEBUG     0x28
set REG_CLK_CNT   0x2C
set REG_CLK_SLW   0x30
set REG_ACT_INFO  0x34
set REG_DESC_INFO 0x38
set REG_Q8_DEBUG  0x3C
set REG_Q8_NUM_GROUPS 0x40

proc gp0_read {reg} {
    global GP0_BASE
    return [read32 [format 0x%08x [expr $GP0_BASE + $reg]]]
}

proc gp0_write {reg val} {
    global GP0_BASE
    write32 [format 0x%08x [expr $GP0_BASE + $reg]] $val
}

# Write descriptor at physical addr
# type: 15 = CPU_OP (default), 0 = Q8 compute, others = reserved
proc write_desc {addr next_addr weight_addr act_addr res_addr bytes {tensor_type 15}} {
    write32 $addr                  $next_addr   ;# bytes 0-3:  next_addr
    write32 [expr $addr + 4]       $weight_addr ;# bytes 4-7:  weight_addr
    write32 [expr $addr + 8]       $act_addr    ;# bytes 8-11: act_addr
    write32 [expr $addr + 12]      $res_addr    ;# bytes 12-15: result_addr
    # tensor_type stored as 16-bit at bytes 16-17 (little-endian)
    write32 [expr $addr + 16]      $tensor_type ;# bytes 16-19: {reserved[15:0], tensor_type}
    write32 [expr $addr + 20]      0            ;# bytes 20-23: reserved
    write32 [expr $addr + 24]      $bytes       ;# bytes 24-27: act_total_bytes
    write32 [expr $addr + 28]      0            ;# bytes 28-31: reserved
}

# Shorthand for CPU_OP descriptor
proc write_desc_cpu {addr next_addr act_addr res_addr bytes} {
    write_desc $addr $next_addr 0 $act_addr $res_addr $bytes 15
}

# Write incrementing pattern (nbytes must be multiple of 4)
proc write_pattern_inc {addr nbytes} {
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        set val [expr {0x03020100 + $j * 0x04040404}]
        write32 [expr $addr + $j * 4] $val
    }
}

# Write constant byte pattern
proc write_pattern_const {addr nbytes byte_val} {
    set word_val [expr {($byte_val << 24) | ($byte_val << 16) | ($byte_val << 8) | $byte_val}]
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        write32 [expr $addr + $j * 4] $word_val
    }
}

# Write checkerboard pattern (A5/5A)
proc write_pattern_checker {addr nbytes} {
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        if {$j & 1} {
            write32 [expr $addr + $j * 4] 0x5A5A5A5A
        } else {
            write32 [expr $addr + $j * 4] 0xA5A5A5A5
        }
    }
}

# Zero-fill DDR region
proc zero_fill {addr nbytes} {
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        write32 [expr $addr + $j * 4] 0x00000000
    }
}

# Verify incrementing pattern
proc verify_inc {addr nbytes test_id} {
    set ok 1
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        set expected [expr {0x03020100 + $j * 0x04040404}]
        set expected [expr {$expected & 0xFFFFFFFF}]
        set got [read32 [expr $addr + $j * 4]]
        if {$got != $expected} {
            puts "  FAIL[${test_id}]: addr=[format 0x%08x [expr $addr + $j * 4]] expected=[format 0x%08x $expected] got=[format 0x%08x $got]"
            set ok 0
        }
    }
    if {$ok} { puts "  Test $test_id: PASS" } else { puts "  Test $test_id: FAIL" }
    return $ok
}

# Verify constant byte pattern
proc verify_const {addr nbytes byte_val test_id} {
    set ok 1
    set expected [expr {($byte_val << 24) | ($byte_val << 16) | ($byte_val << 8) | $byte_val}]
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        set got [read32 [expr $addr + $j * 4]]
        if {$got != $expected} {
            puts "  FAIL[${test_id}]: addr=[format 0x%08x [expr $addr + $j * 4]] expected=[format 0x%08x $expected] got=[format 0x%08x $got]"
            set ok 0
        }
    }
    if {$ok} { puts "  Test $test_id: PASS" } else { puts "  Test $test_id: FAIL" }
    return $ok
}

# Verify checkerboard pattern
proc verify_checker {addr nbytes test_id} {
    set ok 1
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        if {$j & 1} { set expected 0x5A5A5A5A } else { set expected 0xA5A5A5A5 }
        set got [read32 [expr $addr + $j * 4]]
        if {$got != $expected} {
            puts "  FAIL[${test_id}]: addr=[format 0x%08x [expr $addr + $j * 4]] expected=[format 0x%08x $expected] got=[format 0x%08x $got]"
            set ok 0
        }
    }
    if {$ok} { puts "  Test $test_id: PASS" } else { puts "  Test $test_id: FAIL" }
    return $ok
}

# Start chain and wait for HEAD to reach expected_head
proc run_chain {desc_base expected_head} {
    global REG_DESC_BASE REG_DESC_TAIL REG_START REG_DESC_HEAD REG_STATUS REG_DEBUG REG_CLK_CNT
    gp0_write $REG_DESC_BASE $desc_base
    after 10
    gp0_write $REG_DESC_TAIL 1
    after 10
    gp0_write $REG_START 1
    after 10

    set timeout_ms 30000
    set start [clock milliseconds]
    set done 0
    while {[expr {[clock milliseconds] - $start}] < $timeout_ms} {
        set head [gp0_read $REG_DESC_HEAD]
        if {$head >= $expected_head} {
            set elapsed [expr {[clock milliseconds] - $start}]
            puts "  Done at ${elapsed}ms (HEAD=$head)"
            set done 1
            break
        }
        after 50
    }

    if {!$done} {
        set status [gp0_read $REG_STATUS]
        set dbg [gp0_read $REG_DEBUG]
        set head [gp0_read $REG_DESC_HEAD]
        set clk [gp0_read $REG_CLK_CNT]
        puts "  TIMEOUT: STATUS=[format 0x%08x $status] DEBUG=[format 0x%08x $dbg] HEAD=$head CLK_CNT=[format 0x%08x $clk]"
        set state [expr ($dbg >> 28) & 0x1F]
        set state_names {IDLE FETCH_DESC FETCH_DESC_W LOAD_ACT LOAD_ACT_W WRITE_RES WRITE_RES_W DONE LOAD_WEIGHT LOAD_WEIGHT_W LOAD_SCALES LOAD_SCALES_W COPY_ACT_TO_CORE COMPUTE COMPUTE_W READ_RES READ_RES_ACC COPY_ACC_TO_BUF}
        if {$state < [llength $state_names]} { set sname [lindex $state_names $state] } else { set sname "UNK" }
        puts "  FSM state=$state ($sname)"
        return 0
    }
    return 1
}

# ===== MAIN =====
puts "=============================================="
puts "  HP FSM Comprehensive Hardware Test (7+1 tests)"
puts "=============================================="

configparams force-mem-accesses 1
connect
after 15000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}
after 200

puts "=== Step 1: Load bitstream ==="
fpga -file {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.bit}
after 1000
configparams force-mem-accesses 1

puts "=== Step 2: Full ps7_init ==="
source {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.gen/sources_1/bd/system/ip/system_ps7_0/ps7_init.tcl}
ps7_mio_init_data_3_0
ps7_pll_init_data_3_0
ps7_clock_init_data_3_0
ps7_ddr_init_data_3_0
ps7_peripherals_init_data_3_0
after 300
ps7_post_config_3_0
after 200

# Verify PLL lock
set pll_status [read32 0xF800010C]
puts "  PLL_STATUS=[format 0x%08x $pll_status]"
if {[expr ($pll_status & 7)] != 7} { puts "  ERROR: PLLs not locked!"; exit 1 }
puts "  PLLs all locked OK"

puts "=== Step 3: PL clock check ==="
set clk_cnt [gp0_read $REG_CLK_CNT]
puts "  Clock counter = [format 0x%08x $clk_cnt]"
if {$clk_cnt < 1000} { puts "  ERROR: PL clock not running!"; exit 1 }
puts "  PL clock OK"

puts "=== Step 4: DAP AFI config ==="
mwr -force 0xF8000008 0x0000DF0D  ;# unlock SLCR
after 20
mwr -force 0xF8000910 0x0000000F  ;# LVL_SHFTR_EN
after 20
mwr -force 0xF8008000 0x00000005  ;# AFI0_CTRL
after 20
mwr -force 0xF8008004 0x00000044  ;# AFI0_PART
after 20
mwr -force 0xF8008008 0x00000001  ;# AFI0_WRCHAN
after 20
mwr -force 0xF8000004 0x0000767B  ;# lock SLCR
after 50
puts "  CTRL=   [format 0x%08x [read32 0xF8008000]]"
puts "  STATUS= [format 0x%08x [read32 0xF8008014]]"

puts "=== Step 5: Verify GP0 access ==="
set status_init [gp0_read $REG_STATUS]
puts "  REG_STATUS = [format 0x%08x $status_init]"
gp0_write $REG_DESC_BASE 0xDEADBEEF
after 10
set db_check [gp0_read $REG_DESC_BASE]
if {$db_check != 0xDEADBEEF} { puts "  ERROR: GP0 write FAILED!"; exit 1 }
puts "  GP0 access OK"

set pass_count 0
set fail_count 0

# ===================================================================
# Test 1: Basic 64 bytes (regression — matches working hardware test)
# ===================================================================
puts "\n--- Test 1: Basic 64 bytes ---"
write_desc_cpu 0x00100000 0 0x00101000 0x00102000 64
write_pattern_inc 0x00101000 64
zero_fill 0x00102000 64
if {[run_chain 0x00100000 1]} {
    set status [gp0_read $REG_STATUS]
    set head [gp0_read $REG_DESC_HEAD]
    puts "  STATUS=[format 0x%08x $status] (expect 0x300) HEAD=$head (expect 1)"
    set s [verify_inc 0x00102000 64 1]
    if {$s} { incr pass_count } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 2: Minimum 8 bytes (1 word)
# ===================================================================
puts "\n--- Test 2: Minimum 8 bytes ---"
write_desc_cpu 0x00100020 0 0x00101040 0x00102040 8
write32 0x00101040 0xAABBCCDD
zero_fill 0x00102040 8
if {[run_chain 0x00100020 1]} {
    set status [gp0_read $REG_STATUS]
    puts "  STATUS=[format 0x%08x $status] (expect 0x300)"
    set got [read32 0x00102040]
    if {$got == 0xAABBCCDD} {
        puts "  Test 2: PASS"; incr pass_count
    } else {
        puts "  FAIL[2]: res=[format 0x%08x $got] expected 0xAABBCCDD"; incr fail_count
    }
} else { incr fail_count }

# ===================================================================
# Test 3: 128 bytes (2 HP read bursts)
# ===================================================================
puts "\n--- Test 3: 128 bytes (2 bursts) ---"
write_desc_cpu 0x00100040 0 0x00101100 0x00102100 128
write_pattern_inc 0x00101100 128
zero_fill 0x00102100 128
if {[run_chain 0x00100040 1]} {
    set status [gp0_read $REG_STATUS]
    puts "  STATUS=[format 0x%08x $status] (expect 0x300)"
    set s [verify_inc 0x00102100 128 3]
    if {$s} { incr pass_count } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 4: 256 bytes max (4 HP read bursts)
# ===================================================================
puts "\n--- Test 4: 256 bytes max (4 bursts) ---"
write_desc_cpu 0x00100060 0 0x00101200 0x00102200 256
write_pattern_inc 0x00101200 256
zero_fill 0x00102200 256
if {[run_chain 0x00100060 1]} {
    set status [gp0_read $REG_STATUS]
    puts "  STATUS=[format 0x%08x $status] (expect 0x300)"
    set s [verify_inc 0x00102200 256 4]
    if {$s} { incr pass_count } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 5: Chain of 2 descriptors
# ===================================================================
puts "\n--- Test 5: Chain of 2 descriptors ---"
write_desc_cpu 0x00100100 0x00100120 0x00101300 0x00102300 64
write_desc_cpu 0x00100120 0           0x00101340 0x00102340 32
write_pattern_inc   0x00101300 64
write_pattern_const 0x00101340 32 0xFF
zero_fill 0x00102300 64
zero_fill 0x00102340 32
if {[run_chain 0x00100100 2]} {
    set status [gp0_read $REG_STATUS]
    set head [gp0_read $REG_DESC_HEAD]
    puts "  STATUS=[format 0x%08x $status] (expect 0x300) HEAD=$head (expect 2)"
    set s0 [verify_inc   0x00102300 64 5]
    set s1 [verify_const 0x00102340 32 0xFF 5]
    if {$s0 && $s1} { incr pass_count } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 6: Chain of 3 descriptors
# ===================================================================
puts "\n--- Test 6: Chain of 3 descriptors ---"
write_desc_cpu 0x00100140 0x00100160 0x00101400 0x00102400 48
write_desc_cpu 0x00100160 0x00100180 0x00101430 0x00102430 64
write_desc_cpu 0x00100180 0           0x00101470 0x00102470 40
write_pattern_inc     0x00101400 48
write_pattern_const   0x00101430 64 0xFF
write_pattern_checker 0x00101470 40
zero_fill 0x00102400 48
zero_fill 0x00102430 64
zero_fill 0x00102470 40
if {[run_chain 0x00100140 3]} {
    set status [gp0_read $REG_STATUS]
    set head [gp0_read $REG_DESC_HEAD]
    puts "  STATUS=[format 0x%08x $status] (expect 0x300) HEAD=$head (expect 3)"
    set s0 [verify_inc     0x00102400 48 6]
    set s1 [verify_const   0x00102430 64 0xFF 6]
    set s2 [verify_checker 0x00102470 40 6]
    if {$s0 && $s1 && $s2} { incr pass_count } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 7: Re-start from DONE
# ===================================================================
puts "\n--- Test 7: Re-start from DONE ---"
write_desc_cpu 0x00100200 0 0x00101500 0x00102500 64
write_pattern_inc 0x00101500 64
zero_fill 0x00102500 64
if {[run_chain 0x00100200 1]} {
    set status [gp0_read $REG_STATUS]
    set head [gp0_read $REG_DESC_HEAD]
    puts "  First run: STATUS=[format 0x%08x $status] HEAD=$head (expect 1)"
    set s1 [verify_inc 0x00102500 64 7]
    if {$s1} {
        # Second run: different descriptor, same FSM
        write_desc_cpu 0x00100220 0 0x00101540 0x00102540 32
        write_pattern_const 0x00101540 32 0x5A
        zero_fill 0x00102540 32
        if {[run_chain 0x00100220 1]} {
            set status [gp0_read $REG_STATUS]
            set head [gp0_read $REG_DESC_HEAD]
            puts "  Second run: STATUS=[format 0x%08x $status] HEAD=$head (expect 1)"
            set s2a [verify_inc   0x00102500 64 7]
            set s2b [verify_const 0x00102540 32 0x5A 7]
            if {$s1 && $s2a && $s2b} { incr pass_count } else { incr fail_count }
        } else { incr fail_count }
    } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 8: Q8 compute — single 64x64 tile, all weights=1, scale=1.0, act=1
#   Expected: each row sum = 64 (64 columns × 1 × 1)
# ===================================================================
puts "\n--- Test 8: Q8 compute (all-1s pattern) ---"

set Q8_WEIGHT_ADDR  0x00103000
set Q8_SCALE_ADDR   [expr $Q8_WEIGHT_ADDR + 4096]
set Q8_ACT_ADDR     0x00105000
set Q8_RES_ADDR     0x00106000
set Q8_DESC_ADDR    0x00100240

# Fill weights: all INT8 = 1 (4096 bytes = 1024 words of 0x01010101)
write_pattern_const $Q8_WEIGHT_ADDR 4096 0x01

# Fill scales: UQ8.8 1.0 = 0x0100 (256 bytes = 64 words of 0x01000100 packed)
# Write 32-bit words at 4-byte-aligned addresses (2 scales per word)
set sc_pair [expr {0x0100 | (0x0100 << 16)}]
for {set j 0} {$j < 64} {incr j} {
    write32 [expr $Q8_SCALE_ADDR + $j * 4] $sc_pair
}

# Fill activations: 64 x int16 = 1 (128 bytes, 2 per 32-bit word)
set act_pair [expr {0x0001 | (0x0001 << 16)}]
for {set j 0} {$j < 32} {incr j} {
    write32 [expr $Q8_ACT_ADDR + $j * 4] $act_pair
}

# Result buffer: zero-fill 512 bytes
zero_fill $Q8_RES_ADDR 512

# Descriptor: tensor_type=0 (compute path), act_total_bytes=128
write_desc $Q8_DESC_ADDR 0 $Q8_WEIGHT_ADDR $Q8_ACT_ADDR $Q8_RES_ADDR 128 0

if {[run_chain $Q8_DESC_ADDR 1]} {
    set status [gp0_read $REG_STATUS]
    set head [gp0_read $REG_DESC_HEAD]
    set q8_dbg [gp0_read $REG_Q8_DEBUG]
    puts "  STATUS=[format 0x%08x $status] (expect 0x300) HEAD=$head (expect 1) Q8_DEBUG=[format 0x%08x $q8_dbg]"

    # Verify results: each 64-bit word should = 64 (0x40) for all 64 rows
    set ok 1
    for {set j 0} {$j < 64} {incr j} {
        set addr [expr $Q8_RES_ADDR + $j * 8]
        set lo [read32 $addr]
        set hi [read32 [expr $addr + 4]]
        # S24.8: result = 64 = 0x40. In 64-bit word: bytes 0-5 = 48-bit result, bytes 6-7 = 0
        # Little-endian: lo = bytes 0-3, hi = bytes 4-7
        # lo should = 0x00000040, hi should = 0x00000000
        if {$lo != 64 || $hi != 0} {
            puts "  FAIL[8]: row $j addr=[format 0x%08x $addr] lo=[format 0x%08x $lo] hi=[format 0x%08x $hi] (expected lo=0x00000040 hi=0x00000000)"
            set ok 0
        }
    }
    if {$ok} {
        puts "  Test 8: PASS (all 64 rows = 64)"
        incr pass_count
    } else {
        incr fail_count
    }
} else { incr fail_count }

# ===================================================================
# Summary
# ===================================================================
puts "\n=============================================="
if {$fail_count == 0} {
    puts "  ALL $pass_count TESTS PASSED"
} else {
    puts "  $fail_count TESTS FAILED (of [expr $pass_count + $fail_count])"
}
puts "=============================================="

# Halt ARM for clean state
targets -set -filter {name =~ "*Cortex-A9*#0*"}
after 50
stop
after 200
puts "ARM halted."

exit
