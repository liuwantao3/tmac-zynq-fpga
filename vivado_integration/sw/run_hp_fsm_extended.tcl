# HP FSM Extended Hardware Tests (Tests 9b-16)
# Assumes board is already configured from run_hp_fsm_comprehensive.tcl
# If running standalone, you MUST power-cycle and configure first:
#   C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_hp_fsm_comprehensive.tcl
# Then WITHOUT disconnecting:
#   source {D:/Users/u/tmac-zynq-fpga/vivado_integration/sw/run_hp_fsm_extended.tcl}
#
# Or run standalone (reloads bitstream):
#   C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_hp_fsm_extended.tcl

# ===== Helpers =====
proc read32 {addr} {
    set r [mrd $addr 1]
    if {[regexp {:\s+([0-9A-Fa-f]+)} $r full data]} { return [expr "0x$data"] }
    return -1
}
proc write32 {addr val} { mwr -force $addr $val }

set DDR_BASE 0x00100000
set GP0_BASE 0x43C00000
set REG_WR_ADDR    0x04
set REG_WR_DATA    0x08
set REG_START      0x00
set REG_Q8_NUM_GROUPS 0x10
set REG_STATUS     0x14
set REG_DESC_BASE  0x18
set REG_DESC_TAIL  0x1C
set REG_DESC_HEAD  0x20
set REG_DEBUG      0x28
set REG_CLK_CNT    0x2C
set REG_Q8_DEBUG   0x3C

proc gp0_read {reg} { global GP0_BASE; return [read32 [format 0x%08x [expr $GP0_BASE + $reg]]] }
proc gp0_write {reg val} { global GP0_BASE; write32 [format 0x%08x [expr $GP0_BASE + $reg]] $val }

proc write_desc {addr next_addr weight_addr act_addr res_addr bytes {tensor_type 15} {num_groups 0}} {
    write32 $addr                  $next_addr
    write32 [expr $addr + 4]       $weight_addr
    write32 [expr $addr + 8]       $act_addr
    write32 [expr $addr + 12]      $res_addr
    write32 [expr $addr + 16]      $tensor_type
    write32 [expr $addr + 20]      $num_groups
    write32 [expr $addr + 24]      $bytes
    write32 [expr $addr + 28]      0
}
proc write_desc_cpu {addr next_addr act_addr res_addr bytes} {
    write_desc $addr $next_addr 0 $act_addr $res_addr $bytes 15
}
proc write_pattern_inc {addr nbytes} {
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        set val [expr {0x03020100 + $j * 0x04040404}]
        write32 [expr $addr + $j * 4] [expr {$val & 0xFFFFFFFF}]
    }
}
proc write_pattern_const {addr nbytes byte_val} {
    set word_val [expr {($byte_val << 24) | ($byte_val << 16) | ($byte_val << 8) | $byte_val}]
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        write32 [expr $addr + $j * 4] $word_val
    }
}
proc write_pattern_checker {addr nbytes} {
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        if {$j & 1} { write32 [expr $addr + $j * 4] 0x5A5A5A5A } \
        else        { write32 [expr $addr + $j * 4] 0xA5A5A5A5 }
    }
}
proc zero_fill {addr nbytes} {
    for {set j 0} {$j < $nbytes / 4} {incr j} { write32 [expr $addr + $j * 4] 0 }
}
proc verify_inc {addr nbytes test_id} {
    set ok 1
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        set expected [expr {0x03020100 + $j * 0x04040404}]
        set got [read32 [expr $addr + $j * 4]]
        if {$got != $expected} { puts "  FAIL[${test_id}]: addr=[format 0x%08x [expr $addr + $j * 4]] expected=[format 0x%08x $expected] got=[format 0x%08x $got]"; set ok 0 }
    }
    if {$ok} { puts "  Test $test_id: PASS" } else { puts "  Test $test_id: FAIL" }; return $ok
}
proc verify_const {addr nbytes byte_val test_id} {
    set ok 1
    set expected [expr {($byte_val << 24) | ($byte_val << 16) | ($byte_val << 8) | $byte_val}]
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        set got [read32 [expr $addr + $j * 4]]
        if {$got != $expected} { puts "  FAIL[${test_id}]: addr=[format 0x%08x [expr $addr + $j * 4]] expected=[format 0x%08x $expected] got=[format 0x%08x $got]"; set ok 0 }
    }
    if {$ok} { puts "  Test $test_id: PASS" } else { puts "  Test $test_id: FAIL" }; return $ok
}
proc verify_checker {addr nbytes test_id} {
    set ok 1
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        if {$j & 1} { set expected 0x5A5A5A5A } else { set expected 0xA5A5A5A5 }
        set got [read32 [expr $addr + $j * 4]]
        if {$got != $expected} { puts "  FAIL[${test_id}]: addr=[format 0x%08x [expr $addr + $j * 4]] expected=[format 0x%08x $expected] got=[format 0x%08x $got]"; set ok 0 }
    }
    if {$ok} { puts "  Test $test_id: PASS" } else { puts "  Test $test_id: FAIL" }; return $ok
}

# Verify Q8 compute results: all 64 rows must equal expected integer value
proc verify_q8_rows {res_addr expected test_id} {
    set ok 1
    for {set j 0} {$j < 64} {incr j} {
        set addr [expr $res_addr + $j * 8]
        set lo [read32 $addr]
        set hi [read32 [expr $addr + 4]]
        if {$lo != $expected || $hi != 0} {
            puts "  FAIL[${test_id}]: row $j lo=[format 0x%08x $lo] hi=[format 0x%08x $hi] (expected lo=[format 0x%08x $expected] hi=0x00000000)"
            set ok 0
        }
    }
    if {$ok} { puts "  Test $test_id: PASS (all 64 rows = $expected)" } else { puts "  Test $test_id: FAIL" }
    return $ok
}

# Fill Q8 scale data for N groups at weight_addr+4096
proc fill_q8_scales {weight_addr num_groups scale_val} {
    set sc_pair [expr {($scale_val << 16) | $scale_val}]
    for {set g 0} {$g < $num_groups} {incr g} {
        set grp_addr [expr $weight_addr + 4096 + $g * 256]
        for {set j 0} {$j < 64} {incr j} {
            write32 [expr $grp_addr + $j * 4] $sc_pair
        }
    }
}

# Fill Q8 activation data for N groups at act_addr
proc fill_q8_acts {act_addr num_groups act_val} {
    set act_pair [expr {($act_val << 16) | $act_val}]
    for {set g 0} {$g < $num_groups} {incr g} {
        set grp_addr [expr $act_addr + $g * 128]
        for {set j 0} {$j < 32} {incr j} {
            write32 [expr $grp_addr + $j * 4] $act_pair
        }
    }
}

# Run chain and wait for HEAD
proc run_chain {desc_base expected_head} {
    global REG_DESC_BASE REG_DESC_TAIL REG_START REG_DESC_HEAD REG_STATUS REG_DEBUG REG_Q8_DEBUG REG_WR_ADDR REG_WR_DATA
    gp0_write $REG_DESC_BASE $desc_base; after 10
    gp0_write $REG_DESC_TAIL 1; after 10
    gp0_write $REG_START 1; after 10
    set timeout_ms 30000
    set start [clock milliseconds]
    set done 0
    while {[expr {[clock milliseconds] - $start}] < $timeout_ms} {
        set head [gp0_read $REG_DESC_HEAD]
        if {$head >= $expected_head} {
            set elapsed [expr {[clock milliseconds] - $start}]
            puts "  Done at ${elapsed}ms (HEAD=$head)"
            set wr_addr [gp0_read $REG_WR_ADDR]
            set wr_data [gp0_read $REG_WR_DATA]
            puts "  REG_WR_ADDR=[format 0x%08x $wr_addr] REG_WR_DATA=[format 0x%08x $wr_data]"
            set done 1; break
        }
        after 50
    }
    if {!$done} {
        set status [gp0_read $REG_STATUS]
        set dbg [gp0_read $REG_DEBUG]
        set q8_dbg [gp0_read $REG_Q8_DEBUG]
        set head [gp0_read $REG_DESC_HEAD]
        set state_mask [expr ($dbg >> 27) & 0x1F]
        set state_names {IDLE FETCH_DESC FETCH_DESC_W LOAD_ACT LOAD_ACT_W WRITE_RES WRITE_RES_W DONE LOAD_WEIGHT LOAD_WEIGHT_W LOAD_SCALES LOAD_SCALES_W COPY_ACT_TO_CORE COMPUTE COMPUTE_W READ_RES READ_RES_ACC COPY_ACC_TO_BUF TIMEOUT_ERROR WRITE_RES_BURST}
        if {$state_mask < [llength $state_names]} { set sname [lindex $state_names $state_mask] } else { set sname "UNK" }
        puts "  TIMEOUT: STATUS=[format 0x%08x $status] DEBUG=[format 0x%08x $dbg] Q8_DEBUG=[format 0x%08x $q8_dbg] HEAD=$head"
        puts "  FSM state=$state_mask ($sname)"
        return 0
    }
    return 1
}

# ===== Main =====
puts "=============================================="
puts "  HP FSM Extended Tests (9b-16)"
puts "=============================================="

# Check if already connected; if not, run full init
if {[catch {gp0_read $REG_CLK_CNT} result] || $result < 1000} {
    puts "Board not configured — running full init..."
    configparams force-mem-accesses 1
    connect; after 15000
    catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200
    fpga -file {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.bit}; after 1000
    configparams force-mem-accesses 1
    source {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.gen/sources_1/bd/system/ip/system_ps7_0/ps7_init.tcl}
    ps7_mio_init_data_3_0; ps7_pll_init_data_3_0; ps7_clock_init_data_3_0
    ps7_ddr_init_data_3_0; ps7_peripherals_init_data_3_0; after 300
    ps7_post_config_3_0; after 200
    set pll_status [read32 0xF800010C]
    puts "  PLL_STATUS=[format 0x%08x $pll_status]"
    if {[expr ($pll_status & 7)] != 7} { puts "PLLs not locked!"; exit 1 }
    mwr -force 0xF8000008 0x0000DF0D; after 20
    mwr -force 0xF8000910 0x0000000F; after 20
    mwr -force 0xF8008000 0x00000005; after 20
    mwr -force 0xF8008004 0x00000044; after 20
    mwr -force 0xF8008008 0x00000001; after 20
    mwr -force 0xF8000004 0x0000767B; after 50
}

set clk_cnt [gp0_read $REG_CLK_CNT]
puts "  Clock counter = [format 0x%08x $clk_cnt]"
if {$clk_cnt < 1000} { puts "ERROR: PL clock not running!"; exit 1 }

# Reset num_groups to 0 before starting
gp0_write $REG_Q8_NUM_GROUPS 0; after 10

set pass_count 0
set fail_count 0

# ===================================================================
# Test 9b: Q8 full 64×896 tile (14 groups) — all weights=1, scale=1.0, act=1
#   Expected: each row = 64 × 14 = 896 = 0x380
#   Validates: full multi-group iteration, col_group 0→13, acc_buf accumulation,
#             scale offset per group (256B stride), act offset per group (128B stride)
# ===================================================================
puts "\n--- Test 9b: Q8 full 64x896 tile (14 groups) ---"
set T_WEIGHT  0x00108000
set T_ACT     0x0010A000
set T_RES     0x0010C000
set T_DESC    0x00100400
set T_NGROUPS 14
set T_EXPECTED [expr 64 * $T_NGROUPS]

write_pattern_const $T_WEIGHT 4096 0x01
fill_q8_scales $T_WEIGHT $T_NGROUPS 0x0100
fill_q8_acts $T_ACT $T_NGROUPS 0x0001
zero_fill $T_RES 512
write_desc $T_DESC 0 $T_WEIGHT $T_ACT $T_RES 128 0 $T_NGROUPS

if {[run_chain $T_DESC 1]} {
    set s [verify_q8_rows $T_RES $T_EXPECTED 9b]
    if {$s} { incr pass_count } else { incr fail_count }
} else { incr fail_count }
gp0_write $REG_Q8_NUM_GROUPS 0; after 10

# ===================================================================
# Test 10: Q8 compute with weight=2 (non-identity weight value)
#   All weights=2, scale=1.0, act=1 → each row = 128 = 0x80
#   Validates: weight data loaded correctly through q8_wt_din (reg) path
# ===================================================================
puts "\n--- Test 10: Q8 weight=2 (non-identity) ---"
write_pattern_const $T_WEIGHT 4096 0x02
fill_q8_scales $T_WEIGHT 1 0x0100
fill_q8_acts $T_ACT 1 0x0001
zero_fill $T_RES 512
write_desc 0x00100420 0 $T_WEIGHT $T_ACT $T_RES 128 0

if {[run_chain 0x00100420 1]} {
    set s [verify_q8_rows $T_RES 128 10]
    if {$s} { incr pass_count } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 11: Q8 compute with zero activations
#   All weights=1, scale=1.0, act=0 → each row = 0
#   Validates: zero-act path, core produces no output
# ===================================================================
puts "\n--- Test 11: Q8 zero activations ---"
write_pattern_const $T_WEIGHT 4096 0x01
fill_q8_scales $T_WEIGHT 1 0x0100
fill_q8_acts $T_ACT 1 0x0000
zero_fill $T_RES 512
write_desc 0x00100440 0 $T_WEIGHT $T_ACT $T_RES 128 0

if {[run_chain 0x00100440 1]} {
    set s [verify_q8_rows $T_RES 0 11]
    if {$s} { incr pass_count } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 12: Q8 compute with max INT8 weight (127)
#   All weights=127, scale=1.0, act=1 → each row = 127×64 = 8128 = 0x1FC0
#   Validates: S24.8 accumulator at near-max single-column value
# ===================================================================
puts "\n--- Test 12: Q8 max weight (127) ---"
write_pattern_const $T_WEIGHT 4096 0x7F
fill_q8_scales $T_WEIGHT 1 0x0100
fill_q8_acts $T_ACT 1 0x0001
zero_fill $T_RES 512
write_desc 0x00100460 0 $T_WEIGHT $T_ACT $T_RES 128 0

if {[run_chain 0x00100460 1]} {
    set s [verify_q8_rows $T_RES 8128 12]
    if {$s} { incr pass_count } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 13: Long chain of 5 CPU_OP descriptors
#   Each writes a different pattern (inc/const/checker/inc/const)
#   Validates: descriptor chain length > 3, HEAD counting to 5,
#             FSM loops FETCH_DESC→LOAD_ACT→WRITE_RES repeatedly
# ===================================================================
puts "\n--- Test 13: Long chain of 5 descriptors ---"
write_desc_cpu 0x00100500 0x00100520 0x0010D000 0x0010E000 64
write_desc_cpu 0x00100520 0x00100540 0x0010D040 0x0010E040 48
write_desc_cpu 0x00100540 0x00100560 0x0010D070 0x0010E070 56
write_desc_cpu 0x00100560 0x00100580 0x0010D0A8 0x0010E0A8 32
write_desc_cpu 0x00100580 0           0x0010D0C8 0x0010E0C8 64
write_pattern_inc     0x0010D000 64
write_pattern_const   0x0010D040 48 0x5A
write_pattern_checker 0x0010D070 56
write_pattern_inc     0x0010D0A8 32
write_pattern_const   0x0010D0C8 64 0xFF
zero_fill 0x0010E000 64
zero_fill 0x0010E040 48
zero_fill 0x0010E070 56
zero_fill 0x0010E0A8 32
zero_fill 0x0010E0C8 64

if {[run_chain 0x00100500 5]} {
    set status [gp0_read $REG_STATUS]
    set head [gp0_read $REG_DESC_HEAD]
    puts "  STATUS=[format 0x%08x $status] HEAD=$head (expect 5)"
    set s0 [verify_inc     0x0010E000 64  13a]
    set s1 [verify_const   0x0010E040 48 0x5A 13b]
    set s2 [verify_checker 0x0010E070 56 13c]
    set s3 [verify_inc     0x0010E0A8 32 13d]
    set s4 [verify_const   0x0010E0C8 64 0xFF 13e]
    if {$s0 && $s1 && $s2 && $s3 && $s4} { incr pass_count } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 14: Mixed chain — CPU_OP → Q8 compute → CPU_OP
#   Realistic inference scenario: CPU ops interleaved with FPGA matmuls
#   Desc 0: CPU_OP (64B passthrough, inc pattern)
#   Desc 1: Q8 compute (all-1s, 64×64, single group)
#   Desc 2: CPU_OP (32B passthrough, const pattern 0xAB)
#   Validates: FSM switching between tensor_type 15 and 0 mid-chain,
#             weight/scale registers don't leak between desc types
# ===================================================================
puts "\n--- Test 14: Mixed chain CPU_OP + Q8 compute + CPU_OP ---"
# Use completely separate regions for all 3 descriptors
set M14_WEIGHT 0x00108000
set M14_ACT0   0x0010D200
set M14_RES0   0x0010E200
set M14_ACT1   0x0010D240
set M14_RES1   0x0010E400   ;# 512 bytes, no overlap with RES0/2
set M14_ACT2   0x0010D300
set M14_RES2   0x0010E300   ;# 64 bytes, no overlap with RES0/1

# Re-init weight/scales for Q8 compute
write_pattern_const $M14_WEIGHT 4096 0x01
fill_q8_scales $M14_WEIGHT 1 0x0100   ;# UQ8.8 1.0 → expected row result = 64
fill_q8_acts $M14_ACT1 1 0x0001

# Desc 0: CPU_OP (type=15), 64 bytes
write_desc_cpu 0x00100600 0x00100620 $M14_ACT0 $M14_RES0 64
# Desc 1: Q8 compute (type=0), act from M14_ACT1, res to M14_RES1
write_desc 0x00100620 0x00100640 $M14_WEIGHT $M14_ACT1 $M14_RES1 128 0
# Desc 2: CPU_OP (type=15), 32 bytes
write_desc_cpu 0x00100640 0 $M14_ACT2 $M14_RES2 32

write_pattern_inc   0x0010D200 64
write_pattern_const $M14_ACT2 32 0xAB
zero_fill $M14_RES0 64
zero_fill $M14_RES1 512
zero_fill $M14_RES2 32

# Debug: dump weight, scale, act data before chain
puts "  DEBUG PRE: weight[0]=[format 0x%08x [read32 $M14_WEIGHT]]"
puts "  DEBUG PRE: scale[0]=[format 0x%08x [read32 [expr $M14_WEIGHT+4096]]]"
puts "  DEBUG PRE: act[0]=[format 0x%08x [read32 $M14_ACT1]]"
puts "  DEBUG PRE: desc[0..7]=[read32 0x00100600] [read32 0x00100604] [read32 0x00100608] [read32 0x0010060C] [read32 0x00100610] [read32 0x00100614] [read32 0x00100618] [read32 0x0010061C]"
puts "  DEBUG PRE: desc2[0..7]=[read32 0x00100620] [read32 0x00100624] [read32 0x00100628] [read32 0x0010062C] [read32 0x00100630] [read32 0x00100634] [read32 0x00100638] [read32 0x0010063C]"
# Must set REG_Q8_NUM_GROUPS=1 for Q8 compute (num_groups=0 in desc means fallback to reg)
gp0_write $REG_Q8_NUM_GROUPS 1
if {[run_chain 0x00100600 3]} {
    set status [gp0_read $REG_STATUS]
    set head [gp0_read $REG_DESC_HEAD]
    set dbg [gp0_read $REG_DEBUG]
    set q8dbg [gp0_read $REG_Q8_DEBUG]
    puts "  STATUS=[format 0x%08x $status] HEAD=$head DEBUG=[format 0x%08x $dbg] Q8_DEBUG=[format 0x%08x $q8dbg] (expect 3)"
    puts "  DEBUG: RES0[0]=[format 0x%08x [read32 $M14_RES0]]"
    puts "  DEBUG: RES1[0]=[format 0x%08x [read32 $M14_RES1]]"
    puts "  DEBUG: RES2[0]=[format 0x%08x [read32 $M14_RES2]]"
    set s0 [verify_inc     $M14_RES0 64  14a]
    set s1 [verify_q8_rows $M14_RES1 64  14b]
    set s2 [verify_const   $M14_RES2 32 0xAB 14c]
    if {$s0 && $s1 && $s2} { incr pass_count } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 17: Q8 → CPU_OP (Q8 first, opposite of Test 14)
#   If this works, the bug is specific to CPU_OP→Q8 chaining.
#   If this fails, Q8 compute is broken in any chain.
# ===================================================================
puts "\n--- Test 17: Q8 → CPU_OP (Q8 first in chain) ---"
set M17_WEIGHT 0x00108000
set M17_ACT0   0x0010D600
set M17_RES0   0x0010E600
set M17_ACT1   0x0010D700
set M17_RES1   0x0010E900

write_pattern_const $M17_WEIGHT 4096 0x01
fill_q8_scales $M17_WEIGHT 1 0x0100
fill_q8_acts $M17_ACT0 1 0x0001
write_pattern_const $M17_ACT1 32 0xCD
zero_fill $M17_RES0 512
zero_fill $M17_RES1 32

# Desc 0: Q8 compute (type=0)
write_desc 0x00100800 0x00100820 $M17_WEIGHT $M17_ACT0 $M17_RES0 128 0
# Desc 1: CPU_OP (type=15)
write_desc_cpu 0x00100820 0 $M17_ACT1 $M17_RES1 32

gp0_write $REG_Q8_NUM_GROUPS 1
if {[run_chain 0x00100800 2]} {
    set status [gp0_read $REG_STATUS]
    set head [gp0_read $REG_DESC_HEAD]
    puts "  STATUS=[format 0x%08x $status] HEAD=$head (expect 2)"
    set s0 [verify_q8_rows $M17_RES0 64 17a]
    set s1 [verify_const $M17_RES1 32 0xCD 17b]
    if {$s0 && $s1} { incr pass_count } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 18: STANDALONE Q8 same data as Test 17
#   If this passes, the Q8 data setup is correct and the bug is in chain mode.
#   If this fails, the data setup (or stale Q8 core state) is the issue.
# ===================================================================
puts "\n--- Test 18: Standalone Q8 (same data as Test 17) ---"
set M18_WEIGHT 0x00108000
set M18_ACT    0x0010D600
set M18_RES    0x0010E800

write_pattern_const $M18_WEIGHT 4096 0x01
fill_q8_scales $M18_WEIGHT 1 0x0100
fill_q8_acts $M18_ACT 1 0x0001
zero_fill $M18_RES 512
write_desc 0x00100900 0 $M18_WEIGHT $M18_ACT $M18_RES 128 0
gp0_write $REG_Q8_NUM_GROUPS 1
if {[run_chain 0x00100900 1]} {
    set status [gp0_read $REG_STATUS]
    set head [gp0_read $REG_DESC_HEAD]
    puts "  STATUS=[format 0x%08x $status] HEAD=$head (expect 1)"
    set s0 [verify_q8_rows $M18_RES 64 18]
    if {$s0} { incr pass_count } else { incr fail_count }
} else { incr fail_count }

# ===================================================================
# Test 15: Re-run same descriptor 3 times
#   CPU_OP: inc pattern, run 3x, verify all 3 runs correct
#   Validates: FSM clean reset from DONE state, no residual state
#             between runs
# ===================================================================
puts "\n--- Test 15: Re-run same descriptor 3 times ---"
write_desc_cpu 0x00100700 0 0x0010D400 0x0010E400 64
write_pattern_inc 0x0010D400 64

set r15_ok 1
for {set run 0} {$run < 3} {incr run} {
    zero_fill 0x0010E400 64
    if {[run_chain 0x00100700 1]} {
        set status [gp0_read $REG_STATUS]
        set head [gp0_read $REG_DESC_HEAD]
        puts "  Run ${run}: STATUS=[format 0x%08x $status] HEAD=$head (expect 1)"
        set s [verify_inc 0x0010E400 64 15]
        if {!$s} { set r15_ok 0 }
    } else { set r15_ok 0 }
}
if {$r15_ok} { puts "  Test 15: PASS"; incr pass_count } else { puts "  Test 15: FAIL"; incr fail_count }

# ===================================================================
# Test 16: CPU_OP with act_total_bytes = 0
#   Descriptor with zero activation bytes → FSM should skip LOAD_ACT,
#   go to WRITE_RES with wr_remaining=0, advance to DONE
#   Validates: act_remaining=0 edge case in LOAD_ACT state
# ===================================================================
puts "\n--- Test 16: Zero-byte CPU_OP ---"
write_desc_cpu 0x00100720 0 0x0010D500 0x0010E500 0

if {[run_chain 0x00100720 1]} {
    set status [gp0_read $REG_STATUS]
    set head [gp0_read $REG_DESC_HEAD]
    puts "  STATUS=[format 0x%08x $status] HEAD=$head (expect 1)"
    if {$head == 1} {
        puts "  Test 16: PASS (zero-byte desc completed)"
        incr pass_count
    } else {
        puts "  Test 16: FAIL (HEAD=$head)"
        incr fail_count
    }
} else { incr fail_count }

# ===================================================================
# Summary
# ===================================================================
puts "\n=============================================="
if {$fail_count == 0} {
    puts "  ALL $pass_count EXTENDED TESTS PASSED"
} else {
    puts "  $fail_count EXTENDED TESTS FAILED (of [expr $pass_count + $fail_count])"
}
puts "=============================================="

targets -set -filter {name =~ "*Cortex-A9*#0*"}; after 50; stop; after 200
puts "ARM halted."
exit
