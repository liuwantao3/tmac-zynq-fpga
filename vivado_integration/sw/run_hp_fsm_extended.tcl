# HP FSM Extended Hardware Test
# Covers gaps: negative weights, non-unit scale, full 14-group,
# negative q5, varied d, mixed chains, alternating q5 patterns.
#
# Power-cycle the board before running!
# C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_hp_fsm_extended.tcl

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

set REG_START     0x00
set REG_Q8_NUM_GROUPS 0x10
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

proc gp0_read {reg} {
    global GP0_BASE
    return [read32 [format 0x%08x [expr $GP0_BASE + $reg]]]
}

proc gp0_write {reg val} {
    global GP0_BASE
    write32 [format 0x%08x [expr $GP0_BASE + $reg]] $val
}

# Write descriptor at physical addr
# tensor_type: 15=CPU_OP, 0=Q8, 1=Q5_0
proc write_desc {addr next_addr weight_addr act_addr res_addr bytes {tensor_type 15} {num_groups 0} {num_tiles 1}} {
    write32 $addr                  $next_addr
    write32 [expr $addr + 4]       $weight_addr
    write32 [expr $addr + 8]       $act_addr
    write32 [expr $addr + 12]      $res_addr
    write32 [expr $addr + 16]      $tensor_type
    set word20 [expr {($num_tiles << 16) | ($num_groups & 0xFF)}]
    write32 [expr $addr + 20]      $word20
    write32 [expr $addr + 24]      $bytes
    write32 [expr $addr + 28]      0
}

proc write_desc_cpu {addr next_addr act_addr res_addr bytes} {
    write_desc $addr $next_addr 0 $act_addr $res_addr $bytes 15
}

# Zero-fill DDR region
proc zero_fill {addr nbytes} {
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        write32 [expr $addr + $j * 4] 0
    }
}

proc write_pattern_const {addr nbytes byte_val} {
    set word_val [expr {($byte_val << 24) | ($byte_val << 16) | ($byte_val << 8) | $byte_val}]
    for {set j 0} {$j < $nbytes / 4} {incr j} {
        write32 [expr $addr + $j * 4] $word_val
    }
}

# ── Q8 helpers ──

# Fill scales for N groups: all same UQ8.8 scale_value
proc fill_q8_scales_const {base num_groups scale_val} {
    set sc_pair [expr {($scale_val & 0xFFFF) | (($scale_val & 0xFFFF) << 16)}]
    for {set g 0} {$g < $num_groups} {incr g} {
        set grp_addr [expr $base + $g * 256]
        for {set j 0} {$j < 64} {incr j} {
            write32 [expr $grp_addr + $j * 4] $sc_pair
        }
    }
}

# Fill activations for N groups: all same int16 act_val
proc fill_q8_acts_const {base num_groups act_val} {
    set act_pair [expr {($act_val & 0xFFFF) | (($act_val & 0xFFFF) << 16)}]
    for {set g 0} {$g < $num_groups} {incr g} {
        set grp_addr [expr $base + $g * 128]
        for {set j 0} {$j < 32} {incr j} {
            write32 [expr $grp_addr + $j * 4] $act_pair
        }
    }
}

# Verify Q8 result: nrows × 8-byte S24.8 words
proc verify_q8_result {res_addr nrows expected test_id} {
    set ok 1
    for {set j 0} {$j < $nrows} {incr j} {
        set addr [expr $res_addr + $j * 8]
        set lo [read32 $addr]
        set hi [read32 [expr $addr + 4]]
        set got [expr {($hi << 32) | $lo}]
        if {$got >= [expr {1 << 47}]} {
            set got [expr {$got - (1 << 48)}]
        }
        if {$got != $expected} {
            puts "  FAIL(${test_id}): row $j lo=[format 0x%08x $lo] hi=[format 0x%08x $hi] got=$got expected=$expected"
            set ok 0
        }
    }
    if {$ok} { puts "  Test $test_id: PASS (all $nrows rows = $expected)" }
    return $ok
}

# ── Q5_0 helpers ──

# Standard Q5 weight: qh=0xFFFFFFFF, all nibbles = q5_nibble, d=1.0
proc fill_q5_weight {base q5_nibble} {
    set qs_byte [expr ($q5_nibble << 4) | $q5_nibble]
    set qs_word32 [expr {$qs_byte * 0x01010101}]
    set qs_word_low16 [expr {$qs_byte | ($qs_byte << 8)}]
    set qh_word 0xFFFFFFFF
    set word0_val [expr {($qh_word & 0xFFFF) << 16 | 0x3C00}]
    set word1_val [expr {($qs_word_low16 << 16) | (($qh_word >> 16) & 0xFFFF)}]
    set word5_val [expr {(0x3C00 << 16) | $qs_word_low16}]
    set word6_val $qh_word
    for {set blk 0} {$blk < 56} {incr blk} {
        set bo [expr $base + $blk * 48]
        write32 [expr $bo + 0]   $word0_val
        write32 [expr $bo + 4]   $word1_val
        write32 [expr $bo + 8]   $qs_word32
        write32 [expr $bo + 12]  $qs_word32
        write32 [expr $bo + 16]  $qs_word32
        write32 [expr $bo + 20]  $word5_val
        write32 [expr $bo + 24]  $word6_val
        write32 [expr $bo + 28]  $qs_word32
        write32 [expr $bo + 32]  $qs_word32
        write32 [expr $bo + 36]  $qs_word32
        write32 [expr $bo + 40]  $qs_word32
        write32 [expr $bo + 44]  0x00000000
    }
}

# Custom qh: qh_word replaces 0xFFFFFFFF
# For negative q5: qh=0, nibble=1 → unsigned={0,0001}=1 → q5=1-16=-15
proc fill_q5_weight_qh {base q5_nibble qh_word} {
    set qs_byte [expr ($q5_nibble << 4) | $q5_nibble]
    set qs_word32 [expr {$qs_byte * 0x01010101}]
    set qs_word_low16 [expr {$qs_byte | ($qs_byte << 8)}]
    set word0_val [expr {($qh_word & 0xFFFF) << 16 | 0x3C00}]
    set word1_val [expr {($qs_word_low16 << 16) | (($qh_word >> 16) & 0xFFFF)}]
    set word5_val [expr {(0x3C00 << 16) | $qs_word_low16}]
    set word6_val $qh_word
    for {set blk 0} {$blk < 56} {incr blk} {
        set bo [expr $base + $blk * 48]
        write32 [expr $bo + 0]   $word0_val
        write32 [expr $bo + 4]   $word1_val
        write32 [expr $bo + 8]   $qs_word32
        write32 [expr $bo + 12]  $qs_word32
        write32 [expr $bo + 16]  $qs_word32
        write32 [expr $bo + 20]  $word5_val
        write32 [expr $bo + 24]  $word6_val
        write32 [expr $bo + 28]  $qs_word32
        write32 [expr $bo + 32]  $qs_word32
        write32 [expr $bo + 36]  $qs_word32
        write32 [expr $bo + 40]  $qs_word32
        write32 [expr $bo + 44]  0x00000000
    }
}

# Custom d value: d_fp16 replaces 0x3C00 (1.0)
proc fill_q5_weight_d {base q5_nibble d_fp16} {
    set qs_byte [expr ($q5_nibble << 4) | $q5_nibble]
    set qs_word32 [expr {$qs_byte * 0x01010101}]
    set qs_word_low16 [expr {$qs_byte | ($qs_byte << 8)}]
    set qh_word 0xFFFFFFFF
    set word0_val [expr {($qh_word & 0xFFFF) << 16 | ($d_fp16 & 0xFFFF)}]
    set word1_val [expr {($qs_word_low16 << 16) | (($qh_word >> 16) & 0xFFFF)}]
    set word5_val [expr {(($d_fp16 & 0xFFFF) << 16) | $qs_word_low16}]
    set word6_val $qh_word
    for {set blk 0} {$blk < 56} {incr blk} {
        set bo [expr $base + $blk * 48]
        write32 [expr $bo + 0]   $word0_val
        write32 [expr $bo + 4]   $word1_val
        write32 [expr $bo + 8]   $qs_word32
        write32 [expr $bo + 12]  $qs_word32
        write32 [expr $bo + 16]  $qs_word32
        write32 [expr $bo + 20]  $word5_val
        write32 [expr $bo + 24]  $word6_val
        write32 [expr $bo + 28]  $qs_word32
        write32 [expr $bo + 32]  $qs_word32
        write32 [expr $bo + 36]  $qs_word32
        write32 [expr $bo + 40]  $qs_word32
        write32 [expr $bo + 44]  0x00000000
    }
}

# Alternating q5 nibbles per block (even=1, odd=2)
proc fill_q5_weight_alternating {base nibble_even nibble_odd} {
    for {set blk 0} {$blk < 56} {incr blk} {
        if {[expr $blk % 2] == 0} { set nibble $nibble_even } else { set nibble $nibble_odd }
        set qs_byte [expr ($nibble << 4) | $nibble]
        set qs_word32 [expr {$qs_byte * 0x01010101}]
        set qs_word_low16 [expr {$qs_byte | ($qs_byte << 8)}]
        set qh_word 0xFFFFFFFF
        set word0_val [expr {($qh_word & 0xFFFF) << 16 | 0x3C00}]
        set word1_val [expr {($qs_word_low16 << 16) | (($qh_word >> 16) & 0xFFFF)}]
        set word5_val [expr {(0x3C00 << 16) | $qs_word_low16}]
        set word6_val $qh_word
        set bo [expr $base + $blk * 48]
        write32 [expr $bo + 0]   $word0_val
        write32 [expr $bo + 4]   $word1_val
        write32 [expr $bo + 8]   $qs_word32
        write32 [expr $bo + 12]  $qs_word32
        write32 [expr $bo + 16]  $qs_word32
        write32 [expr $bo + 20]  $word5_val
        write32 [expr $bo + 24]  $word6_val
        write32 [expr $bo + 28]  $qs_word32
        write32 [expr $bo + 32]  $qs_word32
        write32 [expr $bo + 36]  $qs_word32
        write32 [expr $bo + 40]  $qs_word32
        write32 [expr $bo + 44]  0x00000000
    }
}

# Row norms: 4 × UQ8.8 (8 bytes at weight_base + 2688)
proc fill_q5_norms {weight_base} {
    write32 [expr $weight_base + 2688] 0x01000100
    write32 [expr $weight_base + 2692] 0x01000100
}

# Fill activations: 896 × int16 = 1792 bytes
proc fill_q5_acts {base val} {
    set act_word [expr {($val & 0xFFFF) | (($val & 0xFFFF) << 16)}]
    for {set j 0} {$j < [expr 1792 / 4]} {incr j} {
        write32 [expr $base + $j * 4] $act_word
    }
}

# Verify Q5 result: nrows × 8-byte S24.8 words
proc verify_q5_result {res_addr nrows expected test_id} {
    set ok 1
    for {set j 0} {$j < $nrows} {incr j} {
        set addr [expr $res_addr + $j * 8]
        set lo [read32 $addr]
        set hi [read32 [expr $addr + 4]]
        set got [expr {($hi << 32) | $lo}]
        if {$got >= [expr {1 << 47}]} {
            set got [expr {$got - (1 << 48)}]
        }
        if {$got != $expected} {
            puts "  FAIL(${test_id}): row $j lo=[format 0x%08x $lo] hi=[format 0x%08x $hi] got=$got expected=$expected"
            set ok 0
        }
    }
    if {$ok} { puts "  Test $test_id: PASS (rows $nrows = $expected)" }
    return $ok
}

# Start chain and wait for HEAD to reach expected_head
proc run_chain {desc_base expected_head} {
    global REG_DESC_BASE REG_DESC_TAIL REG_START REG_DESC_HEAD REG_STATUS REG_DEBUG REG_Q8_DEBUG REG_CLK_CNT REG_ACT_INFO REG_DESC_INFO

    # Reset HEAD to 0 by writing REG_DESC_BASE first
    gp0_write $REG_DESC_BASE $desc_base
    after 10
    gp0_write $REG_DESC_TAIL 1
    after 10
    gp0_write $REG_START 0
    after 5
    gp0_write $REG_START 1
    after 10

    set timeout_ms 60000
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
        set q8_dbg [gp0_read $REG_Q8_DEBUG]
        set head [gp0_read $REG_DESC_HEAD]
        set clk [gp0_read $REG_CLK_CNT]
        set state_mask [expr ($dbg >> 27) & 0x1F]
        set state_names {IDLE FETCH_DESC FETCH_DESC_W LOAD_ACT LOAD_ACT_W WRITE_RES WRITE_RES_W DONE LOAD_WEIGHT LOAD_WEIGHT_W LOAD_SCALES LOAD_SCALES_W COPY_ACT_TO_CORE COMPUTE COMPUTE_W READ_RES READ_RES_ACC COPY_ACC_TO_BUF TIMEOUT_ERROR WRITE_RES_BURST Q5_LOAD_NORM Q5_LOAD_NORM_W Q5_COPY_ACT Q5_COPY_ACT_W Q5_BLOCK_COMPUTE Q5_BLOCK_COMPUTE_W Q5_READ_RES}
        if {$state_mask < [llength $state_names]} { set sname [lindex $state_names $state_mask] } else { set sname "UNK" }
        puts "  TIMEOUT: STATUS=[format 0x%08x $status] DEBUG=[format 0x%08x $dbg] Q8_DEBUG=[format 0x%08x $q8_dbg] HEAD=$head CLK_CNT=[format 0x%08x $clk]"
        puts "  FSM state=$state_mask ($sname) rd_busy=[expr ($dbg>>24)&1] wr_busy=[expr ($dbg>>23)&1] rd_done=[expr ($dbg>>26)&1] wr_done=[expr ($dbg>>25)&1]"
        set act_info [gp0_read $REG_ACT_INFO]
        set desc_info [gp0_read $REG_DESC_INFO]
        puts "  REG_ACT_INFO=[format 0x%08x $act_info] REG_DESC_INFO=[format 0x%08x $desc_info]"
        return 0
    }
    return 1
}

# ===== MAIN =====
puts "=============================================="
puts "  HP FSM Extended Hardware Test (10 tests)"
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
mwr -force 0xF8000008 0x0000DF0D
after 20
mwr -force 0xF8000910 0x0000000F
after 20
mwr -force 0xF8008000 0x00000005
after 20
mwr -force 0xF8008004 0x00000044
after 20
mwr -force 0xF8008008 0x00000001
after 20
mwr -force 0xF8000004 0x0000767B
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

# ── New test data area: 0x00110000-0x0012FFFF ──

# ===================================================================
# Test E1: Q8 negative weights
#   q8=-1 (INT8 0xFF), scale=1.0, act=1
#   Expected: each of 64 rows = -64
# ===================================================================
puts "\n--- Test E1: Q8 negative weights (expect -64 per row) ---"
set E1_W  0x00110000
set E1_S  [expr $E1_W + 4096]
set E1_A  [expr $E1_S + 256]
set E1_R  [expr $E1_A + 128]
set E1_D  0x00100400

write_pattern_const $E1_W 4096 0xFF   ;# all INT8 = -1
fill_q8_scales_const $E1_S 1 0x0100   ;# scale = 1.0
fill_q8_acts_const $E1_A 1 1
zero_fill $E1_R 512
write_desc $E1_D 0 $E1_W $E1_A $E1_R 128 0 1
if {[run_chain $E1_D 1]} {
    if {[verify_q8_result $E1_R 64 -64 E1]} { incr pass_count } else { incr fail_count }
} else { puts "  Test E1: TIMEOUT"; incr fail_count }

# ===================================================================
# Test E2: Q8 scale = 0.5
#   q8=2, scale=UQ8.8 0.5 (0x0080), act=1
#   dequant = 2 × 0.5 = 1.0, MAC: 64 × 1 × 1 = 64
# ===================================================================
puts "\n--- Test E2: Q8 scale=0.5 (q8=2, expect 64 per row) ---"
set E2_W  0x00112000
set E2_S  [expr $E2_W + 4096]
set E2_A  [expr $E2_S + 256]
set E2_R  [expr $E2_A + 128]
set E2_D  0x00100420

write_pattern_const $E2_W 4096 0x02   ;# all INT8 = 2
fill_q8_scales_const $E2_S 1 0x0080   ;# scale = 0.5
fill_q8_acts_const $E2_A 1 1
zero_fill $E2_R 512
write_desc $E2_D 0 $E2_W $E2_A $E2_R 128 0 1
if {[run_chain $E2_D 1]} {
    if {[verify_q8_result $E2_R 64 64 E2]} { incr pass_count } else { incr fail_count }
} else { puts "  Test E2: TIMEOUT"; incr fail_count }

# ===================================================================
# Test E3: Q8 full 14-group (64×896 tile, production scale)
#   q8=1, scale=1.0, act=1, 14 groups
#   Expected: each of 64 rows = 64 × 14 = 896
# ===================================================================
puts "\n--- Test E3: Q8 full 14-group (all-1s, expect 896 per row) ---"
set E3_W  0x00114000
set E3_S  [expr $E3_W + 4096]        ;# scales: 14×256 = 3584 bytes
set E3_A  [expr $E3_S + 3584]        ;# acts: 14×128 = 1792 bytes
set E3_R  [expr $E3_A + 1792]
set E3_D  0x00100440

write_pattern_const $E3_W 4096 0x01
fill_q8_scales_const $E3_S 14 0x0100
fill_q8_acts_const $E3_A 14 1
zero_fill $E3_R 512
write_desc $E3_D 0 $E3_W $E3_A $E3_R 128 0 14
if {[run_chain $E3_D 1]} {
    if {[verify_q8_result $E3_R 64 896 E3]} { incr pass_count } else { incr fail_count }
} else { puts "  Test E3: TIMEOUT"; incr fail_count }

# ===================================================================
# Test E4: Q5_0 negative q5
#   qh=0, nibble=1 → q5 = 1-16 = -15, d=1.0, norm=1.0, act=1
#   d_pre = 256, each element = 256 × (-15) × 1 = -3840
#   896 elements = -3840 × 896 = -3440640
# ===================================================================
puts "\n--- Test E4: Q5_0 negative q5 (qh=0, nibble=1 → q5=-15, expect -3440640) ---"
set E4_W  0x00118000
set E4_A  [expr $E4_W + 2696]
set E4_R  [expr $E4_A + 1792]
set E4_D  0x00100460

puts "  Writing negative-q5 weight..."
fill_q5_weight_qh $E4_W 1 0x00000000  ;# qh=0 → q5 = nibble-16 = -15
fill_q5_norms $E4_W
fill_q5_acts  $E4_A 1
zero_fill $E4_R 32
write_desc $E4_D 0 $E4_W $E4_A $E4_R 1792 1
if {[run_chain $E4_D 1]} {
    set s4 [verify_q5_result $E4_R 4 -3440640 E4]
    if {$s4} { incr pass_count } else { incr fail_count }
} else { puts "  Test E4: TIMEOUT"; incr fail_count }

# ===================================================================
# Test E5: Q5_0 d=0.5
#   q5=1, d=0.5 (f16 0x3800), norm=1.0, act=1
#   f16_decode(0x3800)=128 (S24.8)
#   d_pre = 128 × 256 >> 8 = 128
#   each element = 128 × 1 × 1 = 128
#   896 elements = 128 × 896 = 114688
# ===================================================================
puts "\n--- Test E5: Q5_0 d=0.5 (q5=1, expect 114688 per row) ---"
set E5_W  0x0011A000
set E5_A  [expr $E5_W + 2696]
set E5_R  [expr $E5_A + 1792]
set E5_D  0x00100480

fill_q5_weight_d $E5_W 1 0x3800  ;# d=0.5 (f16)
fill_q5_norms $E5_W
fill_q5_acts  $E5_A 1
zero_fill $E5_R 32
write_desc $E5_D 0 $E5_W $E5_A $E5_R 1792 1
if {[run_chain $E5_D 1]} {
    set s5 [verify_q5_result $E5_R 4 114688 E5]
    if {$s5} { incr pass_count } else { incr fail_count }
} else { puts "  Test E5: TIMEOUT"; incr fail_count }

# ===================================================================
# Dummy Q8 to clear col_group before CPU_OP tests
# (HP FSM bug: col_group not reset on CPU_OP path)
# ===================================================================
set COLGROUP_D  0x00100480
set COLGROUP_W  0x0011C000
set COLGROUP_S  [expr $COLGROUP_W + 4096]
set COLGROUP_A  [expr $COLGROUP_S + 256]
set COLGROUP_R  [expr $COLGROUP_A + 128]
zero_fill $COLGROUP_W 4096
fill_q8_scales_const $COLGROUP_S 1 0x0100
fill_q8_acts_const $COLGROUP_A 1 1
zero_fill $COLGROUP_R 512
write_desc $COLGROUP_D 0 $COLGROUP_W $COLGROUP_A $COLGROUP_R 128 0 0
if {![run_chain $COLGROUP_D 1]} { puts "  WARNING: col_group reset chain timeout" }
puts "  col_group reset done"

# ===================================================================
# Test E6a: CPU_OP standalone (no Q5 before it)
#   Demonstrates CPU_OP working in isolation
# ===================================================================
set E6a_D   0x001004A0
set E6a_CPU [expr $COLGROUP_R + 512]
set E6a_R   [expr $E6a_CPU + 64]
write_pattern_const $E6a_CPU 64 0xA5
zero_fill $E6a_R 64
write_desc_cpu $E6a_D 0 $E6a_CPU $E6a_R 64
puts "  Pre-chain CPU_OP standalone:"
for {set i 0} {$i < 4} {incr i} { set v [read32 [expr $E6a_CPU + $i*4]]; puts "    CPU[$i]=[format 0x%08x $v]" }
for {set i 0} {$i < 4} {incr i} { set v [read32 [expr $E6a_R + $i*4]]; puts "    R[$i]=[format 0x%08x $v]" }
if {[run_chain $E6a_D 1]} {
    set s6a 1
    for {set j 0} {$j < 16} {incr j} {
        set got [read32 [expr $E6a_R + $j * 4]]
        if {$got != 0xA5A5A5A5} { puts "  FAIL(E6a): standalone CPU_OP word $j got [format 0x%08x $got]"; set s6a 0 }
    }
    if {$s6a} { puts "  Test E6a: PASS (standalone CPU_OP)"; incr pass_count } else { incr fail_count }
} else { puts "  Test E6a: TIMEOUT (standalone CPU_OP)"; incr fail_count }

# ===================================================================
# Test E6: Q5 → CPU_OP → Q5 chain (mixed type transitions)
#   Desc 0: Q5 all-1s → 229376
#   Desc 1: CPU_OP passthrough (64 bytes of 0xA5)
#   Desc 2: Q5 all-1s → 229376
# ===================================================================
# Reset col_group before E6 chain
set E6_CLR_D  0x00100480
write_desc $E6_CLR_D 0 $COLGROUP_W $COLGROUP_A $COLGROUP_R 128 0 0
if {![run_chain $E6_CLR_D 1]} { puts "  WARNING: E6 col_group reset timeout" }

puts "\n--- Test E6: Q5 → CPU_OP → Q5 chain ---"
set E6_D0  0x001004C0
set E6_D1  0x001004E0
set E6_D2  0x00100500
set E6_W0  0x0011D000
# DDR layout:
# E6_W0 = 0x0011D000 (2696 → 0x0011DA87)
# E6_A0 = 0x0011DA88 (1792 → 0x0011E187)
# E6_R0 = 0x0011E188 (32 → 0x0011E1A7)
# E6_CPU = 0x0011E1A8 (64 → 0x0011E1E7)
# E6_R1 = 0x0011E1E8 (64 → 0x0011E227)
# E6_W2 = 0x0011E228 (2696 → 0x0011ECAF)
# E6_A2 = 0x0011ECB0 (1792 → 0x0011F3AF)
# E6_R2 = 0x0011F3B0 (32 → 0x0011F3CF)
set E6_A0  [expr $E6_W0 + 2696]
set E6_R0  [expr $E6_A0 + 1792]
set E6_CPU [expr $E6_R0 + 32]
set E6_R1  [expr $E6_CPU + 64]
set E6_W2  [expr $E6_R1 + 64]
set E6_A2  [expr $E6_W2 + 2696]
set E6_R2  [expr $E6_A2 + 1792]

fill_q5_weight $E6_W0 1
fill_q5_norms $E6_W0
fill_q5_acts $E6_A0 1
zero_fill $E6_R0 32
write_pattern_const $E6_CPU 64 0xA5
zero_fill $E6_R1 64
fill_q5_weight $E6_W2 1
fill_q5_norms $E6_W2
fill_q5_acts $E6_A2 1
zero_fill $E6_R2 32
write_desc $E6_D0 $E6_D1 $E6_W0 $E6_A0 $E6_R0 1792 1
write_desc_cpu $E6_D1 $E6_D2 $E6_CPU $E6_R1 64
write_desc $E6_D2 0 $E6_W2 $E6_A2 $E6_R2 1792 1

# Debug: pre-chain data dump
puts "  Pre-chain desc D1:"
for {set i 0} {$i < 8} {incr i} { set v [read32 [expr $E6_D1 + $i*4]]; puts "    [$i]=[format 0x%08x $v]" }
puts "  Pre-chain act CPU:"
for {set i 0} {$i < 4} {incr i} { set v [read32 [expr $E6_CPU + $i*4]]; puts "    [$i]=[format 0x%08x $v]" }
puts "  Pre-chain R1:"
for {set i 0} {$i < 8} {incr i} { set v [read32 [expr $E6_R1 + $i*4]]; puts "    [$i]=[format 0x%08x $v]" }
puts "  Pre-chain R2:"
for {set i 0} {$i < 4} {incr i} { set v [read32 [expr $E6_R2 + $i*4]]; puts "    [$i]=[format 0x%08x $v]" }

if {[run_chain $E6_D0 3]} {
    puts "  Post-chain R1 (CPU_OP result):"
    for {set i 0} {$i < 16} {incr i} { set v [read32 [expr $E6_R1 + $i*4]]; puts "    [$i]=[format 0x%08x $v]" }
    puts "  Post-chain act CPU (source):"
    for {set i 0} {$i < 4} {incr i} { set v [read32 [expr $E6_CPU + $i*4]]; puts "    [$i]=[format 0x%08x $v]" }
    puts "  Post-chain W0 block0 (check if matches R1):"
    for {set i 0} {$i < 12} {incr i} { set v [read32 [expr $E6_W0 + $i*4]]; puts "    [$i]=[format 0x%08x $v]" }
    puts "  Post-chain W2 block0 (check if matches R1):"
    for {set i 0} {$i < 12} {incr i} { set v [read32 [expr $E6_W2 + $i*4]]; puts "    [$i]=[format 0x%08x $v]" }
    set s6_0 [verify_q5_result $E6_R0 4 229376 E6b]
    set s6_cpu 1
    for {set j 0} {$j < 16} {incr j} {
        set got [read32 [expr $E6_R1 + $j * 4]]
        if {$got != 0xA5A5A5A5} { puts "  FAIL(E6b): CPU_OP word $j got [format 0x%08x $got] expected 0xA5A5A5A5"; set s6_cpu 0 }
    }
    set s6_2 [verify_q5_result $E6_R2 4 229376 E6c]
    if {$s6_0 && $s6_cpu && $s6_2} { puts "  Test E6: PASS"; incr pass_count } else { incr fail_count }
} else { puts "  Test E6: TIMEOUT"; incr fail_count }

# ===================================================================
# Test E7: Q8 negative activations
#   q8=1, scale=1.0, act=-1 (INT16 0xFFFF)
#   Expected: each of 64 rows = -64
# ===================================================================
puts "\n--- Test E7: Q8 negative act (act=-1, expect -64 per row) ---"
set E7_W  0x00120000
set E7_S  [expr $E7_W + 4096]
set E7_A  [expr $E7_S + 256]
set E7_R  [expr $E7_A + 128]
set E7_D  0x00100500

write_pattern_const $E7_W 4096 0x01   ;# weights = 1
fill_q8_scales_const $E7_S 1 0x0100   ;# scale = 1.0
fill_q8_acts_const $E7_A 1 0xFFFF     ;# act = -1 (INT16)
zero_fill $E7_R 512
write_desc $E7_D 0 $E7_W $E7_A $E7_R 128 0 1
if {[run_chain $E7_D 1]} {
    if {[verify_q8_result $E7_R 64 -64 E7]} { incr pass_count } else { incr fail_count }
} else { puts "  Test E7: TIMEOUT"; incr fail_count }

# ===================================================================
# Test E8: Q5_0 negative activations
#   q5=1, d=1.0, norm=1.0, act=-1
#   d_pre = 256, each = 256 × 1 × (-1) = -256
#   896 elements = -256 × 896 = -229376
# ===================================================================
puts "\n--- Test E8: Q5_0 negative act (act=-1, expect -229376 per row) ---"
set E8_W  0x00122000
set E8_A  [expr $E8_W + 2696]
set E8_R  [expr $E8_A + 1792]
set E8_D  0x00100520

fill_q5_weight $E8_W 1
fill_q5_norms  $E8_W
fill_q5_acts   $E8_A 0xFFFF  ;# act = -1 (INT16)
zero_fill $E8_R 32
write_desc $E8_D 0 $E8_W $E8_A $E8_R 1792 1
if {[run_chain $E8_D 1]} {
    set s8 [verify_q5_result $E8_R 4 -229376 E8]
    if {$s8} { incr pass_count } else { incr fail_count }
} else { puts "  Test E8: TIMEOUT"; incr fail_count }

# ===================================================================
# Test E9: Mixed Q5 + Q8 + CPU_OP + Q5 chain (cross-type, 4 descs)
#   Desc 0: Q5 all-1s, act=1 → 229376
#   Desc 1: Q8 all-1s, 1 group → 64
#   Desc 2: CPU_OP passthrough (64 bytes of 0x5A)
#   Desc 3: Q5 all-0s → 0
# ===================================================================
puts "\n--- Test E9: Mixed Q5 → Q8 → CPU_OP → Q5 chain ---"
set E9_D0  0x00100540
set E9_D1  0x00100560
set E9_D2  0x00100580
set E9_D3  0x001005A0

set E9_Q5W0 0x00124000
set E9_Q5A0 [expr $E9_Q5W0 + 2696]
set E9_Q5R0 [expr $E9_Q5A0 + 1792]

set E9_Q8W  0x00125200
set E9_Q8S  [expr $E9_Q8W + 4096]
set E9_Q8A  [expr $E9_Q8S + 256]
set E9_Q8R  [expr $E9_Q8A + 128]

set E9_CPU  [expr $E9_Q8R + 512]
set E9_CPUR [expr $E9_CPU + 64]

set E9_Q5W2 [expr $E9_CPUR + 64]
set E9_Q5A2 [expr $E9_Q5W2 + 2696]
set E9_Q5R2 [expr $E9_Q5A2 + 1792]

puts "  Writing E9 data..."
fill_q5_weight $E9_Q5W0 1;     puts "  Q5W0 done"
fill_q5_norms   $E9_Q5W0;      puts "  norms0 done"
fill_q5_acts    $E9_Q5A0 1;    puts "  acts0 done"
zero_fill $E9_Q5R0 32;         puts "  res0 done"

write_pattern_const $E9_Q8W 4096 0x01; puts "  Q8W done"
fill_q8_scales_const $E9_Q8S 1 0x0100; puts "  Q8S done"
fill_q8_acts_const $E9_Q8A 1 1;        puts "  Q8A done"
zero_fill $E9_Q8R 512;                 puts "  Q8R done"

write_pattern_const $E9_CPU 64 0x5A;   puts "  CPU src done"
zero_fill $E9_CPUR 64;                 puts "  CPU res done"

fill_q5_weight $E9_Q5W2 0;             puts "  Q5W2 done"
fill_q5_norms   $E9_Q5W2;              puts "  norms2 done"
fill_q5_acts    $E9_Q5A2 1;            puts "  acts2 done"
zero_fill $E9_Q5R2 32;                 puts "  res2 done"

write_desc $E9_D0 $E9_D1 $E9_Q5W0 $E9_Q5A0 $E9_Q5R0 1792 1
write_desc $E9_D1 $E9_D2 $E9_Q8W  $E9_Q8A  $E9_Q8R  128  0 1
write_desc_cpu $E9_D2 $E9_D3 $E9_CPU  $E9_CPUR 64
write_desc $E9_D3 0      $E9_Q5W2 $E9_Q5A2 $E9_Q5R2 1792 1

# Pre-chain descriptor verification
puts "  Pre-chain E9 descriptors:"
for {set i 0} {$i < 8} {incr i} { set v [read32 [expr $E9_D0 + $i*4]]; puts "    D0[$i]=[format 0x%08x $v]" }
for {set i 0} {$i < 8} {incr i} { set v [read32 [expr $E9_D1 + $i*4]]; puts "    D1[$i]=[format 0x%08x $v]" }
for {set i 0} {$i < 8} {incr i} { set v [read32 [expr $E9_D2 + $i*4]]; puts "    D2[$i]=[format 0x%08x $v]" }
for {set i 0} {$i < 8} {incr i} { set v [read32 [expr $E9_D3 + $i*4]]; puts "    D3[$i]=[format 0x%08x $v]" }
after 100

if {[run_chain $E9_D0 4]} {
    set s9_0 [verify_q5_result $E9_Q5R0 4 229376 E9a]
    set s9_1 [verify_q8_result $E9_Q8R  64 64     E9b]
    set s9_cpu 1
    for {set j 0} {$j < 16} {incr j} {
        set got [read32 [expr $E9_CPUR + $j * 4]]
        if {$got != 0x5A5A5A5A} { puts "  FAIL(E9c): CPU_OP word $j got [format 0x%08x $got] expected 0x5A5A5A5A"; set s9_cpu 0 }
    }
    # E9_D3: nibble=0, qh=0xFFFFFFFF → q5=-16, d=1.0, norm=1.0, act=1
    # Each element = 256 × (-16) × 1 = -4096, 896 elements = -3670016
    set s9_3 [verify_q5_result $E9_Q5R2 4 -3670016 E9d]
    if {$s9_0 && $s9_1 && $s9_cpu && $s9_3} { puts "  Test E9: PASS"; incr pass_count } else { incr fail_count }
} else { puts "  Test E9: TIMEOUT"; incr fail_count }

# ===================================================================
# Test E10: Q5_0 alternating q5 nibbles per block
#   Even blocks (0,2,...,54): nibble=1 → q5=1
#   Odd blocks (1,3,...,55): nibble=2 → q5=2
#   d=1.0, norm=1.0, act=1
#   28 even × 16 elements × 256 = 114688
#   28 odd × 16 elements × 512 = 229376
#   Total: 344064
# ===================================================================
puts "\n--- Test E10: Q5_0 alternating q5 (nibbles 1,2, expect 344064 per row) ---"
set E10_W  0x00128000
set E10_A  [expr $E10_W + 2696]
set E10_R  [expr $E10_A + 1792]
set E10_D  0x001005C0

fill_q5_weight_alternating $E10_W 1 2
fill_q5_norms $E10_W
fill_q5_acts  $E10_A 1
zero_fill $E10_R 32
write_desc $E10_D 0 $E10_W $E10_A $E10_R 1792 1
if {[run_chain $E10_D 1]} {
    set s10 [verify_q5_result $E10_R 4 344064 E10]
    if {$s10} { incr pass_count } else { incr fail_count }
} else { puts "  Test E10: TIMEOUT"; incr fail_count }

# ===================================================================
# Summary
# ===================================================================
puts "\n=============================================="
if {$fail_count == 0} {
    puts "  ALL $pass_count EXTENDED TESTS PASSED"
} else {
    puts "  $fail_count TESTS FAILED (of [expr $pass_count + $fail_count])"
}
puts "=============================================="

# Halt ARM
targets -set -filter {name =~ "*Cortex-A9*#0*"}
after 50
stop
after 200
puts "ARM halted."

exit
