# HP FSM Q5_0 Compute Hardware Test
# Interleaved 48-byte per-block DDR layout (2 cores, 56 blocks/tile).
# All-1s pattern: d=1.0 (fp16 0x3C00), qh=0xFFFFFFFF, qs nibble=1 (q5=1),
#   row_norm=UQ8.8 1.0 (0x0100), act=1 (int16)
# Expected result: each row = 229376 (896 columns x q5=1 x act=1, S24.8 fixed-point)
#
# Power-cycle the board before running!
# C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_hp_fsm_q5_0.tcl

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
# num_groups: Q8 column groups (ignored by Q5_0)
# num_tiles: tiles per descriptor (bytes [22:23], 0→1 backward compat)
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

# Build Q5_0 weight in 48-byte per-block interleaved format (2696 bytes/tile):
#   56 blocks × 48 bytes = 2688 bytes, + 8 bytes norms at offset 2688
# Per block (12 × 32-bit writes):
#   Word  0: d0=0x3C00 | qh0_low=0xFFFF << 16           → 0xFFFF3C00
#   Word  1: qh0_hi=0xFFFF | (qs_word_low << 16)         → 0xQQQQFFFF
#   Word  2: qs_word                                       → 0xQQQQQQQQ
#   Word  3: qs_word                                       → 0xQQQQQQQQ
#   Word  4: qs_word                                       → 0xQQQQQQQQ
#   Word  5: d1=0x3C00 << 16 | qs_word_low                → 0x3C00QQQQ
#   Word  6: qh1=0xFFFFFFFF                               → 0xFFFFFFFF
#   Word  7: qs_word                                       → 0xQQQQQQQQ
#   Word  8: qs_word                                       → 0xQQQQQQQQ
#   Word  9: qs_word                                       → 0xQQQQQQQQ
#   Word 10: qs_word                                       → 0xQQQQQQQQ
#   Word 11: padding=0                                     → 0x00000000
proc fill_q5_weight {base q5_nibble} {
    set qs_byte [expr ($q5_nibble << 4) | $q5_nibble]
    set qs_word32 [expr {$qs_byte * 0x01010101}]
    set qs_word_low16 [expr {$qs_byte | ($qs_byte << 8)}]
    set word1_val [expr {($qs_word_low16 << 16) | 0x0000FFFF}]
    set word5_val [expr {(0x3C00 << 16) | $qs_word_low16}]
    for {set blk 0} {$blk < 56} {incr blk} {
        set bo [expr $base + $blk * 48]
        write32 [expr $bo + 0]   0xFFFF3C00
        write32 [expr $bo + 4]   $word1_val
        write32 [expr $bo + 8]   $qs_word32
        write32 [expr $bo + 12]  $qs_word32
        write32 [expr $bo + 16]  $qs_word32
        write32 [expr $bo + 20]  $word5_val
        write32 [expr $bo + 24]  0xFFFFFFFF
        write32 [expr $bo + 28]  $qs_word32
        write32 [expr $bo + 32]  $qs_word32
        write32 [expr $bo + 36]  $qs_word32
        write32 [expr $bo + 40]  $qs_word32
        write32 [expr $bo + 44]  0x00000000
    }
}

# Fill row_norms at weight_base + 2688: 4 × UQ8.8 1.0 = 0x0100
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

# Verify rows at result_addr (nrows, each 8 bytes S24.8)
proc verify_q5_result {res_addr nrows expected test_id} {
    set ok 1
    for {set j 0} {$j < $nrows} {incr j} {
        set addr [expr $res_addr + $j * 8]
        set lo [read32 $addr]
        set hi [read32 [expr $addr + 4]]
        set got_signed [expr {($hi << 32) | $lo}]
        if {$got_signed >= [expr {1 << 47}]} {
            set got_signed [expr {$got_signed - (1 << 48)}]
        }
        if {$got_signed == $expected} {
            puts "  Row $j: PASS (got $got_signed)"
        } else {
            puts "  FAIL[${test_id}]: row $j lo=[format 0x%08x $lo] hi=[format 0x%08x $hi] got=$got_signed expected=$expected"
            set ok 0
        }
    }
    return $ok
}

# Zero result buffer
proc zero_res {addr nbytes} {
    for {set j 0} {$j < [expr $nbytes / 4]} {incr j} {
        write32 [expr $addr + $j * 4] 0
    }
}

# Start chain and wait for HEAD to reach expected_head
proc run_chain {desc_base expected_head} {
    global REG_DESC_BASE REG_DESC_TAIL REG_START REG_DESC_HEAD REG_STATUS REG_DEBUG REG_Q8_DEBUG REG_ACT_INFO REG_DESC_INFO
    gp0_write $REG_DESC_BASE $desc_base
    after 10
    gp0_write $REG_DESC_TAIL 1
    after 10
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
        puts "  FSM state=$state_mask ($sname), rd_busy=[expr ($dbg>>24)&1] wr_busy=[expr ($dbg>>23)&1] rd_done=[expr ($dbg>>26)&1] wr_done=[expr ($dbg>>25)&1]"
        set act_info [gp0_read $REG_ACT_INFO]
        set desc_info [gp0_read $REG_DESC_INFO]
        puts "  REG_ACT_INFO=[format 0x%08x $act_info] REG_DESC_INFO=[format 0x%08x $desc_info]"
        return 0
    }
    return 1
}

# ===== MAIN =====
puts "=============================================="
puts "  HP FSM Q5_0 Compute Hardware Test"
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

# ===================================================================
# Test 1: Single Q5_0 descriptor, all-1s pattern
#   q5_nibble=1 → q5=1, d=1.0, norm=1.0, act=1
#   Per row = 896 × 256 = 229376
# ===================================================================
puts "\n--- Test 1: Single Q5_0 all-1s (4 rows, expect 229376 each) ---"

set W1_ADDR  0x00103000
set A1_ADDR  0x00104000
set R1_ADDR  0x00104800
set D1_ADDR  0x00100280

puts "  Writing Q5_0 weight (2696 bytes, 56 blocks)..."
fill_q5_weight $W1_ADDR 1
fill_q5_norms  $W1_ADDR
fill_q5_acts   $A1_ADDR 1
zero_res $R1_ADDR 32

write_desc $D1_ADDR 0 $W1_ADDR $A1_ADDR $R1_ADDR 1792 1 0 1

if {[run_chain $D1_ADDR 1]} {
    set s1 [verify_q5_result $R1_ADDR 4 229376 1]
    puts "  Test 1: [expr {$s1 ? "PASS" : "FAIL"}]"
} else {
    puts "  Test 1: TIMEOUT"
}

# ===================================================================
# Test 2: Chain of 2 Q5_0 descriptors
#   Desc 0: all-1s → 229376 per row
#   Desc 1: all-0s (qs nibble=0, q5=0) → 0 per row
# ===================================================================
puts "\n--- Test 2: Chain of 2 Q5_0 (all-1s → all-0s) ---"

set D2_0_ADDR  0x001002C0
set D2_1_ADDR  0x00100300
set W2_0_ADDR  0x00103000
set W2_1_ADDR  0x00105000
set A2_0_ADDR  0x00104000
set A2_1_ADDR  0x00105800
set R2_0_ADDR  0x00104800
set R2_1_ADDR  0x00106000

puts "  Writing desc 0 (all-1s)..."
fill_q5_weight $W2_0_ADDR 1
fill_q5_norms  $W2_0_ADDR
fill_q5_acts   $A2_0_ADDR 1
zero_res $R2_0_ADDR 32

puts "  Writing desc 1 (all-0s, q5=0, expect 0)..."
fill_q5_weight $W2_1_ADDR 0
fill_q5_norms  $W2_1_ADDR
fill_q5_acts   $A2_1_ADDR 1
zero_res $R2_1_ADDR 32

write_desc $D2_0_ADDR $D2_1_ADDR $W2_0_ADDR $A2_0_ADDR $R2_0_ADDR 1792 1
write_desc $D2_1_ADDR 0          $W2_1_ADDR $A2_1_ADDR $R2_1_ADDR 1792 1

if {[run_chain $D2_0_ADDR 2]} {
    set head [gp0_read $REG_DESC_HEAD]
    puts "  HEAD=$head (expect 2)"
    set s2_0 [verify_q5_result $R2_0_ADDR 4 229376 2]
    set s2_1 [verify_q5_result $R2_1_ADDR 4 0 2]
    puts "  Test 2: [expr {$s2_0 && $s2_1 ? "PASS" : "FAIL"}]"
} else {
    puts "  Test 2: TIMEOUT"
}

# ===================================================================
# Test 3: Multi-tile Q5_0 — 2 tiles × 4 rows = 8 rows
#   num_tiles=2, all weights=1, act=1
#   Expected: all 8 rows = 229376
#   Tile 0 at R3_ADDR+0, Tile 1 at R3_ADDR+32
# ===================================================================
puts "\n--- Test 3: Multi-tile Q5_0 (2 tiles, 8 rows, expect 229376 each) ---"

set D3_ADDR    0x00100340
set W3_ADDR    0x00107000
set A3_ADDR    0x00108000
set R3_ADDR    0x00108800

# Tile 0 weight (2696 bytes)
puts "  Writing tile 0 weight..."
fill_q5_weight $W3_ADDR 1
fill_q5_norms  $W3_ADDR

# Tile 1 weight at W3_ADDR + 2696
puts "  Writing tile 1 weight..."
fill_q5_weight [expr $W3_ADDR + 2696] 1
fill_q5_norms  [expr $W3_ADDR + 2696]

# Shared activation
fill_q5_acts $A3_ADDR 1

# Result buffer: 2 tiles × 32 bytes = 64 bytes
zero_res $R3_ADDR 64

# Descriptor: num_tiles=2 -> process both tiles in one go
write_desc $D3_ADDR 0 $W3_ADDR $A3_ADDR $R3_ADDR 1792 1 0 2

if {[run_chain $D3_ADDR 1]} {
    set s3_0 [verify_q5_result $R3_ADDR 4 229376 3t0]
    set s3_1 [verify_q5_result [expr $R3_ADDR + 32] 4 229376 3t1]
    puts "  Test 3: [expr {$s3_0 && $s3_1 ? "PASS" : "FAIL"}]"
} else {
    puts "  Test 3: TIMEOUT"
}

# ===================================================================
# Summary
# ===================================================================
puts "\n=============================================="
puts "  Q5_0 Hardware Test Complete"
puts "=============================================="

# Halt ARM for clean state
targets -set -filter {name =~ "*Cortex-A9*#0*"}
after 50
stop
after 200
puts "ARM halted."

exit
