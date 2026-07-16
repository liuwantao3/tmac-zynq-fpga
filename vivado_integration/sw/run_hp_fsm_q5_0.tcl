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
set REG_Q5_DEBUG  0x40
set REG_Q5_DBG_CAP0  0x44
set REG_Q5_DBG_CAP1  0x48
set REG_Q5_DBG_TRIG  0x4C
set REG_Q5_DBG_LIVE  0x50
set REG_Q5_DBG_CAP2  0x54
set REG_Q5_DBG_CAP3  0x58
set REG_Q5_DBG_CAP4  0x5C
set REG_Q5_DBG_CAP5  0x60
set REG_Q5_DBG_SNAP  0x64
set REG_Q5_DBG_WI_START 0x68

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
    # qh always all-1s: in Q5_0 encoding, unsigned_5bit = {qh, ql}, q5 = unsigned - 16.
    # To get q5 = q5_nibble, need unsigned = 16 + q5_nibble → {qh=1, ql=q5_nibble}.
    # qh=0 gives unsigned in [0,15] → q5 negative; only correct for negative Q5_0 values.
    set qh_word 0xFFFFFFFF
    # Word 0 = {qh0_low(16), d0(16)}
    set word0_val [expr {($qh_word & 0xFFFF) << 16 | 0x3C00}]
    # Word 1 = {qs_word_low16(16), qh0_hi(16)}
    set word1_val [expr {($qs_word_low16 << 16) | (($qh_word >> 16) & 0xFFFF)}]
    # Word 5 = {d1=0x3C00(16), qs_word_low16(16)}
    set word5_val [expr {(0x3C00 << 16) | $qs_word_low16}]
    # Word 6 = qh1
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

# Q5_0 debug register dump
# cap0 (0x44): [31:16]=core0_d_pre, [15:0]=core1_d_pre
# cap1 (0x48): [31:16]=core0_d_fp[15:0], [15:9]=norm[15:9], [8:6]=core0_state, [5:0]=blk_counter
# trig (0x4C): [31:16]=live core0_d_pre, [15:10]=trig_blk, [9]=frozen, [8]=busy,
#              [7:5]=core0_state, [0]=armed
proc q5_dbg_dump {} {
    global REG_Q5_DBG_CAP0 REG_Q5_DBG_CAP1 REG_Q5_DBG_CAP2 REG_Q5_DBG_CAP3 REG_Q5_DBG_CAP4 REG_Q5_DBG_CAP5 REG_Q5_DBG_TRIG REG_Q5_DBG_LIVE REG_Q5_DBG_WI_START REG_Q5_DBG_SNAP
    set cap0 [gp0_read $REG_Q5_DBG_CAP0]
    set cap1 [gp0_read $REG_Q5_DBG_CAP1]
    set cap2 [gp0_read $REG_Q5_DBG_CAP2]
    set cap3 [gp0_read $REG_Q5_DBG_CAP3]
    set cap4 [gp0_read $REG_Q5_DBG_CAP4]
    set cap5 [gp0_read $REG_Q5_DBG_CAP5]
    set trig [gp0_read $REG_Q5_DBG_TRIG]
    set live [gp0_read $REG_Q5_DBG_LIVE]
    set snap [gp0_read $REG_Q5_DBG_SNAP]
    set core0_dpre [expr {($cap0 >> 16) & 0xFFFF}]
    set core1_dpre [expr {$cap0 & 0xFFFF}]
    set core0_dfp_lo [expr {($cap1 >> 16) & 0xFFFF}]
    set norm [expr {($cap1 >> 9) & 0x7F}]
    set state [expr {($cap1 >> 6) & 0x7}]
    set blk [expr {$cap1 & 0x3F}]
    set live_dpre_c0 [expr {($trig >> 16) & 0xFFFF}]
    set frozen [expr {($trig >> 9) & 1}]
    set busy [expr {($trig >> 8) & 1}]
    set armed [expr {$trig & 1}]
    set live_state [expr {($live >> 22) & 0x1F}]
    set live_frozen [expr {($live >> 21) & 1}]
    set live_busy [expr {($live >> 20) & 1}]
    set live_tile [expr {($live >> 14) & 0x3F}]
    set live_blk [expr {($live >> 8) & 0x3F}]
    set live_cstate [expr {($live >> 2) & 0x7}]
    # Snap (0x64): [31:27]=c0_wi, [26:22]=c1_wi, [21:17]=c0_q5, [16:12]=c1_q5,
    #               [11:9]=c0_state, [8:6]=c1_state, [5:0]=blk_counter
    set snap_c0_wi [expr {($snap >> 27) & 0x1F}]
    set snap_c1_wi [expr {($snap >> 22) & 0x1F}]
    set snap_c0_q5 [expr {($snap >> 17) & 0x1F}]
    set snap_c1_q5 [expr {($snap >> 12) & 0x1F}]
    set snap_c0_state [expr {($snap >> 9) & 0x7}]
    set snap_c1_state [expr {($snap >> 6) & 0x7}]
    set snap_blk [expr {$snap & 0x3F}]
    set core_state_names {IDLE SETUP_D SETUP_D2 SETUP_D3 SETUP_D4 COMPUTE DRAIN}
    if {$snap_c0_state < [llength $core_state_names]} { set snap_c0_sname [lindex $core_state_names $snap_c0_state] } else { set snap_c0_sname "UNK" }
    if {$snap_c1_state < [llength $core_state_names]} { set snap_c1_sname [lindex $core_state_names $snap_c1_state] } else { set snap_c1_sname "UNK" }
    set state_names {IDLE FETCH_DESC FETCH_DESC_W LOAD_ACT LOAD_ACT_W WRITE_RES WRITE_RES_W DONE LOAD_WEIGHT LOAD_WEIGHT_W LOAD_SCALES LOAD_SCALES_W COPY_ACT_TO_CORE COMPUTE COMPUTE_W READ_RES READ_RES_ACC COPY_ACC_TO_BUF TIMEOUT_ERROR WRITE_RES_BURST Q5_LOAD_NORM Q5_LOAD_NORM_W Q5_COPY_ACT Q5_COPY_ACT_W Q5_BLOCK_COMPUTE Q5_BLOCK_COMPUTE_W Q5_READ_RES}
    set core_state_names {IDLE SETUP_D SETUP_D2 SETUP_D3 SETUP_D4 COMPUTE DRAIN}
    if {$live_state < [llength $state_names]} { set lname [lindex $state_names $live_state] } else { set lname "UNK" }
    if {$live_cstate < [llength $core_state_names]} { set cname [lindex $core_state_names $live_cstate] } else { set cname "UNK" }
    puts "  DBG: cap0=0x[format %08x $cap0] cap1=0x[format %08x $cap1] trig=0x[format %08x $trig] live=0x[format %08x $live]"
    # cap4: [31:27]=c0_q5, [26:16]=0, [15:0]=c0_act_r
    set core0_q5 [expr {($cap4 >> 27) & 0x1F}]
    set core0_act_r [expr {$cap4 & 0xFFFF}]
    # cap5: [31:27]=c1_q5, [26:16]=0, [15:0]=c1_act_r
    set core1_q5 [expr {($cap5 >> 27) & 0x1F}]
    set core1_act_r [expr {$cap5 & 0xFFFF}]
    puts "  DBG: cap2=0x[format %08x $cap2] cap3=0x[format %08x $cap3]"
    puts "  DBG: cap4=0x[format %08x $cap4] cap5=0x[format %08x $cap5]"
    puts "  DBG: c0_q5=$core0_q5 c0_act_r=$core0_act_r c1_q5=$core1_q5 c1_act_r=$core1_act_r"
    puts "  DBG: c0_d_pre=$core0_dpre c1_d_pre=$core1_dpre c0_d_fp_lo=$core0_dfp_lo norm=$norm blk=$blk core_state=$cname"
    puts "  DBG: live_c0_d_pre=$live_dpre_c0 frozen=$frozen busy=$busy armed=$armed"
    puts "  DBG: FSM=$lname tile=$live_tile blk=$live_blk core=$cname frozen=$live_frozen busy=$live_busy"
    puts "  DBG: core1.res1_lo=0x[format %08x $cap2] core1.res1_hi=0x[format %04x [expr ($cap3 >> 16) & 0xFFFF]] core1.res0_lo=[expr $cap3 & 0xFFFF]"
    puts "  DBG: snap=0x[format %08x $snap] c0_wi=$snap_c0_wi c1_wi=$snap_c1_wi c0_q5=$snap_c0_q5 c1_q5=$snap_c1_q5 c0_state=$snap_c0_sname c1_state=$snap_c1_sname blk=$snap_blk"
    # wi_start: captures wi on first blk_entry after clr_acc (0x68)
    set wi_start [gp0_read $REG_Q5_DBG_WI_START]
    set c0_wi [expr {($wi_start >> 27) & 0x1F}]
    set c1_wi [expr {($wi_start >> 22) & 0x1F}]
    set wi_blk [expr {$wi_start & 0x3F}]
    puts "  DBG: wi_start=0x[format %08x $wi_start] c0_wi=$c0_wi c1_wi=$c1_wi blk=$wi_blk"
}

# Start chain and wait for HEAD to reach expected_head
proc run_chain {desc_base expected_head} {
    global REG_DESC_BASE REG_DESC_TAIL REG_START REG_DESC_HEAD REG_STATUS REG_DEBUG REG_Q8_DEBUG REG_Q5_DEBUG REG_ACT_INFO REG_DESC_INFO REG_CLK_CNT
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
        set rd_state [expr ($dbg>>16) & 0x7]
        set rd_state_names {IDLE SEND_AR READ_BEAT PRESENT}
        set rd_sname [lindex $rd_state_names $rd_state]
        puts "  FSM state=$state_mask ($sname), rd_state=$rd_state ($rd_sname) rd_busy=[expr ($dbg>>24)&1] wr_busy=[expr ($dbg>>23)&1] rd_done=[expr ($dbg>>26)&1] wr_done=[expr ($dbg>>25)&1]"
        set act_info [gp0_read $REG_ACT_INFO]
        set desc_info [gp0_read $REG_DESC_INFO]
        set afi_status [read32 0xF8008014]
        set afi_ctrl [read32 0xF8008000]
        puts "  AFI_STATUS=[format 0x%08x $afi_status] AFI_CTRL=[format 0x%08x $afi_ctrl]"
        set rd_fifo [expr {($afi_status >> 0) & 0xFF}]
        set wr_fifo [expr {($afi_status >> 8) & 0xFF}]
        puts "  RD_FIFO_CNT=$rd_fifo WR_FIFO_CNT=$wr_fifo"
        puts "  REG_ACT_INFO=[format 0x%08x $act_info] REG_DESC_INFO=[format 0x%08x $desc_info]"
        set q5_dbg [gp0_read $REG_Q5_DEBUG]
        set q5_unpack_word [expr {($q5_dbg >> 15) & 0x3F}]
        set q5_blk_ctr [expr {($q5_dbg >> 9) & 0x3F}]
        puts "  REG_Q5_DEBUG=[format 0x%08x $q5_dbg] any_busy=[expr {($q5_dbg >> 24) & 1}] busy_both=[expr {($q5_dbg >> 23) & 1}] done0=[expr {($q5_dbg >> 22) & 1}] done1=[expr {($q5_dbg >> 21) & 1}] blk_valid=[expr {($q5_dbg >> 25) & 1}] rd_unpack=[expr {($q5_dbg >> 26) & 1}]"
        puts "  q5_unpack_word=$q5_unpack_word q5_blk_counter=$q5_blk_ctr"
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
# DDR readback verification
set w0 [read32 $W1_ADDR]
set w1 [read32 [expr {$W1_ADDR + 4}]]
set w48 [read32 [expr {$W1_ADDR + 2688}]]
set a0 [read32 $A1_ADDR]
puts "  DDR check: W[0]=[format 0x%08x $w0] W[1]=[format 0x%08x $w1] Norms=[format 0x%08x $w48] Act[0]=[format 0x%08x $a0]"

write_desc $D1_ADDR 0 $W1_ADDR $A1_ADDR $R1_ADDR 1792 1 0 1

if {[run_chain $D1_ADDR 1]} {
    set s1 [verify_q5_result $R1_ADDR 4 229376 1]
    q5_dbg_dump
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
set A2_1_ADDR  0x00106800
set R2_0_ADDR  0x00104800
set R2_1_ADDR  0x00106000

puts "  Writing desc 0 (all-1s)..."
fill_q5_weight $W2_0_ADDR 1
fill_q5_norms  $W2_0_ADDR
fill_q5_acts   $A2_0_ADDR 1
zero_res $R2_0_ADDR 32

puts "  Writing desc 1 (all-0s, q5=0, act=2, expect 0)..."
fill_q5_weight $W2_1_ADDR 0
fill_q5_acts   $A2_1_ADDR 2
fill_q5_norms  $W2_1_ADDR
zero_res $R2_1_ADDR 32

write_desc $D2_0_ADDR $D2_1_ADDR $W2_0_ADDR $A2_0_ADDR $R2_0_ADDR 1792 1
write_desc $D2_1_ADDR 0          $W2_1_ADDR $A2_1_ADDR $R2_1_ADDR 1792 1

if {[run_chain $D2_0_ADDR 2]} {
    set head [gp0_read $REG_DESC_HEAD]
    puts "  HEAD=$head (expect 2)"
    set s2_0 [verify_q5_result $R2_0_ADDR 4 229376 2]
    set s2_1 [verify_q5_result $R2_1_ADDR 4 0 2]
    # Force snapshot capture before reading debug registers
    gp0_write $REG_Q5_DBG_SNAP 0
    after 20
    q5_dbg_dump
    if {!$s2_1} {
        for {set i 0} {$i < 12} {incr i} { set addr [expr {$W2_1_ADDR + $i*4}]; set v [read32 $addr]; puts "    W[format %02d $i]=[format 0x%08x $v]" }
        set naddr [expr {$W2_1_ADDR + 2688}]
        puts "  === DIAG: norms ==="
        set n0 [read32 $naddr]; set n1 [read32 [expr $naddr+4]]
        puts "    N0=[format 0x%08x $n0] N1=[format 0x%08x $n1]"
        set GP0_BASE 0x43C00000
        puts "  === DIAG: Q5 regs live ==="
        puts "    DEBUG=[format 0x%08x [read32 [expr $GP0_BASE + 0x28]]]"
        puts "    Q5DBG=[format 0x%08x [read32 [expr $GP0_BASE + 0x40]]]"
        puts "    CAP0 =[format 0x%08x [read32 [expr $GP0_BASE + 0x44]]]"
        puts "    CAP1 =[format 0x%08x [read32 [expr $GP0_BASE + 0x48]]]"
        puts "    TRIG =[format 0x%08x [read32 [expr $GP0_BASE + 0x4C]]]"
        puts "    LIVE =[format 0x%08x [read32 [expr $GP0_BASE + 0x50]]]"
        puts "    CAP2 =[format 0x%08x [read32 [expr $GP0_BASE + 0x54]]]"
        puts "    CAP3 =[format 0x%08x [read32 [expr $GP0_BASE + 0x58]]]"
        puts "    CAP4 =[format 0x%08x [read32 [expr $GP0_BASE + 0x5C]]]"
        puts "    CAP5 =[format 0x%08x [read32 [expr $GP0_BASE + 0x60]]]"
    }
    # Unfreeze debug capture for subsequent tests
    gp0_write $REG_Q5_DBG_TRIG 0
    after 10
    puts "  Test 2: [expr {$s2_0 && $s2_1 ? "PASS" : "FAIL"}]"
} else {
    puts "  Test 2: TIMEOUT"
}

# ===================================================================
# Test 3: Single all-0s descriptor (no chain, q5=0, expect 0)
# ===================================================================
puts "\n--- Test 3: Single Q5_0 all-0s (q5=0, expect 0) ---"
set D3_ADDR    0x00100340
set W3_ADDR    0x00107000
set A3_ADDR    0x00108000
set R3_ADDR    0x00108800
fill_q5_weight $W3_ADDR 0
fill_q5_norms  $W3_ADDR
fill_q5_acts   $A3_ADDR 1
zero_res $R3_ADDR 32
write_desc $D3_ADDR 0 $W3_ADDR $A3_ADDR $R3_ADDR 1792 1
if {[run_chain $D3_ADDR 1]} {
    set s3 [verify_q5_result $R3_ADDR 4 0 3]
    q5_dbg_dump
    puts "  Test 3: [expr {$s3 ? "PASS" : "FAIL"}]"
} else {
    puts "  Test 3: TIMEOUT"
}

# ===================================================================
# Test 4: Multi-tile Q5_0 — 2 tiles × 4 rows = 8 rows
#   num_tiles=2, all weights=1, act=1
#   Expected: all 8 rows = 229376
#   Tile 0 at R4_ADDR+0, Tile 1 at R4_ADDR+32
# ===================================================================
puts "\n--- Test 4: Multi-tile Q5_0 (2 tiles, 8 rows, expect 229376 each) ---"

set D4_ADDR    0x00100380
set W4_ADDR    0x00107800
set A4_ADDR    0x00109000
set R4_ADDR    0x00109800

# Tile 0 weight (2696 bytes)
puts "  Writing tile 0 weight..."
fill_q5_weight $W4_ADDR 1
fill_q5_norms  $W4_ADDR

# Tile 1 weight at W4_ADDR + 2696
puts "  Writing tile 1 weight..."
fill_q5_weight [expr $W4_ADDR + 2696] 1
fill_q5_norms  [expr $W4_ADDR + 2696]

# Shared activation
fill_q5_acts $A4_ADDR 1

# Result buffer: 2 tiles × 32 bytes = 64 bytes
zero_res $R4_ADDR 64

# Descriptor: num_tiles=2 -> process both tiles in one go
write_desc $D4_ADDR 0 $W4_ADDR $A4_ADDR $R4_ADDR 1792 1 0 2

if {[run_chain $D4_ADDR 1]} {
    set s4_0 [verify_q5_result $R4_ADDR 4 229376 4t0]
    set s4_1 [verify_q5_result [expr $R4_ADDR + 32] 4 229376 4t1]
    q5_dbg_dump
    puts "  Test 4: [expr {$s4_0 && $s4_1 ? "PASS" : "FAIL"}]"
} else {
    puts "  Test 4: TIMEOUT"
}

# ===================================================================
# Test 5: Standalone all-0s with trigger armed (capture q5/act_r at blk=0)
# ===================================================================
puts "\n--- Test 5: Standalone all-0s, trigger armed for blk=0 ---"
set D5_ADDR    0x001003C0
set W5_ADDR    0x0010A000
set A5_ADDR    0x0010B000
set R5_ADDR    0x0010B800
fill_q5_weight $W5_ADDR 0
fill_q5_norms  $W5_ADDR
fill_q5_acts   $A5_ADDR 1
zero_res $R5_ADDR 32
write_desc $D5_ADDR 0 $W5_ADDR $A5_ADDR $R5_ADDR 1792 1
# Arm trigger: [15:10]=0 (blk=0), [0]=1 (arm)
gp0_write $REG_Q5_DBG_TRIG 0x0001
after 10
if {[run_chain $D5_ADDR 1]} {
    set s5 [verify_q5_result $R5_ADDR 4 0 5]
    q5_dbg_dump
    set frozen [expr {[gp0_read $REG_Q5_DBG_TRIG] >> 9 & 1}]
    puts "  Trigger frozen=$frozen (1=good, means capture fired)"
    puts "  Test 5: [expr {$s5 ? "PASS" : "FAIL"}]"
} else {
    puts "  Test 5: TIMEOUT"
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
