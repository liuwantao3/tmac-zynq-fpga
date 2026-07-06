# HP FSM Q5_0 Compute Hardware Test
# All-1s pattern: d=1.0 (fp16 0x3C00), qh=0xFFFFFFFF, qs=0x11 (q5=1),
#   row_scale=0x0100 (UQ8.8 1.0), act=1 (int16)
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

# Build Q5_0 weight data in separated format (2464 bytes total):
#   Core0 headers: 56×6B = 336B at base+0
#   Core1 headers: 56×6B = 336B at base+336
#   Core0 qs: 56×16B = 896B at base+672
#   Core1 qs: 56×16B = 896B at base+1568
#   Scales: 8B at base+2464 (handled separately)
proc fill_q5_weight {base q5_nibble} {
    set qs_byte [expr ($q5_nibble << 4) | $q5_nibble]
    set bytes {}
    # Core0 headers: 56 blocks × 6 bytes
    for {set blk 0} {$blk < 56} {incr blk} {
        lappend bytes 0x00 0x3C 0xFF 0xFF 0xFF 0xFF
    }
    # Core1 headers: 56 blocks × 6 bytes
    for {set blk 0} {$blk < 56} {incr blk} {
        lappend bytes 0x00 0x3C 0xFF 0xFF 0xFF 0xFF
    }
    # Core0 qs: 56 blocks × 16 bytes
    for {set blk 0} {$blk < 56} {incr blk} {
        for {set k 0} {$k < 16} {incr k} { lappend bytes $qs_byte }
    }
    # Core1 qs: 56 blocks × 16 bytes
    for {set blk 0} {$blk < 56} {incr blk} {
        for {set k 0} {$k < 16} {incr k} { lappend bytes $qs_byte }
    }
    # Now total should be 336 + 336 + 896 + 896 = 2464 bytes = 616 × 32-bit words
    if {[llength $bytes] != 2464} { puts "ERROR: byte list length [llength $bytes] != 2464"; return }
    for {set i 0} {$i < 616} {incr i} {
        set b0 [lindex $bytes [expr $i*4]]
        set b1 [lindex $bytes [expr $i*4+1]]
        set b2 [lindex $bytes [expr $i*4+2]]
        set b3 [lindex $bytes [expr $i*4+3]]
        set word [expr {($b3 << 24) | ($b2 << 16) | ($b1 << 8) | $b0}]
        write32 [expr $base + $i*4] $word
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
        set state_names {IDLE FETCH_DESC FETCH_DESC_W LOAD_ACT LOAD_ACT_W WRITE_RES WRITE_RES_W DONE LOAD_WEIGHT LOAD_WEIGHT_W LOAD_SCALES LOAD_SCALES_W COPY_ACT_TO_CORE COMPUTE COMPUTE_W READ_RES READ_RES_ACC COPY_ACC_TO_BUF TIMEOUT_ERROR WRITE_RES_BURST}
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
# Q5_0 Compute Test — all-1s pattern
#   d = fp16(1.0) = 0x3C00
#   qh = 0xFFFFFFFF (all hyper-bits set)
#   qs = 0x11 (ql_nibble=1)
#   Decoded: q5 = (1<<4 | 1) - 16 = 1
#   row_scale = UQ8.8 1.0 = 0x0100
#   act = int16 1
#   Each MAC = d_fp * q5 * scale >> 8 * act
#            = 256 * 1 * 256 >> 8 * 1 = 256
#   Each row = 896 * 256 = 229376 (S24.8 fixed-point)
# ===================================================================
puts "\n--- Test Q5_0: all-1s pattern (expect each row = 229376) ---"

set Q5_WEIGHT_ADDR  0x00103000
set Q5_ACT_ADDR     0x00106000
set Q5_RES_ADDR     0x00107000
set Q5_DESC_ADDR    0x00100280

# --- Write Q5_0 weight data: separated headers+qs, 2464 bytes ---
puts "  Writing Q5_0 weight data (2464 bytes)..."
fill_q5_weight $Q5_WEIGHT_ADDR 1

puts "  Writing row scales (8 bytes, UQ8.8=1.0)..."
fill_q5_scales $Q5_WEIGHT_ADDR

# --- Write activations: 896 x int16 = 1792 bytes at ACT_ADDR ---
# All values = 1 (int16). As 32-bit words (2 per word): 0x00010001
puts "  Writing activations (1792 bytes)..."
set act_word [expr {0x0001 | (0x0001 << 16)}]
for {set j 0} {$j < [expr 1792 / 4]} {incr j} {
    write32 [expr $Q5_ACT_ADDR + $j * 4] $act_word
}

# --- Zero result buffer: 64 bytes (8 rows x 8 bytes) ---
puts "  Zeroing result buffer (64 bytes)..."
for {set j 0} {$j < 16} {incr j} {
    write32 [expr $Q5_RES_ADDR + $j * 4] 0
}

# --- Write descriptor: tensor_type=1 (Q5_0), act_total_bytes=1792 ---
write_desc $Q5_DESC_ADDR 0 $Q5_WEIGHT_ADDR $Q5_ACT_ADDR $Q5_RES_ADDR 1792 1

puts "  Descriptor at [format 0x%08x $Q5_DESC_ADDR]:"
puts "    next=0  weight=[format 0x%08x $Q5_WEIGHT_ADDR]  act=[format 0x%08x $Q5_ACT_ADDR]  res=[format 0x%08x $Q5_RES_ADDR]"
puts "    tensor_type=1 (Q5_0)  act_total_bytes=1792"

# --- Run chain ---
if {[run_chain $Q5_DESC_ADDR 1]} {
    set status [gp0_read $REG_STATUS]
    set head [gp0_read $REG_DESC_HEAD]
    set dbg [gp0_read $REG_DEBUG]
    set q8_dbg [gp0_read $REG_Q8_DEBUG]
    puts "  STATUS=[format 0x%08x $status] (expect 0x300) HEAD=$head (expect 1)"
    puts "  REG_DEBUG=[format 0x%08x $dbg]"
    puts "  REG_Q8_DEBUG=[format 0x%08x $q8_dbg]"

    # --- Verify results: 4 rows, each 8 bytes (48-bit S24.8 fixed-point) ---
    # Expected: 229376 (896 columns x q5=1 x act=1, S24.8 = 896 << 8)
    set ok 1
    set expected 229376
    for {set j 0} {$j < 4} {incr j} {
        set addr [expr $Q5_RES_ADDR + $j * 8]
        set lo [read32 $addr]
        set hi [read32 [expr $addr + 4]]
        # Construct 48-bit value: {hi[15:0], lo[31:0]}
        set got_48 [expr {($hi << 32) | $lo}]
        # Interpret as signed 48-bit
        if {$got_48 >= [expr {1 << 47}]} {
            set got_signed [expr {$got_48 - (1 << 48)}]
        } else {
            set got_signed $got_48
        }
        if {$got_signed == $expected} {
            puts "  Row $j: PASS (got $got_signed)"
        } else {
            puts "  Row $j: FAIL (lo=[format 0x%08x $lo] hi=[format 0x%08x $hi] got=$got_signed expected=$expected)"
            set ok 0
        }
    }
    if {$ok} {
        puts "  Q5_0 Test: PASS (all 4 rows = $expected)"
    } else {
        puts "  Q5_0 Test: FAIL"
    }
} else {
    puts "  Q5_0 Test: TIMEOUT"
}

# ===================================================================
# Test 2: Chain of 2 Q5_0 descriptors
#   Desc 0: all-1s -> expect 229376 per row
#   Desc 1: all-0s -> expect 0 per row
# ===================================================================
puts "\n--- Test 2: Chain of 2 Q5_0 (all-1s -> all-0s) (4 rows each) ---"

# Helper to fill Q5_0 weight (112 blocks = 2464 bytes) at given base addr
# Uses efficient 32-bit writes. Each block: d=1.0, qh=0xFFFFFFFF,
# qs nibble = q5_nibble for all 32 weights.
proc fill_q5_weight {base q5_nibble} {
    set qs_byte [expr ($q5_nibble << 4) | $q5_nibble]
    # Build 2464-byte list
    set bytes {}
    for {set blk 0} {$blk < 112} {incr blk} {
        lappend bytes 0x00 0x3C 0xFF 0xFF 0xFF 0xFF
        for {set k 0} {$k < 16} {incr k} { lappend bytes $qs_byte }
    }
    # Write as 32-bit words (616 writes)
    for {set i 0} {$i < 616} {incr i} {
        set b0 [lindex $bytes [expr $i*4]]
        set b1 [lindex $bytes [expr $i*4+1]]
        set b2 [lindex $bytes [expr $i*4+2]]
        set b3 [lindex $bytes [expr $i*4+3]]
        set word [expr {($b3 << 24) | ($b2 << 16) | ($b1 << 8) | $b0}]
        write32 [expr $base + $i*4] $word
    }
}

# Helper to fill scales at weight_base + 2464
# Scale = 0x0100 (UQ8.8 = 1.0). Little-endian: byte0=0x00, byte1=0x01.
# After FSM byte swap: q5_sc_din = {0x01, 0x00} = 0x0100.
proc fill_q5_scales {weight_base} {
    # Two scales per 32-bit word: scale[0]=0x0100, scale[1]=0x0100
    # In little-endian: addr+0=0x00, addr+1=0x01, addr+2=0x00, addr+3=0x01
    # So the 32-bit value is 0x00 + (0x01<<8) + (0x00<<16) + (0x01<<24) = 0x01000100
    set sc_word 0x01000100
    for {set j 0} {$j < 2} {incr j} {
        write32 [expr $weight_base + 2464 + $j * 4] $sc_word
    }
}

# Helper to fill activations
proc fill_q5_acts {base val} {
    set act_word [expr {($val & 0xFFFF) | (($val & 0xFFFF) << 16)}]
    for {set j 0} {$j < [expr 1792 / 4]} {incr j} {
        write32 [expr $base + $j * 4] $act_word
    }
}

# Helper to verify 8 rows at result_addr
proc verify_q5_result {res_addr expected test_id} {
    set ok 1
    for {set j 0} {$j < 4} {incr j} {
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

# Zero 64 bytes
proc zero_res {addr} {
    for {set j 0} {$j < 16} {incr j} {
        write32 [expr $addr + $j * 4] 0
    }
}

set Q5_DESC0_ADDR 0x001002C0
set Q5_DESC1_ADDR 0x00100300
set Q5_W1_ADDR    0x00108000
set Q5_A1_ADDR    0x00109000
set Q5_R0_ADDR    0x0010A000
set Q5_R1_ADDR    0x0010A040

# Write weight/act for desc 0 (all-1s -> 896)
puts "  Writing desc 0 weight/act (all-1s)..."
fill_q5_weight $Q5_WEIGHT_ADDR 1
fill_q5_scales  $Q5_WEIGHT_ADDR
fill_q5_acts    $Q5_ACT_ADDR 1
zero_res $Q5_RES_ADDR

# Write weight/act for desc 1 (all-0s -> 0)
puts "  Writing desc 1 weight/act (all-0s)..."
fill_q5_weight $Q5_W1_ADDR 0
fill_q5_scales  $Q5_W1_ADDR
fill_q5_acts    $Q5_A1_ADDR 1
zero_res $Q5_R1_ADDR

# Chain: desc0 -> desc1 -> 0
write_desc $Q5_DESC0_ADDR $Q5_DESC1_ADDR $Q5_WEIGHT_ADDR $Q5_ACT_ADDR $Q5_R0_ADDR 1792 1
write_desc $Q5_DESC1_ADDR 0 $Q5_W1_ADDR $Q5_A1_ADDR $Q5_R1_ADDR 1792 1

if {[run_chain $Q5_DESC0_ADDR 2]} {
    set head [gp0_read $REG_DESC_HEAD]
    puts "  HEAD=$head (expect 2)"
    set s0 [verify_q5_result $Q5_R0_ADDR 229376 2]
    set s1 [verify_q5_result $Q5_R1_ADDR 0 2]
    if {$s0 && $s1} {
        puts "  Test 2: PASS"
    } else {
        puts "  Test 2: FAIL"
    }
} else {
    puts "  Test 2: TIMEOUT"
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