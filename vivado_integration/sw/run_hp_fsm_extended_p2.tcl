# HP FSM Extended Hardware Test — Part 2 (E9-E10)
# Power-cycle board first!
# C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vivado_integration/sw/run_hp_fsm_extended_p2.tcl

set DDR_BASE  0x00100000
set GP0_BASE  0x43C00000

proc read32 {addr} {
    set r [mrd $addr 1]
    if {[regexp {:\s+([0-9A-Fa-f]+)} $r full data]} { return [expr "0x$data"] }
    return -1
}
proc write32 {addr val} { mwr -force $addr $val }
proc gp0_read {reg} { global GP0_BASE; return [read32 [format 0x%08x [expr $GP0_BASE + $reg]]] }
proc gp0_write {reg val} { global GP0_BASE; write32 [format 0x%08x [expr $GP0_BASE + $reg]] $val }

proc zero_fill {addr n} { for {set j 0} {$j < $n/4} {incr j} { write32 [expr $addr+$j*4] 0 } }
proc write_pattern_const {addr n b} {
    set w [expr {($b<<24)|($b<<16)|($b<<8)|$b}]
    for {set j 0} {$j < $n/4} {incr j} { write32 [expr $addr+$j*4] $w }
}
proc write_desc {addr na wa aa ra bytes {tt 15} {ng 0} {nt 1}} {
    write32 $addr $na; write32 [expr $addr+4] $wa; write32 [expr $addr+8] $aa
    write32 [expr $addr+12] $ra; write32 [expr $addr+16] $tt
    set w20 [expr {($nt << 16) | ($ng & 0xFF)}]
    write32 [expr $addr+20] $w20; write32 [expr $addr+24] $bytes; write32 [expr $addr+28] 0
}
proc fill_q5_weight {base n} {
    set qs_byte [expr ($n<<4)|$n]; set qs_w32 [expr {$qs_byte*0x01010101}]; set qs_w16 [expr {$qs_byte|($qs_byte<<8)}]
    for {set b 0} {$b < 56} {incr b} {
        set bo [expr $base+$b*48]
        write32 [expr $bo+0] 0xFFFF3C00; write32 [expr $bo+4] [expr {($qs_w16<<16)|0xFFFF}]
        write32 [expr $bo+8] $qs_w32; write32 [expr $bo+12] $qs_w32; write32 [expr $bo+16] $qs_w32
        write32 [expr $bo+20] [expr {0x3C00<<16|$qs_w16}]; write32 [expr $bo+24] 0xFFFFFFFF
        write32 [expr $bo+28] $qs_w32; write32 [expr $bo+32] $qs_w32; write32 [expr $bo+36] $qs_w32
        write32 [expr $bo+40] $qs_w32; write32 [expr $bo+44] 0
    }
}
proc fill_q5_weight_alternating {base ne no} {
    for {set b 0} {$b < 56} {incr b} {
        set n [expr {$b%2==0?$ne:$no}]; set qs_byte [expr ($n<<4)|$n]
        set qs_w32 [expr {$qs_byte*0x01010101}]; set qs_w16 [expr {$qs_byte|($qs_byte<<8)}]
        set bo [expr $base+$b*48]
        write32 [expr $bo+0] 0xFFFF3C00; write32 [expr $bo+4] [expr {($qs_w16<<16)|0xFFFF}]
        write32 [expr $bo+8] $qs_w32; write32 [expr $bo+12] $qs_w32; write32 [expr $bo+16] $qs_w32
        write32 [expr $bo+20] [expr {0x3C00<<16|$qs_w16}]; write32 [expr $bo+24] 0xFFFFFFFF
        write32 [expr $bo+28] $qs_w32; write32 [expr $bo+32] $qs_w32; write32 [expr $bo+36] $qs_w32
        write32 [expr $bo+40] $qs_w32; write32 [expr $bo+44] 0
    }
}
proc fill_q5_norms {base} { write32 [expr $base+2688] 0x01000100; write32 [expr $base+2692] 0x01000100 }
proc fill_q5_acts {base v} {
    set w [expr {($v&0xFFFF)|(($v&0xFFFF)<<16)}]
    for {set j 0} {$j < 448} {incr j} { write32 [expr $base+$j*4] $w }
}
proc fill_q8_scales_const {base ng sv} {
    set p [expr {($sv&0xFFFF)|(($sv&0xFFFF)<<16)}]
    for {set g 0} {$g < $ng} {incr g} { for {set j 0} {$j < 64} {incr j} { write32 [expr $base+$g*256+$j*4] $p } }
}
proc fill_q8_acts_const {base ng av} {
    set p [expr {($av&0xFFFF)|(($av&0xFFFF)<<16)}]
    for {set g 0} {$g < $ng} {incr g} { for {set j 0} {$j < 32} {incr j} { write32 [expr $base+$g*128+$j*4] $p } }
}
proc verify_q5_result {res_addr nrows expected test_id} {
    set ok 1
    for {set j 0} {$j < $nrows} {incr j} {
        set lo [read32 [expr $res_addr+$j*8]]; set hi [read32 [expr $res_addr+$j*8+4]]
        set got [expr {($hi<<32)|$lo}]
        if {$got >= [expr {1<<47}]} { set got [expr {$got-(1<<48)}] }
        if {$got != $expected} { puts "  FAIL(${test_id}): row $j got=$got"; set ok 0 }
    }
    if {$ok} { puts "  Test $test_id: PASS (rows $nrows = $expected)" }
    return $ok
}
proc verify_q8_result {res_addr nrows expected test_id} {
    set ok 1
    for {set j 0} {$j < $nrows} {incr j} {
        set lo [read32 [expr $res_addr+$j*8]]; set hi [read32 [expr $res_addr+$j*8+4]]
        set got [expr {($hi<<32)|$lo}]
        if {$got >= [expr {1<<47}]} { set got [expr {$got-(1<<48)}] }
        if {$got != $expected} { puts "  FAIL(${test_id}): row $j got=$got"; set ok 0 }
    }
    if {$ok} { puts "  Test $test_id: PASS (all $nrows rows = $expected)" }
    return $ok
}
proc run_chain {desc_base expected_head} {
    gp0_write 0x18 $desc_base; after 10; gp0_write 0x1C 1; after 10
    gp0_write 0x00 0; after 5; gp0_write 0x00 1; after 10
    set start [clock milliseconds]; set done 0
    while {[expr {[clock milliseconds]-$start}] < 60000} {
        set head [gp0_read 0x20]
        if {$head >= $expected_head} { set done 1; break }
        after 50
    }
    if {!$done} {
        set dbg [gp0_read 0x28]; set q8_dbg [gp0_read 0x3C]; set head [gp0_read 0x20]
        set smask [expr ($dbg>>27)&0x1F]
        set snames {IDLE FETCH_DESC FETCH_DESC_W LOAD_ACT LOAD_ACT_W WRITE_RES WRITE_RES_W DONE LOAD_WEIGHT LOAD_WEIGHT_W LOAD_SCALES LOAD_SCALES_W COPY_ACT_TO_CORE COMPUTE COMPUTE_W READ_RES READ_RES_ACC COPY_ACC_TO_BUF TIMEOUT_ERROR WRITE_RES_BURST Q5_LOAD_NORM Q5_LOAD_NORM_W Q5_COPY_ACT Q5_COPY_ACT_W Q5_BLOCK_COMPUTE Q5_BLOCK_COMPUTE_W Q5_READ_RES}
        set sname [lindex $snames $smask]
        if {$sname eq ""} { set sname "UNK" }
        puts "  TIMEOUT: DEBUG=[format 0x%08x $dbg] Q8_DEBUG=[format 0x%08x $q8_dbg] HEAD=$head state=$smask ($sname)"
        return 0
    }
    return 1
}

# ===== MAIN =====
puts "=============================================="
puts "  HP FSM Extended Test — Part 2 (E9-E10)"
puts "=============================================="

configparams force-mem-accesses 1; connect; after 15000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200

puts "=== Step 1: Load bitstream ==="
fpga -file {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.bit}; after 1000
configparams force-mem-accesses 1

puts "=== Step 2: ps7_init ==="
source {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.gen/sources_1/bd/system/ip/system_ps7_0/ps7_init.tcl}
ps7_mio_init_data_3_0; ps7_pll_init_data_3_0; ps7_clock_init_data_3_0
ps7_ddr_init_data_3_0; ps7_peripherals_init_data_3_0; after 300; ps7_post_config_3_0; after 200
set pll_status [read32 0xF800010C]
puts "  PLL_STATUS=[format 0x%08x $pll_status]"
if {[expr ($pll_status & 7)] != 7} { puts "  ERROR: PLLs not locked!"; exit 1 }

set clk_cnt [gp0_read 0x2C]
puts "  CLK_CNT=[format 0x%08x $clk_cnt]"
if {$clk_cnt < 1000} { puts "  ERROR: PL clock not running!"; exit 1 }

mwr -force 0xF8000008 0x0000DF0D; after 20
mwr -force 0xF8000910 0x0000000F; after 20
mwr -force 0xF8008000 0x00000005; after 20
mwr -force 0xF8008004 0x00000044; after 20
mwr -force 0xF8008008 0x00000001; after 20
mwr -force 0xF8000004 0x0000767B; after 50

gp0_write 0x18 0xDEADBEEF; after 10
if {[gp0_read 0x18] != 0xDEADBEEF} { puts "  ERROR: GP0 FAILED!"; exit 1 }

set pass_count 0; set fail_count 0

# ===================================================================
# Test E9: Mixed Q5 + Q8 + CPU_OP + Q5 chain (4 descs)
# ===================================================================
puts "\n--- Test E9: Mixed Q5 → Q8 → CPU_OP → Q5 chain ---"
set D0 0x00100540; set D1 0x00100560; set D2 0x00100580; set D3 0x001005A0
set Q5W0 0x00124000; set Q5A0 [expr $Q5W0+2696]; set Q5R0 [expr $Q5A0+1792]
set Q8W  0x00125200; set Q8S  [expr $Q8W+4096]; set Q8A  [expr $Q8S+256]; set Q8R [expr $Q8A+128]
set CPU  [expr $Q8R+512]; set CPUR [expr $CPU+64]
set Q5W2 [expr $CPUR+64]; set Q5A2 [expr $Q5W2+2696]; set Q5R2 [expr $Q5A2+1792]

fill_q5_weight $Q5W0 1; fill_q5_norms $Q5W0; fill_q5_acts $Q5A0 1; zero_fill $Q5R0 32
write_pattern_const $Q8W 4096 0x01; fill_q8_scales_const $Q8S 1 0x0100; fill_q8_acts_const $Q8A 1 1; zero_fill $Q8R 512
write_pattern_const $CPU 64 0x5A; zero_fill $CPUR 64
fill_q5_weight $Q5W2 0; fill_q5_norms $Q5W2; fill_q5_acts $Q5A2 1; zero_fill $Q5R2 32

write_desc $D0 $D1 $Q5W0 $Q5A0 $Q5R0 1792 1
write_desc $D1 $D2 $Q8W  $Q8A  $Q8R  128  0 1
write_desc $D2 $D3 0 $CPU $CPUR 64 15 0
write_desc $D3 0 $Q5W2 $Q5A2 $Q5R2 1792 1

if {[run_chain $D0 4]} {
    set s0 [verify_q5_result $Q5R0 4 229376 E9a]
    set s1 [verify_q8_result $Q8R  64 64     E9b]
    set sc 1
    for {set j 0} {$j < 16} {incr j} { if {[read32 [expr $CPUR+$j*4]] != 0x5A5A5A5A} { puts "  FAIL(E9c): word $j"; set sc 0 } }
    # q5=0 → q5=0 (qh=1 → unsigned, core decode: {qh,ql}-16 = 16-16 = 0), d=1.0, act=1 → 0 × 896 = 0
    set s3 [verify_q5_result $Q5R2 4 0 E9d]
    if {$s0 && $s1 && $sc && $s3} { puts "  Test E9: PASS"; incr pass_count } else { incr fail_count }
} else { puts "  Test E9: TIMEOUT"; incr fail_count }

# ===================================================================
# Test E10: Q5_0 alternating q5 nibbles per block
# ===================================================================
puts "\n--- Test E10: Q5_0 alternating q5 nibbles (1,2, expect 344064 per row) ---"
set W 0x00128000; set A [expr $W+2696]; set R [expr $A+1792]; set D 0x001005C0
fill_q5_weight_alternating $W 1 2; fill_q5_norms $W; fill_q5_acts $A 1; zero_fill $R 32
write_desc $D 0 $W $A $R 1792 1
if {[run_chain $D 1]} { if {[verify_q5_result $R 4 344064 E10]} { incr pass_count } else { incr fail_count } } else { puts "  E10 TIMEOUT"; incr fail_count }

# ===== Summary =====
puts "\n=============================================="
if {$fail_count == 0} {
    puts "  PART 2: ALL $pass_count TESTS PASSED"
} else {
    puts "  PART 2: $fail_count FAILED (of [expr $pass_count+$fail_count])"
}
puts "=============================================="

targets -set -filter {name =~ "*Cortex-A9*#0*"}; after 50; stop; after 200
exit
