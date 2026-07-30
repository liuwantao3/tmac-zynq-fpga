# Debug E9: Mixed Q5 + Q8 + CPU_OP + Q5 chain
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
proc gp0_read {reg} { global GP0_BASE; return [read32 [format 0x%08x [expr $GP0_BASE + $reg]]] }
proc gp0_write {reg val} { global GP0_BASE; write32 [format 0x%08x [expr $GP0_BASE + $reg]] $val }

configparams force-mem-accesses 1
connect
after 15000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}
after 200

puts "=== Load bitstream ==="
fpga -file {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.bit}
after 1000
configparams force-mem-accesses 1

puts "=== ps7_init ==="
source {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.gen/sources_1/bd/system/ip/system_ps7_0/ps7_init.tcl}
ps7_mio_init_data_3_0; ps7_pll_init_data_3_0; ps7_clock_init_data_3_0
ps7_ddr_init_data_3_0; ps7_peripherals_init_data_3_0; after 300; ps7_post_config_3_0; after 200

set pll_status [read32 0xF800010C]
puts "  PLL_STATUS=[format 0x%08x $pll_status]"

set clk_cnt [gp0_read 0x2C]
puts "  CLK_CNT=[format 0x%08x $clk_cnt]"

mwr -force 0xF8000008 0x0000DF0D; after 20
mwr -force 0xF8000910 0x0000000F; after 20
mwr -force 0xF8008000 0x00000005; after 20
mwr -force 0xF8008004 0x00000044; after 20
mwr -force 0xF8008008 0x00000001; after 20
mwr -force 0xF8000004 0x0000767B; after 50

# ── Setup E9 data ──
proc zero_fill {addr nbytes} { for {set j 0} {$j < $nbytes / 4} {incr j} { write32 [expr $addr + $j * 4] 0 } }
proc write_pattern_const {addr nbytes byte_val} {
    set w [expr {($byte_val<<24)|($byte_val<<16)|($byte_val<<8)|$byte_val}]
    for {set j 0} {$j < $nbytes / 4} {incr j} { write32 [expr $addr + $j * 4] $w }
}
proc write_desc {addr na wa aa ra bytes {tt 15} {ng 0} {nt 1}} {
    write32 $addr $na; write32 [expr $addr+4] $wa; write32 [expr $addr+8] $aa
    write32 [expr $addr+12] $ra; write32 [expr $addr+16] $tt
    set w20 [expr {($nt << 16) | ($ng & 0xFF)}]
    write32 [expr $addr+20] $w20; write32 [expr $addr+24] $bytes; write32 [expr $addr+28] 0
}
proc fill_q5_weight {base n} {
    set qs_byte [expr ($n<<4)|$n]
    set qs_w32 [expr {$qs_byte*0x01010101}]
    set qs_w16 [expr {$qs_byte|($qs_byte<<8)}]
    for {set b 0} {$b < 56} {incr b} {
        set bo [expr $base + $b*48]
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
    for {set j 0} {$j < 448} {incr j} { write32 [expr $base + $j*4] $w }
}
proc fill_q8_scales_const {base ng sv} {
    set p [expr {($sv&0xFFFF)|(($sv&0xFFFF)<<16)}]
    for {set g 0} {$g < $ng} {incr g} { for {set j 0} {$j < 64} {incr j} { write32 [expr $base+$g*256+$j*4] $p } }
}
proc fill_q8_acts_const {base ng av} {
    set p [expr {($av&0xFFFF)|(($av&0xFFFF)<<16)}]
    for {set g 0} {$g < $ng} {incr g} { for {set j 0} {$j < 32} {incr j} { write32 [expr $base+$g*128+$j*4] $p } }
}
proc run_chain {desc_base expected_head} {
    gp0_write 0x18 $desc_base; after 10; gp0_write 0x1C 1; after 10
    gp0_write 0x00 0; after 5; gp0_write 0x00 1; after 10
    set start [clock milliseconds]; set done 0
    while {[expr {[clock milliseconds] - $start}] < 30000} {
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
        puts "  TIMEOUT @[expr {[clock milliseconds]-$start}]ms: DEBUG=[format 0x%08x $dbg] Q8_DEBUG=[format 0x%08x $q8_dbg] HEAD=$head state=$smask ($sname)"
        puts "    col=[expr ($dbg>>11)&0xF] rd_busy=[expr ($dbg>>24)&1] wr_busy=[expr ($dbg>>23)&1]"
        puts "    rd_done=[expr ($dbg>>26)&1] wr_done=[expr ($dbg>>25)&1]"
        puts "    act_info=[format 0x%08x [gp0_read 0x34]] desc_info=[format 0x%08x [gp0_read 0x38]]"
        return 0
    }
    puts "  Done @[expr {[clock milliseconds]-$start}]ms HEAD=$head"
    return 1
}

# ── Setup E9 ──
set D0 0x00100540; set D1 0x00100560; set D2 0x00100580; set D3 0x001005A0
set W0 0x00124000; set A0 [expr $W0+2696]; set R0 [expr $A0+1792]
set W1 0x00125200; set S1 [expr $W1+4096]; set A1 [expr $S1+256]; set R1 [expr $A1+128]
set CP  [expr $R1+512]; set CPR [expr $CP+64]
set W2 [expr $CPR+64]; set A2 [expr $W2+2696]; set R2 [expr $A2+1792]

puts "  W0=$W0 A0=$A0 R0=$R0"
puts "  W1=$W1 A1=$A1 R1=$R1"
puts "  CP=$CP CPR=$CPR"
puts "  W2=$W2 A2=$A2 R2=$R2"

fill_q5_weight $W0 1; fill_q5_norms $W0; fill_q5_acts $A0 1; zero_fill $R0 32
write_pattern_const $W1 4096 0x01; fill_q8_scales_const $S1 1 0x0100; fill_q8_acts_const $A1 1 1; zero_fill $R1 512
write_pattern_const $CP 64 0x5A; zero_fill $CPR 64
fill_q5_weight $W2 0; fill_q5_norms $W2; fill_q5_acts $A2 1; zero_fill $R2 32

write_desc $D0 $D1 $W0 $A0 $R0 1792 1 1
write_desc $D1 $D2 $W1 $A1 $R1 128 0 1
write_desc $D2 $D3 0 $CP $CPR 64 15 0
write_desc $D3 0 $W2 $A2 $R2 1792 1 1

puts "\n=== Run E9 chain ==="
run_chain $D0 4

# Read results
puts "\n=== Results ==="
for {set j 0} {$j < 4} {incr j} {
    set lo [read32 [expr $R0+$j*8]]; set hi [read32 [expr $R0+$j*8+4]]
    set got [expr {($hi<<32)|$lo}]
    if {$got >= [expr {1<<47}]} { set got [expr {$got-(1<<48)}] }
    puts "  D0(row $j) = $got"
}
set ok 1
for {set j 0} {$j < 64} {incr j} {
    set lo [read32 [expr $R1+$j*8]]; set hi [read32 [expr $R1+$j*8+4]]
    set got [expr {($hi<<32)|$lo}]
    if {$got >= [expr {1<<47}]} { set got [expr {$got-(1<<48)}] }
    if {$got != 64} { puts "  D1(row $j) = $got"; set ok 0 }
}
if {$ok} { puts "  D1: ALL 64 rows = 64 PASS" }
set cpu_ok 1
for {set j 0} {$j < 16} {incr j} {
    set got [read32 [expr $CPR+$j*4]]
    if {$got != 0x5A5A5A5A} { puts "  D2(CPU_OP word $j) = [format 0x%08x $got]"; set cpu_ok 0 }
}
if {$cpu_ok} { puts "  D2: CPU_OP PASSTHROUGH PASS" }

stop; puts "ARM halted."; exit
