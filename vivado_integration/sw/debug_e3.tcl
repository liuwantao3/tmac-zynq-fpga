# Debug E3: Q8 14-group timeout
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
proc gp0_read {reg} {
    global GP0_BASE
    return [read32 [format 0x%08x [expr $GP0_BASE + $reg]]]
}
proc gp0_write {reg val} {
    global GP0_BASE
    write32 [format 0x%08x [expr $GP0_BASE + $reg]] $val
}

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
ps7_mio_init_data_3_0
ps7_pll_init_data_3_0
ps7_clock_init_data_3_0
ps7_ddr_init_data_3_0
ps7_peripherals_init_data_3_0
after 300
ps7_post_config_3_0
after 200

set pll_status [read32 0xF800010C]
puts "  PLL_STATUS=[format 0x%08x $pll_status]"

puts "=== PL clock ==="
set clk_cnt [gp0_read 0x2C]
puts "  CLK_CNT=[format 0x%08x $clk_cnt]"

puts "=== AFI config ==="
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

# Set up E3: Q8 full 14-group
set E3_W  0x00114000
set E3_S  [expr $E3_W + 4096]
set E3_A  [expr $E3_S + 3584]
set E3_R  [expr $E3_A + 1792]
set E3_D  0x00100440

# Zero-fill
for {set j 0} {$j < 1024} {incr j} { write32 [expr $E3_W + $j*4] 0 }
# Write weights = 1
set w1 [expr {0x01010101}]
for {set j 0} {$j < 1024} {incr j} { write32 [expr $E3_W + $j*4] $w1 }
# Write scales = 1.0 (UQ8.8 = 0x0100)
set sc_pair [expr {0x01000100}]
for {set g 0} {$g < 14} {incr g} {
    set grp_addr [expr $E3_S + $g * 256]
    for {set j 0} {$j < 64} {incr j} { write32 [expr $grp_addr + $j * 4] $sc_pair }
}
# Write acts = 1
set act_pair [expr {0x00010001}]
for {set g 0} {$g < 14} {incr g} {
    set grp_addr [expr $E3_A + $g * 128]
    for {set j 0} {$j < 32} {incr j} { write32 [expr $grp_addr + $j * 4] $act_pair }
}
# Zero-fill result
for {set j 0} {$j < 128} {incr j} { write32 [expr $E3_R + $j*4] 0 }
# Write descriptor (tensor_type=0, num_groups=14, bytes=128)
write32 [expr $E3_D + 0]  0
write32 [expr $E3_D + 4]  $E3_W
write32 [expr $E3_D + 8]  $E3_A
write32 [expr $E3_D + 12] $E3_R
write32 [expr $E3_D + 16] 0
write32 [expr $E3_D + 20] [expr {14 & 0xFF}]
write32 [expr $E3_D + 24] 128
write32 [expr $E3_D + 28] 0

# Start
gp0_write 0x18 $E3_D
after 10
gp0_write 0x1C 1
after 10
gp0_write 0x00 1
after 10

# Poll with register dump
set timeout_ms 60000
set start [clock milliseconds]
set last_state -1
while {[expr {[clock milliseconds] - $start}] < $timeout_ms} {
    set head [gp0_read 0x20]
    set dbg [gp0_read 0x28]
    set q8_dbg [gp0_read 0x3C]
    set state_mask [expr ($dbg >> 27) & 0x1F]
    
    if {$state_mask != $last_state} {
        set state_names {IDLE FETCH_DESC FETCH_DESC_W LOAD_ACT LOAD_ACT_W WRITE_RES WRITE_RES_W DONE LOAD_WEIGHT LOAD_WEIGHT_W LOAD_SCALES LOAD_SCALES_W COPY_ACT_TO_CORE COMPUTE COMPUTE_W READ_RES READ_RES_ACC COPY_ACC_TO_BUF TIMEOUT_ERROR WRITE_RES_BURST Q5_LOAD_NORM Q5_LOAD_NORM_W Q5_COPY_ACT Q5_COPY_ACT_W Q5_BLOCK_COMPUTE Q5_BLOCK_COMPUTE_W Q5_READ_RES}
        set sname [lindex $state_names $state_mask]
        if {$sname eq ""} { set sname "UNK" }
        set col_grp [expr ($dbg >> 11) & 0xF]
        set rd_busy  [expr ($dbg >> 24) & 1]
        set wr_busy  [expr ($dbg >> 23) & 1]
        set rd_done  [expr ($dbg >> 26) & 1]
        set wr_done  [expr ($dbg >> 25) & 1]
        set q8_busy  [expr ($q8_dbg >> 26) & 1]
        set q8_done  [expr ($q8_dbg >> 25) & 1]
        set q8_k     [expr ($q8_dbg >> 11) & 0x3F]
        set now [expr {[clock milliseconds] - $start}]
        puts "  state=$state_mask ($sname) col=$col_grp rd_busy=$rd_busy wr_busy=$wr_busy rd_done=$rd_done wr_done=$wr_done q8_busy=$q8_busy q8_done=$q8_done q8_k=$q8_k h=$head t=${now}ms"
        set last_state $state_mask
    }
    
    if {$head >= 1} {
        puts "  Done at [expr {[clock milliseconds] - $start}]ms, HEAD=$head"
        # Read results
        set ok 1
        for {set j 0} {$j < 64} {incr j} {
            set addr [expr $E3_R + $j * 8]
            set lo [read32 $addr]
            set hi [read32 [expr $addr + 4]]
            set got [expr {($hi << 32) | $lo}]
            if {$got >= [expr {1 << 47}]} { set got [expr {$got - (1 << 48)}] }
            if {$got != 896} { puts "  row $j = $got (expected 896)"; set ok 0 }
        }
        if {$ok} { puts "  ALL 64 ROWS = 896 PASS" }
        break
    }
    after 50
}

# Final dump
set dbg [gp0_read 0x28]
set q8_dbg [gp0_read 0x3C]
puts "  Final DEBUG=[format 0x%08x $dbg] Q8_DEBUG=[format 0x%08x $q8_dbg]"
set state_mask [expr ($dbg >> 27) & 0x1F]
set state_names {IDLE FETCH_DESC FETCH_DESC_W LOAD_ACT LOAD_ACT_W WRITE_RES WRITE_RES_W DONE LOAD_WEIGHT LOAD_WEIGHT_W LOAD_SCALES LOAD_SCALES_W COPY_ACT_TO_CORE COMPUTE COMPUTE_W READ_RES READ_RES_ACC COPY_ACC_TO_BUF TIMEOUT_ERROR WRITE_RES_BURST Q5_LOAD_NORM Q5_LOAD_NORM_W Q5_COPY_ACT Q5_COPY_ACT_W Q5_BLOCK_COMPUTE Q5_BLOCK_COMPUTE_W Q5_READ_RES}
set sname [lindex $state_names $state_mask]
puts "  State=$state_mask ($sname)"
puts "  col_group=[expr ($dbg>>11)&0xF] sc_byte_idx=[expr $dbg&0xFF]"
puts "  ACT_INFO=[format 0x%08x [gp0_read 0x34]] DESC_INFO=[format 0x%08x [gp0_read 0x38]]"

stop
puts "ARM halted."
exit
