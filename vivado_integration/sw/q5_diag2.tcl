# Q5 diagnostic: dump DDR at the desc1 weight+act overlap region
configparams force-mem-accesses 1
connect
after 15000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}
after 200

proc read32 {addr} {
    set r [mrd $addr 1]
    if {[regexp {:\s+([0-9A-Fa-f]+)} $r full data]} { return [expr "0x$data"] }
    return -1
}
proc hx {v} { return [format 0x%08x $v] }

set GP0_BASE 0x43C00000
set W_BASE 0x00105000
set A_BASE 0x00105800
set N_OFF 2688

puts "=== DDR region check ==="
puts "Weight base:   [hx $W_BASE]..[hx [expr $W_BASE + 2687]] (2688 bytes)"
puts "Act base:      [hx $A_BASE]..[hx [expr $A_BASE + 1791]] (1792 bytes)"
set overlap_start [expr {$W_BASE + 2688 > $A_BASE ? $W_BASE + 2688 : $A_BASE}]
set overlap_end [expr {$W_BASE + 2687 < $A_BASE + 1791 ? $W_BASE + 2687 : $A_BASE + 1791}]
if {$overlap_start < $overlap_end} {
    puts "OVERLAP: [hx $overlap_start]..[hx $overlap_end]"
}

puts "\n=== First 48 bytes of weight ==="
for {set i 0} {$i < 12} {incr i} {
    set addr [expr {$W_BASE + $i*4}]
    puts "  [hx $addr]: [hx [read32 $addr]]"
}

puts "\n=== Block 28 (row1 start) of weight ==="
set blk28_off [expr {28 * 48}]
for {set i 0} {$i < 12} {incr i} {
    set addr [expr {$W_BASE + $blk28_off + $i*4}]
    puts "  [hx $addr]: [hx [read32 $addr]]"
}

puts "\n=== Block 43 (overlap region) of weight ==="
set blk43_off [expr {43 * 48}]
for {set i 0} {$i < 12} {incr i} {
    set addr [expr {$W_BASE + $blk43_off + $i*4}]
    puts "  [hx $addr]: [hx [read32 $addr]]"
}

puts "\n=== Norm location (weight+2688) ==="
puts "  Norm[0:1]=[hx [read32 [expr $W_BASE + 2688]]]"
puts "  Norm[2:3]=[hx [read32 [expr $W_BASE + 2692]]]"

puts "\n=== Current Q5 debug registers ==="
puts "  HEAD    =[hx [read32 [expr $GP0_BASE + 0x20]]]"
puts "  STATUS  =[hx [read32 [expr $GP0_BASE + 0x14]]]"
puts "  DEBUG   =[hx [read32 [expr $GP0_BASE + 0x28]]]"
puts "  CAP0    =[hx [read32 [expr $GP0_BASE + 0x44]]]"
puts "  CAP1    =[hx [read32 [expr $GP0_BASE + 0x48]]]"
puts "  CAP2    =[hx [read32 [expr $GP0_BASE + 0x54]]]"
puts "  CAP3    =[hx [read32 [expr $GP0_BASE + 0x58]]]"
puts "  CAP4    =[hx [read32 [expr $GP0_BASE + 0x5C]]]"
puts "  CAP5    =[hx [read32 [expr $GP0_BASE + 0x60]]]"
puts "  LIVE    =[hx [read32 [expr $GP0_BASE + 0x50]]]"

# Unfreeze debug capture
gp0_write 0x4C 0; after 10

puts "\n=== Arm trigger for block 28, then run chain ==="
# Arm: [15:10]=28 (blk to capture), [0]=1 (arm)
# Need to write with blk=28, arm=1
gp0_write 0x4C [expr {(28 << 10) | 1}]
after 10

# Now write overlapped weight+act data and run test 2
# Use non-overlapping addresses for diagnostic
puts "\n=== Rewriting with NON-OVERLAPPING addresses ==="
set W2_DIAG 0x0010C000
set A2_DIAG 0x0010D000
set R2_DIAG 0x0010E000
set D2_DIAG 0x00100400

# fill_q5_weight with nibble=0 (all zeros)
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
proc fill_q5_norms {weight_base} {
    write32 [expr $weight_base + 2688] 0x01000100
    write32 [expr $weight_base + 2692] 0x01000100
}
proc fill_q5_acts {base val} {
    set act_word [expr {($val & 0xFFFF) | (($val & 0xFFFF) << 16)}]
    for {set j 0} {$j < [expr 1792 / 4]} {incr j} {
        write32 [expr $base + $j * 4] $act_word
    }
}

puts "  Writing weight+act+norm to non-overlapping DDR..."
fill_q5_weight $W2_DIAG 0
fill_q5_norms  $W2_DIAG
fill_q5_acts   $A2_DIAG 2
# Zero result
for {set j 0} {$j < 8} {incr j} { write32 [expr $R2_DIAG + $j*4] 0 }

# Write descriptor
proc write_desc {addr next_addr weight_addr act_addr res_addr bytes {tensor_type 1} {num_groups 0} {num_tiles 1}} {
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
write_desc $D2_DIAG 0 $W2_DIAG $A2_DIAG $R2_DIAG 1792 1

puts "\n=== Verify DDR data loaded correctly ==="
puts "  Descriptor base=[hx $D2_DIAG]"
puts "  Weight[0:1]=[hx [read32 $W2_DIAG]] [hx [read32 [expr $W2_DIAG+4]]]"
puts "  Weight[2688:2692]=[hx [read32 [expr $W2_DIAG+2688]]] [hx [read32 [expr $W2_DIAG+2692]]]"
puts "  Act[0]=[hx [read32 $A2_DIAG]]"
puts "  Res[0]=[hx [read32 $R2_DIAG]]"

puts "\n=== Run single desc (non-overlapping) ==="
gp0_write 0x18 $D2_DIAG; after 5
gp0_write 0x1C 1; after 5
gp0_write 0x00 0; after 5
gp0_write 0x00 1; after 5

set timeout_ms 30000
set start [clock milliseconds]
set done 0
while {[expr {[clock milliseconds] - $start}] < $timeout_ms} {
    set head [read32 [expr $GP0_BASE + 0x20]]
    if {$head >= 1} { set done 1; break }
    after 50
}
if {!$done} { puts "  TIMEOUT!"; set st [read32 [expr $GP0_BASE + 0x14]]; set db [read32 [expr $GP0_BASE + 0x28]]; puts "  STATUS=[hx $st] DEBUG=[hx $db]" }

puts "\n=== Results (non-overlapping) ==="
for {set j 0} {$j < 4} {incr j} {
    set addr [expr $R2_DIAG + $j*8]
    set lo [read32 $addr]
    set hi [read32 [expr $addr + 4]]
    set got [expr {($hi << 32) | $lo}]
    if {$got >= [expr {1 << 47}]} { set got [expr {$got - (1 << 48)}] }
    puts "  Row $j: lo=[hx $lo] hi=[hx $hi] got=$got (expect 0)"
}

puts "\n=== Debug state ==="
puts "  DEBUG=[hx [read32 [expr $GP0_BASE + 0x28]]]"
puts "  LIVE=[hx [read32 [expr $GP0_BASE + 0x50]]]"
set trig [read32 [expr $GP0_BASE + 0x4C]]
puts "  TRIG=[hx $trig] frozen=[expr {($trig>>9)&1}]"
gp0_write 0x4C 0; after 10

puts "\n=== Check original overlapping configuration too ==="
# Now check what the original test weight+act region looks like
puts "  Original weight blk28 (overlap check):"
for {set i 0} {$i < 12} {incr i} {
    set addr [expr {0x00105000 + 28*48 + $i*4}]
    puts "    [hx $addr]: [hx [read32 $addr]]"
}

puts "\n=== DONE ==="
exit
