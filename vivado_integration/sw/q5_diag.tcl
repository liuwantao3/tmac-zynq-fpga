proc read32 {addr} {
    set r [mrd $addr 1]
    if {[regexp {:\s+([0-9A-Fa-f]+)} $r full data]} { return [expr "0x$data"] }
    return -1
}
configparams force-mem-accesses 1
connect
after 15000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}
after 200

set waddr 0x00105000
puts "=== desc1 weight first 48 bytes (block 0) ==="
for {set i 0} {$i < 12} {incr i} {
    set addr [expr {$waddr + $i*4}]
    set val [read32 $addr]
    puts "  W[format %02d $i] @ [format 0x%08x $addr] = [format 0x%08x $val]"
}

set naddr [expr {$waddr + 2688}]
puts "\n=== Norms @ $naddr ==="
set n0 [read32 $naddr]
set n1 [read32 [expr {$naddr + 4}]]
puts "  N0=[format 0x%08x $n0] N1=[format 0x%08x $n1]"

puts "\n=== Result at 0x00106018 ==="
set rlo [read32 0x00106018]
set rhi [read32 0x0010601C]
puts "  lo=[format 0x%08x $rlo] hi=[format 0x%08x $rhi]"

set GP0_BASE 0x43C00000
puts "\n=== Q5 registers ==="
puts "  DEBUG   =[format 0x%08x [read32 [expr $GP0_BASE + 0x28]]]"
puts "  Q5_DEBUG=[format 0x%08x [read32 [expr $GP0_BASE + 0x40]]]"
puts "  CAP0    =[format 0x%08x [read32 [expr $GP0_BASE + 0x44]]]"
puts "  CAP1    =[format 0x%08x [read32 [expr $GP0_BASE + 0x48]]]"
puts "  TRIG    =[format 0x%08x [read32 [expr $GP0_BASE + 0x4C]]]"
puts "  LIVE    =[format 0x%08x [read32 [expr $GP0_BASE + 0x50]]]"
puts "  HEAD    =[format 0x%08x [read32 [expr $GP0_BASE + 0x20]]]"
puts "  STATUS  =[format 0x%08x [read32 [expr $GP0_BASE + 0x14]]]"
exit
