configparams force-mem-accesses 1
connect -url TCP:127.0.0.1:3121
after 5000

set targets [targets -type APU]
puts "Targets: $targets"
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}} err
puts "Catch result: $err"

targets 1
after 100
puts "Now connected: [targets -set -filter {name =~ "*Cortex-A9*#0*"}]"

proc read32 {addr} {
    set r [mrd $addr 1]
    if {[regexp {:\s+([0-9A-Fa-f]+)} $r full data]} { return [expr "0x$data"] }
    return -1
}

puts "=== desc1 weight first 48 bytes ==="
for {set i 0} {$i < 12} {incr i} {
    set addr [expr {0x00105000 + $i*4}]
    set val [read32 $addr]
    puts "  W[format %02d $i] = [format 0x%08x $val]"
}

puts "\n=== Norms ==="
set n0 [read32 0x00105A80]; set n1 [read32 0x00105A84]
puts "  N0=[format 0x%08x $n0] N1=[format 0x%08x $n1]"

set GP0_BASE 0x43C00000
puts "\n=== Head = [read32 [expr $GP0_BASE + 0x20]]"
puts "=== Status = [read32 [expr $GP0_BASE + 0x14]]"
puts "=== All 4 result rows ==="
for {set j 0} {$j < 4} {incr j} {
    set addr [expr {0x00106000 + $j*8}]
    set lo [read32 $addr]
    set hi [read32 [expr $addr+4]]
    puts "  Row $j: lo=[format 0x%08x $lo] hi=[format 0x%08x $hi]"
}
exit
