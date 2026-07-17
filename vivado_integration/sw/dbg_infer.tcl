proc read32 {addr} {
    set r [mrd $addr 1]
    if {[regexp {:\s+([0-9A-Fa-f]+)} $r full data]} { return [expr "0x$data"] }
    return -1
}
proc write32 {addr val} { mwr -force $addr $val }

connect
after 3000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}
after 100
catch {stop}
after 200

puts "=== FPGA Registers ==="
puts "  CHAIN_CTRL=[read32 0x43C00004] GIE=[read32 0x43C00008] ISR=[read32 0x43C0000C]"
puts "  STATUS=[read32 0x43C00014] HEAD=[read32 0x43C00020]"
puts "  DEBUG=[read32 0x43C00028] Q8DBG=[read32 0x43C0003C]"
puts "  CLK_CNT=[read32 0x43C0002C]"

puts "=== Output Buffer ==="
set n_out [read32 0x1F000000]
puts "  n_output=$n_out"
if {$n_out > 0 && $n_out < 100} {
    for {set i 0} {$i < $n_out} {incr i} {
        puts "  token $i: [read32 [expr 0x1F000004 + $i*4]]"
    }
}

puts "=== Check if program completed ==="
# Write a marker, continue for 5s, then check if marker still there
write32 0x1F000000 0xDEAD
puts "  Wrote marker 0xDEAD to output[0]. Continuing for 5s..."
con
after 5000
catch {stop}
after 200
set n_out2 [read32 0x1F000000]
puts "  After 5s: n_output=$n_out2"
if {$n_out2 == 0xDEAD} {
    puts "  Marker unchanged — program hung or crashed"
} elseif {$n_out2 > 0 && $n_out2 < 100} {
    puts "  Program completed! Output tokens:"
    for {set i 0} {$i < $n_out2} {incr i} {
        puts "  token $i: [read32 [expr 0x1F000004 + $i*4]]"
    }
} else {
    puts "  Output changed but unexpected value: $n_out2"
}
exit
