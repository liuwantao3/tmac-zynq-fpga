# Read the BOOT_MODE register at 0xF800025C that the FSBL actually uses.
connect; after 5000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 200
catch {stop}; after 200
puts "=== BOOT_MODE_REG (0xF800025C, what FSBL reads) ==="
catch {mrd 0xF800025C 1} bm; puts "  = $bm"
puts "=== neighbor: REBOOT_STATUS (0xF8000258) ==="
catch {mrd 0xF8000258 1} rs; puts "  = $rs"
puts "=== GPIO MIO[6:4] (from earlier) ==="
catch {mrd 0xE000A060 1} g0; puts "GPIO0_DATA_RO = $g0"
if {[regexp {:\s+([0-9A-Fa-f]+)} $g0 -> h]} {
    set v [expr "0x$h"]
    puts "  MIO4=[expr {($v>>4)&1}] MIO5=[expr {($v>>5)&1}] MIO6=[expr {($v>>6)&1}] -> [expr {($v>>4)&0x7}]"
}
