open_hw_manager
connect_hw_server
open_hw_target
current_hw_device [lindex [get_hw_devices] end]
puts "Device props related to scan/debug:"
set props [list_property [current_hw_device]]
foreach p $props {
    if {[string match -nocase "*scan*" $p] || [string match -nocase "*debug*" $p] || [string match -nocase "*bscan*" $p]} {
        puts "  [get_property $p [current_hw_device]] <- $p"
    }
}
