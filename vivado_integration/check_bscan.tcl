set props [list_property -class design]
puts "All design properties containing 'CONFIG':"
foreach p $props {
    if {[string match -nocase "*config*" $p]} {
        puts "  $p"
    }
}
