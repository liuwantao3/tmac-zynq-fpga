# XSDB script: init PS7, load and run hp_test.elf
# This runs AFTER debug_hp.tcl has programmed the FPGA

connect
after 500
puts "Targets:"
targets

# Try selecting xc7z010 first to run ps7_init
if {[catch {targets -set -filter {name =~ "xc7z010"}} err]} {
    puts "No xc7z010 target, trying DAP"
    targets -set -filter {name =~ "DAP"}
}

source {D:/Users/u/tmac-zynq-fpga/vivado_integration/ps7_init.tcl}
puts "Running ps7_init..."
catch {ps7_init} msg
puts "ps7_init: $msg"
catch {ps7_post_config} msg
puts "ps7_post_config: $msg"

# Check for ARM core
after 500
puts "Targets after init:"
targets

if {[catch {targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}} err]} {
    puts "ARM core not found! Trying to run ps7_init again..."
    after 1000
    targets -set -filter {name =~ "DAP"}
    catch {ps7_init} msg
    catch {ps7_post_config} msg
    after 500
    targets
    if {[catch {targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}} err2]} {
        puts "Still cannot find ARM core: $err2"
        exit 1
    }
}

rst -processor
dow {D:/Users/u/tmac-zynq-fpga/vivado_integration/vitis_ws/hp_test/Debug/hp_test.elf}
con
puts "ELF loaded and running"
