# XSDB: minimal flow — FPGA + PS7 + load ELF + run
configparams force-mem-accesses 1
connect
after 5000

proc list_targets {} {
    puts "=== All targets ==="
    if {[catch {targets} err]} {
        puts "No targets listed: $err"
    }
}
proc select_dap {} {
    foreach pattern {"*DAP*" "*APU*" "*ps7*" "*PS7*"} {
        if {![catch {targets -set -filter [list name =~ $pattern]} err]} {
            if {![catch {mrd 0xF8000090 1}]} { return 1 }
        }
    }
    for {set tn 1} {$tn <= 10} {incr tn} {
        if {![catch {targets -set $tn} err]} {
            if {![catch {mrd 0xF8000090 1}]} { return 1 }
        }
    }
    return 0
}
proc select_arm0 {} {
    foreach pattern {"*Cortex-A9*#0*" "*Cortex-A9*0*" "*A9*#0*" "#0*"} {
        if {![catch {targets -set -filter [list name =~ $pattern]} err]} {
            if {![catch {rrd pc}]} { return 1 }
        }
    }
    for {set tn 1} {$tn <= 10} {incr tn} {
        if {![catch {targets -set $tn} err]} {
            if {![catch {rrd pc}]} { return 1 }
        }
    }
    return 0
}

list_targets

# Step 1: Program FPGA
puts "=== Programming FPGA ==="
if {[catch {fpga -file {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.bit}} err]} {
    puts "fpga failed: $err"
    list_targets
    exit 1
}
puts "FPGA programmed"
after 2000
list_targets

# Step 2: Initialize PS7
source {D:/Users/u/tmac-zynq-fpga/vivado_integration/ps7_init.tcl}
after 500
puts "=== Selecting DAP ==="
if {![select_dap]} {
    puts "ERROR: Could not select DAP target"
    list_targets
    exit 1
}
puts "DAP target selected"
after 500
puts "Running ps7_init..."
if {[catch {ps7_init} err]} {
    puts "ps7_init failed: $err"
    list_targets
    exit 1
}
puts "PS7 initialized"

# Step 3: Deassert PL reset via DAP
puts "=== Deassert PL reset ==="
puts "FPGA_RST_CTRL before: [mrd 0xF8000708 1]"
catch {mwr 0xF8000708 0x00010600}
after 100
puts "FPGA_RST_CTRL after: [mrd 0xF8000708 1]"
after 1000

# Step 4: Load ELF and continue CPU
puts "=== Selecting ARM core ==="
if {![select_arm0]} {
    puts "ERROR: Could not select ARM core"
    list_targets
    exit 1
}
puts "ARM core selected"
catch {stop}
after 500
puts "Loading ELF..."
if {[catch {dow {D:/Users/u/workspace/tmac/Debug/tmac.elf}} err]} {
    puts "dow failed: $err"
    exit 1
}
after 500

puts "Continuing CPU..."
catch {con}
after 1000

# Step 5: Let CPU run for 30s, then halt and read PC
puts "=== Letting CPU run (30s)... ==="
after 30000
catch {stop}
after 500
if {[catch {set pc_after [rrd pc]} e]} {
    puts "Can't read PC: $e"
} else {
    puts "PC after run: $pc_after"
}
if {[catch {set cpsr_val [rrd cpsr]} e]} {
    puts "Can't read CPSR: $e"
} else {
    puts "CPSR: $cpsr_val"
}

# Read PL registers via DAP to see if markers appeared
select_dap
after 500
puts "=== PL reg at 0x43C00004 (REG_GIE) ==="
catch {puts [mrd 0x43C00004 1]}
puts "=== DDR at ELF entry ==="
catch {puts [mrd 0x00100000 8]}

puts "=== DONE ==="
