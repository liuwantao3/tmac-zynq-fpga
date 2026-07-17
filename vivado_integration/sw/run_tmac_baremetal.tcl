proc read32 {addr} {
    set r [mrd $addr 1]
    if {[regexp {:\s+([0-9A-Fa-f]+)} $r full data]} { return [expr "0x$data"] }
    return -1
}
proc write32 {addr val} { mwr -force $addr $val }
proc gp0_read {reg} { return [read32 [expr 0x43C00000 + $reg]] }
proc gp0_write {reg val} { write32 [expr 0x43C00000 + $reg] $val }

set BITSTREAM  {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.runs/impl_1/system_wrapper.bit}
set PS7_INIT   {D:/Users/u/tmac-zynq-fpga/vivado_integration/proj_bd/matmul_bd.gen/sources_1/bd/system/ip/system_ps7_0/ps7_init.tcl}
set ELF_PATH   {D:/Users/u/tmac-zynq-fpga/vivado_integration/sw/tmac_baremetal.elf}
set MODEL_PATH {D:/Users/u/tmac-zynq-fpga/models/model.tmac}
set MODEL_BASE 0x00200000
set DESC_BASE  0x1F001000

puts "=============================================="
puts "  T-MAC Bare-Metal Inference Test"
puts "=============================================="

configparams force-mem-accesses 1
connect
after 15000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}
after 200

# ===== Load bitstream =====
puts "Loading bitstream..."
fpga -file $BITSTREAM
after 2000
configparams force-mem-accesses 1
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}
after 200

# ===== PS7 Init =====
puts "PS7 init..."
source $PS7_INIT
ps7_mio_init_data_3_0; after 20
ps7_pll_init_data_3_0; after 20
ps7_clock_init_data_3_0; after 20
ps7_ddr_init_data_3_0; after 200
ps7_peripherals_init_data_3_0; after 20
ps7_post_config_3_0; after 200

set pll_status [read32 0xF800010C]
puts "  PLL_STATUS=[format 0x%08x $pll_status]"
if {[expr ($pll_status & 7)] != 7} { puts "  ERROR: PLLs not locked!"; exit 1 }

# ===== AFI config =====
puts "AFI config..."
mwr -force 0xF8000008 0x0000DF0D; after 10
mwr -force 0xF8000910 0x0000000F; after 10
mwr -force 0xF8008000 0x00000005; after 10
mwr -force 0xF8008004 0x00000044; after 10
mwr -force 0xF8008008 0x00000001; after 10
mwr -force 0xF8000004 0x0000767B; after 20

# ===== PL clock check =====
set clk_cnt [gp0_read 0x2C]
puts "  CLK_CNT = [format 0x%08X $clk_cnt]"
if {$clk_cnt == 0} { puts "  WARNING: CLK_CNT=0, PL may not be running" }

# ===== Verify XSDB DDR access =====
puts "Verifying DDR access..."
write32 0x1F000800 0xCAFEBABE
set v [read32 0x1F000800]
puts "  Wrote 0xCAFEBABE to 0x1F000800, read back 0x[format %08x $v]"
if {$v != 0xCAFEBABE} { puts "  WARNING: DDR access unreliable!" }

# ===== Load model to DDR =====
puts "Loading model from $MODEL_PATH..."
dow -data $MODEL_PATH $MODEL_BASE
after 200
puts "  Model loaded"
set n_bytes [file size $MODEL_PATH]
puts "  Model size: $n_bytes bytes"

# ===== Verify model header =====
set magic [read32 $MODEL_BASE]
puts "  Model magic=0x[format %08x $magic] (expect 0x43414D54=TMAC)"
set nten [read32 [expr $MODEL_BASE + 4]]
puts "  n_tensors=$nten"

# ===== Write prompt buffer: 1 token =====
puts "Writing prompt buffer..."
write32 $DESC_BASE 1
write32 [expr $DESC_BASE + 4] 151646  ;# Qwen2 BOS token

# ===== Clear progress/exception buffers =====
write32 0x1F000800 0
write32 0x1F000900 0
write32 0x1F000904 0
write32 0x1F000908 0
write32 0x1F00090C 0

# ===== Load ELF =====
puts "Loading ELF..."
dow $ELF_PATH
after 200
puts "  ELF loaded"

# ===== Run =====
puts "Running (900s timeout)..."
con
after 900000
catch {targets -set -filter {name =~ "*Cortex-A9*#0*"}}; after 100
stop; after 200

# ===== Check FPGA registers =====
puts "\n--- FPGA State ---"
puts "  REG_DEBUG=0x[format %08x [gp0_read 0x28]]"
puts "  REG_Q8_DEBUG=0x[format %08x [gp0_read 0x3C]]"
puts "  REG_Q5_DEBUG=0x[format %08x [gp0_read 0x40]]"
puts "  CHAIN_CTRL=0x[format %08x [gp0_read 0x04]] GIE=0x[format %02x [gp0_read 0x08]] ISR=0x[format %02x [gp0_read 0x0C]]"
puts "  STATUS=0x[format %04x [gp0_read 0x14]]"
puts "  CLK_CNT=0x[format %08x [gp0_read 0x2C]]"

# ===== Check exception info =====
puts "\n--- Exception Check ---"
set fault_magic [read32 0x1F000800]
puts "  FAULT_MAGIC=0x[format %08x $fault_magic]"
if {$fault_magic == 0x44414254} { puts "  => DATA ABORT!" }
if {$fault_magic == 0x50414254} { puts "  => PREFETCH ABORT!" }
if {$fault_magic == 0x554E4446} { puts "  => UNDEFINED INSTRUCTION!" }
if {$fault_magic == 0} { puts "  => No exception" }
puts "  FAULT_LR  =0x[format %08x [read32 0x1F000810]]"
puts "  FAULT_ADDR=0x[format %08x [read32 0x1F000814]]"
puts "  FAULT_STAT=0x[format %08x [read32 0x1F000818]]"

# ===== Check progress =====
puts "\n--- Progress Monitor ---"
puts "  PROG_CNT=[read32 0x1F000900]"
puts "  PROG_LAYER=[read32 0x1F000904]"
puts "  PROG_MATMUL=[read32 0x1F000908]"
puts "  PROG_TOKEN=[read32 0x1F00090C]"

# ===== Check output =====
puts "\n--- Output Buffer ---"
set n_out [read32 0x1F000000]
puts "  n_output=$n_out"
if {$n_out > 0 && $n_out < 100} {
    for {set i 0} {$i < $n_out} {incr i} {
        puts "  token $i: [read32 [expr 0x1F000004 + $i*4]]"
    }
} elseif {$n_out == 0} {
    puts "  (program did not write output)"
}

puts ""
puts "ARM halted."
exit
