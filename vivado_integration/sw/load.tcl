# XSDB script: init PS7, program FPGA, load test_int16.elf
connect
targets -set -filter {name =~ "ARM Cortex-A9 MPCore #0"}

# Step 1: Initialize PS7 (clocks, DDR, MIO)
source ../proj_bd/ps7_init.tcl
ps7_init
ps7_post_config

# Step 2: Program FPGA fabric (PL)
fpga -file ../proj_bd/matmul_bd.bit

# Step 3: Reset CPU, load ELF, run
rst -processor
dow test_int16.elf
con
