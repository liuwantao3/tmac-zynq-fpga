# Vitis batch build
set origin "D:/Users/u/tmac-zynq-fpga"
set xsa  "$origin/vivado_integration/proj_bd/matmul_bd.xsa"
set ws   "$origin/vivado_integration/vitis_ws"
set src  "$origin/vivado_integration/sw"

file delete -force $ws
file mkdir $ws

# Create platform
platform create -name matmul_platform -hw $xsa -proc ps7_cortexa9_0 -os standalone -out $ws
platform write

# Create app and add source
app create -name hp_test -platform matmul_platform -template "Empty Application(C)" -out $ws
imports sources $src/hp_test.c

# Build
app build hp_test

puts "=== BUILD DONE ==="
puts "ELF: $ws/hp_test/Debug/hp_test.elf"
