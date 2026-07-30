# Build Zynq FSBL using XSCT
set xsa "D:/Users/u/tmac-zynq-fpga/linux/boot/matmul_bd.xsa"
set out "D:/Users/u/tmac-zynq-fpga/linux/boot"
set sw "C:/Xilinx/Vitis/2023.1/data/embeddedsw"

setws $out

# Create platform
platform create -name "fsbl_platform" -hw $xsa -proc ps7_cortexa9_0 -os standalone
platform active "fsbl_platform"

# Register the main embeddedsw repo (not the sw_apps subfolder)
repo -set "$sw/lib/sw_apps"

# Create FSBL app
app create -name "fsbl_app" -platform "fsbl_platform" -template "Zynq FSBL" \
    -proc ps7_cortexa9_0 -os standalone -lang c

# Add xilffs library (required by FSBL for SD boot)
bsp addlib "fsbl_app" "xilffs"
bsp config "xilffs" "use_lfn" "false"

# Build
app build -name "fsbl_app"

# Copy result
file copy -force "$out/fsbl_app/Release/fsbl_app.elf" "$out/fsbl.elf"
puts "FSBL generated: $out/fsbl.elf"
exit
