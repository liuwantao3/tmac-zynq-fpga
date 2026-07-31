# build.tcl — create the vitis_bm bare-metal Vitis workspace (platform + serial app).
#
# Usage (PowerShell):
#   C:\Xilinx\Vitis\2023.1\bin\xsct.bat vitis_bm\build.tcl
#
# The Vitis workspace IS this directory (like the reference 03_dma project).
# Regenerates (all gitignored):
#   z7_bm/       — standalone platform from ../vitis_linux/matmul_bd.xsa (ps7_cortexa9_0)
#   tmac_serial/ — bare-metal C app (sources imported from app/src)
#   .metadata/   — Vitis workspace metadata
#
# Afterwards open the workspace in the Vitis GUI:
#   C:\Xilinx\Vitis\2023.1\bin\vitis.bat  (workspace = vitis_bm)
#   then Run As -> Launch Hardware.

set script_dir [file dirname [info script]]
cd $script_dir

# Remove previously generated projects/workspace metadata for a clean build.
foreach p {z7_bm tmac_serial tmac_serial_system .metadata .Xil} {
    if {[file exists $p]} { file delete -force $p }
}

setws .

platform create -name {z7_bm} \
    -hw {../vitis_linux/matmul_bd.xsa} \
    -proc {ps7_cortexa9_0} -os {standalone} -out {.}

platform active {z7_bm}
platform generate

app create -name tmac_serial \
    -platform {z7_bm} \
    -template {Empty Application(C)} -lang c -os standalone

importsources -name tmac_serial -path {app/src}
app build -name tmac_serial

puts "\n=== Build complete ==="
puts "Open Vitis GUI with workspace $script_dir and Run As -> Launch Hardware."
puts "Console output appears on the USB-UART terminal (115200 8N1)."
