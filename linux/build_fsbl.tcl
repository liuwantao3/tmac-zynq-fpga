# Build Zynq FSBL using XSCT - standard Vitis flow (matches MicroPhase reference).
#
# The reference (03_dma/arm/arm_build.tcl) does ONLY:
#   platform create ... ; platform generate
# and the auto-generated "zynq_fsbl" boot domain is built as part of platform
# generation. Its BSP auto-selects the boot drivers (sdps, qspips, ...) from
# the hardware spec (XSA). No manual app creation or xilffs setlib is needed.
set xsa "D:/Users/u/tmac-zynq-fpga/linux/boot/matmul_bd.xsa"
set out "D:/Users/u/tmac-zynq-fpga/linux/boot"

# Clean any stale platform so the new XSA is imported fresh (critical: an old
# platform would retain the no-SD hardware and omit the sdps driver).
if {[file exists "$out/fsbl_platform"]} {
    file delete -force "$out/fsbl_platform"
}
if {[file exists "$out/fsbl_app"]} {
    file delete -force "$out/fsbl_app"
}
if {[file exists "$out/fsbl_app_system"]} {
    file delete -force "$out/fsbl_app_system"
}

setws $out

# Create platform. This auto-creates the zynq_fsbl boot domain (with its BSP
# config: xilffs + xilrsa) plus the standalone_domain.
platform create -name "fsbl_platform" -hw $xsa -proc ps7_cortexa9_0 -os standalone
platform active "fsbl_platform"

# Build both BSPs (zynq_fsbl + standalone_domain) and the auto-generated FSBL.
# The zynq_fsbl BSP picks up the sdps driver from the SD-enabled hardware.
platform generate

# NOTE: FSBL_DEBUG_INFO cannot be set via `app config -name "zynq_fsbl"` —
# zynq_fsbl is a platform boot component, not a workspace app. To enable debug
# output, edit fsbl_platform/zynq_fsbl/Makefile CFLAGS := -DFSBL_DEBUG_INFO and
# rebuild with make_4.2.exe (see linux/rebuild_fsbl_debug.tcl). The platform
# generate above already produced an SD-capable fsbl.elf; this block is optional.
# app config -name "zynq_fsbl" define-compiler-symbols FSBL_DEBUG_INFO
# app build -name "zynq_fsbl"

# Copy result
file copy -force "$out/fsbl_platform/zynq_fsbl/fsbl.elf" "$out/fsbl.elf"
puts "FSBL generated: $out/fsbl.elf"
exit
