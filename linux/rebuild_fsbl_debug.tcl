# Rebuild the platform's auto-generated FSBL with FSBL_DEBUG_INFO debug output.
#
# NOTE: `app config -name "zynq_fsbl"` FAILS because zynq_fsbl is a platform
# boot component, not a workspace app. The reliable way is to edit the FSBL
# component's Makefile CFLAGS and run the ARM toolchain make directly.
# (See AGENTS.md Key Decision #23, 2026-08-04.)

set fsbl_dir "D:/Users/u/tmac-zynq-fpga/linux/boot/fsbl_platform/zynq_fsbl"
set out "D:/Users/u/tmac-zynq-fpga/linux/boot"

# 1. Ensure CFLAGS carries -DFSBL_DEBUG_INFO (idempotent)
set mf "$fsbl_dir/Makefile"
set fd [open $mf r]
set content [read $fd]
close $fd
if {![regexp {-DFSBL_DEBUG_INFO} $content]} {
    regsub {CFLAGS := } $content {CFLAGS := -DFSBL_DEBUG_INFO } content
    set fd [open $mf w]
    puts -nonewline $fd $content
    close $fd
    puts "Added -DFSBL_DEBUG_INFO to $mf"
} else {
    puts "-DFSBL_DEBUG_INFO already present"
}

# 2. Rebuild via make_4.2.exe (gnuwin make.exe crashes with 0xc0000005 on CMD
#    SHELL=command.com). Requires the ARM toolchain + sh.exe in PATH.
#    (Run externally from bash/PowerShell; XSCT cannot set env here.)
puts "Build it externally:"
puts "  cd $fsbl_dir"
puts "  make_4.2.exe -C zynq_fsbl_bsp -j1 && make_4.2.exe"
puts "then copy fsbl.elf to $out/fsbl.elf"
exit
