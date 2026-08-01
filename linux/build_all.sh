#!/bin/bash
# Build U-Boot + Linux kernel + Buildroot for Zynq-7010 SD card boot
# Works in BOTH environments:
#   - Lima ARM64 Ubuntu VM (recommended): apt gcc-arm-linux-gnueabihf + libssl-dev
#   - macOS host: clang wrapper via setup_toolchain.sh + brew binutils + openssl
# Requires: bash linux/clone_repos.sh completed (or repos in workdir)
# Usage: bash linux/build_all.sh [workdir] [fpga_root]

set -euo pipefail

WORKDIR="${1:-/tmp/arm-build}"
FPGA_ROOT="${2:-$(cd "$(dirname "$0")/.." && pwd)}"
BOOT_DIR="$FPGA_ROOT/linux/boot"
CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo "=== Build environment ==="
echo "  workdir:   $WORKDIR"
echo "  fpga_root: $FPGA_ROOT"
echo "  boot_dir:  $BOOT_DIR"
echo "  cores:     $CORES"
echo ""

# ── Auto-setup toolchain ──
if ! which arm-linux-gnueabihf-gcc >/dev/null 2>&1; then
    TOOLS="/tmp/arm-toolchain/bin"
    if [[ -x "$TOOLS/arm-linux-gnueabihf-gcc" ]]; then
        export PATH="$TOOLS:$PATH"
        echo "  toolchain: clang-based at $TOOLS"
    else
        echo "ERROR: arm-linux-gnueabihf-gcc not found in PATH."
        echo "Run: bash linux/setup_toolchain.sh   (clang-based, no downloads needed)"
        echo "Or install ARM GCC from: https://developer.arm.com/downloads/-/arm-gnu-toolchain-downloads"
        exit 1
    fi
fi
GCC_VER=$(arm-linux-gnueabihf-gcc --version 2>&1 | head -1)
echo "  compiler:  $GCC_VER"
echo ""

export CROSS_COMPILE=arm-linux-gnueabihf-
TOOLCHAIN_DIR=$(dirname "$(which arm-linux-gnueabihf-gcc)")
export PATH="$TOOLCHAIN_DIR:$PATH"

# Use GNU Make 4.x (macOS ships 3.81 which is too old for U-Boot/kernel)
if command -v gmake >/dev/null 2>&1; then
    export MAKE=gmake
fi
echo "=== [1/4] Building U-Boot ==="
cd "$WORKDIR/u-boot-xlnx"

# ── Reset to clean state (idempotent for re-runs) ──
git checkout -- . 2>/dev/null || true

# ── Patch macOS SDK conflicts (Python is more reliable than sed) ──
python3 << 'PYEOF'
import re

# 1. Undefine macOS secure string macros before ARM asm/string.h
P = "arch/arm/include/asm/string.h"
with open(P) as f: s = f.read()
guard = """#ifdef __APPLE__
#undef memcpy
#undef memmove
#undef memset
#undef memcmp
#undef strcpy
#undef strncpy
#undef strcat
#undef strncat
#undef strlcpy
#undef strlcat
#endif
"""
with open(P,'w') as f: f.write(re.sub(r'(?=^extern void \* memcpy)', guard, s, count=1, flags=re.M))

# 2. Undef strlcat + guard strchrnul (conflicts with macOS SDK)
P = "include/linux/string.h"
with open(P) as f: lines = f.readlines()
out = []
for ln in lines:
    if '#ifndef __HAVE_ARCH_STRLCAT' in ln:
        out.append('#ifdef __APPLE__\n#undef strlcat\n#endif\n')
    if ln.rstrip() == 'const char *strchrnul(const char *s, int c);':
        out.append('#ifndef __APPLE__\n' + ln + '#endif\n'); continue
    out.append(ln)
with open(P,'w') as f: f.writelines(out)

# 3. sbrk signature differs on macOS
P = "include/malloc.h"
with open(P) as f: s = f.read()
s = s.replace('extern Void_t*     sbrk(ptrdiff_t);',
              '#ifndef __APPLE__\nextern Void_t*     sbrk(ptrdiff_t);\n#else\n#include <unistd.h>\n#endif')
with open(P,'w') as f: f.write(s)

# 4. Skip mkeficapsule (EFI headers incompatible with Apple clang/SDK)
P = "tools/Makefile"
with open(P) as f: lines = f.readlines()
out = [l for l in lines if 'mkeficapsule' not in l or 'hostprogs' not in l]
with open(P,'w') as f: f.writelines(out)

# 5. macOS dd requires cbs for conv=block; use cp instead
P = "scripts/Makefile.spl"
with open(P) as f: s = f.read()
# Heredoc is quoted so $ is literal in Python string
s = s.replace('\t@dd if=$< of=$@ conv=block,sync bs=4 2>/dev/null;', '\t@cp $< $@')
s = s.replace('INPUTS-$(CONFIG_ARCH_ZYNQ)\t\t+= $(obj)/boot.bin',
              '# INPUTS-$(CONFIG_ARCH_ZYNQ)\t\t+= $(obj)/boot.bin')
with open(P,'w') as f: f.write(s)

# 6. Disable EFI in defconfig
P = "configs/xilinx_zynq_virt_defconfig"
with open(P) as f: content = f.read()
if 'CONFIG_EFI_LOADER=n' not in content:
    content += '\n# CONFIG_EFI_LOADER is not set\n'
with open(P,'w') as f: f.write(content)
# 7. Move debug_uart_init BEFORE initf_dm — the zc702 serial DM probe
#    hangs against ps7_init's Z7-Lite MIO/clock config. Calling
#    debug_uart_init early guarantees UART0 output before any DM init.
#    NOTE: init_sequence_f entries are `int (*)(void)`, but debug_uart_init
#    returns void — so we insert a small int-returning wrapper, plus the
#    <debug_uart.h> include (board_f.c does not include it by default).
import re
P = "common/board_f.c"
with open(P) as f: s = f.read()
if '#include <debug_uart.h>' not in s:
    s = s.replace('#include <serial.h>', '#include <debug_uart.h>\n#include <serial.h>', 1)
if 'debug_uart_init_wrap' not in s:
    wrap = '''
#ifdef CONFIG_DEBUG_UART
static int debug_uart_init_wrap(void)
{
\tdebug_uart_init();
\treturn 0;
}
#endif
'''
    s = s.replace('static int initf_dm(void)', wrap + 'static int initf_dm(void)', 1)
    s = re.sub(
        r'^(\s*)initf_dm,$',
        r'#ifdef CONFIG_DEBUG_UART\n\1debug_uart_init_wrap,\n#endif\n\1initf_dm,',
        s, flags=re.MULTILINE
    )
with open(P,'w') as f: f.write(s)
PYEOF
echo "  patches applied"

# ── Build ──
export CROSS_COMPILE=arm-linux-gnueabihf-
# macOS host: U-Boot host tools need clang + Homebrew openssl. Inside the Lima
# Ubuntu VM (recommended) these are left to defaults (gcc + libssl-dev).
if [ "$(uname -s)" = "Darwin" ]; then
  export HOSTCC=clang
  export HOSTCFLAGS="-I$(brew --prefix openssl 2>/dev/null || echo /opt/homebrew/opt/openssl@3)/include"
  export HOSTLDFLAGS="-L$(brew --prefix openssl 2>/dev/null || echo /opt/homebrew/opt/openssl@3)/lib"
fi
# Console = UART0: The default device tree (zynq-zc706) uses UART1
# (MIO 48/49), but the Z7-Lite board has the CH340 USB-UART on UART0
# (MIO 14/15). Switch to zynq-zc702 DTB, which routes the console to
# UART0/serial0. Also embed the DTB in the ELF (CONFIG_OF_EMBED=y) so
# that JTAG-booted U-Boot has its device tree. Enable DEBUG_UART_ZYNQ
# for early boot output that writes directly to UART0 registers before
# the driver model initializes. A Python patch (above) moves
# debug_uart_init before initf_dm in board_f.c so it actually runs
# before the DM serial probe hangs on the Z7-Lite's non-standard MIO.
${MAKE:-make} xilinx_zynq_virt_defconfig
echo 'CONFIG_DEFAULT_DEVICE_TREE="zynq-zc702"' >> .config
echo 'CONFIG_OF_EMBED=y' >> .config
echo 'CONFIG_DEBUG_UART=y' >> .config
echo 'CONFIG_DEBUG_UART_ZYNQ=y' >> .config
echo 'CONFIG_DEBUG_UART_BASE=0xE0000000' >> .config
echo 'CONFIG_DEBUG_UART_CLOCK=100000000' >> .config
${MAKE:-make} olddefconfig
${MAKE:-make} -j"$CORES" u-boot spl/u-boot-spl.bin
cp u-boot "$BOOT_DIR/u-boot.elf"
echo "  → u-boot.elf copied to $BOOT_DIR/"

# ── Boot script (boot.scr): CONFIG_DISTRO_DEFAULTS=y makes U-Boot auto-run
#    boot.scr from the FAT32 partition, so SD boot needs no interactive prompt.
#    Use U-Boot's own mkimage (NOT kernel u-boot-tools) for a legacy script image.
./tools/mkimage -A arm -T script -C none -n "Boot" -d "$BOOT_DIR/boot.cmd" "$BOOT_DIR/boot.scr"
echo "  → boot.scr copied to $BOOT_DIR/"
echo ""

# ── 2. Linux Kernel ──
echo "=== [2/4] Building Linux Kernel ==="
cd "$WORKDIR/linux-xlnx"

# macOS-only: minimal elf.h for host tools (Linux hosts have real ELF headers)
if [ "$(uname -s)" = "Darwin" ]; then
  [ -f /tmp/arm-toolchain/elf.h ] || {
      echo "ERROR: /tmp/arm-toolchain/elf.h not found. Run: bash linux/setup_toolchain.sh"
      exit 1
  }
  export HOSTCFLAGS="-I/tmp/arm-toolchain"
fi

${MAKE:-make} ARCH=arm xilinx_zynq_defconfig

# UART0 console (ttyPS0): bake the command line into CONFIG_CMDLINE so both
# SD-boot (U-Boot) and JTAG-boot (boot_linux_jtag.tcl, empty DT /chosen/bootargs)
# get the console on the USB-UART. The stock defconfig already sets
# CONFIG_SERIAL_XILINX_PS_UART(_CONSOLE)=y; this REPLACES the DCC earlycon
# (earlycon=dcc console=hvc0) that was baked in on 2026-07-30.
./scripts/config --enable SERIAL_XILINX_PS_UART_CONSOLE
./scripts/config --set-str CMDLINE "earlycon console=ttyPS0,115200 root=/dev/ram0 rw iomem=relaxed"
${MAKE:-make} -j"$CORES" ARCH=arm olddefconfig
${MAKE:-make} -j"$CORES" ARCH=arm UIMAGE_LOADADDR=0x8000 uImage
${MAKE:-make} ARCH=arm dtbs
cp arch/arm/boot/uImage "$BOOT_DIR/"
cp arch/arm/boot/dts/zynq-zc702.dtb "$BOOT_DIR/devicetree.dtb"
echo "  → uImage, devicetree.dtb copied to $BOOT_DIR/"
echo ""

# ── 3. Buildroot (rootfs) ──
echo "=== [3/4] Building Buildroot rootfs ==="
cd "$WORKDIR/buildroot"
${MAKE:-make} qemu_arm_vexpress_defconfig

# Enable NEON/VFPv3 for Cortex-A9
cat >> .config << 'BRCFG'
BR2_ARM_ENABLE_VFP=y
BR2_ARM_ENABLE_NEON=y
BRCFG
${MAKE:-make} olddefconfig
${MAKE:-make} -j"$CORES"

cp output/images/rootfs.cpio.uboot "$BOOT_DIR/uramdisk.image.gz"
echo "  → uramdisk.image.gz copied to $BOOT_DIR/"

# ── JTAG-boot artifacts (U-Boot-less hand boot via boot_linux_jtag.tcl) ──
# 1. Raw gzipped cpio initramfs (no U-Boot legacy header): the kernel reads the
#    initrd directly from DDR via /chosen/linux,initrd-start/end. If buildroot
#    ever wraps it in a 64-byte mkimage header (magic 0x27051956), strip it.
SRC="$BOOT_DIR/uramdisk.image.gz"
DST="$BOOT_DIR/initramfs.cpio.gz"
if [ "$(head -c4 "$SRC" | od -An -tx1 | tr -d ' \n')" = "27051956" ]; then
  dd if="$SRC" of="$DST" bs=64 skip=1 2>/dev/null
else
  cp "$SRC" "$DST"
fi
echo "  → initramfs.cpio.gz copied to $BOOT_DIR/ ($(wc -c < "$DST") bytes)"

# 2. DTB with /chosen/linux,initrd-start/end baked in (start must match the
#    RAMFS_LOAD in boot_linux_jtag.tcl, 0x03000000).
INITRD_SIZE=$(stat -f%z "$DST" 2>/dev/null || stat -c%s "$DST" 2>/dev/null)
python3 "$FPGA_ROOT/linux/patch_dtb_initrd.py" "$BOOT_DIR/devicetree.dtb" \
  0x03000000 "$INITRD_SIZE" "$BOOT_DIR/devicetree-jtag.dtb"
echo "  → devicetree-jtag.dtb copied to $BOOT_DIR/"

# 3. Mirror the JTAG artifacts into vitis_linux/prebuilt (where the tcl looks)
VITIS_PREBUILT="$FPGA_ROOT/vitis_linux/prebuilt"
if [ -d "$VITIS_PREBUILT" ]; then
  cp "$DST" "$VITIS_PREBUILT/initramfs.cpio.gz"
  cp "$BOOT_DIR/devicetree-jtag.dtb" "$VITIS_PREBUILT/devicetree-jtag.dtb"
  echo "  → mirrored to $VITIS_PREBUILT/"
fi
echo ""

# ── 4. BOOT.BIN — Mac-side bootgen via buildroot (no Windows needed) ──
echo "=== [4/4] Creating BOOT.BIN (buildroot host-bootgen) ==="
bash "$FPGA_ROOT/linux/build_bootbin.sh" "$WORKDIR" "$FPGA_ROOT"
echo ""

# ── Summary ──
echo "============================================"
echo "  Build complete. Boot files in $BOOT_DIR/:"
echo "============================================"
ls -lh "$BOOT_DIR"/{u-boot.elf,boot.scr,uImage,devicetree.dtb,devicetree-jtag.dtb,uramdisk.image.gz,initramfs.cpio.gz,BOOT.BIN} 2>/dev/null
echo ""
echo "Next steps (all on the Mac — no Windows needed for the SD flow):"
echo "  1. Format SD: diskutil partitionDisk /dev/diskX MBR FAT32 SD_BOOT 128M FAT32 SD_DATA R"
echo "  2. Copy BOOT.BIN + uImage + devicetree.dtb + uramdisk.image.gz + boot.scr → FAT32 p1"
echo "  3. Copy model.tmac + tmac → FAT32 p2 (initramfs mounts /dev/mmcblk0p2 on /root)"
echo "  4. Insert SD, set J1 boot mode to SD, power on — U-Boot auto-runs boot.scr (UART0 console)"
echo ""
echo "JTAG boot (no SD): devicetree-jtag.dtb + initramfs.cpio.gz were mirrored to"
echo "  vitis_linux/prebuilt/ if present — run vitis_linux/scripts/boot_linux_jtag.tcl."
