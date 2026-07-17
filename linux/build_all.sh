#!/bin/bash
# Build U-Boot + Linux kernel + Buildroot for Zynq-7010 SD card boot
# Requires: step 1 (clone) completed, arm-linux-gnueabihf- toolchain in PATH
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
echo "=== [1/3] Building U-Boot ==="
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
PYEOF
echo "  patches applied"

# ── Build ──
export CROSS_COMPILE=arm-linux-gnueabihf-
export HOSTCC=clang
export HOSTCFLAGS="-I$(brew --prefix openssl 2>/dev/null || echo /opt/homebrew/opt/openssl@3)/include"
export HOSTLDFLAGS="-L$(brew --prefix openssl 2>/dev/null || echo /opt/homebrew/opt/openssl@3)/lib"
${MAKE:-make} xilinx_zynq_virt_defconfig
${MAKE:-make} -j"$CORES" u-boot spl/u-boot-spl.bin
cp u-boot "$BOOT_DIR/u-boot.elf"
echo "  → u-boot.elf copied to $BOOT_DIR/"
echo ""

# ── 2. Linux Kernel ──
echo "=== [2/3] Building Linux Kernel ==="
cd "$WORKDIR/linux-xlnx"
${MAKE:-make} ARCH=arm xilinx_zynq_defconfig
# macOS lacks elf.h; disable build-time table sort (not needed for boot)
echo 'CONFIG_BUILDTIME_TABLE_SORT=n' >> .config
${MAKE:-make} ARCH=arm olddefconfig
${MAKE:-make} -j"$CORES" ARCH=arm UIMAGE_LOADADDR=0x8000 uImage
${MAKE:-make} ARCH=arm dtbs
cp arch/arm/boot/uImage "$BOOT_DIR/"
cp arch/arm/boot/dts/zynq-zc702.dtb "$BOOT_DIR/devicetree.dtb"
echo "  → uImage, devicetree.dtb copied to $BOOT_DIR/"
echo ""

# ── 3. Buildroot (rootfs) ──
echo "=== [3/3] Building Buildroot rootfs ==="
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
echo ""

# ── Summary ──
echo "============================================"
echo "  Build complete. Boot files in $BOOT_DIR/:"
echo "============================================"
ls -lh "$BOOT_DIR"/{u-boot.elf,uImage,devicetree.dtb,uramdisk.image.gz} 2>/dev/null
echo ""
echo "Next steps:"
echo "  1. On Windows with Vivado: cd $BOOT_DIR && bootgen -image boot.bif -o BOOT.BIN -w"
echo "  2. Format SD card: partition 1 FAT32, partition 2 ext4"
echo "  3. Copy BOOT.BIN + uImage + devicetree.dtb + uramdisk.image.gz → FAT32 partition"
echo "  4. Copy model.tmac + tmac → ext4 partition"
echo "  5. Insert SD, set boot mode to SD, power on"
