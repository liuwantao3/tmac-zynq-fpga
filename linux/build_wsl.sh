#!/bin/bash
# Rebuild the Linux SD-boot artifacts for the MicroPhase Z7-Lite in WSL.
#
# Windows(WSL)-only flow (no Lima/macOS). The frozen, verified boot artifacts
# in linux/boot/ are COMMITTED; this script regenerates them from source when
# the kernel/DTB/initramfs need to change (or to prove reproducibility).
#
# Environment (Ubuntu 24.04 WSL):
#   sudo apt install gcc-arm-linux-gnueabihf build-essential bison flex bc \
#       libssl-dev libelf-dev u-boot-tools device-tree-compiler cpio wget
#   git clone --depth=1 --branch xilinx-v2024.1 \
#       https://github.com/Xilinx/linux-xlnx.git  (see clone_repos.sh)
#
# Usage: bash linux/build_wsl.sh [linux-xlnx dir]
#
# Outputs (overwritten in linux/boot/):
#   uImage               : kernel uImage (load addr 0x8000)
#   devicetree.dtb       : Z7-Lite patched DTB (SD boot)
#   devicetree-jtag.dtb  : same + /chosen/linux,initrd-start/end (JTAG hand-boot)
#   uramdisk.image.gz    : wrapped initramfs (U-Boot ramdisk header)
#   initramfs.cpio.gz    : raw gzipped cpio (for the JTAG hand-boot)

set -euo pipefail

LINUX_DIR="${1:-/home/u/linux-xlnx}"
FPGA_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BOOT_DIR="$FPGA_ROOT/linux/boot"
CORES=$(nproc 2>/dev/null || echo 4)

export CROSS_COMPILE=arm-linux-gnueabihf-
export ARCH=arm

echo "=== WSL Linux build (Z7-Lite) ==="
echo "  kernel:    $LINUX_DIR"
echo "  boot_dir:  $BOOT_DIR"
echo "  cores:     $CORES"

[ -d "$LINUX_DIR" ] || { echo "ERROR: $LINUX_DIR not found — run linux/clone_repos.sh"; exit 1; }
which arm-linux-gnueabihf-gcc >/dev/null || { echo "ERROR: arm-linux-gnueabihf-gcc not found"; exit 1; }
which mkimage >/dev/null || { echo "ERROR: mkimage (u-boot-tools) not found"; exit 1; }

cd "$LINUX_DIR"

# ── 1. Kernel config: stock + UART0 console + early-boot debug (DEBUG_LL) ──
#    DEBUG_LL/EARLY_PRINTK/DEBUG_ZYNQ_UART0 are kept on: they print to UART0
#    before the console driver exists, which proved essential for diagnosing
#    the 2026-08-11 "Starting kernel..." hang. Harmless at runtime.
make xilinx_zynq_defconfig
./scripts/config --enable SERIAL_XILINX_PS_UART_CONSOLE
./scripts/config --set-str CMDLINE "earlycon console=ttyPS0,115200 root=/dev/ram0 rw iomem=relaxed"
./scripts/config --enable EARLY_PRINTK
./scripts/config --enable DEBUG_LL
./scripts/config --enable DEBUG_ZYNQ_UART0
make olddefconfig

# ── 2. DTB patch: strip zc702 config that conflicts with Z7-Lite hardware ──
#    The zc702 reference DTB is used as a base but its MIO assignments clash
#    with the Z7-Lite (MIO 14/15 = UART0, SDIO0 MIO 40-45, no CD/WP pins).
#    See docs/z7lite-vs-zc702.md for the full delta.
PYTHON_PATCH=$(cat << 'PYEOF'
import re
p = "arch/arm/boot/dts/xilinx/zynq-zc702.dts"
s = open(p).read()

# sdhci0: remove CD/WP pinctrl (Z7-Lite has CD/WP off; zc702 maps MIO0/MIO15
# to sdio0_cd/wp, and MIO15 is UART0 RX). Add xlnx,has-* = 0.
old = """&sdhci0 {
	bootph-all;
	status = "okay";
	pinctrl-names = "default";
	pinctrl-0 = <&pinctrl_sdhci0_default>;
};"""
new = """&sdhci0 {
	bootph-all;
	status = "okay";
	xlnx,has-cd = <0x0>;
	xlnx,has-power = <0x0>;
	xlnx,has-wp = <0x0>;
};"""
if old in s:
    s = s.replace(old, new)

# gpio0: remove pinctrl (zc702 claims MIO 7-14 as GPIO; MIO 14/15 are UART0)
old = """&gpio0 {
	pinctrl-names = "default";
	pinctrl-0 = <&pinctrl_gpio0_default>;
};"""
new = """&gpio0 {
};"""
if old in s:
    s = s.replace(old, new)

# gpio-keys node: uses GPIO 14 (= MIO14 = UART0 TX)
old = """	gpio-keys {
		compatible = "gpio-keys";
		autorepeat;
		switch-14 {
			label = "sw14";
			gpios = <&gpio0 12 0>;
			linux,code = <108>; /* down */
			wakeup-source;
			autorepeat;
		};
		switch-13 {
			label = "sw13";
			gpios = <&gpio0 14 0>;
			linux,code = <103>; /* up */
			wakeup-source;
			autorepeat;
		};
	};
"""
s = s.replace(old, "")

# leds node: uses GPIO 10
old = """	leds {
		compatible = "gpio-leds";

		led-ds23 {
			label = "ds23";
			gpios = <&gpio0 10 0>;
			linux,default-trigger = "heartbeat";
		};
	};
"""
s = s.replace(old, "")

open(p, "w").write(s)
print("  DTS patched (sdhci0/gpio0/gpio-keys/leds)")
PYEOF
)
python3 -c "$PYTHON_PATCH"

# ── 3. Build kernel uImage + dtbs ──
make -j"$CORES" ARCH=arm olddefconfig
make -j"$CORES" ARCH=arm UIMAGE_LOADADDR=0x8000 uImage dtbs

cp arch/arm/boot/uImage "$BOOT_DIR/uImage"
cp arch/arm/boot/dts/xilinx/zynq-zc702.dtb "$BOOT_DIR/devicetree.dtb"
echo "  → uImage + devicetree.dtb"

echo "=== kernel + DTB done (initramfs next) ==="

# ── 5. Busybox initramfs (32-bit ARM static) ──
#    The stock buildroot cpio had an AArch64 busybox -> "Failed to execute
#    /init (error -8)" on the 32-bit Zynq. Build busybox from source with the
#    ARM cross-compiler instead. CONFIG_TC is disabled (new kernel headers
#    removed TCA_CBQ_* needed by busybox tc).
BB_VER=1.36.1
BB_DIR=/tmp/busybox-$BB_VER
if [ ! -x "$BB_DIR/busybox" ]; then
    cd /tmp
    if [ ! -f busybox-$BB_VER.tar.bz2 ]; then
        wget -q "https://busybox.net/downloads/busybox-$BB_VER.tar.bz2" -O busybox-$BB_VER.tar.bz2
    fi
    tar xjf busybox-$BB_VER.tar.bz2
    cd "$BB_DIR"
    make clean >/dev/null 2>&1 || true
    make defconfig >/dev/null
    sed -i 's/^# CONFIG_STATIC is not set/CONFIG_STATIC=y/' .config
    sed -i 's/^CONFIG_TC=y/# CONFIG_TC is not set/' .config
    make oldconfig >/dev/null 2>&1 || true
    make -j"$CORES"
fi
file "$BB_DIR/busybox" | grep -q "32-bit" || { echo "ERROR: busybox not 32-bit"; exit 1; }

# ── 6. Assemble initramfs ──
WORK=/tmp/initramfs-new
GEN="$LINUX_DIR/usr/gen_init_cpio"
rm -rf "$WORK"
mkdir -p "$WORK"/{bin,dev,proc,sys,root,tmp,etc}
cp "$BB_DIR/busybox" "$WORK/bin/busybox"
cd "$WORK"
for cmd in sh mount umount ls cat echo mknod sleep dmesg cp mv rm \
    grep sed awk hexdump md5sum devmem ps kill top free vi \
    fdisk mkfs.ext2 blkid ifconfig ping wget modprobe sync \
    reboot poweroff halt setsid cttyhack; do
    ln -sf /bin/busybox bin/$cmd
done
cp "$BOOT_DIR/tmac" bin/ 2>/dev/null || true

cat > init << 'INIT'
#!/bin/sh
mount -t proc none /proc
mount -t sysfs none /sys
mount -t devtmpfs none /dev
echo ""
echo "=== FPGA Linux Boot - Zynq 7010 (UART0 console) ==="
echo "Console: USB-UART0 (CH340), 115200 8N1"
echo ""
mount /dev/mmcblk0p1 /root 2>/dev/null && echo "SD data mounted at /root" || echo "No SD data partition"
echo "FPGA base: 0x43C00000"
echo "Ready. Commands: /root/tmac | devmem 0x43C00014"
echo ""
# setsid + cttyhack give the shell a controlling terminal on /dev/console,
# without which the keyboard input does not work.
setsid /bin/cttyhack /bin/sh
exec sh </dev/ttyPS0 >/dev/ttyPS0 2>&1
INIT
chmod +x init

# cpio via gen_init_cpio so /dev nodes exist without root
: > /tmp/initramfs.manifest
add_dir()  { echo "dir $1 0755 0 0" >> /tmp/initramfs.manifest; }
add_slink(){ echo "slink $1 $2 0777 0 0" >> /tmp/initramfs.manifest; }
add_file() { echo "file $1 $2 0755 0 0" >> /tmp/initramfs.manifest; }
add_node() { echo "nod $1 $2 0 0 $3 $4 $5" >> /tmp/initramfs.manifest; }
add_dir /dev
add_node /dev/console 622 c 5 1
add_node /dev/null    666 c 1 3
add_node /dev/tty     666 c 5 0
(cd "$WORK" && find . -mindepth 1 | sort | while read -r p; do
    p="${p#./}"
    if [ -d "$p" ]; then add_dir "/$p"
    elif [ -L "$p" ]; then add_slink "/$p" "$(readlink "$p")"
    else add_file "/$p" "$WORK/$p"; fi
done)
sort /tmp/initramfs.manifest -o /tmp/initramfs.manifest
"$GEN" /tmp/initramfs.manifest > /tmp/initramfs.cpio
gzip -c /tmp/initramfs.cpio > "$BOOT_DIR/initramfs.cpio.gz"

# ── 7. Wrap initramfs with a U-Boot ramdisk header ──
#    initramfs.cpio.gz is the raw gzip for the JTAG hand-boot; uramdisk.image.gz
#    is the U-Boot legacy-header version for `bootm`.
cd "$BOOT_DIR"
mkimage -A arm -O linux -T ramdisk -C gzip -a 0x02000000 -e 0x02000000 \
    -n 'uramdisk.image.gz' -d initramfs.cpio.gz uramdisk.image.gz >/dev/null

# ── 8. JTAG DTB (needs the real initramfs size) ──
INITRD_SIZE=$(stat -c%s "$BOOT_DIR/initramfs.cpio.gz")
python3 "$FPGA_ROOT/linux/patch_dtb_initrd.py" "$BOOT_DIR/devicetree.dtb" \
    0x03000000 "$INITRD_SIZE" "$BOOT_DIR/devicetree-jtag.dtb"

# ── 9. Mirror to vitis_linux/prebuilt (JTAG boot script reads from there) ──
VITIS_PREBUILT="$FPGA_ROOT/vitis_linux/prebuilt"
if [ -d "$VITIS_PREBUILT" ]; then
    cp "$BOOT_DIR/uImage"              "$VITIS_PREBUILT/uImage"
    cp "$BOOT_DIR/devicetree.dtb"      "$VITIS_PREBUILT/devicetree.dtb"
    cp "$BOOT_DIR/devicetree-jtag.dtb" "$VITIS_PREBUILT/devicetree-jtag.dtb"
    cp "$BOOT_DIR/uramdisk.image.gz"   "$VITIS_PREBUILT/uramdisk.image.gz"
    cp "$BOOT_DIR/initramfs.cpio.gz"   "$VITIS_PREBUILT/initramfs.cpio.gz"
    cp "$LINUX_DIR/arch/arm/boot/zImage" "$VITIS_PREBUILT/zImage"
fi

echo ""
echo "=== Done. Boot artifacts in $BOOT_DIR: ==="
ls -la "$BOOT_DIR"/{uImage,devicetree.dtb,devicetree-jtag.dtb,uramdisk.image.gz,initramfs.cpio.gz} 2>/dev/null
