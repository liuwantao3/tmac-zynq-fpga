# FPGA Accelerator Linux-on-SD Boot

**Two-machine split:** Windows (Vivado) → bitstream + FSBL + BOOT.BIN. Mac (Lima VM) → U-Boot + kernel + initramfs.

Hardware: MicroPhase Z7-Lite (xc7z010clg400-1), UART0 MIO14/15 CH340 broken — DCC console via JTAG.

## Quickstart: macOS Build

### One-time: Lima VM setup

```bash
brew install lima
limactl start --name=linux-build --cpus=8 --memory=16 template://ubuntu
limactl shell linux-build
sudo apt update && sudo apt install -y \
    gcc-arm-linux-gnueabihf build-essential flex bison bc libelf-dev libssl-dev \
    busybox-static cpio git
```

### Clone repo (includes FPGA bitstream + XSA from Windows)

```bash
git clone <your-repo-url> ~/tmac-zynq-fpga
cd ~/tmac-zynq-fpga
```

### 1. Build U-Boot (with DCC console)

```bash
cd ~
git clone --depth=1 --branch xilinx-v2022.1 \
    https://github.com/Xilinx/u-boot-xlnx.git
cd ~/u-boot-xlnx
export CROSS_COMPILE=arm-linux-gnueabihf-

# Enable DCC console (ARM JTAG Debug Communication Channel)
cat >> configs/xilinx_zynq_virt_defconfig << 'EOF'
CONFIG_ARM_DCC=y
CONFIG_SERIAL_ARM_DCC=y
CONFIG_BAUDRATE=115200
EOF
make xilinx_zynq_virt_defconfig
# Fix SPL build for Zynq 7010
sed -i 's|@dd if=$$< of=$$@ conv=block,sync bs=4 2>/dev/null;|@cp $$< $$@|' scripts/Makefile.spl
make -j$(nproc)
cp u-boot u-boot.elf
cp u-boot-spl.bin u-boot-spl.bin
```

### 2. Build Linux kernel (with DCC earlycon)

```bash
cd ~
git clone --depth=1 --branch xilinx-v2024.1 \
    https://github.com/Xilinx/linux-xlnx.git
cd ~/linux-xlnx
export CROSS_COMPILE=arm-linux-gnueabihf-

make ARCH=arm xilinx_zynq_defconfig
# Enable DCC early console
./scripts/config --enable SERIAL_ARM_DCC
./scripts/config --enable SERIAL_ARM_DCC_CONSOLE
./scripts/config --enable DEBUG_LL
./scripts/config --enable EARLY_PRINTK
./scripts/config --set-str CMDLINE "earlycon=dcc console=hvc0 root=/dev/ram0 rw iomem=relaxed"
make -j$(nproc) ARCH=arm UIMAGE_LOADADDR=0x8000 uImage dtbs
cp arch/arm/boot/uImage ~/tmac-zynq-fpga/linux/boot/
cp arch/arm/boot/dts/zynq-zc702.dtb ~/tmac-zynq-fpga/linux/boot/devicetree.dtb
```

### 3. Build initramfs with tmac

```bash
cd ~/tmac-zynq-fpga/linux
# Build tmac-static ARM binary
arm-linux-gnueabihf-gcc -static -O2 -o tmac tmac_linux.c -lm
cp tmac boot/

# Create initramfs
mkdir -p /tmp/initramfs/{bin,dev,proc,sys,root,tmp,etc}
cp /bin/busybox /tmp/initramfs/bin/
cd /tmp/initramfs
for cmd in sh mount umount ls cat echo mknod sleep dmesg cp mv rm \
    grep sed awk hexdump md5sum devmem ps kill top free vi \
    fdisk mkfs.ext2 blkid ifconfig ping wget modprobe sync \
    reboot poweroff halt; do
    ln -sf /bin/busybox bin/$cmd
done
cp ~/tmac-zynq-fpga/linux/boot/tmac bin/

cat > init << 'INIT'
#!/bin/sh
mount -t proc none /proc
mount -t sysfs none /sys
mount -t devtmpfs none /dev
echo "=== FPGA Linux Boot — Zynq 7010 (DCC console) ==="
echo "DCC: JTAG debug channel — connect XSDB and run:"
echo "  xsdb> readjtaguart -start"
echo ""
# Mount SD ext4 partition if present
mount /dev/mmcblk0p2 /root 2>/dev/null && echo "SD ext4 mounted at /root"
echo "Ready. Run /root/tmac for FPGA test."
exec /bin/sh
INIT
chmod +x init
find . | cpio -o -H newc | gzip > ~/tmac-zynq-fpga/linux/boot/uramdisk.image.gz
```

### 4. Deploy U-Boot files

```bash
cp ~/u-boot-xlnx/u-boot.elf ~/tmac-zynq-fpga/linux/boot/
cp ~/u-boot-xlnx/u-boot-spl.bin ~/tmac-zynq-fpga/linux/boot/
```

### 5. Build BOOT.BIN (must be done on Windows with Vivado)

See "Windows: BOOT.BIN + SD Card" section below.

---

## Windows: BOOT.BIN + SD Card

**Prerequisites:** Vivado 2023.1 (provides `bootgen` and FSBL build).

### Step 1: Ensure boot files

```
linux/boot/
<<<<<<< HEAD
├── system_wrapper.bit     ← already in repo (FPGA bitstream from Vivado)
├── matmul_bd.xsa          ← already in repo (hardware handoff)
├── boot.bif               ← already in repo (bootgen config)
├── fsbl.elf               ← build in Vivado SDK (see below)
├── u-boot-spl.bin         ← copy from ~/arm-build/
├── u-boot.img             ← copy from ~/arm-build/
├── uImage                 ← copy from ~/arm-build/
├── devicetree.dtb          ← copy from ~/arm-build/
└── uramdisk.image.gz       ← copy from ~/arm-build/
=======
├── system_wrapper.bit   ← committed (from repo, built in Vivado on Windows)
├── matmul_bd.xsa        ← committed (hardware handoff)
├── boot.bif             ← committed (bootgen config)
├── u-boot.elf           ← copy from Mac build
├── u-boot-spl.bin       ← copy from Mac build
├── uImage               ← copy from Mac build
├── devicetree.dtb        ← copy from Mac build
├── uramdisk.image.gz     ← copy from Mac build
└── tmac                 ← copy from Mac build
>>>>>>> 8607191 (DCC console integration + cleanup + Mac rebuild guide)
```

### Step 2: Build FSBL

```tcl
# In Vivado 2023.1 Tcl Console or XSCT:
hsi::open_hw_design linux/boot/matmul_bd.xsa
hsi::generate_app -hw linux/boot/matmul_bd.xsa -os standalone -proc ps7_cortexa9_0 -app zynq_fsbl
# Copy fsbl.elf from generated SDK project to linux/boot/
```

### Step 3: Create BOOT.BIN

```cmd
cd D:\Users\u\tmac-zynq-fpga\linux\boot
bootgen -image boot.bif -o BOOT.BIN -w
```

`boot.bif` contents:
```
the_ROM_image:
{
    [bootloader] u-boot-spl.bin
    system_wrapper.bit
    u-boot-spl.bin
}
```

### Step 4: Format SD Card

| Partition | Type | Size | Contents |
|-----------|------|------|----------|
| 1 | FAT32 | 64 MB | BOOT.BIN, uImage, devicetree.dtb, uramdisk.image.gz |
| 2 | ext4 | Rest | model.tmac (~374 MB), tmac |

### Step 5: Boot

<<<<<<< HEAD
```
Partition 1 (FAT32):
    BOOT.BIN
    u-boot.img
    uImage
    devicetree.dtb
    uramdisk.image.gz
=======
1. Power-cycle the board (required — PLL re-init hangs on warm reset)
2. Insert SD, set boot mode DIP to SD
3. Connect JTAG, open XSDB, capture DCC console:
   ```tcl
   xsdb> connect
   xsdb> readjtaguart -start
   ```
4. Power on — U-Boot boots from SD, loads Linux, runs initramfs
5. All console output appears via JTAG DCC (captured by `readjtaguart`)
6. To stop capture: `xsdb> readjtaguart -stop`
>>>>>>> 8607191 (DCC console integration + cleanup + Mac rebuild guide)

### U-Boot Manual Boot (if auto-boot fails)

```
U-Boot> fatload mmc 0 0x3000000 uImage
U-Boot> fatload mmc 0 0x2A00000 devicetree.dtb
U-Boot> fatload mmc 0 0x2000000 uramdisk.image.gz
U-Boot> setenv bootargs "earlycon=dcc console=hvc0 root=/dev/ram0 rw iomem=relaxed"
U-Boot> bootm 0x3000000 0x2000000 0x2A00000
```

---

## JTAG Boot (no SD card, for testing)

Use `xsdb` to load everything over JTAG:

```tcl
# From repo root:
xsdb linux/boot/jtag_boot.tcl
```

This loads bitstream → PS7 init → kernel/DTB/initrd/tmac to DDR → starts U-Boot.
Console via DCC: `readjtaguart -start` before `con`.

---

## DCC Console vs UART

| Feature | UART | DCC (recommended) |
|---------|------|-------------------|
| Hardware | CH340 USB-UART (broken) | JTAG (Digilent HS-2, already connected) |
| Speed | 115200 baud (~11 KB/s) | ~200-500 KB/s |
| Console | `ttyPS0` | `hvc0` |
| U-Boot config | default | `CONFIG_ARM_DCC=y` |
| Kernel bootargs | `console=ttyPS0,115200` | `earlycon=dcc console=hvc0` |
| Capture | Serial terminal (PuTTY) | `readjtaguart -start` in XSDB |

The physical CH340 RX pin is dead. PS7 UART TX works but nobody can hear it.
DCC uses the same JTAG cable already used for FPGA programming and debug — no extra hardware.
