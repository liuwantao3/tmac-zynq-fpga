# MicroPhase Z7-Lite SD boot — U-Boot distro boot auto-runs boot.scr from the
# FAT32 partition. Load addresses match the "manual boot" commands in
# linux/README.md; bootm relocates the dtb/initrd out of the kernel's
# decompression range automatically.
setenv bootargs "console=ttyPS0,115200 root=/dev/ram0 rw iomem=relaxed"
fatload mmc 0 0x03000000 uImage
fatload mmc 0 0x02A00000 devicetree.dtb
fatload mmc 0 0x02000000 uramdisk.image.gz
bootm 0x03000000 0x02000000 0x02A00000
