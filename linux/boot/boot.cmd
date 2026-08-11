# MicroPhase Z7-Lite SD boot — U-Boot distro boot auto-runs boot.scr from the
# FAT32 partition. Load addresses match the proven-working manual boot
# (kernel at 0x03000000; bootm relocates dtb/initrd out of the way).
setenv boot_targets mmc0
setenv bootargs "console=ttyPS0,115200 root=/dev/ram0 rw iomem=relaxed"
fatload mmc 0 0x03000000 uImage
fatload mmc 0 0x02A00000 devicetree.dtb
fatload mmc 0 0x02000000 uramdisk.image.gz
bootm 0x03000000 0x02000000 0x02A00000
