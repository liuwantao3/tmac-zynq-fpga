LLVM_DIR = C:/Xilinx/Vivado/2023.1/tps/llvm/7.0/win64/bin
CC  = $(LLVM_DIR)/clang.exe
LD  = $(LLVM_DIR)/ld.lld.exe
OBJDUMP = $(LLVM_DIR)/llvm-objdump.exe
OBJCOPY = $(LLVM_DIR)/llvm-objcopy.exe
SIZE = $(LLVM_DIR)/llvm-size.exe

CFLAGS  = --target=armv7a-none-eabi -mcpu=cortex-a9 -O2 -g -nostdlib -ffreestanding -ffast-math
CXXFLAGS = --target=armv7a-none-eabi -mcpu=cortex-a9 -O1 -g -nostdlib -ffreestanding -ffast-math -fno-exceptions -fno-rtti
LDFLAGS = -T link.ld

OBJS_COMMON = startup.o

all: tmac_baremetal.elf tmac_baremetal.bin test_fpga_cores.elf test_fpga_cores.bin uart_test.elf uart_test.bin

# ===== T-MAC Bare-Metal Inference Engine =====
tmac_baremetal.elf: $(OBJS_COMMON) tmac_baremetal.o link.ld tmac_baremetal.h
	$(LD) $(LDFLAGS) $(OBJS_COMMON) tmac_baremetal.o -o $@
	$(SIZE) $@

tmac_baremetal.bin: tmac_baremetal.elf
	$(OBJCOPY) -O binary $< $@

# ===== Standalone FPGA Core Test Wrapper =====
test_fpga_cores.elf: $(OBJS_COMMON) test_fpga_cores.o link.ld tmac_baremetal.h
	$(LD) $(LDFLAGS) $(OBJS_COMMON) test_fpga_cores.o -o $@
	$(SIZE) $@

test_fpga_cores.bin: test_fpga_cores.elf
	$(OBJCOPY) -O binary $< $@

# ===== UART test =====
uart_test.elf: startup.o uart_test.o link.ld
	$(LD) $(LDFLAGS) startup.o uart_test.o -o $@
	$(SIZE) $@

uart_test.bin: uart_test.elf
	$(OBJCOPY) -O binary $< $@

# Compilation rules
%.o: %.c regs.h tmac_baremetal.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.cpp regs.h tmac_baremetal.h
	$(CC) $(CXXFLAGS) -c -o $@ $<

%.o: %.s
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	-rm -f *.o *.elf *.bin

.PHONY: all clean
