LLVM_DIR = C:/Xilinx/Vivado/2023.1/tps/llvm/7.0/win64/bin
CC  = $(LLVM_DIR)/clang.exe
LD  = $(LLVM_DIR)/ld.lld.exe
OBJDUMP = $(LLVM_DIR)/llvm-objdump.exe
OBJCOPY = $(LLVM_DIR)/llvm-objcopy.exe
SIZE = $(LLVM_DIR)/llvm-size.exe

CFLAGS  = --target=armv7a-none-eabi -mcpu=cortex-a9 -O2 -g -nostdlib -ffreestanding
LDFLAGS = -T link.ld -Map=hp_baremetal.map

OBJS_COMMON = startup.o

all: hp_baremetal.elf hp_baremetal.bin test_int16.elf test_int16.bin

# Baremetal HP test (no BSP, MMU off before main)
hp_baremetal.elf: $(OBJS_COMMON) hp_baremetal.o link.ld
	$(LD) $(LDFLAGS) $(OBJS_COMMON) hp_baremetal.o -o $@
	$(SIZE) $@

hp_baremetal.bin: hp_baremetal.elf
	$(OBJCOPY) -O binary $< $@

# Legacy int16 test
test_int16.elf: $(OBJS_COMMON) main.o link.ld
	$(LD) $(LDFLAGS) $(OBJS_COMMON) main.o -o $@
	$(SIZE) $@

test_int16.bin: test_int16.elf
	$(OBJCOPY) -O binary $< $@

%.o: %.c regs.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.s
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	-rm -f *.o *.elf *.bin

.PHONY: all clean
