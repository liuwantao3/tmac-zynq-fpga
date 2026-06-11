LLVM_DIR = C:/Xilinx/Vivado/2023.1/tps/llvm/7.0/win64/bin
CC  = $(LLVM_DIR)/clang.exe
LD  = $(LLVM_DIR)/ld.lld.exe
OBJDUMP = $(LLVM_DIR)/llvm-objdump.exe
OBJCOPY = $(LLVM_DIR)/llvm-objcopy.exe
SIZE = $(LLVM_DIR)/llvm-size.exe

CFLAGS  = --target=armv7a-none-eabi -mthumb -mcpu=cortex-a9 -O2 -g -nostdlib -ffreestanding -I.
LDFLAGS = -T link.ld

OBJS = startup.o main.o

all: test_int16.elf test_int16.bin

test_int16.elf: $(OBJS) link.ld
	$(LD) $(LDFLAGS) $(OBJS) -o $@
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
