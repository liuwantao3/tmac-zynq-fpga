CROSS = arm-none-eabi-
CC    = $(CROSS)gcc
LD    = $(CROSS)gcc
OBJDUMP = $(CROSS)objdump
SIZE  = $(CROSS)size

CFLAGS  = -mcpu=cortex-a9 -mthumb -O2 -g -nostdlib -ffreestanding -I.
LDFLAGS = -mcpu=cortex-a9 -mthumb -T link.ld -Wl,-gc-sections -nostdlib

OBJS = startup.o main.o

all: test_int16.elf test_int16.bin

test_int16.elf: $(OBJS) link.ld
	$(LD) $(LDFLAGS) -o $@ $(OBJS)
	$(SIZE) $@

test_int16.bin: test_int16.elf
	$(CROSS)objcopy -O binary $< $@

%.o: %.c regs.h
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.s
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o *.elf *.bin

.PHONY: all clean
