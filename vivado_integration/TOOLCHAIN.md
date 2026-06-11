# Toolchain (Windows, Vivado 2023.1)

## Tools

| Tool | Path |
|------|------|
| Vivado | `C:\Xilinx\Vivado\2023.1\bin\vivado.bat` |
| Vitis  | `C:\Xilinx\Vitis\2023.1\bin\vitis.bat` |
| LLVM/clang (ARM cross-compiler) | `C:\Xilinx\Vivado\2023.1\tps\llvm\7.0\win64\bin\clang.exe` |

## Device

- Part: `xc7z010clg400-1` (Zynq 7010)

## Vivado Flow (Batch)

Run from `vivado_integration/`:

```powershell
Remove-Item -Recurse -Force proj_bd -ErrorAction SilentlyContinue
C:\Xilinx\Vivado\2023.1\bin\vivado.bat -mode batch -source build_bd.tcl
```

## Standalone ARM Build (no Vitis)

Uses Vivado's bundled LLVM clang 7.0 as ARM cross-compiler:

```powershell
cd vivado_integration\sw
C:\Xilinx\Vivado\2023.1\tps\llvm\7.0\win64\bin\clang.exe --target=armv7a-none-eabi -mthumb -mcpu=cortex-a9 -O2 -nostdlib -ffreestanding -I. -c main.c -o main.o
```

## Vitis Flow (After bitstream)

1. `File → Export → Export Hardware` (include bitstream) from Vivado
2. `Tools → Launch Vitis`
3. Create new platform from exported `.xsa`
4. Import `sw/` source files
5. Build, run via JTAG

## XSDB JTAG Load

```powershell
C:\Xilinx\Vivado\2023.1\bin\xsdb.bat D:\Users\u\tmac-zynq-fpga\vivado_integration\sw\load.tcl
```

## Notes

- `vivado-local-test` branch only — master branch is for C++ simulation
- Git proxy: `http://127.0.0.1:10810`
