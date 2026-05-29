# Toolchain (Windows, Vivado 2019.2)

## Tools

| Tool | Path |
|------|------|
| Vivado | `D:\Xilinx\Vivado\2019.2\bin\vivado.bat` |
| Vitis  | `D:\Xilinx\Vitis\2019.2\bin\vitis.bat` |

## Device

- Part: `xc7z010clg400-1` (Zynq 7010)

## Vivado Flow (Batch)

Run from `vivado_integration/`:

```powershell
Remove-Item -Recurse -Force proj_bd -ErrorAction SilentlyContinue
D:\Xilinx\Vivado\2019.2\bin\vivado.bat -mode batch -source build_bd.tcl
```

## Vitis Flow (After bitstream)

1. `File → Export → Export Hardware` (include bitstream)
2. `Tools → Launch Vitis`
3. Create new platform from exported `.xsa`
4. Import `sw/` source files
5. Build, run via JTAG

## Notes

- `vivado-local-test` branch only — master branch is for C++ simulation
- Git proxy: `http://127.0.0.1:10810`
- Run Vivado on Windows; use git-bash or PowerShell for git operations
