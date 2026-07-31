# vitis_bm — Bare-Metal Vitis GUI Workspace

Standard Vitis 2023.1 bare-metal project (standalone BSP) for the MicroPhase Z7-Lite.
Demonstrates the **USB-UART0 serial console** (MIO 14/15, 115200 8N1) plus live FPGA
register readback over AXI4-Lite, runnable from the Vitis GUI over JTAG (same flow as
the reference MicroPhase `03_dma` project).

The board has **no Ethernet**, so the standard GUI "Run As" (TCF agent over Ethernet)
cannot be used — the app is launched on hardware via JTAG (System Debugger /
"Launch Hardware"). Console output appears on the USB-UART (CH340) terminal.

## Layout

| Path | Description |
|------|-------------|
| `build.tcl` | XSCT script that regenerates the workspace (platform + app + build) |
| `app/src/tmac_serial.c` | App source (imported into the app project by `build.tcl`) |
| `scripts/run_serial.tcl` | Headless XSDB runner (same result as the GUI launch) |
| `z7_bm/`, `tmac_serial/`, `.metadata/` | Generated — **gitignored**, regenerable |

The Vitis workspace is `vitis_bm/` itself (same layout as the reference `03_dma/arm`).
The hardware handoff is shared with the Linux workspace: `../vitis_linux/matmul_bd.xsa`.

## Build (headless)

```
C:\Xilinx\Vitis\2023.1\bin\xsct.bat vitis_bm\build.tcl
```

Creates `z7_bm` (standalone platform, `ps7_cortexa9_0`) and `tmac_serial`
(bare-metal C app), then builds the app.

## Run in the Vitis GUI

1. Power-cycle the board (ps7_init PLL re-lock hang otherwise).
2. Open Vitis: `C:\Xilinx\Vitis\2023.1\bin\vitis.bat`, workspace = `vitis_bm`.
3. Select `tmac_serial` -> Run As -> **Launch Hardware** (Programs FPGA, runs the
   platform FSBL which calls ps7_init, then loads and runs the app).
4. Open a 115200 8N1 terminal on the USB-UART COM port and confirm the banner +
   `tick` output (both direct-register UART and BSP `xil_printf` paths).

## Run headless (no GUI)

```
C:\Xilinx\Vivado\2023.1\bin\xsdb.bat vitis_bm\scripts\run_serial.tcl
```

Loads `vivado_integration/sw/uart_test.elf` (the clang-built standalone serial test,
same UART0 register programming as `tmac_serial.c`). Opens a terminal on the COM port
first to see the output.
