.PHONY: all hls vivado clean help feedback

WORKSPACE := /Users/arctic/fpga
HLS_DIR := $(WORKSPACE)/hls
VIVADO_DIR := $(WORKSPACE)/vivado
SCRIPTS_DIR := $(WORKSPACE)/scripts

all: hls vivado

hls: hls-int16 hls-q8

hls-int16:
	@echo "Running HLS synthesis (INT16)..."
	cd $(HLS_DIR) && /opt/Xilinx/Vitis_HLS/2023.1/bin/vitis_hls -s script_int16.tcl -l hls_int16.log

hls-q8:
	@echo "Running HLS synthesis (Q8_0 LUT path)..."
	cd $(HLS_DIR) && /opt/Xilinx/Vitis_HLS/2023.1/bin/vitis_hls -s script_q8.tcl -l hls_q8.log

vivado:
	@echo "Running Vivado implementation..."
	cd $(VIVADO_DIR) && /opt/Xilinx/Vivado/2023.1/bin/vivado -mode batch -source block_design.tcl -log vivado.log

feedback:
	@python3 $(SCRIPTS_DIR)/feedback_parser.py $(WORKSPACE)/logs

iterate:
	@bash $(SCRIPTS_DIR)/design_iteration.sh all

clean:
	@rm -rf $(HLS_DIR)/solution*
	@rm -rf $(HLS_DIR)/*.log
	@rm -rf $(VIVADO_DIR)/.Xil
	@rm -rf $(VIVADO_DIR)/*.log
	@rm -rf $(WORKSPACE)/logs/*.log
	@rm -f $(WORKSPACE)/logs/feedback_*.json

help:
	@echo "FPGA Development Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  make hls-int16 - Run HLS synthesis (INT16 path)"
	@echo "  make hls-q8    - Run HLS synthesis (Q8_0 LUT path)"
	@echo "  make hls       - Run all HLS synthesis"
	@echo "  make vivado    - Run Vivado implementation"
	@echo "  make all       - Run full pipeline (HLS + Vivado)"
	@echo "  make feedback  - Parse and display feedback"
	@echo "  make iterate   - Run iteration with feedback"
	@echo "  make clean     - Clean build artifacts"