# Verilog Simulation Makefile
# Requires: iverilog, vvp (from icarus verilog)
# Install on macOS: brew install icarus-verilog
#
# Cosimulation workflow:
#   1. echo "YOUR_PROMPT" | ../sim/tmac_gguf /tmp/model.tmac --fpga-q8 --dump-tiles 10
#   2. make cosim

IVERILOG = iverilog
VVP      = vvp
FLAGS    = -Wall -g2012

SOURCES     = matmul_q8_core.v dequant_lut.v systolic_8x8.v tb_matmul_q8.v
TOP         = tb_matmul_q8
COSIM_TOP   = tb_cosim
COSIM_SRC   = tb_cosim.v matmul_q8_core.v dequant_lut.v systolic_8x8.v

all: sim

sim: $(SOURCES)
	$(IVERILOG) $(FLAGS) -o $(TOP).vvp $(SOURCES)
	$(VVP) $(TOP).vvp

cosim: $(COSIM_SRC)
	$(IVERILOG) $(FLAGS) -o $(COSIM_TOP).vvp $(COSIM_SRC)
	$(VVP) $(COSIM_TOP).vvp

waves:
	gtkwave tb_matmul_q8.vcd

cosim-waves:
	gtkwave tb_cosim.vcd

clean:
	rm -f *.vvp *.vcd

.PHONY: all sim cosim waves cosim-waves clean
