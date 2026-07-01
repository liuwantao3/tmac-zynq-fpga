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

# ---- Q8_0 core test ----
Q8_SRC  = matmul_q8_core.v dequant_lut.v systolic_8x8.v tb_matmul_q8.v
Q8_TOP  = tb_matmul_q8

sim_q8: $(Q8_SRC)
	$(IVERILOG) $(FLAGS) -o $(Q8_TOP).vvp $(Q8_SRC)
	$(VVP) $(Q8_TOP).vvp

# ---- Q4_K core test ----
Q4K_SRC  = matmul_q4k_core.v tb_matmul_q4k.v
Q4K_TOP  = tb_matmul_q4k

sim_q4k: $(Q4K_SRC)
	$(IVERILOG) $(FLAGS) -o $(Q4K_TOP).vvp $(Q4K_SRC)
	$(VVP) $(Q4K_TOP).vvp

# ---- Q4_K minimal smoke test ----
Q4K_MIN_SRC  = matmul_q4k_core.v tb_minimal_q4k.v
Q4K_MIN_TOP  = tb_minimal_q4k

sim_q4k_min: $(Q4K_MIN_SRC)
	$(IVERILOG) $(FLAGS) -o $(Q4K_MIN_TOP).vvp $(Q4K_MIN_SRC)
	$(VVP) $(Q4K_MIN_TOP).vvp

# ---- Co-simulation (Q8_0 only, requires /tmp/cosim_tiles.bin) ----
COSIM_TOP = tb_cosim
COSIM_SRC = tb_cosim.v matmul_q8_core.v dequant_lut.v systolic_8x8.v

sim_cosim: $(COSIM_SRC)
	$(IVERILOG) $(FLAGS) -o $(COSIM_TOP).vvp $(COSIM_SRC)
	$(VVP) $(COSIM_TOP).vvp

# ---- Co-simulation Q4_K (requires /tmp/cosim_tiles_q4k.bin) ----
COSIM_Q4K_TOP = tb_cosim_q4k
COSIM_Q4K_SRC = tb_cosim_q4k.v matmul_q4k_core.v

sim_cosim_q4k: $(COSIM_Q4K_SRC)
	$(IVERILOG) $(FLAGS) -o $(COSIM_Q4K_TOP).vvp $(COSIM_Q4K_SRC)
	$(VVP) $(COSIM_Q4K_TOP).vvp

# ---- Q5_0 core test ----
Q5_0_SRC  = matmul_q5_0_core.v tb_matmul_q5_0.v
Q5_0_TOP  = tb_matmul_q5_0

sim_q5_0: $(Q5_0_SRC)
	$(IVERILOG) $(FLAGS) -o $(Q5_0_TOP).vvp $(Q5_0_SRC)
	$(VVP) $(Q5_0_TOP).vvp

# ---- Q6_K core test ----
Q6_K_SRC  = matmul_q6_k_core.v tb_matmul_q6_k.v
Q6_K_TOP  = tb_matmul_q6_k

sim_q6_k: $(Q6_K_SRC)
	$(IVERILOG) $(FLAGS) -o $(Q6_K_TOP).vvp $(Q6_K_SRC)
	$(VVP) $(Q6_K_TOP).vvp

# ---- INT16 core smoke test ----
INT16_SRC  = matmul_int16_core.v tb_int16_smoke.v
INT16_TOP  = tb_int16_smoke

sim_int16: $(INT16_SRC)
	$(IVERILOG) $(FLAGS) -o $(INT16_TOP).vvp $(INT16_SRC)
	$(VVP) $(INT16_TOP).vvp

# ---- Q6_K cosim ----
COSIM_Q6K_TOP = tb_cosim_q6_k
COSIM_Q6K_SRC = tb_cosim_q6_k.v matmul_q6_k_core.v

sim_cosim_q6k: $(COSIM_Q6K_SRC)
	$(IVERILOG) $(FLAGS) -o $(COSIM_Q6K_TOP).vvp $(COSIM_Q6K_SRC)
	$(VVP) $(COSIM_Q6K_TOP).vvp

# ---- Q5_0 cosim ----
COSIM_Q5_0_TOP = tb_cosim_q5_0
COSIM_Q5_0_SRC = tb_cosim_q5_0.v matmul_q5_0_core.v

sim_cosim_q5_0: $(COSIM_Q5_0_SRC)
	$(IVERILOG) $(FLAGS) -o $(COSIM_Q5_0_TOP).vvp $(COSIM_Q5_0_SRC)
	$(VVP) $(COSIM_Q5_0_TOP).vvp

# ---- HP FSM comprehensive test (7 test cases) ----
HPFSM_TOP = tb_hw_fsm_comprehensive
HPFSM_SRC = tb_hw_fsm_comprehensive.v ../vivado_integration/rtl/hp_fsm_top.v \
             axihp_read_master.v axihp_write_master.v matmul_q8_core.v

sim_hpfsm: $(HPFSM_SRC)
	$(IVERILOG) $(FLAGS) -o $(HPFSM_TOP).vvp $(HPFSM_SRC)
	$(VVP) $(HPFSM_TOP).vvp

# ---- Phase B descriptor chain test ----
PHASEB_TOP = tb_phaseb
PHASEB_SRC = tb_phaseb.v matmul_top.v sim_ddr_axi_hp.v \
             matmul_int16_core.v matmul_q8_core.v \
             matmul_q4k_core.v matmul_q5_0_core.v matmul_q6_k_core.v \
             axihp_read_master.v axihp_write_master.v

sim_phaseb: $(PHASEB_SRC)
	$(IVERILOG) $(FLAGS) -pARRAY_SIZE_LIMIT=268435456 -o $(PHASEB_TOP).vvp $(PHASEB_SRC)
	$(VVP) $(PHASEB_TOP).vvp

# ---- Run all tests (core unit tests only — phaseb requires C++ dump) ----
all: sim_q8 sim_q4k sim_int16 sim_q5_0 sim_q6_k sim_hpfsm

# ---- Waveform viewing ----
waves-q8:
	gtkwave tb_matmul_q8.vcd

waves-q4k:
	gtkwave tb_matmul_q4k.vcd

waves-cosim:
	gtkwave tb_cosim.vcd

clean:
	rm -f *.vvp *.vcd

.PHONY: all sim_q8 sim_q4k sim_q4k_min sim_cosim sim_cosim_q4k sim_phaseb waves-q8 waves-q4k waves-cosim clean
