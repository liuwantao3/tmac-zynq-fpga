`timescale 1ns / 1ps

module matmul_q8_core (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire         op_vecmul,
    output reg          done,
    output reg          busy,
    input  wire         wt_we,
    input  wire [8:0]   wt_addr,
    input  wire [63:0]  wt_din,
    input  wire         sc_we,
    input  wire [6:0]   sc_addr,
    input  wire [15:0]  sc_din,
    input  wire         act_we,
    input  wire [5:0]   act_addr,
    input  wire [15:0]  act_din,
    input  wire [5:0]   res_addr,
    output wire [47:0]  res_dout,
    output wire [2:0]   dbg_state,
    output wire [5:0]   dbg_k,
    output wire [2:0]   dbg_g
);

    localparam IDLE      = 4'd0;
    localparam CLEAR_ACC = 4'd1;
    localparam COMPUTE   = 4'd2;
    localparam DRAIN     = 4'd3;
    localparam DRAIN2    = 4'd4;
    localparam DRAIN3    = 4'd5;
    localparam DRAIN4    = 4'd6;

    reg [3:0] state;
    reg [5:0] k;
    reg [2:0] g;
    assign dbg_state = state;
    assign dbg_k = k;
    assign dbg_g = g;
    integer wi_i;

    // ======================================================================
    // Weight storage: 8 BRAM banks, one per column lane
    //   Each bank = 512 x 8-bit (padded to BRAM18 min depth)
    //   bank_i stores byte lane i for all (group, column) pairs.
    //   Write: broadcast 64-bit word, bank_i <= wt_din[i*8+7:i*8]
    //   Read:  all banks read at address {g, k}, output byte per lane
    // ======================================================================
    (* ram_style = "block" *) reg [7:0] wmem_bank0 [0:511];
    (* ram_style = "block" *) reg [7:0] wmem_bank1 [0:511];
    (* ram_style = "block" *) reg [7:0] wmem_bank2 [0:511];
    (* ram_style = "block" *) reg [7:0] wmem_bank3 [0:511];
    (* ram_style = "block" *) reg [7:0] wmem_bank4 [0:511];
    (* ram_style = "block" *) reg [7:0] wmem_bank5 [0:511];
    (* ram_style = "block" *) reg [7:0] wmem_bank6 [0:511];
    (* ram_style = "block" *) reg [7:0] wmem_bank7 [0:511];

    always @(posedge clk) begin
        if (wt_we) begin
            wmem_bank0[wt_addr] <= wt_din[7:0];
            wmem_bank1[wt_addr] <= wt_din[15:8];
            wmem_bank2[wt_addr] <= wt_din[23:16];
            wmem_bank3[wt_addr] <= wt_din[31:24];
            wmem_bank4[wt_addr] <= wt_din[39:32];
            wmem_bank5[wt_addr] <= wt_din[47:40];
            wmem_bank6[wt_addr] <= wt_din[55:48];
            wmem_bank7[wt_addr] <= wt_din[63:56];
        end
    end

    // ======================================================================
    // Scale storage: 8 BRAM banks (one per column lane)
    //   Each bank = 512 x 16-bit (padded, 16 entries used)
    //   bank wi_i holds scales for column lane wi_i.
    //   Read address (shared across all banks): {g, k[5]} (4 bits, 0..15)
    //   Write: bank = sc_addr[3:1], local_addr = {sc_addr[6:4], sc_addr[0]}
    // ======================================================================
    (* ram_style = "block" *) reg [15:0] smem_bank0 [0:511];
    (* ram_style = "block" *) reg [15:0] smem_bank1 [0:511];
    (* ram_style = "block" *) reg [15:0] smem_bank2 [0:511];
    (* ram_style = "block" *) reg [15:0] smem_bank3 [0:511];
    (* ram_style = "block" *) reg [15:0] smem_bank4 [0:511];
    (* ram_style = "block" *) reg [15:0] smem_bank5 [0:511];
    (* ram_style = "block" *) reg [15:0] smem_bank6 [0:511];
    (* ram_style = "block" *) reg [15:0] smem_bank7 [0:511];

    // Write address decode
    wire [2:0] smem_wbank = sc_addr[3:1];
    wire [3:0] smem_waddr = {sc_addr[6:4], sc_addr[0]};

    always @(posedge clk) begin
        if (sc_we) begin
            case (smem_wbank)
                0: smem_bank0[smem_waddr] <= sc_din;
                1: smem_bank1[smem_waddr] <= sc_din;
                2: smem_bank2[smem_waddr] <= sc_din;
                3: smem_bank3[smem_waddr] <= sc_din;
                4: smem_bank4[smem_waddr] <= sc_din;
                5: smem_bank5[smem_waddr] <= sc_din;
                6: smem_bank6[smem_waddr] <= sc_din;
                7: smem_bank7[smem_waddr] <= sc_din;
            endcase
        end
    end

    // ======================================================================
    // Activation BRAM: 512 x 16-bit (64 entries used)
    // ======================================================================
    (* ram_style = "block" *) reg signed [15:0] act_bram [0:511];

    always @(posedge clk) begin
        if (act_we)
            act_bram[act_addr] <= act_din;
    end

    // ======================================================================
    // Accumulator banks: 8 banks x 48-bit BRAM (padded to 512 depth)
    // ======================================================================
    (* ram_style = "block" *) reg signed [47:0] acc_b0 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b1 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b2 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b3 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b4 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b5 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b6 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b7 [0:511];

    // BRAM read outputs
    reg signed [47:0] acc_r [0:7];

    always @(posedge clk) begin
        acc_r[0] <= acc_b0[pre_read_g];
        acc_r[1] <= acc_b1[pre_read_g];
        acc_r[2] <= acc_b2[pre_read_g];
        acc_r[3] <= acc_b3[pre_read_g];
        acc_r[4] <= acc_b4[pre_read_g];
        acc_r[5] <= acc_b5[pre_read_g];
        acc_r[6] <= acc_b6[pre_read_g];
        acc_r[7] <= acc_b7[pre_read_g];
    end

    // Result read: pipelined from specific bank
    reg [47:0] res_dout_r;
    always @(posedge clk) begin
        case (res_addr[2:0])
            3'd0: res_dout_r <= acc_b0[res_addr[5:3]];
            3'd1: res_dout_r <= acc_b1[res_addr[5:3]];
            3'd2: res_dout_r <= acc_b2[res_addr[5:3]];
            3'd3: res_dout_r <= acc_b3[res_addr[5:3]];
            3'd4: res_dout_r <= acc_b4[res_addr[5:3]];
            3'd5: res_dout_r <= acc_b5[res_addr[5:3]];
            3'd6: res_dout_r <= acc_b6[res_addr[5:3]];
            3'd7: res_dout_r <= acc_b7[res_addr[5:3]];
        endcase
    end
    assign res_dout = res_dout_r;

    // BRAM pre-read address (alias g from dq pipeline)
    reg [2:0] pre_read_g;
    reg [4:0] acc_clr_cnt;

    // ======================================================================
    // BRAM read data (registered outputs from wmem and smem banks, act_bram)
    // ======================================================================
    reg [7:0]  wmem_rdata [0:7];
    reg [15:0] smem_rdata [0:7];
    reg signed [15:0] act_rdata;

    // ======================================================================
    // PRE stage: register addresses for BRAM reads
    //   pre_k/pre_g: g,k values for the current iteration (captured before
    //   counter update). Anticipates k+1 when g wraps.
    // ======================================================================
    reg [5:0]  pre_k;
    reg [2:0]  pre_g;
    reg        pre_valid;

    // ======================================================================
    // Pipeline registers (6-stage)
    //   PRE:   register BRAM read addresses
    //   Stage 0: capture BRAM outputs (wmem_rdata, smem_rdata, act_rdata)
    //   Stage 1a: dequant (wmem_rdata * sc >> 8)
    //   Stage 1b: multiply with activation (dq_act * dq_deq)
    //   Stage 2a: BRAM acc read (1-cycle latency)
    //   Stage 2b: accumulate into acc BRAM
    // ======================================================================
    reg [2:0]  p1_g;
    reg [5:0]  p1_k;
    reg [5:0]  p1_row_base;
    reg signed [15:0] p1_act;
    reg [15:0] p1_sc [0:7];
    reg        p1_valid;

    reg signed [15:0] dq_deq [0:7];
    reg signed [15:0] dq_act;
    reg [5:0]  dq_row_base;
    reg        dq_valid;

    reg signed [47:0] p2_partial [0:7];
    reg [5:0]  p2_row_base;
    reg        p2_valid;

    reg        p2a_valid;

`ifdef __ICARUS__
    integer _init_i;
    initial begin
        for (_init_i = 0; _init_i < 512; _init_i = _init_i + 1) begin
            wmem_bank0[_init_i] = 0; wmem_bank1[_init_i] = 0;
            wmem_bank2[_init_i] = 0; wmem_bank3[_init_i] = 0;
            wmem_bank4[_init_i] = 0; wmem_bank5[_init_i] = 0;
            wmem_bank6[_init_i] = 0; wmem_bank7[_init_i] = 0;
            smem_bank0[_init_i] = 0; smem_bank1[_init_i] = 0;
            smem_bank2[_init_i] = 0; smem_bank3[_init_i] = 0;
            smem_bank4[_init_i] = 0; smem_bank5[_init_i] = 0;
            smem_bank6[_init_i] = 0; smem_bank7[_init_i] = 0;
            act_bram[_init_i] = 0;
            acc_b0[_init_i] = 0; acc_b1[_init_i] = 0;
            acc_b2[_init_i] = 0; acc_b3[_init_i] = 0;
            acc_b4[_init_i] = 0; acc_b5[_init_i] = 0;
            acc_b6[_init_i] = 0; acc_b7[_init_i] = 0;
        end
    end
`endif

    // ======================================================================
    // FSM
    // ======================================================================
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            k <= 0;
            g <= 0;
            done <= 0;
            busy <= 0;
            pre_valid <= 0;
            pre_g <= 0;
            pre_k <= 0;
            pre_wmem_addr <= 0;
            pre_smem_addr <= 0;
            pre_act_addr <= 0;
            p1_valid <= 0;
            dq_valid <= 0;
            p2_valid <= 0;
            p2a_valid <= 0;
            pre_read_g <= 0;
            acc_clr_cnt <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    busy <= 0;
                    pre_valid <= 0;
                    p1_valid <= 0;
                    dq_valid <= 0;
                    p2_valid <= 0;
                    p2a_valid <= 0;
                    if (start) begin
                        busy <= 1;
                        k <= 0;
                        g <= 0;
                        acc_clr_cnt <= 0;
                        state <= CLEAR_ACC;
                    end
                end

                CLEAR_ACC: begin
                    for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1) begin
                        case (wi_i)
                            0: acc_b0[acc_clr_cnt] <= 0;
                            1: acc_b1[acc_clr_cnt] <= 0;
                            2: acc_b2[acc_clr_cnt] <= 0;
                            3: acc_b3[acc_clr_cnt] <= 0;
                            4: acc_b4[acc_clr_cnt] <= 0;
                            5: acc_b5[acc_clr_cnt] <= 0;
                            6: acc_b6[acc_clr_cnt] <= 0;
                            7: acc_b7[acc_clr_cnt] <= 0;
                        endcase
                    end
                    if (acc_clr_cnt == 7) begin
                        // Pre-load BRAM addresses for first COMPUTE read (g=0,k=0)
                        pre_wmem_addr <= {3'd0, 6'd0};
                        pre_smem_addr <= {3'd0, 1'b0};
                        pre_act_addr <= 6'd0;
                        pre_valid <= 1;  // pre-valid for first Stage 0 capture
                        state <= COMPUTE;
                    end else begin
                        acc_clr_cnt <= acc_clr_cnt + 1;
                    end
                end

                COMPUTE: begin
                    // === Stage 2b: accumulate into BRAM (read-modify-write) ===
                    if (p2_valid) begin
                        acc_b0[p2_row_base[5:3]] <= acc_b0[p2_row_base[5:3]] + p2_partial[0];
                        acc_b1[p2_row_base[5:3]] <= acc_b1[p2_row_base[5:3]] + p2_partial[1];
                        acc_b2[p2_row_base[5:3]] <= acc_b2[p2_row_base[5:3]] + p2_partial[2];
                        acc_b3[p2_row_base[5:3]] <= acc_b3[p2_row_base[5:3]] + p2_partial[3];
                        acc_b4[p2_row_base[5:3]] <= acc_b4[p2_row_base[5:3]] + p2_partial[4];
                        acc_b5[p2_row_base[5:3]] <= acc_b5[p2_row_base[5:3]] + p2_partial[5];
                        acc_b6[p2_row_base[5:3]] <= acc_b6[p2_row_base[5:3]] + p2_partial[6];
                        acc_b7[p2_row_base[5:3]] <= acc_b7[p2_row_base[5:3]] + p2_partial[7];
                    end

                    // === Stage 2a: pre-read acc BRAM ===
                    if (dq_valid) begin
                        pre_read_g <= dq_row_base[5:3];
                        p2a_valid <= 1;
                    end else begin
                        p2a_valid <= 0;
                    end

                    // === Stage 1b: multiply ===
                    if (dq_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            p2_partial[wi_i] <= $signed(dq_act) * dq_deq[wi_i];
                        p2_row_base <= dq_row_base;
                        p2_valid <= 1;
                    end else begin
                        p2_valid <= 0;
                    end

                    // === Stage 1a: dequant ===
                    if (p1_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            dq_deq[wi_i] <= dequant_q8(
                                $signed(wmem_rdata[wi_i]), p1_sc[wi_i]);
                        dq_act <= p1_act;
                        dq_row_base <= p1_row_base;
                        dq_valid <= 1;
                    end else begin
                        dq_valid <= 0;
                    end

                    // === Stage 0: read from BRAM outputs ===
                    p1_g <= pre_g;
                    p1_k <= pre_k;
                    p1_row_base <= {pre_g, 3'b0};
                    p1_act <= act_rdata;
                    for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                        p1_sc[wi_i] <= smem_rdata[wi_i];
                    p1_valid <= pre_valid;

                    // === PRE stage: register anticipated (g,k) for pipeline ===
                    // BRAM addr reg sets addresses for next (g_next,k_next), and
                    // the BRAM reads them 1 cycle later when counter has updated.
                    // PRE anticipates the same (g_next,k_next) so pipeline meta
                    // matches the BRAM data that arrives next cycle.
                    if (g == 7) begin
                        pre_k <= k + 1;
                        pre_g <= 0;
                    end else begin
                        pre_k <= k;
                        pre_g <= g + 1;
                    end
                    pre_valid <= 1;

                    // Counter update
                    if (g == 7) begin
                        g <= 0;
                        if (k == 63)
                            state <= DRAIN;
                        else
                            k <= k + 1;
                    end else begin
                        g <= g + 1;
                    end
                end

                DRAIN: begin
                    // Stage 2b
                    if (p2_valid) begin
                        acc_b0[p2_row_base[5:3]] <= acc_b0[p2_row_base[5:3]] + p2_partial[0];
                        acc_b1[p2_row_base[5:3]] <= acc_b1[p2_row_base[5:3]] + p2_partial[1];
                        acc_b2[p2_row_base[5:3]] <= acc_b2[p2_row_base[5:3]] + p2_partial[2];
                        acc_b3[p2_row_base[5:3]] <= acc_b3[p2_row_base[5:3]] + p2_partial[3];
                        acc_b4[p2_row_base[5:3]] <= acc_b4[p2_row_base[5:3]] + p2_partial[4];
                        acc_b5[p2_row_base[5:3]] <= acc_b5[p2_row_base[5:3]] + p2_partial[5];
                        acc_b6[p2_row_base[5:3]] <= acc_b6[p2_row_base[5:3]] + p2_partial[6];
                        acc_b7[p2_row_base[5:3]] <= acc_b7[p2_row_base[5:3]] + p2_partial[7];
                    end

                    // Stage 2a
                    if (dq_valid) begin
                        pre_read_g <= dq_row_base[5:3];
                    end

                    // Stage 1b
                    if (dq_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            p2_partial[wi_i] <= $signed(dq_act) * dq_deq[wi_i];
                        p2_row_base <= dq_row_base;
                        p2_valid <= 1;
                    end else begin
                        p2_valid <= 0;
                    end

                    // Stage 1a: process last entry from S0
                    if (p1_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            dq_deq[wi_i] <= dequant_q8(
                                $signed(wmem_rdata[wi_i]), p1_sc[wi_i]);
                        dq_act <= p1_act;
                        dq_row_base <= p1_row_base;
                        dq_valid <= 1;
                    end else begin
                        dq_valid <= 0;
                    end

                    // No Stage 0 — pipeline drain only, no new BRAM data
                    p1_valid <= 0;

                    // PRE stops
                    pre_valid <= 0;
                    state <= DRAIN2;
                end

                DRAIN2: begin
                    // Stage 2b
                    if (p2_valid) begin
                        acc_b0[p2_row_base[5:3]] <= acc_b0[p2_row_base[5:3]] + p2_partial[0];
                        acc_b1[p2_row_base[5:3]] <= acc_b1[p2_row_base[5:3]] + p2_partial[1];
                        acc_b2[p2_row_base[5:3]] <= acc_b2[p2_row_base[5:3]] + p2_partial[2];
                        acc_b3[p2_row_base[5:3]] <= acc_b3[p2_row_base[5:3]] + p2_partial[3];
                        acc_b4[p2_row_base[5:3]] <= acc_b4[p2_row_base[5:3]] + p2_partial[4];
                        acc_b5[p2_row_base[5:3]] <= acc_b5[p2_row_base[5:3]] + p2_partial[5];
                        acc_b6[p2_row_base[5:3]] <= acc_b6[p2_row_base[5:3]] + p2_partial[6];
                        acc_b7[p2_row_base[5:3]] <= acc_b7[p2_row_base[5:3]] + p2_partial[7];
                    end

                    // Stage 2a: nothing (dq_valid might still be active)
                    if (dq_valid) begin
                        pre_read_g <= dq_row_base[5:3];
                    end

                    // Stage 1b
                    if (dq_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            p2_partial[wi_i] <= $signed(dq_act) * dq_deq[wi_i];
                        p2_row_base <= dq_row_base;
                        p2_valid <= 1;
                    end else begin
                        p2_valid <= 0;
                    end

                    // Stage 1a: last from p1_valid (which was set in DRAIN)
                    if (p1_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            dq_deq[wi_i] <= dequant_q8(
                                $signed(wmem_rdata[wi_i]), p1_sc[wi_i]);
                        dq_act <= p1_act;
                        dq_row_base <= p1_row_base;
                        dq_valid <= 1;
                        p1_valid <= 0;
                    end else begin
                        dq_valid <= 0;
                    end

                    state <= DRAIN3;
                end

                DRAIN3: begin
                    // Stage 2b
                    if (p2_valid) begin
                        acc_b0[p2_row_base[5:3]] <= acc_b0[p2_row_base[5:3]] + p2_partial[0];
                        acc_b1[p2_row_base[5:3]] <= acc_b1[p2_row_base[5:3]] + p2_partial[1];
                        acc_b2[p2_row_base[5:3]] <= acc_b2[p2_row_base[5:3]] + p2_partial[2];
                        acc_b3[p2_row_base[5:3]] <= acc_b3[p2_row_base[5:3]] + p2_partial[3];
                        acc_b4[p2_row_base[5:3]] <= acc_b4[p2_row_base[5:3]] + p2_partial[4];
                        acc_b5[p2_row_base[5:3]] <= acc_b5[p2_row_base[5:3]] + p2_partial[5];
                        acc_b6[p2_row_base[5:3]] <= acc_b6[p2_row_base[5:3]] + p2_partial[6];
                        acc_b7[p2_row_base[5:3]] <= acc_b7[p2_row_base[5:3]] + p2_partial[7];
                    end

                    // Stage 1b
                    if (dq_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            p2_partial[wi_i] <= $signed(dq_act) * dq_deq[wi_i];
                        p2_row_base <= dq_row_base;
                        p2_valid <= 1;
                    end else begin
                        p2_valid <= 0;
                    end

                    dq_valid <= 0;
                    state <= DRAIN4;
                end

                DRAIN4: begin
                    if (p2_valid) begin
                        acc_b0[p2_row_base[5:3]] <= acc_b0[p2_row_base[5:3]] + p2_partial[0];
                        acc_b1[p2_row_base[5:3]] <= acc_b1[p2_row_base[5:3]] + p2_partial[1];
                        acc_b2[p2_row_base[5:3]] <= acc_b2[p2_row_base[5:3]] + p2_partial[2];
                        acc_b3[p2_row_base[5:3]] <= acc_b3[p2_row_base[5:3]] + p2_partial[3];
                        acc_b4[p2_row_base[5:3]] <= acc_b4[p2_row_base[5:3]] + p2_partial[4];
                        acc_b5[p2_row_base[5:3]] <= acc_b5[p2_row_base[5:3]] + p2_partial[5];
                        acc_b6[p2_row_base[5:3]] <= acc_b6[p2_row_base[5:3]] + p2_partial[6];
                        acc_b7[p2_row_base[5:3]] <= acc_b7[p2_row_base[5:3]] + p2_partial[7];
                    end
                    p2_valid <= 0;
                    busy <= 0;
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

    // ======================================================================
    // BRAM read address registration and read data capture
    // These must be outside the FSM case block because they run every cycle
    // regardless of state (BRAM reads happen on every clock edge, even when
    // data is not valid — the outputs are simply not captured by pipeline
    // registers unless p1_valid/dq_valid etc. indicate valid data).
    //
    // However, to save power, we only update addresses when in COMPUTE/DRAIN.
    // ======================================================================

    // PRE-stage address registration (only update in COMPUTE state)
    reg [8:0]  pre_wmem_addr;  // {g,k} for wmem banks
    reg [3:0]  pre_smem_addr;  // {g,k[5]} for smem banks
    reg [5:0]  pre_act_addr;   // k for act_bram

    always @(posedge clk) begin
        if (state == COMPUTE) begin
            // Anticipatory BRAM address registration: set address for the
            // NEXT iteration's (g,k). The BRAM reads this address on the
            // next cycle, by which time the counter has updated to (g_next,k_next).
            if (g == 7) begin
                pre_wmem_addr <= {3'd0, k + 6'd1};  // g=0, k+1
                pre_smem_addr <= {3'd0, (k + 6'd1) >= 32};  // (k+1)[5]
                pre_act_addr <= k + 6'd1;
            end else begin
                pre_wmem_addr <= {g + 3'd1, k};  // g+1, k
                pre_smem_addr <= {g + 3'd1, k[5]};
                pre_act_addr <= k;
            end
        end
    end

    // BRAM synchronous reads (1-cycle latency)
    // These run every cycle, but their outputs are only captured by pipeline
    // registers when pre_valid/p1_valid chain indicates valid data.
    always @(posedge clk) begin
        wmem_rdata[0] <= wmem_bank0[pre_wmem_addr];
        wmem_rdata[1] <= wmem_bank1[pre_wmem_addr];
        wmem_rdata[2] <= wmem_bank2[pre_wmem_addr];
        wmem_rdata[3] <= wmem_bank3[pre_wmem_addr];
        wmem_rdata[4] <= wmem_bank4[pre_wmem_addr];
        wmem_rdata[5] <= wmem_bank5[pre_wmem_addr];
        wmem_rdata[6] <= wmem_bank6[pre_wmem_addr];
        wmem_rdata[7] <= wmem_bank7[pre_wmem_addr];
    end

    always @(posedge clk) begin
        smem_rdata[0] <= smem_bank0[pre_smem_addr];
        smem_rdata[1] <= smem_bank1[pre_smem_addr];
        smem_rdata[2] <= smem_bank2[pre_smem_addr];
        smem_rdata[3] <= smem_bank3[pre_smem_addr];
        smem_rdata[4] <= smem_bank4[pre_smem_addr];
        smem_rdata[5] <= smem_bank5[pre_smem_addr];
        smem_rdata[6] <= smem_bank6[pre_smem_addr];
        smem_rdata[7] <= smem_bank7[pre_smem_addr];
    end

    always @(posedge clk) begin
        act_rdata <= act_bram[pre_act_addr];
    end

    // ======================================================================
    // Q8 dequant: q8 (INT8) x scale (UQ8.8) >> 8 -> INT16
    // ======================================================================
    function automatic signed [15:0] dequant_q8;
        input signed [7:0]  q8;
        input [15:0] sc;
        begin
            dequant_q8 = $signed(q8 * $signed({8'b0, sc})) >>> 8;
        end
    endfunction

endmodule
