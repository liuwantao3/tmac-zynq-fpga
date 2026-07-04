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

    localparam IDLE     = 3'd0;
    localparam COMPUTE  = 3'd1;
    localparam DRAIN    = 3'd2;
    localparam DRAIN2   = 3'd3;
    localparam DRAIN3   = 3'd4;

    // FSM state
    reg [2:0] state;
    reg [5:0] k;
    reg [2:0] g;
    assign dbg_state = state;
    assign dbg_k = k;
    assign dbg_g = g;
    integer wi_i;

    // ======================================================================
    // Weight storage: 512 x 64-bit (BRAM)
    //   addr[8:0] = {bank[2:0], col[5:0]} = bank*64 + col
    //   word[63:0] = {row(bank*8+7), ..., row(bank*8+0)} for column
    //   DDR data is pre-transposed to column-major (bank-major) format.
    // ======================================================================
    (* ram_style = "block" *) reg [63:0] wmem [0:511];

    always @(posedge clk) begin
        if (wt_we) wmem[wt_addr] <= wt_din;
    end

    // BRAM synchronous read (1-cycle latency)
    reg [63:0] wmem_rdata;
    always @(posedge clk) begin
        wmem_rdata <= wmem[{g, k}];
    end

    // ======================================================================
    // Scale storage: 128 x 16-bit (distributed RAM)
    // ======================================================================
    reg [15:0] smem [0:127];

    always @(posedge clk) begin
        if (sc_we)
            smem[sc_addr] <= sc_din;
    end

    // ======================================================================
    // Activation registers: 64 x 16-bit
    // ======================================================================
    reg signed [15:0] act_reg [0:63];

    always @(posedge clk) begin
        if (act_we)
            act_reg[act_addr] <= act_din;
    end

    // ======================================================================
    // Accumulator banks: 8 banks × 8 entries × 48-bit BRAM
    //   addr[5:3] = bank (0-7), addr[2:0] = g (0-7)
    //   bank_i holds: acc[i], acc[8+i], ..., acc[56+i]
    //   Each bank has 1 read port + 1 write port (simple-dual-port BRAM).
    //   Pad depth to 512 for BRAM minimum depth requirement.
    // ======================================================================
    (* ram_style = "block" *) reg signed [47:0] acc_b0 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b1 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b2 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b3 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b4 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b5 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b6 [0:511];
    (* ram_style = "block" *) reg signed [47:0] acc_b7 [0:511];

    // BRAM read outputs (1 cycle latency, read address = g via pre_read_g)
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

    // Result read: combinatorial read from specific bank
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

    // BRAM pre-read address (alias g from dq pipeline, 1 cycle ahead of p2)
    reg [2:0] pre_read_g;
    reg [4:0] acc_clr_cnt;

`ifdef __ICARUS__
    integer _init_i;
    initial begin
        for (_init_i = 0; _init_i < 512; _init_i = _init_i + 1) wmem[_init_i] = 0;
        for (_init_i = 0; _init_i < 512; _init_i = _init_i + 1) begin
            acc_b0[_init_i] = 0; acc_b1[_init_i] = 0; acc_b2[_init_i] = 0; acc_b3[_init_i] = 0;
            acc_b4[_init_i] = 0; acc_b5[_init_i] = 0; acc_b6[_init_i] = 0; acc_b7[_init_i] = 0;
        end
    end
`endif

    // ======================================================================
    // Pipeline registers (5-stage)
    //   Stage 0: address BRAM, read act + scales
    //   Stage 1a: dequant (wmem_rdata * sc >> 8)
    //   Stage 1b: multiply with activation (p1_act * dq_deq)
    //   Stage 2a: BRAM acc read (1-cycle latency, address from dq_row_base)
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

    reg        p2a_valid;      // valid for stage 2a (acc BRAM read in progress)

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
                    // Sequential bulk-clear: 8 cycles to clear all 8 groups
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
                        state <= COMPUTE;
                    end else begin
                        acc_clr_cnt <= acc_clr_cnt + 1;
                    end
                end

                COMPUTE: begin
                    // Stage 2b: accumulate into BRAM (using acc_r from previous pre-read)
                    if (p2_valid) begin
                        acc_b0[p2_row_base[5:3]] <= acc_r[0] + p2_partial[0];
                        acc_b1[p2_row_base[5:3]] <= acc_r[1] + p2_partial[1];
                        acc_b2[p2_row_base[5:3]] <= acc_r[2] + p2_partial[2];
                        acc_b3[p2_row_base[5:3]] <= acc_r[3] + p2_partial[3];
                        acc_b4[p2_row_base[5:3]] <= acc_r[4] + p2_partial[4];
                        acc_b5[p2_row_base[5:3]] <= acc_r[5] + p2_partial[5];
                        acc_b6[p2_row_base[5:3]] <= acc_r[6] + p2_partial[6];
                        acc_b7[p2_row_base[5:3]] <= acc_r[7] + p2_partial[7];
                    end

                    // Stage 2a: pre-read acc BRAM for next accumulate
                    // (read starts from dq_row_base, arrives in acc_r 1 cycle later)
                    if (dq_valid) begin
                        pre_read_g <= dq_row_base[5:3];
                        p2a_valid <= 1;
                    end else begin
                        p2a_valid <= 0;
                    end

                    // Stage 1b: multiply by activation (from dq pipeline)
                    if (dq_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            p2_partial[wi_i] <= $signed(dq_act) * dq_deq[wi_i];
                        p2_row_base <= dq_row_base;
                        p2_valid <= 1;
                    end else begin
                        p2_valid <= 0;
                    end

                    // Stage 1a: dequant (from p1 pipeline)
                    if (p1_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            dq_deq[wi_i] <= dequant_q8(
                                $signed(wmem_rdata[wi_i*8 +: 8]), p1_sc[wi_i]);
                        dq_act <= p1_act;
                        dq_row_base <= p1_row_base;
                        dq_valid <= 1;
                    end else begin
                        dq_valid <= 0;
                    end

                    // Stage 0: address BRAM, read act + scales
                    p1_g <= g;
                    p1_k <= k;
                    p1_row_base <= {g, 3'b0};
                    p1_act <= act_reg[k];
                    for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                        p1_sc[wi_i] <= smem[(g * 8 + wi_i) * 2 + k[5]];
                    p1_valid <= 1;

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
                    // Stage 2b: final accumulate
                    if (p2_valid) begin
                        acc_b0[p2_row_base[5:3]] <= acc_r[0] + p2_partial[0];
                        acc_b1[p2_row_base[5:3]] <= acc_r[1] + p2_partial[1];
                        acc_b2[p2_row_base[5:3]] <= acc_r[2] + p2_partial[2];
                        acc_b3[p2_row_base[5:3]] <= acc_r[3] + p2_partial[3];
                        acc_b4[p2_row_base[5:3]] <= acc_r[4] + p2_partial[4];
                        acc_b5[p2_row_base[5:3]] <= acc_r[5] + p2_partial[5];
                        acc_b6[p2_row_base[5:3]] <= acc_r[6] + p2_partial[6];
                        acc_b7[p2_row_base[5:3]] <= acc_r[7] + p2_partial[7];
                    end

                    // Stage 2a: pre-read (no-op, pipeline draining)
                    if (dq_valid) begin
                        pre_read_g <= dq_row_base[5:3];
                    end

                    // Stage 1b: multiply
                    if (dq_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            p2_partial[wi_i] <= $signed(dq_act) * dq_deq[wi_i];
                        p2_row_base <= dq_row_base;
                        p2_valid <= 1;
                    end else begin
                        p2_valid <= 0;
                    end

                    // Stage 1a: dequant
                    if (p1_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            dq_deq[wi_i] <= dequant_q8(
                                $signed(wmem_rdata[wi_i*8 +: 8]), p1_sc[wi_i]);
                        dq_act <= p1_act;
                        dq_row_base <= p1_row_base;
                        dq_valid <= 1;
                    end else begin
                        dq_valid <= 0;
                    end

                    p1_valid <= 0;
                    state <= DRAIN2;
                end

                DRAIN2: begin
                    if (p2_valid) begin
                        acc_b0[p2_row_base[5:3]] <= acc_r[0] + p2_partial[0];
                        acc_b1[p2_row_base[5:3]] <= acc_r[1] + p2_partial[1];
                        acc_b2[p2_row_base[5:3]] <= acc_r[2] + p2_partial[2];
                        acc_b3[p2_row_base[5:3]] <= acc_r[3] + p2_partial[3];
                        acc_b4[p2_row_base[5:3]] <= acc_r[4] + p2_partial[4];
                        acc_b5[p2_row_base[5:3]] <= acc_r[5] + p2_partial[5];
                        acc_b6[p2_row_base[5:3]] <= acc_r[6] + p2_partial[6];
                        acc_b7[p2_row_base[5:3]] <= acc_r[7] + p2_partial[7];
                    end

                    if (dq_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            p2_partial[wi_i] <= $signed(dq_act) * dq_deq[wi_i];
                        p2_row_base <= dq_row_base;
                        p2_valid <= 1;
                    end else begin
                        p2_valid <= 0;
                    end

                    dq_valid <= 0;
                    state <= DRAIN3;
                end

                DRAIN3: begin
                    if (p2_valid) begin
                        acc_b0[p2_row_base[5:3]] <= acc_r[0] + p2_partial[0];
                        acc_b1[p2_row_base[5:3]] <= acc_r[1] + p2_partial[1];
                        acc_b2[p2_row_base[5:3]] <= acc_r[2] + p2_partial[2];
                        acc_b3[p2_row_base[5:3]] <= acc_r[3] + p2_partial[3];
                        acc_b4[p2_row_base[5:3]] <= acc_r[4] + p2_partial[4];
                        acc_b5[p2_row_base[5:3]] <= acc_r[5] + p2_partial[5];
                        acc_b6[p2_row_base[5:3]] <= acc_r[6] + p2_partial[6];
                        acc_b7[p2_row_base[5:3]] <= acc_r[7] + p2_partial[7];
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
    // Q8 dequant: q8 (INT8) x scale (UQ8.8) >> 8 -> INT16
    // ======================================================================
    function automatic signed [15:0] dequant_q8;
        input signed [7:0]  q8;
        input [15:0] sc;
        begin
            // prod = q8 * sc (sc is UQ8.8), result in S24.8 format
            // No saturation needed: max product = 127*65535 = 8322945 < 8388607,
            // min product = -128*65535 = -8388480 > -8388608.
            dequant_q8 = $signed(q8 * $signed({8'b0, sc})) >>> 8;
        end
    endfunction

endmodule
