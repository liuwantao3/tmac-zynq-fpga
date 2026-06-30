`timescale 1ns / 1ps

module matmul_q8_core (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire         op_vecmul,
    output reg          done,
    output reg          busy,
    input  wire         wt_we,
    input  wire [11:0]  wt_addr,
    input  wire  [7:0]  wt_din,
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
    // Weight storage: 512 x 64-bit (flattened, BRAM-friendly)
    //   addr = bank * 64 + col = {bank[2:0], col[5:0]}
    //   byte lane = row % 8 = wt_addr[8:6]
    // ======================================================================
    (* ram_style = "block" *) reg [63:0] wmem [0:511];

    always @(posedge clk) begin
        if (wt_we) begin
            case (wt_addr[8:6])
                3'd0: wmem[{wt_addr[11:9], wt_addr[5:0]}][7:0]   <= wt_din;
                3'd1: wmem[{wt_addr[11:9], wt_addr[5:0]}][15:8]  <= wt_din;
                3'd2: wmem[{wt_addr[11:9], wt_addr[5:0]}][23:16] <= wt_din;
                3'd3: wmem[{wt_addr[11:9], wt_addr[5:0]}][31:24] <= wt_din;
                3'd4: wmem[{wt_addr[11:9], wt_addr[5:0]}][39:32] <= wt_din;
                3'd5: wmem[{wt_addr[11:9], wt_addr[5:0]}][47:40] <= wt_din;
                3'd6: wmem[{wt_addr[11:9], wt_addr[5:0]}][55:48] <= wt_din;
                3'd7: wmem[{wt_addr[11:9], wt_addr[5:0]}][63:56] <= wt_din;
            endcase
        end
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
    // Single-bank accumulators: 64 x 48-bit
    // ======================================================================
    reg signed [47:0] acc [0:63];

    assign res_dout = acc[res_addr];

`ifdef __ICARUS__
    integer _init_i;
    initial begin
        for (_init_i = 0; _init_i < 512; _init_i = _init_i + 1) wmem[_init_i] = 0;
        for (_init_i = 0; _init_i < 64; _init_i = _init_i + 1) acc[_init_i] = 0;
    end
`endif

    // ======================================================================
    // Pipeline registers (4-stage)
    //   Stage 0: address BRAM, read act + scales
    //   Stage 1a: dequant (wmem_rdata * sc >> 8)
    //   Stage 1b: multiply with activation (p1_act * dq_deq)
    //   Stage 2: accumulate
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
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    busy <= 0;
                    p1_valid <= 0;
                    dq_valid <= 0;
                    p2_valid <= 0;
                    if (start) begin
                        for (wi_i = 0; wi_i < 64; wi_i = wi_i + 1)
                            acc[wi_i] <= 0;
                        busy <= 1;
                        k <= 0;
                        g <= 0;
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    // Stage 2: accumulate
                    if (p2_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            acc[p2_row_base + wi_i] <= acc[p2_row_base + wi_i] + p2_partial[wi_i];
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
                    // Stage 2: accumulate
                    if (p2_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            acc[p2_row_base + wi_i] <= acc[p2_row_base + wi_i] + p2_partial[wi_i];
                    end

                    // Stage 1b: multiply by activation (from dq)
                    if (dq_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            p2_partial[wi_i] <= $signed(dq_act) * dq_deq[wi_i];
                        p2_row_base <= dq_row_base;
                        p2_valid <= 1;
                    end else begin
                        p2_valid <= 0;
                    end

                    // Stage 1a: dequant (from p1)
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
                    // Stage 2: accumulate
                    if (p2_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            acc[p2_row_base + wi_i] <= acc[p2_row_base + wi_i] + p2_partial[wi_i];
                    end

                    // Stage 1b: multiply (from dq, no new dq)
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
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            acc[p2_row_base + wi_i] <= acc[p2_row_base + wi_i] + p2_partial[wi_i];
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
        reg signed [23:0] prod;
        begin
            prod = q8 * $signed({8'b0, sc});
            if (prod > 24'sd8388607)
                dequant_q8 = 16'sd32767;
            else if (prod < -24'sd8388608)
                dequant_q8 = -16'sd32768;
            else
                dequant_q8 = prod[23:8];
        end
    endfunction

endmodule
