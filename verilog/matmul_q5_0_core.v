`timescale 1ns / 1ps

module matmul_q5_0_core (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output reg          done,
    output reg          busy,
    input  wire         wt_we,
    input  wire [2:0]   wt_bank,
    input  wire [9:0]   wt_addr,
    input  wire  [7:0]  wt_din,
    input  wire         sc_we,
    input  wire [2:0]   sc_addr,
    input  wire [15:0]  sc_din,
    input  wire         act_we,
    input  wire [9:0]   act_addr,
    input  wire [15:0]  act_din,
    input  wire [0:0]   res_addr,
    output wire [47:0]  res_dout,
    input  wire [1:0]   core_id,
    input  wire         dbg_tile_start,
    output reg  [31:0]  dbg_tile_cycles,
    output reg  [7:0]   dbg_tile_id,
    input  wire         dbg_verbose
);

    // 4-cycle-pipeline FSM (R, D1, D2, A). Combinational BRAM addresses
    // from current ei fix the off-by-one bug; each pipline stage has 1
    // DSP multiply so timing is met (84 endpoints at -2.5ns worst-case).
    //
    // R:   BRAM read latency (combinational ei -> settle into BRAM addr)
    // D1:  d_fp * q5 -> pipe_dq
    // D2:  dq * scale >> 8 + clamp -> pipe_dec
    // A:   acc[row] += pipe_dec * act_r; ei++; back to R.
    localparam IDLE  = 3'd0;
    localparam R     = 3'd1;
    localparam D1    = 3'd2;
    localparam D2    = 3'd3;
    localparam DRAIN = 3'd4;
    localparam A     = 3'd5;
    localparam MAX_EI = 1791;

    reg [2:0] state;
    reg [10:0] ei;
    reg [31:0] cycle_cnt;
    reg [31:0] tile_start_cycle;
    reg [7:0]  tile_counter;

    // BRAMs
    (* ram_style = "block" *) reg [7:0] bank0 [0:1023];
    (* ram_style = "block" *) reg [7:0] bank1 [0:1023];
    (* ram_style = "block" *) reg [7:0] bank2 [0:1023];
    (* ram_style = "block" *) reg [7:0] bank3 [0:1023];
    (* ram_style = "block" *) reg [7:0] bank4 [0:1023];
    (* ram_style = "block" *) reg [7:0] bank5 [0:1023];
    (* ram_style = "block" *) reg [7:0] bank6 [0:1023];
    (* ram_style = "block" *) reg [15:0] act_mem [0:1023];

    reg [15:0] row_scale [0:7];
    reg signed [47:0] acc [0:1];
    assign res_dout = acc[res_addr];

    // BRAM read outputs
    reg [7:0]  hdr0_r, hdr1_r, hdr2_r, hdr3_r, hdr4_r, hdr5_r;
    reg [7:0]  qs_r;
    reg [15:0] act_r;

    // Pipeline regs
    reg signed [47:0] pipe_dq;
    reg signed [15:0] pipe_dec;

    // Combinational BRAM addresses (from current ei)
    wire        row_high_w  = (ei >= 11'd896);
    wire [10:0] ei_low_raw  = row_high_w ? (ei - 11'd896) : ei;
    wire [9:0]  ei_low      = ei_low_raw[9:0];
    wire [4:0]  blk_in_row  = ei_low[9:5];
    wire [9:0]  blk_idx     = (row_high_w ? 10'd28 : 10'd0) + {5'b0, blk_in_row};
    wire [4:0]  wi          = ei[4:0];
    wire [9:0]  qs_addr_w   = (blk_idx * 10'd16) + {5'b0, wi[4:1]};
    wire [9:0]  act_addr_w  = ei_low;

    // Combinational MAC helpers
    wire [15:0] f16_w       = {hdr1_r, hdr0_r};
    wire [31:0] qh_w        = {hdr5_r, hdr4_r, hdr3_r, hdr2_r};
    wire [4:0]  exp_d_w     = f16_w[14:10];
    wire [9:0]  mant_d_w    = f16_w[9:0];
    wire        qh_bit_w    = qh_w[wi];
    wire [3:0]  ql_nibble_w = wi[4] ? (qs_r >> 4) : (qs_r & 4'hF);
    wire signed [4:0]  q5_w = ((qh_bit_w << 4) | ql_nibble_w) - 16;
    wire signed [31:0] d_fp_w =
        (exp_d_w == 5'd0 || exp_d_w == 5'd31) ? 32'sd0 :
        (exp_d_w >= 5'd17) ? $signed((32'd1024 + mant_d_w) << (exp_d_w - 5'd17)) :
        $signed(({1'b0, 32'd1024 + mant_d_w} + (32'd1 << (5'd17 - exp_d_w - 5'd1)))
                >>> (5'd17 - exp_d_w));

    // D2 helpers
    wire signed [16:0] scale_w = $signed({1'b0, row_scale[row_high_w]});
    wire signed [47:0] val_norm_w = pipe_dq * scale_w;
    wire signed [47:0] val_shr_w  = val_norm_w >>> 8;
    wire signed [15:0] pipe_dec_next =
        (val_shr_w > 32'sd32767)  ? 16'sh7FFF :
        (val_shr_w < -32'sd32768) ? 16'sh8000 :
        val_shr_w[15:0];

    // A helpers
    wire signed [31:0] prod_w = pipe_dec * $signed(act_r);

    // BRAM block: synchronous read every cycle from combinational addresses
    always @(posedge clk) begin
        if (wt_we) begin
            case (wt_bank)
                3'd0: bank0[wt_addr] <= wt_din;
                3'd1: bank1[wt_addr] <= wt_din;
                3'd2: bank2[wt_addr] <= wt_din;
                3'd3: bank3[wt_addr] <= wt_din;
                3'd4: bank4[wt_addr] <= wt_din;
                3'd5: bank5[wt_addr] <= wt_din;
                3'd6: bank6[wt_addr] <= wt_din;
            endcase
        end
        if (act_we) act_mem[act_addr] <= act_din;
        if (sc_we)  row_scale[sc_addr] <= sc_din;
        hdr0_r <= bank0[blk_idx];
        hdr1_r <= bank1[blk_idx];
        hdr2_r <= bank2[blk_idx];
        hdr3_r <= bank3[blk_idx];
        hdr4_r <= bank4[blk_idx];
        hdr5_r <= bank5[blk_idx];
        qs_r   <= bank6[qs_addr_w];
        act_r  <= act_mem[act_addr_w];
    end

    // Main FSM
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state    <= IDLE; ei <= 11'd0;
            done     <= 1'b0; busy <= 1'b0;
            acc[0]   <= 48'd0; acc[1] <= 48'd0;
            cycle_cnt <= 32'd0; tile_counter <= 8'd0;
            tile_start_cycle <= 32'd0;
            dbg_tile_cycles <= 32'd0; dbg_tile_id <= 8'd0;
            pipe_dq <= 48'd0; pipe_dec <= 16'd0;
        end else begin
            cycle_cnt <= cycle_cnt + 1;
            done <= 1'b0;
            case (state)
                IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        acc[0] <= 48'd0; acc[1] <= 48'd0;
                        ei <= 11'd0; busy <= 1'b1;
                        tile_start_cycle <= cycle_cnt;
                        tile_counter <= tile_counter + 8'd1;
                        state <= R;
                    end
                end
                R: state <= D1;
                D1: begin
                    pipe_dq <= d_fp_w * q5_w;
                    state <= D2;
                end
                D2: begin
                    pipe_dec <= pipe_dec_next;
                    state <= A;
                end
                A: begin
                    acc[row_high_w] <= acc[row_high_w] + prod_w;
                    if (ei == MAX_EI) state <= DRAIN;
                    else begin ei <= ei + 1; state <= R; end
                end
                DRAIN: begin
                    dbg_tile_cycles <= cycle_cnt - tile_start_cycle;
                    dbg_tile_id <= tile_counter;
                    if (dbg_verbose) $display("[CORE%0d] Tile %02d DONE: cycles=%0d acc=[%0d,%0d]",
                        core_id, tile_counter, dbg_tile_cycles, acc[0], acc[1]);
                    done <= 1'b1; busy <= 1'b0; state <= IDLE;
                end
                default: state <= IDLE;
            endcase
        end
    end

`ifdef __ICARUS__
    initial begin
        for (integer k = 0; k < 1024; k = k + 1) begin
            bank0[k] = 8'd0; bank1[k] = 8'd0; bank2[k] = 8'd0; bank3[k] = 8'd0;
            bank4[k] = 8'd0; bank5[k] = 8'd0; bank6[k] = 8'd0;
            act_mem[k] = 16'd0;
        end
    end
`endif

endmodule