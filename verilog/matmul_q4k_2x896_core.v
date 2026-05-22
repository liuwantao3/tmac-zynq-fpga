`timescale 1ns / 1ps

module matmul_q4k_2x896_core (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output reg          done,
    output reg          busy,
    input  wire         wt_we,
    input  wire [9:0]   wt_addr,
    input  wire [7:0]   wt_din,
    input  wire         sc_we,
    input  wire [0:0]   sc_addr,
    input  wire [15:0]  sc_din,
    input  wire         act_we,
    input  wire [9:0]   act_addr,
    input  wire [15:0]  act_din,
    input  wire [0:0]   res_addr,
    output wire [47:0]  res_dout
);

    localparam IDLE = 2'd0;
    localparam LD   = 2'd1;
    localparam DEC  = 2'd2;
    localparam DRAIN = 2'd3;

    localparam NUM_BLOCKS = 7;
    localparam TILE_COLS  = 896;
    localparam TOTAL_ELTS = NUM_BLOCKS * 256;

    reg [1:0] state;
    reg [11:0] ei;

    reg [7:0] block_buf [0:1007];
    always @(posedge clk) begin
        if (wt_we) block_buf[wt_addr] <= wt_din;
    end

    reg [15:0] row_scale [0:1];
    always @(posedge clk) begin
        if (sc_we) row_scale[sc_addr] <= sc_din;
    end

    reg [15:0] act_reg [0:895];
    always @(posedge clk) begin
        if (act_we) act_reg[act_addr] <= act_din;
    end

    reg signed [47:0] acc [0:1];
    assign res_dout = acc[res_addr];

    reg [15:0] r_f16_d;
    reg [15:0] r_f16_dmin;
    reg [7:0]  r_qs_b;
    reg [5:0]  r_sc;
    reg [5:0]  r_m;
    reg [3:0]  r_sub;
    reg [0:0]  r_row;
    reg [9:0]  r_col;

    reg signed [31:0] r_prod;
    reg [0:0] r_prow;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; ei <= 0;
            done <= 0; busy <= 0;
            acc[0] <= 0; acc[1] <= 0;
            r_prod <= 0; r_prow <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0; busy <= 0;
                    if (start) begin
                        acc[0] <= 0; acc[1] <= 0;
                        ei <= 0; busy <= 1;
                        state <= LD;
                    end
                end

                LD: begin
                    r_f16_d <= {block_buf[(ei/256)*144 + 1], block_buf[(ei/256)*144]};
                    r_f16_dmin <= {block_buf[(ei/256)*144 + 3], block_buf[(ei/256)*144 + 2]};
                    r_qs_b <= block_buf[(ei/256)*144 + 16 + ((ei%256)/64)*32 + (ei%256)%32];
                    r_sub <= (ei % 256) / 32;
                    r_row <= ei / TILE_COLS;
                    r_col <= ei % TILE_COLS;
                    if ((ei % 256) / 32 < 4) begin
                        r_sc <= block_buf[(ei/256)*144 + 4 + (ei%256)/32][5:0];
                        r_m  <= block_buf[(ei/256)*144 + 8 + (ei%256)/32][5:0];
                    end else begin
                        r_sc <= {block_buf[(ei/256)*144 + 4 + (ei%256)/32 - 4][7:6],
                                 block_buf[(ei/256)*144 + 8 + (ei%256)/32][3:0]};
                        r_m  <= {block_buf[(ei/256)*144 + 8 + (ei%256)/32 - 4][7:6],
                                 block_buf[(ei/256)*144 + 8 + (ei%256)/32][7:4]};
                    end
                    state <= DEC;
                end

                DEC: begin
                    reg [4:0] exp_d, exp_dmin;
                    reg [9:0] mant_d, mant_dmin;
                    reg signed [31:0] d_fp, dmin_fp, a, b;
                    reg signed [47:0] val, val_norm;
                    reg [3:0] q4;
                    reg signed [15:0] dec;

                    exp_d = r_f16_d[14:10]; mant_d = r_f16_d[9:0];
                    if (exp_d == 0 || exp_d == 31) d_fp = 0;
                    else if (exp_d >= 17) d_fp = (32'd1024 + mant_d) << (exp_d - 17);
                    else d_fp = (32'd1024 + mant_d) >> (17 - exp_d);

                    exp_dmin = r_f16_dmin[14:10]; mant_dmin = r_f16_dmin[9:0];
                    if (exp_dmin == 0 || exp_dmin == 31) dmin_fp = 0;
                    else if (exp_dmin >= 17) dmin_fp = (32'd1024 + mant_dmin) << (exp_dmin - 17);
                    else dmin_fp = (32'd1024 + mant_dmin) >> (17 - exp_dmin);

                    q4 = (r_sub % 2 == 0) ? r_qs_b[3:0] : r_qs_b[7:4];
                    a = d_fp * r_sc;
                    b = dmin_fp * r_m;
                    val = (a * q4 - b) >>> 8;
                    val_norm = val * $signed(row_scale[r_row]) >>> 8;

                    if (val_norm > 32767) dec = 32767;
                    else if (val_norm < -32768) dec = -32768;
                    else dec = val_norm[15:0];

                    r_prod <= $signed(dec) * $signed(act_reg[r_col]);
                    r_prow <= r_row;

                    acc[r_prow] <= acc[r_prow] + r_prod;

                    if (ei == TOTAL_ELTS - 1) begin
                        state <= DRAIN;
                    end else begin
                        ei <= ei + 1;
                        state <= LD;
                    end
                end
                DRAIN: begin
                    acc[r_prow] <= acc[r_prow] + r_prod;
                    done <= 1; busy <= 0;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule