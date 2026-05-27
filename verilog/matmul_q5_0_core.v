`timescale 1ns / 1ps

module matmul_q5_0_core (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    output reg          done,
    output reg          busy,
    input  wire         wt_we,
    input  wire [12:0]  wt_addr,
    input  wire  [7:0]  wt_din,
    input  wire         sc_we,
    input  wire [2:0]   sc_addr,
    input  wire [15:0]  sc_din,
    input  wire         act_we,
    input  wire [9:0]   act_addr,
    input  wire [15:0]  act_din,
    input  wire [2:0]   res_addr,
    output wire [47:0]  res_dout,
    input  wire         dbg_tile_start,
    output reg  [31:0]  dbg_tile_cycles,
    output reg  [7:0]   dbg_tile_id,
    input  wire         dbg_verbose
);
    reg [31:0] cycle_cnt;
    reg [31:0] tile_start_cycle;
    reg [31:0] tile_end_cycle;
    reg [7:0]  tile_counter;

    localparam IDLE  = 2'd0;
    localparam LD    = 2'd1;
    localparam DEC   = 2'd2;
    localparam DRAIN = 2'd3;

    localparam NUM_BLOCKS = 224;
    localparam TILE_COLS  = 896;
    localparam TOTAL_ELTS = NUM_BLOCKS * 32;

    localparam BLOCK_BYTES = 22;

    reg [1:0] state;
    reg [12:0] ei;

    reg [7:0] block_buf [0:8191];
    always @(posedge clk) begin
        if (wt_we) block_buf[wt_addr] <= wt_din;
    end

    reg [15:0] row_scale [0:7];
    always @(posedge clk) begin
        if (sc_we) row_scale[sc_addr] <= sc_din;
    end

    reg [15:0] act_reg [0:895];
    always @(posedge clk) begin
        if (act_we) act_reg[act_addr] <= act_din;
    end

    reg signed [47:0] acc [0:7];
    assign res_dout = acc[res_addr];

    reg [15:0] r_f16_d;
    reg [31:0] r_qh;
    reg [7:0]  r_qs_byte;
    reg [4:0]  r_wi;
    reg [2:0]  r_row;
    reg [9:0]  r_col;

    reg signed [31:0] r_prod;
    reg [2:0] r_prow;

    // NOTE: All reg declarations must be at module level, NOT inside
    // named begin..end blocks in case arms. Vivado (and most real
    // tools) reject reg declarations inside case block bodies, even
    // inside an explicit begin..end. Only some simulators (old
    // iVerilog) accept it. This rule applies to ALL Verilog files in
    // this project — keep regs at the top level.
    reg [4:0]  exp_d;
    reg [9:0]  mant_d;
    reg signed [31:0] d_fp;
    reg signed [15:0] q5;
    reg signed [47:0] val_norm;
    reg signed [15:0] dec;
    reg qh_bit;
    reg [3:0] ql_nibble;
    reg [7:0] qs_byte;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; ei <= 0;
            done <= 0; busy <= 0;
            acc[0] <= 0; acc[1] <= 0; acc[2] <= 0; acc[3] <= 0;
            acc[4] <= 0; acc[5] <= 0; acc[6] <= 0; acc[7] <= 0;
            r_prod <= 0; r_prow <= 0;
            cycle_cnt <= 0;
            tile_counter <= 0;
            tile_start_cycle <= 0;
            tile_end_cycle <= 0;
            dbg_tile_cycles <= 0;
            dbg_tile_id <= 0;
        end else begin
            cycle_cnt <= cycle_cnt + 1;
            case (state)
                IDLE: begin
                    done <= 0; busy <= 0;
                    if (start) begin
                        acc[0] <= 0; acc[1] <= 0; acc[2] <= 0; acc[3] <= 0;
                        acc[4] <= 0; acc[5] <= 0; acc[6] <= 0; acc[7] <= 0;
                        ei <= 0; busy <= 1;
                        tile_start_cycle <= cycle_cnt;
                        tile_counter <= tile_counter + 1;
                        state <= LD;
                    end
                end

                LD: begin
                    r_f16_d <= {block_buf[(ei/32)*BLOCK_BYTES + 1], block_buf[(ei/32)*BLOCK_BYTES]};
                    r_qh <= {block_buf[(ei/32)*BLOCK_BYTES + 5],
                             block_buf[(ei/32)*BLOCK_BYTES + 4],
                             block_buf[(ei/32)*BLOCK_BYTES + 3],
                             block_buf[(ei/32)*BLOCK_BYTES + 2]};
                    r_wi <= ei % 32;
                    r_row <= (ei / TILE_COLS) % 8;
                    r_col <= ei % TILE_COLS;
                    state <= DEC;
                end

                DEC: begin
                    qs_byte = block_buf[(ei/32)*BLOCK_BYTES + 6 + r_wi[3:0]];

                    exp_d = r_f16_d[14:10];
                    mant_d = r_f16_d[9:0];
                    if (exp_d == 0 || exp_d == 31) d_fp = 0;
                    else if (exp_d >= 17) d_fp = (32'd1024 + mant_d) << (exp_d - 17);
                    else d_fp = ({1'b0, 32'd1024 + mant_d} + (1 << (17 - exp_d - 1))) >> (17 - exp_d);

                    qh_bit = r_qh[r_wi];

                    ql_nibble = r_wi[4] ? (qs_byte >> 4) : (qs_byte & 4'hF);

                    q5 = ((qh_bit << 4) | ql_nibble) - 16;

                    val_norm = d_fp * q5;
                    val_norm = val_norm * $signed({1'b0, row_scale[r_row]});
                    val_norm = val_norm >>> 8;

                    if (val_norm > 32767) dec = 32767;
                    else if (val_norm < -32768) dec = -32768;
                    else dec = val_norm[15:0];

                    r_prod <= $signed(dec) * $signed(act_reg[r_col]);

                    acc[r_row] <= acc[r_row] + $signed(dec) * $signed(act_reg[r_col]);

                    if (dbg_verbose && ei < 5) begin
                        $display("[CORE] ei=%0d row=%0d col=%0d d_fp=%0d q5=%0d val_norm=%0d dec=%0d act=%0d prod=%0d acc[%0d]=%0d",
                                 ei, r_row, r_col, d_fp, q5, val_norm, dec, act_reg[r_col], r_prod, r_row, acc[r_row]);
                    end

                    if (ei == TOTAL_ELTS - 1) begin
                        state <= DRAIN;
                    end else begin
                        ei <= ei + 1;
                        state <= LD;
                    end
                end

                DRAIN: begin
                    tile_end_cycle <= cycle_cnt - 1;
                    dbg_tile_cycles <= cycle_cnt - tile_start_cycle;
                    dbg_tile_id <= tile_counter;
                    $display("[CORE] Tile %02d DONE: start=%0d end=%0d cycles=%0d acc=[%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d]",
                             tile_counter, tile_start_cycle, cycle_cnt - 1,
                             dbg_tile_cycles,
                             acc[0], acc[1], acc[2], acc[3], acc[4], acc[5], acc[6], acc[7]);
                    done <= 1; busy <= 0;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule