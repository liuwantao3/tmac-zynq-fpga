`timescale 1ns / 1ps

module matmul_q6_k_core (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire         op_vecmul,
    output reg          done,
    output reg          busy,
    input  wire         wt_we,
    input  wire [12:0]  wt_addr,
    input  wire  [7:0]  wt_din,
    input  wire         sc_we,
    input  wire [4:0]   sc_addr,
    input  wire [15:0]  sc_din,
    input  wire         act_we,
    input  wire [7:0]   act_addr,
    input  wire [15:0]  act_din,
    input  wire [4:0]   res_addr,
    output wire [47:0]  res_dout,
    input  wire         mode_block_load,
    output reg          decode_busy
);

    localparam IDLE     = 2'd0;
    localparam COMPUTE  = 2'd1;
    localparam DRAIN    = 2'd2;

    localparam TILE_ROWS = 32;
    localparam TILE_COLS = 256;
    localparam TOTAL_ELTS = TILE_ROWS * TILE_COLS;
    localparam BLOCK_BYTES = 210;
    localparam BLOCK_BUF_BYTES = TILE_ROWS * BLOCK_BYTES;

    reg [1:0] state;
    reg [12:0] ei;
    integer i;

    reg [7:0] block_buf [0:6719];
    reg [12:0] write_ptr;

    always @(posedge clk) begin
        if (wt_we && mode_block_load && write_ptr < BLOCK_BUF_BYTES)
            block_buf[write_ptr[12:0]] <= wt_din;
        if (wt_we && mode_block_load)
            write_ptr <= write_ptr + 1;
    end

    reg signed [15:0] act_reg [0:255];
    always @(posedge clk) begin
        if (act_we) act_reg[act_addr] <= act_din;
    end

    reg [15:0] row_scale [0:31];
    always @(posedge clk) begin
        if (sc_we) row_scale[sc_addr] <= sc_din;
    end

    reg signed [47:0] acc [0:31];
    assign res_dout = acc[res_addr];

    reg [4:0] p_row;
    reg [7:0] p_col;
    reg signed [15:0] p_val;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; ei <= 0;
            done <= 0; busy <= 0; decode_busy <= 0;
            write_ptr <= 0; p_val <= 0; p_row <= 0; p_col <= 0;
            for (i = 0; i < 32; i = i + 1) acc[i] <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0; busy <= 0; decode_busy <= 0;
                    if (start) begin
                        for (i = 0; i < 32; i = i + 1) acc[i] <= 0;
                        busy <= 1; decode_busy <= 1;
                        ei <= 0; state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    acc[p_row] <= acc[p_row] + $signed(p_val) * $signed(act_reg[p_col]);
                    p_row <= ei / TILE_COLS;
                    p_col <= ei % TILE_COLS;
                    p_val <= decode_one(ei / TILE_COLS, ei % TILE_COLS);
                    if (ei == TOTAL_ELTS - 1) begin
                        state <= DRAIN;
                    end else begin
                        ei <= ei + 1;
                    end
                end

                DRAIN: begin
                    acc[p_row] <= acc[p_row] + $signed(p_val) * $signed(act_reg[p_col]);
                    done <= 1; busy <= 0; decode_busy <= 0;
                    state <= IDLE;
                end
            endcase
        end
    end

    function automatic signed [15:0] decode_one;
        input [4:0] bi;
        input [7:0] wi;
        reg [1:0] half;
        reg [6:0] pos;
        reg [4:0] l;
        reg [1:0] sub;
        reg [7:0] ql_byte;
        reg [7:0] qh_byte;
        reg [3:0] ql_nibble;
        reg [1:0] qh_bits;
        reg signed [15:0] q6;
        reg signed [15:0] scale_val;
        reg [15:0] f16_d;
        reg [4:0] exp;
        reg [9:0] mant;
        reg signed [31:0] d_fp;
        reg signed [47:0] val_norm;
        integer base;
        begin
            half = wi / 128;
            pos = wi % 128;
            l = pos % 32;
            sub = pos / 32;

            base = bi * BLOCK_BYTES;

            f16_d = {block_buf[base+209], block_buf[base+208]};
            exp = f16_d[14:10]; mant = f16_d[9:0];
            if (exp == 0 || exp == 31) d_fp = 0;
            else if (exp >= 17) d_fp = (32'd1024 + mant) << (exp - 17);
            else d_fp = (32'd1024 + mant) >> (17 - exp);

            if (half == 0)
                ql_byte = block_buf[base + l + (sub[0] * 32)];
            else
                ql_byte = block_buf[base + 64 + l + (sub[0] * 32)];

            ql_nibble = (sub == 0 || sub == 1) ? ql_byte[3:0] : ql_byte[7:4];

            qh_byte = block_buf[base + 128 + (half * 32) + l];
            qh_bits = (qh_byte >> (sub * 2)) & 2'b11;

            q6 = ({qh_bits, ql_nibble}) - 16'd32;

            scale_val = $signed(block_buf[base + 192 + (l / 16) + sub * 2 + half * 8]);

            val_norm = d_fp * scale_val * q6;
            val_norm = val_norm * $signed(row_scale[bi]);
            val_norm = val_norm >>> 8;

            if (val_norm > 32767) decode_one = 32767;
            else if (val_norm < -32768) decode_one = -32768;
            else decode_one = val_norm[15:0];
        end
    endfunction

endmodule
