`timescale 1ns / 1ps

module matmul_q4k_64x896_core (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire         op_vecmul,
    output reg          done,
    output reg          busy,
    input  wire         wt_we,
    input  wire [12:0]  wt_addr,
    input  wire  [7:0]   wt_din,
    input  wire         sc_we,
    input  wire [9:0]   sc_addr,
    input  wire [15:0]  sc_din,
    input  wire         act_we,
    input  wire [5:0]   act_addr,
    input  wire [15:0]  act_din,
    input  wire [5:0]   res_addr,
    output wire [47:0]  res_dout,
    input  wire         mode_block_load,
    output reg          decode_busy
);

    localparam IDLE   = 3'd0;
    localparam DECODE = 3'd1;
    localparam DW0    = 3'd2;
    localparam DW1    = 3'd3;
    localparam COMPUTE= 3'd4;
    localparam DRAIN  = 3'd5;
    localparam DRAIN2 = 3'd6;

    localparam TILE_ROWS = 64;
    localparam TILE_COLS = 896;
    localparam NUM_BLOCKS = 224;
    localparam NUM_ROW_GROUPS = 8;
    localparam COLS_PER_GROUP = 64;

    reg [2:0] state;
    reg [9:0] k;
    reg [2:0] g;
    integer wi_i;

    reg [14:0] blk_load_ptr;
    reg [7:0] blk_idx;
    reg [7:0] blk_w_idx;

    always @(posedge clk) begin
        if (!rst_n) begin
            blk_load_ptr <= 0;
        end else if (wt_we && blk_load_ptr < 32256) begin
            blk_load_ptr <= blk_load_ptr + 1;
        end
    end

    reg [7:0] block_buf [0:32255];
    always @(posedge clk) begin
        if (wt_we && blk_load_ptr < 32256)
            block_buf[blk_load_ptr[14:0]] <= wt_din;
    end

    reg [15:0] act_reg [0:895];
    always @(posedge clk) begin
        if (act_we)
            act_reg[act_addr] <= act_din;
    end

    reg [15:0] row_scale [0:63];
    always @(posedge clk) begin
        if (sc_we) row_scale[sc_addr] <= sc_din;
    end

    (* ram_style = "block" *) reg [63:0] wmem_lo [0:511];
    (* ram_style = "block" *) reg [63:0] wmem_hi [0:511];

    reg [127:0] wmem_rdata;
    always @(posedge clk) begin
        wmem_rdata <= {wmem_hi[{(k[9:6]), g[2:0]}], wmem_lo[{(k[9:6]), g[2:0]}]};
    end

    reg signed [47:0] acc [0:63];
    assign res_dout = acc[res_addr];

    reg [2:0] p1_g;
    reg [9:0] p1_k;
    reg [5:0] p1_row_base;
    reg signed [15:0] p1_act;
    reg        p1_valid;

    reg signed [31:0] p2_partial [0:7];
    reg [5:0]  p2_row_base;
    reg        p2_valid;

    reg [3:0] dec_byte_lane;
    reg [8:0] dec_entry;
    reg [15:0] dec_computed_val;
    reg dec_done;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;  k <= 0; g <= 0; done <= 0; busy <= 0;
            p1_valid <= 0; p2_valid <= 0;
            blk_load_ptr <= 0; blk_idx <= 0; blk_w_idx <= 0;
            decode_busy <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0; busy <= 0; decode_busy <= 0;
                    p1_valid <= 0; p2_valid <= 0;
                    blk_load_ptr <= 0;
                    if (start && op_vecmul) begin
                        for (wi_i = 0; wi_i < 64; wi_i = wi_i + 1) acc[wi_i] <= 0;
                        busy <= 1;
                        blk_idx <= 0; blk_w_idx <= 0;
                        decode_busy <= 1;
                        state <= DECODE;
                    end
                end

                DECODE: begin
                    if (blk_w_idx == 255) begin
                        blk_w_idx <= 0;
                        if (blk_idx == NUM_BLOCKS - 1) begin
                            decode_busy <= 0;
                            k <= 0; g <= 0;
                            state <= COMPUTE;
                        end else begin
                            blk_idx <= blk_idx + 1;
                        end
                    end else begin
                        if (blk_idx == 0 && blk_w_idx < 3) begin
                            $display("DECODE->DW0: bi=%d wi=%d", blk_idx, blk_w_idx);
                        end
                        dec_computed_val <= decode_one_64x896(blk_idx, blk_w_idx);
                        dec_byte_lane <= dec_byte_lane_val_64x896(blk_idx, blk_w_idx);
                        dec_entry <= dec_entry_val_64x896(blk_idx, blk_w_idx);
                        state <= DW0;
                    end
                end

                DW0: begin
                    case (dec_byte_lane)
                        4'd0: wmem_lo[dec_entry][7:0]   <= dec_computed_val[7:0];
                        4'd1: wmem_lo[dec_entry][15:8]  <= dec_computed_val[7:0];
                        4'd2: wmem_lo[dec_entry][23:16] <= dec_computed_val[7:0];
                        4'd3: wmem_lo[dec_entry][31:24] <= dec_computed_val[7:0];
                        4'd4: wmem_lo[dec_entry][39:32] <= dec_computed_val[7:0];
                        4'd5: wmem_lo[dec_entry][47:40] <= dec_computed_val[7:0];
                        4'd6: wmem_lo[dec_entry][55:48] <= dec_computed_val[7:0];
                        4'd7: wmem_lo[dec_entry][63:56] <= dec_computed_val[7:0];
                        4'd8: wmem_hi[dec_entry][7:0]   <= dec_computed_val[7:0];
                        4'd9: wmem_hi[dec_entry][15:8]  <= dec_computed_val[7:0];
                        4'd10: wmem_hi[dec_entry][23:16] <= dec_computed_val[7:0];
                        4'd11: wmem_hi[dec_entry][31:24] <= dec_computed_val[7:0];
                        4'd12: wmem_hi[dec_entry][39:32] <= dec_computed_val[7:0];
                        4'd13: wmem_hi[dec_entry][47:40] <= dec_computed_val[7:0];
                        4'd14: wmem_hi[dec_entry][55:48] <= dec_computed_val[7:0];
                        4'd15: wmem_hi[dec_entry][63:56] <= dec_computed_val[7:0];
                    endcase
                    state <= DW1;
                end

                DW1: begin
                    case (dec_byte_lane + 4'd1)
                        4'd0: wmem_lo[dec_entry][7:0]   <= dec_computed_val[15:8];
                        4'd1: wmem_lo[dec_entry][15:8]  <= dec_computed_val[15:8];
                        4'd2: wmem_lo[dec_entry][23:16] <= dec_computed_val[15:8];
                        4'd3: wmem_lo[dec_entry][31:24] <= dec_computed_val[15:8];
                        4'd4: wmem_lo[dec_entry][39:32] <= dec_computed_val[15:8];
                        4'd5: wmem_lo[dec_entry][47:40] <= dec_computed_val[15:8];
                        4'd6: wmem_lo[dec_entry][55:48] <= dec_computed_val[15:8];
                        4'd7: wmem_lo[dec_entry][63:56] <= dec_computed_val[15:8];
                        4'd8: wmem_hi[dec_entry][7:0]   <= dec_computed_val[15:8];
                        4'd9: wmem_hi[dec_entry][15:8]  <= dec_computed_val[15:8];
                        4'd10: wmem_hi[dec_entry][23:16] <= dec_computed_val[15:8];
                        4'd11: wmem_hi[dec_entry][31:24] <= dec_computed_val[15:8];
                        4'd12: wmem_hi[dec_entry][39:32] <= dec_computed_val[15:8];
                        4'd13: wmem_hi[dec_entry][47:40] <= dec_computed_val[15:8];
                        4'd14: wmem_hi[dec_entry][55:48] <= dec_computed_val[15:8];
                        4'd15: wmem_hi[dec_entry][63:56] <= dec_computed_val[15:8];
                    endcase
                    blk_w_idx <= blk_w_idx + 1;
                    state <= DECODE;
                end

                COMPUTE: begin
                    if (k == 0 && g == 0 && state != DRAIN) begin
                        $display("COMPUTE: k=%d g=%d p1_valid=%d p2_valid=%d act=%d at %t", k, g, p1_valid, p2_valid, p1_act, $time);
                    end
                    if (p2_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            acc[p2_row_base + wi_i] <= acc[p2_row_base + wi_i] + p2_partial[wi_i];
                    end
                    if (p1_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            p2_partial[wi_i] <= p1_act * $signed(wmem_rdata[wi_i*16 +: 16]);
                        p2_row_base <= p1_row_base; p2_valid <= 1;
                    end else begin
                        p2_valid <= 0;
                    end
                    p1_g <= g; p1_k <= k;
                    p1_row_base <= {g, 3'b0};
                    p1_act <= act_reg[k]; p1_valid <= 1;
                    if (g == NUM_ROW_GROUPS - 1) begin
                        g <= 0;
                        if (k == TILE_COLS - 1) state <= DRAIN;
                        else k <= k + 1;
                    end else begin
                        g <= g + 1;
                    end
                end

                DRAIN: begin
                    if (p2_valid)
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            acc[p2_row_base + wi_i] <= acc[p2_row_base + wi_i] + p2_partial[wi_i];
                    if (p1_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            p2_partial[wi_i] <= p1_act * $signed(wmem_rdata[wi_i*16 +: 16]);
                        p2_row_base <= p1_row_base; p2_valid <= 1;
                    end else begin
                        p2_valid <= 0;
                    end
                    p1_valid <= 0; state <= DRAIN2;
                end

                DRAIN2: begin
                    if (p2_valid)
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            acc[p2_row_base + wi_i] <= acc[p2_row_base + wi_i] + p2_partial[wi_i];
                    p2_valid <= 0; 
                    $display("FSM: entering IDLE (done)");
                    busy <= 0; done <= 1; state <= IDLE;
                end
            endcase
        end
    end

    function automatic [3:0] dec_byte_lane_val_64x896;
        input [7:0] bi;
        input [7:0] w_idx;
        reg [9:0] flat_idx;
        reg [5:0] tile_row;
        begin
            flat_idx = bi * 256 + w_idx;
            tile_row = flat_idx / TILE_COLS;
            dec_byte_lane_val_64x896 = (tile_row % 8) * 2;
        end
    endfunction

    function automatic [8:0] dec_entry_val_64x896;
        input [7:0] bi;
        input [7:0] w_idx;
        reg [9:0] flat_idx;
        reg [9:0] tile_col;
        reg [5:0] tile_row;
        begin
            flat_idx = bi * 256 + w_idx;
            tile_col = flat_idx % TILE_COLS;
            tile_row = flat_idx / TILE_COLS;
            dec_entry_val_64x896 = (tile_col / COLS_PER_GROUP) * 8 + (tile_row % 8);
        end
    endfunction

    function automatic signed [15:0] decode_one_64x896;
        input [7:0] bi;
        input [7:0] wi;
        reg [9:0] flat_idx;
        reg [9:0] tile_col;
        reg [5:0] tile_row;
        reg [3:0] sub;
        reg [4:0] j;
        reg [15:0] f16_d, f16_dmin;
        reg [4:0] exp;
        reg [9:0] mant;
        reg [7:0] qs_b;
        reg [3:0] q4;
        reg [5:0] sc_used, m_used;
        reg signed [31:0] d_fp, dmin_fp, a, b;
        reg signed [47:0] val;
        reg signed [47:0] val_norm;
        integer base;
        begin
            flat_idx = bi * 256 + wi;
            tile_col = flat_idx % TILE_COLS;
            tile_row = flat_idx / TILE_COLS;

            sub = wi / 32;
            j = wi % 32;
            base = bi * 144;

            f16_d = {block_buf[base+1], block_buf[base+0]};
            exp = f16_d[14:10]; mant = f16_d[9:0];
            if (exp == 0 || exp == 31) d_fp = 0;
            else if (exp >= 17) d_fp = (32'd1024 + mant) << (exp - 17);
            else d_fp = (32'd1024 + mant) >> (17 - exp);

            f16_dmin = {block_buf[base+3], block_buf[base+2]};
            exp = f16_dmin[14:10]; mant = f16_dmin[9:0];
            if (exp == 0 || exp == 31) dmin_fp = 0;
            else if (exp >= 17) dmin_fp = (32'd1024 + mant) << (exp - 17);
            else dmin_fp = (32'd1024 + mant) >> (17 - exp);

            if (sub < 4) begin
                sc_used = block_buf[base + 4 + sub][5:0];
                m_used  = block_buf[base + 8 + sub][5:0];
            end else begin
                sc_used = {block_buf[base + 4 + sub - 4][7:6], block_buf[base + 8 + sub][3:0]};
                m_used  = {block_buf[base + 8 + sub - 4][7:6], block_buf[base + 8 + sub][7:4]};
            end

            qs_b = block_buf[base + 16 + (sub/2) * 32 + j];
            q4 = (sub % 2 == 0) ? qs_b[3:0] : qs_b[7:4];

            a = d_fp * sc_used;
            b = dmin_fp * m_used;
            val = (a * q4 - b) >>> 8;

            val_norm = val * $signed(row_scale[tile_row]);
            val_norm = val_norm >>> 8;

            if (val_norm > 32767) decode_one_64x896 = 32767;
            else if (val_norm < -32768) decode_one_64x896 = -32768;
            else decode_one_64x896 = val_norm[15:0];
        end
    endfunction

endmodule