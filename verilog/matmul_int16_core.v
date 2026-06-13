`timescale 1ns / 1ps

module matmul_int16_core (
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
    input  wire [6:0]   sc_addr,
    input  wire [15:0]  sc_din,
    input  wire         act_we,
    input  wire [5:0]   act_addr,
    input  wire [15:0]  act_din,
    input  wire [5:0]   res_addr,
    output wire [47:0]  res_dout
);

    localparam IDLE   = 2'd0;
    localparam COMPUTE= 2'd1;
    localparam DRAIN  = 2'd2;
    localparam DRAIN2 = 2'd3;

    reg [1:0] state;
    reg [5:0] k;
    reg [2:0] g;

    (* ram_style = "distributed" *) reg [63:0] wmem_lo [0:511];
    (* ram_style = "distributed" *) reg [63:0] wmem_hi [0:511];

    reg [127:0] wmem_rdata;
    always @(posedge clk) begin
        wmem_rdata <= {wmem_hi[{k, g}], wmem_lo[{k, g}]};
    end

    always @(posedge clk) begin
        if (wt_we) begin
            case (wt_addr[12:9])
                4'd0: wmem_lo[wt_addr[8:0]][7:0]   <= wt_din;
                4'd1: wmem_lo[wt_addr[8:0]][15:8]  <= wt_din;
                4'd2: wmem_lo[wt_addr[8:0]][23:16] <= wt_din;
                4'd3: wmem_lo[wt_addr[8:0]][31:24] <= wt_din;
                4'd4: wmem_lo[wt_addr[8:0]][39:32] <= wt_din;
                4'd5: wmem_lo[wt_addr[8:0]][47:40] <= wt_din;
                4'd6: wmem_lo[wt_addr[8:0]][55:48] <= wt_din;
                4'd7: wmem_lo[wt_addr[8:0]][63:56] <= wt_din;
                4'd8: wmem_hi[wt_addr[8:0]][7:0]   <= wt_din;
                4'd9: wmem_hi[wt_addr[8:0]][15:8]  <= wt_din;
                4'd10: wmem_hi[wt_addr[8:0]][23:16] <= wt_din;
                4'd11: wmem_hi[wt_addr[8:0]][31:24] <= wt_din;
                4'd12: wmem_hi[wt_addr[8:0]][39:32] <= wt_din;
                4'd13: wmem_hi[wt_addr[8:0]][47:40] <= wt_din;
                4'd14: wmem_hi[wt_addr[8:0]][55:48] <= wt_din;
                4'd15: wmem_hi[wt_addr[8:0]][63:56] <= wt_din;
            endcase
        end
    end

    reg signed [15:0] act_reg [0:63];

    always @(posedge clk) begin
        if (act_we)
            act_reg[act_addr] <= act_din;
    end

    reg signed [47:0] acc [0:63];
    assign res_dout = acc[res_addr];

    reg [2:0]  p1_g;
    reg [5:0]  p1_k;
    reg [5:0]  p1_row_base;
    reg signed [15:0] p1_act;
    reg        p1_valid;

    reg signed [31:0] p2_partial [0:7];
    reg [5:0]  p2_row_base;
    reg        p2_valid;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            k <= 0; g <= 0; done <= 0; busy <= 0;
            p1_valid <= 0; p2_valid <= 0;
            p2_row_base <= 0;
            p1_row_base <= 0;
            p1_act <= 0;
            p2_partial[0] <= 0; p2_partial[1] <= 0; p2_partial[2] <= 0; p2_partial[3] <= 0;
            p2_partial[4] <= 0; p2_partial[5] <= 0; p2_partial[6] <= 0; p2_partial[7] <= 0;
            acc[0] <= 0; acc[1] <= 0; acc[2] <= 0; acc[3] <= 0;
            acc[4] <= 0; acc[5] <= 0; acc[6] <= 0; acc[7] <= 0;
            acc[8] <= 0; acc[9] <= 0; acc[10] <= 0; acc[11] <= 0;
            acc[12] <= 0; acc[13] <= 0; acc[14] <= 0; acc[15] <= 0;
            acc[16] <= 0; acc[17] <= 0; acc[18] <= 0; acc[19] <= 0;
            acc[20] <= 0; acc[21] <= 0; acc[22] <= 0; acc[23] <= 0;
            acc[24] <= 0; acc[25] <= 0; acc[26] <= 0; acc[27] <= 0;
            acc[28] <= 0; acc[29] <= 0; acc[30] <= 0; acc[31] <= 0;
            acc[32] <= 0; acc[33] <= 0; acc[34] <= 0; acc[35] <= 0;
            acc[36] <= 0; acc[37] <= 0; acc[38] <= 0; acc[39] <= 0;
            acc[40] <= 0; acc[41] <= 0; acc[42] <= 0; acc[43] <= 0;
            acc[44] <= 0; acc[45] <= 0; acc[46] <= 0; acc[47] <= 0;
            acc[48] <= 0; acc[49] <= 0; acc[50] <= 0; acc[51] <= 0;
            acc[52] <= 0; acc[53] <= 0; acc[54] <= 0; acc[55] <= 0;
            acc[56] <= 0; acc[57] <= 0; acc[58] <= 0; acc[59] <= 0;
            acc[60] <= 0; acc[61] <= 0; acc[62] <= 0; acc[63] <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0; busy <= 0;
                    p1_valid <= 0; p2_valid <= 0;
                    if (start) begin
                        acc[0] <= 0; acc[1] <= 0; acc[2] <= 0; acc[3] <= 0;
                        acc[4] <= 0; acc[5] <= 0; acc[6] <= 0; acc[7] <= 0;
                        acc[8] <= 0; acc[9] <= 0; acc[10] <= 0; acc[11] <= 0;
                        acc[12] <= 0; acc[13] <= 0; acc[14] <= 0; acc[15] <= 0;
                        acc[16] <= 0; acc[17] <= 0; acc[18] <= 0; acc[19] <= 0;
                        acc[20] <= 0; acc[21] <= 0; acc[22] <= 0; acc[23] <= 0;
                        acc[24] <= 0; acc[25] <= 0; acc[26] <= 0; acc[27] <= 0;
                        acc[28] <= 0; acc[29] <= 0; acc[30] <= 0; acc[31] <= 0;
                        acc[32] <= 0; acc[33] <= 0; acc[34] <= 0; acc[35] <= 0;
                        acc[36] <= 0; acc[37] <= 0; acc[38] <= 0; acc[39] <= 0;
                        acc[40] <= 0; acc[41] <= 0; acc[42] <= 0; acc[43] <= 0;
                        acc[44] <= 0; acc[45] <= 0; acc[46] <= 0; acc[47] <= 0;
                        acc[48] <= 0; acc[49] <= 0; acc[50] <= 0; acc[51] <= 0;
                        acc[52] <= 0; acc[53] <= 0; acc[54] <= 0; acc[55] <= 0;
                        acc[56] <= 0; acc[57] <= 0; acc[58] <= 0; acc[59] <= 0;
                        acc[60] <= 0; acc[61] <= 0; acc[62] <= 0; acc[63] <= 0;
                        busy <= 1; k <= 0; g <= 0;
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    if (p2_valid) begin
                        acc[p2_row_base+0] <= acc[p2_row_base+0] + p2_partial[0];
                        acc[p2_row_base+1] <= acc[p2_row_base+1] + p2_partial[1];
                        acc[p2_row_base+2] <= acc[p2_row_base+2] + p2_partial[2];
                        acc[p2_row_base+3] <= acc[p2_row_base+3] + p2_partial[3];
                        acc[p2_row_base+4] <= acc[p2_row_base+4] + p2_partial[4];
                        acc[p2_row_base+5] <= acc[p2_row_base+5] + p2_partial[5];
                        acc[p2_row_base+6] <= acc[p2_row_base+6] + p2_partial[6];
                        acc[p2_row_base+7] <= acc[p2_row_base+7] + p2_partial[7];
                    end
                    if (p1_valid) begin
                        p2_partial[0] <= $signed(p1_act) * $signed(wmem_rdata[0*16 +: 16]);
                        p2_partial[1] <= $signed(p1_act) * $signed(wmem_rdata[1*16 +: 16]);
                        p2_partial[2] <= $signed(p1_act) * $signed(wmem_rdata[2*16 +: 16]);
                        p2_partial[3] <= $signed(p1_act) * $signed(wmem_rdata[3*16 +: 16]);
                        p2_partial[4] <= $signed(p1_act) * $signed(wmem_rdata[4*16 +: 16]);
                        p2_partial[5] <= $signed(p1_act) * $signed(wmem_rdata[5*16 +: 16]);
                        p2_partial[6] <= $signed(p1_act) * $signed(wmem_rdata[6*16 +: 16]);
                        p2_partial[7] <= $signed(p1_act) * $signed(wmem_rdata[7*16 +: 16]);
                        p2_row_base <= p1_row_base; p2_valid <= 1;
                    end else begin
                        p2_valid <= 0;
                    end
                    p1_g <= g; p1_k <= k;
                    p1_row_base <= {g, 3'b0};
                    p1_act <= act_reg[k]; p1_valid <= 1;
                    if (g == 7) begin
                        g <= 0;
                        if (k == 63) state <= DRAIN; else k <= k + 1;
                    end else begin
                        g <= g + 1;
                    end
                end

                DRAIN: begin
                    if (p2_valid) begin
                        acc[p2_row_base+0] <= acc[p2_row_base+0] + p2_partial[0];
                        acc[p2_row_base+1] <= acc[p2_row_base+1] + p2_partial[1];
                        acc[p2_row_base+2] <= acc[p2_row_base+2] + p2_partial[2];
                        acc[p2_row_base+3] <= acc[p2_row_base+3] + p2_partial[3];
                        acc[p2_row_base+4] <= acc[p2_row_base+4] + p2_partial[4];
                        acc[p2_row_base+5] <= acc[p2_row_base+5] + p2_partial[5];
                        acc[p2_row_base+6] <= acc[p2_row_base+6] + p2_partial[6];
                        acc[p2_row_base+7] <= acc[p2_row_base+7] + p2_partial[7];
                    end
                    if (p1_valid) begin
                        p2_partial[0] <= $signed(p1_act) * $signed(wmem_rdata[0*16 +: 16]);
                        p2_partial[1] <= $signed(p1_act) * $signed(wmem_rdata[1*16 +: 16]);
                        p2_partial[2] <= $signed(p1_act) * $signed(wmem_rdata[2*16 +: 16]);
                        p2_partial[3] <= $signed(p1_act) * $signed(wmem_rdata[3*16 +: 16]);
                        p2_partial[4] <= $signed(p1_act) * $signed(wmem_rdata[4*16 +: 16]);
                        p2_partial[5] <= $signed(p1_act) * $signed(wmem_rdata[5*16 +: 16]);
                        p2_partial[6] <= $signed(p1_act) * $signed(wmem_rdata[6*16 +: 16]);
                        p2_partial[7] <= $signed(p1_act) * $signed(wmem_rdata[7*16 +: 16]);
                        p2_row_base <= p1_row_base; p2_valid <= 1;
                    end else begin
                        p2_valid <= 0;
                    end
                    p1_valid <= 0; state <= DRAIN2;
                end

                DRAIN2: begin
                    if (p2_valid) begin
                        acc[p2_row_base+0] <= acc[p2_row_base+0] + p2_partial[0];
                        acc[p2_row_base+1] <= acc[p2_row_base+1] + p2_partial[1];
                        acc[p2_row_base+2] <= acc[p2_row_base+2] + p2_partial[2];
                        acc[p2_row_base+3] <= acc[p2_row_base+3] + p2_partial[3];
                        acc[p2_row_base+4] <= acc[p2_row_base+4] + p2_partial[4];
                        acc[p2_row_base+5] <= acc[p2_row_base+5] + p2_partial[5];
                        acc[p2_row_base+6] <= acc[p2_row_base+6] + p2_partial[6];
                        acc[p2_row_base+7] <= acc[p2_row_base+7] + p2_partial[7];
                    end
                    p2_valid <= 0; busy <= 0; done <= 1; state <= IDLE;
                end
            endcase
        end
    end

endmodule
