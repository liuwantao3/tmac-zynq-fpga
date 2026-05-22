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
    integer wi_i;

    (* ram_style = "block" *) reg [63:0] wmem_lo [0:511];
    (* ram_style = "block" *) reg [63:0] wmem_hi [0:511];

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

    reg [127:0] wmem_rdata;
    always @(posedge clk) begin
        wmem_rdata <= {wmem_hi[{k, g}], wmem_lo[{k, g}]};
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
        end else begin
            case (state)
                IDLE: begin
                    done <= 0; busy <= 0;
                    p1_valid <= 0; p2_valid <= 0;
                    if (start) begin
                        for (wi_i = 0; wi_i < 64; wi_i = wi_i + 1) acc[wi_i] <= 0;
                        busy <= 1; k <= 0; g <= 0;
                        state <= COMPUTE;
                    end
                end

                COMPUTE: begin
                    if (p2_valid)
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            acc[p2_row_base + wi_i] <= acc[p2_row_base + wi_i] + p2_partial[wi_i];
                    if (p1_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            p2_partial[wi_i] <= $signed(p1_act) *
                                $signed(wmem_rdata[wi_i*16 +: 16]);
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
                    if (p2_valid)
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            acc[p2_row_base + wi_i] <= acc[p2_row_base + wi_i] + p2_partial[wi_i];
                    if (p1_valid) begin
                        for (wi_i = 0; wi_i < 8; wi_i = wi_i + 1)
                            p2_partial[wi_i] <= $signed(p1_act) *
                                $signed(wmem_rdata[wi_i*16 +: 16]);
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
                    p2_valid <= 0; busy <= 0; done <= 1; state <= IDLE;
                end
            endcase
        end
    end

endmodule
