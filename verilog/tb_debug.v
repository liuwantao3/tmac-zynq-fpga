`timescale 1ns / 1ps
module tb_debug;
    reg clk, rst_n, start;
    wire done, busy;
    reg wt_we, sc_we, act_we;
    reg [8:0] wt_addr; reg [63:0] wt_din;
    reg [6:0] sc_addr; reg [15:0] sc_din;
    reg [5:0] act_addr, res_addr;
    reg [15:0] act_din;
    wire [47:0] res_dout;
    wire [2:0] dbg_state;
    wire [5:0] dbg_k;
    wire [2:0] dbg_g;

    matmul_q8_core uut (
        .clk(clk), .rst_n(rst_n), .start(start), .op_vecmul(1'b1),
        .done(done), .busy(busy),
        .wt_we(wt_we), .wt_addr(wt_addr), .wt_din(wt_din),
        .sc_we(sc_we), .sc_addr(sc_addr), .sc_din(sc_din),
        .act_we(act_we), .act_addr(act_addr), .act_din(act_din),
        .res_addr(res_addr), .res_dout(res_dout),
        .dbg_state(dbg_state), .dbg_k(dbg_k), .dbg_g(dbg_g)
    );

    always #5 clk = ~clk;

    integer i, k, g;
    initial begin
        $dumpfile("tb_debug.vcd"); $dumpvars(0, tb_debug);
        clk=0; rst_n=0; start=0;
        wt_we=0; sc_we=0; act_we=0;
        #15 rst_n=1;
        #10;

        for (g=0; g<8; g=g+1)
            for (k=0; k<64; k=k+1) begin
                @(negedge clk); wt_we<=1; wt_addr<=g*64+k; wt_din<=64'h0101010101010101;
            end
        @(negedge clk); wt_we<=0;

        for (i=0; i<128; i=i+1) begin
            @(negedge clk); sc_we<=1; sc_addr<=i; sc_din<=256;
        end
        @(negedge clk); sc_we<=0;

        for (i=0; i<64; i=i+1) begin
            @(negedge clk); act_we<=1; act_addr<=i; act_din<=1;
        end
        @(negedge clk); act_we<=0;

        @(negedge clk); start<=1;
        @(negedge clk); start<=0;

        for (i=0; i<2000 && !done; i=i+1) @(posedge clk);

        for (i=0; i<8; i=i+1) begin
            @(negedge clk); res_addr<=i;
            @(posedge clk); #1;
            $display("res[%0d]=%0d", i, res_dout);
        end
        #100 $finish;
    end
endmodule
