`timescale 1ns / 1ps

module tb_simple2;

    reg         clk = 0;
    reg         rst_n = 0;
    reg         start = 0;
    wire        done;
    wire        busy;
    reg         wt_we = 0;
    reg [11:0]  wt_addr = 0;
    reg [7:0]   wt_din = 0;
    reg         sc_we = 0;
    reg [6:0]   sc_addr = 0;
    reg [15:0]  sc_din = 0;
    reg         act_we = 0;
    reg [5:0]   act_addr = 0;
    reg [15:0]  act_din = 0;
    reg [5:0]   res_addr = 0;
    wire [47:0] res_dout;

    matmul_q8_core u_core (
        .clk       (clk),
        .rst_n     (rst_n),
        .start     (start),
        .op_vecmul (1'b1),
        .done      (done),
        .busy      (busy),
        .wt_we     (wt_we),
        .wt_addr   (wt_addr),
        .wt_din    (wt_din),
        .sc_we     (sc_we),
        .sc_addr   (sc_addr),
        .sc_din    (sc_din),
        .act_we    (act_we),
        .act_addr  (act_addr),
        .act_din   (act_din),
        .res_addr  (res_addr),
        .res_dout  (res_dout)
    );

    initial forever #5 clk = ~clk;

    initial begin
        $display("Time 0, starting test");

        repeat (4) @(posedge clk);
        $display("After 4 posedges");
        rst_n = 1;
        repeat (2) @(posedge clk);
        $display("After reset");

        @(negedge clk);
        wt_we = 1;
        wt_addr = 0;
        wt_din = 8'd1;
        @(posedge clk);
        wt_we = 0;
        $display("After weight write");

        @(negedge clk);
        sc_we = 1;
        sc_addr = 0;
        sc_din = 16'd256;
        @(posedge clk);
        sc_we = 0;
        $display("After scale write");

        @(negedge clk);
        act_we = 1;
        act_addr = 0;
        act_din = 16'd1;
        @(posedge clk);
        act_we = 0;
        $display("After act write");

        @(negedge clk);
        start = 1;
        @(posedge clk);
        $display("After start, busy=%b", busy);
        start = 0;

        repeat (100) @(posedge clk);
        $display("After 100 cycles, busy=%b done=%b", busy, done);

        $finish;
    end

endmodule
