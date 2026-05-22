// INT16 core smoke test: 64 active cols, all weights=1, verify row 0 = 64
`timescale 1ns / 1ps
module tb_int16_smoke;
    reg clk, rst_n, start; wire done, busy;
    reg wt_we; reg [12:0] wt_addr; reg [7:0] wt_din;
    reg act_we; reg [5:0] act_addr; reg [15:0] act_din;
    reg [5:0] res_addr; wire [47:0] res_dout;
    wire sc_we = 0; wire [6:0] sc_addr = 0; wire [15:0] sc_din = 0;

    matmul_int16_core uut(.clk(clk),.rst_n(rst_n),.start(start),.op_vecmul(1),.done(done),.busy(busy),
        .wt_we(wt_we),.wt_addr(wt_addr),.wt_din(wt_din),
        .sc_we(sc_we),.sc_addr(sc_addr),.sc_din(sc_din),
        .act_we(act_we),.act_addr(act_addr),.act_din(act_din),
        .res_addr(res_addr),.res_dout(res_dout));

    always #5 clk = ~clk;
    integer i, k;
    reg signed [47:0] expected;

    initial begin
        for (i = 0; i < 512; i++) begin
            uut.wmem_lo[i] = 64'd0;
            uut.wmem_hi[i] = 64'd0;
        end
    end

    initial begin
        $dumpfile("tb_int16_smoke.vcd"); $dumpvars(0, tb_int16_smoke);
        clk=0; rst_n=0; start=0; wt_we=0; act_we=0;
        repeat(4) @(posedge clk); rst_n=1;
        repeat(2) @(posedge clk);

        $display("Writing weights (64 cols × 1 INT16 = 1)...");
        for (k = 0; k < 64; k++) begin
            @(negedge clk); wt_we=1; wt_din=8'd1;  wt_addr <= {4'd0, k[5:0], 3'b000};
            @(negedge clk); wt_we=1; wt_din=8'd0;  wt_addr <= {4'd1, k[5:0], 3'b000};
        end
        @(negedge clk); wt_we=0;

        $display("Writing acts (all 64 × 1)...");
        for (i = 0; i < 64; i++) begin
            @(negedge clk); act_we=1; act_addr=i; act_din=16'd1;
        end
        @(negedge clk); act_we=0;

        $display("Starting compute...");
        @(negedge clk); start=1;
        @(negedge clk); start=0;
        wait(done);
        @(posedge clk);

        $display("Checking row 0 = 64...");
        expected = 64;
        res_addr = 0; @(negedge clk);
        if (res_dout !== expected) begin
            $display("FAIL: row 0 = %0d (expected %0d)", res_dout, expected);
            $finish;
        end
        $display("PASS: row 0 = %0d", res_dout);

        $display("Checking row 63 = 0 (no weight set for hi entries)...");
        expected = 0;
        res_addr = 63; @(negedge clk);
        if (res_dout !== expected) begin
            $display("FAIL: row 63 = %0d (expected %0d)", res_dout, expected);
            $finish;
        end
        $display("PASS: row 63 = %0d", res_dout);

        $display("\n=== INT16 SMOKE TEST PASSED ===");
        $finish;
    end
endmodule
