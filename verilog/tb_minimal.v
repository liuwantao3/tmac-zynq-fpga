`timescale 1ns / 1ps

module tb_minimal;

    reg         clk = 0;
    reg         rst_n = 0;
    reg         start = 0;
    wire        done, busy;
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

    integer i, r, c, refval;
    integer errors;

    task wr_w(input integer addr, input [7:0] val);
        begin @(negedge clk); wt_we <= 1; wt_addr <= addr; wt_din <= val; end
    endtask
    task wr_s(input integer addr, input [15:0] val);
        begin @(negedge clk); sc_we <= 1; sc_addr <= addr; sc_din <= val; end
    endtask
    task wr_a(input integer addr, input [15:0] val);
        begin @(negedge clk); act_we <= 1; act_addr <= addr; act_din <= val; end
    endtask
    task we_off;
        begin @(negedge clk); wt_we <= 0; sc_we <= 0; act_we <= 0; end
    endtask
    task do_start;
        begin @(negedge clk); start <= 1; @(negedge clk); start <= 0; end
    endtask

    initial begin
        $dumpfile("tb_minimal.vcd");
        $dumpvars(0, tb_minimal);

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        // Write exactly one row of weights: row 0, all 64 columns = 1
        for (c = 0; c < 64; c = c + 1) wr_w(c, 8'd1);  // row 0, col c = 1
        // Initialize row 1 as well (to avoid X propagation in row-group 0)
        for (c = 0; c < 64; c = c + 1) wr_w(64 + c, 8'd0);  // row 1 = 0
        // Row 2,3,4,5,6,7
        for (r = 2; r < 8; r = r + 1)
            for (c = 0; c < 64; c = c + 1)
                wr_w(r * 64 + c, 8'd0);

        // All scales = 256 (1.0)
        for (i = 0; i < 128; i = i + 1) wr_s(i, 16'd256);
        // Activation = 1
        for (i = 0; i < 64; i = i + 1) wr_a(i, 16'd1);

        we_off();
        $display("Data loaded");

        // Reference: sum over k of weight[row][k] * (scale[row][k/32]) >> 8 * act[k]
        // With all weights=1, scale=256, act=1: dequant = 1, result = 64
        $display("Expected result[0] = 64, result[1..7] = 0");

        do_start();

        for (i = 0; i < 600; i = i + 1) begin
            @(posedge clk);
            if (done) begin $display("DONE at cycle %0d", i+1); break; end
        end
        if (i >= 600) $display("TIMEOUT busy=%b done=%b", busy, done);

        // Debug: read internal acc directly
        #1;
        $display("DEBUG acc[0..7] = %0d %0d %0d %0d %0d %0d %0d %0d",
            u_core.acc[0], u_core.acc[1], u_core.acc[2], u_core.acc[3],
            u_core.acc[4], u_core.acc[5], u_core.acc[6], u_core.acc[7]);

        errors = 0;
        for (i = 0; i < 8; i = i + 1) begin
            @(negedge clk); res_addr <= i;
            @(posedge clk); #1;
            refval = (i == 0) ? 64 : 0;
            if (res_dout !== refval) begin
                $display("  MISMATCH[%0d]: HW=%0d REF=%0d", i, res_dout, refval);
                errors = errors + 1;
            end
        end
        if (errors == 0) $display("PASS"); else $display("FAIL (%0d)", errors);

        repeat (10) @(posedge clk);
        $finish;
    end

endmodule
