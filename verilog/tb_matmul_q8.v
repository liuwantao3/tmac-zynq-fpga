`timescale 1ns / 1ps

module tb_matmul_q8;

    reg         clk;
    reg         rst_n;
    reg         start;
    wire        done;
    wire        busy;

    reg        wt_we;
    reg [8:0]  wt_addr;
    reg [63:0] wt_din;
    reg         sc_we;
    reg [6:0]   sc_addr;
    reg [15:0]  sc_din;
    reg         act_we;
    reg [5:0]   act_addr;
    reg [15:0]  act_din;
    reg [5:0]   res_addr;
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

    always #5 clk = ~clk;

    integer i, row, col, errors, total;

    task wr(input integer we, addr, input [63:0] din);
        begin
            @(negedge clk);
            if (we == 0) begin wt_we <= 1; wt_addr <= addr; wt_din <= din; end
            if (we == 1) begin sc_we <= 1; sc_addr <= addr; sc_din <= din[15:0]; end
            if (we == 2) begin act_we <= 1; act_addr <= addr; act_din <= din[15:0]; end
        end
    endtask

    task load_all_weights(input [7:0] val);
        reg [63:0] word;
        begin
            word = {8{val}};
            for (row = 0; row < 8; row = row + 1)
                for (col = 0; col < 64; col = col + 1)
                    wr(0, row * 64 + col, word);
            @(negedge clk); wt_we <= 0;
        end
    endtask

    task load_all_scales(input [15:0] val);
        begin
            for (i = 0; i < 128; i = i + 1)
                wr(1, i, val);
            @(negedge clk); sc_we <= 0;
        end
    endtask

    task load_all_acts(input [15:0] val);
        begin
            for (i = 0; i < 64; i = i + 1)
                wr(2, i, val);
            @(negedge clk); act_we <= 0;
        end
    endtask

    task start_and_wait;
        begin
            @(negedge clk); start <= 1;
            @(negedge clk); start <= 0;
            for (i = 0; i < 2000 && !done; i = i + 1) begin
                @(posedge clk);
            end
            if (i >= 2000) $display("  TIMEOUT");
        end
    endtask

    task check_results(input signed [47:0] expected);
        integer ok;
        begin
            ok = 1;
            for (i = 0; i < 64; i = i + 1) begin
                @(negedge clk);
                res_addr <= i;
                @(posedge clk);
                #1;
                if (res_dout !== expected) begin
                    if (ok) $display("  FAIL at row %0d: got %0d, expected %0d", i, res_dout, expected);
                    ok = 0;
                    errors = errors + 1;
                end
            end
            total = total + 64;
            if (ok) $display("  PASS: all 64 rows = %0d", expected);
        end
    endtask

    initial begin
        $dumpfile("tb_matmul_q8.vcd");
        $dumpvars(0, tb_matmul_q8);

        clk = 0; rst_n = 0; start = 0;
        wt_we = 0; sc_we = 0; act_we = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        // ---------------------------------------------------------------
        // Test 1: all weights=1, all scales=1.0 (UQ8.8=256), all acts=1
        //   dequant_q8(1, 256) = 1*256>>8 = 1
        //   Each row sum = 64 cols × 1 act × 1 wt = 64
        // ---------------------------------------------------------------
        $display("Test 1: all-ones (wt=1, sc=1.0, act=1) → expect 64");
        load_all_weights(8'd1);
        load_all_scales(16'd256);
        load_all_acts(16'd1);
        start_and_wait;
        check_results(64'd64);

        // ---------------------------------------------------------------
        // Test 2: all weights=0 → expect 0
        // ---------------------------------------------------------------
        $display("Test 2: all weights=0 → expect 0");
        load_all_weights(8'd0);
        load_all_scales(16'd256);
        load_all_acts(16'd1);
        start_and_wait;
        check_results(64'd0);

        // ---------------------------------------------------------------
        // Test 3: all weights=1, scales=2.0 (UQ8.8=512), acts=1
        //   dequant_q8(1, 512) = 1*512>>8 = 2
        //   Each row = 64 * 2 = 128
        // ---------------------------------------------------------------
        $display("Test 3: wt=1, sc=2.0, act=1 → expect 128");
        load_all_weights(8'd1);
        load_all_scales(16'd512);
        load_all_acts(16'd1);
        start_and_wait;
        check_results(128);

        // ---------------------------------------------------------------
        // Test 4: all weights=1, scales=1.0, acts=2
        //   dequant_q8(1, 256) = 1,  act=2
        //   Each row = 64 * 1 * 2 = 128
        // ---------------------------------------------------------------
        $display("Test 4: wt=1, sc=1.0, act=2 → expect 128");
        load_all_weights(8'd1);
        load_all_scales(16'd256);
        load_all_acts(16'd2);
        start_and_wait;
        check_results(128);

        // ---------------------------------------------------------------
        // Test 5: all weights=1, scales=1.0, acts=0 → expect 0
        // ---------------------------------------------------------------
        $display("Test 5: wt=1, sc=1.0, act=0 → expect 0");
        load_all_weights(8'd1);
        load_all_scales(16'd256);
        load_all_acts(16'd0);
        start_and_wait;
        check_results(0);

        // ---------------------------------------------------------------
        // Test 6: negative weights (-1), scales=1.0, acts=1
        //   dequant_q8(-1, 256) = (-1)*256>>8 = -1
        //   Each row = 64 * (-1) = -64
        // ---------------------------------------------------------------
        $display("Test 6: wt=-1, sc=1.0, act=1 → expect -64");
        load_all_weights(8'd255);  // -1 in 2's complement
        load_all_scales(16'd256);
        load_all_acts(16'd1);
        start_and_wait;
        check_results(-64);

        // ---------------------------------------------------------------
        // Summary
        // ---------------------------------------------------------------
        $display("========================================");
        if (errors === 0)
            $display("ALL %0d TESTS PASSED", total / 64);
        else
            $display("%0d / %0d CHECKS FAILED", errors, total);

        repeat (20) @(posedge clk);
        $finish;
    end

endmodule
