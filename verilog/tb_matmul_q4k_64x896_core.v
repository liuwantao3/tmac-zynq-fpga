`timescale 1ns / 1ps

module tb_matmul_q4k_64x896;

    reg         clk;
    reg         rst_n;
    reg         start;
    wire        done;
    wire        busy;

    reg         wt_we;
    reg [12:0]  wt_addr;
    reg  [7:0]   wt_din;
    reg         act_we;
    reg [5:0]   act_addr;
    reg [15:0]  act_din;
    reg [5:0]   res_addr;
    wire [47:0] res_dout;

    reg         sc_we;
    reg [9:0]   sc_addr;
    reg [15:0]  sc_din;

    wire        mode_block_load = 1'b1;
    wire        decode_busy;

    matmul_q4k_64x896_core u_core (
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
        .res_dout  (res_dout),
        .mode_block_load (mode_block_load),
        .decode_busy     (decode_busy)
    );

    always #5 clk = ~clk;

    integer i, errors, total;

    task f16_encode;
        input real val;
        output reg [7:0] lo;
        output reg [7:0] hi;
        reg signed [15:0] bits;
        begin
            bits = $shortrealtobits(val);
            lo = bits[7:0];
            hi = bits[15:8];
        end
    endtask

    task load_block;
        input [3:0] q4_const;
        input [5:0] sc_val;
        input [5:0] m_val;
        input [7:0] block_idx;
        reg [7:0] lo, hi;
        integer base;
        begin
            base = block_idx * 144;
            f16_encode(1.0, lo, hi);
            u_core.block_buf[base+0] = lo;
            u_core.block_buf[base+1] = hi;
            f16_encode(0.0, lo, hi);
            u_core.block_buf[base+2] = lo;
            u_core.block_buf[base+3] = hi;
            for (i = 0; i < 8; i = i + 1) begin
                u_core.block_buf[base + 4 + i] = sc_val;
                u_core.block_buf[base + 8 + i] = m_val;
            end
            for (i = 0; i < 256; i = i + 1) begin
                u_core.block_buf[base + 16 + i] = q4_const;
            end
        end
    endtask

    task load_all_blocks;
        input [3:0] q4_const;
        integer b;
        begin
            for (b = 0; b < 224; b = b + 1) begin
                load_block(q4_const, 1, 0, b);
            end
        end
    endtask

    task start_and_wait;
        begin
            @(negedge clk); start <= 1;
            @(negedge clk); start <= 0;
            for (i = 0; i < 300000; i = i + 1) begin
                if (done) begin
                    $display("  DONE at cycle %0d", i);
                    return;
                end
                @(posedge clk);
            end
            $display("  TIMEOUT");
        end
    endtask

    task check_results;
        input [47:0] expected;
        integer r, ok;
        reg [47:0] got;
        begin
            ok = 1;
            for (r = 0; r < 64; r = r + 1) begin
                @(negedge clk); res_addr <= r;
                @(posedge clk); #1;
                got = res_dout;
                if (got !== expected) begin
                    if (ok && errors < 3) begin
                        $display("  FAIL row %d: got %d expected %d", r, got, expected);
                    end
                    ok = 0;
                    errors = errors + 1;
                end
            end
            total = total + 64;
            if (ok) $display("  PASS: all 64 rows = %d", expected);
        end
    endtask

    initial begin
        $dumpfile("tb_q4k_64x896.vcd");
        $dumpvars(0, tb_matmul_q4k_64x896);

        clk = 0; rst_n = 0; start = 0;
        wt_we = 0; act_we = 0; sc_we = 0;
        errors = 0; total = 0;

        repeat(4) @(posedge clk);
        rst_n <= 1;
        repeat(2) @(posedge clk);

        for (i = 0; i < 512; i = i + 1) begin
            u_core.wmem_lo[i] = 0;
            u_core.wmem_hi[i] = 0;
        end

        $display("Test 1: all weights=1, all acts=1, scale=256 → expect %d", 64 * 896);
        load_all_blocks(4'd1);
        for (i = 0; i < 64; i = i + 1) begin sc_we = 1; sc_addr = i; sc_din = 256; @(posedge clk); end
        sc_we = 0; @(posedge clk);
        for (i = 0; i < 896; i = i + 1) begin act_we = 1; act_addr = i; act_din = 1; @(posedge clk); end
        act_we = 0; @(posedge clk);
        start_and_wait;
        check_results(64'd64 * 896);

        $display("Test 2: all weights=0 → expect 0");
        rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
        load_all_blocks(4'd0);
        for (i = 0; i < 64; i = i + 1) begin sc_we = 1; sc_addr = i; sc_din = 256; @(posedge clk); end
        sc_we = 0; @(posedge clk);
        for (i = 0; i < 896; i = i + 1) begin act_we = 1; act_addr = i; act_din = 1; @(posedge clk); end
        act_we = 0; @(posedge clk);
        start_and_wait;
        check_results(64'd0);

        $display("Test 3: q4=2, act=2, scale=256 → expect 256");
        rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
        load_all_blocks(4'd2);
        for (i = 0; i < 64; i = i + 1) begin sc_we = 1; sc_addr = i; sc_din = 256; @(posedge clk); end
        sc_we = 0; @(posedge clk);
        for (i = 0; i < 896; i = i + 1) begin act_we = 1; act_addr = i; act_din = 2; @(posedge clk); end
        act_we = 0; @(posedge clk);
        start_and_wait;
        check_results(64'd256);

        $display("Test 4: all acts=0 → expect 0");
        rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
        load_all_blocks(4'd1);
        for (i = 0; i < 64; i = i + 1) begin sc_we = 1; sc_addr = i; sc_din = 256; @(posedge clk); end
        sc_we = 0; @(posedge clk);
        for (i = 0; i < 896; i = i + 1) begin act_we = 1; act_addr = i; act_din = 0; @(posedge clk); end
        act_we = 0; @(posedge clk);
        start_and_wait;
        check_results(64'd0);

        $display("\nTotal: %d/%d correct", total - errors, total);
        $finish;
    end

endmodule