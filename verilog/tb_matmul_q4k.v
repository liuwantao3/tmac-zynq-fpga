`timescale 1ns / 1ps

module tb_matmul_q4k;

    reg         clk;
    reg         rst_n;
    reg         start;
    wire        done;
    wire        busy;

    reg         wt_we;
    reg [12:0]  wt_addr;
    reg [7:0]   wt_din;
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

    // Q4_K block constants
    reg [7:0] q4k_block_mem [0:143];

    matmul_q4k_core u_core (
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

    integer i, j, row, col, errors, total;

    // Encode float16 into two bytes (little-endian)
    task f16_encode(input real val, output reg [7:0] lo, output reg [7:0] hi);
        reg signed [15:0] bits;
        reg sign;
        reg [4:0] exp;
        reg [9:0] mant;
        real abs_val;
        integer e;
        real frac;
        begin
            sign = (val < 0) ? 1 : 0;
            abs_val = (val < 0) ? -val : val;
            if (abs_val == 0.0) begin exp = 0; mant = 0; end
            else begin
                frac = $realtobits(abs_val);
                e = 0;
                while (abs_val >= 2.0) begin abs_val = abs_val / 2.0; e = e + 1; end
                while (abs_val < 1.0 && e > -14) begin abs_val = abs_val * 2.0; e = e - 1; end
                exp = e + 15;
                mant = $rtoi((abs_val - 1.0) * 1024.0 + 0.5);
                if (exp >= 31) begin exp = 31; mant = 0; end
            end
            bits = {sign, exp, mant};
            lo = bits[7:0];
            hi = bits[15:8];
        end
    endtask

    // Generate a Q4_K block with constant weight value
    // d = 1.0, dmin = 0.0, sc = 1, m = 0, all quants = q4_const
    task gen_const_block(input [3:0] q4_const);
        reg [7:0] lo, hi;
        integer bi, bj;
        reg [7:0] qs_byte;
        begin
            f16_encode(1.0, lo, hi);
            q4k_block_mem[0] = lo; q4k_block_mem[1] = hi;
            f16_encode(0.0, lo, hi);
            q4k_block_mem[2] = lo; q4k_block_mem[3] = hi;
            for (bi = 4; bi < 12; bi = bi + 1) q4k_block_mem[bi] = 8'd1;
            for (bi = 12; bi < 16; bi = bi + 1) q4k_block_mem[bi] = 8'h11;
            qs_byte = {q4_const, q4_const};
            for (bi = 16; bi < 144; bi = bi + 1) q4k_block_mem[bi] = qs_byte;
        end
    endtask

    // Generate a Q4_K block with varying weights per column
    task gen_var_block(input [3:0] base);
        reg [7:0] lo, hi;
        integer bi, bj, sub, idx, qval;
        reg [7:0] qs_byte;
        begin
            f16_encode(1.0, lo, hi);
            q4k_block_mem[0] = lo; q4k_block_mem[1] = hi;
            f16_encode(0.0, lo, hi);
            q4k_block_mem[2] = lo; q4k_block_mem[3] = hi;
            for (bi = 4; bi < 12; bi = bi + 1) q4k_block_mem[bi] = 8'd1;
            for (bi = 12; bi < 16; bi = bi + 1) q4k_block_mem[bi] = 8'h11;
            // sub = 0..7, j = 0..31
            for (sub = 0; sub < 8; sub = sub + 1) begin
                for (bj = 0; bj < 32; bj = bj + 1) begin
                    idx = sub * 32 + bj;
                    if (sub % 2 == 0)
                        qval = base + (idx / 32);
                    else
                        qval = base + (idx / 32);
                    q4k_block_mem[16 + (sub/2)*32 + bj/2] =
                        (sub % 2 == 0)
                            ? (q4k_block_mem[16 + (sub/2)*32 + bj/2] & 8'hF0) | (qval & 8'h0F)
                            : (q4k_block_mem[16 + (sub/2)*32 + bj/2] & 8'h0F) | ((qval & 8'h0F) << 4);
                end
            end
        end
    endtask

    // Load a single Q4_K block into block_buf via wt_we
    task load_block;
        integer bi;
        begin
            for (bi = 0; bi < 144; bi = bi + 1) begin
                @(negedge clk);
                wt_we  <= 1;
                wt_din <= q4k_block_mem[bi];
            end
            @(negedge clk); wt_we <= 0;
        end
    endtask

    // Load multiple blocks (first block uses gen_const_block, rest are copies)
    task load_all_blocks(input [3:0] q4_const);
        integer b;
        begin
            gen_const_block(q4_const);
            for (b = 0; b < 224; b = b + 1) load_block;
        end
    endtask

    // Load identity row_scale (256 for all 896 rows)
    task load_identity_scale;
        integer r;
        begin
            for (r = 0; r < 896; r = r + 1) begin
                @(negedge clk);
                sc_we   <= 1;
                sc_addr <= r;
                sc_din  <= 16'd256;
            end
            @(negedge clk); sc_we <= 0;
        end
    endtask

    // Load activations (all = val)
    task load_acts(input signed [15:0] val);
        begin
            for (i = 0; i < 64; i = i + 1) begin
                @(negedge clk);
                act_we   <= 1;
                act_addr <= i;
                act_din  <= val;
            end
            @(negedge clk); act_we <= 0;
        end
    endtask

    // Start compute and wait for done (up to 200K cycles)
    task start_and_wait;
        begin
            @(negedge clk); start <= 1;
            @(negedge clk); start <= 0;
            for (i = 0; i < 200000; i = i + 1) begin
                if (done) break;
                @(posedge clk);
            end
            if (i >= 200000) $display("  TIMEOUT");
            else $display("  DONE at cycle %0d", i);
        end
    endtask

    // Check first 64 results against expected value
    task check_results(input signed [63:0] expected);
        integer ok;
        begin
            ok = 1;
            for (i = 0; i < 64; i = i + 1) begin
                @(negedge clk);
                res_addr <= i;
                @(posedge clk);
                #1;
                if ($signed(res_dout) !== expected) begin
                    if (ok) $display("  FAIL at row %0d: got %0d, expected %0d", i,
                                     $signed(res_dout), expected);
                    ok = 0;
                    errors = errors + 1;
                end
            end
            total = total + 64;
            if (ok) $display("  PASS: all 64 rows = %0d", $signed(expected));
        end
    endtask

    initial begin
        $dumpfile("tb_matmul_q4k.vcd");
        $dumpvars(0, tb_matmul_q4k);

        clk = 0; rst_n = 0; start = 0;
        wt_we = 0; act_we = 0; sc_we = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        // Force wmem to 0 to avoid x propagation
        for (i = 0; i < 512; i = i + 1) begin
            u_core.wmem_lo[i] = 64'd0;
            u_core.wmem_hi[i] = 64'd0;
        end

        // ---------------------------------------------------------------
        // Test 1: all weights=1, all acts=1 → expect 64 × 1 = 64
        //   d=1.0, sc=1, q4=1, dmin=0, m=0 → decoded=1, val_norm=1
        //   Each row = Σ(1 × 1) over 64 cols = 64
        // ---------------------------------------------------------------
        $display("Test 1: all-ones (q4_const=1, act=1) → expect 64");
        begin
            load_all_blocks(4'd1);
            load_identity_scale;
            load_acts(16'd1);
            start_and_wait;
            check_results(64'd64);
        end

        // ---------------------------------------------------------------
        // Test 2: all weights=0 → expect 0
        // ---------------------------------------------------------------
        $display("Test 2: all weights=0 → expect 0");
        begin
            load_all_blocks(4'd0);
            load_identity_scale;
            load_acts(16'd1);
            start_and_wait;
            check_results(64'd0);
        end

        // ---------------------------------------------------------------
        // Test 3: q4_const=2, all acts=2 → expect 64 × 2 × 2 = 256
        //   decoded=2, val_norm=2, sum=64 × 2 × act=2 = 256
        // ---------------------------------------------------------------
        $display("Test 3: q4=2, act=2 → expect 256");
        begin
            load_all_blocks(4'd2);
            load_identity_scale;
            load_acts(16'd2);
            start_and_wait;
            check_results(64'd256);
        end

        // ---------------------------------------------------------------
        // Test 4: all acts=0 → expect 0
        // ---------------------------------------------------------------
        $display("Test 4: act=0 → expect 0");
        begin
            load_all_blocks(4'd1);
            load_identity_scale;
            load_acts(16'd0);
            start_and_wait;
            check_results(64'd0);
        end

        // ---------------------------------------------------------------
        // Test 5: q4_const=0, act=1 → expect 0
        // ---------------------------------------------------------------
        $display("Test 5: q4=0, act=1 → expect 0");
        begin
            load_all_blocks(4'd0);
            load_identity_scale;
            load_acts(16'd1);
            start_and_wait;
            check_results(64'd0);
        end

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
