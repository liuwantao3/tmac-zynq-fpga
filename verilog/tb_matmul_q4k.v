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

    // Q4K core doesn't use scales — tie off
    wire        sc_we = 1'b0;
    wire [6:0]  sc_addr = 7'd0;
    wire [15:0] sc_din = 16'd0;

    // Module-level arrays for weights and activations (used by tasks/functions)
    reg signed [15:0] tile_w [0:4095];
    reg signed [15:0] tile_a [0:63];

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
        .res_dout  (res_dout)
    );

    always #5 clk = ~clk;

    integer i, row, col, errors, total;

    // Write all 4096 INT16 weights from module-level tile_w
    task load_all_weights;
        integer load_addr;
        reg [15:0] wval;
        begin
            for (load_addr = 0; load_addr < 8192; load_addr = load_addr + 1) begin
                @(negedge clk);
                wval = tile_w[load_addr >> 1];
                if (load_addr[0] == 0)
                    wt_din <= wval[7:0];
                else
                    wt_din <= wval[15:8];
                wt_we   <= 1;
                // addr format: {byte_lane[3:0], entry[8:0]}
                wt_addr <= {load_addr[3:0], load_addr[12:4]};
            end
            @(negedge clk); wt_we <= 0;
        end
    endtask

    task load_all_acts;
        begin
            for (i = 0; i < 64; i = i + 1) begin
                @(negedge clk);
                act_we   <= 1;
                act_addr <= i;
                act_din  <= tile_a[i];
            end
            @(negedge clk); act_we <= 0;
        end
    endtask

    // Fill tile_w with constant value
    task gen_const_weights(input signed [15:0] val);
        begin
            for (i = 0; i < 4096; i = i + 1) tile_w[i] = val;
        end
    endtask

    // Fill tile_w with column-varying values
    task gen_col_weights(input signed [15:0] base);
        begin
            for (col = 0; col < 64; col = col + 1)
                for (row = 0; row < 64; row = row + 1)
                    tile_w[col * 64 + row] = base + col;
        end
    endtask

    // Fill tile_a with constant value
    task gen_acts(input signed [15:0] val);
        begin
            for (i = 0; i < 64; i = i + 1) tile_a[i] = val;
        end
    endtask

    // Computes expected dot product for row r using tile_w and tile_a
    function signed [63:0] dot_row;
        input integer r;
        reg signed [63:0] acc;
        integer c;
        begin
            acc = 0;
            for (c = 0; c < 64; c = c + 1)
                acc = acc + $signed(tile_w[c * 64 + r]) * $signed(tile_a[c]);
            dot_row = acc;
        end
    endfunction

    task start_and_wait;
        begin
            @(negedge clk); start <= 1;
            @(negedge clk); start <= 0;
            for (i = 0; i < 2000; i = i + 1) begin
                if (done) break;
                @(posedge clk);
            end
            if (i >= 2000) $display("  TIMEOUT");
        end
    endtask

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

    task check_results_ref;
        integer ok;
        reg signed [63:0] expected;
        begin
            ok = 1;
            for (i = 0; i < 64; i = i + 1) begin
                expected = dot_row(i);
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
            if (ok) $display("  PASS: all 64 rows match reference");
        end
    endtask

    initial begin
        $dumpfile("tb_matmul_q4k.vcd");
        $dumpvars(0, tb_matmul_q4k);

        clk = 0; rst_n = 0; start = 0;
        wt_we = 0; act_we = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        // ---------------------------------------------------------------
        // Test 1: all weights=1, all acts=1 → expect 64
        //   Each row = Σ(1 × 1) over 64 cols = 64
        // ---------------------------------------------------------------
        $display("Test 1: all-ones (wt=1, act=1) → expect 64");
        begin
            gen_const_weights(16'd1);
            gen_acts(16'd1);
            load_all_weights;
            load_all_acts;
            start_and_wait;
            check_results(64'd64);
        end

        // ---------------------------------------------------------------
        // Test 2: all weights=0 → expect 0
        // ---------------------------------------------------------------
        $display("Test 2: all weights=0 → expect 0");
        begin
            gen_const_weights(16'd0);
            gen_acts(16'd1);
            load_all_weights;
            load_all_acts;
            start_and_wait;
            check_results(64'd0);
        end

        // ---------------------------------------------------------------
        // Test 3: all weights=1, all acts=2 → expect 128
        //   Each row = 64 × 1 × 2 = 128
        // ---------------------------------------------------------------
        $display("Test 3: wt=1, act=2 → expect 128");
        begin
            gen_const_weights(16'd1);
            gen_acts(16'd2);
            load_all_weights;
            load_all_acts;
            start_and_wait;
            check_results(64'd128);
        end

        // ---------------------------------------------------------------
        // Test 4: all acts=0 → expect 0
        // ---------------------------------------------------------------
        $display("Test 4: wt=1, act=0 → expect 0");
        begin
            gen_const_weights(16'd1);
            gen_acts(16'd0);
            load_all_weights;
            load_all_acts;
            start_and_wait;
            check_results(64'd0);
        end

        // ---------------------------------------------------------------
        // Test 5: negative weights (-1), acts=1 → expect -64
        // ---------------------------------------------------------------
        $display("Test 5: wt=-1, act=1 → expect -64");
        begin
            gen_const_weights(-16'd1);
            gen_acts(16'd1);
            load_all_weights;
            load_all_acts;
            start_and_wait;
            check_results(-64'd64);
        end

        // ---------------------------------------------------------------
        // Test 6: varying weights per column, acts=1
        //   Row r = Σ(col=0..63) (base+col) × 1 = 64*base + Σcol = 64*base + 2016
        // ---------------------------------------------------------------
        $display("Test 6: col-varying weights, act=1");
        begin
            gen_col_weights(16'd10);
            gen_acts(16'd1);
            load_all_weights;
            load_all_acts;
            start_and_wait;
            // Each row: Σ(10+col) over 64 cols = 64*10 + Σ(0..63) = 640 + 2016 = 2656
            check_results(64'd2656);
        end

        // ---------------------------------------------------------------
        // Test 7: mixed values, reference-check each row
        // ---------------------------------------------------------------
        $display("Test 7: mixed values, per-row reference check");
        begin
            // Generate varying weights: w[col*64+row] = (col*64+row) mod 200 - 100
            for (col = 0; col < 64; col = col + 1)
                for (row = 0; row < 64; row = row + 1)
                    tile_w[col * 64 + row] = (col * 64 + row) % 200 - 100;
            // Generate varying acts: a[i] = i * 2 - 63
            for (i = 0; i < 64; i = i + 1)
                tile_a[i] = i * 2 - 63;
            load_all_weights;
            load_all_acts;
            start_and_wait;
            check_results_ref;
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
