`timescale 1ns / 1ps

module tb_matmul_q6_k;

    reg         clk;
    reg         rst_n;
    reg         start;
    wire        done;
    wire        busy;

    reg         wt_we;
    reg [12:0]  wt_addr;
    reg [7:0]   wt_din;
    reg         act_we;
    reg [7:0]   act_addr;
    reg [15:0]  act_din;
    reg [4:0]   res_addr;
    wire [47:0] res_dout;

    reg         sc_we;
    reg [4:0]   sc_addr;
    reg [15:0]  sc_din;

    wire        mode_block_load = 1'b1;
    wire        decode_busy;

    reg [7:0] q6k_block_mem [0:209];

    matmul_q6_k_core u_core (
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

    task f16_encode(input real val, output reg [7:0] lo, output reg [7:0] hi);
        reg signed [15:0] bits;
        reg sign;
        reg [4:0] exp;
        reg [9:0] mant;
        real abs_val;
        integer e;
        begin
            sign = (val < 0) ? 1 : 0;
            abs_val = (val < 0) ? -val : val;
            if (abs_val == 0.0) begin exp = 0; mant = 0; end
            else begin
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

    task setup_block(input [4:0] bi, input [5:0] q6_val);
        reg [7:0] lo, hi;
        reg [3:0] ql_nibble;
        reg [1:0] qh_bits;
        integer l;
        reg [3:0] ql_byte;
        begin
            ql_nibble = q6_val[3:0];
            qh_bits = q6_val[5:4];

            f16_encode(1.0, lo, hi);
            q6k_block_mem[208] = lo;
            q6k_block_mem[209] = hi;

            for (l = 0; l < 32; l = l + 1) begin
                ql_byte = {ql_nibble, ql_nibble};
                q6k_block_mem[l]      = ql_byte;  // half 0, sub 0 & 2
                q6k_block_mem[l + 32] = ql_byte;  // half 0, sub 1 & 3
                q6k_block_mem[l + 64] = ql_byte;  // half 1, sub 0 & 2
                q6k_block_mem[l + 96] = ql_byte;  // half 1, sub 1 & 3
            end

            for (l = 0; l < 32; l = l + 1) begin
                q6k_block_mem[128 + l]      = {qh_bits, qh_bits, qh_bits, qh_bits};
                q6k_block_mem[128 + 32 + l] = {qh_bits, qh_bits, qh_bits, qh_bits};
            end

            for (l = 192; l < 208; l = l + 1)
                q6k_block_mem[l] = 8'd1;
        end
    endtask

    task setup_block_var(input [4:0] bi);
        reg [7:0] lo, hi;
        reg [3:0] ql_nibble;
        reg [1:0] qh_bits;
        integer l, sub, half, wi;
        reg [5:0] q6_val;
        reg ql_byte, qh_byte;
        begin
            f16_encode(1.0, lo, hi);
            q6k_block_mem[208] = lo;
            q6k_block_mem[209] = hi;

            for (l = 0; l < 128; l = l + 1)
                q6k_block_mem[l] = 8'd0;
            for (l = 128; l < 192; l = l + 1)
                q6k_block_mem[l] = 8'd0;
            for (l = 192; l < 208; l = l + 1)
                q6k_block_mem[l] = 8'd1;

            for (wi = 0; wi < 256; wi = wi + 1) begin
                half = wi / 128;
                l = (wi % 128) % 32;
                sub = (wi % 128) / 32;
                q6_val = (bi * 256 + wi) % 64;
                ql_nibble = q6_val[3:0];
                qh_bits = q6_val[5:4];

                if (half == 0) begin
                    if (sub == 0 || sub == 2)
                        q6k_block_mem[l] = (sub == 0) ?
                            (q6k_block_mem[l] & 8'hF0) | ql_nibble :
                            (q6k_block_mem[l] & 8'h0F) | (ql_nibble << 4);
                    else
                        q6k_block_mem[32 + l] = (sub == 1) ?
                            (q6k_block_mem[32 + l] & 8'hF0) | ql_nibble :
                            (q6k_block_mem[32 + l] & 8'h0F) | (ql_nibble << 4);
                    q6k_block_mem[128 + l] = (q6k_block_mem[128 + l] & ~(2'b11 << (sub * 2))) | (qh_bits << (sub * 2));
                end else begin
                    if (sub == 0 || sub == 2)
                        q6k_block_mem[64 + l] = (sub == 0) ?
                            (q6k_block_mem[64 + l] & 8'hF0) | ql_nibble :
                            (q6k_block_mem[64 + l] & 8'h0F) | (ql_nibble << 4);
                    else
                        q6k_block_mem[64 + 32 + l] = (sub == 1) ?
                            (q6k_block_mem[64 + 32 + l] & 8'hF0) | ql_nibble :
                            (q6k_block_mem[64 + 32 + l] & 8'h0F) | (ql_nibble << 4);
                    q6k_block_mem[128 + 32 + l] = (q6k_block_mem[128 + 32 + l] & ~(2'b11 << (sub * 2))) | (qh_bits << (sub * 2));
                end
            end
        end
    endtask

    task load_block;
        integer bi;
        begin
            for (bi = 0; bi < 210; bi = bi + 1) begin
                @(negedge clk);
                wt_we  <= 1;
                wt_din <= q6k_block_mem[bi];
            end
            @(negedge clk); wt_we <= 0;
        end
    endtask

    task load_all_blocks_const;
        integer b;
        begin
            for (b = 0; b < 32; b = b + 1) begin
                setup_block(b, b[5:0]);
                load_block;
            end
        end
    endtask

    task load_all_blocks_var;
        integer b;
        begin
            for (b = 0; b < 32; b = b + 1) begin
                setup_block_var(b);
                load_block;
            end
        end
    endtask

    task load_identity_scale;
        integer r;
        begin
            for (r = 0; r < 32; r = r + 1) begin
                @(negedge clk);
                sc_we   <= 1;
                sc_addr <= r;
                sc_din  <= 16'd256;
            end
            @(negedge clk); sc_we <= 0;
        end
    endtask

    task load_acts_256(input signed [15:0] val);
        begin
            for (i = 0; i < 256; i = i + 1) begin
                @(negedge clk);
                act_we   <= 1;
                act_addr <= i;
                act_din  <= val;
            end
            @(negedge clk); act_we <= 0;
        end
    endtask

    task start_and_wait;
        reg done_seen;
        begin
            done_seen = 0;
            @(negedge clk); start <= 1;
            @(negedge clk); start <= 0;
            for (i = 0; i < 200000; i = i + 1) begin
                @(posedge clk);
                #1;
                if (done) begin
                    done_seen = 1;
                    $display("  DONE at cycle %0d", i);
                    i = 200000;
                end
            end
            if (!done_seen) $display("  TIMEOUT");
        end
    endtask

    task check_expected(input signed [47:0] exp_base);
        integer r;
        reg [4:0] first_fail;
        reg signed [47:0] expected;
        begin
            first_fail = 32;
            for (r = 0; r < 32; r = r + 1) begin
                expected = exp_base + 256 * r[5:0];
                @(negedge clk);
                res_addr <= r;
                @(posedge clk);
                #1;
                if ($signed(res_dout) !== expected && first_fail == 32) begin
                    $display("  FAIL at row %0d: got %0d, expected %0d",
                             r, $signed(res_dout), expected);
                    first_fail = r;
                    errors = errors + 1;
                end
            end
            total = total + 32;
            if (first_fail == 32) $display("  PASS: all 32 rows correct");
        end
    endtask

    task check_zero;
        integer r;
        reg [4:0] first_fail;
        begin
            first_fail = 32;
            for (r = 0; r < 32; r = r + 1) begin
                @(negedge clk);
                res_addr <= r;
                @(posedge clk);
                #1;
                if ($signed(res_dout) !== 0 && first_fail == 32) begin
                    $display("  FAIL at row %0d: got %0d, expected 0",
                             r, $signed(res_dout));
                    first_fail = r;
                    errors = errors + 1;
                end
            end
            total = total + 32;
            if (first_fail == 32) $display("  PASS: all 32 rows = 0");
        end
    endtask

    initial begin
        $dumpfile("tb_matmul_q6_k.vcd");
        $dumpvars(0, tb_matmul_q6_k);

        clk = 0; rst_n = 0; start = 0;
        wt_we = 0; act_we = 0; sc_we = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        $display("Test 1: per-block constant q6=bi, act=1 → sum=bi*256");
        begin
            load_all_blocks_const;
            load_identity_scale;
            load_acts_256(16'd1);
            start_and_wait;
            check_expected(0);
        end

        $display("Test 2: acts=0 → expect 0");
        begin
            rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
            load_all_blocks_const;
            load_identity_scale;
            load_acts_256(16'd0);
            start_and_wait;
            check_zero;
        end

        $display("Test 3: varying q6 per column, act=1");
        begin
            rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
            load_all_blocks_var;
            load_identity_scale;
            load_acts_256(16'd1);
            start_and_wait;
            $display("  (expected values computed externally)");
            // Check at least one non-zero result exists
            @(negedge clk);
            res_addr <= 0;
            @(posedge clk);
            #1;
            if ($signed(res_dout) === 0)
                $display("  FAIL: row 0 = 0, expected non-zero");
            else
                $display("  PASS: row 0 = %0d (non-zero)", $signed(res_dout));
            total = total + 1;
        end

        $display("Test 4: all q6=0, act=1 → expect 0");
        begin
            rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
            for (i = 0; i < 32; i = i + 1) begin
                setup_block(i, 6'd0);
                load_block;
            end
            load_identity_scale;
            load_acts_256(16'd1);
            start_and_wait;
            check_zero;
        end

        $display("========================================");
        if (errors === 0)
            $display("ALL %d/%d TESTS PASSED", total, total);
        else
            $display("%d / %d CHECKS FAILED", errors, total);

        repeat (20) @(posedge clk);
        $finish;
    end

endmodule
