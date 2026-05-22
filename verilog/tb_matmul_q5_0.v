`timescale 1ns / 1ps

module tb_matmul_q5_0;

    reg         clk;
    reg         rst_n;
    reg         start;
    wire        done;
    wire        busy;

    reg         wt_we;
    reg [12:0]  wt_addr;
    reg [7:0]   wt_din;
    reg         sc_we;
    reg [2:0]   sc_addr;
    reg [15:0]  sc_din;
    reg         act_we;
    reg [9:0]   act_addr;
    reg [15:0]  act_din;
    reg [2:0]   res_addr;
    wire [47:0] res_dout;

    matmul_q5_0_core u_core (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (start),
        .done     (done),
        .busy     (busy),
        .wt_we    (wt_we),
        .wt_addr  (wt_addr),
        .wt_din   (wt_din),
        .sc_we    (sc_we),
        .sc_addr  (sc_addr),
        .sc_din   (sc_din),
        .act_we   (act_we),
        .act_addr (act_addr),
        .act_din  (act_din),
        .res_addr (res_addr),
        .res_dout (res_dout)
    );

    always #5 clk = ~clk;

    integer i, errors, total;

    task write_byte;
        input [12:0] addr;
        input [7:0] data;
        begin
            @(negedge clk);
            wt_we <= 1; wt_addr <= addr; wt_din <= data;
            @(posedge clk);
            wt_we <= 0;
        end
    endtask

    task gen_block;
        input [7:0] bi;
        input [3:0] q5_val;
        integer base;
        integer k;
        begin
            base = bi * 22;
            f16_encode(1.0, base+0, base+1);
            write_byte(base+2, 8'hFF);
            write_byte(base+3, 8'hFF);
            write_byte(base+4, 8'hFF);
            write_byte(base+5, 8'hFF);
            for (k = 0; k < 16; k = k + 1)
                write_byte(base+6+k, {q5_val, q5_val});
        end
    endtask

    task f16_encode;
        input real val;
        input [12:0] addr_lo;
        input [12:0] addr_hi;
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
            write_byte(addr_lo, bits[7:0]);
            write_byte(addr_hi, bits[15:8]);
        end
    endtask

    task load_all;
        input [3:0] q5_val;
        integer b;
        begin
            for (b = 0; b < 224; b = b + 1) gen_block(b, q5_val);
        end
    endtask

    task load_scales;
        integer s;
        begin
            for (s = 0; s < 8; s = s + 1) begin
                @(negedge clk); sc_we <= 1; sc_addr <= s; sc_din <= 1; @(posedge clk); sc_we <= 0;
            end
        end
    endtask

    task load_acts;
        input [15:0] act_val;
        integer a;
        begin
            for (a = 0; a < 896; a = a + 1) begin
                @(negedge clk); act_we <= 1; act_addr <= a; act_din <= act_val; @(posedge clk); act_we <= 0;
            end
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
                if (done) begin
                    done_seen = 1;
                    $display("  DONE at cycle %0d", i);
                end
            end
            if (!done_seen) $display("  TIMEOUT");
        end
    endtask

    task check_result;
        input [2:0] row;
        input [47:0] expected;
        reg [47:0] got;
        begin
            res_addr <= row; @(posedge clk); #1; got = res_dout;
            if (got !== expected) begin
                $display("  FAIL row %d: got %d expected %d", row, got, expected);
                errors = errors + 1;
            end else begin
                $display("  PASS row %d: %d", row, got);
            end
            total = total + 1;
        end
    endtask

    task check_all_rows;
        input [47:0] expected;
        integer r;
        begin
            for (r = 0; r < 8; r = r + 1) check_result(r, expected);
        end
    endtask

    initial begin
        $dumpfile("tb_matmul_q5_0.vcd");
        $dumpvars(0, tb_matmul_q5_0);

        clk = 0; rst_n = 0; start = 0;
        wt_we = 0; act_we = 0; sc_we = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        $display("Test 1: all weights=1, all acts=1, scales=1");
        $display("  qh=0xFFFFFFFF, qs_nibble=1 -> q5 = ((1<<4)|1)-16 = 1");
        $display("  d=1.0 -> d_fp=256, val_norm = 256*1*1>>8 = 1");
        $display("  Expected per row = 896 * 1 * 1 = 896");
        load_all(4'd1);
        load_scales;
        load_acts(1);
        start_and_wait;
        check_all_rows(896);

        $display("");
        $display("Test 2: all weights=0 -> expect 0");
        rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
        gen_block(0, 4'd0);
        load_all(4'd0);
        load_scales;
        load_acts(1);
        start_and_wait;
        check_all_rows(0);

        $display("");
        $display("Test 3: acts=0 -> expect 0");
        load_all(4'd1);
        load_scales;
        load_acts(0);
        start_and_wait;
        check_all_rows(0);

        $display("");
        $display("Test 4: weights=2, acts=2, scales=1");
        $display("  q5=2, val_norm=256*2*1>>8=2, prod=2*2=4");
        $display("  Expected per row = 896 * 4 = 3584");
        rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
        load_all(4'd2);
        load_scales;
        load_acts(2);
        start_and_wait;
        check_all_rows(3584);

        $display("");
        $display("=== Total: %d/%d correct ===", total-errors, total);
        $finish;
    end

endmodule