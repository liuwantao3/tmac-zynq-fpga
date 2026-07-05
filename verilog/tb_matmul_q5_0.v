`timescale 1ns / 1ps

module tb_matmul_q5_0;

    reg         clk;
    reg         rst_n;
    reg         start;
    wire        done;
    wire        busy;

    reg         wt_we;
    reg [2:0]   wt_bank;
    reg [9:0]   wt_addr;
    reg [7:0]   wt_din;
    reg         sc_we;
    reg [2:0]   sc_addr;
    reg [15:0]  sc_din;
    reg         act_we;
    reg [9:0]   act_addr;
    reg [15:0]  act_din;
    reg [0:0]   res_addr;
    wire [47:0] res_dout;
    reg [1:0]   core_id;

    matmul_q5_0_core u_core (
        .clk      (clk),
        .rst_n    (rst_n),
        .start    (start),
        .done     (done),
        .busy     (busy),
        .wt_we    (wt_we),
        .wt_bank  (wt_bank),
        .wt_addr  (wt_addr),
        .wt_din   (wt_din),
        .sc_we    (sc_we),
        .sc_addr  (sc_addr),
        .sc_din   (sc_din),
        .act_we   (act_we),
        .act_addr (act_addr),
        .act_din  (act_din),
        .res_addr (res_addr),
        .res_dout (res_dout),
        .core_id  (core_id),
        .dbg_verbose(1'b1)
    );

    always #5 clk = ~clk;

    integer i, errors, total;

    // Write 1 byte to a specific bank+addr
    task write_bank;
        input [2:0] bank;
        input [9:0] addr;
        input [7:0] data;
        begin
            @(negedge clk);
            wt_we <= 1; wt_bank <= bank; wt_addr <= addr; wt_din <= data;
            @(posedge clk);
            wt_we <= 0;
        end
    endtask

    // Encode f16 as 2 bytes (bank 0=lo, bank 1=hi) at given block_local
    task f16_encode_banked;
        input real    val;
        input [5:0]   block_local;  // 0..55
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
                e = 0;
                while (abs_val >= 2.0) begin abs_val = abs_val / 2.0; e = e + 1; end
                while (abs_val < 1.0 && e > -14) begin abs_val = abs_val * 2.0; e = e - 1; end
                exp = e + 15;
                mant = $rtoi((abs_val - 1.0) * 1024.0 + 0.5);
                if (exp >= 31) begin exp = 31; mant = 0; end
            end
            bits = {sign, exp, mant};
            write_bank(0, block_local, bits[7:0]);
            write_bank(1, block_local, bits[15:8]);
        end
    endtask

    // Generate one Q5_0 block (22 bytes) in banked format for given block_local
    task gen_block_banked;
        input [5:0]  block_local;  // 0..55
        input [3:0]  q5_val;       // 4-bit ql value for all 32 elements
        integer k;
        begin
            // Bytes 0-1: f16 scale = 1.0
            f16_encode_banked(1.0, block_local);
            // Bytes 2-5: qh = 0xFFFFFFFF (all high bits)
            write_bank(2, block_local, 8'hFF);
            write_bank(3, block_local, 8'hFF);
            write_bank(4, block_local, 8'hFF);
            write_bank(5, block_local, 8'hFF);
            // Bytes 6-21: qs array (16 bytes, each containing two q5_val nibbles)
            for (k = 0; k < 16; k = k + 1)
                write_bank(6, block_local * 16 + k, {q5_val, q5_val});
        end
    endtask

    // Load 56 blocks (2 rows) for core_id=0
    task load_all_banked;
        input [3:0] q5_val;
        integer b;
        begin
            for (b = 0; b < 56; b = b + 1) gen_block_banked(b, q5_val);
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
        input [0:0] row;
        input [47:0] expected;
        reg [47:0] got;
        begin
            res_addr <= row; @(posedge clk); #1; got = res_dout;
            if (got !== expected) begin
                $display("  FAIL row %0d: got %0d expected %0d", row, got, expected);
                errors = errors + 1;
            end else begin
                $display("  PASS row %0d: %0d", row, got);
            end
            total = total + 1;
        end
    endtask

    initial begin
        $dumpfile("tb_matmul_q5_0.vcd");
        $dumpvars(0, tb_matmul_q5_0);

        clk = 0; rst_n = 0; start = 0;
        wt_we = 0; act_we = 0; sc_we = 0;
        core_id = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        $display("=== Q5_0 BRAM Core Unit Tests (core_id=0, rows 0-1) ===");

        $display("");
        $display("Test 1: all weights=1, all acts=1, scales=1");
        $display("  qh=0xFFFFFFFF, qs_nibble=1 -> q5 = ((1<<4)|1)-16 = 1");
        $display("  d=1.0 -> d_fp=256, val_norm = 256*1*1>>8 = 1");
        $display("  Expected per row = 896 * 1 * 1 = 896");
        load_all_banked(4'd1);
        load_scales;
        load_acts(1);
        start_and_wait;
        check_result(0, 896);
        check_result(1, 896);

        $display("");
        $display("Test 2: all weights=0 -> expect 0");
        rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
        gen_block_banked(0, 4'd0);
        load_all_banked(4'd0);
        load_scales;
        load_acts(1);
        start_and_wait;
        check_result(0, 0);
        check_result(1, 0);

        $display("");
        $display("Test 3: acts=0 -> expect 0");
        load_all_banked(4'd1);
        load_scales;
        load_acts(0);
        start_and_wait;
        check_result(0, 0);
        check_result(1, 0);

        $display("");
        $display("Test 4: weights=2, acts=2, scales=1");
        $display("  q5=2, val_norm=256*2*1>>8=2, prod=2*2=4");
        $display("  Expected per row = 896 * 4 = 3584");
        rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
        load_all_banked(4'd2);
        load_scales;
        load_acts(2);
        start_and_wait;
        check_result(0, 3584);
        check_result(1, 3584);

        $display("");
        $display("=== Total: %0d/%0d correct ===", total-errors, total);
        $finish;
    end

endmodule
