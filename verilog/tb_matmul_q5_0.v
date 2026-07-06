`timescale 1ns / 1ps

module tb_matmul_q5_0;

    reg         clk;
    reg         rst_n;
    reg         start;
    wire        done;
    wire        busy;

    reg         hdr_we;
    reg [2:0]   hdr_bank;
    reg [5:0]   hdr_addr;
    reg [7:0]   hdr_din;
    reg         qs_we;
    reg [1:0]   qs_addr;
    reg [31:0]  qs_din;
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
        .hdr_we   (hdr_we),
        .hdr_bank (hdr_bank),
        .hdr_addr (hdr_addr),
        .hdr_din  (hdr_din),
        .qs_we    (qs_we),
        .qs_addr  (qs_addr),
        .qs_din   (qs_din),
        .sc_we    (sc_we),
        .sc_addr  (sc_addr),
        .sc_din   (sc_din),
        .act_we   (act_we),
        .act_addr (act_addr),
        .act_din  (act_din),
        .res_addr (res_addr),
        .res_dout (res_dout),
        .core_id  (core_id),
        .dbg_tile_start(1'b0),
        .dbg_verbose(1'b1)
    );

    always #5 clk = ~clk;

    integer i, b, w, errors, total;

    // Write 1 header byte to LUTRAM bank+addr
    task write_hdr;
        input [2:0] bank;
        input [5:0] addr;
        input [7:0] data;
        begin
            @(negedge clk);
            hdr_we <= 1; hdr_bank <= bank; hdr_addr <= addr; hdr_din <= data;
            @(posedge clk);
            hdr_we <= 0;
        end
    endtask

    // Encode f16 as 2 header bytes (bank 0=lo, bank 1=hi) at given block_local
    task f16_encode_hdr;
        input real    val;
        input [5:0]   block_local;  // 0..55
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
            write_hdr(0, block_local, bits[7:0]);
            write_hdr(1, block_local, bits[15:8]);
        end
    endtask

    // Write one block's headers (6 bytes) to LUTRAM banks 0-5
    task write_block_headers;
        input [5:0]  block_local;
        input [3:0]  q5_val;
        begin
            f16_encode_hdr(1.0, block_local);
            write_hdr(2, block_local, 8'hFF);
            write_hdr(3, block_local, 8'hFF);
            write_hdr(4, block_local, 8'hFF);
            write_hdr(5, block_local, 8'hFF);
        end
    endtask

    // Write one block's qs data (4 × 32-bit words)
    task write_block_qs;
        input [5:0]  block_local;
        input [3:0]  q5_val;
        integer w;
        reg [31:0] qs_word;
        begin
            qs_word = {4{{q5_val, q5_val}}};
            for (w = 0; w < 4; w = w + 1) begin
                @(negedge clk);
                qs_we <= 1; qs_addr <= w[1:0]; qs_din <= qs_word;
                @(posedge clk);
                qs_we <= 0;
            end
        end
    endtask

    // Process a single block: write qs, start, wait for done
    task process_block;
        input [5:0]  block_local;
        input [3:0]  q5_val;
        integer poll;
        begin
            write_block_qs(block_local, q5_val);
            @(negedge clk); start <= 1;
            @(negedge clk); start <= 0;
            poll = 200000;
            for (i = 0; i < 200000 && poll == 200000; i = i + 1) begin
                @(posedge clk);
                if (done) poll = i + 1;
            end
            if (poll == 200000) $display("  TIMEOUT block %0d", block_local);
        end
    endtask

    // Write scales (shared, 8 × UQ16.8)
    task load_scales;
        integer s;
        begin
            for (s = 0; s < 8; s = s + 1) begin
                @(negedge clk); sc_we <= 1; sc_addr <= s; sc_din <= 1; @(posedge clk); sc_we <= 0;
            end
        end
    endtask

    // Write acts (shared, 896 × int16)
    task load_acts;
        input [15:0] act_val;
        integer a;
        begin
            for (a = 0; a < 896; a = a + 1) begin
                @(negedge clk); act_we <= 1; act_addr <= a; act_din <= act_val; @(posedge clk); act_we <= 0;
            end
        end
    endtask

    // Write all 56 block headers (persist in LUTRAM)
    task load_all_headers;
        input [3:0] q5_val;
        begin
            for (b = 0; b < 56; b = b + 1) write_block_headers(b[5:0], q5_val);
        end
    endtask

    // Process all 56 blocks sequentially
    task process_all_blocks;
        input [3:0] q5_val;
        begin
            for (b = 0; b < 56; b = b + 1) process_block(b[5:0], q5_val);
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
        hdr_we = 0; qs_we = 0; act_we = 0; sc_we = 0;
        core_id = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        $display("=== Q5_0 Core Unit Tests (core_id=0, rows 0-1) ===");

        $display("");
        $display("Test 1: all weights=1, all acts=1, scales=1");
        $display("  qh=0xFFFFFFFF, qs_nibble=1 -> q5 = ((1<<4)|1)-16 = 1");
        $display("  d=1.0 -> d_fp=256, val_norm = 256*1*1>>8 = 1");
        $display("  Expected per row = 896 * 1 * 1 = 896");
        load_all_headers(4'd1);
        load_scales;
        load_acts(1);
        process_all_blocks(4'd1);
        check_result(0, 896);
        check_result(1, 896);

        $display("");
        $display("Test 2: all weights=0 -> expect 0");
        rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
        load_all_headers(4'd0);
        load_scales;
        load_acts(1);
        process_all_blocks(4'd0);
        check_result(0, 0);
        check_result(1, 0);

        $display("");
        $display("Test 3: acts=0 -> expect 0");
        load_all_headers(4'd1);
        load_scales;
        load_acts(0);
        process_all_blocks(4'd1);
        check_result(0, 0);
        check_result(1, 0);

        $display("");
        $display("Test 4: weights=2, acts=2, scales=1");
        $display("  q5=2, val_norm=256*2*1>>8=2, prod=2*2=4");
        $display("  Expected per row = 896 * 4 = 3584");
        rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
        load_all_headers(4'd2);
        load_scales;
        load_acts(2);
        process_all_blocks(4'd2);
        check_result(0, 3584);
        check_result(1, 3584);

        $display("");
        $display("=== Total: %0d/%0d correct ===", total-errors, total);
        $finish;
    end

endmodule
