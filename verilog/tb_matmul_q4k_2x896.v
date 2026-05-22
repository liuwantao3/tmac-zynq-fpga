`timescale 1ns / 1ps

module tb_matmul_q4k_2x896;

    reg         clk;
    reg         rst_n;
    reg         start;
    wire        done;
    wire        busy;

    reg         wt_we;
    reg [9:0]   wt_addr;
    reg [7:0]   wt_din;
    reg         sc_we;
    reg [0:0]   sc_addr;
    reg [15:0]  sc_din;
    reg         act_we;
    reg [9:0]   act_addr;
    reg [15:0]  act_din;
    reg [0:0]   res_addr;
    wire [47:0] res_dout;

    matmul_q4k_2x896_core u_core (
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
        input [9:0] addr;
        input [7:0] data;
        begin
            @(negedge clk);
            wt_we <= 1; wt_addr <= addr; wt_din <= data;
            @(posedge clk);
            wt_we <= 0;
        end
    endtask

    task gen_block;
        input [2:0] bi;
        input [3:0] q4_val;
        integer base;
        integer k;
        begin
            base = bi * 144;
            // f16 d = 1.0 = 0x3C00 (lo=00, hi=3C)
            write_byte(base+0, 8'h00);
            write_byte(base+1, 8'h3C);
            // f16 dmin = 0.0 = 0x0000
            write_byte(base+2, 8'h00);
            write_byte(base+3, 8'h00);
            // sc subs 0-3: each = 6'd1, placed at [5:0] of base+4..7
            // sc subs 4-7: top 2 at [7:6] of base+4..7, low 4 at [3:0] of base+12..15
            // For sc=1: low 4 bits = 4'd1, top 2 bits = 2'd0
            // So base+4..7: subs 0-3 sc at [5:0]=6'd1, [7:6]=2'd0 for sub 4-7 top
            // → byte = {2'b00, 6'b000001} = 8'h01
            for (k = 0; k < 4; k = k + 1) write_byte(base+4+k, 8'h01);
            // m subs 0-3: each = 6'd0, at [5:0] of base+8..11
            // m subs 4-7: top 2 at [7:6] of base+8..11, low 4 at [7:4] of base+12..15
            // For m=0: low 4 bits = 0, top 2 bits = 0
            for (k = 0; k < 4; k = k + 1) write_byte(base+8+k, 8'h00);
            // base+12: sub4 m_low[3:0] at [7:4], sub4 sc_low[3:0] at [3:0]
            // m_low=0 → [7:4]=0000, sc_low=4'd1 → [3:0]=0001 → byte = 8'h01
            write_byte(base+12, 8'h01);
            // base+13: sub5 m_low at [7:4], sub5 sc_low at [3:0]
            write_byte(base+13, 8'h01);
            // base+14: sub6
            write_byte(base+14, 8'h01);
            // base+15: sub7
            write_byte(base+15, 8'h01);
            // qs bytes: 128 bytes at base+16..143
            // each byte: low nibble = sub_even q4, high nibble = sub_odd q4
            // q4=1 for all → byte = 8'h11
            for (k = 0; k < 128; k = k + 1) write_byte(base+16+k, {q4_val, q4_val});
        end
    endtask

    task load_all;
        input [3:0] q4_val;
        integer b;
        begin
            for (b = 0; b < 7; b = b + 1) gen_block(b, q4_val);
        end
    endtask

    task load_scales;
        begin
            @(negedge clk); sc_we <= 1; sc_addr <= 0; sc_din <= 256; @(posedge clk); sc_we <= 0;
            @(negedge clk); sc_we <= 1; sc_addr <= 1; sc_din <= 256; @(posedge clk); sc_we <= 0;
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
        begin
            @(negedge clk); start <= 1;
            @(negedge clk); start <= 0;
            for (i = 0; i < 10000; i = i + 1) begin
                @(posedge clk);
                if (done) begin
                    $display("  DONE at cycle %0d", i);
                    return;
                end
            end
            $display("  TIMEOUT");
        end
    endtask

    task check_results;
        input [47:0] exp0;
        input [47:0] exp1;
        reg ok;
        reg [47:0] got;
        begin
            ok = 1;
            @(negedge clk);
            res_addr = 0; @(posedge clk); #1; got = res_dout;
            if (got !== exp0) begin
                $display("  FAIL row 0: got %d expected %d", got, exp0);
                ok = 0; errors = errors + 1;
            end
            res_addr = 1; @(posedge clk); #1; got = res_dout;
            if (got !== exp1) begin
                $display("  FAIL row 1: got %d expected %d", got, exp1);
                ok = 0; errors = errors + 1;
            end
            total = total + 2;
            if (ok) $display("  PASS: [0]=%d [1]=%d", exp0, exp1);
        end
    endtask

    initial begin
        $dumpfile("tb_q4k_2x896.vcd");
        $dumpvars(0, tb_matmul_q4k_2x896);

        clk = 0; rst_n = 0; start = 0;
        wt_we = 0; act_we = 0; sc_we = 0;
        errors = 0; total = 0;

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        $display("Test 1: all weights=1, all acts=1 → expect %d", 896);
        load_all(4'd1);
        load_scales;
        load_acts(1);
        start_and_wait;
        check_results(896, 896);

        $display("");
        $display("Test 2: all weights=0 → expect 0");
        rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
        load_all(4'd0);
        load_scales;
        load_acts(1);
        start_and_wait;
        check_results(0, 0);

        $display("");
        $display("Test 3: act=0 → expect 0");
        load_all(4'd1);
        load_scales;
        load_acts(0);
        start_and_wait;
        check_results(0, 0);

        $display("");
        $display("Test 4: q4=2, act=2 → expect %d", 2*896*2);
        rst_n = 0; @(posedge clk); rst_n = 1; @(posedge clk);
        load_all(4'd2);
        load_scales;
        load_acts(2);
        start_and_wait;
        check_results(3584, 3584);

        $display("");
        $display("=== Total: %d/%d correct ===", total-errors, total);
        $finish;
    end

endmodule