`timescale 1ns / 1ps

module tb_q5_off_by_one;
    reg clk, rst_n, start;
    wire done, busy;
    reg hdr_we; reg [2:0] hdr_bank; reg [5:0] hdr_addr; reg [7:0] hdr_din;
    reg qs_we; reg [1:0] qs_addr; reg [31:0] qs_din;
    reg sc_we; reg [2:0] sc_addr; reg [15:0] sc_din;
    reg act_we; reg [9:0] act_addr; reg [15:0] act_din;
    reg [0:0] res_addr;
    wire [47:0] res_dout;
    reg [1:0] core_id;

    matmul_q5_0_core u_core (
        .clk(clk), .rst_n(rst_n), .start(start),
        .done(done), .busy(busy),
        .hdr_we(hdr_we), .hdr_bank(hdr_bank), .hdr_addr(hdr_addr), .hdr_din(hdr_din),
        .qs_we(qs_we), .qs_addr(qs_addr), .qs_din(qs_din),
        .sc_we(sc_we), .sc_addr(sc_addr), .sc_din(sc_din),
        .act_we(act_we), .act_addr(act_addr), .act_din(act_din),
        .res_addr(res_addr), .res_dout(res_dout),
        .core_id(core_id),
        .dbg_tile_start(1'b0), .dbg_tile_cycles(), .dbg_tile_id(), .dbg_verbose(1'b0)
    );

    always #5 clk = ~clk;
    integer b, k, i, w;
    reg [31:0] qs_word;

    task write_hdr;
        input [2:0] bank;
        input [5:0] addr;
        input [7:0] data;
        begin
            @(negedge clk);
            hdr_we <= 1; hdr_bank = bank; hdr_addr = addr; hdr_din = data;
            @(posedge clk);
            hdr_we <= 0;
        end
    endtask

    // Write one block's headers (6 bytes)
    task write_block_headers;
        input [5:0] block_local;
        input [3:0] q5_val;
        begin
            write_hdr(3'd0, block_local, 8'h00);
            write_hdr(3'd1, block_local, 8'h3C);
            write_hdr(3'd2, block_local, 8'hFF);
            write_hdr(3'd3, block_local, 8'hFF);
            write_hdr(3'd4, block_local, 8'hFF);
            write_hdr(3'd5, block_local, 8'hFF);
        end
    endtask

    // Write one block's qs (4 × 32-bit)
    task write_block_qs;
        input [5:0] block_local;
        input [3:0] q5_val;
        begin
            qs_word = {4{{q5_val, q5_val}}};
            for (w = 0; w < 4; w = w + 1) begin
                @(negedge clk);
                qs_we <= 1; qs_addr = w[1:0]; qs_din = qs_word;
                @(posedge clk);
                qs_we <= 0;
            end
        end
    endtask

    // Process block: write qs, start, wait
    task process_block;
        input [5:0] block_local;
        input [3:0] q5_val;
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
        end
    endtask

    task load_scales;
        integer s;
        begin
            for (s = 0; s < 8; s = s + 1) begin
                @(negedge clk);
                sc_we <= 1; sc_addr = s[2:0]; sc_din = 16'h0001;
                @(posedge clk);
                sc_we <= 0;
            end
        end
    endtask

    task load_acts;
        input [15:0] val;
        integer a;
        begin
            for (a = 0; a < 896; a = a + 1) begin
                @(negedge clk);
                act_we <= 1; act_addr = a[9:0]; act_din = val;
                @(posedge clk);
                act_we <= 0;
            end
        end
    endtask

    initial begin
        $dumpfile("tb_q5_off_by_one.vcd");
        $dumpvars(0, tb_q5_off_by_one);
        clk = 0; rst_n <= 0; start <= 0;
        hdr_we <= 0; qs_we <= 0; act_we <= 0; sc_we <= 0;
        core_id <= 0;
        $display("=== Q5_0 Off-by-One Bug Verification ===");
        $display("");

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        // Write headers for all 56 blocks
        for (b = 0; b < 56; b = b + 1)
            write_block_headers(b[5:0], b[3:0]);

        $display("Loading scales...");
        load_scales;

        $display("Loading activations (all = 1)...");
        load_acts(16'd1);

        $display("Processing 56 blocks per-block...");
        for (b = 0; b < 56; b = b + 1)
            process_block(b[5:0], b[3:0]);

        $display("");
        // Read results
        res_addr <= 0; @(posedge clk); #1;
        $display("Row 0 result = %0d", res_dout);
        if (res_dout == 5952) $display("  => NO OFF-BY-ONE BUG (matches expected 5952)");
        else if (res_dout == 5941) $display("  => OFF-BY-ONE BUG CONFIRMED (got 5941, expected 5952, loss of 11)");
        else $display("  => UNEXPECTED (got %0d, expected 5952 or 5941)", res_dout);

        res_addr <= 1; @(posedge clk); #1;
        $display("Row 1 result = %0d", res_dout);
        if (res_dout == 6464) $display("  => NO OFF-BY-ONE BUG (matches expected 6464)");
        else $display("  => Got %0d; with off-by-one would have similar loss structure", res_dout);

        #100 $finish;
    end
    initial #5000000 begin $display("TIMEOUT"); $finish; end
endmodule
