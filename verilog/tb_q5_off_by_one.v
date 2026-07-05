`timescale 1ns / 1ps

module tb_q5_off_by_one;
    reg clk, rst_n, start;
    wire done, busy;
    reg wt_we; reg [2:0] wt_bank; reg [9:0] wt_addr; reg [7:0] wt_din;
    reg sc_we; reg [2:0] sc_addr; reg [15:0] sc_din;
    reg act_we; reg [9:0] act_addr; reg [15:0] act_din;
    reg [0:0] res_addr;
    wire [47:0] res_dout;
    reg [1:0] core_id;

    matmul_q5_0_core u_core (
        .clk(clk), .rst_n(rst_n), .start(start),
        .done(done), .busy(busy),
        .wt_we(wt_we), .wt_bank(wt_bank), .wt_addr(wt_addr), .wt_din(wt_din),
        .sc_we(sc_we), .sc_addr(sc_addr), .sc_din(sc_din),
        .act_we(act_we), .act_addr(act_addr), .act_din(act_din),
        .res_addr(res_addr), .res_dout(res_dout),
        .core_id(core_id),
        .dbg_tile_start(1'b0), .dbg_tile_cycles(), .dbg_tile_id(), .dbg_verbose(1'b0)
    );

    always #5 clk = ~clk;
    integer b, k, i;

    task write_bank;
        input [2:0] bank;
        input [9:0] addr;
        input [7:0] data;
        begin
            @(negedge clk);
            wt_we <= 1; wt_bank = bank; wt_addr = addr; wt_din = data;
            @(posedge clk);
            wt_we <= 0;
        end
    endtask

    // Load block with given q5_val (4-bit ql value for all 32 weights)
    task gen_block;
        input [5:0]  block_local;
        input [3:0]  q5_val;
        begin
            // bank0/1: f16 d = 1.0 = 0x3C00 (bank0=lo, bank1=hi)
            write_bank(3'd0, block_local, 8'h00);
            write_bank(3'd1, block_local, 8'h3C);
            // bank2..5: qh = 0xFFFFFFFF (hyper-bits all set)
            write_bank(3'd2, block_local, 8'hFF);
            write_bank(3'd3, block_local, 8'hFF);
            write_bank(3'd4, block_local, 8'hFF);
            write_bank(3'd5, block_local, 8'hFF);
            // bank6: qs array (16 bytes; {q5_val, q5_val} for both nibbles)
            for (k = 0; k < 16; k = k + 1)
                write_bank(3'd6, block_local * 16 + k, {q5_val, q5_val});
        end
    endtask

    task load_scales;
        integer s;
        begin
            // row_scale[i] = 0x0001 (UQ8.8 1/256) for all 8 rows
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
        wt_we <= 0; act_we <= 0; sc_we <= 0;
        core_id <= 0;
        $display("=== Q5_0 Off-by-One Bug Verification ===");
        $display("");
        // Test: Block k has q5_val = k mod 16 (varying per block, all 32 weights in block use same q5_val)
        // With d=1.0, scale=1/256, act=1: each weight contributes q5_val.
        // Expected row 0 sum (no bug) = 32 * sum_{k=0..27}(k mod 16)
        //   = 32 * ((0+1+...+15) + (0+1+...+11))
        //   = 32 * (120 + 66) = 32 * 186 = 5952
        // With off-by-one bug: row 0 sum = 5941 (loss of 11)

        repeat (4) @(posedge clk);
        rst_n <= 1;
        repeat (2) @(posedge clk);

        $display("Loading 56 blocks with q5_val = block mod 16...");
        for (b = 0; b < 56; b = b + 1)
            gen_block(b[5:0], b[3:0]);  // q5_val = b mod 16

        $display("Loading scales...");
        load_scales;

        $display("Loading activations (all = 1)...");
        load_acts(16'd1);

        $display("Resetting and starting...");
        rst_n <= 0; @(posedge clk); rst_n <= 1; @(posedge clk);
        repeat (2) @(posedge clk);
        start <= 1;
        @(posedge clk);
        start <= 0;
        // Wait for done
        while (!done) @(posedge clk);
        @(posedge clk);

        // Read results
        res_addr <= 0; @(posedge clk); #1;
        $display("");
        $display("Row 0 result = %0d", res_dout);
        if (res_dout == 5952) $display("  => NO OFF-BY-ONE BUG (matches expected 5952)");
        else if (res_dout == 5941) $display("  => OFF-BY-ONE BUG CONFIRMED (got 5941, expected 5952, loss of 11)");
        else $display("  => UNEXPECTED (got %0d, expected 5952 or 5941)", res_dout);

        res_addr <= 1; @(posedge clk); #1;
        $display("Row 1 result = %0d", res_dout);
        // Row 1 uses blocks 28..55, q5 per block = (b mod 16) for b=28..55
        // Sum = 32 * sum_{b=28..55}(b mod 16)
        // b mod 16 for b=28: 12, b=29: 13, ..., b=43: 11, b=44: 12, ..., b=55: 7
        // Block pattern: 28..43: 12..15, 0..11 (= sum 0..15 = 120, then 0..11=66)
        //   Wait, b=28 mod 16 = 12, b=29=13, b=30=14, b=31=15, b=32=0, ..., b=43=11; sum=66+66+66=122+... let me just direct compute
        // Actually sum for b=28..55 of (b mod 16):
        //   b=28..31: 12+13+14+15=54
        //   b=32..47: 0+1+...+15=120 (one full cycle)
        //   b=48..55: 0+1+...+7=28
        //   Total = 54+120+28 = 202
        // Expected row 1 (no bug) = 32 * 202 = 6464
        // With off-by-one bug: 1 wrong ei per block-transition = 27 losses
        //   block k=28 (no transition from block 27 via ei=896 boundary, LD sets addr=28)
        //   Actually wait: when entering row 1, the very first block in row 1 is block 28
        //   with ei=896. With off-by-one, DEC at ei=896 sees bank0[addr(895)] = block 0's data
        //   instead of block 28. So row 1's first ei=896 gets wrong data.
        //   Subsequent transitions within row 1: similar bug pattern
        //   Let me compute the predicted buggy value:
        //   31*q5(k) + q5(k-1) sum for k=29..55 (block transition points)
        //   k=29: q5(28)=12 vs q5(29)=13. Loss = 1.
        //   k=30: loss 1. k=31: loss 1. k=32: q5(31)=15 vs 0, loss -15
        //   k=33..43: loss 1 each (11 transitions).
        //   k=44: q5(43)=11 vs 12, loss 1
        //   k=45..55: loss 1 each (11 transitions)
        //   Total = ?
        //   Actually let me just report both predicted values
        if (res_dout == 6464) $display("  => NO OFF-BY-ONE BUG (matches expected 6464)");
        else $display("  => Got %0d; with off-by-one would have similar loss structure", res_dout);

        #100 $finish;
    end
    initial #5000000 begin $display("TIMEOUT"); $finish; end
endmodule
