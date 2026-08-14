`timescale 1ns / 1ps

// Focused test: verify matmul_q5_0_core f16_decode sign handling directly via
// dbg_d_fp (registered f16_decode output) and dbg_d_pre (S16 d_pre).
// Feed one block with d and sample after the SETUP_D/SETUP_D3 registers latch.
module tb_q5_negd;

    reg clk = 0;
    reg rst_n = 0;
    reg core_id = 0;
    reg norm_we = 0;
    reg [1:0] norm_addr = 0;
    reg [15:0] norm_din = 0;
    reg [15:0] blk_d = 0;
    reg [31:0] blk_qh = 0;
    reg [127:0] blk_qs = 0;
    reg blk_valid = 0;
    reg act_we = 0;
    reg [9:0] act_addr = 0;
    reg [15:0] act_din = 0;
    reg clr_acc = 0;
    wire [47:0] res0, res1;
    wire done, busy;
    wire [15:0] dbg_d_pre;
    wire [31:0] dbg_d_fp;

    matmul_q5_0_core u_core (
        .clk(clk), .rst_n(rst_n), .core_id(core_id),
        .norm_we(norm_we), .norm_addr(norm_addr), .norm_din(norm_din),
        .blk_d(blk_d), .blk_qh(blk_qh), .blk_qs(blk_qs), .blk_valid(blk_valid),
        .act_we(act_we), .act_addr(act_addr), .act_din(act_din),
        .clr_acc(clr_acc),
        .res0(res0), .res1(res1), .done(done), .busy(busy),
        .dbg_d_pre(dbg_d_pre),
        .dbg_blk_counter(), .dbg_d_fp(dbg_d_fp), .dbg_norm(), .dbg_state(),
        .dbg_act_r(), .dbg_q5(), .dbg_wi()
    );

    always #5 clk = ~clk;

    integer failures = 0;

    // Feed one block and return the sampled dbg_d_fp (S24.16) and dbg_d_pre (S16).
    task check_d(input [15:0] d16, input signed [31:0] exp_fp, input signed [15:0] exp_pre);
        begin
            rst_n = 0;
            repeat (2) @(posedge clk);
            rst_n = 1;
            repeat (2) @(posedge clk);
            norm_we = 1; norm_addr = 0; norm_din = 16'h0100;  // norm = 1.0
            @(posedge clk);
            norm_we = 0;
            clr_acc = 1; @(posedge clk); clr_acc = 0;
            @(posedge clk);
            // feed one block
            blk_d = d16; blk_qh = 32'h0; blk_qs = 128'd0; blk_valid = 1;
            @(posedge clk);   // IDLE -> SETUP_D (d_fp_r latched)
            blk_valid = 0;
            @(posedge clk);   // SETUP_D -> SETUP_D2
            // dbg_d_fp is d_fp_r (latched at the end of SETUP_D, i.e. now valid)
            if ($signed(dbg_d_fp) !== exp_fp) begin
                $display("FAIL d=0x%04h dbg_d_fp=%d expect=%d", d16, $signed(dbg_d_fp), exp_fp);
                failures = failures + 1;
            end else
                $display("PASS d=0x%04h dbg_d_fp=%d", d16, $signed(dbg_d_fp));
            @(posedge clk);   // SETUP_D2 -> SETUP_D3
            @(posedge clk);   // SETUP_D3 -> SETUP_D4 (d_pre latched)
            @(posedge clk);   // extra settle
            if ($signed(dbg_d_pre) !== exp_pre) begin
                $display("FAIL d=0x%04h dbg_d_pre=%d expect=%d", d16, $signed(dbg_d_pre), exp_pre);
                failures = failures + 1;
            end else
                $display("PASS d=0x%04h dbg_d_pre=%d", d16, $signed(dbg_d_pre));
        end
    endtask

    initial begin
        $display("START tb_q5_negd");
        // d=+1.0 (0x3C00): f16_decode=+65536, d_pre=+65536*256>>8=+65536 (clamped to +32767)
        check_d(16'h3C00, 32'sd65536, 16'sd32767);
        // d=-1.0 (0xBC00): f16_decode=-65536, d_pre=-65536*256>>8=-65536 (clamped to -32768)
        check_d(16'hBC00, -32'sd65536, -16'sd32768);
        // d=+0.25 (0x3400): f16_decode=+16384, d_pre=+16384
        check_d(16'h3400, 32'sd16384, 16'sd16384);
        // d=-0.25 (0xB400): f16_decode=-16384, d_pre=-16384 (sign must hold)
        check_d(16'hB400, -32'sd16384, -16'sd16384);
        if (failures == 0) $display("ALL NEG-D TESTS PASSED");
        else $display("%0d NEG-D FAILURES", failures);
        $finish;
    end

    initial begin
        #200000;
        $display("TIMEOUT");
        $finish;
    end

endmodule
