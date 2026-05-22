// Minimal Q4K test: 64 active cols, all weights=1, verify row 0 = 64
// wmem addr = {k[5:0], g[2:0]} = k*8+g
`timescale 1ns / 1ps
module tb_minimal_q4k;
    reg clk, rst_n, start; wire done, busy;
    reg wt_we; reg [12:0] wt_addr; reg [7:0] wt_din;
    reg act_we; reg [5:0] act_addr; reg [15:0] act_din;
    reg [5:0] res_addr; wire [47:0] res_dout;
    wire sc_we = 0; wire [6:0] sc_addr = 0; wire [15:0] sc_din = 0;
    wire mode_block_load = 0; wire decode_busy;

    matmul_q4k_core uut(.clk(clk),.rst_n(rst_n),.start(start),.op_vecmul(1),.done(done),.busy(busy),
        .wt_we(wt_we),.wt_addr(wt_addr),.wt_din(wt_din),
        .sc_we(sc_we),.sc_addr(sc_addr),.sc_din(sc_din),
        .act_we(act_we),.act_addr(act_addr),.act_din(act_din),
        .res_addr(res_addr),.res_dout(res_dout),
        .mode_block_load(mode_block_load),.decode_busy(decode_busy));

    always #5 clk = ~clk;
    integer i, k;

    // Force all wmem entries to 0 (reg defaults to x)
    initial begin
        for (i = 0; i < 512; i++) begin
            uut.wmem_lo[i] = 64'd0;
            uut.wmem_hi[i] = 64'd0;
        end
    end

    initial begin
        $dumpfile("tb_minimal_q4k.vcd"); $dumpvars(0, tb_minimal_q4k);
        clk=0; rst_n=0; start=0; wt_we=0; act_we=0;
        repeat(4) @(posedge clk); rst_n=1;
        repeat(2) @(posedge clk);

        // Write wmem[entry={k,0}] INT16[0] = 1 for all 64 columns
        // entry = k*8 + 0 = k[5:0] << 3 = {k[5:0], 3'b000}
        // addr = {4'b0000 (byte_lane), 9'b{entry}} = {4'd0, k[5:0], 3'b000}
        $display("Writing weights (64 cols × 1 INT16 = 1)...");
        for (k = 0; k < 64; k++) begin
            @(negedge clk); wt_we=1; wt_din=8'd1;  wt_addr <= {4'd0, k[5:0], 3'b000};
            @(negedge clk); wt_we=1; wt_din=8'd0;  wt_addr <= {4'd1, k[5:0], 3'b000};
        end
        @(negedge clk); wt_we=0;

        // Write all 64 activations = 1
        $display("Writing acts (all 64 × 1)...");
        for (i = 0; i < 64; i++) begin
            @(negedge clk); act_we=1; act_addr=i; act_din=16'd1;
        end
        @(negedge clk); act_we=0;

        // Start
        $display("Starting compute...");
        @(negedge clk); start=1;
        @(negedge clk); start=0;

        // Wait for done
        for (i = 0; i < 2000; i++) begin
            @(posedge clk);
            if (done) begin
                $display("DONE after %0d cycles", i);
                // Read all 8 rows in group 0 (rows 0-7)
                for (k = 0; k < 8; k++) begin
                    @(negedge clk); res_addr=k;
                    @(posedge clk); #1;
                    $display("  Result[%0d] = %0d (expect %0d)", k, $signed(res_dout), k==0?64:0);
                end
                @(negedge clk); res_addr=0;
                @(posedge clk); #1;
                if ($signed(res_dout) === 64'sd64) begin
                    $display("PASS");
                end else begin
                    $display("FAIL (got %0d expected 64)", $signed(res_dout));
                end
                $finish;
            end
        end
        $display("TIMEOUT");
        $finish;
    end
endmodule
