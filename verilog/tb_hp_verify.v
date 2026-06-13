// Verify core: direct byte-by-byte write matching HP read order
`timescale 1ns / 1ps
module tb_hp_verify;
    reg clk, rst_n, start; wire done, busy;
    reg wt_we; reg [12:0] wt_addr; reg [7:0] wt_din;
    reg act_we; reg [5:0] act_addr; reg [15:0] act_din;
    reg [5:0] res_addr; wire [47:0] res_dout;
    wire sc_we = 0; wire [6:0] sc_addr = 0; wire [15:0] sc_din = 0;

    matmul_int16_core uut(.clk(clk),.rst_n(rst_n),.start(start),.op_vecmul(1),.done(done),.busy(busy),
        .wt_we(wt_we),.wt_addr(wt_addr),.wt_din(wt_din),
        .sc_we(sc_we),.sc_addr(sc_addr),.sc_din(sc_din),
        .act_we(act_we),.act_addr(act_addr),.act_din(act_din),
        .res_addr(res_addr),.res_dout(res_dout));

    always #5 clk = ~clk;
    integer k, g, wi, w, i, b;
    reg [47:0] golden;
    reg [15:0] val1, val2;
    reg [7:0] data [0:8191]; // DDR contents (same as hp_test writes)

    // Fill DDR array with hp_test weight data (NEW formula)
    initial begin
        for (k = 0; k < 64; k++)
            for (g = 0; g < 8; g++)
                for (wi = 0; wi < 4; wi++) begin
                    w = k*32 + g*4 + wi;
                    val1 = k*64 + (g*8+wi*2);
                    val2 = k*64 + (g*8+wi*2+1);
                    // 32-bit entry in little-endian bytes
                    data[w*4+0] = val1[7:0];
                    data[w*4+1] = val1[15:8];
                    data[w*4+2] = val2[7:0];
                    data[w*4+3] = val2[15:8];
                end

        // Compute golden: acc[out] = Σ(k+1) * (k*64+out)
        golden = 0;
        for (k = 0; k < 64; k++)
            golden = golden + (k+1) * (k*64 + 0);
        $display("Golden[0] = %0d (0x%h)", golden, golden);
    end

    initial begin
        $dumpfile("tb_hp_verify.vcd"); $dumpvars(0, tb_hp_verify);
        clk=0; rst_n=0; start=0; wt_we=0; act_we=0;
        repeat(4) @(posedge clk); rst_n=1;
        repeat(2) @(posedge clk);

        // Write weights via byte stream (matching HP read order)
        $display("Writing weights (8192 bytes, HP order)...");
        for (b = 0; b < 8192; b++) begin
            @(negedge clk);
            wt_we = 1;
            wt_addr = {b[3:0], b[12:4]};  // swizzle matches FSM
            wt_din = data[b];
        end
        @(negedge clk); wt_we = 0;
        $display("Weights done");

        // Write acts: act[k] = k+1
        $display("Writing acts...");
        for (i = 0; i < 64; i++) begin
            @(negedge clk); act_we=1; act_addr=i; act_din=i+1;
        end
        @(negedge clk); act_we=0;

        // Compute
        $display("Starting compute...");
        @(negedge clk); start=1;
        @(negedge clk); start=0;
        wait(done);
        @(posedge clk);

        // Check result[0]
        res_addr = 0; @(negedge clk);
        $display("Result[0] = %0d (0x%h) = %0d in lower 32 bits", res_dout, res_dout, res_dout[31:0]);
        if (res_dout[31:0] === golden[31:0])
            $display("PASS: result[0] matches golden");
        else
            $display("FAIL: result[0] = 0x%h, expected lower 32 = 0x%h", res_dout[31:0], golden[31:0]);

        // Check result[1]
        res_addr = 1; @(negedge clk);
        $display("Result[1] = %0d (0x%h)", res_dout, res_dout);

        $display("\n=== VERIFICATION DONE ===");
        $finish;
    end
endmodule
