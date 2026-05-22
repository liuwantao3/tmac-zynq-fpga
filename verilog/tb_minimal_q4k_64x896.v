`timescale 1ns / 1ps

module tb_minimal_q4k_64x896;

    reg         clk = 0;
    reg         rst_n = 0;
    reg         start = 0;
    wire        done;
    wire        busy;

    reg         wt_we = 0;
    reg [12:0]  wt_addr = 0;
    reg  [7:0]   wt_din = 0;
    reg         act_we = 0;
    reg [5:0]   act_addr = 0;
    reg [15:0]  act_din = 0;
    reg [5:0]   res_addr = 0;
    wire [47:0] res_dout;

    reg         sc_we = 0;
    reg [9:0]   sc_addr = 0;
    reg [15:0]  sc_din = 0;

    wire        mode_block_load = 1'b1;
    wire        decode_busy;

    matmul_q4k_64x896_core u_core (
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

    integer errors = 0;
    integer run_count = 0;

    task init_single_block;
        integer base;
        begin
            base = 0;
            u_core.block_buf[base+0] = 8'h00;
            u_core.block_buf[base+1] = 8'h3C;  // d = 1.0
            u_core.block_buf[base+2] = 8'h00;
            u_core.block_buf[base+3] = 8'h00;  // dmin = 0
            u_core.block_buf[base+4] = 1;  // scale[0]=1
            u_core.block_buf[base+5] = 1;
            u_core.block_buf[base+6] = 1;
            u_core.block_buf[base+7] = 1;
            u_core.block_buf[base+8] = 0;  // min[0]=0
            u_core.block_buf[base+9] = 0;
            u_core.block_buf[base+10] = 0;
            u_core.block_buf[base+11] = 0;
            u_core.block_buf[base+16] = 1;  // q4 values = 1
            u_core.block_buf[base+17] = 1;
            u_core.block_buf[base+18] = 1;
            u_core.block_buf[base+19] = 1;
            u_core.block_buf[base+20] = 1;
            u_core.block_buf[base+21] = 1;
            u_core.block_buf[base+22] = 1;
            u_core.block_buf[base+23] = 1;
        end
    endtask

    task reset_all;
        integer i;
        begin
            rst_n = 0; #20;
            rst_n = 1; #10;
            for (i = 0; i < 512; i = i + 1) begin
                u_core.wmem_lo[i] = 0;
                u_core.wmem_hi[i] = 0;
            end
        end
    endtask

    task test_decode_only;
        begin
            $display("Test 1: Decode only (single block)");
            init_single_block();
            start = 1; #10; start = 0;
            
            wait(done);
            $display("  DONE at cycle %d", run_count);
            run_count = run_count + 1;
        end
    endtask

    initial begin
        $dumpfile("tb_minimal.vcd");
        $dumpvars(0, tb_minimal_q4k_64x896);
        
        #5 rst_n = 1; #10;
        init_single_block();
        
        $display("=== Running tests ===");
        
        test_decode_only();
        
        $display("\n=== Summary ===");
        $finish;
    end

endmodule