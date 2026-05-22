module top_tb;
    reg clk = 0;
    reg rst_n = 0;
    reg start;
    wire done, busy;
    
    reg wt_we = 0;
    reg [12:0] wt_addr;
    reg [7:0] wt_din;
    reg act_we = 0;
    reg [5:0] act_addr;
    reg [15:0] act_din;
    reg sc_we = 0;
    reg [9:0] sc_addr;
    reg [15:0] sc_din;
    reg [5:0] res_addr;
    wire [47:0] res_dout;
    wire mode_block_load = 1;
    wire decode_busy;
    
    matmul_q4k_64x896_core u_core (
        .clk, .rst_n, .start, .op_vecmul(1), .done, .busy,
        .wt_we, .wt_addr, .wt_din,
        .sc_we, .sc_addr, .sc_din,
        .act_we, .act_addr, .act_din,
        .res_addr, .res_dout,
        .mode_block_load, .decode_busy
    );
    
    always #5 clk = ~clk;
    
    integer i;
    
    task load_block;
        input [3:0] q4_const;
        input [5:0] sc_val;
        input [5:0] m_val;
        input [7:0] block_idx;
        reg [7:0] lo, hi;
        integer base;
        begin
            base = block_idx * 144;
            u_core.block_buf[base+0] = 8'h00;  // d = 1.0 f16
            u_core.block_buf[base+1] = 8'h3C;
            u_core.block_buf[base+2] = 8'h00;  // dmin = 0
            u_core.block_buf[base+3] = 8'h00;
            for (i = 0; i < 8; i = i + 1) begin
                u_core.block_buf[base + 4 + i] = sc_val;
                u_core.block_buf[base + 8 + i] = m_val;
            end
            for (i = 0; i < 256; i = i + 1) begin
                u_core.block_buf[base + 16 + i] = q4_const;
            end
        end
    endtask
    
    initial begin
        $dumpfile("top.vcd");
        $dumpvars;
        
        rst_n = 0;
        #15 rst_n = 1;
        #10;
        
        $display("Loading blocks...");
        for (i = 0; i < 224; i = i + 1) load_block(4'd1, 1, 0, i);
        $display("Loading scale...");
        for (i = 0; i < 64; i = i + 1) begin sc_we = 1; sc_addr = i; sc_din = 256; @(posedge clk); end
        sc_we = 0; @(posedge clk);
        $display("Loading acts...");
        for (i = 0; i < 896; i = i + 1) begin act_we = 1; act_addr = i; act_din = 1; @(posedge clk); end
        act_we = 0; @(posedge clk);
        
        $display("Starting...");
        start = 1; @(posedge clk); start = 0;
        
        wait(done);
        @(posedge clk);
        
        $display("Checking result[0]...");
        res_addr = 0; @(posedge clk);
        #1;
        $display("Result[0] = %d", res_dout);
        $finish;
    end
endmodule