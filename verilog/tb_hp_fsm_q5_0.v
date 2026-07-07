`timescale 1ns / 1ps
module tb_hp_fsm_q5_0;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    reg  [15:0] axil_awaddr; reg axil_awvalid; wire axil_awready;
    reg  [31:0] axil_wdata;  reg [3:0] axil_wstrb;
    reg  axil_wvalid; wire axil_wready;
    wire [1:0]  axil_bresp; wire axil_bvalid; reg axil_bready;
    reg  [15:0] axil_araddr; reg axil_arvalid; wire axil_arready;
    wire [31:0] axil_rdata; wire [1:0] axil_rresp;
    wire axil_rvalid; reg axil_rready;

    wire [5:0]  m_axi_awid;     wire [31:0] m_axi_awaddr;
    wire        m_axi_awvalid, m_axi_awready;
    wire [7:0]  m_axi_awlen;    wire [2:0]  m_axi_awsize, m_axi_arsize;
    wire [1:0]  m_axi_awburst, m_axi_arburst;
    wire [63:0] m_axi_wdata;    wire m_axi_wvalid, m_axi_wready, m_axi_wlast;
    wire [7:0]  m_axi_wstrb;    wire [5:0]  m_axi_wid;
    wire        m_axi_bvalid, m_axi_bready;
    wire [1:0]  m_axi_bresp;    wire [5:0]  m_axi_bid;
    wire [5:0]  m_axi_arid;     wire [31:0] m_axi_araddr;
    wire        m_axi_arvalid, m_axi_arready;
    wire [7:0]  m_axi_arlen;
    wire [63:0] m_axi_rdata;    wire [1:0]  m_axi_rresp;
    wire [5:0]  m_axi_rid;      wire m_axi_rvalid, m_axi_rready, m_axi_rlast;

    reg [63:0] ddr_mem [0:524287];

    task ddr_write32(input [31:0] addr, input [31:0] val);
        if (addr[2]) ddr_mem[addr[31:3]][63:32] = val;
        else         ddr_mem[addr[31:3]][31:0] = val;
    endtask
    function [31:0] ddr_read32(input [31:0] addr);
        if (addr[2]) ddr_read32 = ddr_mem[addr[31:3]][63:32];
        else         ddr_read32 = ddr_mem[addr[31:3]][31:0];
    endfunction
    task ddr_write8(input [31:0] addr, input [7:0] data);
        reg [31:0] wa; reg [31:0] wv;
        wa = addr & 32'hFFFFFFFC;
        wv = ddr_read32(wa);
        case (addr[1:0])
            0: wv[7:0]   = data;
            1: wv[15:8]  = data;
            2: wv[23:16] = data;
            3: wv[31:24] = data;
        endcase
        ddr_write32(wa, wv);
    endtask

    // HP write model
    localparam WR_IDLE = 0, WR_WRITE = 1, WR_WAIT_B = 2;
    reg [1:0] wr_state;
    reg [31:0] wr_addr_cur;
    reg [7:0]  wr_beats_rem;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin wr_state <= WR_IDLE; wr_addr_cur <= 0; wr_beats_rem <= 0; end
        else begin
            case (wr_state)
                WR_IDLE: if (m_axi_awvalid) begin
                    wr_addr_cur <= m_axi_awaddr; wr_beats_rem <= m_axi_awlen; wr_state <= WR_WRITE;
                end
                WR_WRITE: if (m_axi_wvalid && m_axi_wready) begin
                    if (m_axi_wstrb[3:0]) ddr_mem[wr_addr_cur[31:3]][31:0] <= m_axi_wdata[31:0];
                    if (m_axi_wstrb[7:4]) ddr_mem[wr_addr_cur[31:3]][63:32] <= m_axi_wdata[63:32];
                    if (m_axi_wlast) wr_state <= WR_WAIT_B;
                    else begin wr_addr_cur <= wr_addr_cur + 8; wr_beats_rem <= wr_beats_rem - 1; end
                end
                WR_WAIT_B: if (m_axi_bready) wr_state <= WR_IDLE;
            endcase
        end
    end

    // HP read model
    reg [7:0]  rd_beats_done, rd_beats_total;
    reg [31:0] rd_addr_base;
    reg        rd_busy;
    wire [31:0] rd_idx  = (rd_addr_base[31:2] + rd_beats_done);
    wire [31:0] rd_half = rd_idx[0] ? ddr_mem[rd_idx >> 1][63:32] : ddr_mem[rd_idx >> 1][31:0];
    assign m_axi_awready = (wr_state == WR_IDLE);
    assign m_axi_wready  = (wr_state == WR_WRITE);
    assign m_axi_bvalid  = (wr_state == WR_WAIT_B);
    assign m_axi_bresp   = 2'd0; assign m_axi_bid = 6'd0;
    assign m_axi_arready = !rd_busy;
    assign m_axi_rdata   = {32'd0, rd_half};
    assign m_axi_rvalid  = rd_busy;
    assign m_axi_rresp   = 2'd0; assign m_axi_rid = 6'd0;
    assign m_axi_rlast   = rd_busy && (rd_beats_done == rd_beats_total);
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_busy <= 0; rd_beats_done <= 0; rd_beats_total <= 0; rd_addr_base <= 0;
        end else begin
            if (!rd_busy && m_axi_arvalid) begin
                rd_addr_base <= m_axi_araddr; rd_beats_done <= 0;
                rd_beats_total <= m_axi_arlen; rd_busy <= 1;
            end
            if (rd_busy && m_axi_rready) begin
                if (rd_beats_done == rd_beats_total) rd_busy <= 0;
                else rd_beats_done <= rd_beats_done + 1;
            end
        end
    end

    hp_fsm_top uut (
        .clk(clk), .rst_n(rst_n),
        .s_axil_awaddr(axil_awaddr), .s_axil_awvalid(axil_awvalid), .s_axil_awready(axil_awready),
        .s_axil_wdata(axil_wdata), .s_axil_wstrb(axil_wstrb),
        .s_axil_wvalid(axil_wvalid), .s_axil_wready(axil_wready),
        .s_axil_bresp(axil_bresp), .s_axil_bvalid(axil_bvalid), .s_axil_bready(axil_bready),
        .s_axil_araddr(axil_araddr), .s_axil_arvalid(axil_arvalid), .s_axil_arready(axil_arready),
        .s_axil_rdata(axil_rdata), .s_axil_rresp(axil_rresp),
        .s_axil_rvalid(axil_rvalid), .s_axil_rready(axil_rready),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr), .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_awlock(), .m_axi_awcache(), .m_axi_awprot(),
        .m_axi_wdata(m_axi_wdata), .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_wlast(m_axi_wlast), .m_axi_wstrb(m_axi_wstrb), .m_axi_wid(m_axi_wid),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready), .m_axi_bresp(m_axi_bresp), .m_axi_bid(m_axi_bid),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr), .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_arlock(), .m_axi_arcache(), .m_axi_arprot(),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp), .m_axi_rid(m_axi_rid),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready), .m_axi_rlast(m_axi_rlast)
    );

    reg [31:0] rd_val;
    integer i, j, fail_count, test_num;

    task axil_write(input [15:0] addr, input [31:0] data);
        @(negedge clk);
        axil_awaddr = addr; axil_awvalid = 1;
        axil_wdata = data; axil_wvalid = 1; axil_wstrb = 4'hF;
        axil_bready = 1;
        @(negedge clk);
        while (!axil_bvalid) @(negedge clk);
        @(negedge clk);
        axil_awaddr = 0; axil_awvalid = 0;
        axil_wdata = 0; axil_wvalid = 0; axil_wstrb = 0;
        axil_bready = 0;
    endtask

    task axil_read(input [15:0] addr, output [31:0] data);
        @(negedge clk);
        axil_araddr = addr; axil_arvalid = 1; axil_rready = 1;
        @(negedge clk);
        while (!axil_rvalid) @(negedge clk);
        data = axil_rdata;
        @(negedge clk);
        axil_araddr = 0; axil_arvalid = 0; axil_rready = 0;
    endtask

    // --- Q5_0 descriptor setup ---
    task setup_q5_desc(input [31:0] desc_addr, input [31:0] next_addr,
                       input [31:0] weight_addr, input [31:0] act_addr,
                       input [31:0] res_addr, input [23:0] act_bytes,
                       input [15:0] num_tiles);
        ddr_write32(desc_addr + 0,  next_addr);
        ddr_write32(desc_addr + 4,  weight_addr);
        ddr_write32(desc_addr + 8,  act_addr);
        ddr_write32(desc_addr + 12, res_addr);
        ddr_write32(desc_addr + 16, 32'h00000001);  // tensor_type=1 (Q5_0)
        ddr_write32(desc_addr + 20, {num_tiles[15:8], num_tiles[7:0], 8'h00, 8'h00});
        ddr_write32(desc_addr + 24, {8'h00, act_bytes});
        ddr_write32(desc_addr + 28, 32'h00000000);
    endtask

    // --- CPU_OP descriptor setup ---
    task setup_cpu_desc(input [31:0] desc_addr, input [31:0] next_addr,
                        input [31:0] act_addr, input [31:0] res_addr,
                        input [23:0] act_bytes);
        ddr_write32(desc_addr + 0,  next_addr);
        ddr_write32(desc_addr + 4,  0);              // weight_addr = 0 (unused)
        ddr_write32(desc_addr + 8,  act_addr);
        ddr_write32(desc_addr + 12, res_addr);
        ddr_write32(desc_addr + 16, 32'h0000000F);  // tensor_type=15 (CPU_OP)
        ddr_write32(desc_addr + 20, 32'h00000000);
        ddr_write32(desc_addr + 24, {8'h00, act_bytes});
        ddr_write32(desc_addr + 28, 32'h00000000);
    endtask

    task start_chain(input [31:0] desc_base);
        axil_write(16'h18, desc_base);
        axil_write(16'h1C, 32'd1);
        axil_write(16'h00, 32'd1);
    endtask

    task wait_done;
        input [31:0] expected_head;
        integer timeout;
        reg completed;
        reg [31:0] head_val;
        timeout = 0; completed = 0;
        while (timeout < 1000000) begin
            axil_read(16'h20, head_val);
            if (head_val >= expected_head) begin completed = 1; timeout = 1000000; end
            else begin timeout = timeout + 1; #10; end
        end
        if (!completed) $display("  WARN: wait_done timeout");
    endtask

    // Write Q5_0 weight data in per-block layout:
    //   Per block (48 bytes): core0_d[2] + qh[4] + qs[16] + core1_d[2] + qh[4] + qs[16] + pad[4]
    //   56 blocks × 48 = 2688 bytes, then 4 × UQ8.8 norm values at offset 2688
    task fill_q5_weight(input [31:0] base, input [3:0] q5_val);
        integer b, k;
        begin
            for (b = 0; b < 56; b = b + 1) begin
                // Core0 d = 0x3C00 (f16 1.0)
                ddr_write8(base + b*48 + 0, 8'h00);
                ddr_write8(base + b*48 + 1, 8'h3C);
                // Core0 qh = 0xFFFFFFFF (all high bits = 1)
                ddr_write8(base + b*48 + 2, 8'hFF);
                ddr_write8(base + b*48 + 3, 8'hFF);
                ddr_write8(base + b*48 + 4, 8'hFF);
                ddr_write8(base + b*48 + 5, 8'hFF);
                // Core0 qs: each nibble = q5_val
                for (k = 0; k < 16; k = k + 1)
                    ddr_write8(base + b*48 + 6 + k, {q5_val, q5_val});
                // Core1 d = 0x3C00
                ddr_write8(base + b*48 + 22, 8'h00);
                ddr_write8(base + b*48 + 23, 8'h3C);
                // Core1 qh = 0xFFFFFFFF
                ddr_write8(base + b*48 + 24, 8'hFF);
                ddr_write8(base + b*48 + 25, 8'hFF);
                ddr_write8(base + b*48 + 26, 8'hFF);
                ddr_write8(base + b*48 + 27, 8'hFF);
                // Core1 qs: each nibble = q5_val
                for (k = 0; k < 16; k = k + 1)
                    ddr_write8(base + b*48 + 28 + k, {q5_val, q5_val});
                // Bytes 44-47: padding (unused)
            end
        end
    endtask

    // Fill 4 row_norm values at weight_base + 2688
    // norm = 0x0100 (UQ8.8 = 1.0). Little-endian: byte0=0x00, byte1=0x01 gives 0x0100
    task fill_q5_scales(input [31:0] weight_base);
        integer s;
        begin
            for (s = 0; s < 4; s = s + 1) begin
                ddr_write8(weight_base + 2688 + s*2, 8'd0);
                ddr_write8(weight_base + 2688 + s*2 + 1, 8'd1);
            end
        end
    endtask

    // Fill 896 x int16 activations
    task fill_q5_acts(input [31:0] base, input [15:0] val);
        integer a;
        begin
            for (a = 0; a < 896; a = a + 1) begin
                ddr_write8(base + a*2, val[7:0]);
                ddr_write8(base + a*2 + 1, val[15:8]);
            end
        end
    endtask

    // Verify 4 rows of Q5_0 result at res_addr
    task verify_q5_result(input [31:0] res_addr, input [47:0] expected, input integer tn);
        integer r;
        reg [31:0] lo, hi;
        reg signed [47:0] got;
        begin
            for (r = 0; r < 4; r = r + 1) begin
                lo = ddr_read32(res_addr + r*8);
                hi = ddr_read32(res_addr + r*8 + 4);
                got = {hi[15:0], lo};
                if (got == expected) begin
                    $display("  Row %0d: PASS (got %0d)", r, got);
                end else begin
                    $display("  FAIL[%0d]: row %0d got %0d expected %0d", tn, r, got, expected);
                    fail_count = fail_count + 1;
                end
            end
        end
    endtask

    // Zero 32 bytes (4 rows x 8 bytes)
    task zero_res(input [31:0] addr);
        integer z;
        begin
            for (z = 0; z < 8; z = z + 1)
                ddr_write32(addr + z*4, 32'h00000000);
        end
    endtask

    initial begin
        $dumpfile("tb_hp_fsm_q5_0.vcd");
        $dumpvars(0, tb_hp_fsm_q5_0);
        $dumpon;
        $display("==============================================");
        $display("  HP FSM Q5_0 Multi-Descriptor Tests");
        $display("==============================================");
        #12 rst_n = 1; #20;
        fail_count = 0; test_num = 0;

        // =================================================================
        // Test 1: Single Q5_0 descriptor, all-1s -> expect 896 per row
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Single Q5_0 all-1s (expect each row=229376) ---", test_num);
        setup_q5_desc(32'h00300000, 32'h00000000,
                      32'h00310000, 32'h00320000, 32'h00330000, 1792, 16'd1);
        fill_q5_weight(32'h00310000, 4'd1);
        fill_q5_scales(32'h00310000);
        fill_q5_acts(32'h00320000, 16'd1);
        zero_res(32'h00330000);
        start_chain(32'h00300000);
        wait_done(1);
        axil_read(16'h14, rd_val); $display("  STATUS=0x%08x", rd_val);
        axil_read(16'h28, rd_val); $display("  DEBUG=0x%08x", rd_val);
        axil_read(16'h20, rd_val); $display("  HEAD=%d", rd_val);
        verify_q5_result(32'h00330000, 229376, test_num);

        // =================================================================
        // Test 2: Chain of 2 Q5_0 descriptors
        //   Desc 0: all-1s -> expect 896 per row
        //   Desc 1: all-0s -> expect 0 per row
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Chain of 2 Q5_0 (all-1s, then all-0s) ---", test_num);
        setup_q5_desc(32'h00300020, 32'h00300040,   // desc 0 -> desc 1
                      32'h00310000, 32'h00320000, 32'h00330040, 1792, 16'd1);
        setup_q5_desc(32'h00300040, 32'h00000000,   // desc 1 -> end
                      32'h00313000, 32'h00322000, 32'h00330080, 1792, 16'd1);
        fill_q5_weight(32'h00310000, 4'd1);   // Desc 0: weights=1 -> expect 229376
        fill_q5_scales(32'h00310000);
        fill_q5_weight(32'h00313000, 4'd0);   // Desc 1: weights=0 -> expect 0
        fill_q5_scales(32'h00313000);
        fill_q5_acts(32'h00320000, 16'd1);    // Desc 0 acts = 1
        fill_q5_acts(32'h00322000, 16'd1);    // Desc 1 acts = 1
        zero_res(32'h00330040);
        zero_res(32'h00330080);
        start_chain(32'h00300020);
        wait_done(2);
        axil_read(16'h14, rd_val); $display("  STATUS=0x%08x", rd_val);
        axil_read(16'h20, rd_val); $display("  HEAD=%d (expect 2)", rd_val);
        $display("  Verifying Desc 0 (weights=1, expect 229376):");
        verify_q5_result(32'h00330040, 229376, test_num);
        $display("  Verifying Desc 1 (weights=0, expect 0):");
        verify_q5_result(32'h00330080, 0, test_num);

        // =================================================================
        // Test 3: Mixed chain CPU_OP -> Q5_0
        //   Desc 0: CPU_OP passthrough 64 bytes
        //   Desc 1: Q5_0 all-1s -> expect 896
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Mixed CPU_OP -> Q5_0 ---", test_num);
        // CPU_OP: act=pattern, res=0x00340000 (passthrough)
        setup_cpu_desc(32'h00300060, 32'h00300080,
                       32'h00340000, 32'h00340040, 64);
        // Q5_0: weights=1, acts=1 -> 896
        setup_q5_desc(32'h00300080, 32'h00000000,
                      32'h00314000, 32'h00323000, 32'h003300C0, 1792, 16'd1);
        // CPU_OP pattern: incrementing 64 bytes (0x00..0x3F)
        for (j = 0; j < 16; j = j + 1)
            ddr_write32(32'h00340000 + j*4,
                (4*j+0) + ((4*j+1) << 8) + ((4*j+2) << 16) + ((4*j+3) << 24));
        zero_res(32'h00340040);
        $display("  Writing weight/act for Q5_0 desc 1...");
        fill_q5_weight(32'h00314000, 4'd1);
        fill_q5_scales(32'h00314000);
        fill_q5_acts(32'h00323000, 16'd1);
        zero_res(32'h003300C0);

        start_chain(32'h00300060);
        wait_done(2);
        axil_read(16'h14, rd_val); $display("  STATUS=0x%08x", rd_val);
        axil_read(16'h20, rd_val); $display("  HEAD=%d (expect 2)", rd_val);
        // Verify CPU_OP passthrough
        $display("  Verifying CPU_OP passthrough (64 bytes):");
        for (j = 0; j < 16; j = j + 1) begin
            reg [31:0] expected;
            expected = (4*j+0) + ((4*j+1) << 8) + ((4*j+2) << 16) + ((4*j+3) << 24);
            rd_val = ddr_read32(32'h00340040 + j*4);
            if (rd_val == expected)
                $display("    Word %0d: PASS (0x%08x)", j, rd_val);
            else begin
                $display("    FAIL[%0d]: word %0d got 0x%08x expected 0x%08x", test_num, j, rd_val, expected);
                fail_count = fail_count + 1;
            end
        end
        // Verify Q5_0 result
        $display("  Verifying Q5_0 desc 1 (expect 229376 per row):");
        verify_q5_result(32'h003300C0, 229376, test_num);

        // =================================================================
        // Test 4: Chain of 3: Q5_0 -> CPU_OP -> Q5_0
        //   Desc 0: Q5_0 all-1s -> 896
        //   Desc 1: CPU_OP passthrough 64 bytes
        //   Desc 2: Q5_0 all-2s -> 3584
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Chain of 3 (Q5_0 -> CPU_OP -> Q5_0) ---", test_num);
        setup_q5_desc(32'h003000A0, 32'h003000C0,   // desc 0 -> desc 1
                      32'h00315000, 32'h00324000, 32'h00330100, 1792, 16'd1);
        setup_cpu_desc(32'h003000C0, 32'h003000E0,   // desc 1 -> desc 2
                       32'h00340100, 32'h00340140, 64);
        setup_q5_desc(32'h003000E0, 32'h00000000,   // desc 2 -> end
                      32'h00317000, 32'h00325000, 32'h00330180, 1792, 16'd1);
        // Desc 0: Q5_0 weights=1, acts=1 -> 896
        fill_q5_weight(32'h00315000, 4'd1);
        fill_q5_scales(32'h00315000);
        fill_q5_acts(32'h00324000, 16'd1);
        zero_res(32'h00330100);
        // Desc 1: CPU_OP incrementing pattern
        for (j = 0; j < 16; j = j + 1)
            ddr_write32(32'h00340100 + j*4, (4*j+0) + ((4*j+1) << 8) + ((4*j+2) << 16) + ((4*j+3) << 24));
        zero_res(32'h00340140);
        // Desc 2: Q5_0 weights=2, acts=2 -> 3584
        fill_q5_weight(32'h00317000, 4'd2);
        fill_q5_scales(32'h00317000);
        fill_q5_acts(32'h00325000, 16'd2);
        zero_res(32'h00330180);

        start_chain(32'h003000A0);
        wait_done(3);
        axil_read(16'h14, rd_val); $display("  STATUS=0x%08x", rd_val);
        axil_read(16'h20, rd_val); $display("  HEAD=%d (expect 3)", rd_val);
        $display("  Verifying Q5_0 desc 0 (weights=1, expect 229376):");
        verify_q5_result(32'h00330100, 229376, test_num);
        $display("  Verifying CPU_OP desc 1 passthrough (64 bytes):");
        for (j = 0; j < 16; j = j + 1) begin
            reg [31:0] expected;
            expected = (4*j+0) + ((4*j+1) << 8) + ((4*j+2) << 16) + ((4*j+3) << 24);
            rd_val = ddr_read32(32'h00340140 + j*4);
            if (rd_val == expected)
                $display("    Word %0d: PASS (0x%08x)", j, rd_val);
            else begin
                $display("    FAIL[%0d]: word %0d got 0x%08x expected 0x%08x", test_num, j, rd_val, expected);
                fail_count = fail_count + 1;
            end
        end
        $display("  Verifying Q5_0 desc 2 (weights=2, acts=2, expect 917504):");
        verify_q5_result(32'h00330180, 917504, test_num);

        // =================================================================
        // Test 5: Zero activations — non-zero weights with all-act=0
        //   Expect: all 4 rows = 0 (accumulator never changes from 0)
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Zero activations (expect all rows=0) ---", test_num);
        setup_q5_desc(32'h003000A0, 32'h00000000,
                      32'h00318000, 32'h00326000, 32'h00330200, 1792, 16'd1);
        fill_q5_weight(32'h00318000, 4'd1);
        fill_q5_scales(32'h00318000);
        fill_q5_acts(32'h00326000, 16'd0);    // act = 0
        zero_res(32'h00330200);
        start_chain(32'h003000A0);
        wait_done(1);
        verify_q5_result(32'h00330200, 0, test_num);

        // =================================================================
        // Test 6: Back-to-back independent chains (restart from DONE)
        //   Chain A: all-1s → 229376
        //   Chain B: all-1s → 229376 (same weight, new descriptor)
        //   Verifies FSM properly re-enters from DONE state
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Back-to-back restart (two chains) ---", test_num);
        // Chain A
        setup_q5_desc(32'h003000C0, 32'h00000000,
                      32'h00319000, 32'h00327000, 32'h00330280, 1792, 16'd1);
        fill_q5_weight(32'h00319000, 4'd1);
        fill_q5_scales(32'h00319000);
        fill_q5_acts(32'h00327000, 16'd1);
        zero_res(32'h00330280);
        start_chain(32'h003000C0);
        wait_done(1);
        $display("  Chain A result:");
        verify_q5_result(32'h00330280, 229376, test_num);
        // Chain B — different descriptor, same pattern
        setup_q5_desc(32'h003000E0, 32'h00000000,
                      32'h00319000, 32'h00327000, 32'h00330300, 1792, 16'd1);
        zero_res(32'h00330300);
        start_chain(32'h003000E0);
        wait_done(1);
        $display("  Chain B result:");
        verify_q5_result(32'h00330300, 229376, test_num);

        // =================================================================
        // Test 7: Long chain (4 descriptors: Q5_0 → Q5_0 → Q5_0 → Q5_0)
        //   Each uses different weight (1, 2, 0, 1) with uniform acts=1
        //   Tests chain depth and varied results
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Chain of 4 Q5_0 ---", test_num);
        // Desc 0 → 1
        setup_q5_desc(32'h00300100, 32'h00300120,
                      32'h0031A000, 32'h00328000, 32'h00330380, 1792, 16'd1);
        // Desc 1 → 2
        setup_q5_desc(32'h00300120, 32'h00300140,
                      32'h0031B000, 32'h00329000, 32'h00330400, 1792, 16'd1);
        // Desc 2 → 3
        setup_q5_desc(32'h00300140, 32'h00300160,
                      32'h0031C000, 32'h0032A000, 32'h00330480, 1792, 16'd1);
        // Desc 3 → end
        setup_q5_desc(32'h00300160, 32'h00000000,
                      32'h0031D000, 32'h0032B000, 32'h00330500, 1792, 16'd1);
        fill_q5_weight(32'h0031A000, 4'd1);  // desc 0: weight=1 → 229376
        fill_q5_scales(32'h0031A000);
        fill_q5_weight(32'h0031B000, 4'd2);  // desc 1: weight=2 → ?
        fill_q5_scales(32'h0031B000);
        fill_q5_weight(32'h0031C000, 4'd0);  // desc 2: weight=0 → 0
        fill_q5_scales(32'h0031C000);
        fill_q5_weight(32'h0031D000, 4'd1);  // desc 3: weight=1 → 229376
        fill_q5_scales(32'h0031D000);
        fill_q5_acts(32'h00328000, 16'd1);
        fill_q5_acts(32'h00329000, 16'd1);
        fill_q5_acts(32'h0032A000, 16'd1);
        fill_q5_acts(32'h0032B000, 16'd1);
        zero_res(32'h00330380);
        zero_res(32'h00330400);
        zero_res(32'h00330480);
        zero_res(32'h00330500);
        start_chain(32'h00300100);
        wait_done(4);
        axil_read(16'h20, rd_val);
        $display("  HEAD=%d (expect 4)", rd_val);
        $display("  Desc 0 (weights=1):");
        verify_q5_result(32'h00330380, 229376, test_num);
        // Desc 1: q5=2 → dq=512 → per block 32*512 → per row 28*16384=458752
        $display("  Desc 1 (weights=2):");
        verify_q5_result(32'h00330400, 458752, test_num);
        $display("  Desc 2 (weights=0):");
        verify_q5_result(32'h00330480, 0, test_num);
        $display("  Desc 3 (weights=1):");
        verify_q5_result(32'h00330500, 229376, test_num);

        // =================================================================
        // Test 8: Negative activations (act=-1, weights=1)
        //   q5=1, d_pre=256 → dq=256 → prod = 256*(-1) = -256
        //   Per block: 32 * (-256) = -8192. Per row: 28 * -8192 = -229376
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Negative activations (expect all rows=-229376) ---", test_num);
        setup_q5_desc(32'h00300180, 32'h00000000,
                      32'h0031E000, 32'h0032C000, 32'h00330600, 1792, 16'd1);
        fill_q5_weight(32'h0031E000, 4'd1);
        fill_q5_scales(32'h0031E000);
        fill_q5_acts(32'h0032C000, 16'hFFFF);   // act = -1
        zero_res(32'h00330600);
        start_chain(32'h00300180);
        wait_done(1);
        verify_q5_result(32'h00330600, -229376, test_num);

        // =================================================================
        // Test 9: 5-descriptor chain (Q5_0 → CPU_OP → Q5_0 → CPU_OP → Q5_0)
        //   Desc 0: Q5_0 weights=1, acts=1 → 229376
        //   Desc 1: CPU_OP passthrough 128 bytes
        //   Desc 2: Q5_0 weights=0, acts=1 → 0
        //   Desc 3: CPU_OP passthrough 128 bytes
        //   Desc 4: Q5_0 weights=1, acts=1 → 229376
        //   Tests deep mixed chain
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: 5-desc mixed chain ---", test_num);
        setup_q5_desc(32'h003001A0, 32'h003001C0,   // desc 0 → 1
                      32'h0031F000, 32'h0032D000, 32'h00330700, 1792, 16'd1);
        setup_cpu_desc(32'h003001C0, 32'h003001E0,   // desc 1 → 2
                       32'h00340000, 32'h00340080, 128);
        setup_q5_desc(32'h003001E0, 32'h00300200,   // desc 2 → 3
                      32'h00320000, 32'h0032E000, 32'h00330780, 1792, 16'd1);
        setup_cpu_desc(32'h00300200, 32'h00300220,   // desc 3 → 4
                       32'h00340100, 32'h00340180, 128);
        setup_q5_desc(32'h00300220, 32'h00000000,   // desc 4 → end
                      32'h00321000, 32'h0032F000, 32'h00330800, 1792, 16'd1);
        // Desc 0: weights=1
        fill_q5_weight(32'h0031F000, 4'd1);
        fill_q5_scales(32'h0031F000);
        // Desc 1: CPU_OP pattern
        for (j = 0; j < 32; j = j + 1)
            ddr_write32(32'h00340000 + j*4,
                (4*j+0) + ((4*j+1) << 8) + ((4*j+2) << 16) + ((4*j+3) << 24));
        zero_res(32'h00340080);
        // Desc 2: weights=0
        fill_q5_weight(32'h00320000, 4'd0);
        fill_q5_scales(32'h00320000);
        // Desc 3: CPU_OP pattern
        for (j = 0; j < 32; j = j + 1)
            ddr_write32(32'h00340100 + j*4,
                (128+4*j+0) + ((128+4*j+1) << 8) + ((128+4*j+2) << 16) + ((128+4*j+3) << 24));
        zero_res(32'h00340180);
        // Desc 4: weights=1
        fill_q5_weight(32'h00321000, 4'd1);
        fill_q5_scales(32'h00321000);
        fill_q5_acts(32'h0032D000, 16'd1);
        fill_q5_acts(32'h0032E000, 16'd1);
        fill_q5_acts(32'h0032F000, 16'd1);
        zero_res(32'h00330700);
        zero_res(32'h00330780);
        zero_res(32'h00330800);

        start_chain(32'h003001A0);
        wait_done(5);
        axil_read(16'h20, rd_val);
        $display("  HEAD=%d (expect 5)", rd_val);
        $display("  Verifying Q5_0 desc 0 (weights=1, expect 229376):");
        verify_q5_result(32'h00330700, 229376, test_num);
        $display("  Verifying CPU_OP desc 1 passthrough (128 bytes):");
        for (j = 0; j < 32; j = j + 1) begin
            reg [31:0] expected;
            expected = (4*j+0) + ((4*j+1) << 8) + ((4*j+2) << 16) + ((4*j+3) << 24);
            rd_val = ddr_read32(32'h00340080 + j*4);
            if (rd_val == expected)
                $display("    Word %0d: PASS (0x%08x)", j, rd_val);
            else begin
                $display("    FAIL[%0d]: word %0d got 0x%08x expected 0x%08x", test_num, j, rd_val, expected);
                fail_count = fail_count + 1;
            end
        end
        $display("  Verifying Q5_0 desc 2 (weights=0, expect 0):");
        verify_q5_result(32'h00330780, 0, test_num);
        $display("  Verifying CPU_OP desc 3 passthrough (128 bytes):");
        for (j = 0; j < 32; j = j + 1) begin
            reg [31:0] expected;
            expected = (128+4*j+0) + ((128+4*j+1) << 8) + ((128+4*j+2) << 16) + ((128+4*j+3) << 24);
            rd_val = ddr_read32(32'h00340180 + j*4);
            if (rd_val == expected)
                $display("    Word %0d: PASS (0x%08x)", j, rd_val);
            else begin
                $display("    FAIL[%0d]: word %0d got 0x%08x expected 0x%08x", test_num, j, rd_val, expected);
                fail_count = fail_count + 1;
            end
        end
        $display("  Verifying Q5_0 desc 4 (weights=1, expect 229376):");
        verify_q5_result(32'h00330800, 229376, test_num);

        // =================================================================
        // Test 10: Multi-tile Q5_0 descriptor (4 tiles, all-1s)
        //   16 rows × 896 cols, all weights=1, acts=1 → each row = 229376
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Multi-tile Q5_0 (4 tiles, expect 16×229376) ---", test_num);
        setup_q5_desc(32'h00300240, 32'h00000000,
                      32'h00322000, 32'h00325000, 32'h00330880, 1792, 16'd4);
        // Fill 4 tiles of weight data at 2696-byte stride
        for (j = 0; j < 4; j = j + 1) begin
            fill_q5_weight(32'h00322000 + j * 2696, 4'd1);
            fill_q5_scales(32'h00322000 + j * 2696);
        end
        fill_q5_acts(32'h00325000, 16'd1);
        for (j = 0; j < 4; j = j + 1)
            zero_res(32'h00330880 + j * 32);
        start_chain(32'h00300240);
        wait_done(1);
        axil_read(16'h14, rd_val); $display("  STATUS=0x%08x", rd_val);
        axil_read(16'h28, rd_val); $display("  DEBUG=0x%08x", rd_val);
        axil_read(16'h20, rd_val); $display("  HEAD=%d (expect 1)", rd_val);
        for (j = 0; j < 4; j = j + 1) begin
            $display("  Verifying tile %0d (rows %0d-%0d):", j, j*4, j*4+3);
            verify_q5_result(32'h00330880 + j * 32, 229376, test_num);
        end

        // Summary
        $display("\n==============================================");
        if (fail_count == 0)
            $display("  ALL %0d TESTS PASSED", test_num);
        else
            $display("  %0d TESTS FAILED (of %0d)", fail_count, test_num);
        $display("==============================================");
        #100 $finish;
    end

    initial #5000000 begin
        $display("TIMEOUT");
        $finish;
    end
endmodule