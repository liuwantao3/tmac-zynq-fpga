`timescale 1ns / 1ps
// Comprehensive HP FSM test: 7 test cases covering edge cases, chaining, restart
module tb_hw_fsm_comprehensive;
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
        if (addr[2])
            ddr_mem[addr[31:3]][63:32] = val;
        else
            ddr_mem[addr[31:3]][31:0] = val;
    endtask
    function [31:0] ddr_read32(input [31:0] addr);
        if (addr[2])
            ddr_read32 = ddr_mem[addr[31:3]][63:32];
        else
            ddr_read32 = ddr_mem[addr[31:3]][31:0];
    endfunction

    // Write state machine — supports INCR burst (awlen > 0, wlast)
    localparam WR_IDLE = 0, WR_WRITE = 1, WR_WAIT_B = 2;
    reg [1:0] wr_state;
    reg [31:0] wr_addr_cur;
    reg [7:0]  wr_beats_rem;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_state <= WR_IDLE;
            wr_addr_cur <= 0;
            wr_beats_rem <= 0;
        end else begin
            case (wr_state)
                WR_IDLE: if (m_axi_awvalid) begin
                    wr_addr_cur <= m_axi_awaddr;
                    wr_beats_rem <= m_axi_awlen;
                    wr_state <= WR_WRITE;
                end
                WR_WRITE: if (m_axi_wvalid && m_axi_wready) begin
                    if (m_axi_wstrb[3:0]) begin
                        ddr_mem[wr_addr_cur[31:3]][31:0] <= m_axi_wdata[31:0];
                        $display("[TB] WR beat addr=0x%08x data[31:0]=0x%08x strb=%b",
                            wr_addr_cur, m_axi_wdata[31:0], m_axi_wstrb[3:0]);
                    end
                    if (m_axi_wstrb[7:4]) begin
                        ddr_mem[wr_addr_cur[31:3]][63:32] <= m_axi_wdata[63:32];
                        $display("[TB] WR beat addr=0x%08x data[63:32]=0x%08x strb=%b",
                            wr_addr_cur, m_axi_wdata[63:32], m_axi_wstrb[7:4]);
                    end
                    if (m_axi_wlast) begin
                        wr_state <= WR_WAIT_B;
                    end else begin
                        wr_addr_cur <= wr_addr_cur + 8;
                        wr_beats_rem <= wr_beats_rem - 1;
                    end
                end
                WR_WAIT_B: if (m_axi_bready) wr_state <= WR_IDLE;
            endcase
        end
    end

    // Read state machine — ARSIZE=2, data on RDATA[31:0]
    reg [7:0]  rd_beats_done, rd_beats_total;
    reg [31:0] rd_addr_base;
    reg        rd_busy;

    wire [31:0] rd_idx  = (rd_addr_base[31:2] + rd_beats_done);
    wire [31:0] rd_half = rd_idx[0] ? ddr_mem[rd_idx >> 1][63:32]
                                     : ddr_mem[rd_idx >> 1][31:0];

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
                rd_addr_base   <= m_axi_araddr;
                rd_beats_done  <= 0;
                rd_beats_total <= m_axi_arlen;
                rd_busy        <= 1;
            end
            if (rd_busy && m_axi_rready) begin
                if (rd_beats_done == rd_beats_total)
                    rd_busy <= 0;
                else
                    rd_beats_done <= rd_beats_done + 1;
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
    integer i, pass_count, fail_count, test_num;

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

    // Write descriptor at desc_addr in DDR
    task setup_desc(input [31:0] desc_addr, input [31:0] next_addr,
                    input [31:0] act_addr, input [31:0] res_addr,
                    input [23:0] act_bytes);
        ddr_write32(desc_addr + 0,  next_addr);
        ddr_write32(desc_addr + 4,  32'h00000000);
        ddr_write32(desc_addr + 8,  act_addr);
        ddr_write32(desc_addr + 12, res_addr);
        ddr_write32(desc_addr + 16, 32'h0000000F);  // tensor_type=15 (CPU_OP)
        ddr_write32(desc_addr + 20, 32'h00000000);
        ddr_write32(desc_addr + 24, {8'h00, act_bytes});
        ddr_write32(desc_addr + 28, 32'h00000000);
    endtask

    // Q8 descriptor: tensor_type=0, num_groups=1 (single column group, 64 cols)
    task setup_q8_desc(input [31:0] desc_addr, input [31:0] next_addr,
                       input [31:0] weight_addr, input [31:0] act_addr,
                       input [31:0] res_addr, input [15:0] num_tiles);
        ddr_write32(desc_addr + 0,  next_addr);
        ddr_write32(desc_addr + 4,  weight_addr);
        ddr_write32(desc_addr + 8,  act_addr);
        ddr_write32(desc_addr + 12, res_addr);
        ddr_write32(desc_addr + 16, 32'h00000000);  // tensor_type=0 (Q8_0), groups=0→GP0 fallback
        ddr_write32(desc_addr + 20, {num_tiles[15:8], num_tiles[7:0], 8'h00, 8'h01});  // num_groups=1, num_tiles
        ddr_write32(desc_addr + 24, 24'd128);  // act_total_bytes = 64 × INT16
        ddr_write32(desc_addr + 28, 32'h00000000);
    endtask

    // Write incrementing pattern to DDR at base_addr for nbytes (must be multiple of 4)
    task write_pattern_inc(input [31:0] base_addr, input [23:0] nbytes);
        integer j;
        for (j = 0; j < nbytes/4; j = j + 1)
            ddr_write32(base_addr + j*4, 32'h03020100 + j * 32'h04040404);
    endtask

    // Write constant byte pattern to DDR
    task write_pattern_const(input [31:0] base_addr, input [23:0] nbytes, input [7:0] byte_val);
        integer j;
        for (j = 0; j < nbytes/4; j = j + 1)
            ddr_write32(base_addr + j*4, {4{byte_val}});
    endtask

    // Write alternating 0xA5/0x5A pattern
    task write_pattern_checker(input [31:0] base_addr, input [23:0] nbytes);
        integer j;
        for (j = 0; j < nbytes/4; j = j + 1)
            ddr_write32(base_addr + j*4, (j & 1) ? 32'h5A5A5A5A : 32'hA5A5A5A5);
    endtask

    // Zero-fill DDR region
    task zero_fill(input [31:0] base_addr, input [23:0] nbytes);
        integer j;
        for (j = 0; j < nbytes/4; j = j + 1)
            ddr_write32(base_addr + j*4, 32'h00000000);
    endtask

    // Fill 4096-byte Q8 weight tile with all 0x01 bytes (INT8 1)
    task fill_q8_weight_all1(input [31:0] base_addr);
        integer j;
        for (j = 0; j < 4096/4; j = j + 1)
            ddr_write32(base_addr + j*4, 32'h01010101);
    endtask

    // Fill 256 bytes of Q8 scales with UQ8.8 256 (1.0, per column group)
    // Each 32-bit word holds two 16-bit UQ8.8 values: {256, 256} = 32'h01000100
    task fill_q8_scales_all1(input [31:0] base_addr);
        integer j;
        for (j = 0; j < 256/4; j = j + 1)
            ddr_write32(base_addr + j*4, 32'h01000100);
    endtask

    // Fill activation with INT16 1 (0x0001 little-endian)
    task fill_act_all1(input [31:0] base_addr, input [23:0] nbytes);
        integer j;
        for (j = 0; j < nbytes/4; j = j + 1)
            ddr_write32(base_addr + j*4, 32'h00010001);
    endtask

    // Verify 64-bit Q8 result word at offset (row * 8) in result buffer
    task verify_q8_row(input [31:0] res_addr, input [31:0] row,
                       input [63:0] expected, input [31:0] test_id);
        reg [63:0] got;
        got[31:0]  = ddr_read32(res_addr + row * 8 + 0);
        got[63:32] = ddr_read32(res_addr + row * 8 + 4);
        if (got !== expected) begin
            $display("  FAIL[%0d]: row %0d addr=0x%08x expected=0x%016x got=0x%016x",
                test_id, row, res_addr + row * 8, expected, got);
            fail_count = fail_count + 1;
        end
    endtask

    // Configure and start the FSM
    task start_chain(input [31:0] desc_base);
        axil_write(16'h18, desc_base);  // REG_DESC_BASE
        axil_write(16'h1C, 32'd1);      // REG_DESC_TAIL = 1
        axil_write(16'h00, 32'd1);      // REG_START = 1
    endtask

    // Wait for HEAD to reach expected_head (poll HEAD register)
    task wait_done;
        input [31:0] expected_head;
        integer timeout;
        reg completed;
        reg [31:0] head_val;
        timeout = 0;
        completed = 0;
        while (timeout < 1000000) begin
            axil_read(16'h20, head_val);
            if (head_val >= expected_head) begin
                completed = 1;
                timeout = 1000000;
            end else begin
                timeout = timeout + 1;
                #10;
            end
        end
        if (!completed) begin
            axil_read(16'h28, rd_val); $display("  WARN: wait_done timeout, DEBUG=0x%08x", rd_val);
            axil_read(16'h14, rd_val); $display("  STATUS=0x%08x", rd_val);
            axil_read(16'h20, rd_val); $display("  HEAD=%d", rd_val);
        end
    endtask

    // Verify incrementing pattern in DDR region
    task verify_pattern_inc(input [31:0] base_addr, input [23:0] nbytes,
                            input [31:0] test_id);
        integer j, ok;
        ok = 1;
        for (j = 0; j < nbytes/4; j = j + 1) begin
            if (ddr_read32(base_addr + j*4) !== (32'h03020100 + j * 32'h04040404)) begin
                $display("  FAIL[%0d]: addr=0x%08x expected=0x%08x got=0x%08x",
                    test_id, base_addr + j*4,
                    32'h03020100 + j * 32'h04040404, ddr_read32(base_addr + j*4));
                ok = 0;
            end
        end
        if (ok) $display("  Test %0d: PASS", test_id);
        else begin fail_count = fail_count + 1; end
    endtask

    // Verify constant byte pattern
    task verify_pattern_const(input [31:0] base_addr, input [23:0] nbytes,
                              input [7:0] byte_val, input [31:0] test_id);
        integer j, ok;
        reg [31:0] expected;
        ok = 1;
        expected = {4{byte_val}};
        for (j = 0; j < nbytes/4; j = j + 1) begin
            if (ddr_read32(base_addr + j*4) !== expected) begin
                $display("  FAIL[%0d]: addr=0x%08x expected=0x%08x got=0x%08x",
                    test_id, base_addr + j*4, expected, ddr_read32(base_addr + j*4));
                ok = 0;
            end
        end
        if (ok) $display("  Test %0d: PASS", test_id);
        else begin fail_count = fail_count + 1; end
    endtask

    // Verify checkerboard pattern
    task verify_pattern_checker(input [31:0] base_addr, input [23:0] nbytes,
                                input [31:0] test_id);
        integer j, ok;
        reg [31:0] expected;
        ok = 1;
        for (j = 0; j < nbytes/4; j = j + 1) begin
            expected = (j & 1) ? 32'h5A5A5A5A : 32'hA5A5A5A5;
            if (ddr_read32(base_addr + j*4) !== expected) begin
                $display("  FAIL[%0d]: addr=0x%08x expected=0x%08x got=0x%08x",
                    test_id, base_addr + j*4, expected, ddr_read32(base_addr + j*4));
                ok = 0;
            end
        end
        if (ok) $display("  Test %0d: PASS", test_id);
        else begin fail_count = fail_count + 1; end
    endtask

    initial begin
        $dumpfile("tb_hw_fsm_comprehensive.vcd");
        $dumpvars(0, tb_hw_fsm_comprehensive);
        $display("==============================================");
        $display("  HP FSM Comprehensive Test Suite (8 tests)");
        $display("==============================================");
        #12 rst_n = 1; #20;
        pass_count = 0; fail_count = 0; test_num = 0;

        // ===================================================================
        // Test 1: Basic 64 bytes (regression — matches working hardware test)
        // ===================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Basic 64 bytes ---", test_num);
        setup_desc(32'h00300000, 32'h00000000, 32'h00301000, 32'h00302000, 64);
        write_pattern_inc(32'h00301000, 64);
        zero_fill(32'h00302000, 64);
        start_chain(32'h00300000);
        wait_done(1);
        axil_read(16'h14, rd_val);
        $display("  STATUS=0x%08x (expect 0x300)", rd_val);
        axil_read(16'h28, rd_val); $display("  DEBUG=0x%08x", rd_val);
        axil_read(16'h20, rd_val); $display("  HEAD=%d", rd_val);
        verify_pattern_inc(32'h00302000, 64, test_num);

        // ===================================================================
        // Test 2: Minimum 8 bytes (1 word — edge case for write master)
        // ===================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Minimum 8 bytes ---", test_num);
        setup_desc(32'h00300020, 32'h00000000, 32'h00301040, 32'h00302040, 8);
        ddr_write32(32'h00301040, 32'hAABBCCDD);
        zero_fill(32'h00302040, 8);
        start_chain(32'h00300020);
        wait_done(1);
        axil_read(16'h14, rd_val);
        $display("  STATUS=0x%08x (expect 0x300)", rd_val);
        if (ddr_read32(32'h00302040) === 32'hAABBCCDD)
            $display("  Test %0d: PASS", test_num);
        else begin
            $display("  FAIL[%0d]: res=0x%08x expected 0xAABBCCDD", test_num, ddr_read32(32'h00302040));
            fail_count = fail_count + 1;
        end

        // ===================================================================
        // Test 3: 128 bytes (2 HP read bursts — verifies multi-burst path)
        // ===================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: 128 bytes (2 bursts) ---", test_num);
        setup_desc(32'h00300040, 32'h00000000, 32'h00301100, 32'h00302100, 128);
        write_pattern_inc(32'h00301100, 128);
        zero_fill(32'h00302100, 128);
        start_chain(32'h00300040);
        wait_done(1);
        axil_read(16'h14, rd_val);
        $display("  STATUS=0x%08x (expect 0x300)", rd_val);
        axil_read(16'h28, rd_val); $display("  DEBUG=0x%08x", rd_val);
        verify_pattern_inc(32'h00302100, 128, test_num);

        // ===================================================================
        // Test 4: 256 bytes max (4 HP read bursts — max act_buf)
        // ===================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: 256 bytes max (4 bursts) ---", test_num);
        setup_desc(32'h00300060, 32'h00000000, 32'h00301200, 32'h00302200, 256);
        write_pattern_inc(32'h00301200, 256);
        zero_fill(32'h00302200, 256);
        start_chain(32'h00300060);
        wait_done(1);
        axil_read(16'h14, rd_val);
        $display("  STATUS=0x%08x (expect 0x300)", rd_val);
        axil_read(16'h28, rd_val); $display("  DEBUG=0x%08x", rd_val);
        verify_pattern_inc(32'h00302200, 256, test_num);

        // ===================================================================
        // Test 5: Chain of 2 descriptors (verifies next_addr traversal)
        // Desc 0: 64 bytes inc pattern, act→res
        // Desc 1: 32 bytes const 0xFF pattern, act→res
        // ===================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Chain of 2 descriptors ---", test_num);
        // Desc 0 at 0x00300100: next = 0x00300120 (next descriptor)
        setup_desc(32'h00300100, 32'h00300120, 32'h00301300, 32'h00302300, 64);
        // Desc 1 at 0x00300120: next = 0 (end of chain)
        setup_desc(32'h00300120, 32'h00000000, 32'h00301340, 32'h00302340, 32);
        // Act data for desc 0: incrementing pattern
        write_pattern_inc(32'h00301300, 64);
        // Act data for desc 1: constant 0xFF
        write_pattern_const(32'h00301340, 32, 8'hFF);
        // Zero fill both result areas
        zero_fill(32'h00302300, 64);
        zero_fill(32'h00302340, 32);
        // Start chain from desc 0
        start_chain(32'h00300100);
        wait_done(2);
        axil_read(16'h14, rd_val);
        $display("  STATUS=0x%08x (expect 0x300)", rd_val);
        axil_read(16'h20, rd_val); $display("  HEAD=%d (expect 2)", rd_val);
        $display("  Verify Desc 0 result:");
        verify_pattern_inc(32'h00302300, 64, test_num);
        $display("  Verify Desc 1 result:");
        verify_pattern_const(32'h00302340, 32, 8'hFF, test_num);

        // ===================================================================
        // Test 6: Chain of 3 descriptors with different patterns
        // Desc 0: 48 bytes inc pattern
        // Desc 1: 64 bytes 0xFF pattern
        // Desc 2: 40 bytes checkerboard
        // ===================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Chain of 3 descriptors ---", test_num);
        setup_desc(32'h00300140, 32'h00300160, 32'h00301400, 32'h00302400, 48);
        setup_desc(32'h00300160, 32'h00300180, 32'h00301430, 32'h00302430, 64);
        setup_desc(32'h00300180, 32'h00000000, 32'h00301470, 32'h00302470, 40);
        write_pattern_inc(32'h00301400, 48);
        write_pattern_const(32'h00301430, 64, 8'hFF);
        write_pattern_checker(32'h00301470, 40);
        zero_fill(32'h00302400, 48);
        zero_fill(32'h00302430, 64);
        zero_fill(32'h00302470, 40);
        start_chain(32'h00300140);
        wait_done(3);
        axil_read(16'h14, rd_val);
        $display("  STATUS=0x%08x (expect 0x300)", rd_val);
        axil_read(16'h20, rd_val); $display("  HEAD=%d (expect 3)", rd_val);
        $display("  Verify Desc 0 (inc 48 bytes):");
        verify_pattern_inc(32'h00302400, 48, test_num);
        $display("  Verify Desc 1 (0xFF 64 bytes):");
        verify_pattern_const(32'h00302430, 64, 8'hFF, test_num);
        $display("  Verify Desc 2 (checker 40 bytes):");
        verify_pattern_checker(32'h00302470, 40, test_num);

        // ===================================================================
        // Test 7: Re-start from DONE
        // Run desc A (64 bytes inc), verify, then re-start with desc B (32 bytes const)
        // ===================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Re-start from DONE ---", test_num);
        // First run
        setup_desc(32'h00300200, 32'h00000000, 32'h00301500, 32'h00302500, 64);
        write_pattern_inc(32'h00301500, 64);
        zero_fill(32'h00302500, 64);
        start_chain(32'h00300200);
        wait_done(1);
        axil_read(16'h14, rd_val);
        $display("  First run STATUS=0x%08x (expect 0x300)", rd_val);
        axil_read(16'h20, rd_val); $display("  First run HEAD=%d (expect 1)", rd_val);
        // Verify first run result
        verify_pattern_inc(32'h00302500, 64, test_num);

        // Set up second descriptor — must use different DDR addresses
        setup_desc(32'h00300220, 32'h00000000, 32'h00301540, 32'h00302540, 32);
        write_pattern_const(32'h00301540, 32, 8'h5A);
        zero_fill(32'h00302540, 32);
        // Re-start from DONE with new desc_base
        start_chain(32'h00300220);
        wait_done(1);
        axil_read(16'h14, rd_val);
        $display("  Second run STATUS=0x%08x (expect 0x300)", rd_val);
        axil_read(16'h20, rd_val); $display("  Second run HEAD=%d (expect 1)", rd_val);
        // Verify second run result (should NOT have corrupted first run)
        $display("  Verify first run still intact:");
        verify_pattern_inc(32'h00302500, 64, test_num);
        $display("  Verify second run result:");
        verify_pattern_const(32'h00302540, 32, 8'h5A, test_num);

        // ===================================================================
        // Test 8: Q8 multi-tile (2 tiles, all-1s, 1 column group)
        //   weight=0x01 (INT8 1), scale=UQ8.8 256 (1.0), act=INT16 1
        //   dequant_q8(1, 256) = 1*256>>8 = 1
        //   Each row sum = 64 cols × 1 × 1 = 64
        //   Two tiles = 128 rows, each = 64
        // ===================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Q8 multi-tile (2 tiles, all-1s) ---", test_num);
        // Q8 weight tile 0 at 0x00310000, scales at offset 4096
        fill_q8_weight_all1(32'h00310000);
        fill_q8_scales_all1(32'h00311000);    // weight_addr + 4096
        // Q8 weight tile 1 at 0x00311100 (4352 stride/tile for 1 column group)
        fill_q8_weight_all1(32'h00311100);
        fill_q8_scales_all1(32'h00312100);    // tile 1 weight + 4096
        // Activation: 64 × INT16 1
        fill_act_all1(32'h00320000, 128);
        // Zero-fill result area
        zero_fill(32'h00321000, 1024);
        // Descriptor: tensor_type=0, num_groups=1, num_tiles=2, act_bytes=128
        setup_q8_desc(32'h00330000, 32'h00000000, 32'h00310000,
                      32'h00320000, 32'h00321000, 16'd2);
        // REG_Q8_NUM_GROUPS fallback not needed (descriptor has num_groups=1)
        start_chain(32'h00330000);
        wait_done(1);
        axil_read(16'h14, rd_val);
        $display("  STATUS=0x%08x (expect 0x300)", rd_val);
        axil_read(16'h20, rd_val); $display("  HEAD=%d (expect 1)", rd_val);
        // Verify tile 0 (rows 0-63) and tile 1 (rows 64-127)
        for (i = 0; i < 64; i = i + 1)
            verify_q8_row(32'h00321000, i, 64'd64, test_num);
        for (i = 0; i < 64; i = i + 1)
            verify_q8_row(32'h00321200, i, 64'd64, test_num);
        $display("  Test %0d: verify done (failures counted above)", test_num);

        // ===================================================================
        // Summary
        // ===================================================================
        $display("\n==============================================");
        if (fail_count == 0)
            $display("  ALL %0d TESTS PASSED", test_num);
        else
            $display("  %0d TESTS FAILED (of %0d)", fail_count, test_num);
        $display("==============================================");
        #100 $finish;
    end

    initial #1000000 begin
        $display("TIMEOUT");
        $finish;
    end
endmodule
