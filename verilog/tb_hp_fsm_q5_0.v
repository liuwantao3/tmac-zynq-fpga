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
                       input [31:0] res_addr, input [23:0] act_bytes);
        ddr_write32(desc_addr + 0,  next_addr);
        ddr_write32(desc_addr + 4,  weight_addr);
        ddr_write32(desc_addr + 8,  act_addr);
        ddr_write32(desc_addr + 12, res_addr);
        ddr_write32(desc_addr + 16, 32'h00000001);  // tensor_type=1 (Q5_0)
        ddr_write32(desc_addr + 20, 32'h00000000);
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

    // Write one Q5_0 block (22 bytes) at DDR byte offset with given q5_val
    task gen_q5_block_at(input [31:0] base, input [15:0] byte_off,
                         input [3:0] q5_val);
        reg [15:0] f16;
        integer k;
        begin
            f16 = 16'h3C00;  // f16(1.0) = 0x3C00
            ddr_write8(base + byte_off + 0, f16[7:0]);
            ddr_write8(base + byte_off + 1, f16[15:8]);
            ddr_write8(base + byte_off + 2, 8'hFF);  // qh = 0xFFFFFFFF
            ddr_write8(base + byte_off + 3, 8'hFF);
            ddr_write8(base + byte_off + 4, 8'hFF);
            ddr_write8(base + byte_off + 5, 8'hFF);
            for (k = 0; k < 16; k = k + 1)
                ddr_write8(base + byte_off + 6 + k, {q5_val, q5_val});
        end
    endtask

    // Fill all 224 blocks with a given q5_val
    task fill_q5_weight(input [31:0] base, input [3:0] q5_val);
        integer b;
        begin
            for (b = 0; b < 224; b = b + 1)
                gen_q5_block_at(base, b * 22, q5_val);
        end
    endtask

    // Fill 8 row scales (UQ8.8 = 0x0001 for scale=1/256)
    task fill_q5_scales(input [31:0] weight_base);
        integer s;
        begin
            for (s = 0; s < 8; s = s + 1) begin
                ddr_write8(weight_base + 4928 + s*2, 8'd1);
                ddr_write8(weight_base + 4928 + s*2 + 1, 8'd0);
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

    // Verify 8 rows of Q5_0 result at res_addr
    task verify_q5_result(input [31:0] res_addr, input [47:0] expected, input integer tn);
        integer r;
        reg [31:0] lo, hi;
        reg signed [47:0] got;
        begin
            for (r = 0; r < 8; r = r + 1) begin
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

    // Zero 64 bytes
    task zero_res(input [31:0] addr);
        integer z;
        begin
            for (z = 0; z < 16; z = z + 1)
                ddr_write32(addr + z*4, 32'h00000000);
        end
    endtask

    initial begin
        $dumpfile("tb_hp_fsm_q5_0.vcd");
        $dumpvars(0, tb_hp_fsm_q5_0);
        $display("==============================================");
        $display("  HP FSM Q5_0 Multi-Descriptor Tests");
        $display("==============================================");
        #12 rst_n = 1; #20;
        fail_count = 0; test_num = 0;

        // =================================================================
        // Test 1: Single Q5_0 descriptor, all-1s -> expect 896 per row
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Single Q5_0 all-1s (expect each row=896) ---", test_num);
        setup_q5_desc(32'h00300000, 32'h00000000,
                      32'h00310000, 32'h00320000, 32'h00330000, 1792);
        fill_q5_weight(32'h00310000, 4'd1);
        fill_q5_scales(32'h00310000);
        fill_q5_acts(32'h00320000, 16'd1);
        zero_res(32'h00330000);
        start_chain(32'h00300000);
        wait_done(1);
        axil_read(16'h14, rd_val); $display("  STATUS=0x%08x", rd_val);
        axil_read(16'h28, rd_val); $display("  DEBUG=0x%08x", rd_val);
        axil_read(16'h20, rd_val); $display("  HEAD=%d", rd_val);
        verify_q5_result(32'h00330000, 896, test_num);

        // =================================================================
        // Test 2: Chain of 2 Q5_0 descriptors
        //   Desc 0: all-1s -> expect 896 per row
        //   Desc 1: all-0s -> expect 0 per row
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Chain of 2 Q5_0 (all-1s, then all-0s) ---", test_num);
        setup_q5_desc(32'h00300020, 32'h00300040,   // desc 0 -> desc 1
                      32'h00310000, 32'h00320000, 32'h00330040, 1792);
        setup_q5_desc(32'h00300040, 32'h00000000,   // desc 1 -> end
                      32'h00313000, 32'h00322000, 32'h00330080, 1792);
        fill_q5_weight(32'h00310000, 4'd1);   // Desc 0: weights=1 -> expect 896
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
        $display("  Verifying Desc 0 (weights=1, expect 896):");
        verify_q5_result(32'h00330040, 896, test_num);
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
                      32'h00314000, 32'h00323000, 32'h003300C0, 1792);
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
        $display("  Verifying Q5_0 desc 1 (expect 896 per row):");
        verify_q5_result(32'h003300C0, 896, test_num);

        // =================================================================
        // Test 4: Chain of 3: Q5_0 -> CPU_OP -> Q5_0
        //   Desc 0: Q5_0 all-1s -> 896
        //   Desc 1: CPU_OP passthrough 64 bytes
        //   Desc 2: Q5_0 all-2s -> 3584
        // =================================================================
        test_num = test_num + 1;
        $display("\n--- Test %0d: Chain of 3 (Q5_0 -> CPU_OP -> Q5_0) ---", test_num);
        setup_q5_desc(32'h003000A0, 32'h003000C0,   // desc 0 -> desc 1
                      32'h00315000, 32'h00324000, 32'h00330100, 1792);
        setup_cpu_desc(32'h003000C0, 32'h003000E0,   // desc 1 -> desc 2
                       32'h00340100, 32'h00340140, 64);
        setup_q5_desc(32'h003000E0, 32'h00000000,   // desc 2 -> end
                      32'h00317000, 32'h00325000, 32'h00330180, 1792);
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
        $display("  Verifying Q5_0 desc 0 (weights=1, expect 896):");
        verify_q5_result(32'h00330100, 896, test_num);
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
        $display("  Verifying Q5_0 desc 2 (weights=2, acts=2, expect 3584):");
        verify_q5_result(32'h00330180, 3584, test_num);

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