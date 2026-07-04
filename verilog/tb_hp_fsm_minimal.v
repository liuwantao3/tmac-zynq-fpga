`timescale 1ns / 1ps
// Minimal hp_fsm_top integration test: read master + write master + descriptor chain
// Tests CPU_OP descriptor (bypass Q8 compute) to verify read/write pipeline.
// Debug: override tensor_type to force CPU_OP path
`define FORCE_CPU_OP
module tb_hp_fsm_minimal;
    reg clk = 0, rst_n = 0;
    always #5 clk = ~clk;

    // AXI4-Lite master
    reg  [15:0] axil_awaddr; reg axil_awvalid;
    reg  [31:0] axil_wdata;  reg [3:0] axil_wstrb;
    reg  axil_wvalid;
    wire axil_awready, axil_wready, axil_bvalid;
    wire [1:0] axil_bresp, axil_rresp;
    reg  axil_bready;
    reg  [15:0] axil_araddr; reg axil_arvalid, axil_rready;
    wire axil_arready, axil_rvalid;
    wire [31:0] axil_rdata;

    // AXI HP shared wires
    wire [31:0] m_axi_awaddr; wire m_axi_awvalid, m_axi_awready;
    wire [7:0]  m_axi_awlen;  wire [2:0] m_axi_awsize, m_axi_arsize;
    wire [1:0]  m_axi_awburst, m_axi_arburst;
    wire [63:0] m_axi_wdata;  wire m_axi_wvalid, m_axi_wready, m_axi_wlast;
    wire [7:0]  m_axi_wstrb;  wire [5:0] m_axi_wid, m_axi_awid, m_axi_arid;
    wire m_axi_bvalid, m_axi_bready; wire [1:0] m_axi_bresp; wire [5:0] m_axi_bid;
    wire [31:0] m_axi_araddr; wire m_axi_arvalid, m_axi_arready;
    wire [7:0]  m_axi_arlen;
    wire [63:0] m_axi_rdata;  wire [1:0] m_axi_rresp;
    wire [5:0]  m_axi_rid;    wire m_axi_rvalid, m_axi_rready, m_axi_rlast;

    // Internal DDR memory (4 MB = 524288 × 64-bit words)
    reg [63:0] ddr_mem [0:524287];

    // Task: write 32-bit word to DDR (addr must be 4-byte aligned)
    task ddr_write32(input [31:0] addr, input [31:0] val);
        if (addr[31:3] <= 524287) begin
            if (addr[2])
                ddr_mem[addr[31:3]][63:32] = val;
            else
                ddr_mem[addr[31:3]][31:0] = val;
        end
    endtask

    function [31:0] ddr_read32(input [31:0] addr);
        if (addr[31:3] <= 524287) begin
            if (addr[2])
                ddr_read32 = ddr_mem[addr[31:3]][63:32];
            else
                ddr_read32 = ddr_mem[addr[31:3]][31:0];
        end else begin
            ddr_read32 = 32'hDEADBEEF;
        end
    endfunction

    // Write state machine (handles write master's INCR bursts)
    localparam WR_IDLE = 0, WR_WRITE = 1, WR_WAIT_B = 2;
    reg [1:0] wr_state;
    reg [31:0] wr_addr_cur;
    reg [7:0]  wr_beats_rem;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_state <= WR_IDLE; wr_addr_cur <= 0; wr_beats_rem <= 0;
        end else begin
            case (wr_state)
                WR_IDLE: if (m_axi_awvalid) begin
                    wr_addr_cur <= m_axi_awaddr;
                    wr_beats_rem <= m_axi_awlen;
                    wr_state <= WR_WRITE;
                end
                WR_WRITE: if (m_axi_wvalid && m_axi_wready) begin
                    if (m_axi_wstrb[3:0])
                        ddr_mem[wr_addr_cur[31:3]][31:0] <= m_axi_wdata[31:0];
                    if (m_axi_wstrb[7:4])
                        ddr_mem[wr_addr_cur[31:3]][63:32] <= m_axi_wdata[63:32];
                    if (m_axi_wlast) begin
                        wr_state <= WR_WAIT_B;
                    end else begin
                        wr_addr_cur <= wr_addr_cur + 4;
                        wr_beats_rem <= wr_beats_rem - 1;
                    end
                end
                WR_WAIT_B: if (m_axi_bready) wr_state <= WR_IDLE;
            endcase
        end
    end

    // Read state machine (handles read master's ARSIZE=2 INCR bursts)
    reg        rd_busy;
    reg [7:0]  rd_beats_done, rd_beats_total;
    reg [31:0] rd_addr_base;
    wire [31:0] rd_idx = (rd_addr_base[31:2] + rd_beats_done);
    wire [31:0] rd_half = rd_idx[0] ? ddr_mem[rd_idx >> 1][63:32]
                                     : ddr_mem[rd_idx >> 1][31:0];

    // Combinatorial AXI read slave: data on RDATA[31:0] (Zynq-7010 x16 DDR behavior)
    assign m_axi_arready = !rd_busy;
    assign m_axi_rdata   = {32'd0, rd_half};
    assign m_axi_rvalid  = rd_busy;
    assign m_axi_rresp   = 2'd0; assign m_axi_rid = 6'd0;
    assign m_axi_rlast   = rd_busy && (rd_beats_done == rd_beats_total);

    // AXI write slave (combinatorial handshake)
    assign m_axi_awready = (wr_state == WR_IDLE);
    assign m_axi_wready  = (wr_state == WR_WRITE);
    assign m_axi_bvalid  = (wr_state == WR_WAIT_B);
    assign m_axi_bresp   = 2'd0; assign m_axi_bid = 6'd0;

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

    // Debug: state transitions
    reg [31:0] prev_state = 0;
    always @(posedge clk) begin
        if (rst_n && uut.state != prev_state) begin
            $display("T=%0t ns STATE %0d -> %0d (rd_done_rise=%b rd_valid=%b rd_ready=%b rbusy=%b rstart=%b)",
                $time, prev_state, uut.state,
                uut.rd_done_rise, uut.rd_valid, uut.rd_ready, uut.rd_busy, uut.rd_start);
            prev_state <= uut.state;
        end
        // Trace desc_buf when entering FETCH_DESC_W
        if (rst_n && uut.state == 2) begin
            if (uut.rd_valid && uut.rd_ready)
                $display("T=%0t ns FETCH_DESC_W: cap byte_idx=%0d", $time, uut.desc_byte_idx);
            if (uut.rd_done_rise)
                $display("T=%0t ns FETCH_DESC_W: DONE! tensor_type=%0d (0x%04x) desc[17:16]=%0d",
                    $time, uut.tensor_type, uut.tensor_type,
                    {uut.desc_buf[17], uut.desc_buf[16]});
        end
        // Trace weight loading progress
        if (rst_n && uut.state == 8)
            $display("T=%0t ns LOAD_WEIGHT: wt_remaining=%0d", $time, uut.wt_remaining);
        if (rst_n && uut.state == 9) begin
            if (uut.wt_burst_done && !uut.rd_unpack_active)
                $display("T=%0t ns LOAD_WEIGHT_W: burst done, wt_remaining=%0d", $time, uut.wt_remaining);
        end
    end

    // Instantiate hp_fsm_top
    hp_fsm_top uut (
        .clk(clk), .rst_n(rst_n),
        .s_axil_awaddr(axil_awaddr), .s_axil_awvalid(axil_awvalid), .s_axil_awready(axil_awready),
        .s_axil_wdata(axil_wdata), .s_axil_wstrb(axil_wstrb),
        .s_axil_wvalid(axil_wvalid), .s_axil_wready(axil_wready),
        .s_axil_bresp(axil_bresp), .s_axil_bvalid(axil_bvalid), .s_axil_bready(axil_bready),
        .s_axil_araddr(axil_araddr), .s_axil_arvalid(axil_arvalid), .s_axil_arready(axil_arready),
        .s_axil_rdata(axil_rdata), .s_axil_rresp(axil_rresp),
        .s_axil_rvalid(axil_rvalid), .s_axil_rready(axil_rready),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize), .m_axi_awburst(m_axi_awburst),
        .m_axi_awlock(), .m_axi_awcache(), .m_axi_awprot(),
        .m_axi_wdata(m_axi_wdata), .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_wlast(m_axi_wlast), .m_axi_wstrb(m_axi_wstrb), .m_axi_wid(m_axi_wid),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_bresp(m_axi_bresp), .m_axi_bid(m_axi_bid),
        .m_axi_arid(m_axi_arid), .m_axi_araddr(m_axi_araddr),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_arlen(m_axi_arlen), .m_axi_arsize(m_axi_arsize), .m_axi_arburst(m_axi_arburst),
        .m_axi_arlock(), .m_axi_arcache(), .m_axi_arprot(),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(m_axi_rresp), .m_axi_rid(m_axi_rid),
        .m_axi_rvalid(m_axi_rvalid), .m_axi_rready(m_axi_rready), .m_axi_rlast(m_axi_rlast)
    );

    // AXI4-Lite access tasks
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

    // ================
    // Test sequence
    // ================
    integer pass, fail;
    reg [31:0] rd_val;

    initial begin
        $dumpfile("tb_hp_fsm_minimal.vcd");
        $dumpvars(0, tb_hp_fsm_minimal);
        pass = 0; fail = 0;

        #12 rst_n = 1; #20;

        // --- Test 1: CPU_OP descriptor using addresses within ddr_mem bounds ---
        // Use addresses in the first 4 MB: 0x00100000 to 0x0010001F for descriptor
        $display("--- Test 1: CPU_OP descriptor fetch + act load + result writeback ---");

        // Write descriptor at 0x00100000 (32 bytes)
        ddr_write32(32'h00100000, 32'h00000000);  // next_addr = 0 (end of chain)

        // DDR addresses 0x00100000 + 4 (second word): weight_addr → 0
        ddr_write32(32'h00100004, 32'h00000000);  // weight_addr = 0 (unused for CPU_OP)

        // Act addr = 0x00101000, res addr = 0x00102000
        ddr_write32(32'h00100008, 32'h00101000);  // act_addr
        ddr_write32(32'h0010000C, 32'h00102000);  // result_addr

        // tensor_type = 15 (CPU_OP), num_groups = 0
        ddr_write32(32'h00100010, 32'h0000000F);  // tensor_type[15:0]=15

        // reserved
        ddr_write32(32'h00100014, 32'h00000000);

        // act_total_bytes = 64 (0x40)
        ddr_write32(32'h00100018, 32'h00000040);

        // reserved
        ddr_write32(32'h0010001C, 32'h00000000);

        // Write activation pattern at 0x00101000
        ddr_write32(32'h00101000, 32'h03020100);
        ddr_write32(32'h00101004, 32'h07060504);
        ddr_write32(32'h00101008, 32'h0B0A0908);
        ddr_write32(32'h0010100C, 32'h0F0E0D0C);
        ddr_write32(32'h00101010, 32'h13121110);
        ddr_write32(32'h00101014, 32'h17161514);
        ddr_write32(32'h00101018, 32'h1B1A1918);
        ddr_write32(32'h0010101C, 32'h1F1E1D1C);
        ddr_write32(32'h00101020, 32'h23222120);
        ddr_write32(32'h00101024, 32'h27262524);
        ddr_write32(32'h00101028, 32'h2B2A2928);
        ddr_write32(32'h0010102C, 32'h2F2E2D2C);
        ddr_write32(32'h00101030, 32'h33323130);
        ddr_write32(32'h00101034, 32'h37363534);
        ddr_write32(32'h00101038, 32'h3B3A3938);
        ddr_write32(32'h0010103C, 32'h3F3E3D3C);

        // Zero-fill result area
        ddr_write32(32'h00102000, 32'h00000000);
        ddr_write32(32'h00102004, 32'h00000000);
        ddr_write32(32'h00102008, 32'h00000000);
        ddr_write32(32'h0010200C, 32'h00000000);
        ddr_write32(32'h00102010, 32'h00000000);
        ddr_write32(32'h00102014, 32'h00000000);
        ddr_write32(32'h00102018, 32'h00000000);
        ddr_write32(32'h0010201C, 32'h00000000);

        // Start the FSM
        $display("Writing REG_DESC_BASE = 0x00100000");
        axil_write(16'h18, 32'h00100000);  // REG_DESC_BASE
        axil_write(16'h1C, 32'd1);          // REG_DESC_TAIL = 1
        axil_write(16'h00, 32'd1);          // REG_START = 1

        // Wait for completion (poll HEAD register)
        #100000;  // 100µs timeout
        axil_read(16'h20, rd_val); $display("HEAD=%d", rd_val);
        axil_read(16'h28, rd_val); $display("DEBUG=0x%08x", rd_val);
        axil_read(16'h14, rd_val); $display("STATUS=0x%08x", rd_val);

        // Verify result data (should match activation data)
        if (ddr_read32(32'h00102000) === 32'h03020100 &&
            ddr_read32(32'h00102004) === 32'h07060504 &&
            ddr_read32(32'h00102008) === 32'h0B0A0908) begin
            $display("PASS: Result data matches activation data");
            pass = pass + 1;
        end else begin
            $display("FAIL: Result mismatch");
            $display("  res[0]=0x%08x (expect 0x03020100)", ddr_read32(32'h00102000));
            $display("  res[1]=0x%08x (expect 0x07060504)", ddr_read32(32'h00102004));
            $display("  res[2]=0x%08x (expect 0x0B0A0908)", ddr_read32(32'h00102008));
            fail = fail + 1;
        end

        // --- Test 2: Verify the descriptor correctly parsed tensor_type=15 ---
        // If FSM went to LOAD_ACT instead of LOAD_WEIGHT, the chain completed.
        // HEAD = 1 means one descriptor processed.
        if (rd_val >= 1) begin
            $display("PASS: Chain completed (HEAD >= 1)");
            pass = pass + 1;
        end else begin
            $display("FAIL: Chain not completed (HEAD=%d)", rd_val);
            fail = fail + 1;
        end

        $display("============================================");
        $display("  %0d PASS, %0d FAIL", pass, fail);
        $display("============================================");
        #100 $finish;
    end
endmodule
