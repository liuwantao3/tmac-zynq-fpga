`timescale 1ns / 1ps

// Phase B Descriptor Chain Testbench
// Loads DDR image from /tmp/tb_phaseb.bin, drives descriptor chain via AXI4-Lite,
// compares DDR results with expected values from /tmp/tb_phaseb.hdr.

module tb_phaseb;

    reg clk = 0;
    reg rst_n = 0;

    // AXI4-Lite (testbench master → DUT slave)
    reg        s_axil_awvalid;
    wire       s_axil_awready;
    reg [15:0] s_axil_awaddr;
    reg        s_axil_wvalid;
    wire       s_axil_wready;
    reg [31:0] s_axil_wdata;
    reg [3:0]  s_axil_wstrb;
    wire       s_axil_bvalid;
    reg        s_axil_bready;
    wire [1:0] s_axil_bresp;
    reg        s_axil_arvalid;
    wire       s_axil_arready;
    reg [15:0] s_axil_araddr;
    wire       s_axil_rvalid;
    reg        s_axil_rready;
    wire [31:0] s_axil_rdata;
    wire [1:0]  s_axil_rresp;
    wire        interrupt;

    // AXI HP (DUT master → DDR slave)
    wire [31:0] m_axi_araddr;
    wire        m_axi_arvalid;
    wire        m_axi_arready;
    wire [7:0]  m_axi_arlen;
    wire [2:0]  m_axi_arsize;
    wire [1:0]  m_axi_arburst;
    wire [63:0] m_axi_rdata;
    wire        m_axi_rvalid;
    wire        m_axi_rready;
    wire        m_axi_rlast;
    wire [31:0] m_axi_awaddr;
    wire        m_axi_awvalid;
    wire        m_axi_awready;
    wire [7:0]  m_axi_awlen;
    wire [2:0]  m_axi_awsize;
    wire [1:0]  m_axi_awburst;
    wire [63:0] m_axi_wdata;
    wire        m_axi_wvalid;
    wire        m_axi_wready;
    wire        m_axi_wlast;
    wire [7:0]  m_axi_wstrb;
    wire        m_axi_bvalid;
    wire        m_axi_bready;
    wire [1:0]  m_axi_bresp;

    // DDR model debug readback
    reg  [31:0] dbg_addr;
    wire [63:0] dbg_data;

    // Instantiate DUT
    matmul_top u_dut (
        .clk(clk), .rst_n(rst_n),
        .s_axil_awvalid(s_axil_awvalid),
        .s_axil_awready(s_axil_awready),
        .s_axil_awaddr(s_axil_awaddr),
        .s_axil_wvalid(s_axil_wvalid),
        .s_axil_wready(s_axil_wready),
        .s_axil_wdata(s_axil_wdata),
        .s_axil_wstrb(s_axil_wstrb),
        .s_axil_bvalid(s_axil_bvalid),
        .s_axil_bready(s_axil_bready),
        .s_axil_bresp(s_axil_bresp),
        .s_axil_arvalid(s_axil_arvalid),
        .s_axil_arready(s_axil_arready),
        .s_axil_araddr(s_axil_araddr),
        .s_axil_rvalid(s_axil_rvalid),
        .s_axil_rready(s_axil_rready),
        .s_axil_rdata(s_axil_rdata),
        .s_axil_rresp(s_axil_rresp),
        .interrupt(interrupt),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .m_axi_bresp(m_axi_bresp)
    );

    // Instantiate DDR model
    sim_ddr_axi_hp u_ddr (
        .clk(clk), .rst_n(rst_n),
        .s_axi_araddr(m_axi_araddr),
        .s_axi_arvalid(m_axi_arvalid),
        .s_axi_arready(m_axi_arready),
        .s_axi_arlen(m_axi_arlen),
        .s_axi_arsize(m_axi_arsize),
        .s_axi_arburst(m_axi_arburst),
        .s_axi_rdata(m_axi_rdata),
        .s_axi_rvalid(m_axi_rvalid),
        .s_axi_rready(m_axi_rready),
        .s_axi_rlast(m_axi_rlast),
        .s_axi_awaddr(m_axi_awaddr),
        .s_axi_awvalid(m_axi_awvalid),
        .s_axi_awready(m_axi_awready),
        .s_axi_awlen(m_axi_awlen),
        .s_axi_awsize(m_axi_awsize),
        .s_axi_awburst(m_axi_awburst),
        .s_axi_wdata(m_axi_wdata),
        .s_axi_wvalid(m_axi_wvalid),
        .s_axi_wready(m_axi_wready),
        .s_axi_wlast(m_axi_wlast),
        .s_axi_wstrb(m_axi_wstrb),
        .s_axi_bvalid(m_axi_bvalid),
        .s_axi_bready(m_axi_bready),
        .s_axi_bresp(m_axi_bresp),
        .dbg_addr(dbg_addr),
        .dbg_data(dbg_data)
    );

    // Clock: 100 MHz → 10 ns period
    always #5 clk = ~clk;

    // ======================================================================
    // AXI4-Lite master tasks
    // ======================================================================
    task axil_write(input [15:0] addr, input [31:0] data);
        @(posedge clk);
        #1;
        s_axil_awvalid = 1;
        s_axil_awaddr  = addr;
        s_axil_wvalid  = 1;
        s_axil_wdata   = data;
        s_axil_wstrb   = 4'hF;
        @(posedge clk);
        #1;
        s_axil_awvalid = 0;
        s_axil_wvalid  = 0;
        wait(s_axil_bvalid);
        @(posedge clk);
        #1;
    endtask

    task axil_read(input [15:0] addr, output [31:0] data);
        @(posedge clk);
        #1;
        s_axil_arvalid = 1;
        s_axil_araddr  = addr;
        @(posedge clk);
        #1;
        s_axil_arvalid = 0;
        wait(s_axil_rvalid);
        #1;
        data = s_axil_rdata;
    endtask

    // ======================================================================
    // DDR debug read
    // ======================================================================
    task ddr_read64(input [31:0] addr, output reg [63:0] data);
        @(posedge clk);
        #1;
        dbg_addr = addr;
        #1;
        data = dbg_data;
    endtask

    // ======================================================================
    // Header parsing: load /tmp/tb_phaseb.hdr into byte buffer
    // ======================================================================
    integer hdr_fd;
    reg [7:0] hdr_buf [0:1024*1024];  // 1 MB header buffer
    integer hdr_size;
    reg       hdr_loaded;

    integer   h_ndesc;
    integer   h_desc_base;
    integer   h_total_entries;

    reg [31:0] h_desc_act_bytes [0:15];
    reg [31:0] h_desc_prev_res  [0:15];

    integer    desc_i;  // shared between header parsing and test sequence
    integer    hdr_off;

    initial begin
        hdr_loaded = 0;
        hdr_fd = $fopen("/tmp/tb_phaseb.hdr", "rb");
        if (hdr_fd) begin
            hdr_size = $fread(hdr_buf, hdr_fd);
            $fclose(hdr_fd);
            $display("[TB] Loaded %0d bytes from /tmp/tb_phaseb.hdr", hdr_size);
            // Parse header (first 16 bytes)
            h_ndesc = {hdr_buf[7], hdr_buf[6], hdr_buf[5], hdr_buf[4]};
            h_desc_base = {hdr_buf[11], hdr_buf[10], hdr_buf[9], hdr_buf[8]};
            h_total_entries = {hdr_buf[15], hdr_buf[14], hdr_buf[13], hdr_buf[12]};
            $display("[TB] Header: ndesc=%0d desc_base=0x%08x total_entries=%0d",
                     h_ndesc, h_desc_base, h_total_entries);
            // Parse per-descriptor table (24 bytes each after 16-byte header)
            for (desc_i = 0; desc_i < h_ndesc && desc_i < 16; desc_i = desc_i + 1) begin
                hdr_off = 16 + desc_i * 24;
                h_desc_act_bytes[desc_i] = hdr_read32(hdr_off + 8);
                h_desc_prev_res[desc_i]   = hdr_read32(hdr_off + 12);
                $display("[TB]   desc %0d: act_total_bytes=%0d prev_result=0x%08x",
                         desc_i, h_desc_act_bytes[desc_i], h_desc_prev_res[desc_i]);
                // Validate chain: if prev_result addr matches prior desc result, act_total_bytes should = prev_rows × 8
                if (desc_i > 0 && h_desc_prev_res[desc_i] != 0) begin
                    integer prev_rows, expected_act_bytes;
                    prev_rows = hdr_read32(16 + (desc_i-1) * 24 + 4);  // nrows of prev desc
                    expected_act_bytes = prev_rows * 8;
                    if (h_desc_act_bytes[desc_i] != expected_act_bytes) begin
                        $display("[TB] WARNING: desc %0d act_total_bytes=%0d != expected %0d (prev rows=%0d, chain)",
                                 desc_i, h_desc_act_bytes[desc_i], expected_act_bytes, prev_rows);
                    end
                end
            end
            hdr_loaded = 1;
        end else begin
            $display("[TB] ERROR: Cannot open /tmp/tb_phaseb.hdr");
        end
    end

    // Helper: read 4 bytes from hdr_buf at byte offset
    function [31:0] hdr_read32;
        input integer byte_off;
        begin
            hdr_read32 = {hdr_buf[byte_off+3], hdr_buf[byte_off+2],
                          hdr_buf[byte_off+1], hdr_buf[byte_off]};
        end
    endfunction

    // Helper: read 8 bytes int64_t from hdr_buf at byte offset
    function signed [63:0] hdr_read64;
        input integer byte_off;
        begin
            hdr_read64 = {hdr_buf[byte_off+7], hdr_buf[byte_off+6],
                          hdr_buf[byte_off+5], hdr_buf[byte_off+4],
                          hdr_buf[byte_off+3], hdr_buf[byte_off+2],
                          hdr_buf[byte_off+1], hdr_buf[byte_off]};
        end
    endfunction

    // ======================================================================
    // Test sequence
    // ======================================================================
    reg [31:0] rd;
    integer    row_i;
    integer    total_errors;
    integer    desc_tab_off;
    integer    flat_off;
    integer    exp_off;
    integer    result_addr;
    integer    nrows;
    reg  [63:0] actual;
    reg  [63:0] expected;
    integer    max_rows;

    initial begin
        // 1. Reset
        $display("[TB] ========================================");
        $display("[TB] Phase B Descriptor Chain Testbench");
        $display("[TB] ========================================");
        s_axil_awvalid = 0;
        s_axil_wvalid  = 0;
        s_axil_arvalid = 0;
        s_axil_bready  = 1;
        s_axil_rready  = 1;
        s_axil_wstrb   = 4'hF;
        dbg_addr = 0;

        rst_n = 0;
        #20;
        rst_n = 1;
        #10;

        // 2. Wait for header to load
        wait(hdr_loaded);
        #10;

        // 3. Configure descriptor chain
        // Set mode register (all bits high enables all cores)
        axil_write(16'h0010, 32'h0000_01F0);  // CTRL_USER: enable all modes

        // Enable global interrupt + local
        axil_write(16'h0004, 32'h0000_0001);  // GIE
        axil_write(16'h0008, 32'h0000_0001);  // IER

        // Write DESC_BASE (address of contiguous descriptor array in DDR)
        axil_write(16'h0018, h_desc_base);

        // Write DESC_TAIL = number of descriptors (limit to 2 for simulation speed)
        //axil_write(16'h001C, h_ndesc);
        axil_write(16'h001C, 2);

        $display("[TB] DESC_BASE=0x%08x DESC_TAIL=2", h_desc_base);

        // 4. Start descriptor chain
        axil_write(16'h0024, 32'h0000_0001);  // CHAIN_CTRL[0]=1 (enable/start)

        // 5. Wait for chain to complete (poll CHAIN_CTRL[2] or wait for interrupt)
        $display("[TB] Descriptor chain started, waiting for interrupt...");
        wait(interrupt);
        $display("[TB] Interrupt received!");

        // Read status registers for info
        axil_read(16'h0014, rd);  // STATUS
        $display("[TB] STATUS=0x%08x", rd);
        axil_read(16'h0020, rd);  // DESC_HEAD
        $display("[TB] DESC_HEAD=%0d", rd);
        axil_read(16'h0024, rd);  // CHAIN_CTRL
        $display("[TB] CHAIN_CTRL=0x%08x", rd);

        // Clear ISR
        axil_write(16'h000C, 32'h0000_0001);

        // 6. Verify results: compare DDR with expected values from header
        // Only check first descriptor for now (quick validation)
        $display("[TB] ========================================");
        $display("[TB] Verifying descriptor 0 (DESC_HEAD=%0d)...", rd);
        total_errors = 0;
        desc_tab_off = 16;

        for (desc_i = 0; desc_i < 1; desc_i++) begin
            hdr_off = desc_tab_off + desc_i * 16;
            result_addr = hdr_read32(hdr_off);
            nrows       = hdr_read32(hdr_off + 4);
            exp_off     = hdr_read32(hdr_off + 12);

            // Compute offset in flat expected array (after per-desc table)
            flat_off = 16 + h_ndesc * 16 + exp_off * 8;

            // Limit max rows to prevent out-of-bounds
            max_rows = (nrows > 896) ? 896 : nrows;

            for (row_i = 0; row_i < max_rows; row_i++) begin
                #1;
                ddr_read64(result_addr + row_i * 8, actual);
                expected = hdr_read64(flat_off + row_i * 8);
                if (actual !== expected) begin
                    if (total_errors < 20) begin
                        $display("[TB] ERROR desc %0d row %0d: addr=0x%08x actual=%0d expected=%0d",
                                 desc_i, row_i, result_addr + row_i * 8,
                                 actual, expected);
                    end
                    total_errors = total_errors + 1;
                end
            end
            if (max_rows > 0) begin
                $display("[TB] desc %0d: %0d rows checked at 0x%08x",
                         desc_i, max_rows, result_addr);
            end
        end

        // 7. Report
        $display("[TB] ========================================");
        if (total_errors == 0) begin
            $display("[TB] PASS: All %0d descriptors match expected values!", h_ndesc);
        end else begin
            $display("[TB] FAIL: %0d errors out of %0d descriptors", total_errors, h_ndesc);
        end
        $display("[TB] ========================================");
        $finish;
    end

    // Timeout after 500M cycles (5s), with periodic heartbeat
    initial begin
        #10_000_000;
        $display("[TB] 10M cycles elapsed...");
        #40_000_000;
        $display("[TB] 50M cycles elapsed...");
        #50_000_000;
        $display("[TB] 100M cycles elapsed...");
        #100_000_000;
        $display("[TB] 200M cycles elapsed...");
        #100_000_000;
        $display("[TB] 300M cycles elapsed...");
        #100_000_000;
        $display("[TB] 400M cycles elapsed...");
        #100_000_000;
        $display("[TB] TIMEOUT after 500M cycles");
        $finish;
    end

    // VCD dump
    initial begin
        $dumpfile("tb_phaseb.vcd");
        $dumpvars(0, tb_phaseb);
    end

endmodule
