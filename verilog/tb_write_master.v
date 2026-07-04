`timescale 1ns / 1ps
// Demonstrate 64-bit axihp_write_master with 8-beat INCR burst (AWSIZE=3)
// Each 64-bit beat carries 8 bytes (8B × 8 = 64B total)
module tb_write_master;
    reg clk, rst_n;
    reg start;
    reg [31:0] dst_addr;
    reg [15:0] word_count;
    wire busy, done;
    reg  [63:0] wdata;
    reg         wvalid;
    wire        wready;
    wire [2:0] dbg_state;

    wire [5:0]  m_axi_awid;
    wire [31:0] m_axi_awaddr;
    wire        m_axi_awvalid;
    reg         m_axi_awready;
    wire [7:0]  m_axi_awlen;
    wire [2:0]  m_axi_awsize;
    wire [1:0]  m_axi_awburst;
    wire [1:0]  m_axi_awlock;
    wire [3:0]  m_axi_awcache;
    wire [2:0]  m_axi_awprot;
    wire [5:0]  m_axi_wid;
    wire [63:0] m_axi_wdata;
    wire        m_axi_wvalid;
    reg         m_axi_wready;
    wire        m_axi_wlast;
    wire [7:0]  m_axi_wstrb;
    reg         m_axi_bvalid;
    wire        m_axi_bready;
    reg  [1:0]  m_axi_bresp;
    reg  [5:0]  m_axi_bid;

    axihp_write_master uut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .dst_addr(dst_addr), .word_count(word_count),
        .busy(busy), .done(done),
        .dbg_state(dbg_state),
        .wdata(wdata), .wvalid(wvalid), .wready(wready),
        .m_axi_awid(m_axi_awid), .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awvalid(m_axi_awvalid), .m_axi_awready(m_axi_awready),
        .m_axi_awlen(m_axi_awlen), .m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst), .m_axi_awlock(m_axi_awlock),
        .m_axi_awcache(m_axi_awcache), .m_axi_awprot(m_axi_awprot),
        .m_axi_wid(m_axi_wid), .m_axi_wdata(m_axi_wdata),
        .m_axi_wvalid(m_axi_wvalid), .m_axi_wready(m_axi_wready),
        .m_axi_wlast(m_axi_wlast), .m_axi_wstrb(m_axi_wstrb),
        .m_axi_bvalid(m_axi_bvalid), .m_axi_bready(m_axi_bready),
        .m_axi_bresp(m_axi_bresp), .m_axi_bid(m_axi_bid)
    );

    always #5 clk = ~clk;

    // AXI slave (no backpressure)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_awready <= 1;
            m_axi_wready  <= 1;
            m_axi_bvalid  <= 0;
        end else begin
            if (m_axi_wvalid && m_axi_wready && m_axi_wlast) begin
                m_axi_bvalid <= 1;
            end else if (m_axi_bready) begin
                m_axi_bvalid <= 0;
            end
        end
    end

    // Send next word when wready is asserted
    reg [3:0] beat_sent;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            beat_sent <= 0;
            wvalid <= 0;
            wdata  <= 0;
        end else if (wready && wvalid) begin
            wdata <= wdata + 64'h0101010101010101;
            beat_sent <= beat_sent + 1;
        end else if (start) begin
            wvalid <= 1;
            wdata  <= 64'h0001020304050607;
        end
    end

    initial begin
        $dumpfile("tb_write_master.vcd");
        $dumpvars(0, tb_write_master);
    end

    reg done_captured;
    always @(posedge clk) if (done) done_captured <= 1;

    initial begin
        clk = 0; rst_n = 0; start = 0; done_captured = 0;
        #15 rst_n = 1;

        @(posedge clk);
        start <= 1; dst_addr <= 32'h00001000; word_count <= 16'd8;

        @(posedge clk);
        start <= 0;

        #2000;
        if (done_captured) $display("PASS: 8-word burst complete");
        else               $display("FAIL: done was never asserted");
        $finish;
    end
endmodule
