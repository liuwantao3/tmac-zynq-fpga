`timescale 1ns / 1ps
// Test axihp_read_master pipelined AR: a second start issued while the first
// burst drains must be queued, its AR issued early, and both bursts' data
// returned in order with a per-burst done pulse.
module tb_read_master_pipe;
    reg clk, rst_n;
    reg start;
    reg [31:0] src_addr;
    reg [7:0]  burst_len;
    wire done, busy;
    wire [63:0] rdata;
    wire        rvalid;
    reg         rready;
    wire [2:0]  dbg_state;

    wire m_axi_arvalid, m_axi_rready;
    wire [31:0] m_axi_araddr;
    wire [7:0]  m_axi_arlen;
    reg  m_axi_arready;
    reg  [63:0] m_axi_rdata;
    reg  m_axi_rvalid, m_axi_rlast;

    axihp_read_master uut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .src_addr(src_addr), .burst_len(burst_len),
        .done(done), .busy(busy),
        .rdata(rdata), .rvalid(rvalid), .rready(rready),
        .dbg_state(dbg_state),
        .m_axi_arid(), .m_axi_araddr(m_axi_araddr),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_arlen(m_axi_arlen), .m_axi_arsize(), .m_axi_arburst(),
        .m_axi_arlock(), .m_axi_arcache(), .m_axi_arprot(),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(2'b00),
        .m_axi_rid(6'd0), .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready), .m_axi_rlast(m_axi_rlast)
    );

    always #5 clk = ~clk;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) rready <= 0;
        else        rready <= rvalid;
    end

    // AXI read slave: 2-deep outstanding-AR FIFO, in-order data
    reg [31:0] arq_addr [0:1];
    reg [7:0]  arq_len  [0:1];
    reg [1:0]  ar_cnt, ar_rd, ar_wr;
    reg        srv_busy;
    reg [7:0]  srv_cnt;
    reg [7:0]  srv_len;
    reg [31:0] beat_addr;

    assign m_axi_arready = (ar_cnt < 2);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ar_cnt <= 0; ar_rd <= 0; ar_wr <= 0;
            srv_busy <= 0; srv_cnt <= 0; srv_len <= 0;
            beat_addr <= 0;
            m_axi_rvalid <= 0; m_axi_rlast <= 0;
        end else begin
            if (m_axi_arvalid && m_axi_arready) begin
                arq_addr[ar_wr] <= m_axi_araddr;
                arq_len[ar_wr]  <= m_axi_arlen;
                ar_wr <= ar_wr + 1;
                ar_cnt <= ar_cnt + 1;
            end
            if (!srv_busy && ar_cnt > 0) begin
                srv_busy  <= 1;
                srv_cnt   <= 0;
                srv_len   <= arq_len[ar_rd];
                beat_addr <= arq_addr[ar_rd];
                ar_rd <= ar_rd + 1;
                ar_cnt <= ar_cnt - 1;
            end
            if (srv_busy) begin
                if (!m_axi_rvalid) begin
                    m_axi_rvalid <= 1;
                    m_axi_rlast  <= (srv_cnt == srv_len);
                    m_axi_rdata  <= {32'd0, beat_addr};
                end
                if (m_axi_rvalid && m_axi_rready) begin
                    m_axi_rvalid <= 0;
                    if (srv_cnt >= srv_len) srv_busy <= 0;
                    else begin
                        srv_cnt <= srv_cnt + 1;
                        beat_addr <= beat_addr + 4;
                    end
                end
            end
        end
    end

    reg [63:0] data_words [0:15];
    reg [4:0]  word_cnt;
    reg [3:0]  done_cnt;
    integer i, fails;

    always @(posedge clk) begin
        if (rvalid && rready) begin
            data_words[word_cnt] <= rdata;
            word_cnt <= word_cnt + 1;
        end
        if (done) done_cnt <= done_cnt + 1;
    end

    task issue_start(input [31:0] addr, input [7:0] len);
        @(posedge clk);
        start <= 1; src_addr <= addr; burst_len <= len;
        @(posedge clk);
        start <= 0;
    endtask

    initial begin
        clk = 0; rst_n = 0; start = 0;
        word_cnt = 0; done_cnt = 0; fails = 0;
        #15 rst_n = 1;

        issue_start(32'h00001000, 8'd11);
        repeat (6) @(posedge clk);
        issue_start(32'h00002000, 8'd11);

        repeat (300) @(posedge clk);

        $display("words received: %0d, done pulses: %0d", word_cnt, done_cnt);
        if (word_cnt != 12) begin
            $display("FAIL: expected 12 words, got %0d", word_cnt);
            fails = 1;
        end
        if (done_cnt != 2) begin
            $display("FAIL: expected 2 done pulses, got %0d", done_cnt);
            fails = 1;
        end
        for (i = 0; i < 6; i = i + 1) begin
            reg [63:0] exp;
            exp = ((32'h00001004 + (i << 3)) << 32) | (32'h00001000 + (i << 3));
            if (data_words[i] !== exp) begin
                $display("FAIL: burst1 word %0d = 0x%016h (expected 0x%016h)", i, data_words[i], exp);
                fails = 1;
            end
        end
        for (i = 0; i < 6; i = i + 1) begin
            reg [63:0] exp;
            exp = ((32'h00002004 + (i << 3)) << 32) | (32'h00002000 + (i << 3));
            if (data_words[6+i] !== exp) begin
                $display("FAIL: burst2 word %0d = 0x%016h (expected 0x%016h)", 6+i, data_words[6+i], exp);
                fails = 1;
            end
        end
        if (fails == 0) $display("ALL TESTS PASSED");
        else            $display("%0d FAILURES", fails);
        $finish;
    end
endmodule
