`timescale 1ns / 1ps
// Test axihp_read_master 64-bit output: 2 × 32-bit AXI beats → 1 × 64-bit word
module tb_read_master;
    reg clk, rst_n;
    reg start;
    reg [31:0] src_addr;
    reg [7:0]  burst_len;
    wire done, busy;
    wire [63:0] rdata;
    wire        rvalid;
    reg         rready;
    wire [2:0]  dbg_state;

    // AXI slave model
    reg         m_axi_arready;
    reg  [63:0] m_axi_rdata;
    reg         m_axi_rvalid;
    reg         m_axi_rlast;
    wire        m_axi_arvalid;
    wire        m_axi_rready;

    axihp_read_master uut (
        .clk(clk), .rst_n(rst_n),
        .start(start), .src_addr(src_addr), .burst_len(burst_len),
        .done(done), .busy(busy),
        .rdata(rdata), .rvalid(rvalid), .rready(rready),
        .dbg_state(dbg_state),
        .m_axi_arid(), .m_axi_araddr(),
        .m_axi_arvalid(m_axi_arvalid), .m_axi_arready(m_axi_arready),
        .m_axi_arlen(), .m_axi_arsize(), .m_axi_arburst(),
        .m_axi_arlock(), .m_axi_arcache(), .m_axi_arprot(),
        .m_axi_rdata(m_axi_rdata), .m_axi_rresp(2'b00),
        .m_axi_rid(6'd0), .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready), .m_axi_rlast(m_axi_rlast)
    );

    always #5 clk = ~clk;

    // AXI read slave with **combinatorial** arready (avoid NBA race)
    assign m_axi_arready = m_axi_arvalid;   // accept AR immediately

    reg [7:0] next_byte;
    reg       ar_hs;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            m_axi_rvalid <= 0;
            m_axi_rlast  <= 0;
            next_byte    <= 0;
            ar_hs        <= 0;
        end else begin
            // Capture AR handshake
            if (m_axi_arvalid && m_axi_arready && !ar_hs) begin
                ar_hs     <= 1;
                next_byte <= 0;
            end

            // Provide R data when rready is high and not already valid
            if (m_axi_rready && !m_axi_rvalid && ar_hs) begin
                m_axi_rvalid <= 1;
                m_axi_rdata[7:0]   <= next_byte;
                m_axi_rdata[15:8]  <= next_byte + 1;
                m_axi_rdata[23:16] <= next_byte + 2;
                m_axi_rdata[31:24] <= next_byte + 3;
                m_axi_rdata[63:32] <= 32'd0;
                m_axi_rlast  <= (next_byte == (burst_len << 2));
                next_byte    <= next_byte + 4;
            end

            // Clear rvalid when accepted
            if (m_axi_rvalid && m_axi_rready) begin
                m_axi_rvalid <= 0;
            end
        end
    end

    initial begin
        $dumpfile("tb_read_master.vcd");
        $dumpvars(0, tb_read_master);
    end

    reg done_captured;
    reg [63:0] data_words[0:7];
    reg [3:0]  word_cnt;
    integer i;

    always @(posedge clk) begin
        if (done) done_captured <= 1;
        if (rvalid && rready) begin
            data_words[word_cnt] <= rdata;
            word_cnt <= word_cnt + 1;
        end
    end

    initial begin
        clk = 0; rst_n = 0; start = 0;
        rready = 0; done_captured = 0;
        word_cnt = 0;
        #15 rst_n = 1;

        // Test: 16 beats (burst_len=15), expect 8 × 64-bit words
        @(posedge clk);
        start <= 1; src_addr <= 32'h00001000; burst_len <= 8'd15;
        rready <= 1;

        @(posedge clk);
        start <= 0;

        #1000;
        if (done_captured) $display("PASS: done asserted, %0d words received", word_cnt);
        else               $display("FAIL: done not asserted");

        // Word 0 = {AXI_beat1[31:0], AXI_beat0[31:0]}
        // beat0: {3,2,1,0} = 0x03020100, beat1: {7,6,5,4} = 0x07060504
        // Word 0 = 0x0706050403020100
        if (data_words[0] === 64'h0706050403020100)
            $display("PASS: word 0 = 0x%016h", data_words[0]);
        else
            $display("FAIL: word 0 = 0x%016h (expected 0x0706050403020100)", data_words[0]);

        // Word 0..7 expected values:
        // word 0: 0x0706050403020100   (beats 0-1: bytes 0-7)
        // word 1: 0x0F0E0D0C0B0A0908   (beats 2-3: bytes 8-15)
        // word 7: 0x3F3E3D3C3B3A3938   (beats 14-15: bytes 56-63)
        for (i = 0; i < word_cnt; i = i + 1) begin
            $display("  word[%0d] = 0x%016h", i, data_words[i]);
        end

        if (word_cnt == 8)
            $display("PASS: all 8 words received");
        else
            $display("FAIL: got %0d words, expected 8", word_cnt);

        $finish;
    end
endmodule
