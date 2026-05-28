`timescale 1ns / 1ps

module axihp_write_master (
    input  wire         clk,
    input  wire         rst_n,
    input  wire         start,
    input  wire [31:0]  dst_addr,
    input  wire [15:0]  word_count,
    output reg          busy,
    output reg          done,
    input  wire [63:0]  wdata,
    input  wire         wvalid,
    output reg          wready,
    output reg  [31:0]  m_axi_awaddr,
    output reg          m_axi_awvalid,
    input  wire         m_axi_awready,
    output reg  [7:0]   m_axi_awlen,
    output reg  [2:0]   m_axi_awsize,
    output reg  [1:0]   m_axi_awburst,
    output reg  [63:0]  m_axi_wdata,
    output reg          m_axi_wvalid,
    input  wire         m_axi_wready,
    output reg          m_axi_wlast,
    output reg  [7:0]   m_axi_wstrb,
    input  wire         m_axi_bvalid,
    output reg          m_axi_bready,
    input  wire  [1:0]  m_axi_bresp
);

    localparam [1:0] IDLE = 0, AW = 1, W = 2, B = 3;
    reg [1:0] state;
    reg [15:0] beat_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state         <= IDLE;
            busy          <= 0;
            done          <= 0;
            m_axi_awvalid <= 0;
            m_axi_wvalid  <= 0;
            m_axi_bready  <= 0;
            wready        <= 0;
            beat_cnt      <= 0;
        end else begin
            done <= 0;

            case (state)
                IDLE: begin
                    if (start && !busy) begin
                        if (word_count > 256) begin
                            $display("[WARN] axihp_write_master: word_count=%0d > 256, truncating to 256", word_count);
                        end
                        busy          <= 1;
                        m_axi_awaddr  <= dst_addr;
                        m_axi_awlen   <= (|word_count[15:8]) ? 8'd255 : word_count[7:0] - 1;
                        m_axi_awsize  <= 3'd3;
                        m_axi_awburst <= 2'd1;
                        m_axi_awvalid <= 1;
                        beat_cnt      <= 0;
                        state         <= AW;
                    end
                end

                AW: begin
                    if (m_axi_awready) begin
                        m_axi_awvalid <= 0;
                        wready        <= 1;
                        state         <= W;
                    end
                end

                W: begin
                    if (wvalid && wready) begin
                        m_axi_wdata  <= wdata;
                        m_axi_wstrb  <= 8'hFF;
                        m_axi_wlast  <= (beat_cnt == word_count - 1);
                        m_axi_wvalid <= 1;
                        wready       <= 0;
                    end
                    if (m_axi_wvalid && m_axi_wready) begin
                        m_axi_wvalid <= 0;
                        if (m_axi_wlast) begin
                            m_axi_bready <= 1;
                            state        <= B;
                        end else begin
                            wready   <= 1;
                            beat_cnt <= beat_cnt + 1;
                        end
                    end
                end

                B: begin
                    if (m_axi_bvalid) begin
                        m_axi_bready <= 0;
                        busy         <= 0;
                        done         <= 1;
                        state        <= IDLE;
                    end
                end
            endcase
        end
    end
endmodule
