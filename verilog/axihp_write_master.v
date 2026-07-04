`timescale 1ns / 1ps

// AXI HP Write Master — 64-bit wdata, INCR burst, AWSIZE=3 (8 bytes/beat)
// Zynq-7010 HP port with x16 DDR3: HP0 is 32-bit, RDATA[63:32]=0.
// AWSIZE=3 sends 8 bytes per beat — the Zynq HP port handles this
// by performing two 32-bit DDR accesses per beat.
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
    output wire [2:0]   dbg_state,
    output reg  [5:0]   m_axi_awid,
    output reg  [31:0]  m_axi_awaddr,
    output reg          m_axi_awvalid,
    input  wire         m_axi_awready,
    output reg  [7:0]   m_axi_awlen,
    output reg  [2:0]   m_axi_awsize,
    output reg  [1:0]   m_axi_awburst,
    output reg  [1:0]   m_axi_awlock,
    output reg  [3:0]   m_axi_awcache,
    output reg  [2:0]   m_axi_awprot,
    output reg  [5:0]   m_axi_wid,
    output reg  [63:0]  m_axi_wdata,
    output reg          m_axi_wvalid,
    input  wire         m_axi_wready,
    output reg          m_axi_wlast,
    output reg  [7:0]   m_axi_wstrb,
    input  wire         m_axi_bvalid,
    output reg          m_axi_bready,
    input  wire  [1:0]  m_axi_bresp,
    input  wire  [5:0]  m_axi_bid
);

    localparam [1:0] IDLE = 0, AW = 1, W = 2, B = 3;
    reg [1:0] state;
    reg [15:0] beat_cnt;
    reg [15:0] beats_total;

    assign dbg_state = {1'b0, state};

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
            beats_total   <= 0;
            m_axi_awid    <= 6'd0;
            m_axi_awlock  <= 2'd0;
            m_axi_awcache <= 4'b0011;
            m_axi_awprot  <= 3'd0;
            m_axi_wid     <= 6'd0;
        end else begin
            done <= 0;

            case (state)
                IDLE: begin
                    if (start && !busy) begin
                        beats_total  <= word_count;
                        busy         <= 1;
                        m_axi_awaddr <= dst_addr;
                        m_axi_awlen  <= (word_count > 0) ? word_count[7:0] - 1 : 8'd0;
                        m_axi_awsize <= 3'd3;
                        m_axi_awburst <= 2'd1;
                        m_axi_awvalid <= 1;
                        beat_cnt     <= 0;
                        state        <= AW;
                    end
                end

                AW: begin
                    if (m_axi_awready) begin
                        m_axi_awvalid <= 0;
                        if (word_count == 0) begin
                            m_axi_bready <= 1;
                            state        <= B;
                        end else begin
                            wready <= 1;
                            state  <= W;
                        end
                    end
                end

                W: begin
                    if (wvalid && wready) begin
                        m_axi_wdata <= wdata;
                        m_axi_wstrb <= 8'hFF;
                        m_axi_wlast <= (beat_cnt == beats_total - 1);
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
