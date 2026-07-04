`timescale 1ns / 1ps

// AXI HP Read Master — 64-bit word output, ARSIZE=2 (32-bit AXI beats)
// Accumulates 2 AXI beats into one 64-bit output word.
// Matches axihp_write_master's wdata[63:0] interface.
//
// Zynq-7010 with x16 DDR: ARSIZE=3 loses upper 32 bits (RDATA[63:32]=0),
// so we keep ARSIZE=2 and combine pairs internally.

module axihp_read_master (
    input  wire         clk,
    input  wire         rst_n,

    input  wire         start,
    input  wire [31:0]  src_addr,
    input  wire [7:0]   burst_len,      // total 32-bit AXI beats - 1 (0..255, same as ARLEN)
    output reg          done,
    output reg          busy,
    output wire [63:0]  rdata,          // 64-bit combined output word
    output reg          rvalid,         // word valid (consumer should capture on this cycle if rready=1)
    input  wire         rready,         // consumer ready
    output wire [2:0]   dbg_state,
    output wire [7:0]   dbg_beat_cnt,

    // AXI HP read interface
    output reg  [5:0]   m_axi_arid,
    output reg  [31:0]  m_axi_araddr,
    output reg          m_axi_arvalid,
    input  wire         m_axi_arready,
    output reg  [7:0]   m_axi_arlen,
    output reg  [2:0]   m_axi_arsize,
    output reg  [1:0]   m_axi_arburst,
    output reg  [1:0]   m_axi_arlock,
    output reg  [3:0]   m_axi_arcache,
    output reg  [2:0]   m_axi_arprot,
    input  wire [63:0]  m_axi_rdata,
    input  wire [1:0]   m_axi_rresp,
    input  wire [5:0]   m_axi_rid,
    input  wire         m_axi_rvalid,
    output reg          m_axi_rready,
    input  wire         m_axi_rlast
);

    localparam [2:0] IDLE      = 3'd0;
    localparam [2:0] SEND_AR   = 3'd1;
    localparam [2:0] READ_BEAT = 3'd2;
    localparam [2:0] PRESENT   = 3'd3;

    reg [2:0] state;
    reg [7:0] beat_cnt;          // 0..burst_len, counts AXI beats received
    reg       even_beat;         // 0=accumulating low half, 1=accumulating high half
    reg [31:0] rdata_lo, rdata_hi;

    assign rdata       = {rdata_hi, rdata_lo};
    assign dbg_state   = state;
    assign dbg_beat_cnt = beat_cnt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= IDLE;
            done           <= 0;
            busy           <= 0;
            rvalid         <= 0;
            m_axi_arvalid  <= 0;
            m_axi_rready   <= 0;
            beat_cnt       <= 0;
            even_beat      <= 0;
            rdata_lo       <= 0;
            rdata_hi       <= 0;
        end else begin
            rvalid <= 0;
            done   <= 0;

            case (state)
                IDLE: begin
                    if (start && !busy) begin
                        busy          <= 1;
                        m_axi_arid    <= 6'd0;
                        m_axi_araddr  <= src_addr;
                        m_axi_arlen   <= burst_len;
                        m_axi_arsize  <= 3'd2;    // 4 bytes per beat
                        m_axi_arburst <= 2'd1;     // INCR
                        m_axi_arlock  <= 2'd0;
                        m_axi_arcache <= 4'b0011;
                        m_axi_arprot  <= 3'd0;
                        m_axi_arvalid <= 1;
                        beat_cnt      <= 0;
                        even_beat     <= 0;
                        state         <= SEND_AR;
                    end
                end

                SEND_AR: begin
                    if (m_axi_arready) begin
                        m_axi_arvalid <= 0;
                        m_axi_rready  <= 1;
                        state         <= READ_BEAT;
                    end
                end

                READ_BEAT: begin
                    if (m_axi_rvalid) begin
                        m_axi_rready <= 0;
                        if (!even_beat) begin
                            // First beat of pair → lower 32 bits
                            rdata_lo <= m_axi_rdata[31:0];
                            if (beat_cnt == burst_len) begin
                                // Last beat is odd (only 1 beat in this pair)
                                rdata_hi  <= 32'd0;
                                even_beat <= 1;
                                state     <= PRESENT;
                            end else begin
                                beat_cnt   <= beat_cnt + 1;
                                even_beat  <= 1;
                                m_axi_rready <= 1;  // request next beat immediately
                            end
                        end else begin
                            // Second beat of pair → upper 32 bits
                            rdata_hi  <= m_axi_rdata[31:0];
                            even_beat <= 0;
                            if (beat_cnt == burst_len) begin
                                state <= PRESENT;   // last pair complete
                            end else begin
                                beat_cnt <= beat_cnt + 1;
                                state     <= PRESENT;
                            end
                        end
                    end
                end

                PRESENT: begin
                    rvalid <= 1;
                    if (rready) begin
                        rvalid <= 0;  // clear rvalid on handshake to prevent double-capture
                        if (beat_cnt >= burst_len && !even_beat) begin
                            // All AXI beats consumed, all words presented
                            busy  <= 0;
                            done  <= 1;
                            state <= IDLE;
                        end else begin
                            // More AXI beats to read for next word
                            if (!even_beat) begin
                                m_axi_rready <= 1;
                                state <= READ_BEAT;
                            end else begin
                                // even_beat=1 means solo beat was presented (odd count)
                                // That was the last beat, so we're done
                                busy  <= 0;
                                done  <= 1;
                                state <= IDLE;
                            end
                        end
                    end
                end
            endcase
        end
    end
endmodule
