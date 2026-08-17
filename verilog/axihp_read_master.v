`timescale 1ns / 1ps

// AXI HP Read Master — 64-bit word output, ARSIZE=2 (32-bit AXI beats)
// Accumulates 2 AXI beats into one 64-bit output word.
// Matches axihp_write_master's wdata[63:0] interface.
//
// Zynq-7010 with x16 DDR: ARSIZE=3 loses upper 32 bits (RDATA[63:32]=0),
// so we keep ARSIZE=2 and combine pairs internally.
//
// Step 3a (2026-08-17): pipelined AR support. A `start` received while a burst
// drains is accepted into a 1-deep queue and its AR is issued as soon as the AR
// channel is free, so consecutive bursts pipeline on the PS7 HP0 (which
// supports outstanding reads). `done` pulses per burst (even when a queued burst
// is promoted), so per-burst consumers (Q5 blocks, Q8 weight bursts) are
// unchanged. Single-flight callers (Q8, CPU_OP) are unaffected.

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
    reg [7:0] beat_cnt;          // 0..cur_len, counts AXI beats received
    reg       even_beat;         // 0=accumulating low half, 1=accumulating high half
    reg [31:0] rdata_lo, rdata_hi;
    reg       start_prev;
    wire      start_rise = start && !start_prev;

    // ---- pipelined AR support (Step 3a) ----
    reg [31:0] cur_addr;         // active burst DDR address (survives promote)
    reg [7:0]  cur_len;          // active burst beat count (survives promote)
    reg       queued_valid;      // a next burst was accepted while busy
    reg [31:0] queued_addr;
    reg [7:0]  queued_len;
    reg       nxt_arsent;        // queued burst's AR already issued

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
            start_prev     <= 0;
            cur_addr       <= 0;
            cur_len        <= 0;
            queued_valid   <= 0;
            queued_addr    <= 0;
            queued_len     <= 0;
            nxt_arsent     <= 0;
        end else begin
            rvalid <= 0;
            done   <= 0;
            start_prev <= start;

            // Accept a new start while busy into the 1-deep queue (pipelined AR).
            // In IDLE, busy=0 so the IDLE case handles the first start normally.
            if (busy && !queued_valid && start_rise) begin
                queued_valid <= 1;
                queued_addr  <= src_addr;
                queued_len   <= burst_len;
            end

            case (state)
                IDLE: begin
                    if (start_rise) begin
                        busy          <= 1;
                        cur_addr      <= src_addr;
                        cur_len       <= burst_len;
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
                        // Pipeline: if a next burst is queued and not yet AR'd,
                        // send its AR immediately (DDR latency overlaps drain).
                        if (queued_valid && !nxt_arsent) begin
                            m_axi_araddr  <= queued_addr;
                            m_axi_arlen   <= queued_len;
                            m_axi_arvalid <= 1;
                            nxt_arsent    <= 1;
                        end
                        m_axi_rready <= 1;
                        state         <= READ_BEAT;
                    end
                end

                READ_BEAT: begin
                    // Complete a queued AR handshake, or start one if the AR
                    // channel is free and a queued burst is not yet AR'd.
                    if (m_axi_arvalid && nxt_arsent) begin
                        if (m_axi_arready) m_axi_arvalid <= 0;
                    end else if (!m_axi_arvalid && queued_valid && !nxt_arsent) begin
                        m_axi_araddr  <= queued_addr;
                        m_axi_arlen   <= queued_len;
                        m_axi_arvalid <= 1;
                        nxt_arsent    <= 1;
                    end
                    if (m_axi_rvalid) begin
                        m_axi_rready <= 0;
                        if (!even_beat) begin
                            // First beat of pair → lower 32 bits
                            rdata_lo <= m_axi_rdata[31:0];
                            if (beat_cnt == cur_len) begin
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
                            if (beat_cnt == cur_len) begin
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
                    // Complete a pipelined AR handshake if arready arrives while
                    // in PRESENT. The slave accepts the next burst's AR exactly
                    // when the current burst finishes, which is when the master is
                    // here presenting the last word; without this, arvalid would
                    // stay asserted forever (only SEND_AR/READ_BEAT clear it) and
                    // the queued burst would never promote.
                    if (m_axi_arvalid && nxt_arsent) begin
                        if (m_axi_arready) m_axi_arvalid <= 0;
                    end
                    if (rready) begin
                        rvalid <= 0;  // clear rvalid on handshake to prevent double-capture
                        if (beat_cnt >= cur_len && !even_beat) begin
                            // All AXI beats consumed, all words presented.
                            done <= 1;   // per-burst done, even when promoting
                            if (queued_valid) begin
                                // Promote the queued burst to active and continue.
                                cur_addr      <= queued_addr;
                                cur_len       <= queued_len;
                                queued_valid  <= 0;
                                beat_cnt      <= 0;
                                even_beat     <= 0;
                                if (nxt_arsent) begin
                                    // Its AR already went out — data is arriving;
                                    // keep consuming (no new AR latency).
                                    nxt_arsent    <= 0;
                                    m_axi_rready  <= 1;
                                    state         <= READ_BEAT;
                                end else begin
                                    // AR not sent yet — send it now.
                                    m_axi_araddr  <= queued_addr;
                                    m_axi_arlen   <= queued_len;
                                    m_axi_arvalid <= 1;
                                    nxt_arsent    <= 0;
                                    state         <= SEND_AR;
                                end
                            end else begin
                                busy  <= 0;
                                state <= IDLE;
                            end
                        end else begin
                            // More AXI beats to read for next word
                            if (!even_beat) begin
                                m_axi_rready <= 1;
                                state <= READ_BEAT;
                            end else begin
                                // even_beat=1 means solo beat was presented (odd count)
                                // That was the last beat, so we're done
                                done  <= 1;
                                busy  <= 0;
                                state <= IDLE;
                            end
                        end
                    end
                end
            endcase
        end
    end
endmodule
