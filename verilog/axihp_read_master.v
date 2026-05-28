`timescale 1ns / 1ps

// AXI HP Read Master — single-INCR-burst, byte-stream output
// Performs one AXI4 INCR burst of (arlen+1) × 8 bytes from src_addr.
// Drains returned data byte-by-byte on data_out/data_valid/data_ready.
// For transfers > 256 beats × 8 B = 2048 B, the caller must issue
// multiple bursts (up to Zynq HP's max 256-beat limit per burst).

module axihp_read_master (
    input  wire         clk,
    input  wire         rst_n,

    // Control — pulse start to begin one burst
    input  wire         start,
    input  wire [31:0]  src_addr,
    input  wire [7:0]   burst_len,    // number of beats − 1 (0–255)
    output reg          done,          // pulse when burst complete
    output reg          busy,

    // Byte stream output (one byte per cycle when valid & ready)
    output reg  [7:0]   data_out,
    output reg          data_valid,
    input  wire         data_ready,

    // AXI4 (full) read master
    output reg  [31:0]  m_axi_araddr,
    output reg          m_axi_arvalid,
    input  wire         m_axi_arready,
    output reg  [7:0]   m_axi_arlen,
    output reg  [2:0]   m_axi_arsize,
    output reg  [1:0]   m_axi_arburst,
    input  wire [63:0]  m_axi_rdata,
    input  wire         m_axi_rvalid,
    output reg          m_axi_rready,
    input  wire         m_axi_rlast
);

    localparam IDLE       = 3'd0;
    localparam SEND_AR    = 3'd1;
    localparam WAIT_R     = 3'd2;
    localparam DRAIN_BYTE = 3'd3;

    reg [2:0]  state;
    reg [7:0]  beat_cnt;
    reg [63:0] rdata_buf;
    reg [2:0]  buf_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state          <= IDLE;
            done           <= 0;
            busy           <= 0;
            data_valid     <= 0;
            m_axi_arvalid  <= 0;
            m_axi_rready   <= 0;
            buf_idx        <= 0;
            beat_cnt       <= 0;
        end else begin
            done       <= 0;
            data_valid <= 0;

            case (state)
                IDLE: begin
                    if (start) begin
                        busy          <= 1;
                        m_axi_araddr  <= src_addr;
                        m_axi_arlen   <= burst_len;
                        m_axi_arsize  <= 3'd3;  // 8 bytes per beat
                        m_axi_arburst <= 2'd1;  // INCR
                        m_axi_arvalid <= 1;
                        beat_cnt      <= 0;
                        state         <= SEND_AR;
                    end
                end

                SEND_AR: begin
                    if (m_axi_arready) begin
                        m_axi_arvalid <= 0;
                        m_axi_rready  <= 1;
                        state         <= WAIT_R;
                    end
                end

                WAIT_R: begin
                    if (m_axi_rvalid) begin
                        // Capture beat; stall R channel while draining
                        m_axi_rready <= 0;
                        rdata_buf    <= m_axi_rdata;
                        buf_idx      <= 0;
                        state        <= DRAIN_BYTE;
                    end
                end

                DRAIN_BYTE: begin
                    data_out   <= rdata_buf[buf_idx * 8 +: 8];
                    data_valid <= 1;
                    if (data_ready) begin
                        if (buf_idx == 7) begin
                            // Beat fully drained
                            buf_idx <= 0;
                            if (beat_cnt == burst_len) begin
                                // Last beat of burst
                                state <= IDLE;
                                busy  <= 0;
                                done  <= 1;
                            end else begin
                                // More beats in this burst
                                beat_cnt <= beat_cnt + 1;
                                m_axi_rready <= 1;
                                state <= WAIT_R;
                            end
                        end else begin
                            buf_idx <= buf_idx + 1;
                        end
                    end
                end
            endcase
        end
    end

endmodule
